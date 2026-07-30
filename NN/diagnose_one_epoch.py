from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import (
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    LEARNING_RATE,
    N_HEAD,
    N_LAYERS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import TauSpinDataset, collate_events, extract_event
from hpo_utils import shutdown_loader_workers
from model import TauSpinTransformer
from train import (
    binary_roc_auc,
    choose_device,
    move_batch,
    require_finite,
    require_finite_gradients,
    set_random_seed,
)


LOGGER = logging.getLogger("diagnose_one_epoch")
VALIDATION_PER_CLASS = 2048
VALIDATION_INTERVAL = 100
MOVING_AVERAGE_WINDOW = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where the baseline TauSpin training saturates within "
            "one epoch."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR / "old-full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUTPUT_DIR
            / "diagnostics"
            / "one-epoch-baseline-v1"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )
    parser.add_argument(
        "--device",
        choices=("cuda",),
        default="cuda",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Use the validated torch float32 matmul precision='high' runtime.",
    )
    parser.add_argument(
        "--dataset-label",
        default="old sample",
        help="Short label used in metadata and the diagnostic plot.",
    )
    parser.add_argument(
        "--purpose",
        default=(
            "Diagnose one-epoch baseline saturation on the old full sample; "
            "not a physics result."
        ),
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", mode="w"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def git_state(repository: Path) -> dict:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.rstrip()

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD"),
        "working_tree_clean": not bool(status),
        "status_porcelain": status.splitlines(),
        "diff_stat": run("diff", "--stat").splitlines(),
    }


def build_configuration(
    arguments: argparse.Namespace,
    metadata: dict,
    repository: Path,
) -> dict:
    return {
        "purpose": arguments.purpose,
        "dataset_label": arguments.dataset_label,
        "epochs": 1,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "dropout": DROPOUT,
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
        "scheduler": None,
        "early_stopping": False,
        "validation_interval_optimizer_steps": VALIDATION_INTERVAL,
        "validation_subset_per_class": VALIDATION_PER_CLASS,
        "moving_average_window": MOVING_AVERAGE_WINDOW,
        "num_workers": arguments.num_workers,
        "prefetch_factor": arguments.prefetch_factor,
        "tf32": arguments.tf32,
        "d_model": D_MODEL,
        "n_heads": N_HEAD,
        "n_layers": N_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "processed_dataset_path": str(arguments.processed_dir.resolve()),
        "dataset_metadata": metadata,
        "git": git_state(repository),
    }


def events_fingerprint(events: Iterable[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        for name in sorted(event):
            tensor = event[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_validation_class(
    processed_dir: Path,
    metadata: dict,
    sample: str,
    count: int,
) -> list[dict[str, torch.Tensor]]:
    events: list[dict[str, torch.Tensor]] = []
    for record in metadata["shards"]["validation"][sample]:
        shard = torch.load(
            processed_dir / record["path"],
            map_location="cpu",
            weights_only=True,
        )
        for index in range(int(shard["labels"].shape[0])):
            events.append(extract_event(shard, index))
            if len(events) == count:
                return events
    raise RuntimeError(
        f"Validation split has only {len(events)} {sample} events; "
        f"{count} are required"
    )


def fixed_validation_subset(
    processed_dir: Path,
    metadata: dict,
) -> list[dict[str, torch.Tensor]]:
    h_events = load_validation_class(
        processed_dir, metadata, "H", VALIDATION_PER_CLASS
    )
    z_events = load_validation_class(
        processed_dir, metadata, "Z", VALIDATION_PER_CLASS
    )
    events: list[dict[str, torch.Tensor]] = []
    for h_event, z_event in zip(h_events, z_events):
        events.extend((h_event, z_event))
    labels = torch.stack([event["label"] for event in events])
    if int((labels == 1).sum()) != VALIDATION_PER_CLASS:
        raise RuntimeError("Fixed validation subset has the wrong H count")
    if int((labels == 0).sum()) != VALIDATION_PER_CLASS:
        raise RuntimeError("Fixed validation subset has the wrong Z count")
    return events


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


def assert_parameters_unchanged(
    model: nn.Module,
    before: list[torch.Tensor],
) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(before):
        raise RuntimeError("Model parameter count changed during validation")
    for (name, parameter), reference in zip(
        model.named_parameters(), before
    ):
        if not torch.equal(parameter.detach(), reference):
            raise RuntimeError(
                f"Model parameter changed during validation: {name}"
            )


def evaluate_without_updates(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    name: str,
    expected_labels: np.ndarray | None = None,
    expected_event_numbers: np.ndarray | None = None,
) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    was_training = model.training
    before = parameter_snapshot(model)
    model.eval()
    loss_sum = 0.0
    event_count = 0
    labels = []
    scores = []
    event_numbers = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            require_finite(logits, f"{name} logits")
            loss = loss_function(logits, batch["labels"])
            require_finite(loss, f"{name} loss")
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            event_count += batch_size
            labels.append(batch["labels"].detach().cpu())
            scores.append(torch.sigmoid(logits).detach().cpu())
            event_numbers.append(batch["event_numbers"].detach().cpu())

    assert_parameters_unchanged(model, before)
    if was_training:
        model.train()

    if event_count == 0:
        raise RuntimeError(f"{name} loader produced no events")
    label_array = torch.cat(labels).numpy()
    score_array = torch.cat(scores).numpy()
    number_array = torch.cat(event_numbers).numpy()
    if expected_labels is not None and not np.array_equal(
        label_array, expected_labels
    ):
        raise RuntimeError("Fixed validation subset label order changed")
    if expected_event_numbers is not None and not np.array_equal(
        number_array, expected_event_numbers
    ):
        raise RuntimeError(
            "Fixed validation subset event order changed"
        )
    return (
        loss_sum / event_count,
        binary_roc_auc(label_array, score_array),
        event_count,
        label_array,
        number_array,
    )


def save_history(
    output_dir: Path,
    configuration: dict,
    train_history: list[dict],
    validation_history: list[dict],
    full_validation: dict | None,
) -> None:
    document = {
        "configuration": configuration,
        "train_steps": train_history,
        "validation_subset": validation_history,
        "full_validation": full_validation,
    }
    (output_dir / "diagnostic_history.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )

    columns = (
        "record_type",
        "optimizer_step",
        "raw_train_loss",
        "train_loss_ma50",
        "batch_events",
        "validation_loss",
        "validation_auc",
        "validation_events",
        "elapsed_seconds",
        "learning_rate",
    )
    with (output_dir / "diagnostic_history.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in train_history:
            writer.writerow(
                {
                    "record_type": "train",
                    "optimizer_step": record["optimizer_step"],
                    "raw_train_loss": record["raw_batch_train_loss"],
                    "train_loss_ma50": record["train_loss_ma50"],
                    "batch_events": record["batch_events"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "learning_rate": record["learning_rate"],
                }
            )
        for record in validation_history:
            writer.writerow(
                {
                    "record_type": "validation_subset",
                    "optimizer_step": record["optimizer_step"],
                    "validation_loss": record["validation_loss"],
                    "validation_auc": record["validation_auc"],
                    "validation_events": record["validation_events"],
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
        if full_validation is not None:
            writer.writerow(
                {
                    "record_type": "full_validation",
                    "optimizer_step": full_validation[
                        "optimizer_step"
                    ],
                    "validation_loss": full_validation[
                        "validation_loss"
                    ],
                    "validation_auc": full_validation[
                        "validation_auc"
                    ],
                    "validation_events": full_validation[
                        "validation_events"
                    ],
                    "elapsed_seconds": full_validation[
                        "elapsed_seconds"
                    ],
                }
            )


def plot_diagnostics(
    train_history: list[dict],
    validation_history: list[dict],
    output_path: Path,
    dataset_label: str,
) -> None:
    train_steps = [
        int(record["optimizer_step"]) for record in train_history
    ]
    raw_loss = [
        float(record["raw_batch_train_loss"])
        for record in train_history
    ]
    moving_loss = [
        float(record["train_loss_ma50"])
        for record in train_history
    ]
    validation_steps = [
        int(record["optimizer_step"])
        for record in validation_history
    ]
    validation_loss = [
        float(record["validation_loss"])
        for record in validation_history
    ]
    validation_auc = [
        float(record["validation_auc"])
        for record in validation_history
    ]
    best_loss_index = int(np.argmin(validation_loss))
    best_auc_index = int(np.argmax(validation_auc))
    epoch_end = train_steps[-1]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        layout="constrained",
    )
    axes[0].plot(
        train_steps,
        raw_loss,
        color="tab:blue",
        alpha=0.22,
        linewidth=0.7,
        label="Raw batch loss",
    )
    axes[0].plot(
        train_steps,
        moving_loss,
        color="tab:blue",
        linewidth=1.8,
        label="50-batch moving average",
    )
    axes[0].set_ylabel("Train BCE loss")

    axes[1].plot(
        validation_steps,
        validation_loss,
        marker="o",
        markersize=3,
        color="tab:orange",
    )
    axes[1].scatter(
        [validation_steps[best_loss_index]],
        [validation_loss[best_loss_index]],
        color="red",
        zorder=3,
        label=(
            "Best loss: "
            f"step {validation_steps[best_loss_index]}"
        ),
    )
    axes[1].set_ylabel("Validation subset BCE loss")

    axes[2].plot(
        validation_steps,
        validation_auc,
        marker="o",
        markersize=3,
        color="tab:green",
    )
    axes[2].scatter(
        [validation_steps[best_auc_index]],
        [validation_auc[best_auc_index]],
        color="red",
        zorder=3,
        label=(
            "Best AUC: "
            f"step {validation_steps[best_auc_index]}"
        ),
    )
    axes[2].set_ylabel("Validation subset ROC AUC")
    axes[2].set_xlabel("Optimizer step")

    for axis in axes:
        axis.axvline(
            0,
            color="0.4",
            linestyle=":",
            linewidth=1,
            label="Step 0",
        )
        axis.axvline(
            epoch_end,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Epoch end: {epoch_end}",
        )
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0)
        axis.legend()

    figure.suptitle(
        "TauSpin one-epoch baseline diagnostic\n"
        f"{dataset_label}"
    )
    figure.savefig(output_path)
    plt.close(figure)


def diagnostic_summary(
    configuration: dict,
    train_history: list[dict],
    validation_history: list[dict],
    full_validation: dict,
    total_seconds: float,
) -> dict:
    best_auc = max(
        validation_history, key=lambda record: record["validation_auc"]
    )
    best_loss = min(
        validation_history, key=lambda record: record["validation_loss"]
    )
    target_auc = float(best_auc["validation_auc"]) - 0.002
    first_within = next(
        record
        for record in validation_history
        if float(record["validation_auc"]) >= target_auc
    )
    adjacent = list(
        zip(validation_history[:-1], validation_history[1:])
    )
    largest_gain_start, largest_gain_end = max(
        adjacent,
        key=lambda pair: (
            float(pair[1]["validation_auc"])
            - float(pair[0]["validation_auc"])
        ),
    )
    return {
        "purpose": configuration["purpose"],
        "completed": True,
        "total_elapsed_seconds": total_seconds,
        "optimizer_steps": len(train_history),
        "train_events_seen": sum(
            int(record["batch_events"]) for record in train_history
        ),
        "step_0_validation_auc": validation_history[0][
            "validation_auc"
        ],
        "maximum_subset_validation_auc": {
            "optimizer_step": best_auc["optimizer_step"],
            "value": best_auc["validation_auc"],
        },
        "minimum_subset_validation_loss": {
            "optimizer_step": best_loss["optimizer_step"],
            "value": best_loss["validation_loss"],
        },
        "first_step_within_0p002_of_maximum_auc": {
            "optimizer_step": first_within["optimizer_step"],
            "value": first_within["validation_auc"],
            "threshold": target_auc,
        },
        "largest_adjacent_auc_gain": {
            "start_step": largest_gain_start["optimizer_step"],
            "end_step": largest_gain_end["optimizer_step"],
            "gain": (
                float(largest_gain_end["validation_auc"])
                - float(largest_gain_start["validation_auc"])
            ),
        },
        "full_validation": full_validation,
        "fixed_subset": {
            "events": validation_history[0]["validation_events"],
            "h_events": VALIDATION_PER_CLASS,
            "z_events": VALIDATION_PER_CLASS,
            "fingerprint_sha256": configuration[
                "validation_subset_fingerprint_sha256"
            ],
            "identical_across_evaluations": True,
        },
        "validation_parameter_updates_detected": False,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if arguments.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    configure_logging(arguments.output_dir)
    run_started = time.perf_counter()

    repository = Path(__file__).resolve().parent.parent
    processed_dir = arguments.processed_dir.resolve()
    metadata = json.loads(
        (processed_dir / "metadata.json").read_text()
    )
    configuration = build_configuration(
        arguments, metadata, repository
    )
    set_random_seed(RANDOM_SEED)
    device = choose_device(arguments.device)
    torch.set_float32_matmul_precision(
        "high" if arguments.tf32 else "highest"
    )

    train_dataset = TauSpinDataset(
        processed_dir,
        split="train",
        shuffle=True,
        balanced=True,
        seed=RANDOM_SEED,
    )
    validation_dataset = TauSpinDataset(
        processed_dir,
        split="validation",
        shuffle=False,
        balanced=False,
        seed=RANDOM_SEED,
    )
    fixed_events = fixed_validation_subset(processed_dir, metadata)
    subset_fingerprint = events_fingerprint(fixed_events)
    configuration["validation_subset_fingerprint_sha256"] = (
        subset_fingerprint
    )
    configuration["device"] = str(device)
    configuration["torch_version"] = str(torch.__version__)
    configuration["cuda_runtime"] = torch.version.cuda
    configuration["gpu_name"] = torch.cuda.get_device_name(0)
    (arguments.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )

    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": True,
        "collate_fn": collate_events,
    }
    if arguments.num_workers > 0:
        loader_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": arguments.prefetch_factor,
            }
        )
    train_loader = DataLoader(train_dataset, **loader_options)
    subset_loader = DataLoader(
        fixed_events,
        shuffle=False,
        **loader_options,
    )
    full_validation_loader = DataLoader(
        validation_dataset,
        **loader_options,
    )
    expected_labels = np.asarray(
        [int(event["label"]) for event in fixed_events],
        dtype=np.float32,
    )
    expected_event_numbers = np.asarray(
        [int(event["event_number"]) for event in fixed_events],
    )
    reported_steps = math.ceil(
        len(train_dataset) / arguments.batch_size
    )

    model = TauSpinTransformer(
        metadata["feature_dimensions"],
        metadata["tau_decay_num_embeddings"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_history: list[dict] = []
    validation_history: list[dict] = []
    recent_losses: deque[float] = deque(
        maxlen=MOVING_AVERAGE_WINDOW
    )

    LOGGER.info("Repository: %s", repository)
    LOGGER.info("Git HEAD: %s", configuration["git"]["head"])
    LOGGER.info(
        "Working tree clean: %s",
        configuration["git"]["working_tree_clean"],
    )
    LOGGER.info("Processed dataset: %s", processed_dir)
    LOGGER.info(
        "Events: train=%d balanced, full validation=%d, "
        "fixed validation=%d (H=%d, Z=%d)",
        len(train_dataset),
        len(validation_dataset),
        len(fixed_events),
        VALIDATION_PER_CLASS,
        VALIDATION_PER_CLASS,
    )
    LOGGER.info(
        "DataLoader reports approximately %d optimizer steps; "
        "validation interval: %d",
        reported_steps,
        VALIDATION_INTERVAL,
    )
    LOGGER.info(
        "PyTorch=%s CUDA=%s visible_devices=%d GPU=%s",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.device_count(),
        torch.cuda.get_device_name(0),
    )

    (
        subset_loss,
        subset_auc,
        subset_count,
        labels_at_step_zero,
        numbers_at_step_zero,
    ) = evaluate_without_updates(
        model,
        subset_loader,
        loss_function,
        device,
        "step 0 validation subset",
        expected_labels,
        expected_event_numbers,
    )
    validation_history.append(
        {
            "optimizer_step": 0,
            "validation_loss": subset_loss,
            "validation_auc": subset_auc,
            "validation_events": subset_count,
            "elapsed_seconds": time.perf_counter() - run_started,
        }
    )
    if not np.array_equal(labels_at_step_zero, expected_labels):
        raise RuntimeError("Step 0 validation subset labels differ")
    if not np.array_equal(
        numbers_at_step_zero, expected_event_numbers
    ):
        raise RuntimeError("Step 0 validation subset order differs")
    LOGGER.info(
        "Step %d subset validation: loss=%.6f AUC=%.6f",
        0,
        subset_loss,
        subset_auc,
    )

    train_dataset.set_epoch(0)
    model.train()
    train_events_seen = 0
    for optimizer_step, batch in enumerate(train_loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        require_finite(logits, "train logits")
        loss = loss_function(logits, batch["labels"])
        require_finite(loss, "train loss")
        loss.backward()
        require_finite_gradients(model)
        optimizer.step()

        raw_loss = float(loss.detach().cpu())
        if not math.isfinite(raw_loss):
            raise FloatingPointError("Non-finite raw train loss")
        recent_losses.append(raw_loss)
        train_events_seen += int(batch["labels"].shape[0])
        train_history.append(
            {
                "optimizer_step": optimizer_step,
                "raw_batch_train_loss": raw_loss,
                "train_loss_ma50": float(
                    np.mean(recent_losses)
                ),
                "batch_events": int(batch["labels"].shape[0]),
                "elapsed_seconds": time.perf_counter() - run_started,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )

        if optimizer_step % VALIDATION_INTERVAL == 0:
            (
                subset_loss,
                subset_auc,
                subset_count,
                _,
                _,
            ) = evaluate_without_updates(
                model,
                subset_loader,
                loss_function,
                device,
                f"step {optimizer_step} validation subset",
                expected_labels,
                expected_event_numbers,
            )
            validation_history.append(
                {
                    "optimizer_step": optimizer_step,
                    "validation_loss": subset_loss,
                    "validation_auc": subset_auc,
                    "validation_events": subset_count,
                    "elapsed_seconds": (
                        time.perf_counter() - run_started
                    ),
                }
            )
            save_history(
                arguments.output_dir,
                configuration,
                train_history,
                validation_history,
                None,
            )
            LOGGER.info(
                "Step %d/%d subset validation: "
                "loss=%.6f AUC=%.6f",
                optimizer_step,
                reported_steps,
                subset_loss,
                subset_auc,
            )

    if train_events_seen != len(train_dataset):
        raise RuntimeError(
            f"Observed {train_events_seen} training events; "
            f"expected {len(train_dataset)}"
        )
    last_step = len(train_history)
    if validation_history[-1]["optimizer_step"] != last_step:
        (
            subset_loss,
            subset_auc,
            subset_count,
            _,
            _,
        ) = evaluate_without_updates(
            model,
            subset_loader,
            loss_function,
            device,
            f"step {last_step} validation subset",
            expected_labels,
            expected_event_numbers,
        )
        validation_history.append(
            {
                "optimizer_step": last_step,
                "validation_loss": subset_loss,
                "validation_auc": subset_auc,
                "validation_events": subset_count,
                "elapsed_seconds": time.perf_counter() - run_started,
            }
        )
        LOGGER.info(
            "Step %d/%d subset validation: loss=%.6f AUC=%.6f",
            last_step,
            reported_steps,
            subset_loss,
            subset_auc,
        )

    (
        full_loss,
        full_auc,
        full_count,
        _,
        _,
    ) = evaluate_without_updates(
        model,
        full_validation_loader,
        loss_function,
        device,
        "full validation",
    )
    full_validation = {
        "optimizer_step": last_step,
        "validation_loss": full_loss,
        "validation_auc": full_auc,
        "validation_events": full_count,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    if events_fingerprint(fixed_events) != subset_fingerprint:
        raise RuntimeError(
            "Fixed validation subset content changed during the run"
        )

    total_seconds = time.perf_counter() - run_started
    summary = diagnostic_summary(
        configuration,
        train_history,
        validation_history,
        full_validation,
        total_seconds,
    )
    save_history(
        arguments.output_dir,
        configuration,
        train_history,
        validation_history,
        full_validation,
    )
    plot_diagnostics(
        train_history,
        validation_history,
        arguments.output_dir / "diagnostic_summary.pdf",
        arguments.dataset_label,
    )
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    torch.save(
        {
            "optimizer_step": last_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "configuration": configuration,
            "dataset_metadata": metadata,
            "summary": summary,
        },
        arguments.output_dir / "checkpoint_last.pt",
    )
    worker_shutdown = {
        "train": shutdown_loader_workers(train_loader),
        "subset_validation": shutdown_loader_workers(subset_loader),
        "full_validation": shutdown_loader_workers(
            full_validation_loader
        ),
    }
    (arguments.output_dir / "worker_shutdown.json").write_text(
        json.dumps(worker_shutdown, indent=2) + "\n"
    )
    LOGGER.info(
        "Full validation: loss=%.6f AUC=%.6f events=%d",
        full_loss,
        full_auc,
        full_count,
    )
    LOGGER.info(
        "Completed %d optimizer steps in %.1f seconds. "
        "Outputs: %s",
        last_step,
        total_seconds,
        arguments.output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("One-epoch diagnostic failed")
        raise
