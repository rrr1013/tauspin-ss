from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterator, Mapping

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import (
    LEARNING_RATE,
    OUTPUT_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import TauSpinDataset, collate_events
from hpo_utils import (
    configure_tf32,
    create_model,
    evaluate_model,
    json_ready,
    shutdown_loader_workers,
    strip_evaluation_arrays,
)
from train import (
    choose_device,
    move_batch,
    require_finite,
    require_finite_gradients,
    set_random_seed,
)


LOGGER = logging.getLogger("diagnose_cross_evaluation")
DOMAINS = ("raw", "ptmatched20")
FEATURE_GROUPS = ("event", "tau", "track", "pfo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Raw and pT-matched baselines for a fixed number of "
            "optimizer steps and evaluate both models on both validation "
            "populations."
        )
    )
    parser.add_argument("--raw-processed-dir", type=Path, required=True)
    parser.add_argument("--matched-processed-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUTPUT_DIR
            / "diagnostics"
            / "fixed-partial-v1-raw-vs-ptmatched20-5000-step-2x2-v1"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
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


def run_command(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.rstrip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(json_ready(document), indent=2) + "\n")


def validate_inputs(
    processed_dirs: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_metadata = metadata["raw"]
    matched_metadata = metadata["ptmatched20"]
    compatibility_keys = (
        "feature_names",
        "feature_dimensions",
        "tau_decay_mode_to_id",
        "tau_decay_unknown_id",
        "tau_decay_num_embeddings",
        "split_identity",
    )
    for key in compatibility_keys:
        if raw_metadata[key] != matched_metadata[key]:
            raise ValueError(f"Dataset metadata differs for {key}")

    for domain in DOMAINS:
        if "test" not in metadata[domain]["counts"]:
            raise ValueError(f"{domain} metadata lacks the test count audit")
        if not metadata[domain]["shards"]["validation"]:
            raise ValueError(f"{domain} validation shards are missing")

    stat_checks: dict[str, Any] = {}
    for group in FEATURE_GROUPS:
        source = stats["raw"][group]
        target = stats["ptmatched20"][group]
        for key in ("names", "standardize"):
            if source[key] != target[key]:
                raise ValueError(f"{group} statistics differ for {key}")
        source_mean = np.asarray(source["mean"], dtype=np.float64)
        source_std = np.asarray(source["std"], dtype=np.float64)
        target_mean = np.asarray(target["mean"], dtype=np.float64)
        target_std = np.asarray(target["std"], dtype=np.float64)
        if not all(
            np.isfinite(values).all()
            for values in (
                source_mean,
                source_std,
                target_mean,
                target_std,
            )
        ):
            raise ValueError(f"Non-finite {group} normalization statistics")
        if np.any(source_std <= 0) or np.any(target_std <= 0):
            raise ValueError(f"Non-positive {group} standard deviation")

        probe = np.linspace(
            -3.0, 3.0, source_mean.size, dtype=np.float64
        )
        converted = (
            probe * source_std + source_mean - target_mean
        ) / target_std
        round_trip = (
            converted * target_std + target_mean - source_mean
        ) / source_std
        maximum_error = float(np.max(np.abs(probe - round_trip)))
        if maximum_error > 1.0e-12:
            raise RuntimeError(
                f"{group} normalization round trip failed: {maximum_error}"
            )
        stat_checks[group] = {
            "names_equal": True,
            "standardization_mask_equal": True,
            "round_trip_max_abs_error_float64": maximum_error,
            "maximum_absolute_mean_difference": float(
                np.max(np.abs(source_mean - target_mean))
            ),
            "maximum_relative_std_difference": float(
                np.max(np.abs(source_std / target_std - 1.0))
            ),
        }

    return {
        "processed_dirs": {
            key: str(path.resolve()) for key, path in processed_dirs.items()
        },
        "metadata_sha256": {
            key: sha256_file(path / "metadata.json")
            for key, path in processed_dirs.items()
        },
        "stats_sha256": {
            key: sha256_file(path / "stats.json")
            for key, path in processed_dirs.items()
        },
        "metadata_compatible": True,
        "statistics_checks": stat_checks,
        "test_split_loaded": False,
    }


class ReStandardizedDataset(IterableDataset):
    """Read one processed dataset using another train split's statistics."""

    def __init__(
        self,
        source_processed_dir: Path,
        target_stats: Mapping[str, Mapping[str, Any]],
        *,
        split: str = "validation",
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        if split != "validation":
            raise ValueError("Cross-evaluation may only load validation")
        self.base = TauSpinDataset(
            source_processed_dir,
            split=split,
            shuffle=False,
            balanced=False,
            seed=seed,
        )
        source_stats = load_json(source_processed_dir / "stats.json")
        self.conversion: dict[str, tuple[torch.Tensor, ...]] = {}
        for group in FEATURE_GROUPS:
            source = source_stats[group]
            target = target_stats[group]
            self.conversion[group] = (
                torch.tensor(source["mean"], dtype=torch.float32),
                torch.tensor(source["std"], dtype=torch.float32),
                torch.tensor(target["mean"], dtype=torch.float32),
                torch.tensor(target["std"], dtype=torch.float32),
            )

    def __len__(self) -> int:
        return len(self.base)

    def convert(
        self, values: torch.Tensor, group: str
    ) -> torch.Tensor:
        source_mean, source_std, target_mean, target_std = self.conversion[
            group
        ]
        return (
            values * source_std + source_mean - target_mean
        ) / target_std

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for event in self.base:
            converted = dict(event)
            converted["event_features"] = self.convert(
                event["event_features"], "event"
            )
            converted["tau_features"] = self.convert(
                event["tau_features"], "tau"
            )
            converted["track_features"] = self.convert(
                event["track_features"], "track"
            )
            converted["pfo_features"] = self.convert(
                event["pfo_features"], "pfo"
            )
            yield converted


def loader_options(arguments: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": True,
        "collate_fn": collate_events,
        "drop_last": False,
    }
    if arguments.num_workers > 0:
        options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": arguments.prefetch_factor,
            }
        )
    return options


def create_validation_loaders(
    processed_dirs: Mapping[str, Path],
    target_domain: str,
    stats: Mapping[str, Mapping[str, Any]],
    arguments: argparse.Namespace,
) -> dict[str, DataLoader]:
    loaders = {}
    options = loader_options(arguments)
    for evaluation_domain in DOMAINS:
        source_dir = processed_dirs[evaluation_domain]
        if evaluation_domain == target_domain:
            dataset: IterableDataset = TauSpinDataset(
                source_dir,
                split="validation",
                shuffle=False,
                balanced=False,
                seed=RANDOM_SEED,
            )
        else:
            dataset = ReStandardizedDataset(
                source_dir,
                stats[target_domain],
                split="validation",
                seed=RANDOM_SEED,
            )
        loaders[evaluation_domain] = DataLoader(dataset, **options)
    return loaders


def create_train_loader(
    processed_dir: Path,
    epoch: int,
    arguments: argparse.Namespace,
) -> tuple[TauSpinDataset, DataLoader, Any]:
    dataset = TauSpinDataset(
        processed_dir,
        split="train",
        shuffle=True,
        balanced=True,
        seed=RANDOM_SEED,
    )
    dataset.set_epoch(epoch)
    loader = DataLoader(dataset, **loader_options(arguments))
    iterator = iter(loader)
    return dataset, loader, iterator


def checkpoint_document(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_domain: str,
    metadata: Mapping[str, Any],
    arguments: argparse.Namespace,
    step: int,
    events_seen: int,
    validation_metrics: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "purpose": (
            "Fixed-partial Raw/pT-matched 2x2 saturation diagnostic; "
            "not a final physics result."
        ),
        "training_domain": training_domain,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dimensions": metadata["feature_dimensions"],
        "tau_decay_num_embeddings": metadata[
            "tau_decay_num_embeddings"
        ],
        "tau_decay_mode_to_id": metadata["tau_decay_mode_to_id"],
        "model_profile": "current",
        "hyperparameters": {
            "learning_rate": arguments.learning_rate,
            "dropout": arguments.dropout,
            "weight_decay": arguments.weight_decay,
            "scheduler": "constant",
            "warmup_steps": 0,
            "batch_size": arguments.batch_size,
            "balanced_sampling": True,
            "seed": RANDOM_SEED,
        },
        "training_state": {
            "optimizer_step": step,
            "events_seen": events_seen,
        },
        "validation_metrics": dict(validation_metrics),
        "normalization_target_domain": training_domain,
        "input_audit": dict(input_audit),
        "test_split_loaded": False,
    }


def evaluate_both(
    *,
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    loss_function: nn.Module,
    device: torch.device,
    training_domain: str,
    step: int,
    events_seen: int,
    epoch_equivalent: float,
) -> list[dict[str, Any]]:
    records = []
    for evaluation_domain in DOMAINS:
        metrics = evaluate_model(
            model,
            loaders[evaluation_domain],
            loss_function,
            device,
            (
                f"{training_domain}-trained model on "
                f"{evaluation_domain} validation"
            ),
            verify_parameters_unchanged=True,
        )
        record = {
            "training_domain": training_domain,
            "evaluation_domain": evaluation_domain,
            "optimizer_step": step,
            "events_seen": events_seen,
            "epoch_equivalent": epoch_equivalent,
            **strip_evaluation_arrays(metrics),
        }
        records.append(record)
        LOGGER.info(
            "train=%s step=%d eval=%s auc=%.6f loss=%.6f events=%d",
            training_domain,
            step,
            evaluation_domain,
            metrics["auc"],
            metrics["loss"],
            metrics["event_count"],
        )
    return records


def train_domain_model(
    *,
    training_domain: str,
    processed_dirs: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
    arguments: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    domain_dir = output_dir / f"train_{training_domain}"
    domain_dir.mkdir(parents=True, exist_ok=False)
    set_random_seed(RANDOM_SEED)
    model, parameter_counts = create_model(
        metadata[training_domain],
        "current",
        arguments.dropout,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    validation_loaders = create_validation_loaders(
        processed_dirs, training_domain, stats, arguments
    )
    train_length = 2 * max(
        int(metadata[training_domain]["counts"]["train"]["H"]),
        int(metadata[training_domain]["counts"]["train"]["Z"]),
    )
    train_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    recent_losses: deque[float] = deque(maxlen=50)
    class_counts: Counter[int] = Counter()
    worker_shutdown: list[dict[str, Any]] = []
    events_seen = 0
    valid_tokens_seen = 0
    step = 0
    epoch = 0
    best_native_auc = -math.inf
    best_native_step: int | None = None
    started = time.perf_counter()
    train_loader: DataLoader | None = None
    train_iterator: Any | None = None

    LOGGER.info(
        "Starting %s training: train_events_per_epoch=%d max_steps=%d",
        training_domain,
        train_length,
        arguments.max_steps,
    )
    initial_records = evaluate_both(
        model=model,
        loaders=validation_loaders,
        loss_function=loss_function,
        device=device,
        training_domain=training_domain,
        step=0,
        events_seen=0,
        epoch_equivalent=0.0,
    )
    validation_history.extend(initial_records)

    try:
        while step < arguments.max_steps:
            train_dataset, train_loader, train_iterator = create_train_loader(
                processed_dirs[training_domain], epoch, arguments
            )
            batches_this_epoch = 0
            while step < arguments.max_steps:
                try:
                    cpu_batch = next(train_iterator)
                except StopIteration:
                    break
                batch = move_batch(cpu_batch, device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                require_finite(logits, "train logits")
                loss = loss_function(logits, batch["labels"])
                require_finite(loss, "train loss")
                loss.backward()
                require_finite_gradients(model)
                optimizer.step()

                step += 1
                batches_this_epoch += 1
                batch_events = int(batch["labels"].shape[0])
                batch_tokens = (
                    int((~batch["padding_mask"]).sum().item())
                    + batch_events
                )
                events_seen += batch_events
                valid_tokens_seen += batch_tokens
                labels = (
                    batch["labels"].detach().to(torch.int64).cpu().tolist()
                )
                class_counts.update(labels)
                batch_loss = float(loss.detach().cpu())
                recent_losses.append(batch_loss)
                elapsed = time.perf_counter() - started
                train_history.append(
                    {
                        "training_domain": training_domain,
                        "optimizer_step": step,
                        "epoch_index": epoch,
                        "epoch_equivalent": events_seen / train_length,
                        "events_seen": events_seen,
                        "h_seen": class_counts[1],
                        "z_seen": class_counts[0],
                        "valid_tokens_seen": valid_tokens_seen,
                        "batch_train_loss": batch_loss,
                        "train_loss_ma50": float(np.mean(recent_losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": elapsed,
                    }
                )

                if (
                    step % arguments.eval_every_steps == 0
                    or step == arguments.max_steps
                ):
                    records = evaluate_both(
                        model=model,
                        loaders=validation_loaders,
                        loss_function=loss_function,
                        device=device,
                        training_domain=training_domain,
                        step=step,
                        events_seen=events_seen,
                        epoch_equivalent=events_seen / train_length,
                    )
                    validation_history.extend(records)
                    native = next(
                        record
                        for record in records
                        if record["evaluation_domain"] == training_domain
                    )
                    if float(native["auc"]) > best_native_auc:
                        best_native_auc = float(native["auc"])
                        best_native_step = step
                        torch.save(
                            checkpoint_document(
                                model=model,
                                optimizer=optimizer,
                                training_domain=training_domain,
                                metadata=metadata[training_domain],
                                arguments=arguments,
                                step=step,
                                events_seen=events_seen,
                                validation_metrics=native,
                                input_audit=input_audit,
                            ),
                            domain_dir / "best_native_auc_model.pt",
                        )

            shutdown = shutdown_loader_workers(
                train_loader, train_iterator
            )
            shutdown["epoch_index"] = epoch
            shutdown["batches"] = batches_this_epoch
            worker_shutdown.append(shutdown)
            train_loader = None
            train_iterator = None
            if batches_this_epoch == 0:
                raise RuntimeError("Training epoch produced no batches")
            epoch += 1
    finally:
        if train_loader is not None:
            shutdown = shutdown_loader_workers(
                train_loader, train_iterator
            )
            shutdown["epoch_index"] = epoch
            worker_shutdown.append(shutdown)

    validation_worker_shutdown = {
        domain: shutdown_loader_workers(loader)
        for domain, loader in validation_loaders.items()
    }
    last_records = [
        record
        for record in validation_history
        if int(record["optimizer_step"]) == arguments.max_steps
    ]
    torch.save(
        checkpoint_document(
            model=model,
            optimizer=optimizer,
            training_domain=training_domain,
            metadata=metadata[training_domain],
            arguments=arguments,
            step=arguments.max_steps,
            events_seen=events_seen,
            validation_metrics={
                record["evaluation_domain"]: record
                for record in last_records
            },
            input_audit=input_audit,
        ),
        domain_dir / "last_model.pt",
    )
    if best_native_step is None:
        raise RuntimeError("No native validation checkpoint was selected")

    result = {
        "training_domain": training_domain,
        "optimizer_steps": step,
        "epochs_started": epoch,
        "events_seen": events_seen,
        "class_counts": dict(class_counts),
        "valid_tokens_seen": valid_tokens_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_counts": parameter_counts,
        "best_native_auc": best_native_auc,
        "best_native_auc_step": best_native_step,
        "train_worker_shutdown": worker_shutdown,
        "validation_worker_shutdown": validation_worker_shutdown,
        "finite_training": True,
        "test_split_loaded": False,
    }
    write_json(
        domain_dir / "history.json",
        {
            "train_steps": train_history,
            "full_validation": validation_history,
        },
    )
    write_json(domain_dir / "metrics.json", result)
    return {
        "metrics": result,
        "train_history": train_history,
        "validation_history": validation_history,
    }


def summarize_series(records: list[dict[str, Any]]) -> dict[str, Any]:
    best_auc = max(records, key=lambda record: float(record["auc"]))
    minimum_loss = min(records, key=lambda record: float(record["loss"]))
    final = max(records, key=lambda record: int(record["optimizer_step"]))
    within_threshold = [
        record
        for record in records
        if float(record["auc"]) >= float(best_auc["auc"]) - 0.002
    ]
    record_by_step = {
        int(record["optimizer_step"]): record for record in records
    }
    penultimate = record_by_step.get(
        int(final["optimizer_step"]) - 1000
    )
    return {
        "best_auc": float(best_auc["auc"]),
        "best_auc_step": int(best_auc["optimizer_step"]),
        "minimum_loss": float(minimum_loss["loss"]),
        "minimum_loss_step": int(minimum_loss["optimizer_step"]),
        "final_auc": float(final["auc"]),
        "final_loss": float(final["loss"]),
        "first_step_within_0p002_of_best_auc": int(
            min(
                within_threshold,
                key=lambda record: int(record["optimizer_step"]),
            )["optimizer_step"]
        ),
        "auc_change_final_1000_steps": (
            None
            if penultimate is None
            else float(final["auc"]) - float(penultimate["auc"])
        ),
        "validation_events": int(final["event_count"]),
        "label_counts": dict(final["label_counts"]),
    }


def save_csv(
    path: Path,
    train_results: Mapping[str, Mapping[str, Any]],
) -> None:
    columns = (
        "training_domain",
        "evaluation_domain",
        "optimizer_step",
        "epoch_equivalent",
        "events_seen",
        "loss",
        "auc",
        "event_count",
        "elapsed_seconds",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for training_domain in DOMAINS:
            for record in train_results[training_domain][
                "validation_history"
            ]:
                writer.writerow(
                    {key: record.get(key) for key in columns}
                )


def save_plot(
    path: Path,
    train_results: Mapping[str, Mapping[str, Any]],
) -> None:
    colors = {"raw": "tab:blue", "ptmatched20": "tab:orange"}
    styles = {"raw": "-", "ptmatched20": "--"}
    labels = {"raw": "Raw", "ptmatched20": "pT-matched"}
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 4.8), layout="constrained"
    )
    for training_domain in DOMAINS:
        train_history = train_results[training_domain]["train_history"]
        axes[0].plot(
            [record["optimizer_step"] for record in train_history],
            [record["train_loss_ma50"] for record in train_history],
            color=colors[training_domain],
            label=f"{labels[training_domain]} train",
        )
        for evaluation_domain in DOMAINS:
            records = [
                record
                for record in train_results[training_domain][
                    "validation_history"
                ]
                if record["evaluation_domain"] == evaluation_domain
            ]
            label = (
                f"{labels[training_domain]} train"
                f" \u2192 {labels[evaluation_domain]} val"
            )
            axes[1].plot(
                [record["optimizer_step"] for record in records],
                [record["loss"] for record in records],
                color=colors[training_domain],
                linestyle=styles[evaluation_domain],
                marker="o",
                label=label,
            )
            axes[2].plot(
                [record["optimizer_step"] for record in records],
                [record["auc"] for record in records],
                color=colors[training_domain],
                linestyle=styles[evaluation_domain],
                marker="o",
                label=label,
            )

    axes[0].set_title("Training loss")
    axes[0].set_ylabel("BCE loss (moving average, 50 steps)")
    axes[1].set_title("Full-validation loss")
    axes[1].set_ylabel("BCE loss")
    axes[2].set_title("Full-validation ROC AUC")
    axes[2].set_ylabel("ROC AUC")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Raw / pT-matched 2\u00d72 cross-evaluation (Current Transformer)"
    )
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    if arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if arguments.eval_every_steps <= 0:
        raise ValueError("--eval-every-steps must be positive")
    if arguments.max_steps % arguments.eval_every_steps:
        raise ValueError(
            "--max-steps must be divisible by --eval-every-steps"
        )
    configure_logging(arguments.output_dir)
    repository = Path(__file__).resolve().parent.parent
    processed_dirs = {
        "raw": arguments.raw_processed_dir.resolve(),
        "ptmatched20": arguments.matched_processed_dir.resolve(),
    }
    metadata = {
        key: load_json(path / "metadata.json")
        for key, path in processed_dirs.items()
    }
    stats = {
        key: load_json(path / "stats.json")
        for key, path in processed_dirs.items()
    }
    input_audit = validate_inputs(processed_dirs, metadata, stats)
    set_random_seed(RANDOM_SEED)
    device = choose_device(arguments.device)
    runtime = configure_tf32()
    runtime.update(
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_cuda_devices": torch.cuda.device_count(),
        }
    )
    git = {
        "head": run_command(["git", "rev-parse", "HEAD"], repository),
        "status_short": run_command(
            ["git", "status", "--short"], repository
        ).splitlines(),
        "diff_stat": run_command(
            ["git", "diff", "--stat"], repository
        ).splitlines(),
    }
    config = {
        "purpose": (
            "Compare Raw and 20 GeV pT-matched training with a 2x2 "
            "cross-evaluation before HPO; partial sample only."
        ),
        "max_optimizer_steps": arguments.max_steps,
        "full_validation_every_steps": arguments.eval_every_steps,
        "model_profile": "current",
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "dropout": arguments.dropout,
        "weight_decay": arguments.weight_decay,
        "scheduler": "constant",
        "warmup_steps": 0,
        "balanced_sampling": True,
        "seed": RANDOM_SEED,
        "num_workers": arguments.num_workers,
        "prefetch_factor": arguments.prefetch_factor,
        "runtime": runtime,
        "git": git,
        "input_audit": input_audit,
        "test_split_loaded": False,
    }
    write_json(arguments.output_dir / "config.json", config)
    LOGGER.info("Repository: %s", repository)
    LOGGER.info("Git HEAD: %s", git["head"])
    LOGGER.info("GPU: %s", runtime["gpu_name"])
    LOGGER.info("Input compatibility and normalization checks passed")

    train_results = {}
    for training_domain in DOMAINS:
        train_results[training_domain] = train_domain_model(
            training_domain=training_domain,
            processed_dirs=processed_dirs,
            metadata=metadata,
            stats=stats,
            arguments=arguments,
            device=device,
            output_dir=arguments.output_dir,
            input_audit=input_audit,
        )

    series_summary = {}
    for training_domain in DOMAINS:
        for evaluation_domain in DOMAINS:
            records = [
                record
                for record in train_results[training_domain][
                    "validation_history"
                ]
                if record["evaluation_domain"] == evaluation_domain
            ]
            series_summary[
                f"{training_domain}_train_to_{evaluation_domain}_validation"
            ] = summarize_series(records)

    summary = {
        "series": series_summary,
        "training_runs": {
            domain: train_results[domain]["metrics"]
            for domain in DOMAINS
        },
        "native_final_auc_difference_matched_minus_raw": (
            series_summary[
                "ptmatched20_train_to_ptmatched20_validation"
            ]["final_auc"]
            - series_summary["raw_train_to_raw_validation"]["final_auc"]
        ),
        "raw_model_final_auc_change_when_evaluated_on_matched": (
            series_summary[
                "raw_train_to_ptmatched20_validation"
            ]["final_auc"]
            - series_summary["raw_train_to_raw_validation"]["final_auc"]
        ),
        "matched_model_final_auc_change_when_evaluated_on_raw": (
            series_summary[
                "ptmatched20_train_to_raw_validation"
            ]["final_auc"]
            - series_summary[
                "ptmatched20_train_to_ptmatched20_validation"
            ]["final_auc"]
        ),
        "test_split_loaded": False,
    }
    save_csv(arguments.output_dir / "cross_evaluation_history.csv", train_results)
    save_plot(arguments.output_dir / "cross_evaluation_curves", train_results)
    write_json(arguments.output_dir / "summary.json", summary)
    LOGGER.info("Finished 2x2 diagnostic: %s", arguments.output_dir)
    LOGGER.info("Summary: %s", summary["series"])


if __name__ == "__main__":
    main()
