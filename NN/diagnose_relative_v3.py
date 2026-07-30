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
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import LEARNING_RATE, OUTPUT_DIR, RANDOM_SEED, WEIGHT_DECAY
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


LOGGER = logging.getLogger("diagnose_relative_v3")
FEATURE_SETS = ("absolute-v1", "absolute-plus-parent-relative-v3")
RELATIVE_TRACK_FEATURES = (
    "track_dEta",
    "sin_track_dPhi",
    "cos_track_dPhi",
    "log1p_track_ptFraction",
)
RELATIVE_PFO_FEATURES = (
    "pfo_dEta",
    "sin_pfo_dPhi",
    "cos_pfo_dPhi",
    "log1p_pfo_ptFraction",
)
RELATIVE_EVENT_FEATURES = (
    "abs_tau_pair_dEta",
    "sin_tau_pair_dPhi",
    "cos_tau_pair_dPhi",
    "log_tau_minus_over_plus_pt",
    "sin_met_tau_minus_dPhi",
    "cos_met_tau_minus_dPhi",
    "sin_met_tau_plus_dPhi",
    "cos_met_tau_plus_dPhi",
    "met_over_tau_pair_pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Current Transformer with and without Relative-v3 "
            "constituent, tau-pair, and MET features."
        )
    )
    parser.add_argument("--absolute-processed-dir", type=Path, required=True)
    parser.add_argument("--relative-processed-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            OUTPUT_DIR
            / "diagnostics"
            / "fixed-partial-v1-ptmatched20-relative-v3-7000-step-v1"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=7000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(json_ready(document), indent=2) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.rstrip()


def validate_inputs(
    processed_dirs: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    absolute = metadata["absolute-v1"]
    relative = metadata["absolute-plus-parent-relative-v3"]
    for key in (
        "counts",
        "labels",
        "split_fractions",
        "split_identity",
        "tau_decay_mode_to_id",
        "tau_decay_unknown_id",
        "tau_decay_num_embeddings",
    ):
        if absolute[key] != relative[key]:
            raise ValueError(f"Dataset metadata differs for {key}")

    if relative.get("feature_set") != "absolute-plus-parent-relative-v3":
        raise ValueError("Relative dataset has the wrong feature_set")
    absolute_feature_set = absolute.get("feature_set", "absolute-v1")
    if absolute_feature_set != "absolute-v1":
        raise ValueError("Absolute dataset has the wrong feature_set")

    if relative["feature_names"]["event"] != [
        *absolute["feature_names"]["event"],
        *RELATIVE_EVENT_FEATURES,
    ]:
        raise ValueError("Unexpected Relative-v3 event feature schema")
    if absolute["feature_names"]["tau"] != relative["feature_names"]["tau"]:
        raise ValueError("Unexpected tau feature difference")
    if relative["feature_names"]["track"] != [
        *absolute["feature_names"]["track"],
        *RELATIVE_TRACK_FEATURES,
    ]:
        raise ValueError("Unexpected relative track feature schema")
    if relative["feature_names"]["pfo"] != [
        *absolute["feature_names"]["pfo"],
        *RELATIVE_PFO_FEATURES,
    ]:
        raise ValueError("Unexpected relative PFO feature schema")

    for feature_set in FEATURE_SETS:
        for group in ("event", "tau", "track", "pfo"):
            group_stats = stats[feature_set][group]
            if group_stats["names"] != metadata[feature_set][
                "feature_names"
            ][group]:
                raise ValueError(
                    f"{feature_set} {group} stats/metadata mismatch"
                )
            mean = np.asarray(group_stats["mean"], dtype=np.float64)
            std = np.asarray(group_stats["std"], dtype=np.float64)
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError(f"Non-finite {feature_set} {group} stats")
            if np.any(std <= 0):
                raise ValueError(
                    f"Non-positive {feature_set} {group} standard deviation"
                )

    absolute_selection = absolute.get("event_selection")
    relative_selection = relative.get("event_selection")
    if not absolute_selection or not relative_selection:
        raise ValueError("Both datasets must record event selection")
    if (
        absolute_selection["manifest_sha256"]
        != relative_selection["manifest_sha256"]
    ):
        raise ValueError("Matching manifest differs between feature sets")

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
        "matching_manifest_sha256": absolute_selection["manifest_sha256"],
        "counts_equal": True,
        "base_feature_schema_equal": True,
        "relative_v3_feature_schema_verified": True,
        "test_split_loaded": False,
    }


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


def create_loader(
    processed_dir: Path,
    split: str,
    arguments: argparse.Namespace,
    *,
    epoch: int = 0,
) -> tuple[TauSpinDataset, DataLoader]:
    dataset = TauSpinDataset(
        processed_dir,
        split=split,
        shuffle=split == "train",
        balanced=split == "train",
        seed=RANDOM_SEED,
    )
    dataset.set_epoch(epoch)
    return dataset, DataLoader(dataset, **loader_options(arguments))


def checkpoint_document(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    feature_set: str,
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
            "Relative-v3 ablation on the fixed partial 20 GeV pT-matched "
            "sample; not a final physics result."
        ),
        "feature_set": feature_set,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "feature_dimensions": metadata["feature_dimensions"],
        "feature_names": metadata["feature_names"],
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
        "input_audit": dict(input_audit),
        "test_split_loaded": False,
    }


def train_feature_set(
    *,
    feature_set: str,
    processed_dir: Path,
    metadata: Mapping[str, Any],
    arguments: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    feature_dir = output_dir / feature_set
    feature_dir.mkdir(parents=True, exist_ok=False)
    set_random_seed(RANDOM_SEED)
    model, parameter_counts = create_model(
        metadata, "current", arguments.dropout, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    _, validation_loader = create_loader(
        processed_dir, "validation", arguments
    )
    train_length = 2 * max(
        int(metadata["counts"]["train"]["H"]),
        int(metadata["counts"]["train"]["Z"]),
    )
    train_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    recent_losses: deque[float] = deque(maxlen=50)
    class_counts: Counter[int] = Counter()
    train_worker_shutdown: list[dict[str, Any]] = []
    events_seen = 0
    valid_tokens_seen = 0
    step = 0
    epoch = 0
    best_auc = -math.inf
    best_step: int | None = None
    started = time.perf_counter()

    def evaluate(step_value: int) -> dict[str, Any]:
        metrics = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
            f"{feature_set} validation",
            verify_parameters_unchanged=True,
        )
        record = {
            "feature_set": feature_set,
            "optimizer_step": step_value,
            "events_seen": events_seen,
            "epoch_equivalent": events_seen / train_length,
            **strip_evaluation_arrays(metrics),
        }
        validation_history.append(record)
        LOGGER.info(
            "feature_set=%s step=%d auc=%.6f loss=%.6f",
            feature_set,
            step_value,
            metrics["auc"],
            metrics["loss"],
        )
        return record

    evaluate(0)
    while step < arguments.max_steps:
        _, train_loader = create_loader(
            processed_dir, "train", arguments, epoch=epoch
        )
        train_iterator = iter(train_loader)
        batches_this_epoch = 0
        try:
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
                events_seen += batch_events
                valid_tokens_seen += (
                    int((~batch["padding_mask"]).sum().item())
                    + batch_events
                )
                class_counts.update(
                    batch["labels"]
                    .detach()
                    .to(torch.int64)
                    .cpu()
                    .tolist()
                )
                batch_loss = float(loss.detach().cpu())
                recent_losses.append(batch_loss)
                train_history.append(
                    {
                        "feature_set": feature_set,
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
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if (
                    step % arguments.eval_every_steps == 0
                    or step == arguments.max_steps
                ):
                    record = evaluate(step)
                    if float(record["auc"]) > best_auc:
                        best_auc = float(record["auc"])
                        best_step = step
                        torch.save(
                            checkpoint_document(
                                model=model,
                                optimizer=optimizer,
                                feature_set=feature_set,
                                metadata=metadata,
                                arguments=arguments,
                                step=step,
                                events_seen=events_seen,
                                validation_metrics=record,
                                input_audit=input_audit,
                            ),
                            feature_dir / "best_validation_auc_model.pt",
                        )
        finally:
            shutdown = shutdown_loader_workers(
                train_loader, train_iterator
            )
            shutdown["epoch_index"] = epoch
            shutdown["batches"] = batches_this_epoch
            train_worker_shutdown.append(shutdown)
        if batches_this_epoch == 0:
            raise RuntimeError("Training epoch produced no batches")
        epoch += 1

    validation_worker_shutdown = shutdown_loader_workers(validation_loader)
    last_validation = validation_history[-1]
    torch.save(
        checkpoint_document(
            model=model,
            optimizer=optimizer,
            feature_set=feature_set,
            metadata=metadata,
            arguments=arguments,
            step=step,
            events_seen=events_seen,
            validation_metrics=last_validation,
            input_audit=input_audit,
        ),
        feature_dir / "last_model.pt",
    )
    if best_step is None:
        raise RuntimeError("No best validation checkpoint was selected")

    reloaded, _ = create_model(
        metadata, "current", arguments.dropout, device
    )
    saved = torch.load(
        feature_dir / "best_validation_auc_model.pt",
        map_location=device,
        weights_only=True,
    )
    reloaded.load_state_dict(saved["model_state_dict"])
    _, reload_loader = create_loader(
        processed_dir, "validation", arguments
    )
    reload_metrics = evaluate_model(
        reloaded,
        reload_loader,
        loss_function,
        device,
        f"{feature_set} reloaded best validation",
        verify_parameters_unchanged=True,
    )
    reload_worker_shutdown = shutdown_loader_workers(reload_loader)
    reload_auc_difference = float(reload_metrics["auc"]) - best_auc
    if abs(reload_auc_difference) > 1.0e-12:
        raise RuntimeError(
            "Reloaded checkpoint AUC differs from saved best AUC: "
            f"{reload_auc_difference}"
        )

    result = {
        "feature_set": feature_set,
        "optimizer_steps": step,
        "events_seen": events_seen,
        "class_counts": dict(class_counts),
        "valid_tokens_seen": valid_tokens_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_counts": parameter_counts,
        "best_validation_auc": best_auc,
        "best_validation_auc_step": best_step,
        "reloaded_best_auc": float(reload_metrics["auc"]),
        "reloaded_auc_difference": reload_auc_difference,
        "train_worker_shutdown": train_worker_shutdown,
        "validation_worker_shutdown": validation_worker_shutdown,
        "reload_worker_shutdown": reload_worker_shutdown,
        "finite_training": True,
        "test_split_loaded": False,
    }
    write_json(
        feature_dir / "history.json",
        {
            "train_steps": train_history,
            "full_validation": validation_history,
        },
    )
    write_json(feature_dir / "metrics.json", result)
    return {
        "metrics": result,
        "train_history": train_history,
        "validation_history": validation_history,
    }


def summarize(
    records: list[dict[str, Any]], max_steps: int
) -> dict[str, Any]:
    trained = [
        record for record in records if int(record["optimizer_step"]) > 0
    ]
    best = max(trained, key=lambda record: float(record["auc"]))
    minimum_loss = min(trained, key=lambda record: float(record["loss"]))
    final = max(trained, key=lambda record: int(record["optimizer_step"]))
    late_steps = (max_steps - 1000, max_steps - 500, max_steps)
    by_step = {
        int(record["optimizer_step"]): record for record in trained
    }
    if not all(step in by_step for step in late_steps):
        raise ValueError(f"Missing late validation steps: {late_steps}")
    late_auc = [float(by_step[step]["auc"]) for step in late_steps]
    return {
        "best_auc": float(best["auc"]),
        "best_auc_step": int(best["optimizer_step"]),
        "minimum_loss": float(minimum_loss["loss"]),
        "minimum_loss_step": int(minimum_loss["optimizer_step"]),
        "final_auc": float(final["auc"]),
        "final_loss": float(final["loss"]),
        "late_steps": list(late_steps),
        "late_auc": late_auc,
        "late_auc_mean": float(np.mean(late_auc)),
        "validation_events": int(final["event_count"]),
        "label_counts": dict(final["label_counts"]),
    }


def save_csv(
    path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    columns = (
        "feature_set",
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
        for feature_set in FEATURE_SETS:
            for record in results[feature_set]["validation_history"]:
                writer.writerow({key: record.get(key) for key in columns})


def save_plot(
    path: Path,
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    colors = {
        "absolute-v1": "tab:blue",
        "absolute-plus-parent-relative-v3": "tab:orange",
    }
    labels = {
        "absolute-v1": "Absolute baseline",
        "absolute-plus-parent-relative-v3": "Relative-v3",
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 4.8), layout="constrained"
    )
    for feature_set in FEATURE_SETS:
        train_history = results[feature_set]["train_history"]
        validation_history = results[feature_set]["validation_history"]
        axes[0].plot(
            [record["optimizer_step"] for record in train_history],
            [record["train_loss_ma50"] for record in train_history],
            color=colors[feature_set],
            label=labels[feature_set],
        )
        axes[1].plot(
            [record["optimizer_step"] for record in validation_history],
            [record["loss"] for record in validation_history],
            color=colors[feature_set],
            marker="o",
            label=labels[feature_set],
        )
        axes[2].plot(
            [record["optimizer_step"] for record in validation_history],
            [record["auc"] for record in validation_history],
            color=colors[feature_set],
            marker="o",
            label=labels[feature_set],
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
        axis.legend()
    figure.suptitle(
        "Relative-v3 feature ablation "
        "(20 GeV pT-matched partial sample)"
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
        "absolute-v1": arguments.absolute_processed_dir.resolve(),
        "absolute-plus-parent-relative-v3": (
            arguments.relative_processed_dir.resolve()
        ),
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
            "Test whether Relative-v3 constituent, tau-pair, and MET "
            "features improve the Current Transformer without changing "
            "its structure."
        ),
        "relative_event_features": list(RELATIVE_EVENT_FEATURES),
        "feature_sets": list(FEATURE_SETS),
        "relative_track_features": list(RELATIVE_TRACK_FEATURES),
        "relative_pfo_features": list(RELATIVE_PFO_FEATURES),
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
    LOGGER.info("Input feature schemas and matching manifest verified")

    results = {}
    for feature_set in FEATURE_SETS:
        results[feature_set] = train_feature_set(
            feature_set=feature_set,
            processed_dir=processed_dirs[feature_set],
            metadata=metadata[feature_set],
            arguments=arguments,
            device=device,
            output_dir=arguments.output_dir,
            input_audit=input_audit,
        )

    series = {
        feature_set: summarize(
            results[feature_set]["validation_history"],
            arguments.max_steps,
        )
        for feature_set in FEATURE_SETS
    }
    baseline = series["absolute-v1"]
    relative = series["absolute-plus-parent-relative-v3"]
    summary = {
        "series": series,
        "differences_relative_minus_absolute": {
            "best_auc": relative["best_auc"] - baseline["best_auc"],
            "final_auc": relative["final_auc"] - baseline["final_auc"],
            "late_auc_mean": (
                relative["late_auc_mean"] - baseline["late_auc_mean"]
            ),
            "final_loss": relative["final_loss"] - baseline["final_loss"],
        },
        "training_runs": {
            feature_set: results[feature_set]["metrics"]
            for feature_set in FEATURE_SETS
        },
        "test_split_loaded": False,
    }
    save_csv(arguments.output_dir / "validation_history.csv", results)
    save_plot(arguments.output_dir / "relative_feature_comparison", results)
    write_json(arguments.output_dir / "summary.json", summary)
    LOGGER.info(
        "Finished relative-feature diagnostic: %s",
        arguments.output_dir,
    )
    LOGGER.info(
        "Relative-minus-absolute late AUC mean: %.6f",
        summary["differences_relative_minus_absolute"]["late_auc_mean"],
    )


if __name__ == "__main__":
    main()
