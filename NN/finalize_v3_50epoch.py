from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hpo_utils import (
    configure_tf32,
    create_model,
    create_streaming_loader,
    evaluate_model,
    roc_curve,
    sha256_file,
    shutdown_loader_workers,
    strip_evaluation_arrays,
    write_json,
)
from train import choose_device, set_random_seed


BATCH_SIZE = 512
RELOAD_NUM_WORKERS = 12
PREDICTION_NUM_WORKERS = 1
PREFETCH_FACTOR = 2
RELOAD_WORKER_PARTITION = "event"
PREDICTION_WORKER_PARTITION = "shard"
SEED = 42
EXPECTED_EPOCHS = 50
VALIDATION_TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "After a complete v3 50-epoch retraining, fix the validation-only "
            "center checkpoint, verify reload AUC/loss, then evaluate test "
            "exactly once."
        )
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--retrain-dir", type=Path, required=True)
    parser.add_argument(
        "--event-selection-manifest", type=Path, required=True
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def plot_learning(history: List[Dict[str, Any]], output_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True)
    axes[0].plot(
        epochs,
        [float(row["epoch_train_loss"]) for row in history],
        label="train",
    )
    axes[0].plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="validation",
    )
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epochs,
        [float(row["validation_auc"]) for row in history],
        color="tab:green",
    )
    axes[1].set(xlabel="Epoch", ylabel="Validation ROC AUC")
    axes[1].grid(alpha=0.25)
    figure.suptitle("v3 fixed 50-epoch retraining")
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves_50epoch.png", dpi=180)
    figure.savefig(output_dir / "learning_curves_50epoch.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        epochs,
        [float(row["learning_rate"]) for row in history],
        color="tab:purple",
    )
    axis.set(xlabel="Epoch", ylabel="Learning rate")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_rate_curve.png", dpi=180)
    figure.savefig(output_dir / "learning_rate_curve.pdf")
    plt.close(figure)


def plot_roc(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    auc: float,
    output_dir: Path,
    split: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{split.capitalize()} (AUC={auc:.6f})",
    )
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="Random")
    axis.set(
        xlabel="Background efficiency",
        ylabel="Signal efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
        title=f"v3 final-model {split} ROC",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / f"{split}_roc_curve.png", dpi=180)
    figure.savefig(output_dir / f"{split}_roc_curve.pdf")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    retrain_dir = args.retrain_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to repeat or overwrite final evaluation: {output_dir}"
        )
    required = [
        processed_dir / "metadata.json",
        args.selection_json,
        args.event_selection_manifest,
        args.snapshot_manifest,
        retrain_dir / "config.json",
        retrain_dir / "history.json",
        retrain_dir / "result.json",
        retrain_dir / "retrain_invocation.json",
        retrain_dir / "best_rolling_auc_model.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing finalizer inputs: {missing}")

    selection = json.loads(args.selection_json.read_text())
    binding = selection["data_binding"]
    actual_binding = {
        "processed_metadata_sha256": sha256_file(
            processed_dir / "metadata.json"
        ),
        "event_selection_manifest_sha256": sha256_file(
            args.event_selection_manifest
        ),
        "snapshot_manifest_sha256": sha256_file(args.snapshot_manifest),
    }
    if actual_binding != binding:
        raise RuntimeError("Finalizer data binding mismatch")
    result = json.loads((retrain_dir / "result.json").read_text())
    config = json.loads((retrain_dir / "config.json").read_text())
    history = json.loads((retrain_dir / "history.json").read_text())
    invocation = json.loads(
        (retrain_dir / "retrain_invocation.json").read_text()
    )
    if result["state"] != "COMPLETE":
        raise RuntimeError("50-epoch retraining is not complete")
    if int(result["epochs_completed"]) != EXPECTED_EPOCHS:
        raise RuntimeError("Retraining did not complete exactly 50 epochs")
    if len(history) != EXPECTED_EPOCHS:
        raise RuntimeError("History does not contain exactly 50 epochs")
    if result["stopped_early"] or invocation["early_stopping_enabled"]:
        raise RuntimeError("50-epoch retraining used early stopping")
    if (
        result["test_split_loaded"]
        or config["test_split_loaded"]
        or invocation["test_split_loaded"]
        or selection["test_split_loaded"]
    ):
        raise RuntimeError("Test was accessed before finalization")
    if result["data_binding"] != binding:
        raise RuntimeError("Retraining result binding mismatch")
    if result["parameters"] != selection["parameters"]:
        raise RuntimeError("Retraining parameters differ from selection")

    center_epoch = int(result["best_center_epoch"])
    center_rows = [
        row for row in history if int(row["epoch"]) == center_epoch
    ]
    if len(center_rows) != 1:
        raise RuntimeError("Selected center epoch is not unique in history")
    center_record = center_rows[0]
    source_checkpoint = retrain_dir / "best_rolling_auc_model.pt"
    if sha256_file(source_checkpoint) != result["checkpoint_sha256"]:
        raise RuntimeError("Retraining checkpoint hash mismatch")
    checkpoint = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True
    )
    if int(checkpoint["training_state"]["epoch"]) != center_epoch:
        raise RuntimeError("Checkpoint is not the selected center epoch")
    if checkpoint["hyperparameters"] != selection["parameters"]:
        raise RuntimeError("Checkpoint parameters differ from selection")
    checkpoint_binding = {
        "processed_metadata_sha256": checkpoint["data"][
            "processed_metadata_sha256"
        ],
        "event_selection_manifest_sha256": checkpoint["data"][
            "event_selection_manifest_sha256"
        ],
        "snapshot_manifest_sha256": checkpoint["data"][
            "snapshot_manifest_sha256"
        ],
    }
    if checkpoint_binding != binding:
        raise RuntimeError("Checkpoint data binding mismatch")

    selected_checkpoint_hash = sha256_file(source_checkpoint)
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    set_random_seed(SEED)
    device = choose_device("cuda")
    precision = configure_tf32()
    model, parameter_counts = create_model(
        metadata,
        selection["parameters"]["model_profile"],
        float(selection["parameters"]["dropout"]),
        device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = torch.compile(model, dynamic=True)
    loss_function = nn.BCEWithLogitsLoss()

    _, validation_loader = create_streaming_loader(
        processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=RELOAD_NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
        worker_partition=RELOAD_WORKER_PARTITION,
    )
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        loss_function,
        device,
        "v3 50-epoch selected checkpoint validation reload",
        verify_parameters_unchanged=True,
    )
    validation_shutdown = shutdown_loader_workers(validation_loader)
    auc_difference = float(validation_metrics["auc"]) - float(
        center_record["validation_auc"]
    )
    loss_difference = float(validation_metrics["loss"]) - float(
        center_record["validation_loss"]
    )
    if abs(auc_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(f"Validation reload AUC mismatch: {auc_difference}")
    if abs(loss_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation reload loss mismatch: {loss_difference}"
        )
    output_dir.mkdir(parents=True)
    _, canonical_validation_loader = create_streaming_loader(
        processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=PREDICTION_NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
        worker_partition=PREDICTION_WORKER_PARTITION,
    )
    canonical_validation_metrics = evaluate_model(
        model,
        canonical_validation_loader,
        loss_function,
        device,
        "v3 canonical-order validation predictions",
        verify_parameters_unchanged=True,
    )
    canonical_validation_shutdown = shutdown_loader_workers(
        canonical_validation_loader
    )
    validation_labels = np.asarray(canonical_validation_metrics["labels"])
    validation_scores = np.asarray(canonical_validation_metrics["scores"])
    validation_fpr, validation_tpr = roc_curve(
        validation_labels, validation_scores
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        labels=validation_labels,
        scores=validation_scores,
    )
    with (output_dir / "validation_roc_curve.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["background_efficiency", "signal_efficiency"])
        writer.writerows(zip(validation_fpr, validation_tpr))
    plot_roc(
        validation_fpr,
        validation_tpr,
        float(canonical_validation_metrics["auc"]),
        output_dir,
        "validation",
    )
    selected_checkpoint = output_dir / "selected_checkpoint.pt"
    shutil.copy2(source_checkpoint, selected_checkpoint)
    if sha256_file(selected_checkpoint) != selected_checkpoint_hash:
        raise RuntimeError("Copied checkpoint hash mismatch")
    write_json(
        output_dir / "validation_reload_audit.json",
        {
            "selected_center_epoch": center_epoch,
            "expected_auc": center_record["validation_auc"],
            "expected_loss": center_record["validation_loss"],
            "reloaded_metrics": strip_evaluation_arrays(validation_metrics),
            "auc_difference": auc_difference,
            "loss_difference": loss_difference,
            "worker_shutdown": validation_shutdown,
            "canonical_order_metrics": strip_evaluation_arrays(
                canonical_validation_metrics
            ),
            "canonical_order_worker_shutdown": (
                canonical_validation_shutdown
            ),
            "precision": precision,
            "test_split_loaded": False,
        },
    )
    selected_config = {
        "format_version": 1,
        "selected_before_test": True,
        "selection_scope": (
            "Selected parameters came from validation-only HPO; this "
            "checkpoint is the center of the best three-epoch validation "
            "AUC window after all 50 retraining epochs."
        ),
        "selected_trial_number": selection["selected_trial_number"],
        "selected_center_epoch": center_epoch,
        "best_window_epochs": result["best_window_epochs"],
        "parameters": selection["parameters"],
        "parameter_counts": parameter_counts,
        "data_binding": binding,
        "selected_checkpoint_sha256": selected_checkpoint_hash,
        "test_used_for_selection": False,
        "evaluation_loader": {
            "batch_size": BATCH_SIZE,
            "num_workers": PREDICTION_NUM_WORKERS,
            "worker_partition": PREDICTION_WORKER_PARTITION,
            "order": (
                "metadata sample/shard/event order, retained so saved "
                "predictions align with the pT-matching selection rows"
            ),
        },
    }
    write_json(output_dir / "selected_config.json", selected_config)
    plot_learning(history, output_dir)

    selection_fingerprint = hash_bytes(
        json.dumps(selected_config, sort_keys=True).encode()
    )
    write_json(
        output_dir / "test_evaluation_started.json",
        {
            "started_at": datetime.now().astimezone().isoformat(),
            "selection_fingerprint_sha256": selection_fingerprint,
            "checkpoint_sha256": selected_checkpoint_hash,
            "evaluation_number_for_this_model": 1,
        },
    )
    _, test_loader = create_streaming_loader(
        processed_dir,
        split="test",
        batch_size=BATCH_SIZE,
        num_workers=PREDICTION_NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
        worker_partition=PREDICTION_WORKER_PARTITION,
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
        "v3 final-model single test evaluation",
        verify_parameters_unchanged=True,
    )
    test_shutdown = shutdown_loader_workers(test_loader)
    labels = np.asarray(test_metrics["labels"])
    scores = np.asarray(test_metrics["scores"])
    false_positive_rate, true_positive_rate = roc_curve(labels, scores)
    eligible = np.flatnonzero(true_positive_rate >= 0.70)
    if eligible.size == 0:
        raise RuntimeError("Could not reach 70% signal efficiency")
    selected_index = int(eligible[0])
    background_efficiency = float(false_positive_rate[selected_index])
    background_rejection = (
        math.inf
        if background_efficiency == 0.0
        else 1.0 / background_efficiency
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        labels=labels,
        scores=scores,
    )
    with (output_dir / "test_roc_curve.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["background_efficiency", "signal_efficiency"])
        writer.writerows(zip(false_positive_rate, true_positive_rate))
    plot_roc(
        false_positive_rate,
        true_positive_rate,
        float(test_metrics["auc"]),
        output_dir,
        "test",
    )
    final_summary = {
        "selected_trial_number": selection["selected_trial_number"],
        "selected_center_epoch": center_epoch,
        "validation_auc": validation_metrics["auc"],
        "validation_loss": validation_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_loss": test_metrics["loss"],
        "background_rejection_at_signal_efficiency_0p7": (
            background_rejection
        ),
        "test_worker_shutdown": test_shutdown,
        "test_evaluation_count_for_this_model": 1,
        "test_used_for_selection": False,
        "evaluation_loader": selected_config["evaluation_loader"],
        "checkpoint_sha256": selected_checkpoint_hash,
        "data_binding": binding,
    }
    write_json(output_dir / "final_summary.json", final_summary)
    write_json(
        output_dir / "test_metrics.json",
        {
            **strip_evaluation_arrays(test_metrics),
            "background_efficiency": background_efficiency,
            "background_rejection": background_rejection,
            "test_evaluation_count_for_this_model": 1,
        },
    )
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
