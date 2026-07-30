from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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


BATCH_SIZE = 128
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
SEED = 42
VALIDATION_TOLERANCE = 1.0e-12
EXPECTED_TRIAL = 5
EXPECTED_EPOCHS = 40
FINAL_DIRECTORY_NAME = "final_selection_40epoch_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the validation-selected checkpoint from the fixed "
            "40-epoch trial-5 retraining, then evaluate that new model on "
            "test exactly once."
        )
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    return parser.parse_args()


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def plot_roc(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    auc: float,
    selected_index: int,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"40-epoch final model (AUC = {auc:.6f})",
    )
    axis.plot([0, 1], [0, 1], "--", color="0.55", label="Random")
    axis.scatter(
        [false_positive_rate[selected_index]],
        [true_positive_rate[selected_index]],
        color="tab:red",
        zorder=5,
        label=(
            "signal efficiency "
            f"{true_positive_rate[selected_index]:.3f}"
        ),
    )
    axis.set(
        xlabel="Background efficiency",
        ylabel="Signal efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
        title="TauSpin 40-epoch final-model test ROC",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "test_roc_curve.png", dpi=180)
    figure.savefig(output / "test_roc_curve.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    diagnostic_dir = args.diagnostic_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    final_dir = study_dir / FINAL_DIRECTORY_NAME
    if final_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing final selection: {final_dir}"
        )

    required = [
        diagnostic_dir / "result.json",
        diagnostic_dir / "history.json",
        diagnostic_dir / "config.json",
        diagnostic_dir / "best_rolling_auc_model.pt",
        processed_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    result = json.loads((diagnostic_dir / "result.json").read_text())
    history = json.loads((diagnostic_dir / "history.json").read_text())
    config = json.loads((diagnostic_dir / "config.json").read_text())
    if result["state"] != "COMPLETE":
        raise RuntimeError(f"40-epoch run is not complete: {result['state']}")
    if int(result["trial_number"]) != EXPECTED_TRIAL:
        raise RuntimeError("Unexpected trial number")
    if int(result["epochs_completed"]) != EXPECTED_EPOCHS:
        raise RuntimeError("40-epoch run did not complete all epochs")
    if result["stopped_early"]:
        raise RuntimeError("40-epoch final run unexpectedly stopped early")
    if result["test_split_loaded"] or config["test_split_loaded"]:
        raise RuntimeError("Test was loaded before final checkpoint selection")
    if not result["finite_training"]:
        raise RuntimeError("Training reported non-finite values")
    if len(history) != EXPECTED_EPOCHS:
        raise RuntimeError("History does not contain exactly 40 epochs")

    source_checkpoint = diagnostic_dir / "best_rolling_auc_model.pt"
    checkpoint_sha256 = sha256_file(source_checkpoint)
    if checkpoint_sha256 != result["checkpoint_sha256"]:
        raise RuntimeError("Best-checkpoint hash does not match result")
    center_epoch = int(result["best_center_epoch"])
    center_rows = [
        row for row in history if int(row["epoch"]) == center_epoch
    ]
    if len(center_rows) != 1:
        raise RuntimeError("Could not uniquely identify the center epoch")
    center_record = center_rows[0]

    final_dir.mkdir()
    copied_checkpoint = final_dir / "selected_checkpoint.pt"
    shutil.copy2(source_checkpoint, copied_checkpoint)
    if sha256_file(copied_checkpoint) != checkpoint_sha256:
        raise RuntimeError("Copied checkpoint hash mismatch")

    selection = {
        "selected_before_test": True,
        "selected_at": datetime.now().astimezone().isoformat(),
        "selection_scope": (
            "Trial 5 hyperparameters were fixed by the completed HPO. "
            "This final retraining used a 40-epoch cosine schedule and "
            "selected the center epoch of the best three-epoch moving-average "
            "validation AUC window."
        ),
        "selection_rule": {
            "primary": (
                "maximum three-epoch moving-average validation AUC across "
                "the fixed 40-epoch retraining"
            ),
            "checkpoint": "center epoch of the winning three-epoch window",
            "test_used_for_selection": False,
        },
        "selected_trial": {
            "trial_number": EXPECTED_TRIAL,
            "objective_auc": float(result["objective"]),
            "minimum_validation_loss": float(
                result["minimum_validation_loss"]
            ),
            "best_rolling_loss_3": float(
                result["best_rolling_loss_3"]
            ),
            "best_center_epoch": center_epoch,
            "stopping_epoch": EXPECTED_EPOCHS,
            "stopped_early": False,
            "stop_reason": (
                "Diagnostic/final retraining intentionally completed all "
                "40 epochs; checkpoint selection remained validation-only."
            ),
            "model_profile": result["parameters"]["model_profile"],
            "learning_rate": float(
                result["parameters"]["learning_rate"]
            ),
            "dropout": float(result["parameters"]["dropout"]),
            "schedule_profile": result["parameters"]["schedule_profile"],
            "parameter_count": int(
                result["parameter_counts"]["total"]
            ),
            "best_window_epochs": result["best_window_epochs"],
        },
        "selected_checkpoint_source": str(source_checkpoint),
        "selected_checkpoint_copy": str(copied_checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "source_history": str(
            (diagnostic_dir / "history.json").resolve()
        ),
        "source_result": str((diagnostic_dir / "result.json").resolve()),
        "test_metrics_available_at_selection": False,
    }
    write_json(final_dir / "selected_config.json", selection)

    metadata = json.loads((processed_dir / "metadata.json").read_text())
    checkpoint = torch.load(
        copied_checkpoint, map_location="cpu", weights_only=True
    )
    if int(checkpoint["trial_number"]) != EXPECTED_TRIAL:
        raise RuntimeError("Checkpoint trial number mismatch")
    parameters = checkpoint["hyperparameters"]
    if parameters != result["parameters"]:
        raise RuntimeError("Checkpoint/result hyperparameters mismatch")
    if int(checkpoint["training_state"]["epoch"]) != center_epoch:
        raise RuntimeError("Checkpoint is not the selected center epoch")

    set_random_seed(SEED)
    precision = configure_tf32()
    device = choose_device("cuda")
    model, parameter_counts = create_model(
        metadata,
        parameters["model_profile"],
        float(parameters["dropout"]),
        device,
    )
    if parameter_counts["total"] != result["parameter_counts"]["total"]:
        raise RuntimeError("Reconstructed parameter count mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function = nn.BCEWithLogitsLoss()

    _, validation_loader = create_streaming_loader(
        processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
    )
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        loss_function,
        device,
        "40-epoch final-model validation reload",
        verify_parameters_unchanged=True,
    )
    validation_shutdown = shutdown_loader_workers(validation_loader)
    auc_difference = (
        float(validation_metrics["auc"])
        - float(center_record["validation_auc"])
    )
    loss_difference = (
        float(validation_metrics["loss"])
        - float(center_record["validation_loss"])
    )
    if abs(auc_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation reload AUC mismatch: {auc_difference}"
        )
    if abs(loss_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation reload loss mismatch: {loss_difference}"
        )
    validation_labels = np.asarray(validation_metrics["labels"])
    validation_scores = np.asarray(validation_metrics["scores"])
    np.savez_compressed(
        final_dir / "validation_predictions.npz",
        labels=validation_labels,
        scores=validation_scores,
    )
    validation_audit = {
        "selected_trial": EXPECTED_TRIAL,
        "selected_center_epoch": center_epoch,
        "expected_center_epoch_auc": center_record["validation_auc"],
        "expected_center_epoch_loss": center_record["validation_loss"],
        "reloaded_metrics": strip_evaluation_arrays(validation_metrics),
        "auc_difference": auc_difference,
        "loss_difference": loss_difference,
        "worker_shutdown": validation_shutdown,
        "precision": precision,
        "test_split_loaded": False,
    }
    write_json(final_dir / "validation_reload_audit.json", validation_audit)

    write_json(
        final_dir / "test_evaluation_started.json",
        {
            "started_at": datetime.now().astimezone().isoformat(),
            "selected_trial": EXPECTED_TRIAL,
            "selected_center_epoch": center_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "selection_fingerprint_sha256": hash_bytes(
                json.dumps(selection, sort_keys=True).encode()
            ),
            "evaluation_number_for_this_new_model": 1,
        },
    )
    _, test_loader = create_streaming_loader(
        processed_dir,
        split="test",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
        "40-epoch final-model single test evaluation",
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
        final_dir / "test_predictions.npz",
        labels=labels,
        scores=scores,
    )
    with (final_dir / "test_roc_curve.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["background_efficiency", "signal_efficiency"])
        writer.writerows(zip(false_positive_rate, true_positive_rate))
    plot_roc(
        false_positive_rate,
        true_positive_rate,
        float(test_metrics["auc"]),
        selected_index,
        final_dir,
    )
    test_document = {
        **strip_evaluation_arrays(test_metrics),
        "signal_efficiency_target": 0.70,
        "achieved_signal_efficiency": float(
            true_positive_rate[selected_index]
        ),
        "background_efficiency": background_efficiency,
        "background_rejection": background_rejection,
        "selected_threshold_index": selected_index,
        "test_evaluation_count_for_this_new_model": 1,
        "test_worker_shutdown": test_shutdown,
        "selected_before_test": True,
        "selected_trial": EXPECTED_TRIAL,
        "selected_center_epoch": center_epoch,
        "checkpoint_sha256": checkpoint_sha256,
    }
    write_json(final_dir / "test_metrics.json", test_document)
    final_summary = {
        "selected_trial": EXPECTED_TRIAL,
        "selected_center_epoch": center_epoch,
        "best_window_epochs": result["best_window_epochs"],
        "validation_objective_auc_3epoch_mean": result["objective"],
        "validation_auc_at_selected_epoch": validation_metrics["auc"],
        "validation_loss_at_selected_epoch": validation_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_loss": test_metrics["loss"],
        "background_rejection_at_signal_efficiency_0p7": (
            background_rejection
        ),
        "test_evaluation_count_for_this_new_model": 1,
        "test_used_for_selection": False,
        "checkpoint_sha256": checkpoint_sha256,
    }
    write_json(final_dir / "final_summary.json", final_summary)
    write_json(
        final_dir / "artifact_manifest.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "study_dir": str(study_dir),
            "diagnostic_dir": str(diagnostic_dir),
            "processed_dir": str(processed_dir),
            "selected_checkpoint_sha256": checkpoint_sha256,
            "files": sorted(path.name for path in final_dir.iterdir()),
        },
    )
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
