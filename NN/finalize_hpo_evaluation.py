from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
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
AUC_TIE_THRESHOLD = 1.0e-3
VALIDATION_TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a completed HPO using validation only, then evaluate "
            "the selected checkpoint on test exactly once."
        )
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-validation-only",
        action="store_true",
        help=(
            "Resume only when validation-only selection artifacts exist and "
            "test evaluation has not started."
        ),
    )
    return parser.parse_args()


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_trials(study_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((study_dir / "trials").glob("trial_*/result.json")):
        result = json.loads(path.read_text())
        history = json.loads((path.parent / "history.json").read_text())
        parameters = result["parameters"]
        rows.append(
            {
                "trial_number": int(result["trial_number"]),
                "objective_auc": float(result["objective"]),
                "minimum_validation_loss": float(
                    result["minimum_validation_loss"]
                ),
                "best_rolling_loss_3": float(
                    result["best_rolling_loss_3"]
                ),
                "best_center_epoch": int(result["best_center_epoch"]),
                "stopping_epoch": int(result["epochs_completed"]),
                "stopped_early": bool(result["stopped_early"]),
                "stop_reason": result["stop_reason"],
                "model_profile": parameters["model_profile"],
                "learning_rate": float(parameters["learning_rate"]),
                "dropout": float(parameters["dropout"]),
                "schedule_profile": parameters["schedule_profile"],
                "parameter_count": int(
                    result["parameter_counts"]["total"]
                ),
                "finite_training": bool(result["finite_training"]),
                "test_split_loaded": bool(result["test_split_loaded"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "reloaded_center_auc": float(
                    result["reloaded_center_auc"]
                ),
                "reloaded_center_auc_difference": float(
                    result["reloaded_center_auc_difference"]
                ),
                "trial_dir": str(path.parent.resolve()),
                "result_path": str(path.resolve()),
                "history": history,
            }
        )
    if not rows:
        raise RuntimeError("No completed trial result.json files")
    return rows


def rank_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_auc = max(row["objective_auc"] for row in rows)
    near = [
        row
        for row in rows
        if best_auc - row["objective_auc"] < AUC_TIE_THRESHOLD
    ]
    far = [row for row in rows if row not in near]
    near.sort(
        key=lambda row: (
            row["minimum_validation_loss"],
            row["parameter_count"],
            -row["objective_auc"],
            row["trial_number"],
        )
    )
    far.sort(
        key=lambda row: (
            -row["objective_auc"],
            row["minimum_validation_loss"],
            row["parameter_count"],
            row["trial_number"],
        )
    )
    ranked = near + far
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["auc_gap_from_best"] = best_auc - row["objective_auc"]
        row["within_auc_tie_threshold"] = (
            row["auc_gap_from_best"] < AUC_TIE_THRESHOLD
        )
    return ranked


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "history"
    }


def write_ranking(study_dir: Path, ranked: list[dict[str, Any]]) -> None:
    output = study_dir / "final_selection"
    output.mkdir(exist_ok=False)
    write_json(
        output / "validation_ranking.json",
        [public_row(row) for row in ranked],
    )
    fields = list(public_row(ranked[0]))
    with (output / "validation_ranking.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_row(row) for row in ranked)


def selected_checkpoint_path(row: dict[str, Any]) -> Path:
    return Path(row["trial_dir"]) / "best_rolling_auc_model.pt"


def center_history_record(row: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in row["history"]
        if int(item["epoch"]) == row["best_center_epoch"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify selected center epoch history"
        )
    return matches[0]


def plot_roc(
    *,
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    auc: float,
    selected_index: int,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Selected model (AUC = {auc:.6f})",
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
        title="TauSpin fixed-partial-v2 test ROC",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "test_roc_curve.png", dpi=180)
    figure.savefig(output_dir / "test_roc_curve.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    final_dir = args.study_dir / "final_selection"
    if final_dir.exists() and not args.resume_validation_only:
        raise FileExistsError(
            f"{final_dir} already exists; refusing to repeat test evaluation"
        )
    if args.resume_validation_only:
        required = {
            "selected_config.json",
            "selected_checkpoint.pt",
            "validation_ranking.csv",
            "validation_ranking.json",
        }
        present = {path.name for path in final_dir.glob("*")}
        missing = required - present
        forbidden = {
            "test_evaluation_started.json",
            "test_metrics.json",
            "test_predictions.npz",
            "final_summary.json",
        }
        if not final_dir.is_dir() or missing or present & forbidden:
            raise RuntimeError(
                "Unsafe resume state: "
                f"missing={sorted(missing)}, "
                f"forbidden_present={sorted(present & forbidden)}"
            )

    controller = json.loads(
        (args.study_dir / "controller_complete.json").read_text()
    )
    if controller["trial_state_counts"] != {"COMPLETE": 21}:
        raise RuntimeError(
            f"Unexpected controller state: {controller['trial_state_counts']}"
        )
    rows = load_trials(args.study_dir)
    if len(rows) != 21:
        raise RuntimeError(f"Expected 21 result files, found {len(rows)}")
    if not all(row["finite_training"] for row in rows):
        raise RuntimeError("A trial reported non-finite training")
    if any(row["test_split_loaded"] for row in rows):
        raise RuntimeError("A trial loaded test during HPO")
    if any(
        abs(row["reloaded_center_auc_difference"])
        > VALIDATION_TOLERANCE
        for row in rows
    ):
        raise RuntimeError("A trial failed its checkpoint reload audit")

    ranked = rank_trials(rows)
    selected = ranked[0]
    source_checkpoint = selected_checkpoint_path(selected)
    if sha256_file(source_checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint SHA-256 mismatch")
    copied_checkpoint = final_dir / "selected_checkpoint.pt"
    if args.resume_validation_only:
        previous_selection = json.loads(
            (final_dir / "selected_config.json").read_text()
        )
        if (
            previous_selection["selected_trial"]["trial_number"]
            != selected["trial_number"]
        ):
            raise RuntimeError("Resume selection does not match ranking")
    else:
        write_ranking(args.study_dir, ranked)
        final_dir = args.study_dir / "final_selection"
        copied_checkpoint = final_dir / "selected_checkpoint.pt"
        shutil.copy2(source_checkpoint, copied_checkpoint)
    if sha256_file(copied_checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Copied checkpoint SHA-256 mismatch")

    selection = {
        "selected_before_test": True,
        "selected_at": datetime.now().astimezone().isoformat(),
        "selection_rule": {
            "primary": (
                "maximum best three-epoch moving-average validation AUC"
            ),
            "auc_tie_threshold": AUC_TIE_THRESHOLD,
            "secondary_within_threshold": (
                "minimum validation loss, lower is better"
            ),
            "tertiary": "parameter count, lower is better",
        },
        "selected_trial": public_row(selected),
        "runner_up": public_row(ranked[1]),
        "selected_checkpoint_source": str(source_checkpoint.resolve()),
        "selected_checkpoint_copy": str(copied_checkpoint.resolve()),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "test_metrics_available_at_selection": False,
    }
    if args.resume_validation_only:
        selection = previous_selection
    else:
        write_json(final_dir / "selected_config.json", selection)

    metadata = json.loads(
        (args.processed_dir / "metadata.json").read_text()
    )
    checkpoint = torch.load(
        copied_checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint["trial_number"] != selected["trial_number"]:
        raise RuntimeError("Checkpoint trial number mismatch")
    parameters = checkpoint["hyperparameters"]
    set_random_seed(SEED)
    configure_tf32()
    device = choose_device("cuda")
    model, parameter_counts = create_model(
        metadata,
        parameters["model_profile"],
        float(parameters["dropout"]),
        device,
    )
    if parameter_counts["total"] != selected["parameter_count"]:
        raise RuntimeError("Reconstructed model parameter-count mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_function = nn.BCEWithLogitsLoss()

    _, validation_loader = create_streaming_loader(
        args.processed_dir,
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
        "independent selected-checkpoint validation reload",
        verify_parameters_unchanged=True,
    )
    validation_shutdown = shutdown_loader_workers(validation_loader)
    expected = center_history_record(selected)
    auc_difference = (
        float(validation_metrics["auc"])
        - float(expected["validation_auc"])
    )
    loss_difference = (
        float(validation_metrics["loss"])
        - float(expected["validation_loss"])
    )
    if abs(auc_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation reload AUC mismatch: {auc_difference}"
        )
    if abs(loss_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation reload loss mismatch: {loss_difference}"
        )
    validation_audit = {
        "selected_trial": selected["trial_number"],
        "selected_center_epoch": selected["best_center_epoch"],
        "expected_center_epoch_auc": expected["validation_auc"],
        "expected_center_epoch_loss": expected["validation_loss"],
        "reloaded_metrics": strip_evaluation_arrays(validation_metrics),
        "auc_difference": auc_difference,
        "loss_difference": loss_difference,
        "worker_shutdown": validation_shutdown,
        "test_split_loaded": False,
    }
    write_json(final_dir / "validation_reload_audit.json", validation_audit)

    # This marker is written only after selection and validation reproduction.
    # Its presence prevents an accidental second test evaluation.
    write_json(
        final_dir / "test_evaluation_started.json",
        {
            "started_at": datetime.now().astimezone().isoformat(),
            "selected_trial": selected["trial_number"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_fingerprint_sha256": hash_bytes(
                json.dumps(selection, sort_keys=True).encode()
            ),
        },
    )
    _, test_loader = create_streaming_loader(
        args.processed_dir,
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
        "selected-checkpoint test",
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
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        auc=float(test_metrics["auc"]),
        selected_index=selected_index,
        output_dir=final_dir,
    )
    test_document = {
        **strip_evaluation_arrays(test_metrics),
        "signal_efficiency_target": 0.70,
        "achieved_signal_efficiency": float(
            true_positive_rate[selected_index]
        ),
        "background_efficiency": background_efficiency,
        "background_rejection_inverse_efficiency": background_rejection,
        "selected_trial": selected["trial_number"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "worker_shutdown": test_shutdown,
        "test_evaluation_count_for_selected_model": 1,
    }
    write_json(final_dir / "test_metrics.json", test_document)
    final_summary = {
        "hpo_state_counts": controller["trial_state_counts"],
        "selected_trial": public_row(selected),
        "runner_up": public_row(ranked[1]),
        "auc_difference_to_runner_up": (
            selected["objective_auc"] - ranked[1]["objective_auc"]
        ),
        "validation_reload": validation_audit,
        "test": test_document,
        "artifacts": {
            "selected_checkpoint": str(copied_checkpoint.resolve()),
            "validation_ranking": str(
                (final_dir / "validation_ranking.csv").resolve()
            ),
            "test_roc_png": str(
                (final_dir / "test_roc_curve.png").resolve()
            ),
            "test_roc_pdf": str(
                (final_dir / "test_roc_curve.pdf").resolve()
            ),
        },
    }
    write_json(final_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
