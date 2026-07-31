from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


HPO_FIGURES = [
    "hpo_optimization_history",
    "hpo_profile_performance",
    "hpo_learning_rate_dropout_scatter",
    "hpo_trial_ranking",
    "hpo_summary_panel",
]
FINAL_FIGURES = [
    "roc_validation_test",
    "score_distribution_test",
    "working_point_test",
    "confusion_matrix_wp70",
    "pt_binned_auc_test",
    "learning_curve_loss",
    "learning_curve_auc",
    "learning_rate_curve",
    "final_summary_panel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completed fixed-partial-v3 analysis.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--matching-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--retrain-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--hpo-figures-dir", type=Path, required=True)
    parser.add_argument("--final-figures-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    args = parse_args()
    snapshot_dir = args.snapshot_dir.resolve()
    matching_dir = args.matching_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    study_dir = args.study_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    retrain_dir = args.retrain_dir.resolve()
    final_dir = args.final_dir.resolve()
    hpo_figures_dir = args.hpo_figures_dir.resolve()
    final_figures_dir = args.final_figures_dir.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    ntuple_audit_path = snapshot_dir / "full_ntuple_audit.json"
    dataset_audit_path = matching_dir / "dataset_audit.json"
    study_config_path = study_dir / "study_config.json"
    controller_path = study_dir / "controller_complete.json"
    selection_path = selection_dir / "selected_parameters.json"
    retrain_result_path = retrain_dir / "result.json"
    retrain_invocation_path = retrain_dir / "retrain_invocation.json"
    retrain_history_path = retrain_dir / "history.json"
    final_summary_path = final_dir / "final_summary.json"
    selected_config_path = final_dir / "selected_config.json"
    test_metrics_path = final_dir / "test_metrics.json"
    validation_audit_path = final_dir / "validation_reload_audit.json"
    required = [
        ntuple_audit_path,
        dataset_audit_path,
        study_config_path,
        controller_path,
        selection_path,
        retrain_result_path,
        retrain_invocation_path,
        retrain_history_path,
        final_summary_path,
        selected_config_path,
        test_metrics_path,
        validation_audit_path,
        processed_dir / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    ntuple_audit = json.loads(ntuple_audit_path.read_text())
    dataset_audit = json.loads(dataset_audit_path.read_text())
    study_config = json.loads(study_config_path.read_text())
    controller = json.loads(controller_path.read_text())
    selection = json.loads(selection_path.read_text())
    retrain_result = json.loads(retrain_result_path.read_text())
    retrain_invocation = json.loads(retrain_invocation_path.read_text())
    retrain_history = json.loads(retrain_history_path.read_text())
    final_summary = json.loads(final_summary_path.read_text())
    selected_config = json.loads(selected_config_path.read_text())
    test_metrics = json.loads(test_metrics_path.read_text())
    validation_audit = json.loads(validation_audit_path.read_text())

    if not ntuple_audit["submission_allowed_for_dataset"]:
        failures.append("ntuple_audit")
    if not dataset_audit["approved_for_gpu_smoke"]:
        failures.append("dataset_audit")
    if any(
        not row["source_and_packed_event_numbers_equal"]
        for row in dataset_audit["prediction_order_alignment"].values()
    ):
        failures.append("prediction_order_alignment")

    state_counts = controller["trial_state_counts"]
    if state_counts.get("RUNNING", 0) or state_counts.get("WAITING", 0):
        failures.append("hpo_unfinished_trials")
    if state_counts.get("FAIL", 0):
        warnings.append(f"hpo_failed_trials:{state_counts['FAIL']}")
    binding_names = (
        "processed_metadata_sha256",
        "event_selection_manifest_sha256",
        "snapshot_manifest_sha256",
    )
    study_binding = {
        name: study_config["data_binding"][name] for name in binding_names
    }
    trial_records = []
    nonempty_worker_stderr = []
    for result_path in sorted((study_dir / "trials").glob("trial_*/result.json")):
        result = json.loads(result_path.read_text())
        trial_number = int(result["trial_number"])
        trial_failures = []
        if result["state"] != "COMPLETE":
            trial_failures.append("state")
        if result["test_split_loaded"]:
            trial_failures.append("test_loaded")
        if result["data_binding"] != study_binding:
            trial_failures.append("data_binding")
        for name in ("objective", "minimum_validation_loss", "elapsed_seconds"):
            if not finite(result[name]):
                trial_failures.append(f"nonfinite:{name}")
        if not result["finite_training"]:
            trial_failures.append("finite_training")
        if abs(float(result["reloaded_center_auc_difference"])) > 1.0e-12:
            trial_failures.append("checkpoint_reload")
        checkpoint = result_path.parent / "best_rolling_auc_model.pt"
        if sha256_file(checkpoint) != result["checkpoint_sha256"]:
            trial_failures.append("checkpoint_hash")
        stderr_path = study_dir / "logs" / f"trial_{trial_number:03d}.stderr.log"
        stderr_bytes = stderr_path.stat().st_size
        if stderr_bytes:
            nonempty_worker_stderr.append(
                {"trial": trial_number, "path": str(stderr_path), "bytes": stderr_bytes}
            )
        trial_records.append(
            {
                "trial_number": trial_number,
                "objective": result["objective"],
                "epochs_completed": result["epochs_completed"],
                "elapsed_seconds": result["elapsed_seconds"],
                "stderr_bytes": stderr_bytes,
                "failures": trial_failures,
            }
        )
        if trial_failures:
            failures.append(f"hpo_trial:{trial_number}")
    if len(trial_records) != state_counts.get("COMPLETE", 0):
        failures.append("hpo_result_count")
    if nonempty_worker_stderr:
        warnings.append("nonempty_worker_stderr")

    if selection["test_split_loaded"]:
        failures.append("selection_test_loaded")
    if selection["data_binding"] != study_binding:
        failures.append("selection_binding")
    selected_trial = int(selection["selected_trial_number"])
    if selected_trial not in {row["trial_number"] for row in trial_records}:
        failures.append("selected_trial_missing")

    if (
        retrain_result["state"] != "COMPLETE"
        or int(retrain_result["epochs_completed"]) != 50
        or len(retrain_history) != 50
        or retrain_result["stopped_early"]
        or retrain_result["test_split_loaded"]
        or retrain_invocation["test_split_loaded"]
        or retrain_invocation["early_stopping_enabled"]
    ):
        failures.append("retrain_guards")
    if retrain_result["data_binding"] != study_binding:
        failures.append("retrain_binding")
    if abs(float(retrain_result["reloaded_center_auc_difference"])) > 1.0e-12:
        failures.append("retrain_checkpoint_reload")

    if (
        final_summary["test_evaluation_count_for_this_model"] != 1
        or final_summary["test_used_for_selection"]
        or not selected_config["selected_before_test"]
    ):
        failures.append("test_isolation")
    if final_summary["data_binding"] != study_binding:
        failures.append("final_binding")
    if int(final_summary["selected_trial_number"]) != selected_trial:
        failures.append("final_selected_trial")
    if abs(float(validation_audit["auc_difference"])) > 1.0e-12:
        failures.append("final_validation_auc_reload")
    if abs(float(validation_audit["loss_difference"])) > 1.0e-12:
        failures.append("final_validation_loss_reload")
    for name in ("test_auc", "test_loss", "background_rejection"):
        source = test_metrics if name == "background_rejection" else final_summary
        if not finite(source[name]):
            failures.append(f"nonfinite_final:{name}")
    expected_test_events = int(
        dataset_audit["counts"]["test"]["total"]
    )
    if int(test_metrics["event_count"]) != expected_test_events:
        failures.append("test_event_count")

    missing_figures = []
    for directory, bases in (
        (hpo_figures_dir, HPO_FIGURES),
        (final_figures_dir, FINAL_FIGURES),
    ):
        for base in bases:
            for extension in ("png", "pdf"):
                path = directory / f"{base}.{extension}"
                if not path.is_file() or path.stat().st_size == 0:
                    missing_figures.append(str(path))
    if missing_figures:
        failures.append("missing_figures")

    controller_stderr_path = Path(
        str(study_dir) + ".controller.stderr.log"
    )
    report = {
        "format_version": 1,
        "completed_at": datetime.now().astimezone().isoformat(),
        "approved": not failures,
        "failures": sorted(set(failures)),
        "warnings": warnings,
        "snapshot": {
            "id": ntuple_audit["snapshot_id"],
            "H_selected_entries": ntuple_audit["samples"]["H"]["selected_entries"],
            "Z_selected_entries": ntuple_audit["samples"]["Z"]["selected_entries"],
            "audit_sha256": sha256_file(ntuple_audit_path),
        },
        "dataset": {
            "counts": dataset_audit["counts"],
            "metadata_sha256": dataset_audit["metadata_sha256"],
            "matching_manifest_sha256": dataset_audit[
                "matching_manifest_sha256"
            ],
            "audit_sha256": sha256_file(dataset_audit_path),
        },
        "hpo": {
            "state_counts": state_counts,
            "selected_trial_number": selected_trial,
            "trials": trial_records,
            "nonempty_worker_stderr": nonempty_worker_stderr,
            "controller_stderr": (
                {
                    "path": str(controller_stderr_path),
                    "bytes": controller_stderr_path.stat().st_size,
                }
                if controller_stderr_path.is_file()
                else None
            ),
        },
        "final": {
            "selected_center_epoch": final_summary["selected_center_epoch"],
            "validation_auc": final_summary["validation_auc"],
            "validation_loss": final_summary["validation_loss"],
            "test_auc": final_summary["test_auc"],
            "test_loss": final_summary["test_loss"],
            "background_rejection_at_signal_efficiency_0p7": final_summary[
                "background_rejection_at_signal_efficiency_0p7"
            ],
            "test_event_count": test_metrics["event_count"],
        },
        "missing_figures": missing_figures,
        "input_hashes": {
            str(path): sha256_file(path) for path in required
        },
    }
    args.report.resolve().write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
