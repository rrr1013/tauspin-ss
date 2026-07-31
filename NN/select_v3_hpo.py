from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


AUC_GAP = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select v3 HPO parameters using validation results only. "
            "Trials within 0.001 AUC of the best are ordered by minimum "
            "validation loss, trainable parameter count, then trial number."
        )
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_parameters(parameters: Dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")


def rank_trials(
    trials: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    if not trials:
        raise ValueError("Cannot rank an empty trial list")
    best_auc = max(row["objective_auc"] for row in trials)
    for row in trials:
        row["auc_gap_from_best"] = best_auc - row["objective_auc"]
        row["within_auc_gap"] = row["auc_gap_from_best"] < AUC_GAP
    eligible = [row for row in trials if row["within_auc_gap"]]
    eligible.sort(key=lambda row: (
        row["minimum_validation_loss"],
        row["trainable_parameter_count"],
        row["trial_number"],
    ))
    selected = eligible[0]
    ranked = sorted(trials, key=lambda row: (
        0 if row["within_auc_gap"] else 1,
        (
            row["minimum_validation_loss"]
            if row["within_auc_gap"]
            else -row["objective_auc"]
        ),
        (
            row["trainable_parameter_count"]
            if row["within_auc_gap"]
            else 0
        ),
        row["trial_number"],
    ))
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected"] = row["trial_number"] == selected["trial_number"]
    return ranked, selected, best_auc


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    study_config_path = study_dir / "study_config.json"
    if not study_config_path.is_file():
        raise FileNotFoundError(study_config_path)
    study_config = json.loads(study_config_path.read_text())
    binding = study_config.get("data_binding")
    required_hashes = (
        "processed_metadata_sha256",
        "event_selection_manifest_sha256",
        "snapshot_manifest_sha256",
    )
    if not isinstance(binding, dict) or any(
        not binding.get(name) for name in required_hashes
    ):
        raise RuntimeError("Study config lacks complete data binding")

    trials: List[Dict[str, Any]] = []
    for result_path in sorted((study_dir / "trials").glob("trial_*/result.json")):
        result = json.loads(result_path.read_text())
        if result.get("state") != "COMPLETE":
            continue
        if result.get("test_split_loaded"):
            raise RuntimeError(f"Trial loaded test split: {result_path}")
        if result.get("data_binding") != {
            name: binding[name] for name in required_hashes
        }:
            raise RuntimeError(f"Trial binding mismatch: {result_path}")
        trials.append({
            "trial_number": int(result["trial_number"]),
            "objective_auc": float(result["objective"]),
            "minimum_validation_loss": float(
                result["minimum_validation_loss"]
            ),
            "trainable_parameter_count": int(
                result["parameter_counts"]["trainable"]
            ),
            "parameters": result["parameters"],
            "parameters_canonical": canonical_parameters(
                result["parameters"]
            ),
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "test_split_loaded": False,
        })
    if not trials:
        raise RuntimeError("No completed validation-only trials found")

    ranked, selected, best_auc = rank_trials(trials)

    output_dir.mkdir(parents=True)
    write_json(output_dir / "validation_ranking.json", ranked)
    with (output_dir / "validation_ranking.csv").open(
        "w", newline=""
    ) as stream:
        fields = [
            "selection_rank",
            "selected",
            "trial_number",
            "objective_auc",
            "auc_gap_from_best",
            "within_auc_gap",
            "minimum_validation_loss",
            "trainable_parameter_count",
            "parameters_canonical",
            "result_path",
            "result_sha256",
            "checkpoint_sha256",
            "test_split_loaded",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in fields} for row in ranked
        )
    selection = {
        "format_version": 1,
        "purpose": "Validation-only v3 HPO parameter selection",
        "study_dir": str(study_dir),
        "study_config_path": str(study_config_path.resolve()),
        "study_config_sha256": sha256_file(study_config_path),
        "data_binding": {
            name: binding[name] for name in required_hashes
        },
        "selection_rule": {
            "auc_gap_strictly_less_than": AUC_GAP,
            "eligible_reference_auc": best_auc,
            "eligible_tie_break": [
                "minimum_validation_loss ascending",
                "trainable parameter count ascending",
                "trial_number ascending",
            ],
            "test_used": False,
        },
        "selected_trial_number": selected["trial_number"],
        "selected_objective_auc": selected["objective_auc"],
        "selected_minimum_validation_loss": selected[
            "minimum_validation_loss"
        ],
        "parameters": selected["parameters"],
        "source_result_path": selected["result_path"],
        "source_result_sha256": selected["result_sha256"],
        "test_split_loaded": False,
    }
    write_json(output_dir / "selected_parameters.json", selection)
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
