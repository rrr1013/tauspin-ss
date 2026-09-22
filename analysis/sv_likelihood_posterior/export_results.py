"""Export compact, reviewable tables from the SV-posterior run reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRIMARY_COHORTS = (
    "inclusive",
    "one_sv",
    "both_sv",
    "threeprong_containing",
    "threeprong_x_threeprong",
    "exact_surface_valid",
    "exact_surface_invalid",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-report", type=Path, required=True)
    parser.add_argument("--readout-report", type=Path, required=True)
    parser.add_argument("--surface-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    representation = json.loads(args.representation_report.read_text())
    readout = json.loads(args.readout_report.read_text())
    surface = json.loads(args.surface_report.read_text())

    h_rows: list[dict[str, Any]] = []
    h_metrics = representation["h_metrics_reco_visible_truth_nu_functional"]
    for cohort in PRIMARY_COHORTS:
        if cohort not in h_metrics:
            continue
        for method, values in h_metrics[cohort].items():
            components = values.get("component_mse", [None, None, None])
            correlations = values.get("component_correlation", [None, None, None])
            h_rows.append({
                "cohort": cohort,
                "method": method,
                "events": values.get("events"),
                "mse": values.get("mse"),
                "mse_h_n": components[0],
                "mse_h_r": components[1],
                "mse_h_k": components[2],
                "cosine": values.get("cosine"),
                "corr_h_n": correlations[0],
                "corr_h_r": correlations[1],
                "corr_h_k": correlations[2],
            })
    write_csv(args.output / "h_metrics.csv", h_rows)

    auc_rows: list[dict[str, Any]] = []
    for cohort in PRIMARY_COHORTS:
        if cohort not in readout["auc_metrics"]:
            continue
        for method, values in readout["auc_metrics"][cohort].items():
            auc_rows.append({"cohort": cohort, "method": method, **values})
    write_csv(args.output / "auc_metrics.csv", auc_rows)

    bootstrap_rows: list[dict[str, Any]] = []
    for cohort, comparisons in readout["paired_bootstrap_auc"].items():
        for comparison, values in comparisons.items():
            bootstrap_rows.append({"cohort": cohort, "comparison": comparison, **values})
    write_csv(args.output / "bootstrap_auc.csv", bootstrap_rows)

    quality_rows: list[dict[str, Any]] = []
    for cohort in PRIMARY_COHORTS:
        if cohort not in representation["posterior_quality"]:
            continue
        for method, values in representation["posterior_quality"][cohort].items():
            ess = values["ess"]
            maximum = values["max_weight"]
            entropy = values["normalized_entropy"]
            quality_rows.append({
                "cohort": cohort,
                "method": method,
                "events": ess["events"],
                "ess_mean": ess["mean"],
                "ess_min": ess["q00_q16_q50_q84_q95_q99_q100"][0],
                "ess_median": ess["q00_q16_q50_q84_q95_q99_q100"][2],
                "max_weight_mean": maximum["mean"],
                "max_weight_median": maximum["q00_q16_q50_q84_q95_q99_q100"][2],
                "max_weight_max": maximum["q00_q16_q50_q84_q95_q99_q100"][-1],
                "normalized_entropy_mean": entropy["mean"],
                "collapse_fraction_ess_below_10pct": values[
                    "collapse_fraction_ess_below_10pct"
                ],
                "collapse_fraction_max_weight_above_half": values[
                    "collapse_fraction_max_weight_above_half"
                ],
            })
    write_csv(args.output / "posterior_quality.csv", quality_rows)

    compact = {
        "contract": {
            "representation": representation["contract"],
            "readout": readout["contract"],
            "surface": surface["contract"],
        },
        "counts": {
            "representation": representation["counts"],
            "readout": readout["counts"],
            "surface": {
                "events": surface["events"],
                "valid": surface["surface_valid"],
                "invalid": surface["surface_invalid"],
                "valid_fraction": surface["surface_valid_fraction"],
            },
        },
        "paired_bootstrap_h": representation["paired_bootstrap_h"],
        "paired_bootstrap_auc": readout["paired_bootstrap_auc"],
        "oracle_gap_recovery": readout["oracle_gap_recovery"],
        "response_audit": representation["response_audit"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
