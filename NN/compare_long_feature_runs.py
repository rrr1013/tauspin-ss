from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnose_relative_v3 import save_csv, save_plot, summarize, write_json


FEATURE_SETS = ("absolute-v1", "absolute-plus-parent-relative-v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine parallel long-step feature diagnostics."
    )
    parser.add_argument("--absolute-run-dir", type=Path, required=True)
    parser.add_argument("--relative-v3-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    arguments = parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {arguments.output_dir}"
        )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "absolute-v1": arguments.absolute_run_dir,
        "absolute-plus-parent-relative-v3": arguments.relative_v3_run_dir,
    }
    results: dict[str, dict[str, Any]] = {}
    source_summaries = {}
    source_configs = {}
    for feature_set in FEATURE_SETS:
        run_dir = run_dirs[feature_set]
        history = load_json(
            run_dir / feature_set / "history.json"
        )
        results[feature_set] = {
            "train_history": history["train_steps"],
            "validation_history": history["full_validation"],
        }
        source_summaries[feature_set] = load_json(
            run_dir / "summary.json"
        )
        source_configs[feature_set] = load_json(run_dir / "config.json")

    absolute = summarize(
        results["absolute-v1"]["validation_history"],
        arguments.max_steps,
    )
    relative = summarize(
        results["absolute-plus-parent-relative-v3"][
            "validation_history"
        ],
        arguments.max_steps,
    )
    summary = {
        "series": {
            "absolute-v1": absolute,
            "absolute-plus-parent-relative-v3": relative,
        },
        "differences_relative_minus_absolute": {
            "best_auc": relative["best_auc"] - absolute["best_auc"],
            "final_auc": relative["final_auc"] - absolute["final_auc"],
            "late_auc_mean": (
                relative["late_auc_mean"] - absolute["late_auc_mean"]
            ),
            "final_loss": relative["final_loss"] - absolute["final_loss"],
        },
        "source_summaries": source_summaries,
        "source_configs": source_configs,
        "parallel_gpu_execution": True,
        "test_split_loaded": False,
    }
    save_csv(arguments.output_dir / "validation_history.csv", results)
    save_plot(
        arguments.output_dir / "relative_v3_long_comparison",
        results,
    )
    write_json(arguments.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
