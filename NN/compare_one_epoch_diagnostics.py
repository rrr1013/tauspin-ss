from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Raw and pT-matched one-epoch diagnostics."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_history(path: Path) -> dict:
    return json.loads((path / "diagnostic_history.json").read_text())


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_history(arguments.raw_dir)
    matched = load_history(arguments.matched_dir)
    raw_summary = load_summary(arguments.raw_dir)
    matched_summary = load_summary(arguments.matched_dir)

    figure, axes = plt.subplots(
        3, 1, figsize=(10, 12), layout="constrained"
    )
    colors = {"Raw": "tab:blue", "pT-matched": "tab:orange"}
    for label, history in (("Raw", raw), ("pT-matched", matched)):
        train = history["train_steps"]
        validation = history["validation_subset"]
        axes[0].plot(
            [item["optimizer_step"] for item in train],
            [item["train_loss_ma50"] for item in train],
            color=colors[label],
            linewidth=1.6,
            label=label,
        )
        axes[1].plot(
            [item["optimizer_step"] for item in validation],
            [item["validation_loss"] for item in validation],
            marker="o",
            markersize=3,
            color=colors[label],
            label=label,
        )
        axes[2].plot(
            [item["optimizer_step"] for item in validation],
            [item["validation_auc"] for item in validation],
            marker="o",
            markersize=3,
            color=colors[label],
            label=label,
        )

    axes[0].set_ylabel("Train BCE loss (MA50)")
    axes[1].set_ylabel("Validation subset BCE loss")
    axes[2].set_ylabel("Validation subset ROC AUC")
    axes[2].set_xlabel("Optimizer step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlim(left=0)
    figure.suptitle(
        "Fixed partial sample: Raw vs 20 GeV pT-matched\n"
        "Current Transformer, identical baseline hyperparameters"
    )
    figure.savefig(arguments.output_dir / "raw_vs_ptmatched_one_epoch.pdf")
    figure.savefig(arguments.output_dir / "raw_vs_ptmatched_one_epoch.png")
    plt.close(figure)

    comparison = {
        "raw": raw_summary,
        "pt_matched": matched_summary,
        "full_validation_auc_difference_matched_minus_raw": (
            matched_summary["full_validation"]["validation_auc"]
            - raw_summary["full_validation"]["validation_auc"]
        ),
        "full_validation_loss_difference_matched_minus_raw": (
            matched_summary["full_validation"]["validation_loss"]
            - raw_summary["full_validation"]["validation_loss"]
        ),
        "same_model_and_hyperparameters": True,
        "comparison_note": (
            "Each dataset uses a fixed class-balanced validation subset drawn "
            "from its own validation distribution."
        ),
    }
    (arguments.output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
