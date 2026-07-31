from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


PROFILE_ORDER = ["small", "current", "deep", "wide", "large"]
PROFILE_LABELS = {
    "small": "Small",
    "current": "Current",
    "deep": "Deep",
    "wide": "Wide",
    "large": "Large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validation-only figures for a completed v3 HPO."
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_pair(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=200)
    figure.savefig(output_dir / f"{name}.pdf")
    plt.close(figure)


def scatter_profile(
    axis: plt.Axes, rows: list[dict[str, Any]], selected_trial: int
) -> None:
    rng = np.random.default_rng(42)
    for index, profile in enumerate(PROFILE_ORDER):
        values = [row for row in rows if row["model_profile"] == profile]
        if not values:
            continue
        x = index + rng.uniform(-0.13, 0.13, len(values))
        y = [row["objective_auc"] for row in values]
        axis.scatter(x, y, alpha=0.78, s=42, label=PROFILE_LABELS[profile])
        axis.hlines(
            np.median(y), index - 0.22, index + 0.22, color="black", linewidth=2
        )
    selected = next(row for row in rows if row["trial_number"] == selected_trial)
    selected_x = PROFILE_ORDER.index(selected["model_profile"])
    axis.scatter(
        [selected_x],
        [selected["objective_auc"]],
        marker="*",
        s=220,
        color="gold",
        edgecolor="black",
        zorder=5,
        label=f"Selected trial {selected_trial}",
    )
    axis.set(
        xticks=range(len(PROFILE_ORDER)),
        xticklabels=[PROFILE_LABELS[name] for name in PROFILE_ORDER],
        xlabel="Model profile",
        ylabel="Best 3-epoch validation AUC",
    )
    axis.grid(alpha=0.25)


def main() -> int:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    study_config_path = study_dir / "study_config.json"
    controller_path = study_dir / "controller_complete.json"
    selection_path = selection_dir / "selected_parameters.json"
    ranking_path = selection_dir / "validation_ranking.json"
    for path in (
        study_config_path,
        controller_path,
        selection_path,
        ranking_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    study_config = json.loads(study_config_path.read_text())
    controller = json.loads(controller_path.read_text())
    selection = json.loads(selection_path.read_text())
    ranking = json.loads(ranking_path.read_text())
    binding = selection["data_binding"]
    selected_trial = int(selection["selected_trial_number"])
    rows: list[dict[str, Any]] = []
    rank_by_trial = {
        int(row["trial_number"]): int(row["selection_rank"]) for row in ranking
    }
    for result_path in sorted((study_dir / "trials").glob("trial_*/result.json")):
        result = json.loads(result_path.read_text())
        if result["state"] != "COMPLETE":
            continue
        if result["test_split_loaded"]:
            raise RuntimeError(f"HPO trial accessed test: {result_path}")
        if result["data_binding"] != binding:
            raise RuntimeError(f"Data binding mismatch: {result_path}")
        parameters = result["parameters"]
        rows.append(
            {
                "trial_number": int(result["trial_number"]),
                "selection_rank": rank_by_trial[int(result["trial_number"])],
                "selected": int(result["trial_number"]) == selected_trial,
                "objective_auc": float(result["objective"]),
                "minimum_validation_loss": float(
                    result["minimum_validation_loss"]
                ),
                "model_profile": parameters["model_profile"],
                "learning_rate": float(parameters["learning_rate"]),
                "dropout": float(parameters["dropout"]),
                "schedule_profile": parameters["schedule_profile"],
                "trainable_parameter_count": int(
                    result["parameter_counts"]["trainable"]
                ),
                "epochs_completed": int(result["epochs_completed"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
            }
        )
    if not rows or selected_trial not in {row["trial_number"] for row in rows}:
        raise RuntimeError("Completed HPO results do not contain selected trial")
    output_dir.mkdir(parents=True)
    fields = list(rows[0])
    with (output_dir / "plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    trials = np.asarray([row["trial_number"] for row in rows])
    objective = np.asarray([row["objective_auc"] for row in rows])
    order = np.argsort(trials)
    running_best = np.maximum.accumulate(objective[order])
    selected = next(row for row in rows if row["selected"])

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    axis.plot(trials[order], objective[order], "o", alpha=0.75, label="Trial")
    axis.plot(
        trials[order], running_best, color="tab:orange", linewidth=2, label="Best so far"
    )
    axis.scatter(
        [selected_trial],
        [selected["objective_auc"]],
        marker="*",
        s=220,
        color="gold",
        edgecolor="black",
        zorder=5,
        label=f"Selected trial {selected_trial}",
    )
    axis.set(
        xlabel="Trial number",
        ylabel="Best 3-epoch validation AUC",
        title="v3 HPO optimization history",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_pair(figure, output_dir, "hpo_optimization_history")

    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    scatter_profile(axis, rows, selected_trial)
    axis.set_title("Validation performance by model profile")
    axis.legend(loc="best", fontsize=9)
    save_pair(figure, output_dir, "hpo_profile_performance")

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    markers = {"constant": "o", "cosine_warmup5": "^"}
    color_norm = Normalize(vmin=float(objective.min()), vmax=float(objective.max()))
    for schedule, marker in markers.items():
        subset = [row for row in rows if row["schedule_profile"] == schedule]
        points = axis.scatter(
            [row["learning_rate"] for row in subset],
            [row["dropout"] for row in subset],
            c=[row["objective_auc"] for row in subset],
            cmap="viridis",
            norm=color_norm,
            marker=marker,
            s=75,
            edgecolor="0.2",
            linewidth=0.5,
            label=schedule,
        )
    axis.scatter(
        [selected["learning_rate"]],
        [selected["dropout"]],
        marker="*",
        s=260,
        color="gold",
        edgecolor="black",
        zorder=5,
    )
    axis.set_xscale("log")
    axis.set(
        xlabel="Learning rate",
        ylabel="Dropout",
        title="Observed learning-rate–dropout trials",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.colorbar(points, ax=axis, label="Best 3-epoch validation AUC")
    save_pair(figure, output_dir, "hpo_learning_rate_dropout_scatter")

    ranked = sorted(rows, key=lambda row: row["selection_rank"])[:15]
    figure, axis = plt.subplots(figsize=(9.0, 6.4))
    labels = [
        f"#{row['selection_rank']}  t{row['trial_number']}  "
        f"{PROFILE_LABELS[row['model_profile']]}"
        for row in ranked
    ]
    colors = ["gold" if row["selected"] else "tab:blue" for row in ranked]
    axis.barh(np.arange(len(ranked)), [row["objective_auc"] for row in ranked], color=colors)
    axis.set_yticks(np.arange(len(ranked)), labels=labels)
    axis.invert_yaxis()
    axis.set(
        xlabel="Best 3-epoch validation AUC",
        title="Validation-only HPO ranking (top 15)",
    )
    axis.grid(axis="x", alpha=0.25)
    save_pair(figure, output_dir, "hpo_trial_ranking")

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    axes[0, 0].plot(trials[order], objective[order], "o", alpha=0.7)
    axes[0, 0].plot(trials[order], running_best, color="tab:orange")
    axes[0, 0].set(title="Optimization history", xlabel="Trial", ylabel="Val AUC")
    scatter_profile(axes[0, 1], rows, selected_trial)
    axes[0, 1].set_title("Profiles")
    axes[1, 0].scatter(
        [row["learning_rate"] for row in rows],
        [row["dropout"] for row in rows],
        c=[row["objective_auc"] for row in rows],
        cmap="viridis",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(
        title="Learning rate vs dropout",
        xlabel="Learning rate",
        ylabel="Dropout",
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Completed trials: {len(rows)}",
                f"Selected: trial {selected_trial}",
                f"Profile: {PROFILE_LABELS[selected['model_profile']]}",
                f"Learning rate: {selected['learning_rate']:.4g}",
                f"Dropout: {selected['dropout']:.4f}",
                f"Schedule: {selected['schedule_profile']}",
                f"Objective AUC: {selected['objective_auc']:.6f}",
                f"Minimum val loss: {selected['minimum_validation_loss']:.6f}",
                "Selection used validation only.",
            ]
        ),
        va="top",
        fontsize=13,
    )
    figure.suptitle("fixed-partial-v3 validation-only HPO summary", fontsize=18)
    save_pair(figure, output_dir, "hpo_summary_panel")

    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "study_dir": str(study_dir),
        "selection_dir": str(selection_dir),
        "completed_trial_count": len(rows),
        "controller_state_counts": controller["trial_state_counts"],
        "selected_trial_number": selected_trial,
        "test_data_used": False,
        "data_binding": binding,
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (
                study_config_path,
                controller_path,
                selection_path,
                ranking_path,
            )
        },
        "figure_bases": [
            "hpo_optimization_history",
            "hpo_profile_performance",
            "hpo_learning_rate_dropout_scatter",
            "hpo_trial_ranking",
            "hpo_summary_panel",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
