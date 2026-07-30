from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


STUDY_NAME = "fixed-partial-v2-ptmatched20-relative-v3-final-hpo-v1"
OUTPUT_NAME = "hpo_parameter_space_v1"
EXPECTED_TRIALS = 21
BEST_TRIAL = 5
LR_BOUNDS = (7.0e-5, 2.0e-4)
DROPOUT_BOUNDS = (0.0, 0.25)
PROFILE_ORDER = ["small", "current", "deep", "wide", "large"]
SCHEDULER_ORDER = ["constant", "cosine_warmup5"]

PROFILE_LABELS = {
    "small": "Small",
    "current": "Current",
    "deep": "Deep",
    "wide": "Wide",
    "large": "Large",
}
SCHEDULER_LABELS = {
    "constant": "Constant",
    "cosine_warmup5": "Cosine + 5% warmup",
}
PROFILE_COLORS = {
    "small": "#4E79A7",
    "current": "#F28E2B",
    "deep": "#7A8E36",
    "wide": "#D37295",
    "large": "#B6992D",
}
PROFILE_MARKERS = {
    "small": "o",
    "current": "s",
    "deep": "^",
    "wide": "D",
    "large": "P",
}
SCHEDULER_COLORS = {
    "constant": "#4E79A7",
    "cosine_warmup5": "#E17C05",
}
SCHEDULER_MARKERS = {
    "constant": "o",
    "cosine_warmup5": "^",
}
INK = "#252525"
MID_GREY = "#777777"
LIGHT_GREY = "#D7D7D7"
GRID_GREY = "#E5E5E5"
BEST_EDGE = "#111111"
BEST_FILL = "#FFD166"
OBJECTIVE_CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create observation-based visualizations of a completed HPO."
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default=OUTPUT_NAME,
        help="New subdirectory name under study-dir/figures.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(study_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    controller_path = study_dir / "controller_complete.json"
    controller = json.loads(controller_path.read_text())
    if controller.get("trial_state_counts") != {"COMPLETE": EXPECTED_TRIALS}:
        raise RuntimeError(
            f"Unexpected trial states: {controller.get('trial_state_counts')}"
        )

    ranking_path = study_dir / "final_selection" / "validation_ranking.csv"
    ranking_rows = list(csv.DictReader(ranking_path.open()))
    ranks = {int(row["trial_number"]): int(row["rank"]) for row in ranking_rows}

    source_paths = [controller_path, ranking_path, study_dir / "trials.csv"]
    rows: list[dict[str, Any]] = []
    result_paths = sorted((study_dir / "trials").glob("trial_*/result.json"))
    if len(result_paths) != EXPECTED_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRIALS} result files, found {len(result_paths)}"
        )
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        parameters = result["parameters"]
        trial = int(result["trial_number"])
        state = str(result["state"])
        if state != "COMPLETE":
            raise RuntimeError(f"Trial {trial} has state {state}")
        if not result["finite_training"]:
            raise RuntimeError(f"Trial {trial} reports non-finite training")
        if result["test_split_loaded"]:
            raise RuntimeError(f"Trial {trial} loaded test during HPO")
        lr = float(parameters["learning_rate"])
        objective = float(result["objective"])
        min_loss = float(result["minimum_validation_loss"])
        runtime = float(result["elapsed_seconds"])
        if not all(math.isfinite(x) for x in [lr, objective, min_loss, runtime]):
            raise RuntimeError(f"Trial {trial} has non-finite plot data")
        rows.append(
            {
                "trial": trial,
                "rank": ranks[trial],
                "objective": objective,
                "min_val_loss": min_loss,
                "profile": str(parameters["model_profile"]),
                "lr": lr,
                "log10_lr": math.log10(lr),
                "dropout": float(parameters["dropout"]),
                "scheduler": str(parameters["schedule_profile"]),
                "runtime_sec": runtime,
                "param_count": int(result["parameter_counts"]["total"]),
                "best_epoch": int(result["best_center_epoch"]),
                "stop_epoch": int(result["epochs_completed"]),
                "state": state,
                "is_best": trial == BEST_TRIAL,
                "stopped_early": bool(result["stopped_early"]),
                "stop_reason": result["stop_reason"] or "",
            }
        )
        source_paths.append(result_path)

    rows.sort(key=lambda row: row["trial"])
    if {row["trial"] for row in rows} != set(range(EXPECTED_TRIALS)):
        raise RuntimeError("Trial numbers are not exactly 0 through 20")
    best = [row for row in rows if row["is_best"]]
    if len(best) != 1 or best[0]["rank"] != 1:
        raise RuntimeError("Trial 5 is not the unique validation-selected best trial")
    if set(row["profile"] for row in rows) - set(PROFILE_ORDER):
        raise RuntimeError("Unknown model profile")
    if set(row["scheduler"] for row in rows) - set(SCHEDULER_ORDER):
        raise RuntimeError("Unknown scheduler")
    return rows, source_paths


def save_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trial",
        "rank",
        "objective",
        "min_val_loss",
        "profile",
        "lr",
        "log10_lr",
        "dropout",
        "scheduler",
        "runtime_sec",
        "param_count",
        "best_epoch",
        "stop_epoch",
        "state",
        "is_best",
        "stopped_early",
        "stop_reason",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def arrays(
    rows: list[dict[str, Any]], field: str, dtype=float
) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=dtype)


def objective_limits(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = arrays(rows, "objective")
    span = float(values.max() - values.min())
    pad = max(0.0012, span * 0.08)
    return float(values.min() - pad), float(values.max() + pad)


def add_grid(axis: plt.Axes, axis_name: str = "y") -> None:
    axis.grid(axis=axis_name, color=GRID_GREY, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)


def add_header(
    figure: plt.Figure, title: str, subtitle: str, top: float = 0.88
) -> None:
    figure.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=16, weight="bold")
    figure.text(0.08, 0.935, subtitle, ha="left", fontsize=9.5, color=MID_GREY)
    figure.subplots_adjust(top=top)


def add_footer(
    figure: plt.Figure,
    text: str = (
        "Observed trials only (n=21, TPE search); density reflects the sampler. "
        "Objective = best 3-epoch moving-average validation AUC."
    ),
) -> None:
    figure.text(0.08, 0.018, text, ha="left", va="bottom", fontsize=8.2, color=MID_GREY)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def profile_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=PROFILE_MARKERS[profile],
            color="none",
            markerfacecolor=PROFILE_COLORS[profile],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=PROFILE_LABELS[profile],
        )
        for profile in PROFILE_ORDER
    ]


def scheduler_legend_handles(
    colors: dict[str, str] | None = None,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=SCHEDULER_MARKERS[scheduler],
            color="none",
            markerfacecolor=(
                colors[scheduler] if colors else "#F2F2F2"
            ),
            markeredgecolor=(colors[scheduler] if colors else INK),
            markeredgewidth=1.0,
            markersize=8,
            label=SCHEDULER_LABELS[scheduler],
        )
        for scheduler in SCHEDULER_ORDER
    ]


def highlight_best(
    axis: plt.Axes,
    x: float,
    y: float,
    *,
    text: str = "Trial 5",
    xytext: tuple[float, float] = (12, 13),
) -> None:
    axis.scatter(
        [x],
        [y],
        s=190,
        marker="o",
        facecolors="none",
        edgecolors=BEST_EDGE,
        linewidths=2.2,
        zorder=8,
    )
    axis.scatter(
        [x],
        [y],
        s=60,
        marker="*",
        color=BEST_FILL,
        edgecolors=BEST_EDGE,
        linewidths=0.8,
        zorder=9,
    )
    axis.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=9,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": MID_GREY, "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": LIGHT_GREY},
        zorder=10,
    )


def plot_optimization_history(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 6.0))
    trials = arrays(rows, "trial", int)
    objective = arrays(rows, "objective")
    best_so_far = np.maximum.accumulate(objective)
    axis.step(
        trials,
        best_so_far,
        where="post",
        color=INK,
        linewidth=1.8,
        label="Best so far",
        zorder=2,
    )
    for row in rows:
        axis.scatter(
            row["trial"],
            row["objective"],
            s=72,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            color=PROFILE_COLORS[row["profile"]],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    best = next(row for row in rows if row["is_best"])
    highlight_best(axis, best["trial"], best["objective"], xytext=(12, -31))
    axis.set(
        xlabel="Trial number",
        ylabel="Validation objective AUC",
        xlim=(-0.8, EXPECTED_TRIALS - 0.2),
        ylim=objective_limits(rows),
    )
    axis.set_xticks(range(0, EXPECTED_TRIALS, 2))
    add_grid(axis)
    profile_legend = axis.legend(
        handles=profile_legend_handles(),
        title="Profile (color)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.53),
        ncol=2,
        frameon=True,
        fontsize=8.5,
    )
    axis.add_artist(profile_legend)
    axis.legend(
        handles=[
            *scheduler_legend_handles(),
            Line2D([0], [0], color=INK, lw=1.8, label="Best so far"),
        ],
        title="Scheduler / guide",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
    )
    add_header(
        figure,
        "Optimization history",
        "Validation-only objective by trial; line shows the cumulative best.",
    )
    add_footer(figure)
    figure.subplots_adjust(bottom=0.13, right=0.76)
    save_figure(figure, output_dir, "optimization_history")


def plot_continuous_projection(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    field: str,
    xlabel: str,
    title: str,
    stem: str,
    xlim: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(10.6, 5.9))
    for row in rows:
        axis.scatter(
            row[field],
            row["objective"],
            s=76,
            marker=PROFILE_MARKERS[row["profile"]],
            color=SCHEDULER_COLORS[row["scheduler"]],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.93,
            zorder=3,
        )
    best = next(row for row in rows if row["is_best"])
    highlight_best(axis, best[field], best["objective"], xytext=(12, 12))
    axis.set(
        xlabel=xlabel,
        ylabel="Validation objective AUC",
        xlim=xlim,
        ylim=objective_limits(rows),
    )
    add_grid(axis)
    scheduler_legend = axis.legend(
        handles=scheduler_legend_handles(SCHEDULER_COLORS),
        title="Scheduler (color)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.54),
        fontsize=8.5,
    )
    axis.add_artist(scheduler_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=PROFILE_MARKERS[p],
                color="none",
                markerfacecolor="#EEEEEE",
                markeredgecolor=INK,
                markersize=8,
                label=PROFILE_LABELS[p],
            )
            for p in PROFILE_ORDER
        ],
        title="Profile (marker)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=2,
        fontsize=8.5,
    )
    add_header(
        figure,
        title,
        "Each point is one completed trial; no fitted response curve is shown.",
    )
    add_footer(figure)
    figure.subplots_adjust(bottom=0.13, right=0.72)
    save_figure(figure, output_dir, stem)


def deterministic_jitter(trial: int, scale: float = 0.13) -> float:
    value = ((trial * 37 + 11) % 101) / 100.0
    return (value - 0.5) * 2.0 * scale


def plot_discrete_projection(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    field: str,
    categories: list[str],
    labels: dict[str, str],
    stem: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.7, 5.9))
    positions = np.arange(len(categories))
    grouped = [
        [row["objective"] for row in rows if row[field] == category]
        for category in categories
    ]
    box = axis.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.3},
        boxprops={"facecolor": "#F4F4F4", "edgecolor": "#B8B8B8"},
        whiskerprops={"color": "#B8B8B8"},
        capprops={"color": "#B8B8B8"},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.65)
    for row in rows:
        x = categories.index(row[field]) + deterministic_jitter(row["trial"])
        if field == "profile":
            color = PROFILE_COLORS[row["profile"]]
            marker = SCHEDULER_MARKERS[row["scheduler"]]
        else:
            color = PROFILE_COLORS[row["profile"]]
            marker = PROFILE_MARKERS[row["profile"]]
        axis.scatter(
            x,
            row["objective"],
            s=70,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=4,
        )
    best = next(row for row in rows if row["is_best"])
    best_x = categories.index(best[field]) + deterministic_jitter(best["trial"])
    highlight_best(axis, best_x, best["objective"], xytext=(11, 12))
    counts = Counter(row[field] for row in rows)
    axis.set_xticks(
        positions,
        [f"{labels[c]}\n(n={counts[c]})" for c in categories],
    )
    axis.set(
        ylabel="Validation objective AUC",
        xlim=(-0.65, len(categories) - 0.35),
        ylim=objective_limits(rows),
    )
    add_grid(axis)
    if field == "profile":
        axis.legend(
            handles=scheduler_legend_handles(),
            title="Scheduler (marker)",
            loc="lower right",
            fontsize=8.5,
        )
    else:
        axis.legend(
            handles=profile_legend_handles(),
            title="Profile (color / marker)",
            loc="lower right",
            ncol=2,
            fontsize=8.5,
        )
    add_header(
        figure,
        f"{field.capitalize()} vs validation objective",
        "Raw trial points with a light boxplot; category counts reflect TPE sampling.",
    )
    add_footer(figure)
    figure.subplots_adjust(bottom=0.15)
    save_figure(figure, output_dir, stem)


def plot_lr_dropout_scatter(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.4))
    values = arrays(rows, "objective")
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    for row in rows:
        axis.scatter(
            row["log10_lr"],
            row["dropout"],
            s=105,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            c=[row["objective"]],
            cmap=OBJECTIVE_CMAP,
            norm=norm,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
    best = next(row for row in rows if row["is_best"])
    overlap_count = sum(
        row["log10_lr"] == best["log10_lr"]
        and row["dropout"] == best["dropout"]
        for row in rows
    )
    highlight_best(
        axis,
        best["log10_lr"],
        best["dropout"],
        text=(
            f"Trial 5 · AUC {best['objective']:.6f}\n"
            f"{overlap_count} trials share these LR/dropout coordinates"
        ),
        xytext=(14, 15),
    )
    axis.set(
        xlabel=r"$\log_{10}(\mathrm{learning\ rate})$",
        ylabel="Dropout",
        xlim=(math.log10(LR_BOUNDS[0]) - 0.015, math.log10(LR_BOUNDS[1]) + 0.015),
        ylim=(-0.012, DROPOUT_BOUNDS[1] + 0.012),
    )
    add_grid(axis, "both")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=OBJECTIVE_CMAP)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label("Validation objective AUC")
    axis.legend(
        handles=scheduler_legend_handles(),
        title="Scheduler (marker)",
        loc="upper right",
        fontsize=8.5,
    )
    add_header(
        figure,
        "Learning rate × dropout: observed HPO trials",
        "Color encodes validation objective; no interpolation or response surface.",
        top=0.87,
    )
    add_footer(figure)
    figure.subplots_adjust(bottom=0.13)
    save_figure(figure, output_dir, "lr_dropout_objective_scatter")


def plot_binned_summary(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.4))
    x = arrays(rows, "log10_lr")
    y = arrays(rows, "dropout")
    objective = arrays(rows, "objective")
    x_edges = np.linspace(math.log10(LR_BOUNDS[0]), math.log10(LR_BOUNDS[1]), 5)
    y_edges = np.linspace(DROPOUT_BOUNDS[0], DROPOUT_BOUNDS[1], 5)
    sums = np.zeros((4, 4), dtype=float)
    counts = np.zeros((4, 4), dtype=int)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, 3)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, 3)
    for xi, yi, value in zip(x_bin, y_bin, objective):
        sums[yi, xi] += value
        counts[yi, xi] += 1
    means = np.full((4, 4), np.nan)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask]
    mesh = axis.pcolormesh(
        x_edges,
        y_edges,
        means,
        cmap=OBJECTIVE_CMAP,
        shading="flat",
        edgecolors="white",
        linewidth=1.5,
    )
    for yi in range(4):
        for xi in range(4):
            width = x_edges[xi + 1] - x_edges[xi]
            height = y_edges[yi + 1] - y_edges[yi]
            label_x = x_edges[xi] + 0.08 * width
            label_y = y_edges[yi + 1] - 0.10 * height
            label = f"n={counts[yi, xi]}"
            if counts[yi, xi]:
                label += f"\n{means[yi, xi]:.4f}"
                rgba = plt.get_cmap(OBJECTIVE_CMAP)(mesh.norm(means[yi, xi]))
                luminance = (
                    0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                )
                text_color = "white" if luminance < 0.46 else INK
            else:
                text_color = INK
            axis.text(
                label_x,
                label_y,
                label,
                ha="left",
                va="top",
                fontsize=8.5,
                color=text_color,
            )
    axis.scatter(
        x,
        y,
        s=23,
        facecolors="none",
        edgecolors="#333333",
        linewidths=0.65,
        alpha=0.65,
        zorder=3,
    )
    best = next(row for row in rows if row["is_best"])
    highlight_best(axis, best["log10_lr"], best["dropout"], xytext=(-86, -4))
    axis.set(
        xlabel=r"$\log_{10}(\mathrm{learning\ rate})$",
        ylabel="Dropout",
        xlim=(x_edges[0], x_edges[-1]),
        ylim=(y_edges[0], y_edges[-1]),
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("Mean validation objective in occupied bin")
    add_header(
        figure,
        "Coarse 4×4 binned summary",
        "Descriptive auxiliary view only; labels show trial count and mean AUC.",
        top=0.87,
    )
    add_footer(
        figure,
        "Coarse bins summarize observed TPE trials; empty bins are not evidence of poor performance.",
    )
    figure.subplots_adjust(bottom=0.13)
    save_figure(figure, output_dir, "lr_dropout_binned_summary")


def plot_runtime(rows: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 6.0))
    for row in rows:
        axis.scatter(
            row["runtime_sec"] / 60.0,
            row["objective"],
            s=78,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            color=PROFILE_COLORS[row["profile"]],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    best = next(row for row in rows if row["is_best"])
    highlight_best(
        axis,
        best["runtime_sec"] / 60.0,
        best["objective"],
        xytext=(12, 12),
    )
    axis.set(
        xlabel="Trial runtime [min]",
        ylabel="Validation objective AUC",
        ylim=objective_limits(rows),
    )
    add_grid(axis)
    profile_legend = axis.legend(
        handles=profile_legend_handles(),
        title="Profile (color)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.53),
        ncol=2,
        fontsize=8.5,
    )
    axis.add_artist(profile_legend)
    axis.legend(
        handles=scheduler_legend_handles(),
        title="Scheduler (marker)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
    )
    add_header(
        figure,
        "Runtime vs validation objective",
        "Wall-clock runtime includes each trial's actual stopping epoch.",
    )
    add_footer(figure)
    figure.subplots_adjust(bottom=0.13, right=0.74)
    save_figure(figure, output_dir, "runtime_vs_objective")


def ranked_top5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["rank"])[:5]


def plot_top5_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    top = ranked_top5(rows)
    columns = [
        "Rank",
        "Trial",
        "Objective",
        "Min loss",
        "Profile",
        "LR",
        "Dropout",
        "Scheduler",
        "Best ep.",
        "Stop ep.",
    ]
    cell_text = []
    for row in top:
        cell_text.append(
            [
                f"{row['rank']}",
                f"{row['trial']}",
                f"{row['objective']:.6f}",
                f"{row['min_val_loss']:.6f}",
                PROFILE_LABELS[row["profile"]],
                f"{row['lr']:.3g}",
                f"{row['dropout']:.4f}",
                "cosine+5%" if row["scheduler"] == "cosine_warmup5" else "constant",
                f"{row['best_epoch']}",
                f"{row['stop_epoch']}",
            ]
        )
    figure, axis = plt.subplots(figsize=(13.2, 5.0))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.055, 0.055, 0.105, 0.095, 0.09, 0.095, 0.08, 0.12, 0.075, 0.075],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.65)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D2D2D2")
        if row_index == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        elif row_index == 1:
            cell.set_facecolor("#FFF2CC")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F8F8F8")
    figure.suptitle(
        "Top five validation-ranked trials",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.06,
        0.905,
        "Tie rule: within 0.001 of best objective, lower minimum validation loss wins.",
        ha="left",
        fontsize=9.5,
        color=MID_GREY,
    )
    add_footer(
        figure,
        "Ranking is validation-only. Trial 5 is the selected model; test metrics are not used.",
    )
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.80)
    save_figure(figure, output_dir, "top5_table")


def plot_top5_lr_dropout(rows: list[dict[str, Any]], output_dir: Path) -> None:
    top = ranked_top5(rows)
    figure, axis = plt.subplots(figsize=(8.6, 6.0))
    values = np.asarray([row["objective"] for row in top])
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    label_offsets = {
        1: (8, 8),
        2: (7, 7),
        3: (-8, 14),
        4: (8, -21),
        5: (7, 8),
    }
    for row in top:
        axis.scatter(
            row["log10_lr"],
            row["dropout"],
            s=145,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            c=[row["objective"]],
            cmap=OBJECTIVE_CMAP,
            norm=norm,
            edgecolors=BEST_EDGE if row["is_best"] else "white",
            linewidths=2.2 if row["is_best"] else 0.9,
            zorder=4,
        )
        axis.annotate(
            f"#{row['rank']} · T{row['trial']}",
            (row["log10_lr"], row["dropout"]),
            xytext=label_offsets[row["rank"]],
            textcoords="offset points",
            fontsize=9,
            weight="bold" if row["is_best"] else "normal",
        )
    best = next(row for row in top if row["is_best"])
    axis.scatter(
        best["log10_lr"],
        best["dropout"],
        s=230,
        marker="o",
        facecolors="none",
        edgecolors=BEST_EDGE,
        linewidths=2.2,
        zorder=8,
    )
    axis.scatter(
        best["log10_lr"],
        best["dropout"],
        s=68,
        marker="*",
        color=BEST_FILL,
        edgecolors=BEST_EDGE,
        linewidths=0.8,
        zorder=9,
    )
    axis.set(
        xlabel=r"$\log_{10}(\mathrm{learning\ rate})$",
        ylabel="Dropout",
        xlim=(math.log10(LR_BOUNDS[0]) - 0.015, math.log10(LR_BOUNDS[1]) + 0.015),
        ylim=(-0.012, DROPOUT_BOUNDS[1] + 0.012),
    )
    add_grid(axis, "both")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=OBJECTIVE_CMAP)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis, pad=0.02)
    colorbar.set_label("Validation objective AUC")
    add_header(
        figure,
        "Top five trials in the learning-rate–dropout plane",
        "All five use the Current profile and cosine + 5% warmup scheduler.",
        top=0.87,
    )
    add_footer(
        figure,
        "Ranks follow the validation-only tie rule; positions are observed trials, not fitted optima.",
    )
    figure.subplots_adjust(bottom=0.13)
    save_figure(figure, output_dir, "top5_lr_dropout")


def scatter_optimization_on_axis(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    trials = arrays(rows, "trial", int)
    objective = arrays(rows, "objective")
    axis.step(
        trials,
        np.maximum.accumulate(objective),
        where="post",
        color=INK,
        lw=1.5,
        zorder=2,
    )
    for row in rows:
        axis.scatter(
            row["trial"],
            row["objective"],
            s=35,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            color=PROFILE_COLORS[row["profile"]],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    best = next(row for row in rows if row["is_best"])
    axis.scatter(
        best["trial"],
        best["objective"],
        s=90,
        facecolors="none",
        edgecolors=BEST_EDGE,
        linewidths=1.8,
        zorder=5,
    )
    axis.set(xlabel="Trial", ylabel="Objective AUC", xlim=(-0.8, 20.8))
    axis.set_title("Optimization history", loc="left", weight="bold")
    add_grid(axis)


def discrete_on_axis(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    field: str,
    categories: list[str],
    labels: dict[str, str],
) -> None:
    for row in rows:
        x = categories.index(row[field]) + deterministic_jitter(row["trial"])
        axis.scatter(
            x,
            row["objective"],
            s=34,
            marker=(
                SCHEDULER_MARKERS[row["scheduler"]]
                if field == "profile"
                else PROFILE_MARKERS[row["profile"]]
            ),
            color=PROFILE_COLORS[row["profile"]],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    counts = Counter(row[field] for row in rows)
    axis.set_xticks(
        range(len(categories)),
        [f"{labels[c]}\n{counts[c]}" for c in categories],
        fontsize=8,
    )
    axis.set_ylabel("Objective AUC")
    axis.set_title(
        f"{field.capitalize()} distribution", loc="left", weight="bold"
    )
    add_grid(axis)


def lr_dropout_on_axis(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    values = arrays(rows, "objective")
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    for row in rows:
        axis.scatter(
            row["log10_lr"],
            row["dropout"],
            s=52,
            marker=SCHEDULER_MARKERS[row["scheduler"]],
            c=[row["objective"]],
            cmap=OBJECTIVE_CMAP,
            norm=norm,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    best = next(row for row in rows if row["is_best"])
    axis.scatter(
        best["log10_lr"],
        best["dropout"],
        s=110,
        facecolors="none",
        edgecolors=BEST_EDGE,
        linewidths=1.8,
        zorder=5,
    )
    axis.set(
        xlabel=r"$\log_{10}(\mathrm{LR})$",
        ylabel="Dropout",
        xlim=(math.log10(LR_BOUNDS[0]) - 0.015, math.log10(LR_BOUNDS[1]) + 0.015),
        ylim=(-0.012, DROPOUT_BOUNDS[1] + 0.012),
    )
    axis.set_title("Observed LR × dropout", loc="left", weight="bold")
    overlap_count = sum(
        row["log10_lr"] == best["log10_lr"]
        and row["dropout"] == best["dropout"]
        for row in rows
    )
    axis.text(
        0.02,
        0.97,
        f"Color = objective AUC · n={overlap_count} at trial 5 coordinates",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=MID_GREY,
    )
    add_grid(axis, "both")


def plot_summary_panel(rows: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    scatter_optimization_on_axis(axes[0, 0], rows)
    discrete_on_axis(
        axes[0, 1], rows, "profile", PROFILE_ORDER, PROFILE_LABELS
    )
    discrete_on_axis(
        axes[1, 0], rows, "scheduler", SCHEDULER_ORDER, SCHEDULER_LABELS
    )
    lr_dropout_on_axis(axes[1, 1], rows)
    y_limits = objective_limits(rows)
    axes[0, 0].set_ylim(y_limits)
    axes[0, 1].set_ylim(y_limits)
    axes[1, 0].set_ylim(y_limits)
    figure.suptitle(
        "TauSpin final HPO: validation performance landscape",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.06,
        0.95,
        "21 completed TPE trials · Relative-v3 · fixed-partial-v2 · trial 5 outlined",
        ha="left",
        fontsize=10,
        color=MID_GREY,
    )
    legend_handles = [
        *profile_legend_handles(),
        *scheduler_legend_handles(),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.035),
        frameon=True,
        fontsize=8.5,
    )
    figure.text(
        0.06,
        0.012,
        "Observed points only; TPE point density is not a uniform performance map. "
        "Objective = best 3-epoch moving-average validation AUC.",
        fontsize=8.5,
        color=MID_GREY,
    )
    figure.subplots_adjust(
        left=0.08, right=0.97, top=0.90, bottom=0.12, hspace=0.32, wspace=0.24
    )
    save_figure(figure, output_dir, "hpo_summary_panel")


def write_notes(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    top = ranked_top5(rows)
    counts_profile = Counter(row["profile"] for row in rows)
    counts_scheduler = Counter(row["scheduler"] for row in rows)
    best = top[0]
    runner = top[1]
    note = f"""# HPO性能分布の解釈メモ

## 結論

- validation-onlyの既定規則ではtrial {best['trial']}が最良。objectiveは`{best['objective']:.6f}`で、順位規則上の2位trial {runner['trial']}との差は`{best['objective'] - runner['objective']:.6f}`。
- 上位5 trialはすべてCurrent profileとcosine + 5% warmup schedulerだった。
- 上位5のlearning rateは`{min(row['lr'] for row in top):.3g}`〜`{max(row['lr'] for row in top):.3g}`、dropoutは`{min(row['dropout'] for row in top):.3f}`〜`{max(row['dropout'] for row in top):.3f}`にある。観測点ベースでは、LRがおおよそ`7e-5–1.5e-4`、dropoutが`0.10–0.18`の領域が有望に見える。
- trial 5のlearning rate `1e-4`とdropout `0.1`は、探索境界LR `{LR_BOUNDS[0]:.0e}–{LR_BOUNDS[1]:.0e}`、dropout `{DROPOUT_BOUNDS[0]:.2f}–{DROPOUT_BOUNDS[1]:.2f}`へ張り付いていない。
- trial 5は「唯一絶対の最適値」ではなく、21 trialと事前に固定したvalidation順位規則のもとで選ばれた最良点である。

## 読み方の注意

- 21 trialしかないため、滑らかな性能面や狭い最適領域は断定しない。主図はすべて観測点を表示し、補間等高線を作っていない。
- TPE探索は一様サンプリングではない。点が多い場所は、過去のtrialに基づくsamplerの提案バイアスを含み、「本質的に良い領域の体積」や確率密度を表さない。
- 4×4 binned summaryは補助的な記述図である。binのmean AUCはtrial数に依存し、空binは性能が悪いことを意味しない。
- runtimeはprofileだけでなくearly stoppingの停止epochにも依存するため、純粋なarchitecture速度比較ではない。
- profile別trial数: {dict((PROFILE_LABELS[k], counts_profile[k]) for k in PROFILE_ORDER)}
- scheduler別trial数: {dict((SCHEDULER_LABELS[k], counts_scheduler[k]) for k in SCHEDULER_ORDER)}

## 発表での推奨

1. `lr_dropout_objective_scatter`: 観測された探索点、性能、trial 5の位置を一枚で示す主役図。
2. `projection_profile_vs_objective`: Currentが上位を占め、大型化が必須でなかったことを示す。
3. `optimization_history`: TPE探索の進行とbest-so-farを示す。
4. 全体説明には`hpo_summary_panel`を使い、個別の議論では上の3図を拡大して使う。

## metricと選択規則

- 主指標: best 3-epoch moving-average validation AUC。
- 首位との差が`0.001`未満ならminimum validation loss、さらに同程度ならparameter数で順位付け。
- test結果は作図データにも順位付けにも使用していない。
"""
    (output_dir / "figure_notes.md").write_text(note)
    plain = (
        note.replace("# ", "")
        .replace("## ", "")
        .replace("`", "")
        .replace("- ", "・")
    )
    (output_dir / "figure_notes.txt").write_text(plain)


def write_chart_map(output_dir: Path) -> None:
    rows = [
        (
            "optimization_history",
            "探索順序とbest-so-far",
            "scatter + step line",
            "profile color; scheduler marker",
        ),
        (
            "projection_log10lr_vs_objective",
            "LRとobjectiveの観測関係",
            "scatter",
            "scheduler color; profile marker",
        ),
        (
            "projection_dropout_vs_objective",
            "dropoutとobjectiveの観測関係",
            "scatter",
            "scheduler color; profile marker",
        ),
        (
            "projection_profile_vs_objective",
            "profile別分布",
            "jittered points + light boxplot",
            "profile color; scheduler marker",
        ),
        (
            "projection_scheduler_vs_objective",
            "scheduler別分布",
            "jittered points + light boxplot",
            "profile color and marker",
        ),
        (
            "lr_dropout_objective_scatter",
            "LR×dropout上の観測性能",
            "scatter",
            "continuous objective color; scheduler marker",
        ),
        (
            "lr_dropout_binned_summary",
            "粗い2D記述集約",
            "4×4 binned mean + raw points",
            "continuous mean color; count labels",
        ),
        (
            "runtime_vs_objective",
            "計算時間と性能",
            "scatter",
            "profile color; scheduler marker",
        ),
        (
            "top5_table",
            "上位5設定の精密参照",
            "table",
            "best-row highlight",
        ),
        (
            "top5_lr_dropout",
            "上位5の位置",
            "scatter",
            "objective color; direct rank labels",
        ),
        (
            "hpo_summary_panel",
            "発表用総合",
            "2×2 panel",
            "shared profile and scheduler encodings",
        ),
    ]
    with (output_dir / "chart_map.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stem", "analytical_question", "chart_type", "encoding"])
        writer.writerows(rows)


def write_manifest(
    output_dir: Path,
    source_paths: Iterable[Path],
    rows: list[dict[str, Any]],
) -> None:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "figure_manifest.json":
            continue
        artifacts.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "study": STUDY_NAME,
        "generated_at": datetime.now().astimezone().isoformat(),
        "trial_count": len(rows),
        "complete_trials": sum(row["state"] == "COMPLETE" for row in rows),
        "best_trial": BEST_TRIAL,
        "performance_metric": (
            "best 3-epoch moving-average validation AUC"
        ),
        "test_metrics_used": False,
        "interpolation_used": False,
        "sampling": "TPE; not uniform",
        "source_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "artifacts": artifacts,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def validate_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    expected_stems = [
        "optimization_history",
        "projection_log10lr_vs_objective",
        "projection_dropout_vs_objective",
        "projection_profile_vs_objective",
        "projection_scheduler_vs_objective",
        "lr_dropout_objective_scatter",
        "lr_dropout_binned_summary",
        "runtime_vs_objective",
        "top5_table",
        "top5_lr_dropout",
        "hpo_summary_panel",
    ]
    expected_files = {
        "plot_data.csv",
        "figure_notes.md",
        "figure_notes.txt",
        "chart_map.csv",
        *{f"{stem}.{suffix}" for stem in expected_stems for suffix in ["png", "pdf"]},
    }
    present = {path.name for path in output_dir.iterdir()}
    missing = expected_files - present
    if missing:
        raise RuntimeError(f"Missing output files: {sorted(missing)}")
    for name in expected_files:
        if (output_dir / name).stat().st_size <= 0:
            raise RuntimeError(f"Empty output file: {name}")
    plot_rows = list(csv.DictReader((output_dir / "plot_data.csv").open()))
    if len(plot_rows) != EXPECTED_TRIALS:
        raise RuntimeError("plot_data.csv row count mismatch")
    if sum(row["is_best"] == "True" for row in plot_rows) != 1:
        raise RuntimeError("plot_data.csv best-trial flag mismatch")
    if next(row for row in plot_rows if row["is_best"] == "True")["trial"] != "5":
        raise RuntimeError("plot_data.csv does not identify trial 5 as best")
    for stem in expected_stems:
        png = output_dir / f"{stem}.png"
        image = plt.imread(png)
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 500:
            raise RuntimeError(f"Unexpected PNG dimensions for {stem}: {image.shape}")


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    if study_dir.name != STUDY_NAME:
        raise RuntimeError(f"Unexpected study directory: {study_dir.name}")
    figures_root = study_dir / "figures"
    final_output = figures_root / args.output_name
    if final_output.exists():
        raise FileExistsError(
            f"{final_output} already exists; refusing to overwrite"
        )
    figures_root.mkdir(exist_ok=True)
    temp_output = figures_root / f".{args.output_name}.tmp-{os.getpid()}"
    if temp_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_output}")
    temp_output.mkdir()

    try:
        configure_matplotlib()
        rows, source_paths = load_rows(study_dir)
        save_plot_data(temp_output / "plot_data.csv", rows)
        plot_optimization_history(rows, temp_output)
        plot_continuous_projection(
            rows,
            temp_output,
            field="log10_lr",
            xlabel=r"$\log_{10}(\mathrm{learning\ rate})$",
            title="Learning rate vs validation objective",
            stem="projection_log10lr_vs_objective",
            xlim=(
                math.log10(LR_BOUNDS[0]) - 0.015,
                math.log10(LR_BOUNDS[1]) + 0.015,
            ),
        )
        plot_continuous_projection(
            rows,
            temp_output,
            field="dropout",
            xlabel="Dropout",
            title="Dropout vs validation objective",
            stem="projection_dropout_vs_objective",
            xlim=(-0.012, DROPOUT_BOUNDS[1] + 0.012),
        )
        plot_discrete_projection(
            rows,
            temp_output,
            field="profile",
            categories=PROFILE_ORDER,
            labels=PROFILE_LABELS,
            stem="projection_profile_vs_objective",
        )
        plot_discrete_projection(
            rows,
            temp_output,
            field="scheduler",
            categories=SCHEDULER_ORDER,
            labels=SCHEDULER_LABELS,
            stem="projection_scheduler_vs_objective",
        )
        plot_lr_dropout_scatter(rows, temp_output)
        plot_binned_summary(rows, temp_output)
        plot_runtime(rows, temp_output)
        plot_top5_table(rows, temp_output)
        plot_top5_lr_dropout(rows, temp_output)
        plot_summary_panel(rows, temp_output)
        write_notes(temp_output, rows)
        write_chart_map(temp_output)
        validate_outputs(temp_output, rows)
        write_manifest(temp_output, source_paths, rows)
        temp_output.rename(final_output)
    except Exception:
        failed_output = figures_root / (
            f".{args.output_name}.failed-"
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        )
        if temp_output.exists():
            temp_output.rename(failed_output)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(final_output),
                "trial_count": len(rows),
                "best_trial": BEST_TRIAL,
                "artifact_count": len(list(final_output.iterdir())),
                "test_metrics_used": False,
                "interpolation_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
