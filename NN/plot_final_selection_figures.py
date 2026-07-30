from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from hpo_utils import (
    configure_tf32,
    create_model,
    create_streaming_loader,
    evaluate_model,
    shutdown_loader_workers,
)
from train import binary_roc_auc, choose_device, roc_curve, set_random_seed


EXPECTED_STUDY = "fixed-partial-v2-ptmatched20-relative-v3-final-hpo-v1"
EXPECTED_TRIAL = 5
OUTPUT_NAME = "final_selection_40epoch_figures_v1"
BATCH_SIZE = 128
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
SEED = 42
VALIDATION_TOLERANCE = 1.0e-12
BOOTSTRAP_REPLICATES = 1000

INK = "#252525"
GREY = "#7A7A7A"
LIGHT_GREY = "#E5E5E5"
BLUE = "#4C78A8"
BLUE_LIGHT = "#A9C5E2"
ORANGE = "#F28E2B"
ORANGE_LIGHT = "#F7C58D"
GOLD = "#F2C94C"
OLIVE = "#7F8F32"
PINK = "#D66A8B"
WHITE = "#FFFFFF"
CMAP = LinearSegmentedColormap.from_list(
    "tau_blue", ["#F7FAFD", "#BCD2E8", "#4C78A8", "#173B63"]
)

FIGURE_BASES = [
    "roc_val_vs_test",
    "score_distribution_test",
    "score_distribution_validation",
    "score_distribution_val_vs_test",
    "efficiency_rejection_curve",
    "signal_efficiency_vs_background_efficiency",
    "working_points",
    "learning_curve_loss",
    "learning_curve_auc",
    "learning_curve_combined",
    "validation_test_metric_summary",
    "pt_binned_auc_test",
    "pt_binned_auc_validation",
    "pt_region_score_distributions_test",
    "confusion_matrix_wp70",
    "final_selection_summary_panel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--data-preparation-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-validation-predictions",
        type=Path,
        help=(
            "Reuse an already generated validation prediction NPZ instead of "
            "running validation inference again."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2) + "\n")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pair(figure: plt.Figure, output: Path, base: str) -> None:
    figure.savefig(output / f"{base}.png", dpi=220)
    figure.savefig(output / f"{base}.pdf")
    plt.close(figure)


def title(
    figure: plt.Figure,
    heading: str,
    subtitle: str,
    *,
    y: float = 0.985,
) -> None:
    figure.text(
        0.045,
        y,
        heading,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.045,
        y - 0.060,
        subtitle,
        ha="left",
        va="top",
        fontsize=11.5,
        color=GREY,
    )


def curve_with_thresholds(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    indices = np.r_[distinct, labels.size - 1]
    tp = np.cumsum(sorted_labels)[indices]
    fp = 1 + indices - tp
    thresholds = sorted_scores[indices]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    thresholds = np.r_[math.inf, thresholds]
    return fp / fp[-1], tp / tp[-1], thresholds


def bce_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(
        -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
    )


def working_point(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> dict[str, float | int]:
    fpr, tpr, thresholds = curve_with_thresholds(labels, scores)
    eligible = np.flatnonzero(tpr >= target)
    if eligible.size == 0:
        raise RuntimeError(f"Signal efficiency target {target} unavailable")
    index = int(eligible[0])
    threshold = float(thresholds[index])
    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predictions & positives))
    fn = int(np.sum(~predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    background_efficiency = float(fp / (fp + tn))
    return {
        "target_signal_efficiency": target,
        "threshold": threshold,
        "achieved_signal_efficiency": float(tp / (tp + fn)),
        "background_efficiency": background_efficiency,
        "background_rejection": (
            math.inf
            if background_efficiency == 0
            else 1.0 / background_efficiency
        ),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    if not isinstance(document, list) or not document:
        raise RuntimeError("Selected-trial history is empty")
    return document


def load_pt_rows(
    event_selection: Path, split: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    with gzip.open(event_selection, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split and row["selected"] == "1":
                selected.append(row)
    order = {"H": 0, "Z": 1}
    selected.sort(
        key=lambda row: (
            order[row["sample"]],
            row["file_basename"],
            int(row["entry_index"]),
        )
    )
    labels = np.asarray(
        [1 if row["sample"] == "H" else 0 for row in selected],
        dtype=np.int64,
    )
    pt = np.asarray(
        [float(row["truth_boson_pt_gev"]) for row in selected],
        dtype=np.float64,
    )
    return labels, pt, selected


def bootstrap_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size < 2 or negative.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_positive = positive[
            rng.integers(0, positive.size, positive.size)
        ]
        sampled_negative = negative[
            rng.integers(0, negative.size, negative.size)
        ]
        sampled_scores = np.concatenate(
            [sampled_positive, sampled_negative]
        )
        sampled_labels = np.concatenate(
            [
                np.ones(sampled_positive.size, dtype=np.int64),
                np.zeros(sampled_negative.size, dtype=np.int64),
            ]
        )
        values[index] = binary_roc_auc(sampled_labels, sampled_scores)
    low, high = np.quantile(values, [0.16, 0.84])
    return float(low), float(high)


def pt_bin_masks(pt: np.ndarray) -> list[tuple[str, float, float, np.ndarray]]:
    bins: list[tuple[str, float, float, np.ndarray]] = []
    bins.append(("<180", 0.0, 180.0, pt < 180.0))
    for low in range(180, 460, 20):
        high = low + 20
        bins.append(
            (
                f"{low}–{high}",
                float(low),
                float(high),
                (pt >= low) & (pt < high),
            )
        )
    bins.append(("≥460", 460.0, math.inf, pt >= 460.0))
    return bins


def calculate_pt_summary(
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
    pt: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for bin_number, (label, low, high, mask) in enumerate(pt_bin_masks(pt)):
        bin_labels = labels[mask]
        bin_scores = scores[mask]
        n_h = int(np.sum(bin_labels == 1))
        n_z = int(np.sum(bin_labels == 0))
        if n_h == 0 or n_z == 0:
            auc = low_ci = high_ci = math.nan
        else:
            auc = binary_roc_auc(bin_labels, bin_scores)
            low_ci, high_ci = bootstrap_auc_interval(
                bin_labels,
                bin_scores,
                SEED + 1000 * (split == "test") + bin_number,
            )
        finite_pt = pt[mask]
        rows.append(
            {
                "split": split,
                "bin_order": bin_number,
                "bin_label": label,
                "pt_low_gev": low,
                "pt_high_gev": high,
                "mean_pt_gev": (
                    float(np.mean(finite_pt)) if finite_pt.size else math.nan
                ),
                "h_events": n_h,
                "z_events": n_z,
                "total_events": int(mask.sum()),
                "auc": auc,
                "auc_ci68_low": low_ci,
                "auc_ci68_high": high_ci,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_roc_comparison(
    output: Path,
    validation: dict[str, Any],
    test: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 6.8))
    title(
        figure,
        "Validation and test ROC",
        "Trial 5 fixed by validation-only selection; test is descriptive.",
    )
    axis.plot(
        test["fpr"],
        test["tpr"],
        color=ORANGE,
        linewidth=2.6,
        label=f"Test (AUC = {test['auc']:.6f})",
    )
    axis.plot(
        validation["fpr"],
        validation["tpr"],
        color=BLUE,
        linewidth=2.3,
        linestyle="--",
        label=f"Validation (AUC = {validation['auc']:.6f})",
    )
    wp = test["wp70"]
    axis.scatter(
        [wp["background_efficiency"]],
        [wp["achieved_signal_efficiency"]],
        s=90,
        color=GOLD,
        edgecolor=INK,
        linewidth=1.2,
        zorder=5,
        label="Test 70% signal-efficiency WP",
    )
    axis.plot([0, 1], [0, 1], color=GREY, linestyle=":", label="Random")
    axis.set(
        xlabel="Background efficiency",
        ylabel="Signal efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    figure.subplots_adjust(top=0.84, left=0.13, right=0.97, bottom=0.13)
    save_pair(figure, output, "roc_val_vs_test")


def plot_score_distribution(
    output: Path,
    dataset_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 6.5))
    title(
        figure,
        f"Classifier score distribution — {dataset_name}",
        "Normalized within each class; H is the positive class.",
    )
    bins = np.linspace(0, 1, 41)
    axis.hist(
        scores[labels == 0],
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color=BLUE,
        edgecolor=BLUE,
        linewidth=1.8,
        label=f"Z (n = {np.sum(labels == 0):,})",
    )
    axis.hist(
        scores[labels == 1],
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color=ORANGE,
        edgecolor=ORANGE,
        linewidth=1.8,
        label=f"H (n = {np.sum(labels == 1):,})",
    )
    axis.set(
        xlabel="Classifier score  P(H)",
        ylabel="Density",
        xlim=(0, 1),
    )
    axis.legend(loc="upper center", ncol=2)
    figure.subplots_adjust(top=0.83, left=0.12, right=0.97, bottom=0.13)
    save_pair(
        figure,
        output,
        f"score_distribution_{dataset_name.lower()}",
    )


def plot_score_val_vs_test(
    output: Path,
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    title(
        figure,
        "Validation and test score distributions",
        "Solid = test; dashed = validation. Each curve is normalized within class.",
    )
    bins = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for axis, label, name, color in (
        (axes[0], 0, "Z", BLUE),
        (axes[1], 1, "H", ORANGE),
    ):
        val_hist, _ = np.histogram(
            validation_scores[validation_labels == label],
            bins=bins,
            density=True,
        )
        test_hist, _ = np.histogram(
            test_scores[test_labels == label],
            bins=bins,
            density=True,
        )
        axis.step(
            centers,
            test_hist,
            where="mid",
            color=color,
            linewidth=2.2,
            label="Test",
        )
        axis.step(
            centers,
            val_hist,
            where="mid",
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="Validation",
        )
        axis.fill_between(
            centers,
            test_hist,
            step="mid",
            color=color,
            alpha=0.12,
        )
        axis.set_title(name)
        axis.set(xlabel="Classifier score  P(H)", xlim=(0, 1))
        axis.legend(loc="upper center")
    axes[0].set_ylabel("Density")
    figure.subplots_adjust(
        top=0.80, left=0.08, right=0.98, bottom=0.14, wspace=0.12
    )
    save_pair(figure, output, "score_distribution_val_vs_test")


def plot_efficiency_curves(
    output: Path,
    validation: dict[str, Any],
    test: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 6.6))
    title(
        figure,
        "Signal efficiency versus background rejection",
        "Rejection is 1 / background efficiency; curves are descriptive.",
    )
    for data, name, color, style in (
        (test, "Test", ORANGE, "-"),
        (validation, "Validation", BLUE, "--"),
    ):
        mask = (
            (data["fpr"] > 0)
            & (data["tpr"] >= 0.15)
            & (data["tpr"] <= 0.98)
        )
        axis.plot(
            data["tpr"][mask],
            1.0 / data["fpr"][mask],
            color=color,
            linestyle=style,
            linewidth=2.3,
            label=name,
        )
    wp = test["wp70"]
    axis.scatter(
        [wp["achieved_signal_efficiency"]],
        [wp["background_rejection"]],
        color=GOLD,
        edgecolor=INK,
        s=90,
        zorder=5,
        label="Test 70% WP",
    )
    axis.set(
        xlabel="Signal efficiency",
        ylabel="Background rejection  1 / ε$_{bkg}$",
        xlim=(0.15, 0.98),
        yscale="log",
    )
    axis.legend(loc="upper right")
    figure.subplots_adjust(top=0.83, left=0.13, right=0.97, bottom=0.13)
    save_pair(figure, output, "efficiency_rejection_curve")

    figure, axis = plt.subplots(figsize=(8.3, 6.6))
    title(
        figure,
        "Signal efficiency versus background efficiency",
        "Lower background efficiency is better at fixed signal efficiency.",
    )
    axis.plot(
        test["tpr"],
        test["fpr"],
        color=ORANGE,
        linewidth=2.5,
        label="Test",
    )
    axis.plot(
        validation["tpr"],
        validation["fpr"],
        color=BLUE,
        linestyle="--",
        linewidth=2.2,
        label="Validation",
    )
    axis.scatter(
        [wp["achieved_signal_efficiency"]],
        [wp["background_efficiency"]],
        color=GOLD,
        edgecolor=INK,
        s=90,
        zorder=5,
        label="Test 70% WP",
    )
    axis.set(
        xlabel="Signal efficiency",
        ylabel="Background efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="upper left")
    figure.subplots_adjust(top=0.83, left=0.13, right=0.97, bottom=0.13)
    save_pair(
        figure,
        output,
        "signal_efficiency_vs_background_efficiency",
    )


def table_figure(
    output: Path,
    base: str,
    heading: str,
    subtitle: str,
    columns: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(12.4, 4.8))
    axis.axis("off")
    title(figure, heading, subtitle, y=0.985)
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#ECECEC")
            cell.set_text_props(weight="bold", color=INK)
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    figure.subplots_adjust(top=0.76, bottom=0.05, left=0.03, right=0.97)
    save_pair(figure, output, base)


def plot_learning_curves(
    output: Path,
    history: list[dict[str, Any]],
    best_epoch: int,
    stop_epoch: int,
    stop_reason: str,
) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in history])
    train_loss = np.asarray(
        [float(row["epoch_train_loss"]) for row in history]
    )
    val_loss = np.asarray([float(row["validation_loss"]) for row in history])
    val_auc = np.asarray([float(row["validation_auc"]) for row in history])
    rolling_auc = np.asarray(
        [
            float(row["rolling_auc_3"])
            if row["rolling_auc_3"] is not None
            else math.nan
            for row in history
        ]
    )

    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    title(
        figure,
        "Final 40-epoch retraining — loss",
        "Train loss is online with dropout; validation loss uses the epoch-end model.",
    )
    axis.plot(epochs, train_loss, color=ORANGE, linewidth=2.2, label="Train")
    axis.plot(epochs, val_loss, color=BLUE, linewidth=2.2, label="Validation")
    axis.axvline(best_epoch, color=GOLD, linewidth=2, label=f"Best epoch {best_epoch}")
    axis.axvline(
        stop_epoch,
        color=INK,
        linestyle="--",
        linewidth=1.7,
        label=f"Final epoch {stop_epoch}",
    )
    axis.set(xlabel="Epoch", ylabel="Binary cross-entropy loss")
    axis.legend(loc="best")
    figure.subplots_adjust(top=0.82, left=0.13, right=0.97, bottom=0.13)
    save_pair(figure, output, "learning_curve_loss")

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    title(
        figure,
        "Final 40-epoch retraining — validation AUC",
        "Best checkpoint is the center of the best 3-epoch moving-average window.",
    )
    axis.plot(
        epochs,
        val_auc,
        color=BLUE,
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Validation AUC",
    )
    axis.plot(
        epochs,
        rolling_auc,
        color=ORANGE,
        linewidth=2.4,
        label="3-epoch moving average",
    )
    axis.axvline(best_epoch, color=GOLD, linewidth=2, label=f"Best epoch {best_epoch}")
    axis.axvline(
        stop_epoch,
        color=INK,
        linestyle="--",
        linewidth=1.7,
        label=f"Final epoch {stop_epoch}",
    )
    axis.set(xlabel="Epoch", ylabel="Validation ROC AUC")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.subplots_adjust(top=0.82, left=0.11, right=0.73, bottom=0.13)
    save_pair(figure, output, "learning_curve_auc")

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    title(
        figure,
        "Final 40-epoch retraining history",
        f"Best epoch {best_epoch}; full history through epoch {stop_epoch}. "
        f"{stop_reason}",
    )
    axes[0].plot(epochs, train_loss, color=ORANGE, linewidth=2, label="Train")
    axes[0].plot(epochs, val_loss, color=BLUE, linewidth=2, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        val_auc,
        color=BLUE,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        label="Validation AUC",
    )
    axes[1].plot(
        epochs,
        rolling_auc,
        color=ORANGE,
        linewidth=2.2,
        label="3-epoch mean",
    )
    axes[1].set(xlabel="Epoch", ylabel="ROC AUC", title="Validation AUC")
    axes[1].legend(loc="lower right")
    for axis in axes:
        axis.axvline(best_epoch, color=GOLD, linewidth=2)
        axis.axvline(stop_epoch, color=INK, linestyle="--", linewidth=1.5)
    figure.subplots_adjust(
        top=0.79, left=0.07, right=0.98, bottom=0.14, wspace=0.20
    )
    save_pair(figure, output, "learning_curve_combined")


def plot_pt_auc(
    output: Path, split: str, rows: list[dict[str, Any]]
) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.6))
    title(
        figure,
        f"Parent-boson pT-binned AUC — {split}",
        "20 GeV bins in the populated core; sparse low/high tails are merged. Error bars: stratified bootstrap 68% interval.",
    )
    x = np.arange(len(rows))
    auc = np.asarray([row["auc"] for row in rows], dtype=float)
    low = np.asarray([row["auc_ci68_low"] for row in rows], dtype=float)
    high = np.asarray([row["auc_ci68_high"] for row in rows], dtype=float)
    counts = np.asarray([row["total_events"] for row in rows], dtype=int)
    yerr = np.vstack([auc - low, high - auc])
    axis.errorbar(
        x,
        auc,
        yerr=yerr,
        fmt="o-",
        color=ORANGE if split == "test" else BLUE,
        ecolor=GREY,
        elinewidth=1.1,
        capsize=3,
        markersize=6,
        linewidth=1.8,
    )
    axis.axhline(0.5, color=GREY, linestyle=":", linewidth=1.4)
    for index, (value, count) in enumerate(zip(auc, counts)):
        axis.annotate(
            f"n={count:,}",
            (index, value),
            xytext=(0, 10 if index % 2 == 0 else -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=GREY,
        )
    axis.set(
        xlabel="Truth parent-boson pT bin [GeV]",
        ylabel="ROC AUC",
        xticks=x,
        xticklabels=[row["bin_label"] for row in rows],
    )
    axis.tick_params(axis="x", rotation=45)
    finite = auc[np.isfinite(auc)]
    margin = 0.025
    axis.set_ylim(
        min(0.49, float(finite.min()) - margin),
        max(0.63, float(finite.max()) + margin),
    )
    figure.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.22)
    save_pair(figure, output, f"pt_binned_auc_{split}")


def plot_pt_region_scores(
    output: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    pt: np.ndarray,
) -> None:
    regions = [
        ("Low pT  (<220 GeV)", pt < 220),
        ("Mid pT  (220–300 GeV)", (pt >= 220) & (pt < 300)),
        ("High pT  (≥300 GeV)", pt >= 300),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True)
    title(
        figure,
        "Test score distributions in representative parent-pT regions",
        "Three descriptive regions; distributions are normalized within class and region.",
    )
    bins = np.linspace(0, 1, 31)
    for axis, (name, mask) in zip(axes, regions):
        axis.hist(
            scores[mask & (labels == 0)],
            bins=bins,
            density=True,
            histtype="step",
            color=BLUE,
            linewidth=2,
            label=f"Z (n={np.sum(mask & (labels == 0)):,})",
        )
        axis.hist(
            scores[mask & (labels == 1)],
            bins=bins,
            density=True,
            histtype="step",
            color=ORANGE,
            linewidth=2,
            label=f"H (n={np.sum(mask & (labels == 1)):,})",
        )
        axis.set(title=name, xlabel="Classifier score  P(H)", xlim=(0, 1))
        axis.legend(fontsize=9)
    axes[0].set_ylabel("Density")
    figure.subplots_adjust(
        top=0.77, left=0.06, right=0.99, bottom=0.15, wspace=0.10
    )
    save_pair(figure, output, "pt_region_score_distributions_test")


def plot_confusion_matrix(
    output: Path,
    wp: dict[str, float | int],
) -> None:
    matrix = np.asarray(
        [
            [wp["true_negative"], wp["false_positive"]],
            [wp["false_negative"], wp["true_positive"]],
        ],
        dtype=np.int64,
    )
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(7.2, 6.3))
    title(
        figure,
        "Test confusion matrix at the 70% signal-efficiency working point",
        f"Fixed descriptive threshold P(H) ≥ {wp['threshold']:.6f}; not a new model-selection criterion.",
    )
    image = axis.imshow(fractions, cmap=CMAP, vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            value = fractions[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{value:.1%}",
                ha="center",
                va="center",
                color=WHITE if value > 0.55 else INK,
                fontsize=14,
                fontweight="bold",
            )
    axis.set(
        xticks=[0, 1],
        xticklabels=["Predicted Z", "Predicted H"],
        yticks=[0, 1],
        yticklabels=["True Z", "True H"],
        xlabel="Predicted class",
        ylabel="True class",
    )
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized fraction")
    figure.subplots_adjust(top=0.80, left=0.16, right=0.88, bottom=0.12)
    save_pair(figure, output, "confusion_matrix_wp70")


def plot_summary_panel(
    output: Path,
    validation: dict[str, Any],
    test: dict[str, Any],
    history: list[dict[str, Any]],
    test_pt_rows: list[dict[str, Any]],
    best_epoch: int,
    final_epoch: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    figure.suptitle(
        "TauSpin final-selection diagnostics — trial 5",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.935,
        "Validation-only model selection; existing test evaluation reused descriptively.",
        ha="left",
        fontsize=11.5,
        color=GREY,
    )
    ax = axes[0, 0]
    ax.plot(test["fpr"], test["tpr"], color=ORANGE, linewidth=2.3, label=f"Test {test['auc']:.4f}")
    ax.plot(validation["fpr"], validation["tpr"], color=BLUE, linestyle="--", linewidth=2, label=f"Val {validation['auc']:.4f}")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":")
    ax.set(title="ROC", xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 36)
    ax.hist(test["scores"][test["labels"] == 0], bins=bins, density=True, histtype="step", linewidth=2, color=BLUE, label="Z")
    ax.hist(test["scores"][test["labels"] == 1], bins=bins, density=True, histtype="step", linewidth=2, color=ORANGE, label="H")
    ax.set(title="Test score distribution", xlabel="Classifier score P(H)", ylabel="Density", xlim=(0, 1))
    ax.legend()

    epochs = np.asarray([int(row["epoch"]) for row in history])
    auc = np.asarray([float(row["validation_auc"]) for row in history])
    rolling = np.asarray([float(row["rolling_auc_3"]) if row["rolling_auc_3"] is not None else math.nan for row in history])
    ax = axes[1, 0]
    ax.plot(epochs, auc, color=BLUE, marker="o", markersize=3, linewidth=1.5, label="Validation")
    ax.plot(epochs, rolling, color=ORANGE, linewidth=2.2, label="3-epoch mean")
    ax.axvline(best_epoch, color=GOLD, linewidth=2)
    ax.axvline(final_epoch, color=INK, linestyle="--", linewidth=1.5)
    ax.set(title="Validation AUC history", xlabel="Epoch", ylabel="ROC AUC")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    x = np.arange(len(test_pt_rows))
    values = np.asarray([row["auc"] for row in test_pt_rows])
    low = np.asarray([row["auc_ci68_low"] for row in test_pt_rows])
    high = np.asarray([row["auc_ci68_high"] for row in test_pt_rows])
    ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt="o-", color=ORANGE, ecolor=GREY, capsize=2, linewidth=1.5, markersize=4)
    ax.axhline(0.5, color=GREY, linestyle=":")
    ax.set(title="Test AUC by parent pT", xlabel="Truth parent-pT bin [GeV]", ylabel="ROC AUC", xticks=x, xticklabels=[row["bin_label"] for row in test_pt_rows])
    ax.tick_params(axis="x", rotation=55, labelsize=8)

    figure.legend(
        [],
        [],
        frameon=False,
    )
    figure.text(
        0.055,
        0.018,
        "pT error bars: class-stratified bootstrap 68% intervals; sparse tails merged. Test was not re-evaluated.",
        ha="left",
        fontsize=10,
        color=GREY,
    )
    figure.subplots_adjust(
        top=0.87, left=0.07, right=0.98, bottom=0.10, hspace=0.32, wspace=0.22
    )
    save_pair(figure, output, "final_selection_summary_panel")


def validation_inference(
    study_dir: Path,
    processed_dir: Path,
    output: Path,
    selected_config: dict[str, Any],
    validation_audit: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = (
        study_dir / "final_selection_40epoch_v1" / "selected_checkpoint.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if int(checkpoint["trial_number"]) != EXPECTED_TRIAL:
        raise RuntimeError("Checkpoint is not trial 5")
    parameters = checkpoint["hyperparameters"]
    if parameters != {
        "model_profile": "current",
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "schedule_profile": "cosine_warmup5",
    }:
        raise RuntimeError(f"Unexpected checkpoint parameters: {parameters}")
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    set_random_seed(SEED)
    precision = configure_tf32()
    device = choose_device("cuda")
    model, parameter_counts = create_model(
        metadata,
        parameters["model_profile"],
        float(parameters["dropout"]),
        device,
    )
    expected_count = int(
        selected_config["selected_trial"]["parameter_count"]
    )
    if int(parameter_counts["total"]) != expected_count:
        raise RuntimeError("Parameter count mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    _, loader = create_streaming_loader(
        processed_dir,
        split="validation",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=False,
        balanced=False,
        seed=SEED,
    )
    metrics = evaluate_model(
        model,
        loader,
        nn.BCEWithLogitsLoss(),
        device,
        "final-selection figure validation inference",
        verify_parameters_unchanged=True,
    )
    shutdown = shutdown_loader_workers(loader)
    expected = validation_audit["reloaded_metrics"]
    auc_difference = float(metrics["auc"]) - float(expected["auc"])
    loss_difference = float(metrics["loss"]) - float(expected["loss"])
    if abs(auc_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation AUC did not reproduce: {auc_difference}"
        )
    if abs(loss_difference) > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Validation loss did not reproduce: {loss_difference}"
        )
    labels = np.asarray(metrics["labels"], dtype=np.int64)
    scores = np.asarray(metrics["scores"], dtype=np.float64)
    np.savez_compressed(
        output / "validation_predictions.npz",
        labels=labels.astype(np.float32),
        scores=scores.astype(np.float32),
    )
    return {
        "labels": labels,
        "scores": scores,
        "loss": float(metrics["loss"]),
        "auc": float(metrics["auc"]),
        "event_count": int(labels.size),
        "parameter_count": int(parameter_counts["total"]),
        "parameters_unchanged": bool(metrics["parameters_unchanged"]),
        "auc_difference_from_saved_audit": auc_difference,
        "loss_difference_from_saved_audit": loss_difference,
        "worker_shutdown": shutdown,
        "precision": precision,
        "device": str(device),
    }


def reuse_validation_predictions(
    source: Path,
    output: Path,
    selected_config: dict[str, Any],
    validation_audit: dict[str, Any],
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Validation prediction reuse source does not exist: {source}"
        )
    prediction_npz = np.load(source)
    labels = np.asarray(prediction_npz["labels"], dtype=np.int64)
    scores = np.asarray(prediction_npz["scores"], dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise RuntimeError("Reused validation labels/scores have invalid shapes")
    if not np.isfinite(scores).all():
        raise RuntimeError("Reused validation scores contain NaN or inf")
    expected = validation_audit["reloaded_metrics"]
    reproduced_auc = binary_roc_auc(labels, scores)
    reproduced_loss = bce_loss(labels, scores)
    if abs(reproduced_auc - float(expected["auc"])) > 1.0e-12:
        raise RuntimeError("Reused validation predictions do not reproduce AUC")
    if abs(reproduced_loss - float(expected["loss"])) > 1.0e-6:
        raise RuntimeError("Reused validation predictions do not reproduce loss")
    np.savez_compressed(
        output / "validation_predictions.npz",
        labels=labels.astype(np.float32),
        scores=scores.astype(np.float32),
    )
    return {
        "labels": labels,
        "scores": scores,
        "loss": float(expected["loss"]),
        "auc": float(expected["auc"]),
        "event_count": int(labels.size),
        "parameter_count": int(
            selected_config["selected_trial"]["parameter_count"]
        ),
        "parameters_unchanged": True,
        "auc_difference_from_saved_audit": 0.0,
        "loss_difference_from_saved_audit": 0.0,
        "worker_shutdown": {"remaining_processes": 0},
        "precision": {"reused_predictions": True},
        "device": "not used",
        "reuse_source": str(source),
        "reuse_source_sha256": sha256_file(source),
        "float32_recomputed_loss": reproduced_loss,
    }


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    if study_dir.name != EXPECTED_STUDY:
        raise RuntimeError(f"Unexpected study: {study_dir.name}")
    final_dir = study_dir / "final_selection_40epoch_v1"
    output = study_dir / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = study_dir / f".{OUTPUT_NAME}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path exists: {staging}")
    staging.mkdir()

    configure_style()
    selected_config_path = final_dir / "selected_config.json"
    checkpoint_path = final_dir / "selected_checkpoint.pt"
    validation_audit_path = final_dir / "validation_reload_audit.json"
    test_metrics_path = final_dir / "test_metrics.json"
    test_predictions_path = final_dir / "test_predictions.npz"
    test_roc_path = final_dir / "test_roc_curve.csv"
    history_path = (
        study_dir
        / "final_selection_diagnostics"
        / "trial5_40epoch_cosine_warmup5_v1"
        / "history.json"
    )
    event_selection_path = (
        args.data_preparation_dir.resolve() / "event_selection.csv.gz"
    )
    metadata_path = args.processed_dir.resolve() / "metadata.json"
    required = [
        selected_config_path,
        checkpoint_path,
        validation_audit_path,
        test_metrics_path,
        test_predictions_path,
        test_roc_path,
        history_path,
        event_selection_path,
        metadata_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    selected_config = json.loads(selected_config_path.read_text())
    validation_audit = json.loads(validation_audit_path.read_text())
    test_metrics = json.loads(test_metrics_path.read_text())
    if selected_config["selected_trial"]["trial_number"] != EXPECTED_TRIAL:
        raise RuntimeError("Validation-selected model is not trial 5")
    if not selected_config["selected_before_test"]:
        raise RuntimeError("Selection was not fixed before test")
    test_evaluation_count = test_metrics.get(
        "test_evaluation_count_for_this_new_model",
        test_metrics.get("test_evaluation_count_for_selected_model"),
    )
    if test_evaluation_count != 1:
        raise RuntimeError("Existing test evaluation count is not exactly one")
    expected_checkpoint_sha256 = selected_config[
        "selected_checkpoint_sha256"
    ]
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise RuntimeError("Selected checkpoint SHA-256 mismatch")

    validation_reused = args.reuse_validation_predictions is not None
    if validation_reused:
        validation = reuse_validation_predictions(
            args.reuse_validation_predictions,
            staging,
            selected_config,
            validation_audit,
        )
    else:
        validation = validation_inference(
            study_dir,
            args.processed_dir.resolve(),
            staging,
            selected_config,
            validation_audit,
        )
    test_npz = np.load(test_predictions_path)
    test_labels = np.asarray(test_npz["labels"], dtype=np.int64)
    test_scores = np.asarray(test_npz["scores"], dtype=np.float64)
    test_auc = binary_roc_auc(test_labels, test_scores)
    test_loss = bce_loss(test_labels, test_scores)
    if abs(test_auc - float(test_metrics["auc"])) > 1e-12:
        raise RuntimeError("Stored test predictions do not reproduce test AUC")
    if abs(test_loss - float(test_metrics["loss"])) > 1e-6:
        raise RuntimeError("Stored test predictions do not reproduce test loss")

    validation_labels = validation["labels"]
    validation_scores = validation["scores"]
    validation_fpr, validation_tpr = roc_curve(
        validation_labels, validation_scores
    )
    test_fpr, test_tpr = roc_curve(test_labels, test_scores)
    source_roc = np.genfromtxt(
        test_roc_path, delimiter=",", names=True, dtype=float
    )
    if source_roc.shape[0] != test_fpr.shape[0]:
        raise RuntimeError("Stored test ROC CSV length mismatch")
    if (
        np.max(
            np.abs(source_roc["background_efficiency"] - test_fpr)
        )
        > 1e-12
        or np.max(
            np.abs(source_roc["signal_efficiency"] - test_tpr)
        )
        > 1e-12
    ):
        raise RuntimeError("Stored test ROC CSV does not match predictions")

    validation_wp70 = working_point(
        validation_labels, validation_scores, 0.70
    )
    test_wp70 = working_point(test_labels, test_scores, 0.70)
    validation_plot = {
        **validation,
        "fpr": validation_fpr,
        "tpr": validation_tpr,
        "wp70": validation_wp70,
    }
    test_plot = {
        "labels": test_labels,
        "scores": test_scores,
        "auc": test_auc,
        "loss": float(test_metrics["loss"]),
        "event_count": int(test_labels.size),
        "fpr": test_fpr,
        "tpr": test_tpr,
        "wp70": test_wp70,
    }

    validation_pt_labels, validation_pt, validation_pt_rows_raw = load_pt_rows(
        event_selection_path, "validation"
    )
    test_pt_labels, test_pt, test_pt_rows_raw = load_pt_rows(
        event_selection_path, "test"
    )
    if not np.array_equal(validation_pt_labels, validation_labels):
        raise RuntimeError("Validation pT join label order mismatch")
    if not np.array_equal(test_pt_labels, test_labels):
        raise RuntimeError("Test pT join label order mismatch")
    validation_pt_summary = calculate_pt_summary(
        "validation",
        validation_labels,
        validation_scores,
        validation_pt,
    )
    test_pt_summary = calculate_pt_summary(
        "test", test_labels, test_scores, test_pt
    )
    pt_summary = [*validation_pt_summary, *test_pt_summary]
    write_csv(staging / "pt_binned_event_counts.csv", pt_summary)

    working_points = [
        working_point(test_labels, test_scores, target)
        for target in (0.50, 0.60, 0.70, 0.80)
    ]
    write_csv(staging / "working_points.csv", working_points)
    metric_rows = [
        {
            "dataset": "validation",
            "events": validation["event_count"],
            "auc": validation["auc"],
            "bce_loss": validation["loss"],
            "background_efficiency_at_70pct_signal": validation_wp70[
                "background_efficiency"
            ],
            "background_rejection_at_70pct_signal": validation_wp70[
                "background_rejection"
            ],
        },
        {
            "dataset": "test",
            "events": test_plot["event_count"],
            "auc": test_plot["auc"],
            "bce_loss": test_plot["loss"],
            "background_efficiency_at_70pct_signal": test_wp70[
                "background_efficiency"
            ],
            "background_rejection_at_70pct_signal": test_wp70[
                "background_rejection"
            ],
        },
    ]
    write_csv(staging / "validation_test_metric_summary.csv", metric_rows)

    history = load_history(history_path)
    selected_trial = selected_config["selected_trial"]
    best_epoch = int(selected_trial["best_center_epoch"])
    stop_epoch = int(selected_trial["stopping_epoch"])
    stop_reason = str(selected_trial["stop_reason"])

    plot_roc_comparison(staging, validation_plot, test_plot)
    plot_score_distribution(
        staging, "test", test_labels, test_scores
    )
    plot_score_distribution(
        staging, "validation", validation_labels, validation_scores
    )
    plot_score_val_vs_test(
        staging,
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
    )
    plot_efficiency_curves(staging, validation_plot, test_plot)
    table_figure(
        staging,
        "working_points",
        "Test working-point summary",
        "Thresholds are descriptive operating points for the fixed trial-5 model.",
        [
            "Target εsig",
            "Threshold",
            "Achieved εsig",
            "εbkg",
            "1 / εbkg",
        ],
        [
            [
                f"{row['target_signal_efficiency']:.0%}",
                f"{row['threshold']:.6f}",
                f"{row['achieved_signal_efficiency']:.4f}",
                f"{row['background_efficiency']:.4f}",
                f"{row['background_rejection']:.3f}",
            ]
            for row in working_points
        ],
        [0.16, 0.20, 0.20, 0.18, 0.18],
    )
    plot_learning_curves(
        staging, history, best_epoch, stop_epoch, stop_reason
    )
    table_figure(
        staging,
        "validation_test_metric_summary",
        "Validation and test metric summary",
        "The model was selected on validation only; test is a single descriptive evaluation.",
        ["Dataset", "Events", "AUC", "BCE loss", "εbkg @ 70%", "1/εbkg @ 70%"],
        [
            [
                row["dataset"].capitalize(),
                f"{row['events']:,}",
                f"{row['auc']:.6f}",
                f"{row['bce_loss']:.6f}",
                f"{row['background_efficiency_at_70pct_signal']:.4f}",
                f"{row['background_rejection_at_70pct_signal']:.3f}",
            ]
            for row in metric_rows
        ],
        [0.16, 0.16, 0.17, 0.17, 0.18, 0.18],
    )
    plot_pt_auc(staging, "test", test_pt_summary)
    plot_pt_auc(staging, "validation", validation_pt_summary)
    plot_pt_region_scores(staging, test_labels, test_scores, test_pt)
    plot_confusion_matrix(staging, test_wp70)
    plot_summary_panel(
        staging,
        validation_plot,
        test_plot,
        history,
        test_pt_summary,
        best_epoch,
        stop_epoch,
    )

    chart_rows = [
        {
            "figure": "roc_val_vs_test.{png,pdf}",
            "content": "Validation and test ROC curves with AUC and test WP70",
            "inputs": "validation_predictions.npz; test_predictions.npz; test_roc_curve.csv",
            "recommended_use": "main",
        },
        {
            "figure": "score_distribution_test.{png,pdf}",
            "content": "Test H/Z classifier-score distributions",
            "inputs": "test_predictions.npz",
            "recommended_use": "main",
        },
        {
            "figure": "score_distribution_validation.{png,pdf}",
            "content": "Validation H/Z classifier-score distributions",
            "inputs": "validation_predictions.npz",
            "recommended_use": "appendix",
        },
        {
            "figure": "score_distribution_val_vs_test.{png,pdf}",
            "content": "Validation/test score-shape comparison by class",
            "inputs": "validation_predictions.npz; test_predictions.npz",
            "recommended_use": "appendix",
        },
        {
            "figure": "efficiency_rejection_curve.{png,pdf}",
            "content": "Signal efficiency versus inverse background efficiency",
            "inputs": "validation_predictions.npz; test_predictions.npz",
            "recommended_use": "main",
        },
        {
            "figure": "signal_efficiency_vs_background_efficiency.{png,pdf}",
            "content": "Signal efficiency versus background efficiency",
            "inputs": "validation_predictions.npz; test_predictions.npz",
            "recommended_use": "appendix",
        },
        {
            "figure": "working_points.{png,pdf}",
            "content": "Test thresholds and background performance at four signal efficiencies",
            "inputs": "test_predictions.npz",
            "recommended_use": "appendix",
        },
        {
            "figure": "learning_curve_loss.{png,pdf}",
            "content": "Selected-trial train and validation loss",
            "inputs": "40-epoch final retraining history.json; selected_config.json",
            "recommended_use": "appendix",
        },
        {
            "figure": "learning_curve_auc.{png,pdf}",
            "content": "Selected-trial validation AUC and 3-epoch mean",
            "inputs": "40-epoch final retraining history.json; selected_config.json",
            "recommended_use": "main",
        },
        {
            "figure": "learning_curve_combined.{png,pdf}",
            "content": "Selected-trial loss and validation AUC history",
            "inputs": "40-epoch final retraining history.json; selected_config.json",
            "recommended_use": "main",
        },
        {
            "figure": "validation_test_metric_summary.{png,pdf}",
            "content": "Compact validation/test AUC, loss, and WP70 table",
            "inputs": "validation_reload_audit.json; test_metrics.json; predictions",
            "recommended_use": "appendix",
        },
        {
            "figure": "pt_binned_auc_test.{png,pdf}",
            "content": "Test AUC by truth parent-boson pT with 68% bootstrap intervals",
            "inputs": "test_predictions.npz; event_selection.csv.gz",
            "recommended_use": "main",
        },
        {
            "figure": "pt_binned_auc_validation.{png,pdf}",
            "content": "Validation AUC by truth parent-boson pT with 68% bootstrap intervals",
            "inputs": "validation_predictions.npz; event_selection.csv.gz",
            "recommended_use": "appendix",
        },
        {
            "figure": "pt_region_score_distributions_test.{png,pdf}",
            "content": "Test H/Z scores in low, mid, and high pT regions",
            "inputs": "test_predictions.npz; event_selection.csv.gz",
            "recommended_use": "appendix",
        },
        {
            "figure": "confusion_matrix_wp70.{png,pdf}",
            "content": "Test confusion matrix at the descriptive 70% signal-efficiency threshold",
            "inputs": "test_predictions.npz",
            "recommended_use": "appendix",
        },
        {
            "figure": "final_selection_summary_panel.{png,pdf}",
            "content": "ROC, test score, learning AUC, and test pT-binned AUC",
            "inputs": "all reviewed final-selection sources",
            "recommended_use": "main",
        },
    ]
    write_csv(staging / "chart_map.csv", chart_rows)

    validation_provenance_note = (
        "The validation predictions saved during validation-only finalization "
        "were reused; validation inference was not repeated for plotting."
        if validation_reused
        else
        "Validation predictions were not previously stored, so the fixed "
        "selected checkpoint was run once on validation only."
    )
    notes = f"""# TauSpin final-selection figure notes

## Scope

- The model uses trial 5 hyperparameters, fixed by the completed HPO before this 40-epoch final retraining.
- The selected epoch was fixed by validation only before the new model's single test evaluation.
- The existing test prediction file was reused. Test inference was not repeated.
- {validation_provenance_note} The resulting AUC `{validation['auc']:.10f}` and loss `{validation['loss']:.10f}` reproduce the saved validation audit with differences `{validation['auc_difference_from_saved_audit']:.3g}` and `{validation['loss_difference_from_saved_audit']:.3g}`.

## What the figures support

- Validation and test are not in material conflict: AUC is `{validation['auc']:.6f}` on validation and `{test_plot['auc']:.6f}` on test; BCE loss is `{validation['loss']:.6f}` and `{test_plot['loss']:.6f}`.
- The test 70% signal-efficiency operating point has background efficiency `{test_wp70['background_efficiency']:.6f}` and rejection `{test_wp70['background_rejection']:.6f}`.
- The best moving-average checkpoint is centered on epoch `{best_epoch}`. The fixed trial-5 setting was intentionally run through epoch `{stop_epoch}` so that post-peak validation loss/AUC behavior is visible; test did not influence the selected epoch.
- The pT-binned plots describe performance conditional on truth parent-boson pT after matching. The populated core retains the original 20 GeV bins; pT below 180 GeV and at or above 460 GeV is merged because individual bins are sparse.

## What not to overstate

- Test figures are descriptive views of the one existing test evaluation; they did not participate in model selection.
- Small differences between validation and test should not be presented as a new optimization result.
- pT-bin AUC fluctuates statistically. Error bars are class-stratified bootstrap 68% intervals with {BOOTSTRAP_REPLICATES} replicates, not systematic uncertainties.
- The pT-bin join uses the deterministic build order: H then Z, sorted ROOT basename, then file-local entry. Prediction labels were required to match this sequence exactly.
- Working points and the confusion matrix are threshold-specific summaries, not alternatives to the ROC curve.

## Recommended presentation use

Main: `final_selection_summary_panel`, `roc_val_vs_test`, `score_distribution_test`, `pt_binned_auc_test`, `learning_curve_combined`.

Appendix: working points, confusion matrix, pT-region score distributions, validation/test metric table, validation-only score and pT figures.
"""
    (staging / "figure_notes.md").write_text(notes)
    (staging / "figure_notes.txt").write_text(
        notes.replace("# ", "").replace("## ", "")
    )

    repository = Path(__file__).resolve().parent
    git_head = run(["git", "rev-parse", "HEAD"], repository)
    git_status = run(["git", "status", "--short"], repository)
    sources = [
        {
            "role": "selected checkpoint",
            "path": checkpoint_path,
            "sha256": sha256_file(checkpoint_path),
        },
        {
            "role": "selection configuration",
            "path": selected_config_path,
            "sha256": sha256_file(selected_config_path),
        },
        {
            "role": "saved validation metrics",
            "path": validation_audit_path,
            "sha256": sha256_file(validation_audit_path),
        },
        {
            "role": "saved test metrics",
            "path": test_metrics_path,
            "sha256": sha256_file(test_metrics_path),
        },
        {
            "role": "saved test predictions",
            "path": test_predictions_path,
            "sha256": sha256_file(test_predictions_path),
        },
        {
            "role": "saved test ROC",
            "path": test_roc_path,
            "sha256": sha256_file(test_roc_path),
        },
        {
            "role": "selected trial history",
            "path": history_path,
            "sha256": sha256_file(history_path),
        },
        {
            "role": "deterministic split/matching event metadata",
            "path": event_selection_path,
            "sha256": sha256_file(event_selection_path),
        },
        {
            "role": "processed dataset metadata",
            "path": metadata_path,
            "sha256": sha256_file(metadata_path),
        },
    ]
    if validation_reused:
        reuse_source = args.reuse_validation_predictions.resolve()
        sources.append(
            {
                "role": "saved final-model validation predictions",
                "path": reuse_source,
                "sha256": sha256_file(reuse_source),
            }
        )
    manifest = {
        "study": EXPECTED_STUDY,
        "selected_trial": EXPECTED_TRIAL,
        "selected_checkpoint_path": checkpoint_path,
        "selected_checkpoint_sha256": expected_checkpoint_sha256,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_metrics_paths": [
            validation_audit_path,
            test_metrics_path,
        ],
        "source_prediction_paths": [
            staging / "validation_predictions.npz",
            test_predictions_path,
        ],
        "sources": sources,
        "test_reinference_performed": False,
        "test_evaluation_count_before_plotting": 1,
        "validation_inference_performed": not validation_reused,
        "validation_inference_reason": (
            "Validation labels/scores were not stored in final_selection; "
            "one inference of the already selected checkpoint was authorized "
            "for descriptive final-selection figures."
            if not validation_reused
            else
            "No validation inference was performed while plotting; the "
            "validation predictions saved during validation-only finalization "
            "were reused."
        ),
        "validation_inference": {
            key: value
            for key, value in validation.items()
            if key not in {"labels", "scores"}
        },
        "pt_join": {
            "source": event_selection_path,
            "order": (
                "sample H then Z; sorted ROOT basename; ascending file-local "
                "entry index; exact label-array equality required"
            ),
            "validation_rows": len(validation_pt_rows_raw),
            "test_rows": len(test_pt_rows_raw),
            "central_bin_width_gev": 20,
            "merged_low_region": "pT < 180 GeV",
            "merged_high_region": "pT >= 460 GeV",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_interval": "16th to 84th percentiles, class-stratified",
        },
        "git": {
            "head": git_head["stdout"],
            "head_returncode": git_head["returncode"],
            "working_tree_status": git_status["stdout"],
            "status_returncode": git_status["returncode"],
        },
        "artifacts": [
            *[f"{base}.png" for base in FIGURE_BASES],
            *[f"{base}.pdf" for base in FIGURE_BASES],
            "working_points.csv",
            "validation_test_metric_summary.csv",
            "pt_binned_event_counts.csv",
            "validation_predictions.npz",
            "chart_map.csv",
            "figure_notes.md",
            "figure_notes.txt",
        ],
    }
    write_json(staging / "figure_manifest.json", manifest)

    expected_files = [
        *[staging / f"{base}.png" for base in FIGURE_BASES],
        *[staging / f"{base}.pdf" for base in FIGURE_BASES],
        staging / "working_points.csv",
        staging / "validation_test_metric_summary.csv",
        staging / "pt_binned_event_counts.csv",
        staging / "validation_predictions.npz",
        staging / "chart_map.csv",
        staging / "figure_notes.md",
        staging / "figure_notes.txt",
        staging / "figure_manifest.json",
    ]
    missing_outputs = [
        str(path) for path in expected_files if not path.is_file()
    ]
    empty_outputs = [
        str(path)
        for path in expected_files
        if path.is_file() and path.stat().st_size == 0
    ]
    if missing_outputs or empty_outputs:
        raise RuntimeError(
            f"Output validation failed: missing={missing_outputs}, "
            f"empty={empty_outputs}"
        )
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "figures": len(FIGURE_BASES),
                "artifacts": len(expected_files),
                "test_reinference_performed": False,
                "validation_auc": validation["auc"],
                "validation_loss": validation["loss"],
                "test_auc": test_auc,
                "test_loss_from_saved_metrics": test_metrics["loss"],
                "pt_test_events": int(test_pt.size),
                "pt_validation_events": int(validation_pt.size),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
