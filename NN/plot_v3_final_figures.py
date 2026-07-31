from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from train import binary_roc_auc, roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final v3 figures from saved validation/test predictions."
    )
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--retrain-dir", type=Path, required=True)
    parser.add_argument("--matching-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
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


def selected_rows(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["selected"] == "1" and row["split"] == split:
                rows.append(row)
    sample_order = {"H": 0, "Z": 1}
    rows.sort(
        key=lambda row: (
            sample_order[row["sample"]],
            row["file_basename"],
            int(row["entry_index"]),
        )
    )
    return rows


def packed_event_numbers(processed_dir: Path, split: str) -> np.ndarray:
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    values = []
    for sample in ("H", "Z"):
        for record in metadata["shards"][split][sample]:
            shard = torch.load(
                processed_dir / record["path"],
                map_location="cpu",
                weights_only=True,
            )
            values.append(shard["event_numbers"].numpy())
    return np.concatenate(values)


def working_point(
    labels: np.ndarray, scores: np.ndarray, target: float = 0.70
) -> dict[str, Any]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    indices = np.r_[distinct, labels.size - 1]
    true_positive = np.cumsum(sorted_labels)[indices]
    false_positive = 1 + indices - true_positive
    tpr = np.r_[0, true_positive] / np.sum(labels == 1)
    fpr = np.r_[0, false_positive] / np.sum(labels == 0)
    thresholds = np.r_[math.inf, sorted_scores[indices]]
    eligible = np.flatnonzero(tpr >= target)
    if not eligible.size:
        raise RuntimeError("Signal-efficiency working point is unavailable")
    index = int(eligible[0])
    threshold = float(thresholds[index])
    predictions = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(predictions & positive))
    fn = int(np.sum(~predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))
    background_efficiency = fp / (fp + tn)
    return {
        "target_signal_efficiency": target,
        "threshold": threshold,
        "achieved_signal_efficiency": tp / (tp + fn),
        "background_efficiency": background_efficiency,
        "background_rejection": (
            math.inf if background_efficiency == 0 else 1 / background_efficiency
        ),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "curve_signal_efficiency": tpr,
        "curve_background_efficiency": fpr,
    }


def pt_bins(pt: np.ndarray) -> list[tuple[str, np.ndarray]]:
    bins = [("<180", pt < 180)]
    for low in range(180, 460, 20):
        bins.append((f"{low}–{low + 20}", (pt >= low) & (pt < low + 20)))
    bins.append(("≥460", pt >= 460))
    return bins


def pt_auc_rows(
    labels: np.ndarray, scores: np.ndarray, pt: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for order, (name, mask) in enumerate(pt_bins(pt)):
        bin_labels = labels[mask]
        h_events = int(np.sum(bin_labels == 1))
        z_events = int(np.sum(bin_labels == 0))
        auc = (
            binary_roc_auc(bin_labels, scores[mask])
            if h_events and z_events
            else math.nan
        )
        rows.append(
            {
                "bin_order": order,
                "bin_label": name,
                "h_events": h_events,
                "z_events": z_events,
                "total_events": int(mask.sum()),
                "auc": auc,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    final_dir = args.final_dir.resolve()
    retrain_dir = args.retrain_dir.resolve()
    matching_dir = args.matching_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    selected_config_path = final_dir / "selected_config.json"
    final_summary_path = final_dir / "final_summary.json"
    history_path = retrain_dir / "history.json"
    validation_predictions_path = final_dir / "validation_predictions.npz"
    test_predictions_path = final_dir / "test_predictions.npz"
    event_selection_path = matching_dir / "event_selection.csv.gz"
    dataset_audit_path = matching_dir / "dataset_audit.json"
    required = (
        selected_config_path,
        final_summary_path,
        history_path,
        validation_predictions_path,
        test_predictions_path,
        event_selection_path,
        dataset_audit_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    selected_config = json.loads(selected_config_path.read_text())
    final_summary = json.loads(final_summary_path.read_text())
    history = json.loads(history_path.read_text())
    dataset_audit = json.loads(dataset_audit_path.read_text())
    if final_summary["test_evaluation_count_for_this_model"] != 1:
        raise RuntimeError("Final summary is not a single test evaluation")
    if final_summary["test_used_for_selection"]:
        raise RuntimeError("Final summary reports test-based selection")
    if not dataset_audit["approved_for_gpu_smoke"]:
        raise RuntimeError("Dataset audit did not approve the packed dataset")
    expected_loader = selected_config["evaluation_loader"]
    if (
        int(expected_loader["num_workers"]) != 1
        or expected_loader["worker_partition"] != "shard"
    ):
        raise RuntimeError("Saved predictions do not declare canonical dataset order")

    validation_npz = np.load(validation_predictions_path)
    test_npz = np.load(test_predictions_path)
    prediction_data = {
        "validation": {
            "labels": validation_npz["labels"].astype(np.int64),
            "scores": validation_npz["scores"].astype(np.float64),
        },
        "test": {
            "labels": test_npz["labels"].astype(np.int64),
            "scores": test_npz["scores"].astype(np.float64),
        },
    }
    row_data: dict[str, list[dict[str, str]]] = {}
    pt: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        rows = selected_rows(event_selection_path, split)
        row_data[split] = rows
        labels = np.asarray(
            [1 if row["sample"] == "H" else 0 for row in rows], dtype=np.int64
        )
        if not np.array_equal(labels, prediction_data[split]["labels"]):
            raise RuntimeError(f"{split} prediction labels do not align")
        packed_numbers = packed_event_numbers(processed_dir, split)
        alignment = dataset_audit["prediction_order_alignment"][split]
        if (
            not alignment["source_and_packed_event_numbers_equal"]
            or int(alignment["events"]) != len(packed_numbers)
        ):
            raise RuntimeError(f"{split} dataset order audit did not pass")
        pt[split] = np.asarray(
            [float(row["truth_boson_pt_gev"]) for row in rows],
            dtype=np.float64,
        )

    for split in ("validation", "test"):
        labels = prediction_data[split]["labels"]
        scores = prediction_data[split]["scores"]
        fpr, tpr = roc_curve(labels, scores)
        prediction_data[split].update(
            {
                "fpr": fpr,
                "tpr": tpr,
                "auc": binary_roc_auc(labels, scores),
            }
        )
    wp = working_point(
        prediction_data["test"]["labels"], prediction_data["test"]["scores"]
    )
    test_pt_rows = pt_auc_rows(
        prediction_data["test"]["labels"],
        prediction_data["test"]["scores"],
        pt["test"],
    )
    output_dir.mkdir(parents=True)

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    for split, style in (("validation", "--"), ("test", "-")):
        data = prediction_data[split]
        axis.plot(
            data["fpr"],
            data["tpr"],
            style,
            linewidth=2.2,
            label=f"{split.capitalize()} (AUC={data['auc']:.6f})",
        )
    axis.plot([0, 1], [0, 1], ":", color="0.55")
    axis.scatter(
        [wp["background_efficiency"]],
        [wp["achieved_signal_efficiency"]],
        marker="*",
        s=170,
        color="gold",
        edgecolor="black",
        label="Test 70% signal-efficiency WP",
    )
    axis.set(
        xlabel="Background efficiency",
        ylabel="Signal efficiency",
        title="Validation and single-evaluation test ROC",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_pair(figure, output_dir, "roc_validation_test")

    labels = prediction_data["test"]["labels"]
    scores = prediction_data["test"]["scores"]
    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    bins = np.linspace(0, 1, 41)
    axis.hist(
        scores[labels == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label=f"Z (n={np.sum(labels == 0):,})",
    )
    axis.hist(
        scores[labels == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label=f"H (n={np.sum(labels == 1):,})",
    )
    axis.axvline(
        wp["threshold"], color="black", linestyle="--", label="70% signal-efficiency WP"
    )
    axis.set(
        xlabel="Classifier score P(H)",
        ylabel="Density",
        title="Final test score distribution",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_pair(figure, output_dir, "score_distribution_test")

    curve_fpr = wp["curve_background_efficiency"]
    curve_tpr = wp["curve_signal_efficiency"]
    rejection = np.divide(
        1.0,
        curve_fpr,
        out=np.full_like(curve_fpr, np.inf, dtype=np.float64),
        where=curve_fpr > 0,
    )
    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    finite = np.isfinite(rejection)
    axis.plot(curve_tpr[finite], rejection[finite], linewidth=2.2)
    axis.scatter(
        [wp["achieved_signal_efficiency"]],
        [wp["background_rejection"]],
        marker="*",
        s=170,
        color="gold",
        edgecolor="black",
    )
    axis.set(
        xlabel="Signal efficiency",
        ylabel="Background rejection",
        title="Final test working-point curve",
        yscale="log",
    )
    axis.grid(alpha=0.25)
    save_pair(figure, output_dir, "working_point_test")

    matrix = np.asarray(
        [
            [wp["true_negative"], wp["false_positive"]],
            [wp["false_negative"], wp["true_positive"]],
        ]
    )
    fraction = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    image = axis.imshow(fraction, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{fraction[row, column]:.1%}",
                ha="center",
                va="center",
                color="white" if fraction[row, column] > 0.55 else "black",
                fontsize=13,
            )
    axis.set(
        xticks=[0, 1],
        xticklabels=["Predicted Z", "Predicted H"],
        yticks=[0, 1],
        yticklabels=["True Z", "True H"],
        xlabel="Predicted class",
        ylabel="True class",
        title="Test confusion matrix at 70% signal efficiency",
    )
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="Row-normalized fraction")
    save_pair(figure, output_dir, "confusion_matrix_wp70")

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(test_pt_rows))
    auc = np.asarray([row["auc"] for row in test_pt_rows])
    axis.plot(x, auc, "o-", linewidth=2)
    axis.axhline(0.5, color="0.55", linestyle=":")
    axis.set(
        xticks=x,
        xticklabels=[row["bin_label"] for row in test_pt_rows],
        xlabel="Truth parent-boson pT bin [GeV]",
        ylabel="Test ROC AUC",
        title="Final test AUC by truth parent-boson pT",
    )
    axis.tick_params(axis="x", rotation=45)
    axis.grid(alpha=0.25)
    save_pair(figure, output_dir, "pt_binned_auc_test")

    epochs = np.asarray([int(row["epoch"]) for row in history])
    train_loss = np.asarray([float(row["epoch_train_loss"]) for row in history])
    validation_loss = np.asarray(
        [float(row["validation_loss"]) for row in history]
    )
    validation_auc = np.asarray(
        [float(row["validation_auc"]) for row in history]
    )
    rolling_auc = np.asarray(
        [
            math.nan if row["rolling_auc_3"] is None else float(row["rolling_auc_3"])
            for row in history
        ]
    )
    learning_rate = np.asarray([float(row["learning_rate"]) for row in history])
    selected_epoch = int(selected_config["selected_center_epoch"])

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    axis.plot(epochs, train_loss, label="Train")
    axis.plot(epochs, validation_loss, label="Validation")
    axis.axvline(selected_epoch, color="gold", label=f"Selected epoch {selected_epoch}")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="50-epoch loss history")
    axis.grid(alpha=0.25)
    axis.legend()
    save_pair(figure, output_dir, "learning_curve_loss")

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    axis.plot(epochs, validation_auc, label="Validation AUC")
    axis.plot(epochs, rolling_auc, label="3-epoch mean")
    axis.axvline(selected_epoch, color="gold", label=f"Selected epoch {selected_epoch}")
    axis.set(xlabel="Epoch", ylabel="ROC AUC", title="50-epoch validation AUC")
    axis.grid(alpha=0.25)
    axis.legend()
    save_pair(figure, output_dir, "learning_curve_auc")

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.plot(epochs, learning_rate, color="tab:purple")
    axis.set(xlabel="Epoch", ylabel="Learning rate", title="50-epoch learning-rate schedule")
    axis.grid(alpha=0.25)
    save_pair(figure, output_dir, "learning_rate_curve")

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    axes[0, 0].plot(
        prediction_data["test"]["fpr"],
        prediction_data["test"]["tpr"],
        label=f"Test AUC {prediction_data['test']['auc']:.4f}",
    )
    axes[0, 0].plot([0, 1], [0, 1], ":")
    axes[0, 0].set(title="Test ROC", xlabel="Background efficiency", ylabel="Signal efficiency")
    axes[0, 0].legend()
    axes[0, 1].hist(
        scores[labels == 0], bins=bins, density=True, histtype="step", label="Z"
    )
    axes[0, 1].hist(
        scores[labels == 1], bins=bins, density=True, histtype="step", label="H"
    )
    axes[0, 1].set(title="Test scores", xlabel="P(H)", ylabel="Density")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, train_loss, label="Train loss")
    axes[1, 0].plot(epochs, validation_loss, label="Val loss")
    axes[1, 0].axvline(selected_epoch, color="gold")
    axes[1, 0].set(title="50-epoch loss", xlabel="Epoch", ylabel="BCE")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Selected trial: {selected_config['selected_trial_number']}",
                f"Selected epoch: {selected_epoch}",
                f"Validation AUC: {final_summary['validation_auc']:.6f}",
                f"Validation loss: {final_summary['validation_loss']:.6f}",
                f"Test AUC: {final_summary['test_auc']:.6f}",
                f"Test loss: {final_summary['test_loss']:.6f}",
                f"Background rejection @ 70% signal: {wp['background_rejection']:.4f}",
                "Test was evaluated once after validation-only selection.",
            ]
        ),
        va="top",
        fontsize=13,
    )
    figure.suptitle("fixed-partial-v3 final 50-epoch summary", fontsize=18)
    save_pair(figure, output_dir, "final_summary_panel")

    with (output_dir / "pt_binned_auc_test.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_pt_rows[0]))
        writer.writeheader()
        writer.writerows(test_pt_rows)
    serializable_wp = {
        key: value
        for key, value in wp.items()
        if not isinstance(value, np.ndarray)
    }
    (output_dir / "working_point.json").write_text(
        json.dumps(serializable_wp, indent=2) + "\n"
    )
    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selected_trial_number": selected_config["selected_trial_number"],
        "selected_center_epoch": selected_epoch,
        "validation_auc": prediction_data["validation"]["auc"],
        "test_auc": prediction_data["test"]["auc"],
        "test_evaluation_count_for_this_model": 1,
        "test_used_for_selection": False,
        "prediction_order_audit": {
            "loader": expected_loader,
            "validation_events": len(row_data["validation"]),
            "test_events": len(row_data["test"]),
            "labels_aligned": True,
            "source_and_packed_event_numbers_aligned": True,
            "dataset_audit_sha256": sha256_file(dataset_audit_path),
        },
        "inputs": {str(path): sha256_file(path) for path in required},
        "figure_bases": [
            "roc_validation_test",
            "score_distribution_test",
            "working_point_test",
            "confusion_matrix_wp70",
            "pt_binned_auc_test",
            "learning_curve_loss",
            "learning_curve_auc",
            "learning_rate_curve",
            "final_summary_panel",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
