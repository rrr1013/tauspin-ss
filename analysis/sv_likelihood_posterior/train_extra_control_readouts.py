"""Train deterministic readouts for additional SV controls and placebo cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import jsonable
from train_readouts import (
    FixedMLP,
    SEED,
    auc_block,
    fit_fixed,
    full_scores_from_strict,
    load_existing_score,
    load_npz,
    paired_auc_bootstrap,
    save_model,
)


def plot_control_auc(
    output: Path,
    auc_metrics: dict[str, dict[str, dict[str, float]]],
    bootstrap: dict[str, dict[str, dict[str, float]]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    methods = (
        ("baseline_flow_mean", "baseline", "0.35"),
        ("sv_weighted_mean", "real SV", "#4C78A8"),
        ("offset_shuffle", "offset shuffle", "#F28E2B"),
        ("visible_axis", "visible axis", "#59A14F"),
        ("point_h", "direct point h", "#B279A2"),
    )
    comparisons = (
        ("sv_minus_baseline", "baseline", "0.35"),
        ("sv_minus_offset_shuffle", "offset shuffle", "#F28E2B"),
        ("sv_minus_visible_axis", "visible axis", "#59A14F"),
        ("sv_minus_random_direction_shuffle", "random dir.", "#E45756"),
    )
    cohorts = ("inclusive", "no_sv", "reco_threeprong_x_threeprong")
    for column, cohort in enumerate(cohorts):
        values = [auc_metrics[cohort][name]["weighted_auc"] for name, _, _ in methods]
        axes[0, column].scatter(
            np.arange(len(values)), values,
            color=[color for _, _, color in methods], s=60,
        )
        axes[0, column].set_xticks(
            np.arange(len(values)), [label for _, label, _ in methods],
            rotation=25, ha="right",
        )
        axes[0, column].set(ylabel="weighted H/Z AUC",
                            title=cohort.replace("_", " "))
        margin = max(max(values) - min(values), 1.0e-4) * 0.08
        axes[0, column].set_ylim(min(values) - margin, max(values) + margin)
        for position, value in enumerate(values):
            axes[0, column].annotate(
                f"{value:.6f}", (position, value), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8,
            )

        points = np.array([
            bootstrap[cohort][name]["difference"] for name, _, _ in comparisons
        ])
        lower = np.array([
            bootstrap[cohort][name]["ci_low"] for name, _, _ in comparisons
        ])
        upper = np.array([
            bootstrap[cohort][name]["ci_high"] for name, _, _ in comparisons
        ])
        for position, (point, low, high, (_, _, color)) in enumerate(
            zip(points, lower, upper, comparisons)
        ):
            axes[1, column].errorbar(
                position, point, yerr=[[point - low], [high - point]],
                fmt="none", ecolor=color, capsize=4, lw=2,
            )
        axes[1, column].scatter(
            np.arange(len(points)), points,
            color=[color for _, _, color in comparisons], s=50, zorder=3,
        )
        axes[1, column].axhline(0.0, color="0.5", ls="--", lw=1)
        axes[1, column].set_xticks(
            np.arange(len(points)), [label for _, label, _ in comparisons],
            rotation=25, ha="right",
        )
        axes[1, column].set(
            ylabel="AUC(real SV) - AUC(control)",
            title="paired event bootstrap (95% CI)",
        )
    fig.suptitle("Deterministic-readout SV controls and zero-SV placebo")
    fig.savefig(output / "extra_control_auc.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--train-representations", type=Path, required=True)
    parser.add_argument("--validation-representations", type=Path, required=True)
    parser.add_argument("--train-extra-controls", type=Path, required=True)
    parser.add_argument("--validation-extra-controls", type=Path, required=True)
    parser.add_argument("--primary-readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True); (args.output / "models").mkdir()
    device = torch.device(args.device)

    names = ("global_indices", "labels", "weights", "modes", "reco_h_mode")
    train = load_npz(args.train_inputs, names)
    validation = load_npz(args.validation_inputs, names)
    train_cohorts = load_npz(args.train_representations / "cohorts.npz")
    val_cohorts = load_npz(args.validation_representations / "cohorts.npz")
    if not np.array_equal(train["global_indices"], train_cohorts["global_indices"]):
        raise RuntimeError("Train identity mismatch")
    if not np.array_equal(validation["global_indices"], val_cohorts["global_indices"]):
        raise RuntimeError("Validation identity mismatch")
    train_strict = train_cohorts["strict"].astype(bool)
    val_strict = val_cohorts["strict"].astype(bool)
    train_rows = np.flatnonzero(train_strict); val_rows = np.flatnonzero(val_strict)

    scores: dict[str, np.ndarray] = {}
    histories = {}
    metadata = {}
    for name in ("offset_shuffle", "visible_axis"):
        train_values = np.load(
            args.train_extra_controls / f"mean_{name}.npy", mmap_mode="r"
        )[train_rows]
        val_values = np.load(
            args.validation_extra_controls / f"mean_{name}.npy", mmap_mode="r"
        )[val_rows]
        model, strict_score, history = fit_fixed(
            train_values.reshape(len(train_rows), 6), train["labels"][train_rows],
            val_values.reshape(len(val_rows), 6), validation["labels"][val_rows],
            device, FixedMLP(6), SEED,
        )
        scores[name] = full_scores_from_strict(strict_score, val_rows, len(validation["labels"]))
        histories[name] = history
        save_model(args.output / "models" / f"{name}.pt", model, name)
        metadata[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_events": int(len(train_rows)), "validation_events": int(len(val_rows)),
        }

    for name in (
        "baseline_flow_mean", "sv_weighted_mean", "matched_shuffle_mean",
        "historical_baseline_flow_mean", "point_h", "exact_h",
    ):
        scores[name] = load_existing_score(
            args.primary_readouts / f"{name}_validation_scores.npz",
            validation["global_indices"], val_rows,
        )

    available = val_cohorts["sv_available"].astype(bool)
    reco_modes = validation["reco_h_mode"].astype(np.int8)
    surface_valid = val_cohorts["surface_valid"].astype(bool)
    masks = {
        name.removeprefix("mask_"): value.astype(bool)
        for name, value in val_cohorts.items() if name.startswith("mask_")
    }
    masks.update({
        "no_sv": val_strict & ~available.any(axis=1),
        "reco_threeprong_containing": val_strict & (reco_modes == 3).any(axis=1),
        "reco_threeprong_x_threeprong": val_strict & (reco_modes == 3).all(axis=1),
        "any_sv_exact_surface_valid": val_strict & available.any(axis=1) & surface_valid,
        "any_sv_exact_surface_invalid": val_strict & available.any(axis=1) & ~surface_valid,
    })
    labels = validation["labels"].astype(np.int8)
    event_weights = validation["weights"].astype(np.float64)
    auc_metrics = {
        cohort: {
            name: auc_block(labels, score, event_weights, mask)
            for name, score in scores.items()
        }
        for cohort, mask in masks.items()
    }
    bootstrap = {}
    bootstrap_cohorts = (
        "inclusive", "no_sv", "any_sv", "threeprong_x_threeprong",
        "reco_threeprong_x_threeprong", "any_sv_exact_surface_valid",
        "any_sv_exact_surface_invalid",
    )
    for index, cohort in enumerate(bootstrap_cohorts):
        mask = masks[cohort]
        bootstrap[cohort] = {
            "sv_minus_baseline": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["baseline_flow_mean"], labels,
                event_weights, mask, args.bootstrap_draws, 20261100 + 20 * index,
            ),
            "sv_minus_offset_shuffle": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["offset_shuffle"], labels,
                event_weights, mask, args.bootstrap_draws, 20261101 + 20 * index,
            ),
            "sv_minus_visible_axis": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["visible_axis"], labels,
                event_weights, mask, args.bootstrap_draws, 20261102 + 20 * index,
            ),
            "sv_minus_random_direction_shuffle": paired_auc_bootstrap(
                scores["sv_weighted_mean"], scores["matched_shuffle_mean"], labels,
                event_weights, mask, args.bootstrap_draws, 20261103 + 20 * index,
            ),
        }

    for name in ("offset_shuffle", "visible_axis"):
        np.savez_compressed(
            args.output / f"{name}_validation_scores.npz",
            scores=scores[name][val_rows], labels=labels[val_rows],
            overlap_weights=event_weights[val_rows],
            global_indices=validation["global_indices"][val_rows],
        )

    plot_control_auc(args.output, auc_metrics, bootstrap)

    report = {
        "contract": {
            "readout": "same deterministic reset, seed, minibatch order, architecture, and fixed endpoint as primary readouts-v2",
            "test_loaded": False,
        },
        "metadata": metadata,
        "auc_metrics": auc_metrics,
        "paired_bootstrap_auc": bootstrap,
        "histories": histories,
    }
    (args.output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
