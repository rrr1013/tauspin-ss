"""Audit whether truth and reconstructed tau-neutrino sheets have a stable label.

This is the first work packet of the reco-sheet identifiability study.  It does
not train a classifier and it never reads the test split.  For a deterministic
subset of the training cohort it compares

  * the globally defined truth-surface sheet
    ``2 * root(tau-) + root(tau+)``; and
  * the sheet of the reconstructed visible+MET surface containing the
    hypothesis closest to the two truth tau flight directions.

The comparison deliberately exposes reconstruction-induced migration.  If the
mapping is not stable, the downstream task must be candidate ranking rather
than four-class truth-sheet classification.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAU_MASS_GEV = 1.7769


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--proposal-lattice", type=int, default=64)
    parser.add_argument("--keep", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Stable vectorised 64-bit hash for deterministic audit subsampling."""
    x = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def invariant_mass(four: np.ndarray) -> np.ndarray:
    mass2 = four[..., 3] ** 2 - np.sum(four[..., :3] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def opening_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def normalised_rows(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom > 0)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {key: None for key in ("q05", "q16", "q50", "q84", "q95")}
    q = np.quantile(finite, (0.05, 0.16, 0.50, 0.84, 0.95))
    return {key: float(value) for key, value in zip(("q05", "q16", "q50", "q84", "q95"), q)}


def category_summary(mask: np.ndarray, valid: np.ndarray, match: np.ndarray,
                     best: np.ndarray, second: np.ndarray) -> dict[str, object]:
    count = int(mask.sum())
    valid_mask = mask & valid
    valid_count = int(valid_mask.sum())
    return {
        "events": count,
        "surface_valid": valid_count,
        "surface_valid_fraction": float(valid_count / count) if count else None,
        "truth_sheet_equals_reco_nearest_fraction": (
            float(match[valid_mask].mean()) if valid_count else None
        ),
        "best_rms_angle_mrad": quantiles(best[valid_mask] * 1.0e3),
        "second_minus_best_rms_angle_mrad": quantiles(
            (second[valid_mask] - best[valid_mask]) * 1.0e3
        ),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.module_dir))
    import metspace as ms  # type: ignore
    import sheetspace as ss  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs, allow_pickle=False) as handle:
        required = (
            "labels", "global_indices", "modes", "truth_visible_tau_lab4",
            "truth_neutrino_lab4", "reco_visible_tau_lab4", "reco_met_xy",
        )
        missing = [name for name in required if name not in handle.files]
        if missing:
            raise KeyError(f"Missing required arrays: {missing}")
        arrays = {name: np.asarray(handle[name]) for name in required}

    n_total = len(arrays["global_indices"])
    hashes = splitmix64(arrays["global_indices"], args.seed)
    n_select = min(args.max_events, n_total)
    selected = np.argpartition(hashes, n_select - 1)[:n_select]
    selected = selected[np.argsort(hashes[selected])]

    labels = arrays["labels"][selected].astype(np.int8)
    global_indices = arrays["global_indices"][selected].astype(np.int64)
    modes = arrays["modes"][selected].astype(np.int8)
    truth_visible = arrays["truth_visible_tau_lab4"][selected].astype(np.float64)
    truth_nu = arrays["truth_neutrino_lab4"][selected, ..., :3].astype(np.float64)
    reco_visible = arrays["reco_visible_tau_lab4"][selected].astype(np.float64)
    reco_met = arrays["reco_met_xy"][selected].astype(np.float64)

    truth_label = ss.truth_sheet(truth_visible, truth_nu)
    truth_sheet = np.asarray(truth_label["sheet"], dtype=np.int8)
    truth_root_separation = np.asarray(truth_label["root_separation"], dtype=np.float64)
    truth_tau = truth_visible[..., :3] + truth_nu
    truth_tau /= np.maximum(np.linalg.norm(truth_tau, axis=-1, keepdims=True), 1.0e-300)

    reco_mass = invariant_mass(reco_visible)
    algebraically_possible = (
        np.isfinite(reco_visible).all(axis=(1, 2))
        & np.isfinite(reco_met).all(axis=1)
        & (reco_visible[..., 3] > 0.0).all(axis=1)
        & (reco_mass < TAU_MASS_GEV).all(axis=1)
    )

    n = n_select
    retained = np.zeros(n, dtype=np.int16)
    sheet_present = np.zeros((n, 4), dtype=bool)
    min_rms_angle = np.full((n, 4), np.inf, dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    eligible_indices = np.flatnonzero(algebraically_possible)
    for chunk_start in range(0, len(eligible_indices), args.row_chunk):
        rows = eligible_indices[chunk_start:chunk_start + args.row_chunk]
        vis = reco_visible[rows]
        met = reco_met[rows]
        unit = ms.lattice(args.proposal_lattice, len(rows), rng).transpose(1, 0, 2)
        drawn = ms.sample_region(vis, met, unit)
        picked = ms.compact(drawn["accepted"], args.keep, rng)
        retained[rows] = np.sum(picked >= 0, axis=0).astype(np.int16)
        nu_t = np.take_along_axis(
            drawn["nu_t"], np.maximum(picked, 0)[..., None], axis=0
        )
        solved = ms.solution_sheets(vis, met, nu_t)
        valid = solved["valid"] & (picked >= 0)[:, None, :]
        nu = solved["nu"]
        tau = vis[None, None, ..., :3] + nu
        tau /= np.maximum(np.linalg.norm(tau, axis=-1, keepdims=True), 1.0e-300)
        angle = opening_angle(tau, truth_tau[rows][None, None])
        rms = np.sqrt(np.mean(angle ** 2, axis=-1))
        rms = np.where(valid, rms, np.inf)
        min_rms_angle[rows] = np.min(rms, axis=0).T
        sheet_present[rows] = np.any(valid, axis=0).T

    surface_valid = np.isfinite(min_rms_angle).any(axis=1)
    reco_nearest_sheet = np.full(n, -1, dtype=np.int8)
    reco_nearest_sheet[surface_valid] = np.argmin(
        min_rms_angle[surface_valid], axis=1
    ).astype(np.int8)
    ordered = np.sort(min_rms_angle, axis=1)
    best = ordered[:, 0]
    second = ordered[:, 1]
    match = surface_valid & (reco_nearest_sheet == truth_sheet)

    migration = np.zeros((4, 4), dtype=np.int64)
    np.add.at(migration, (truth_sheet[surface_valid], reco_nearest_sheet[surface_valid]), 1)

    categories: dict[str, object] = {
        "inclusive": category_summary(
            np.ones(n, dtype=bool), surface_valid, match, best, second
        )
    }
    for value, name in ((0, "Z"), (1, "H")):
        categories[f"process_{name}"] = category_summary(
            labels == value, surface_valid, match, best, second
        )
    pair_names = {(0, 0): "pi_pi", (1, 1): "rho_rho", (3, 3): "threeprong_threeprong"}
    for pair, name in pair_names.items():
        subset = (modes[:, 0] == pair[0]) & (modes[:, 1] == pair[1])
        categories[f"mode_{name}"] = category_summary(
            subset, surface_valid, match, best, second
        )

    invalid_mass = (reco_mass >= TAU_MASS_GEV).any(axis=1)
    multiplicity = sheet_present.sum(axis=1)
    report = {
        "contract": {
            "split": "train only",
            "target": "truth-surface global root label",
            "reco_comparator": "sheet containing the reco-surface hypothesis nearest to both truth tau directions",
            "distance": "RMS opening angle over the two tau directions",
            "test_loaded": False,
            "seed": args.seed,
            "max_events": args.max_events,
            "proposal_lattice": args.proposal_lattice,
            "keep": args.keep,
        },
        "counts": {
            "input_events": int(n_total),
            "audit_events": int(n),
            "algebraically_possible": int(algebraically_possible.sum()),
            "surface_valid": int(surface_valid.sum()),
            "invalid_due_to_reco_visible_mass_ge_tau": int(invalid_mass.sum()),
            "retained_draws_quantiles": quantiles(retained.astype(float)),
            "sheet_multiplicity": {
                str(value): int((multiplicity == value).sum()) for value in range(5)
            },
        },
        "truth_sheet_counts": np.bincount(truth_sheet, minlength=4).tolist(),
        "reco_nearest_sheet_counts_valid": np.bincount(
            reco_nearest_sheet[surface_valid], minlength=4
        ).tolist(),
        "migration_counts": migration.tolist(),
        "migration_row_fraction": normalised_rows(migration).tolist(),
        "truth_root_separation_GeV": {
            "side0": quantiles(truth_root_separation[:, 0]),
            "side1": quantiles(truth_root_separation[:, 1]),
            "minimum": quantiles(np.min(truth_root_separation, axis=1)),
        },
        "categories": categories,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        args.output / "audit_events.npz",
        global_indices=global_indices,
        labels=labels,
        modes=modes,
        truth_sheet=truth_sheet,
        reco_nearest_sheet=reco_nearest_sheet,
        surface_valid=surface_valid,
        algebraically_possible=algebraically_possible,
        reco_visible_mass=reco_mass,
        retained=retained,
        sheet_present=sheet_present,
        min_rms_angle=min_rms_angle,
        best_rms_angle=best,
        second_rms_angle=second,
        truth_root_separation=truth_root_separation,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    fraction = normalised_rows(migration)
    image = axes[0, 0].imshow(fraction, vmin=0.0, vmax=max(0.5, float(fraction.max())), cmap="Blues")
    for i in range(4):
        for j in range(4):
            axes[0, 0].text(j, i, f"{fraction[i, j]:.2f}\n({migration[i, j]})",
                            ha="center", va="center", fontsize=9)
    axes[0, 0].set(xlabel="Nearest reco-surface sheet", ylabel="Truth-surface sheet",
                   title="Truth → reco sheet migration (valid surfaces)")
    axes[0, 0].set_xticks(range(4)); axes[0, 0].set_yticks(range(4))
    fig.colorbar(image, ax=axes[0, 0], label="row fraction")

    finite = surface_valid & np.isfinite(second)
    bins = np.geomspace(1.0e-2, 2.0e3, 70)
    axes[0, 1].hist(best[finite] * 1.0e3, bins=bins, histtype="step", lw=2,
                    label="nearest sheet")
    axes[0, 1].hist(second[finite] * 1.0e3, bins=bins, histtype="step", lw=2,
                    label="second-nearest sheet")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel="RMS tau-direction distance [mrad]", ylabel="events",
                   title="Reco-surface match distances")
    axes[0, 1].legend()

    labels_cut = ["all", "$m_{vis}<m_\\tau$", "surface valid"]
    values_cut = [1.0, float(algebraically_possible.mean()), float(surface_valid.mean())]
    axes[1, 0].bar(labels_cut, values_cut, color=("0.55", "#4C78A8", "#59A14F"))
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("fraction of audit cohort")
    axes[1, 0].set_title("Operational reco-surface coverage")
    for i, value in enumerate(values_cut):
        axes[1, 0].text(i, value + 0.02, f"{value:.3f}", ha="center")

    root_min = np.min(truth_root_separation, axis=1)
    valid_root = surface_valid & np.isfinite(root_min)
    edges = np.quantile(root_min[valid_root], np.linspace(0.0, 1.0, 6))
    centres, fidelity, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = valid_root & (root_min >= low) & (root_min <= high)
        centres.append(math.sqrt(max(low, 1.0e-12) * max(high, 1.0e-12)))
        fidelity.append(float(match[in_bin].mean()) if in_bin.any() else np.nan)
        counts.append(int(in_bin.sum()))
    axes[1, 1].plot(centres, fidelity, marker="o", lw=2)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set(xlabel="minimum truth root separation |Δνz| [GeV]",
                   ylabel="same global sheet fraction",
                   title="Label stability versus branch separation")
    for x, y, count in zip(centres, fidelity, counts):
        axes[1, 1].annotate(f"n={count}", (x, y), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("TauSpin WP-A: is a four-class reco sheet target well defined?", fontsize=14)
    fig.savefig(args.output / "label_stability.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(json.dumps(report["categories"]["inclusive"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
