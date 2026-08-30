#!/usr/bin/env python3
"""Plot thesis-defined truth 1p1n x 1p1n cos(theta) maps.

The starting cohort is exactly the final cohort used by
``truth_1p1n_cospsi_population_maps.py``.  Oshihara thesis Equation 3.4 is
then evaluated for tau -> rho nu with x=E_rho/E_tau and a=m_rho/m_tau.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from reco_1p0n_population_maps import (
    SAMPLE_NAMES,
    choose_device,
    infer_split,
    join_reference_scores,
    load_models,
    sha256_file,
    structured_identity,
)
from truth_1p0n_population_maps import find_unique_objects, validate_event_identity
from truth_1p1n_cospsi_population_maps import (
    CLOSURE_TOLERANCE,
    COS_TOLERANCE,
    FIGURE_DPI,
    MAP_RANGE,
    RHO_MASS_GEV,
    ROLE_CHARGED_PION,
    ROLE_NEUTRAL_PION,
    ROLE_TAU_NEUTRINO,
    assign_tertiles,
    attach_fixed_scores,
    class_mask,
    draw,
    histogram2d,
    json_ready,
    population_view,
    select_truth_shard as select_cospsi_truth_shard,
)


TAU_MASS_GEV = 1.77686


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--validation-ensemble", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def thesis_costheta(tau_four: np.ndarray, rho_four: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate thesis Eq. 3.4 for tau -> rho nu in the lab frame."""
    tau_energy = np.asarray(tau_four[..., 3], dtype=np.float64)
    rho_energy = np.asarray(rho_four[..., 3], dtype=np.float64)
    x = rho_energy / tau_energy
    a_squared = (RHO_MASS_GEV / TAU_MASS_GEV) ** 2
    beta_squared = 1.0 - (TAU_MASS_GEV / tau_energy) ** 2
    beta = np.sqrt(beta_squared)
    costheta = (2.0 * x - 1.0 - a_squared) / (beta * (1.0 - a_squared))
    return costheta, x


def validate_thesis_formula() -> float:
    """Validate Eq. 3.4 with an exact two-body tau -> rho nu boost."""
    beta_tau = 0.82
    gamma_tau = 1.0 / np.sqrt(1.0 - beta_tau**2)
    rho_energy_star = (TAU_MASS_GEV**2 + RHO_MASS_GEV**2) / (2.0 * TAU_MASS_GEV)
    rho_momentum_star = (TAU_MASS_GEV**2 - RHO_MASS_GEV**2) / (2.0 * TAU_MASS_GEV)
    expected = np.asarray([-1.0, -0.45, 0.0, 0.3, 1.0], dtype=np.float64)
    tau_four = np.zeros((len(expected), 4), dtype=np.float64)
    tau_four[:, 2] = gamma_tau * beta_tau * TAU_MASS_GEV
    tau_four[:, 3] = gamma_tau * TAU_MASS_GEV
    rho_four = np.zeros_like(tau_four)
    rho_four[:, 0] = rho_momentum_star * np.sqrt(1.0 - expected**2)
    rho_four[:, 2] = gamma_tau * (
        rho_momentum_star * expected + beta_tau * rho_energy_star
    )
    rho_four[:, 3] = gamma_tau * (
        rho_energy_star + beta_tau * rho_momentum_star * expected
    )
    observed, _ = thesis_costheta(tau_four, rho_four)
    error = float(np.max(np.abs(observed - expected)))
    if error > 1.0e-12:
        raise RuntimeError(f"thesis cos(theta) synthetic validation failed: {error}")
    return error


def select_costheta_shard(
    data: Mapping[str, np.ndarray], rows: np.ndarray, path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Start from the exact final cos(psi) cohort, then require valid cos(theta)."""
    cospsi_record, cospsi_audit = select_cospsi_truth_shard(data, rows, path)
    base_rows = np.asarray(cospsi_record["row"], dtype=np.int64)
    base_mask = np.isin(rows, base_rows)
    selected_events = np.flatnonzero(base_mask)
    if not np.array_equal(rows[selected_events], base_rows):
        raise RuntimeError(f"{path.name}: cos(psi) base cohort row ordering changed")

    event_count = len(rows)
    tau_event = np.asarray(data["tau_event_local_index"], dtype=np.int64)
    tau_pdg = np.asarray(data["tau_pdg_id"], dtype=np.int16)
    tau_exact = (
        (np.asarray(data["tau_mode_code"]) == 1)
        & (np.asarray(data["tau_n_charged_pion"]) == 1)
        & (np.asarray(data["tau_n_neutral_pion"]) == 1)
        & (np.asarray(data["tau_n_tau_neutrino"]) == 1)
        & (np.asarray(data["tau_n_other_neutrino"]) == 0)
        & (np.asarray(data["tau_n_photon"]) == 0)
        & (np.asarray(data["tau_object_count"]) == 3)
        & (np.asarray(data["tau_decay_vector_match"]) == 1)
        & (np.abs(tau_pdg) == 15)
    )
    tau_by_event = np.full((event_count, 2), -1, dtype=np.int64)
    exact_indices = np.flatnonzero(tau_exact)
    charge_slot = (tau_pdg[exact_indices] == -15).astype(np.int8)
    tau_by_event[tau_event[exact_indices], charge_slot] = exact_indices
    tau_indices = tau_by_event[selected_events]
    if np.any(tau_indices < 0):
        raise RuntimeError(f"{path.name}: tau charge ordering failed in cos(theta) cohort")

    flat_taus = tau_indices.reshape(-1)
    charged_indices = find_unique_objects(data, flat_taus, ROLE_CHARGED_PION).reshape(-1, 2)
    neutral_indices = find_unique_objects(data, flat_taus, ROLE_NEUTRAL_PION).reshape(-1, 2)
    neutrino_indices = find_unique_objects(data, flat_taus, ROLE_TAU_NEUTRINO).reshape(-1, 2)
    charged_pdg = np.asarray(data["object_pdg_id"])[charged_indices]
    neutral_pdg = np.asarray(data["object_pdg_id"])[neutral_indices]
    neutrino_pdg = np.asarray(data["object_pdg_id"])[neutrino_indices]
    expected_charged = np.tile(np.asarray([-211, 211], dtype=np.int16), (len(selected_events), 1))
    expected_neutrino = np.tile(np.asarray([16, -16], dtype=np.int16), (len(selected_events), 1))
    if not np.array_equal(charged_pdg, expected_charged):
        raise RuntimeError(f"{path.name}: charged-pion daughter does not match tau charge")
    if np.any(neutral_pdg != 111):
        raise RuntimeError(f"{path.name}: neutral-pion daughter PDG identity failed")
    if not np.array_equal(neutrino_pdg, expected_neutrino):
        raise RuntimeError(f"{path.name}: tau-neutrino daughter does not match tau charge")

    tau_four = np.asarray(data["tau_px_py_pz_e"], dtype=np.float64)[tau_indices]
    charged_four = np.asarray(data["object_px_py_pz_e"], dtype=np.float64)[charged_indices]
    neutral_four = np.asarray(data["object_px_py_pz_e"], dtype=np.float64)[neutral_indices]
    neutrino_four = np.asarray(data["object_px_py_pz_e"], dtype=np.float64)[neutrino_indices]
    rho_four = charged_four + neutral_four
    closure_scale = np.maximum(tau_four[..., 3], 1.0)
    closure = np.max(np.abs(tau_four - rho_four - neutrino_four), axis=2) / closure_scale
    if np.any(~np.isfinite(closure)) or np.any(closure > CLOSURE_TOLERANCE):
        raise RuntimeError(f"{path.name}: cos(psi) base cohort lost tau -> rho nu closure")

    tau_energy = tau_four[..., 3]
    valid_energy = np.all(
        np.isfinite(tau_energy)
        & np.isfinite(rho_four[..., 3])
        & (tau_energy > TAU_MASS_GEV),
        axis=1,
    )
    raw_costheta = np.full((len(selected_events), 2), np.nan, dtype=np.float64)
    x = np.full_like(raw_costheta, np.nan)
    if np.any(valid_energy):
        raw_costheta[valid_energy], x[valid_energy] = thesis_costheta(
            tau_four[valid_energy], rho_four[valid_energy]
        )
    finite_event = np.all(np.isfinite(raw_costheta), axis=1)
    range_excess = np.maximum(np.abs(raw_costheta) - 1.0, 0.0)
    physical_event = finite_event & np.all(range_excess <= COS_TOLERANCE, axis=1)
    raw_selected_costheta = raw_costheta[physical_event]
    x_selected = x[physical_event]
    beta = np.sqrt(1.0 - (TAU_MASS_GEV / tau_energy[physical_event]) ** 2)
    a_squared = (RHO_MASS_GEV / TAU_MASS_GEV) ** 2
    x_back = 0.5 * (
        1.0 + a_squared + beta * (1.0 - a_squared) * raw_selected_costheta
    )
    inverse_error = np.abs(x_back - x_selected)
    costheta = np.clip(raw_selected_costheta, -1.0, 1.0)
    if np.any(~np.isfinite(costheta)) or np.any(np.abs(costheta) > 1.0):
        raise RuntimeError(f"{path.name}: final cos(theta) validity failed")
    if np.max(inverse_error, initial=0.0) > 1.0e-12:
        raise RuntimeError(f"{path.name}: x/cos(theta) inverse consistency failed")

    final_events = selected_events[physical_event]
    record = {
        "row": rows[final_events],
        "split_id": np.asarray(data["event_split_id"], dtype=np.uint8)[final_events],
        "sample_id": np.asarray(data["event_sample_id"], dtype=np.uint8)[final_events],
        "costheta": costheta,
    }
    audit = {
        "cospsi_base_audit": cospsi_audit,
        "cospsi_base_events": int(len(selected_events)),
        "failed_costheta_finite_or_energy": int(len(selected_events) - np.sum(finite_event)),
        "failed_costheta_physical_range": int(np.sum(finite_event) - len(final_events)),
        "selected_costheta_events": int(len(final_events)),
        "max_relative_tau_rho_nu_closure": float(np.max(closure, initial=0.0)),
        "max_finite_costheta_range_excess": float(
            np.nanmax(range_excess[finite_event], initial=0.0)
        ),
        "max_x_inverse_error": float(np.max(inverse_error, initial=0.0)),
    }
    return record, audit


def load_costheta_cohort(
    truth_dir: Path, row_map: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = sorted(truth_dir.glob("[HZ]-*.npz"))
    if not paths:
        raise FileNotFoundError(f"no truth-object shards found in {truth_dir}")
    seen_rows = np.zeros(len(row_map["sample_id"]), dtype=bool)
    records: list[dict[str, np.ndarray]] = []
    audits: dict[str, Any] = {}
    role_contract = None
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            data = {key: np.asarray(source[key]) for key in source.files}
        contract = (tuple(data["role_names"].tolist()), tuple(data["role_values"].tolist()))
        if role_contract is None:
            role_contract = contract
        elif role_contract != contract:
            raise RuntimeError(f"{path.name}: truth role schema differs")
        rows = validate_event_identity(data, row_map, seen_rows, path)
        record, audit = select_costheta_shard(data, rows, path)
        records.append(record)
        audits[path.name] = audit
    if not np.all(seen_rows):
        raise RuntimeError(f"truth shards miss {int(np.sum(~seen_rows))} canonical rows")
    cohort = {key: np.concatenate([record[key] for record in records]) for key in records[0]}
    if len(np.unique(cohort["row"])) != len(cohort["row"]):
        raise RuntimeError("truth cos(theta) cohort rows are not unique")
    return cohort, {
        "shard_count": len(paths),
        "canonical_rows_seen_once": int(np.sum(seen_rows)),
        "shards": audits,
    }


def decorate(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.3)
    axis.set_xlabel(r"truth $\cos\theta_{-}$")
    axis.set_ylabel(r"truth $\cos\theta_{+}$")
    axis.set_xlim(MAP_RANGE)
    axis.set_ylim(MAP_RANGE)
    axis.set_aspect("equal")


def footer(population: str, bins: int | None) -> str:
    suffix = f" • {bins}x{bins} bins" if bins is not None else ""
    return (
        f"H/Z MC samples • {population} • exact truth 1p1n x 1p1n • "
        f"cos(psi)-matched cohort • fixed 3-seed classifier • unit weight • nominal{suffix}"
    )


def save_png(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_inclusive(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    counts: dict[int, np.ndarray] = {}
    densities: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["costheta"][mask, 0], data["costheta"][mask, 1], bins)
        if count.sum() == 0:
            raise RuntimeError(f"{population}: empty {SAMPLE_NAMES[label]} class")
        density = count / count.sum()
        if not np.isclose(density.sum(), 1.0):
            raise RuntimeError(f"{population}: inclusive normalization failed")
        counts[label], densities[label] = count, density
    vmax = max(float(np.max(value)) for value in densities.values())
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        image = draw(axis, densities[label], bins, "viridis", Normalize(0.0, vmax))
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={int(counts[label].sum())}")
    assert image is not None
    figure.colorbar(image, ax=axes, label="class-normalized bin probability")
    figure.suptitle(r"Truth $1p1n \times 1p1n$: $(\cos\theta_{-},\,\cos\theta_{+})$", fontsize=13)
    figure.text(0.5, -0.015, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            "events": int(counts[label].sum()),
            "empty_bins": int(np.sum(counts[label] == 0)),
        }
        for label in (1, 0)
    }


def plot_tertiles(
    data: Mapping[str, np.ndarray], population: str, bins: int, levels: np.ndarray, path: Path
) -> dict[str, Any]:
    counts: dict[int, dict[int, np.ndarray]] = {1: {}, 0: {}}
    densities: list[np.ndarray] = []
    for label in (1, 0):
        for level in range(3):
            mask = class_mask(data, label) & (levels == level)
            count = histogram2d(data["costheta"][mask, 0], data["costheta"][mask, 1], bins)
            if count.sum() == 0:
                raise RuntimeError(f"{population}: empty {SAMPLE_NAMES[label]} tertile {level}")
            density = count / count.sum()
            if not np.isclose(density.sum(), 1.0):
                raise RuntimeError(f"{population}: tertile normalization failed")
            counts[label][level] = count
            densities.append(density)
    vmax = max(float(np.max(value)) for value in densities)
    figure, axes = plt.subplots(2, 3, figsize=(11.3, 7.1), constrained_layout=True, sharex=True, sharey=True)
    names = ("low", "middle", "high")
    image = None
    for row, label in enumerate((1, 0)):
        for level in range(3):
            density = counts[label][level] / counts[label][level].sum()
            image = draw(axes[row, level], density, bins, "viridis", Normalize(0.0, vmax))
            decorate(
                axes[row, level],
                f"{SAMPLE_NAMES[label]} • {names[level]} within-class score • N={int(counts[label][level].sum())}",
            )
    assert image is not None
    figure.colorbar(image, ax=axes, label="panel-normalized bin probability")
    figure.suptitle("Fixed classifier score tertiles within each truth class", fontsize=13)
    figure.text(0.5, -0.01, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            str(level): {
                "events": int(counts[label][level].sum()),
                "empty_bins": int(np.sum(counts[label][level] == 0)),
            }
            for level in range(3)
        }
        for label in (1, 0)
    }


def plot_mean_score(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    counts: dict[int, np.ndarray] = {}
    means: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["costheta"][mask, 0], data["costheta"][mask, 1], bins)
        summed = histogram2d(
            data["costheta"][mask, 0],
            data["costheta"][mask, 1],
            bins,
            weights=data["scores"][mask],
        )
        mean = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
        if not np.isfinite(mean).all():
            raise RuntimeError(f"{population}: mean score contains NaN or infinity")
        counts[label], means[label] = count, mean
    occupied = [means[label][counts[label] > 0] for label in (1, 0)]
    vmin = min(float(np.min(value)) for value in occupied)
    vmax = max(float(np.max(value)) for value in occupied)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#d9d9d9")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        visible = np.ma.masked_where(counts[label] == 0, means[label])
        image = draw(axis, visible, bins, cmap, Normalize(vmin, vmax))
        occupied_counts = counts[label][counts[label] > 0]
        decorate(
            axis,
            f"{SAMPLE_NAMES[label]} truth class • occupied-bin N={int(occupied_counts.min())}–{int(occupied_counts.max())} • gray=empty",
        )
    assert image is not None
    figure.colorbar(image, ax=axes, label="mean fixed-classifier H-like score")
    figure.suptitle(
        r"Mean fixed-classifier score per truth $(\cos\theta_{-},\cos\theta_{+})$ bin",
        fontsize=13,
    )
    figure.text(0.5, -0.015, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            "empty_bins": int(np.sum(counts[label] == 0)),
            "nan_bins": int(np.sum(~np.isfinite(means[label]))),
            "occupied_mean_score_min": float(np.min(means[label][counts[label] > 0])),
            "occupied_mean_score_max": float(np.max(means[label][counts[label] > 0])),
        }
        for label in (1, 0)
    }


def plot_scatter(data: Mapping[str, np.ndarray], population: str, path: Path) -> dict[str, int]:
    colors = {1: "#D55E00", 0: "#0072B2"}
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True, sharex=True, sharey=True)
    result: dict[str, int] = {}
    for axis, label in zip(axes, (1, 0), strict=True):
        mask = class_mask(data, label)
        values = data["costheta"][mask]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{population}: scatter contains NaN or infinity")
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=4,
            alpha=0.10,
            linewidths=0,
            color=colors[label],
            rasterized=True,
            label=f"{SAMPLE_NAMES[label]} truth class",
        )
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={len(values)}")
        axis.legend(loc="upper right", frameon=True, fontsize=8, markerscale=2.5)
        result[SAMPLE_NAMES[label]] = int(len(values))
    figure.suptitle(
        r"Truth $1p1n \times 1p1n$ $(\cos\theta_{-},\cos\theta_{+})$ scatter",
        fontsize=13,
    )
    figure.text(0.5, -0.015, footer(population, None), ha="center", fontsize=7.8)
    save_png(figure, path)
    return result


def main() -> int:
    args = parse_args()
    if args.bins != 20:
        raise ValueError("this task requires exactly 20x20 binning")
    formula_error = validate_thesis_formula()
    repo = args.repo.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with np.load(args.row_map, allow_pickle=False) as source:
        row_map = {key: np.asarray(source[key]) for key in source.files}
    cohort, truth_audit = load_costheta_cohort(args.truth_dir.resolve(), row_map)

    device = choose_device(args.device)
    models, checkpoint_hashes = load_models(repo, processed_dir, args.checkpoint_root.resolve(), device)
    inference: dict[str, dict[str, np.ndarray]] = {}
    validation_parity = None
    for split in ("train", "validation", "test"):
        current = infer_split(processed_dir, split, row_map, models, device, args.batch_size)
        inference[split] = current
        if split == "validation":
            identity = structured_identity(
                {
                    key: row_map[key][current["split_mask"]]
                    for key in ("sample_id", "ntuple_file_index", "ntuple_entry")
                }
            )
            with np.load(args.validation_ensemble, allow_pickle=False) as source:
                reference = {key: np.asarray(source[key]) for key in source.files}
            expected_scores = join_reference_scores(identity, reference)
            absolute = np.abs(current["scores"] - expected_scores)
            validation_parity = {
                "max_abs_difference": float(np.max(absolute)),
                "mean_abs_difference": float(np.mean(absolute)),
                "allclose_atol_5e-4": bool(
                    np.allclose(current["scores"], expected_scores, rtol=0.0, atol=5.0e-4)
                ),
            }
            if not validation_parity["allclose_atol_5e-4"]:
                raise RuntimeError(f"frozen validation score parity failed: {validation_parity}")
    cohort["scores"] = attach_fixed_scores(cohort, row_map, inference)
    del models, inference
    gc.collect()

    populations = {
        "all-data": population_view(cohort, (0, 1, 2)),
        "validation-test": population_view(cohort, (1, 2)),
    }
    labels = {
        "all-data": "all data (train + validation + test)",
        "validation-test": "validation + test",
    }
    manifest: dict[str, Any] = {
        "status": "completed",
        "bins_per_axis": args.bins,
        "device": str(device),
        "checkpoint_sha256": checkpoint_hashes,
        "validation_reference_sha256": sha256_file(args.validation_ensemble),
        "validation_inference_parity": validation_parity,
        "truth_audit": truth_audit,
        "formula_synthetic_max_abs_error": formula_error,
        "definitions": {
            "population": "exact final cohort from the preceding truth 1p1n x 1p1n cos(psi) maps, followed only by cos(theta) finite and physical-range validity",
            "truth_selection": "clean boson tau pair; charge-ordered PDG 15 (tau-) and -15 (tau+); each tau has exactly one charged pion, one neutral pion, and one tau neutrino; tau -> rho nu closure <=1e-4",
            "costheta": "Oshihara thesis Eq. 3.4 for 1pn: x=E_rho/E_tau, a=m_rho/m_tau, beta=sqrt(1-m_tau^2/E_tau^2), cos(theta)=(2x-1-a^2)/(beta*(1-a^2)); m_rho=0.778 GeV, m_tau=1.77686 GeV",
            "axis": "x=tau- cos(theta), y=tau+ cos(theta); rho four-vector is the charge-matched charged pion plus PDG-111 neutral pion",
            "excluded_observables": "cos(psi) is used only to reproduce the preceding final event cohort; no cos(theta) x cos(psi), A_pair, or other polarimeter is computed",
            "score": "arithmetic mean of frozen full-reco seeds 42, 43, and 44 terminal checkpoints; eval-only inference; H-like score",
            "tertiles": "unit-weight score tertiles defined separately within each truth class and each population after final cos(theta) validity",
            "normalization": "inclusive and tertile panels normalized independently to unit sum; mean maps show raw mean H-like score; scatter contains every final selected event",
        },
        "populations": {},
    }
    for key, data in populations.items():
        prefix = f"truth-1p1n-costheta-{key}"
        levels, edges = assign_tertiles(data)
        inclusive = plot_inclusive(data, labels[key], args.bins, output_dir / f"{prefix}-inclusive.png")
        tertiles = plot_tertiles(
            data,
            labels[key],
            args.bins,
            levels,
            output_dir / f"{prefix}-score-tertiles.png",
        )
        mean_score = plot_mean_score(
            data, labels[key], args.bins, output_dir / f"{prefix}-mean-score.png"
        )
        scatter = plot_scatter(data, labels[key], output_dir / f"{prefix}-scatter.png")
        manifest["populations"][key] = {
            "events": int(len(data["row"])),
            "class_counts": {
                SAMPLE_NAMES[label]: int(np.sum(class_mask(data, label))) for label in (1, 0)
            },
            "tertile_edges": {SAMPLE_NAMES[label]: edges[label] for label in (1, 0)},
            "costheta_min_max_tau_minus": [
                float(np.min(data["costheta"][:, 0])),
                float(np.max(data["costheta"][:, 0])),
            ],
            "costheta_min_max_tau_plus": [
                float(np.min(data["costheta"][:, 1])),
                float(np.max(data["costheta"][:, 1])),
            ],
            "inclusive": inclusive,
            "tertiles": tertiles,
            "mean_score": mean_score,
            "scatter": scatter,
            "png": [
                f"{prefix}-inclusive.png",
                f"{prefix}-score-tertiles.png",
                f"{prefix}-mean-score.png",
                f"{prefix}-scatter.png",
            ],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "truth_1p1n_counts": {
                    key: manifest["populations"][key]["class_counts"] for key in populations
                },
                "validation_inference_parity": validation_parity,
                "png_count": sum(
                    len(manifest["populations"][key]["png"]) for key in populations
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
