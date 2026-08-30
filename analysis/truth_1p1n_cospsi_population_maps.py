#!/usr/bin/env python3
"""Plot thesis-defined truth 1p1n x 1p1n cos(psi) maps.

The truth cohort is selected independently from reconstruction.  The observable
is Equation 3.5 of the Oshihara thesis and is not replaced by cos(theta),
``A_pair``, or another polarimeter.
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
from reco_1p1p_score_maps import stable_tertiles
from truth_1p0n_population_maps import find_unique_objects, validate_event_identity


MAP_RANGE = (-1.0, 1.0)
FIGURE_DPI = 240
RHO_MASS_GEV = 0.778
CHARGED_PION_MASS_GEV = 0.13957039
ROLE_CHARGED_PION = 1
ROLE_NEUTRAL_PION = 2
ROLE_TAU_NEUTRINO = 3
CLOSURE_TOLERANCE = 1.0e-4
COS_TOLERANCE = 5.0e-5


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


def thesis_cospsi(charged_four: np.ndarray, neutral_four: np.ndarray) -> np.ndarray:
    """Evaluate Oshihara thesis Eq. 3.5 from lab-frame pion four-vectors."""
    denominator_momentum = np.linalg.norm(charged_four[..., :3] + neutral_four[..., :3], axis=-1)
    mass_factor_squared = RHO_MASS_GEV**2 - 4.0 * CHARGED_PION_MASS_GEV**2
    if mass_factor_squared <= 0.0:
        raise RuntimeError("invalid fixed masses in thesis cos(psi) definition")
    factor = RHO_MASS_GEV / np.sqrt(mass_factor_squared)
    return factor * (charged_four[..., 3] - neutral_four[..., 3]) / denominator_momentum


def validate_thesis_formula() -> float:
    """Check Eq. 3.5 against its defining rho-rest-frame angle."""
    p_star = np.sqrt(RHO_MASS_GEV**2 / 4.0 - CHARGED_PION_MASS_GEV**2)
    energy_star = RHO_MASS_GEV / 2.0
    beta = 0.8
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    expected = np.asarray([-1.0, -0.4, 0.0, 0.35, 1.0], dtype=np.float64)
    charged = np.zeros((len(expected), 4), dtype=np.float64)
    neutral = np.zeros_like(charged)
    charged[:, 0] = p_star * np.sqrt(1.0 - expected**2)
    charged[:, 2] = gamma * (p_star * expected + beta * energy_star)
    charged[:, 3] = gamma * (energy_star + beta * p_star * expected)
    neutral[:, 0] = -charged[:, 0]
    neutral[:, 2] = gamma * (-p_star * expected + beta * energy_star)
    neutral[:, 3] = gamma * (energy_star - beta * p_star * expected)
    observed = thesis_cospsi(charged, neutral)
    error = float(np.max(np.abs(observed - expected)))
    if error > 1.0e-12:
        raise RuntimeError(f"thesis cos(psi) synthetic rest-frame validation failed: {error}")
    return error


def select_truth_shard(
    data: Mapping[str, np.ndarray], rows: np.ndarray, path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
    exact_count = np.bincount(tau_event, weights=tau_exact.astype(np.int8), minlength=event_count)
    minus_count = np.bincount(
        tau_event, weights=(tau_exact & (tau_pdg == 15)).astype(np.int8), minlength=event_count
    )
    plus_count = np.bincount(
        tau_event, weights=(tau_exact & (tau_pdg == -15)).astype(np.int8), minlength=event_count
    )
    structural = (
        (np.asarray(data["event_clean_two_tau"]) == 1)
        & (np.asarray(data["event_boson_tau_count"]) == 2)
        & (exact_count == 2)
        & (minus_count == 1)
        & (plus_count == 1)
    )
    selected_events = np.flatnonzero(structural)
    tau_by_event = np.full((event_count, 2), -1, dtype=np.int64)
    exact_indices = np.flatnonzero(tau_exact)
    charge_slot = (tau_pdg[exact_indices] == -15).astype(np.int8)
    tau_by_event[tau_event[exact_indices], charge_slot] = exact_indices
    tau_indices = tau_by_event[selected_events]
    if np.any(tau_indices < 0):
        raise RuntimeError(f"{path.name}: tau charge ordering failed")

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
    closure_scale = np.maximum(tau_four[..., 3], 1.0)
    closure = np.max(
        np.abs(tau_four - charged_four - neutral_four - neutrino_four), axis=2
    ) / closure_scale
    if np.any(~np.isfinite(closure)):
        raise RuntimeError(f"{path.name}: non-finite truth daughter closure")
    closure_candidate_max = float(np.max(closure, initial=0.0))
    closure_event = np.all(closure <= CLOSURE_TOLERANCE, axis=1)
    structural_count = int(len(selected_events))
    selected_events = selected_events[closure_event]
    tau_indices = tau_indices[closure_event]
    charged_four = charged_four[closure_event]
    neutral_four = neutral_four[closure_event]
    closure = closure[closure_event]

    rho_momentum = np.linalg.norm(charged_four[..., :3] + neutral_four[..., :3], axis=2)
    valid_denominator = np.all(np.isfinite(rho_momentum) & (rho_momentum > 0.0), axis=1)
    raw_cospsi = np.full_like(rho_momentum, np.nan, dtype=np.float64)
    if np.any(valid_denominator):
        raw_cospsi[valid_denominator] = thesis_cospsi(
            charged_four[valid_denominator], neutral_four[valid_denominator]
        )
    finite_event = np.all(np.isfinite(raw_cospsi), axis=1)
    range_excess = np.maximum(np.abs(raw_cospsi) - 1.0, 0.0)
    physical_event = finite_event & np.all(range_excess <= COS_TOLERANCE, axis=1)
    finite_count = int(np.sum(finite_event))
    closure_selected_count = int(len(selected_events))
    selected_events = selected_events[physical_event]
    tau_indices = tau_indices[physical_event]
    cospsi = np.clip(raw_cospsi[physical_event], -1.0, 1.0)
    if np.any(~np.isfinite(cospsi)) or np.any(np.abs(cospsi) > 1.0):
        raise RuntimeError(f"{path.name}: final cos(psi) validity failed")

    excluded = np.asarray(data["tau_excluded_simulation_object_count"])[tau_indices]
    record = {
        "row": rows[selected_events],
        "split_id": np.asarray(data["event_split_id"], dtype=np.uint8)[selected_events],
        "sample_id": np.asarray(data["event_sample_id"], dtype=np.uint8)[selected_events],
        "cospsi": cospsi,
    }
    audit = {
        "events": event_count,
        "structural_truth_1p1n_x_1p1n_candidates": structural_count,
        "failed_three_daughter_four_vector_closure": int(structural_count - closure_selected_count),
        "finite_cospsi_events_after_closure": finite_count,
        "failed_cospsi_finite_or_denominator": int(closure_selected_count - finite_count),
        "failed_cospsi_physical_range": int(finite_count - len(selected_events)),
        "selected_truth_1p1n_x_1p1n": int(len(selected_events)),
        "selected_taus_with_excluded_simulation_objects": int(np.sum(excluded > 0)),
        "max_relative_three_daughter_closure_before_selection": closure_candidate_max,
        "max_relative_three_daughter_closure_selected": float(np.max(closure, initial=0.0)),
        "max_finite_cospsi_range_excess": float(np.nanmax(range_excess[finite_event], initial=0.0)),
    }
    return record, audit


def load_truth_cohort(
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
        record, audit = select_truth_shard(data, rows, path)
        records.append(record)
        audits[path.name] = audit
    if not np.all(seen_rows):
        raise RuntimeError(f"truth shards miss {int(np.sum(~seen_rows))} canonical rows")
    cohort = {key: np.concatenate([record[key] for record in records]) for key in records[0]}
    if len(np.unique(cohort["row"])) != len(cohort["row"]):
        raise RuntimeError("truth cohort rows are not unique")
    return cohort, {
        "shard_count": len(paths),
        "canonical_rows_seen_once": int(np.sum(seen_rows)),
        "shards": audits,
    }


def attach_fixed_scores(
    cohort: dict[str, np.ndarray], row_map: Mapping[str, np.ndarray],
    split_inference: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    global_scores = np.full(len(row_map["sample_id"]), np.nan, dtype=np.float64)
    for split, inference in split_inference.items():
        mask = np.asarray(inference["split_mask"], dtype=bool)
        if np.any(np.isfinite(global_scores[mask])):
            raise RuntimeError(f"{split}: score rows overlap another split")
        global_scores[mask] = np.asarray(inference["scores"], dtype=np.float64)
    if not np.isfinite(global_scores).all():
        raise RuntimeError("fixed classifier scores do not cover canonical rows")
    rows = cohort["row"]
    if not np.array_equal(cohort["split_id"], np.asarray(row_map["split_id"])[rows]):
        raise RuntimeError("truth split identity changed before score join")
    if not np.array_equal(cohort["sample_id"], np.asarray(row_map["sample_id"])[rows]):
        raise RuntimeError("truth sample identity changed before score join")
    scores = global_scores[rows]
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError("joined fixed classifier score is invalid")
    return scores


def class_mask(data: Mapping[str, np.ndarray], label: int) -> np.ndarray:
    return np.asarray(data["sample_id"]) == (0 if label == 1 else 1)


def histogram2d(x: np.ndarray, y: np.ndarray, bins: int, weights: np.ndarray | None = None) -> np.ndarray:
    values, _, _ = np.histogram2d(x, y, bins=bins, range=(MAP_RANGE, MAP_RANGE), weights=weights)
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("histogram contains NaN or infinity")
    return values


def draw(axis: plt.Axes, values: np.ndarray, bins: int, cmap: Any, norm: Normalize) -> Any:
    edges = np.linspace(MAP_RANGE[0], MAP_RANGE[1], bins + 1)
    return axis.pcolormesh(edges, edges, values.T, cmap=cmap, norm=norm, shading="flat")


def decorate(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.3)
    axis.set_xlabel(r"truth $\cos\psi_{-}$")
    axis.set_ylabel(r"truth $\cos\psi_{+}$")
    axis.set_xlim(MAP_RANGE)
    axis.set_ylim(MAP_RANGE)
    axis.set_aspect("equal")


def footer(population: str, bins: int | None) -> str:
    suffix = f" • {bins}x{bins} bins" if bins is not None else ""
    return (
        f"H/Z MC samples • {population} • exact truth 1p1n x 1p1n • "
        f"fixed 3-seed classifier • unit weight • nominal{suffix}"
    )


def save_png(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_inclusive(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    counts: dict[int, np.ndarray] = {}
    densities: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["cospsi"][mask, 0], data["cospsi"][mask, 1], bins)
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
    figure.suptitle(r"Truth $1p1n \times 1p1n$: $(\cos\psi_{-},\,\cos\psi_{+})$", fontsize=13)
    figure.text(0.5, -0.015, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {"events": int(counts[label].sum()), "empty_bins": int(np.sum(counts[label] == 0))}
        for label in (1, 0)
    }


def assign_tertiles(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    levels = np.full(len(data["scores"]), -1, dtype=np.int8)
    edges: dict[int, tuple[float, float]] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        class_levels, class_edges = stable_tertiles(data["scores"][mask])
        levels[mask] = class_levels
        edges[label] = class_edges
    if np.any(levels < 0):
        raise RuntimeError("not every event received a score tertile")
    return levels, edges


def plot_tertiles(
    data: Mapping[str, np.ndarray], population: str, bins: int, levels: np.ndarray, path: Path
) -> dict[str, Any]:
    counts: dict[int, dict[int, np.ndarray]] = {1: {}, 0: {}}
    densities: list[np.ndarray] = []
    for label in (1, 0):
        for level in range(3):
            mask = class_mask(data, label) & (levels == level)
            count = histogram2d(data["cospsi"][mask, 0], data["cospsi"][mask, 1], bins)
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
            str(level): {"events": int(counts[label][level].sum()), "empty_bins": int(np.sum(counts[label][level] == 0))}
            for level in range(3)
        }
        for label in (1, 0)
    }


def plot_mean_score(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    counts: dict[int, np.ndarray] = {}
    means: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["cospsi"][mask, 0], data["cospsi"][mask, 1], bins)
        summed = histogram2d(
            data["cospsi"][mask, 0], data["cospsi"][mask, 1], bins, weights=data["scores"][mask]
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
    figure.suptitle(r"Mean fixed-classifier score per truth $(\cos\psi_{-},\cos\psi_{+})$ bin", fontsize=13)
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
        values = data["cospsi"][mask]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{population}: scatter contains NaN or infinity")
        axis.scatter(
            values[:, 0], values[:, 1], s=4, alpha=0.10, linewidths=0,
            color=colors[label], rasterized=True, label=f"{SAMPLE_NAMES[label]} truth class",
        )
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={len(values)}")
        axis.legend(loc="upper right", frameon=True, fontsize=8, markerscale=2.5)
        result[SAMPLE_NAMES[label]] = int(len(values))
    figure.suptitle(r"Truth $1p1n \times 1p1n$ $(\cos\psi_{-},\cos\psi_{+})$ scatter", fontsize=13)
    figure.text(0.5, -0.015, footer(population, None), ha="center", fontsize=7.8)
    save_png(figure, path)
    return result


def population_view(cohort: Mapping[str, np.ndarray], split_ids: tuple[int, ...]) -> dict[str, np.ndarray]:
    mask = np.isin(cohort["split_id"], split_ids)
    return {key: np.asarray(value)[mask] for key, value in cohort.items()}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


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
    cohort, truth_audit = load_truth_cohort(args.truth_dir.resolve(), row_map)

    device = choose_device(args.device)
    models, checkpoint_hashes = load_models(repo, processed_dir, args.checkpoint_root.resolve(), device)
    inference: dict[str, dict[str, np.ndarray]] = {}
    validation_parity = None
    for split in ("train", "validation", "test"):
        current = infer_split(processed_dir, split, row_map, models, device, args.batch_size)
        inference[split] = current
        if split == "validation":
            identity = structured_identity(
                {key: row_map[key][current["split_mask"]] for key in ("sample_id", "ntuple_file_index", "ntuple_entry")}
            )
            with np.load(args.validation_ensemble, allow_pickle=False) as source:
                reference = {key: np.asarray(source[key]) for key in source.files}
            expected_scores = join_reference_scores(identity, reference)
            absolute = np.abs(current["scores"] - expected_scores)
            validation_parity = {
                "max_abs_difference": float(np.max(absolute)),
                "mean_abs_difference": float(np.mean(absolute)),
                "allclose_atol_5e-4": bool(np.allclose(current["scores"], expected_scores, rtol=0.0, atol=5.0e-4)),
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
            "population": "same canonical sample/split surface and fixed classifier scores as the preceding truth pi-nu maps; truth 1p1n x 1p1n is independently selected without reco intersection",
            "truth_selection": "clean boson tau pair; charge-ordered PDG 15 (tau-) and -15 (tau+); each tau has exactly one generator-level charged pion, one neutral pion, and one tau neutrino, no other preserved decay objects, and three-daughter four-vector closure <=1e-4",
            "cospsi": "Oshihara thesis Eq. 3.5 evaluated in the lab frame: [m_rho/sqrt(m_rho^2-4*m_pi^2)] * [(E_charged-E_pi0)/|p_charged+p_pi0|], with m_rho=0.778 GeV and m_pi=0.13957039 GeV",
            "axis": "x=tau- cos(psi), y=tau+ cos(psi); charged pion is charge matched and neutral pion has PDG 111",
            "excluded_observables": "cos(theta), A_pair, and other polarimeters are not computed",
            "score": "arithmetic mean of frozen full-reco seeds 42, 43, and 44 terminal checkpoints; eval-only inference; H-like score",
            "tertiles": "unit-weight score tertiles defined separately within each truth class and each population after cos(psi) physical validity",
            "normalization": "inclusive and tertile panels normalized independently to unit sum; mean maps show raw mean H-like score; scatter contains every final selected event",
        },
        "populations": {},
    }
    for key, data in populations.items():
        prefix = f"truth-1p1n-cospsi-{key}"
        levels, edges = assign_tertiles(data)
        inclusive = plot_inclusive(data, labels[key], args.bins, output_dir / f"{prefix}-inclusive.png")
        tertiles = plot_tertiles(data, labels[key], args.bins, levels, output_dir / f"{prefix}-score-tertiles.png")
        mean_score = plot_mean_score(data, labels[key], args.bins, output_dir / f"{prefix}-mean-score.png")
        scatter = plot_scatter(data, labels[key], output_dir / f"{prefix}-scatter.png")
        manifest["populations"][key] = {
            "events": int(len(data["row"])),
            "class_counts": {SAMPLE_NAMES[label]: int(np.sum(class_mask(data, label))) for label in (1, 0)},
            "tertile_edges": {SAMPLE_NAMES[label]: edges[label] for label in (1, 0)},
            "cospsi_min_max_tau_minus": [float(np.min(data["cospsi"][:, 0])), float(np.max(data["cospsi"][:, 0]))],
            "cospsi_min_max_tau_plus": [float(np.min(data["cospsi"][:, 1])), float(np.max(data["cospsi"][:, 1]))],
            "inclusive": inclusive,
            "tertiles": tertiles,
            "mean_score": mean_score,
            "scatter": scatter,
            "png": [
                f"{prefix}-inclusive.png", f"{prefix}-score-tertiles.png",
                f"{prefix}-mean-score.png", f"{prefix}-scatter.png",
            ],
        }
    (output_dir / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "truth_1p1n_counts": {key: manifest["populations"][key]["class_counts"] for key in populations},
        "validation_inference_parity": validation_parity,
        "png_count": sum(len(manifest["populations"][key]["png"]) for key in populations),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
