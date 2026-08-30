#!/usr/bin/env python3
"""Plot truth pi-nu x pi-nu angular maps with the frozen reco classifier.

The truth cohort is selected independently from reconstruction.  The angular
observable follows the thesis energy-fraction definition exactly and does not
use the existing invariant ``A_pair`` observable.
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
    SPLIT_IDS,
    choose_device,
    infer_split,
    join_reference_scores,
    load_models,
    sha256_file,
    structured_identity,
)
from reco_1p1p_score_maps import stable_tertiles


MAP_RANGE = (-1.0, 1.0)
FIGURE_DPI = 240
TAU_MASS_GEV = 1.77686
CHARGED_PION_MASS_GEV = 0.13957039
ROLE_CHARGED_PION = 1
ROLE_TAU_NEUTRINO = 3
COS_TOLERANCE = 5.0e-5
CLOSURE_TOLERANCE = 1.0e-4


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


def find_unique_objects(
    data: Mapping[str, np.ndarray], tau_indices: np.ndarray, role: int
) -> np.ndarray:
    object_mask = np.asarray(data["object_role"]) == role
    object_keys = (
        np.asarray(data["object_event_local_index"])[object_mask].astype(np.int64) * 256
        + np.asarray(data["object_boson_tau_slot"])[object_mask].astype(np.int64)
    )
    object_indices = np.flatnonzero(object_mask)
    order = np.argsort(object_keys, kind="stable")
    object_keys = object_keys[order]
    object_indices = object_indices[order]
    tau_keys = (
        np.asarray(data["tau_event_local_index"])[tau_indices].astype(np.int64) * 256
        + np.asarray(data["tau_boson_slot"])[tau_indices].astype(np.int64)
    )
    left = np.searchsorted(object_keys, tau_keys, side="left")
    right = np.searchsorted(object_keys, tau_keys, side="right")
    if np.any(right - left != 1):
        raise RuntimeError(f"selected tau does not have exactly one truth object with role={role}")
    return object_indices[left]


def validate_event_identity(
    data: Mapping[str, np.ndarray], row_map: Mapping[str, np.ndarray], seen_rows: np.ndarray, path: Path
) -> np.ndarray:
    rows = np.asarray(data["event_row_global_index"], dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(seen_rows)):
        raise RuntimeError(f"{path.name}: invalid or duplicate global rows within shard")
    if np.any(seen_rows[rows]):
        raise RuntimeError(f"{path.name}: global rows overlap an earlier shard")
    checks = {
        "event_sample_id": "sample_id",
        "event_split_id": "split_id",
        "event_ntuple_entry": "ntuple_entry",
        "event_source_file_index": "source_file_index",
        "event_source_event_number": "source_event_number",
    }
    for truth_key, row_key in checks.items():
        if not np.array_equal(np.asarray(data[truth_key]), np.asarray(row_map[row_key])[rows]):
            raise RuntimeError(f"{path.name}: {truth_key} differs from canonical row map {row_key}")
    seen_rows[rows] = True
    return rows


def select_truth_shard(
    data: Mapping[str, np.ndarray], rows: np.ndarray, path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    event_count = len(rows)
    tau_event = np.asarray(data["tau_event_local_index"], dtype=np.int64)
    tau_pdg = np.asarray(data["tau_pdg_id"], dtype=np.int16)
    tau_exact = (
        (np.asarray(data["tau_mode_code"]) == 0)
        & (np.asarray(data["tau_n_charged_pion"]) == 1)
        & (np.asarray(data["tau_n_neutral_pion"]) == 0)
        & (np.asarray(data["tau_n_tau_neutrino"]) == 1)
        & (np.asarray(data["tau_n_other_neutrino"]) == 0)
        & (np.asarray(data["tau_n_photon"]) == 0)
        & (np.asarray(data["tau_object_count"]) == 2)
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
    event_mask = (
        (np.asarray(data["event_clean_two_tau"]) == 1)
        & (np.asarray(data["event_boson_tau_count"]) == 2)
        & (exact_count == 2)
        & (minus_count == 1)
        & (plus_count == 1)
    )
    selected_events = np.flatnonzero(event_mask)
    tau_minus_by_event = np.full(event_count, -1, dtype=np.int64)
    tau_plus_by_event = np.full(event_count, -1, dtype=np.int64)
    minus_indices = np.flatnonzero(tau_exact & (tau_pdg == 15))
    plus_indices = np.flatnonzero(tau_exact & (tau_pdg == -15))
    tau_minus_by_event[tau_event[minus_indices]] = minus_indices
    tau_plus_by_event[tau_event[plus_indices]] = plus_indices
    tau_indices = np.stack(
        [tau_minus_by_event[selected_events], tau_plus_by_event[selected_events]], axis=1
    )
    if np.any(tau_indices < 0):
        raise RuntimeError(f"{path.name}: charge ordering failed for selected events")

    flat_taus = tau_indices.reshape(-1)
    pion_indices = find_unique_objects(data, flat_taus, ROLE_CHARGED_PION).reshape(-1, 2)
    neutrino_indices = find_unique_objects(data, flat_taus, ROLE_TAU_NEUTRINO).reshape(-1, 2)
    tau_four = np.asarray(data["tau_px_py_pz_e"], dtype=np.float64)[tau_indices]
    pion_four = np.asarray(data["object_px_py_pz_e"], dtype=np.float64)[pion_indices]
    neutrino_four = np.asarray(data["object_px_py_pz_e"], dtype=np.float64)[neutrino_indices]
    closure_scale = np.maximum(tau_four[..., 3], 1.0)
    closure = np.max(np.abs(tau_four - pion_four - neutrino_four), axis=2) / closure_scale
    if np.any(~np.isfinite(closure)):
        raise RuntimeError(f"{path.name}: truth pi+nu closure is not finite")
    closure_candidate_max = float(np.max(closure, initial=0.0))
    closure_event = np.all(closure <= CLOSURE_TOLERANCE, axis=1)
    structural_candidate_count = int(len(selected_events))
    selected_events = selected_events[closure_event]
    tau_indices = tau_indices[closure_event]
    pion_indices = pion_indices[closure_event]
    neutrino_indices = neutrino_indices[closure_event]
    tau_four = tau_four[closure_event]
    pion_four = pion_four[closure_event]
    neutrino_four = neutrino_four[closure_event]
    closure = closure[closure_event]

    tau_energy = tau_four[..., 3]
    pion_energy = pion_four[..., 3]
    if np.any(~np.isfinite(tau_energy)) or np.any(~np.isfinite(pion_energy)) or np.any(tau_energy <= TAU_MASS_GEV):
        raise RuntimeError(f"{path.name}: invalid truth energy in selected cohort")
    x = pion_energy / tau_energy
    a_squared = (CHARGED_PION_MASS_GEV / TAU_MASS_GEV) ** 2
    beta_squared = 1.0 - (TAU_MASS_GEV / tau_energy) ** 2
    if np.any(beta_squared <= 0.0):
        raise RuntimeError(f"{path.name}: non-positive beta squared")
    beta = np.sqrt(beta_squared)
    cosines = (2.0 * x - 1.0 - a_squared) / (beta * (1.0 - a_squared))
    if np.any(~np.isfinite(cosines)):
        raise RuntimeError(f"{path.name}: non-finite thesis cos(theta)")
    range_excess = np.maximum(np.abs(cosines) - 1.0, 0.0)
    if np.any(range_excess > COS_TOLERANCE):
        raise RuntimeError(f"{path.name}: thesis cos(theta) outside range, max excess={float(np.max(range_excess))}")
    x_back = 0.5 * (1.0 + a_squared + beta * (1.0 - a_squared) * cosines)
    inverse_error = np.abs(x_back - x)
    if np.max(inverse_error, initial=0.0) > 1.0e-12:
        raise RuntimeError(f"{path.name}: x/cos(theta) inverse consistency failed")
    cosines = np.clip(cosines, -1.0, 1.0)

    tau_mass_squared = tau_energy**2 - np.sum(tau_four[..., :3] ** 2, axis=2)
    pion_mass_squared = pion_energy**2 - np.sum(pion_four[..., :3] ** 2, axis=2)
    tau_mass = np.sqrt(np.maximum(tau_mass_squared, 0.0))
    pion_mass = np.sqrt(np.maximum(pion_mass_squared, 0.0))
    excluded = np.asarray(data["tau_excluded_simulation_object_count"])[tau_indices]
    record = {
        "row": rows[selected_events],
        "split_id": np.asarray(data["event_split_id"], dtype=np.uint8)[selected_events],
        "sample_id": np.asarray(data["event_sample_id"], dtype=np.uint8)[selected_events],
        "cosines": cosines,
        "x": x,
    }
    audit = {
        "events": event_count,
        "structural_truth_pi_nu_x_pi_nu_candidates": structural_candidate_count,
        "failed_pi_nu_four_vector_closure": int(structural_candidate_count - len(selected_events)),
        "selected_truth_pi_nu_x_pi_nu": int(len(selected_events)),
        "excluded_simulation_objects_on_selected_taus": int(np.sum(excluded)),
        "selected_taus_with_excluded_simulation_objects": int(np.sum(excluded > 0)),
        "max_relative_pi_nu_closure_before_selection": closure_candidate_max,
        "max_relative_pi_nu_closure_selected": float(np.max(closure, initial=0.0)),
        "max_cos_range_excess_before_tolerance_clip": float(np.max(range_excess, initial=0.0)),
        "max_x_inverse_error": float(np.max(inverse_error, initial=0.0)),
        "tau_mass_abs_deviation_quantiles_gev": np.quantile(
            np.abs(tau_mass - TAU_MASS_GEV), [0.5, 0.95, 1.0]
        ).tolist() if len(selected_events) else [0.0, 0.0, 0.0],
        "pion_mass_abs_deviation_quantiles_gev": np.quantile(
            np.abs(pion_mass - CHARGED_PION_MASS_GEV), [0.5, 0.95, 1.0]
        ).tolist() if len(selected_events) else [0.0, 0.0, 0.0],
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
    shard_audits: dict[str, Any] = {}
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
        shard_audits[path.name] = audit
    if not np.all(seen_rows):
        raise RuntimeError(f"truth object shards miss {int(np.sum(~seen_rows))} canonical rows")
    cohort = {key: np.concatenate([record[key] for record in records]) for key in records[0]}
    if len(np.unique(cohort["row"])) != len(cohort["row"]):
        raise RuntimeError("truth cohort global rows are not unique")
    return cohort, {
        "shard_count": len(paths),
        "canonical_rows_seen_once": int(np.sum(seen_rows)),
        "shards": shard_audits,
    }


def attach_fixed_scores(
    cohort: dict[str, np.ndarray], row_map: Mapping[str, np.ndarray], split_inference: Mapping[str, Mapping[str, np.ndarray]]
) -> np.ndarray:
    global_scores = np.full(len(row_map["sample_id"]), np.nan, dtype=np.float64)
    for split, inference in split_inference.items():
        mask = np.asarray(inference["split_mask"], dtype=bool)
        if np.any(np.isfinite(global_scores[mask])):
            raise RuntimeError(f"{split}: score rows overlap another split")
        global_scores[mask] = np.asarray(inference["scores"], dtype=np.float64)
    if not np.isfinite(global_scores).all():
        raise RuntimeError("fixed classifier scores do not cover every canonical row")
    rows = cohort["row"]
    if not np.array_equal(cohort["split_id"], np.asarray(row_map["split_id"])[rows]):
        raise RuntimeError("truth cohort split identity changed before score join")
    if not np.array_equal(cohort["sample_id"], np.asarray(row_map["sample_id"])[rows]):
        raise RuntimeError("truth cohort sample identity changed before score join")
    scores = global_scores[rows]
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError("joined fixed classifier score is invalid")
    return scores


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
    axis.set_xlabel(r"truth $\cos\theta_{-}$")
    axis.set_ylabel(r"truth $\cos\theta_{+}$")
    axis.set_xlim(MAP_RANGE)
    axis.set_ylim(MAP_RANGE)
    axis.set_aspect("equal")


def footer(population: str, bins: int) -> str:
    return (
        f"H/Z MC samples • {population} • exact truth pi-nu x pi-nu • "
        f"fixed 3-seed classifier • unit weight • nominal • {bins}x{bins} bins"
    )


def save_png(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def class_mask(data: Mapping[str, np.ndarray], label: int) -> np.ndarray:
    return np.asarray(data["sample_id"]) == (0 if label == 1 else 1)


def plot_inclusive(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    densities: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
        if count.sum() == 0:
            raise RuntimeError(f"{population}: empty {SAMPLE_NAMES[label]} truth class")
        density = count / count.sum()
        if not np.isclose(density.sum(), 1.0):
            raise RuntimeError(f"{population}: inclusive normalization failed")
        counts[label] = count
        densities[label] = density
    vmax = max(float(np.max(value)) for value in densities.values())
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        image = draw(axis, densities[label], bins, "viridis", Normalize(0.0, vmax))
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={int(counts[label].sum())}")
    assert image is not None
    figure.colorbar(image, ax=axes, label="class-normalized bin probability")
    figure.suptitle(r"Truth $\pi\nu \times \pi\nu$: $(\cos\theta_{-},\,\cos\theta_{+})$", fontsize=13)
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
        raise RuntimeError("not every truth-cohort event received a score tertile")
    return levels, edges


def plot_tertiles(
    data: Mapping[str, np.ndarray], population: str, bins: int, levels: np.ndarray, path: Path
) -> dict[str, Any]:
    counts: dict[int, dict[int, np.ndarray]] = {1: {}, 0: {}}
    densities: list[np.ndarray] = []
    for label in (1, 0):
        for level in range(3):
            mask = class_mask(data, label) & (levels == level)
            count = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
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
    means: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for label in (1, 0):
        mask = class_mask(data, label)
        count = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
        summed = histogram2d(
            data["cosines"][mask, 0], data["cosines"][mask, 1], bins, weights=data["scores"][mask]
        )
        mean = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
        if not np.isfinite(mean).all():
            raise RuntimeError(f"{population}: mean score map contains NaN or infinity")
        counts[label] = count
        means[label] = mean
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
    figure.suptitle("Mean fixed-classifier score per truth angular bin", fontsize=13)
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
        values = data["cosines"][mask]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{population}: scatter contains NaN or infinity")
        axis.scatter(
            values[:, 0], values[:, 1], s=5, alpha=0.20, linewidths=0,
            color=colors[label], rasterized=True, label=f"{SAMPLE_NAMES[label]} truth class",
        )
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={len(values)}")
        axis.legend(loc="upper right", frameon=True, fontsize=8, markerscale=2.2)
        result[SAMPLE_NAMES[label]] = int(len(values))
    figure.suptitle(r"Truth $\pi\nu \times \pi\nu$ angular scatter", fontsize=13)
    figure.text(0.5, -0.015, footer(population, 20).replace(" • 20x20 bins", ""), ha="center", fontsize=7.8)
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
        raise ValueError("this fixed exploratory task requires exactly 20x20 binning")
    repo = args.repo.resolve()
    processed_dir = args.processed_dir.resolve()
    truth_dir = args.truth_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with np.load(args.row_map, allow_pickle=False) as source:
        row_map = {key: np.asarray(source[key]) for key in source.files}

    cohort, truth_audit = load_truth_cohort(truth_dir, row_map)
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
        "definitions": {
            "population": "same canonical split/sample surface and fixed classifier scores as the preceding reco maps; truth pi-nu x pi-nu is independently reselected and is not intersected with reco final events",
            "truth_selection": "clean boson tau pair; charge-ordered PDG 15 (tau-) and -15 (tau+); each tau mode code 0 with exactly one generator-level charged pion and one tau neutrino and no other preserved decay objects; pi+nu four-vector closure <=1e-4",
            "truth_cos_theta": "x=E_pi/E_tau; a=m_pi/m_tau; beta=sqrt(1-m_tau^2/E_tau^2); cos(theta)=(2x-1-a^2)/(beta*(1-a^2)); stored truth four-vector energies; nominal m_tau=1.77686 GeV and m_pi=0.13957039 GeV",
            "A_pair": "not used and not treated as identical to the thesis energy-fraction observable",
            "score": "arithmetic mean of frozen full-reco seeds 42, 43, and 44 terminal checkpoints; eval-only inference; H-like score",
            "tertiles": "unit-weight score tertiles defined separately within each truth class and each population",
            "normalization": "inclusive and tertile panels normalized independently to unit sum; mean maps show raw mean H-like score; scatter contains every selected event",
        },
        "populations": {},
    }
    for key, data in populations.items():
        prefix = f"truth-1p0n-x-1p0n-{key}"
        levels, edges = assign_tertiles(data)
        inclusive = plot_inclusive(data, labels[key], args.bins, output_dir / f"{prefix}-inclusive.png")
        tertiles = plot_tertiles(data, labels[key], args.bins, levels, output_dir / f"{prefix}-score-tertiles.png")
        mean_score = plot_mean_score(data, labels[key], args.bins, output_dir / f"{prefix}-mean-score.png")
        scatter = plot_scatter(data, labels[key], output_dir / f"{prefix}-scatter.png")
        manifest["populations"][key] = {
            "events": int(len(data["row"])),
            "class_counts": {SAMPLE_NAMES[label]: int(np.sum(class_mask(data, label))) for label in (1, 0)},
            "tertile_edges": {SAMPLE_NAMES[label]: edges[label] for label in (1, 0)},
            "cos_min_max_tau_minus": [float(np.min(data["cosines"][:, 0])), float(np.max(data["cosines"][:, 0]))],
            "cos_min_max_tau_plus": [float(np.min(data["cosines"][:, 1])), float(np.max(data["cosines"][:, 1]))],
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
        "truth_pi_nu_counts": {key: manifest["populations"][key]["class_counts"] for key in populations},
        "validation_inference_parity": validation_parity,
        "png_count": sum(len(manifest["populations"][key]["png"]) for key in populations),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
