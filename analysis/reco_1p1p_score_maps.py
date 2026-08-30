#!/usr/bin/env python3
"""Prespecified reco 1p0n x 1p0n helicity-angle score diagnostics.

The primary population is the held-out report subset of the fixed-v3 validation
surface.  The script never trains or calibrates a classifier.  It joins the
frozen three-seed ensemble by the exact ROOT identity triple and constructs the
collinear helicity-angle proxy without clipping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


TAU_MASS_GEV = 1.77686
PION_MASS_GEV = 0.13957039
MIN_ABS_SIN_DPHI = 1.0e-6
MAP_BINS = 6
MAP_RANGE = (-1.0, 1.0)
MEAN_SCORE_MIN_OCCUPANCY = 8
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260826
PERMUTATION_REPLICATES = 1000
PERMUTATION_SEED = 20260827
FIGURE_DPI = 240
SAMPLE_NAMES = {1: "H", 0: "Z"}
SAMPLE_IDS = {"H": 0, "Z": 1}
REPORT_DESCRIPTION = "held-out report subset of validation"
COMMON_FOOTER = (
    "Simulation samples • held-out report subset of validation • "
    "unit weight • nominal only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--identity-reference", type=Path, required=True)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--permutation-replicates", type=int, default=PERMUTATION_REPLICATES)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(values: np.ndarray, stats: dict[str, Any], index: int) -> np.ndarray:
    result = values[..., index].astype(np.float64)
    if bool(stats["standardize"][index]):
        result = result * float(stats["std"][index]) + float(stats["mean"][index])
    return result


def _structured_keys(columns: Iterable[np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    arrays = [np.asarray(column) for column in columns]
    if not arrays or any(len(array) != len(arrays[0]) for array in arrays):
        raise RuntimeError("identity columns have inconsistent lengths")
    dtype = np.dtype([(name, array.dtype) for name, array in zip(names, arrays, strict=True)])
    result = np.empty(len(arrays[0]), dtype=dtype)
    for name, array in zip(names, arrays, strict=True):
        result[name] = array
    return result


def exact_join_indices(left: np.ndarray, right: np.ndarray, description: str) -> np.ndarray:
    """Return right-row indices for every left key, failing on any ambiguity."""
    if left.dtype.names != right.dtype.names:
        raise RuntimeError(f"{description}: key schemas differ")
    if len(np.unique(left)) != len(left):
        raise RuntimeError(f"{description}: duplicate keys on left")
    if len(np.unique(right)) != len(right):
        raise RuntimeError(f"{description}: duplicate keys on right")
    if len(left) != len(right):
        raise RuntimeError(f"{description}: row counts differ ({len(left)} != {len(right)})")
    order = np.argsort(right, order=right.dtype.names, kind="stable")
    sorted_right = right[order]
    position = np.searchsorted(sorted_right, left)
    if np.any(position >= len(sorted_right)):
        raise RuntimeError(f"{description}: missing right key")
    matched = sorted_right[position]
    if not np.array_equal(matched, left):
        raise RuntimeError(f"{description}: key sets differ")
    return order[position]


def join_frozen_inputs(
    sample_ids: np.ndarray,
    event_numbers: np.ndarray,
    labels: np.ndarray,
    identity_reference: dict[str, np.ndarray],
    ensemble: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Audit the frozen row-map sidecar, then join scores by exact ROOT identity."""
    # The processed shards intentionally retain eventNumber only as a diagnostic;
    # it is not unique.  Their frozen builder contract writes validation rows in
    # the validation row-map order, and the trainer exports that exact row-map as
    # this identity sidecar.  Validate every available row diagnostic before
    # attaching the sidecar.  The actual score join below is by the complete ROOT
    # identity triple and is therefore independent of ensemble row order.
    expected_length = len(labels)
    if any(len(value) != expected_length for value in identity_reference.values()):
        raise RuntimeError("processed and identity-sidecar row counts differ")
    if not np.array_equal(sample_ids.astype(np.uint8), identity_reference["sample_id"].astype(np.uint8)):
        raise RuntimeError("processed sample blocks disagree with frozen identity-sidecar order")
    if not np.array_equal(event_numbers.astype(np.uint64), identity_reference["source_event_number"].astype(np.uint64)):
        raise RuntimeError("processed event-number diagnostics disagree with frozen identity sidecar")
    if not np.array_equal(labels.astype(np.uint8), identity_reference["labels"].astype(np.uint8)):
        raise RuntimeError("processed labels disagree with frozen identity sidecar")

    identity_names = ("sample_id", "ntuple_file_index", "ntuple_entry")
    reference_keys = _structured_keys(
        (
            identity_reference["sample_id"],
            identity_reference["ntuple_file_index"],
            identity_reference["ntuple_entry"],
        ),
        identity_names,
    )
    ensemble_keys = _structured_keys(
        (ensemble["identity_0"], ensemble["identity_1"], ensemble["identity_2"]),
        identity_names,
    )
    ensemble_rows = exact_join_indices(reference_keys, ensemble_keys, "identity-sidecar-to-ensemble ROOT join")
    joined_labels = ensemble["labels"][ensemble_rows].astype(np.uint8)
    if not np.array_equal(labels.astype(np.uint8), joined_labels):
        raise RuntimeError("processed labels disagree with ensemble after exact ROOT identity join")
    return {
        "score": ensemble["full_reco_scores"][ensemble_rows].astype(np.float64),
        "report_mask": ensemble["report_mask"][ensemble_rows].astype(bool),
        "sample_id": identity_reference["sample_id"].astype(np.int64),
        "ntuple_file_index": identity_reference["ntuple_file_index"].astype(np.int64),
        "ntuple_entry": identity_reference["ntuple_entry"].astype(np.int64),
        "source_file_index": identity_reference["source_file_index"].astype(np.int64),
        "source_event_number": identity_reference["source_event_number"].astype(np.uint64),
    }


def load_processed(processed_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    stats = json.loads((processed_dir / "stats.json").read_text())
    event_fields = ("event_features", "tau_features", "tau_decay_mode", "labels", "event_numbers")
    accumulators: dict[str, list[np.ndarray]] = {key: [] for key in event_fields}
    track_features: list[np.ndarray] = []
    track_sides: list[np.ndarray] = []
    track_offsets: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    track_shift = 0
    shard_paths: list[Path] = []
    for sample in ("H", "Z"):
        for record in metadata["shards"]["validation"][sample]:
            path = processed_dir / record["path"]
            shard_paths.append(path)
            shard = torch.load(path, map_location="cpu", weights_only=True)
            count = len(shard["labels"])
            for key in event_fields:
                accumulators[key].append(shard[key].numpy())
            track_features.append(shard["track_features"].numpy())
            track_sides.append(shard["track_sides"].numpy())
            offsets = shard["track_offsets"].numpy().astype(np.int64) + track_shift
            track_offsets.append(offsets)
            track_shift += int(shard["track_offsets"][-1])
            sample_ids.append(np.full(count, SAMPLE_IDS[sample], dtype=np.uint8))
    data = {key: np.concatenate(values) for key, values in accumulators.items()}
    data["track_features"] = np.concatenate(track_features)
    data["track_sides"] = np.concatenate(track_sides).astype(np.int64)
    data["track_offsets"] = np.concatenate(
        [offsets if index == 0 else offsets[1:] for index, offsets in enumerate(track_offsets)]
    )
    data["sample_ids"] = np.concatenate(sample_ids)
    return data, {"metadata": metadata, "stats": stats, "shard_paths": shard_paths}


def select_tracks(data: dict[str, np.ndarray], specs: dict[str, Any]) -> dict[str, np.ndarray]:
    metadata = specs["metadata"]
    stats = specs["stats"]
    names = metadata["feature_names"]["track"]
    features = data["track_features"]
    core = restore(features, stats["track"], names.index("track_isCore")) > 0.5
    selector = restore(features, stats["track"], names.index("track_passTrkSelector")) > 0.5
    charge = restore(features, stats["track"], names.index("track_charge"))
    count = len(data["labels"])
    selected_index = np.full((count, 2), -1, dtype=np.int64)
    candidate_count = np.zeros((count, 2), dtype=np.int16)
    charge_match = np.zeros((count, 2), dtype=bool)
    for event_index in range(count):
        start = int(data["track_offsets"][event_index])
        stop = int(data["track_offsets"][event_index + 1])
        sides = data["track_sides"][start:stop]
        for side, expected_charge in ((0, -1.0), (1, 1.0)):
            local = np.flatnonzero((sides == side) & core[start:stop] & selector[start:stop])
            candidate_count[event_index, side] = len(local)
            if len(local) == 1:
                index = start + int(local[0])
                selected_index[event_index, side] = index
                charge_match[event_index, side] = bool(charge[index] * expected_charge > 0.5)
    return {
        "selected_index": selected_index,
        "candidate_count": candidate_count,
        "charge_match": charge_match,
        "unique_pair": np.all(candidate_count == 1, axis=1),
        "charge_pair": np.all(charge_match, axis=1),
    }


def solve_collinear(
    tau_pt: np.ndarray,
    tau_phi: np.ndarray,
    met_et: np.ndarray,
    met_phi: np.ndarray,
    min_abs_sin_dphi: float = MIN_ABS_SIN_DPHI,
) -> dict[str, np.ndarray]:
    px0 = tau_pt[:, 0] * np.cos(tau_phi[:, 0])
    py0 = tau_pt[:, 0] * np.sin(tau_phi[:, 0])
    px1 = tau_pt[:, 1] * np.cos(tau_phi[:, 1])
    py1 = tau_pt[:, 1] * np.sin(tau_phi[:, 1])
    metx = met_et * np.cos(met_phi)
    mety = met_et * np.sin(met_phi)
    determinant = px0 * py1 - py0 * px1
    sin_dphi = np.sin(tau_phi[:, 1] - tau_phi[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha0 = (metx * py1 - mety * px1) / determinant
        alpha1 = (px0 * mety - py0 * metx) / determinant
        alpha = np.column_stack((alpha0, alpha1))
        z = 1.0 / (1.0 + alpha)
    finite = np.isfinite(alpha).all(axis=1) & np.isfinite(z).all(axis=1)
    valid = (
        finite
        & (np.abs(sin_dphi) > min_abs_sin_dphi)
        & np.all(alpha >= 0.0, axis=1)
        & np.all((z > 0.0) & (z <= 1.0), axis=1)
    )
    return {
        "alpha": alpha,
        "z": z,
        "determinant": determinant,
        "sin_dphi": sin_dphi,
        "finite": finite,
        "valid": valid,
    }


def helicity_cosine(e_pi: np.ndarray, e_tau: np.ndarray) -> dict[str, np.ndarray]:
    a = PION_MASS_GEV / TAU_MASS_GEV
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_squared = 1.0 - (TAU_MASS_GEV / e_tau) ** 2
        beta = np.sqrt(beta_squared)
        x = e_pi / e_tau
        raw_cos = (2.0 * x - 1.0 - a**2) / (beta * (1.0 - a**2))
    return {"x": x, "beta": beta, "raw_cos": raw_cos}


def reconstruct(data: dict[str, np.ndarray], specs: dict[str, Any], tracks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    metadata = specs["metadata"]
    stats = specs["stats"]
    event_names = metadata["feature_names"]["event"]
    tau_names = metadata["feature_names"]["tau"]
    track_names = metadata["feature_names"]["track"]
    event = data["event_features"]
    tau = data["tau_features"]
    track = data["track_features"]
    tau_pt = np.expm1(restore(tau, stats["tau"], tau_names.index("log1p_tau_pt")))
    tau_eta = restore(tau, stats["tau"], tau_names.index("tau_eta"))
    tau_phi = np.arctan2(
        restore(tau, stats["tau"], tau_names.index("sin_tau_phi")),
        restore(tau, stats["tau"], tau_names.index("cos_tau_phi")),
    )
    tau_mass = np.expm1(restore(tau, stats["tau"], tau_names.index("log1p_tau_m")))
    met_et = np.expm1(restore(event, stats["event"], event_names.index("log1p_met_et")))
    met_phi = np.arctan2(
        restore(event, stats["event"], event_names.index("sin_met_phi")),
        restore(event, stats["event"], event_names.index("cos_met_phi")),
    )
    collinear = solve_collinear(tau_pt, tau_phi, met_et, met_phi)
    visible_momentum = tau_pt * np.cosh(tau_eta)
    e_visible = np.sqrt(np.maximum(0.0, visible_momentum**2 + tau_mass**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        e_tau = e_visible / collinear["z"]

    e_pi = np.full_like(e_tau, np.nan)
    track_pt_all = np.expm1(restore(track, stats["track"], track_names.index("log1p_track_pt")))
    track_eta_all = restore(track, stats["track"], track_names.index("track_eta"))
    selected = tracks["selected_index"]
    for side in (0, 1):
        present = selected[:, side] >= 0
        indices = selected[present, side]
        track_momentum = track_pt_all[indices] * np.cosh(track_eta_all[indices])
        e_pi[present, side] = np.sqrt(track_momentum**2 + PION_MASS_GEV**2)
    angle = helicity_cosine(e_pi, e_tau)
    angle_finite = np.isfinite(angle["raw_cos"]).all(axis=1)
    angle_range = angle_finite & np.all(np.abs(angle["raw_cos"]) <= 1.0, axis=1)
    return {
        **collinear,
        **angle,
        "e_visible": e_visible,
        "e_tau": e_tau,
        "e_pi": e_pi,
        "tau_pt": tau_pt,
        "tau_eta": tau_eta,
        "tau_phi": tau_phi,
        "tau_mass": tau_mass,
        "met_et": met_et,
        "met_phi": met_phi,
        "angle_finite": angle_finite,
        "angle_range": angle_range,
    }


def stable_tertiles(scores: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    if len(scores) < 3 or not np.isfinite(scores).all():
        raise RuntimeError("tertiles require at least three finite scores")
    q1, q2 = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    labels = np.full(len(scores), 2, dtype=np.int8)
    labels[scores <= q1] = 0
    labels[(scores > q1) & (scores <= q2)] = 1
    return labels, (float(q1), float(q2))


def histogram2d(x: np.ndarray, y: np.ndarray, bins: int = MAP_BINS) -> np.ndarray:
    values, _, _ = np.histogram2d(x, y, bins=bins, range=(MAP_RANGE, MAP_RANGE))
    return values.astype(np.float64)


def normalized(histogram: np.ndarray) -> np.ndarray:
    total = float(histogram.sum())
    return histogram / total if total > 0 else np.full_like(histogram, np.nan)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.nansum(np.abs(normalized(left) - normalized(right))))


def binned_mean(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = histogram2d(x, y)
    summed, _, _ = np.histogram2d(x, y, bins=MAP_BINS, range=(MAP_RANGE, MAP_RANGE), weights=values)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = summed / count
    return mean, count


def cluster_bootstrap_tvd(
    cosines: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    observed = {}
    group_info = {}
    for label in (1, 0):
        mask = labels == label
        observed[label] = histogram2d(cosines[mask, 0], cosines[mask, 1])
        unique, inverse = np.unique(groups[mask], return_inverse=True)
        group_info[label] = (cosines[mask], unique, inverse)
    estimate = total_variation(observed[1], observed[0])
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        histograms = {}
        for label in (1, 0):
            values, unique, inverse = group_info[label]
            drawn = rng.integers(0, len(unique), len(unique))
            multiplicity = np.bincount(drawn, minlength=len(unique)).astype(np.float64)
            weights = multiplicity[inverse]
            histograms[label], _, _ = np.histogram2d(
                values[:, 0], values[:, 1], bins=MAP_BINS, range=(MAP_RANGE, MAP_RANGE), weights=weights
            )
        samples[replicate] = total_variation(histograms[1], histograms[0])
    # TVD is a positive, nonlinear plug-in statistic, so the raw bootstrap
    # distribution is visibly upward biased at this occupancy.  Center the
    # bootstrap deviations on the observed estimate for the reported interval,
    # while retaining the raw percentile range as an explicit diagnostic.
    centered_samples = samples - float(np.mean(samples)) + estimate
    return {
        "estimate": estimate,
        "cluster_bootstrap_95": [
            float(np.clip(np.quantile(centered_samples, 0.025), 0.0, 1.0)),
            float(np.clip(np.quantile(centered_samples, 0.975), 0.0, 1.0)),
        ],
        "interval_method": "cluster bootstrap, distribution centered on the plug-in estimate",
        "raw_bootstrap_percentile_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "raw_bootstrap_mean": float(np.mean(samples)),
        "replicates": int(replicates),
        "seed": int(seed),
        "cluster_unit": "ntuple_file_index sampled independently within truth class",
        "source_files": {
            "H": int(len(group_info[1][1])),
            "Z": int(len(group_info[0][1])),
        },
        "rationale": (
            "2000 replicates matches the frozen classifier comparison. The centered interval reports cluster-resampling spread "
            "without presenting the known upward plug-in bias of sparse-bin TVD as a displacement of the point estimate."
        ),
    }


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    figure.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"png": png.name, "svg": svg.name}


def decorate_map_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(r"reco $\cos\theta_{\tau^-}$")
    axis.set_ylabel(r"reco $\cos\theta_{\tau^+}$")
    axis.set_aspect("equal")


def draw_map(axis: plt.Axes, values: np.ndarray, *, cmap: str, norm: Normalize) -> Any:
    edges = np.linspace(MAP_RANGE[0], MAP_RANGE[1], MAP_BINS + 1)
    return axis.pcolormesh(edges, edges, values.T, cmap=cmap, norm=norm, shading="flat")


def class_masks(primary: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {label: primary & (labels == label) for label in (1, 0)}


def plot_inclusive(
    output_dir: Path, cosines: np.ndarray, labels: np.ndarray, primary: np.ndarray
) -> tuple[dict[str, str], dict[int, np.ndarray]]:
    histograms = {
        label: histogram2d(cosines[mask, 0], cosines[mask, 1])
        for label, mask in class_masks(primary, labels).items()
    }
    densities = {label: normalized(histogram) for label, histogram in histograms.items()}
    vmax = max(float(np.nanmax(value)) for value in densities.values())
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        image = draw_map(axis, densities[label], cmap="viridis", norm=Normalize(0.0, vmax))
        decorate_map_axis(axis, f"{SAMPLE_NAMES[label]} sample • N={int(histograms[label].sum())}")
    assert image is not None
    figure.colorbar(image, ax=axes, label="class-normalized bin probability")
    figure.suptitle("Reco 1p0n × 1p0n helicity-angle density", fontsize=13)
    figure.text(0.5, -0.015, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-inclusive-density"), histograms


def plot_difference(output_dir: Path, histograms: dict[int, np.ndarray], tvd: dict[str, Any]) -> dict[str, str]:
    difference = normalized(histograms[1]) - normalized(histograms[0])
    vmax = float(np.nanmax(np.abs(difference))) or 1.0
    figure, axis = plt.subplots(figsize=(5.8, 4.7), constrained_layout=True)
    image = draw_map(axis, difference, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    decorate_map_axis(axis, "H minus Z")
    figure.colorbar(image, ax=axis, label=r"$p_H - p_Z$ per bin")
    lo, hi = tvd["cluster_bootstrap_95"]
    figure.suptitle(
        f"Signed class-shape difference • TVD={tvd['estimate']:.3f} [{lo:.3f}, {hi:.3f}]",
        fontsize=12,
    )
    figure.text(0.5, -0.015, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-hz-difference")


def plot_tertiles(
    output_dir: Path,
    cosines: np.ndarray,
    labels: np.ndarray,
    primary: np.ndarray,
    tertile: np.ndarray,
) -> tuple[dict[str, str], dict[int, dict[int, np.ndarray]]]:
    histograms: dict[int, dict[int, np.ndarray]] = {}
    densities = []
    for label in (1, 0):
        histograms[label] = {}
        for level in range(3):
            mask = primary & (labels == label) & (tertile == level)
            histogram = histogram2d(cosines[mask, 0], cosines[mask, 1])
            histograms[label][level] = histogram
            densities.append(normalized(histogram))
    vmax = max(float(np.nanmax(value)) for value in densities)
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), constrained_layout=True, sharex=True, sharey=True)
    image = None
    level_names = ("low", "middle", "high")
    for row, label in enumerate((1, 0)):
        for level in range(3):
            histogram = histograms[label][level]
            image = draw_map(axes[row, level], normalized(histogram), cmap="viridis", norm=Normalize(0.0, vmax))
            decorate_map_axis(
                axes[row, level],
                f"{SAMPLE_NAMES[label]} • {level_names[level]} class-rank score • N={int(histogram.sum())}",
            )
    assert image is not None
    figure.colorbar(image, ax=axes, label="panel-normalized bin probability")
    figure.suptitle("Fixed H-like score tertiles within each truth class", fontsize=13)
    figure.text(0.5, -0.01, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-score-tertiles"), histograms


def mean_maps(
    cosines: np.ndarray, scores: np.ndarray, labels: np.ndarray, primary: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    means: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    centers: dict[int, float] = {}
    for label in (1, 0):
        mask = primary & (labels == label)
        mean, count = binned_mean(cosines[mask, 0], cosines[mask, 1], scores[mask])
        center = float(np.mean(scores[mask]))
        mean = mean - center
        mean[count < MEAN_SCORE_MIN_OCCUPANCY] = np.nan
        means[label] = mean
        counts[label] = count
        centers[label] = center
    return means, counts, centers


def plot_mean_score(
    output_dir: Path,
    cosines: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    primary: np.ndarray,
) -> tuple[dict[str, str], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    means, counts, centers = mean_maps(cosines, scores, labels, primary)
    vmax = max(float(np.nanmax(np.abs(value))) for value in means.values() if np.isfinite(value).any())
    vmax = vmax or 1.0
    occupancy_vmax = max(float(np.max(value)) for value in counts.values())
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
    mean_image = occupancy_image = None
    for column, label in enumerate((1, 0)):
        mean_image = draw_map(
            axes[0, column], means[label], cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        )
        decorate_map_axis(
            axes[0, column],
            f"{SAMPLE_NAMES[label]} • mean score − {centers[label]:.3f}\nmask: occupancy < {MEAN_SCORE_MIN_OCCUPANCY}",
        )
        occupancy_image = draw_map(
            axes[1, column], counts[label], cmap="Greys", norm=Normalize(0.0, occupancy_vmax)
        )
        decorate_map_axis(axes[1, column], f"{SAMPLE_NAMES[label]} • raw occupancy • N={int(counts[label].sum())}")
    assert mean_image is not None and occupancy_image is not None
    figure.colorbar(mean_image, ax=axes[0, :], label="centered mean H-like score")
    figure.colorbar(occupancy_image, ax=axes[1, :], label="events per bin")
    figure.suptitle("Class-centered fixed score and occupancy", fontsize=13)
    figure.text(0.5, -0.01, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-mean-score"), means, counts, centers


def quantile_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def plot_event_flow(
    output_dir: Path,
    labels: np.ndarray,
    report: np.ndarray,
    mode: np.ndarray,
    unique_track: np.ndarray,
    track: np.ndarray,
    collinear: np.ndarray,
    angle_finite: np.ndarray,
    primary: np.ndarray,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    stages = (
        ("report", report),
        ("reco 1p0n²", mode),
        ("unique tracks", unique_track),
        ("charge ordered", track),
        ("collinear valid", collinear),
        ("finite raw angle", angle_finite),
        ("both |cosθ|≤1", primary),
    )
    rows = []
    figure, axis = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    colors = {1: "#D55E00", 0: "#0072B2"}
    for label in (1, 0):
        denominator = int(np.sum(report & (labels == label)))
        counts = []
        for stage, mask in stages:
            count = int(np.sum(mask & (labels == label)))
            counts.append(count)
            rows.append(
                {
                    "sample": SAMPLE_NAMES[label],
                    "stage": stage,
                    "events": count,
                    "fraction_of_report": count / denominator if denominator else float("nan"),
                }
            )
        fractions = np.asarray(counts, dtype=float) / denominator
        axis.plot(range(len(stages)), fractions, marker="o", linewidth=2, color=colors[label], label=f"{SAMPLE_NAMES[label]} (N={denominator})")
        for index, (fraction, count) in enumerate(zip(fractions, counts, strict=True)):
            axis.annotate(str(count), (index, fraction), xytext=(0, 6 if label == 1 else -12), textcoords="offset points", ha="center", fontsize=7)
    axis.set_xticks(range(len(stages)), [stage for stage, _ in stages], rotation=22, ha="right")
    axis.set_yscale("log")
    axis.set_ylim(0.01, 1.15)
    axis.set_ylabel("fraction of class report subset")
    axis.set_title("Reco 1p0n × 1p0n event flow (raw angles are never clipped)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.text(0.5, -0.02, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-event-flow"), rows


def plot_quality_diagnostics(
    output_dir: Path,
    labels: np.ndarray,
    track: np.ndarray,
    tertile: np.ndarray,
    reco: dict[str, np.ndarray],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    level_names = ("low", "middle", "high")
    colors = {1: "#D55E00", 0: "#0072B2"}
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.1), constrained_layout=True)
    rows: list[dict[str, Any]] = []
    metric_specs = (
        ("collinear valid fraction", None, axes[0, 0]),
        ("z (two tau sides)", "z", axes[0, 1]),
        (r"$|\sin\Delta\phi(\tau^-,\tau^+)|$", "abs_sin", axes[0, 2]),
        ("MET [GeV]", "met", axes[1, 0]),
        (r"tau $p_T$ [GeV] (two sides)", "tau_pt", axes[1, 1]),
        (r"$|\eta_\tau|$ (two sides)", "tau_abs_eta", axes[1, 2]),
    )
    for metric_title, metric, axis in metric_specs:
        for class_offset, label in ((-0.06, 1), (0.06, 0)):
            x_positions = np.arange(3, dtype=float) + class_offset
            medians, low_errors, high_errors = [], [], []
            for level in range(3):
                base = track & (labels == label) & (tertile == level)
                if metric is None:
                    value = float(np.mean(reco["valid"][base])) if np.any(base) else float("nan")
                    q16 = q50 = q84 = value
                    count = int(np.sum(base))
                else:
                    if metric == "z":
                        values = reco["z"][base & reco["valid"]].reshape(-1)
                    elif metric == "abs_sin":
                        values = np.abs(reco["sin_dphi"][base])
                    elif metric == "met":
                        values = reco["met_et"][base]
                    elif metric == "tau_pt":
                        values = reco["tau_pt"][base].reshape(-1)
                    elif metric == "tau_abs_eta":
                        values = np.abs(reco["tau_eta"][base]).reshape(-1)
                    else:
                        raise AssertionError(metric)
                    q16, q50, q84 = quantile_summary(values)
                    count = int(np.isfinite(values).sum())
                medians.append(q50)
                low_errors.append(q50 - q16)
                high_errors.append(q84 - q50)
                rows.append(
                    {
                        "sample": SAMPLE_NAMES[label],
                        "score_tertile": level_names[level],
                        "metric": metric_title,
                        "n_values": count,
                        "q16": q16,
                        "median": q50,
                        "q84": q84,
                    }
                )
            if metric is None:
                axis.plot(x_positions, medians, marker="o", color=colors[label], label=SAMPLE_NAMES[label])
            else:
                axis.errorbar(
                    x_positions,
                    medians,
                    yerr=np.vstack((low_errors, high_errors)),
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    color=colors[label],
                    label=SAMPLE_NAMES[label],
                )
        axis.set_xticks(range(3), level_names)
        axis.set_title(metric_title, fontsize=10)
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Score-rank dependence of collinear quality and reco kinematics", fontsize=13)
    figure.text(0.5, -0.01, COMMON_FOOTER + " • points: median, bars: 16–84%", ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-collinear-diagnostics"), rows


def plot_angle_diagnostics(
    output_dir: Path,
    labels: np.ndarray,
    track: np.ndarray,
    collinear: np.ndarray,
    reco: dict[str, np.ndarray],
) -> dict[str, str]:
    colors = {1: "#D55E00", 0: "#0072B2"}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    for label in (1, 0):
        mask = track & collinear & (labels == label)
        x = reco["x"][mask].reshape(-1)
        raw_cos = reco["raw_cos"][mask].reshape(-1)
        axes[0].hist(x, bins=np.linspace(-0.5, 1.5, 61), density=True, histtype="step", linewidth=1.8, color=colors[label], label=SAMPLE_NAMES[label])
        axes[1].hist(raw_cos, bins=np.linspace(-3.0, 3.0, 81), density=True, histtype="step", linewidth=1.8, color=colors[label], label=SAMPLE_NAMES[label])
    axes[0].set(title=r"$x=E_\pi/E_\tau$", xlabel="raw x (visible plotting window)", ylabel="side-level density")
    axes[1].set(title=r"raw reco $\cos\theta$ (not clipped)", xlabel="raw cosθ (visible plotting window)", ylabel="side-level density")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Collinear angle construction diagnostics", fontsize=13)
    figure.text(0.5, -0.01, COMMON_FOOTER, ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-angle-diagnostics")


def permutation_null(
    cosines: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    track: np.ndarray,
    primary: np.ndarray,
    tertile_edges: dict[int, tuple[float, float]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    first_shuffle_scores = scores.copy()
    results: dict[str, Any] = {}
    first_means: dict[int, np.ndarray] = {}
    first_counts: dict[int, np.ndarray] = {}
    for label in (1, 0):
        class_track_indices = np.flatnonzero(track & (labels == label))
        class_primary = primary & (labels == label)
        q1, q2 = tertile_edges[label]
        observed_level = np.full(len(scores), -1, dtype=np.int8)
        observed_level[class_track_indices] = np.where(
            scores[class_track_indices] <= q1,
            0,
            np.where(scores[class_track_indices] <= q2, 1, 2),
        )
        observed = total_variation(
            histogram2d(cosines[class_primary & (observed_level == 0), 0], cosines[class_primary & (observed_level == 0), 1]),
            histogram2d(cosines[class_primary & (observed_level == 2), 0], cosines[class_primary & (observed_level == 2), 1]),
        )
        null = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            shuffled = rng.permutation(scores[class_track_indices])
            if replicate == 0:
                first_shuffle_scores[class_track_indices] = shuffled
            shuffled_level = np.where(shuffled <= q1, 0, np.where(shuffled <= q2, 1, 2))
            physical_local = primary[class_track_indices]
            values = cosines[class_track_indices]
            null[replicate] = total_variation(
                histogram2d(values[physical_local & (shuffled_level == 0), 0], values[physical_local & (shuffled_level == 0), 1]),
                histogram2d(values[physical_local & (shuffled_level == 2), 0], values[physical_local & (shuffled_level == 2), 1]),
            )
        results[SAMPLE_NAMES[label]] = {
            "observed_low_high_tvd": observed,
            "permuted_tvd_median": float(np.median(null)),
            "permuted_tvd_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
            "permutation_p_ge_observed": float((1 + np.sum(null >= observed)) / (replicates + 1)),
            "replicates": int(replicates),
        }
    first_means, first_counts, _ = mean_maps(cosines, first_shuffle_scores, labels, primary)
    results["seed"] = int(seed)
    results["rationale"] = "1000 within-class permutations give 0.001 tail-probability resolution for a minimum visual null control; no significance claim is made."
    return results, first_means, first_counts


def plot_shuffled_null(
    output_dir: Path,
    means: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
    permutation: dict[str, Any],
) -> dict[str, str]:
    finite_values = [value[np.isfinite(value)] for value in means.values() if np.isfinite(value).any()]
    vmax = max(float(np.max(np.abs(value))) for value in finite_values) if finite_values else 1.0
    vmax = vmax or 1.0
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        image = draw_map(axis, means[label], cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
        result = permutation[SAMPLE_NAMES[label]]
        decorate_map_axis(
            axis,
            f"{SAMPLE_NAMES[label]} shuffled within class\nlow–high TVD obs {result['observed_low_high_tvd']:.3f}, null median {result['permuted_tvd_median']:.3f}",
        )
    assert image is not None
    figure.colorbar(image, ax=axes, label=f"centered shuffled mean score (bins N≥{MEAN_SCORE_MIN_OCCUPANCY})")
    figure.suptitle("Deterministic shuffled-score null", fontsize=13)
    figure.text(0.5, -0.015, COMMON_FOOTER + " • shuffle is not a kinematic deconfounder", ha="center", fontsize=8)
    return save_figure(figure, output_dir, "reco-1p1p-shuffled-null")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts_document(
    labels: np.ndarray,
    report: np.ndarray,
    mode: np.ndarray,
    tracks: dict[str, np.ndarray],
    reco: dict[str, np.ndarray],
) -> dict[str, dict[str, int]]:
    unique = report & mode & tracks["unique_pair"]
    track = unique & tracks["charge_pair"]
    collinear = track & reco["valid"]
    finite = collinear & reco["angle_finite"]
    physical = finite & reco["angle_range"]
    result = {}
    for label in (1, 0):
        class_mask = labels == label
        result[SAMPLE_NAMES[label]] = {
            "validation": int(np.sum(class_mask)),
            "report": int(np.sum(class_mask & report)),
            "mode_1p0n_x_1p0n": int(np.sum(class_mask & report & mode)),
            "unique_core_selector_track_pair": int(np.sum(class_mask & unique)),
            "charge_ordered_track_pair": int(np.sum(class_mask & track)),
            "collinear_valid": int(np.sum(class_mask & collinear)),
            "finite_raw_angles": int(np.sum(class_mask & finite)),
            "both_raw_angles_in_range": int(np.sum(class_mask & physical)),
        }
    return result


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    ensemble_path = args.ensemble.resolve()
    identity_path = args.identity_reference.resolve()
    comparison_path = args.comparison_summary.resolve()
    data, specs = load_processed(processed_dir)
    with np.load(identity_path, allow_pickle=False) as source:
        identity_reference = {key: np.asarray(source[key]) for key in source.files}
    with np.load(ensemble_path, allow_pickle=False) as source:
        ensemble = {key: np.asarray(source[key]) for key in source.files}
    joined = join_frozen_inputs(
        data["sample_ids"], data["event_numbers"].astype(np.uint64), data["labels"], identity_reference, ensemble
    )
    comparison = json.loads(comparison_path.read_text())
    actual_ensemble_hash = sha256_file(ensemble_path)
    declared_ensemble_hash = comparison["ensemble_artifact"]["sha256"]
    if actual_ensemble_hash != declared_ensemble_hash:
        raise RuntimeError("ensemble hash disagrees with frozen comparison summary")
    expected_report = int(comparison["report_partition"]["events"])
    if int(joined["report_mask"].sum()) != expected_report:
        raise RuntimeError("joined report mask count disagrees with frozen comparison summary")
    for label_text, expected_count in comparison["report_partition"]["class_counts"].items():
        label = int(label_text)
        observed_count = int(np.sum(joined["report_mask"] & (data["labels"] == label)))
        if observed_count != int(expected_count):
            raise RuntimeError(
                f"report-mask class count for label {label} disagrees with frozen comparison summary"
            )

    tracks = select_tracks(data, specs)
    reco = reconstruct(data, specs, tracks)
    one_prong_id = int(specs["metadata"]["tau_decay_mode_to_id"]["0"])
    mode = np.all(data["tau_decay_mode"] == one_prong_id, axis=1)
    report = joined["report_mask"]
    count_summary = counts_document(data["labels"], report, mode, tracks, reco)
    if args.counts_only:
        print(json.dumps({"population": REPORT_DESCRIPTION, "map_bins": MAP_BINS, "counts": count_summary}, indent=2))
        return 0
    if args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --counts-only is used")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    unique_track = report & mode & tracks["unique_pair"]
    track = unique_track & tracks["charge_pair"]
    collinear = track & reco["valid"]
    finite_angle = collinear & reco["angle_finite"]
    primary = finite_angle & reco["angle_range"]
    score = joined["score"]
    labels = data["labels"].astype(np.uint8)
    tertile = np.full(len(labels), -1, dtype=np.int8)
    tertile_edges: dict[int, tuple[float, float]] = {}
    tertile_rows: list[dict[str, Any]] = []
    level_names = ("low", "middle", "high")
    for label in (1, 0):
        cohort = track & (labels == label)
        class_levels, edges = stable_tertiles(score[cohort])
        tertile[cohort] = class_levels
        tertile_edges[label] = edges
        for level in range(3):
            base = cohort & (tertile == level)
            tertile_rows.append(
                {
                    "sample": SAMPLE_NAMES[label],
                    "score_tertile": level_names[level],
                    "q1": edges[0],
                    "q2": edges[1],
                    "track_cohort_events": int(np.sum(base)),
                    "collinear_valid_events": int(np.sum(base & reco["valid"])),
                    "physical_map_events": int(np.sum(base & primary)),
                    "collinear_valid_fraction": float(np.mean(reco["valid"][base])),
                    "physical_map_fraction": float(np.mean(primary[base])),
                }
            )

    cosines = reco["raw_cos"]
    physical_labels = labels[primary]
    physical_groups = joined["ntuple_file_index"][primary]
    tvd = cluster_bootstrap_tvd(
        cosines[primary], physical_labels, physical_groups, args.bootstrap_replicates, BOOTSTRAP_SEED
    )

    figures: dict[str, dict[str, str]] = {}
    figures["inclusive_density"], inclusive_histograms = plot_inclusive(output_dir, cosines, labels, primary)
    figures["signed_h_minus_z"] = plot_difference(output_dir, inclusive_histograms, tvd)
    figures["score_tertiles"], tertile_histograms = plot_tertiles(output_dir, cosines, labels, primary, tertile)
    figures["mean_score"], observed_mean, occupancy, class_centers = plot_mean_score(
        output_dir, cosines, score, labels, primary
    )
    figures["event_flow"], flow_rows = plot_event_flow(
        output_dir, labels, report, report & mode, unique_track, track, collinear, finite_angle, primary
    )
    figures["collinear_diagnostics"], diagnostic_rows = plot_quality_diagnostics(
        output_dir, labels, track, tertile, reco
    )
    figures["angle_diagnostics"] = plot_angle_diagnostics(output_dir, labels, track, collinear, reco)
    permutation, shuffled_mean, shuffled_count = permutation_null(
        cosines,
        score,
        labels,
        track,
        primary,
        tertile_edges,
        args.permutation_replicates,
        PERMUTATION_SEED,
    )
    figures["shuffled_score_null"] = plot_shuffled_null(
        output_dir, shuffled_mean, shuffled_count, permutation
    )

    write_csv(output_dir / "event-flow.csv", flow_rows)
    write_csv(output_dir / "score-tertile-summary.csv", tertile_rows)
    write_csv(output_dir / "collinear-diagnostics-summary.csv", diagnostic_rows)
    np.savez_compressed(
        output_dir / "map-data.npz",
        bin_edges=np.linspace(MAP_RANGE[0], MAP_RANGE[1], MAP_BINS + 1),
        inclusive_h=inclusive_histograms[1],
        inclusive_z=inclusive_histograms[0],
        tertile_h=np.stack([tertile_histograms[1][level] for level in range(3)]),
        tertile_z=np.stack([tertile_histograms[0][level] for level in range(3)]),
        mean_score_h=observed_mean[1],
        mean_score_z=observed_mean[0],
        occupancy_h=occupancy[1],
        occupancy_z=occupancy[0],
        shuffled_mean_score_h=shuffled_mean[1],
        shuffled_mean_score_z=shuffled_mean[0],
    )

    input_hashes = {
        "metadata.json": sha256_file(processed_dir / "metadata.json"),
        "stats.json": sha256_file(processed_dir / "stats.json"),
        "terminal-ensembles.npz": actual_ensemble_hash,
        "validation_predictions.npz": sha256_file(identity_path),
        "comparison-summary.json": sha256_file(comparison_path),
    }
    for path in specs["shard_paths"]:
        input_hashes[str(path.relative_to(processed_dir))] = sha256_file(path)
    manifest = {
        "format_version": 1,
        "status": "completed",
        "primary_estimand": (
            "unit-weight nominal shape in the held-out report subset of the fixed-v3 pT-matched validation surface"
        ),
        "population": {
            "partition": REPORT_DESCRIPTION,
            "report_events": int(report.sum()),
            "frozen_summary_report_events": expected_report,
            "counts": count_summary,
            "weights": "unit weight; no additional parent-overlap weight",
        },
        "identity_join": {
            "ensemble_key": ["sample_id", "ntuple_file_index", "ntuple_entry"],
            "processed_to_sidecar": (
                "the processed-shard builder and prediction sidecar share the frozen validation row-map order; all sample_id, "
                "label, and source_event_number rows are required to agree before identities are attached"
            ),
            "event_number_caveat": "source_event_number is diagnostic and non-unique, so it is never used as a join key",
            "semantics": (
                "ensemble scores are joined by unique complete ROOT identity key sets; missing, duplicate, or label-mismatched rows fail closed, "
                "and ensemble input row order is not used"
            ),
        },
        "selection": {
            "decay_mode": "both reconstructed taus raw decay-mode code 0 (processed categorical id 1)",
            "charged_track": "exactly one core && passTrkSelector track per tau, with tau[0]/tau[1] charges -1/+1",
            "tau_side_convention": "tau[0] = tau-; tau[1] = tau+, audited by selected track charge",
            "score_tertile_population": "report subset, reco 1p0n x 1p0n, unique charge-ordered track pair; before collinear validity",
            "score_tertile_rule": "within truth class, unit weight: s<=q1; q1<s<=q2; q2<s",
            "score_tertile_edges": {
                SAMPLE_NAMES[label]: {"q1": edges[0], "q2": edges[1]}
                for label, edges in tertile_edges.items()
            },
        },
        "definition": {
            "score": "H-like score from frozen full-reco arithmetic ensemble of seeds 42, 43, 44 at terminal checkpoints",
            "collinear_equation": "MET_T = alpha_minus pT_vis_minus + alpha_plus pT_vis_plus; z=1/(1+alpha); E_tau=E_vis/z",
            "angle_equation": "cos(theta)=(2*x-1-a^2)/(beta*(1-a^2)); x=E_pi/E_tau; a=m_pi/m_tau; beta=sqrt(1-m_tau^2/E_tau^2)",
            "e_pi": "unique selected charged track with charged-pion mass hypothesis",
            "m_tau_GeV": TAU_MASS_GEV,
            "m_pi_charged_GeV": PION_MASS_GEV,
            "minimum_abs_sin_tau_dphi": MIN_ABS_SIN_DPHI,
            "collinear_valid": "finite alpha,z; |sin(dphi)|>1e-6; alpha>=0; 0<z<=1 on both sides",
            "map_range": list(MAP_RANGE),
            "map_bins_per_axis": MAP_BINS,
            "map_population": "collinear valid, finite raw angle, and both raw |cos(theta)|<=1",
            "clipping": "none; out-of-range raw angles are counted outside the primary map",
            "mean_score_min_occupancy": MEAN_SCORE_MIN_OCCUPANCY,
        },
        "effect_sizes": {"inclusive_h_z_tvd": tvd, "within_class_score_permutation": permutation},
        "lineage": {
            "comparison_status": comparison["status"],
            "comparison_protocol_sha256": comparison["protocol_sha256"],
            "comparison_claim_boundary": comparison["claim_boundary"],
            "terminal_ensemble_auc_full_reco": comparison["terminal_ensemble_auc"]["full_reco"],
            "full_reco_runs": comparison["runs"]["full_reco"],
        },
        "input_sha256": input_hashes,
        "figures": figures,
        "tables": [
            "event-flow.csv",
            "score-tertile-summary.csv",
            "collinear-diagnostics-summary.csv",
            "map-data.npz",
        ],
        "limitations": [
            "No processed-tensor truth pi-nu purity or truth-to-reco angular response is available.",
            "No additional parent pT-|eta| overlap weighting is applied.",
            "Truth-class conditioning removes overall H/Z mixture changes, not within-class kinematics, acceptance, reconstruction, MET, collinear-quality, or direct classifier-input dependence.",
            "Class-specific score tertiles are within-class ranks and are not a common operating threshold.",
            "A shuffled-score map is a visual association null, not a kinematic deconfounder.",
        ],
        "claim_boundary": (
            "These figures test association between a frozen classifier score and the reco helicity-angle proxy in selected simulated reco 1p0n x 1p0n events. "
            "They do not establish that the classifier learned or used spin, nor that an H/Z shape difference is spin-only."
        ),
        "bootstrap_note": "The H/Z TVD interval resamples ROOT ntuple_file_index clusters independently within class.",
        "permutation_note": "The low-high TVD null shuffles scores only within truth class and the fixed track-audited cohort.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact-sha256.json"
    }
    (output_dir / "artifact-sha256.json").write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "counts": count_summary, "tvd": tvd, "permutation": permutation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
