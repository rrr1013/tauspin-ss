#!/usr/bin/env python3
"""Plot the requested reco 1p0n x 1p0n maps for two fixed populations.

This script performs inference only with the already-trained three-seed
full-reco ensemble.  It reuses the established reconstruction helpers from
``reco_1p1p_score_maps.py`` and deliberately produces only the requested six
PNG figures.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from reco_1p1p_score_maps import reconstruct, select_tracks, stable_tertiles


SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
SAMPLE_IDS = {"H": 0, "Z": 1}
SAMPLE_NAMES = {1: "H", 0: "Z"}
SEEDS = (42, 43, 44)
ALLOWED_TYPES = (1, 2, 3, 4)
MAP_RANGE = (-1.0, 1.0)
FIGURE_DPI = 240


class TokenBlockModel(nn.Module):
    """Apply the frozen full-reco token visibility mask."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        lookup = torch.zeros(6, dtype=torch.bool)
        lookup[list(ALLOWED_TYPES)] = True
        self.register_buffer("allowed_type_lookup", lookup, persistent=False)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        visible = self.allowed_type_lookup[batch["object_type"]]
        masked_batch = dict(batch)
        masked_batch["padding_mask"] = batch["padding_mask"] | ~visible
        return self.base(masked_batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--validation-ensemble", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def structured_identity(source: Mapping[str, np.ndarray]) -> np.ndarray:
    fields = (
        ("sample_id", np.asarray(source["sample_id"])),
        ("ntuple_file_index", np.asarray(source["ntuple_file_index"])),
        ("ntuple_entry", np.asarray(source["ntuple_entry"])),
    )
    dtype = np.dtype([(name, values.dtype) for name, values in fields])
    result = np.empty(len(fields[0][1]), dtype=dtype)
    for name, values in fields:
        result[name] = values
    return result


def join_reference_scores(row_identity: np.ndarray, reference: Mapping[str, np.ndarray]) -> np.ndarray:
    reference_source = {
        "sample_id": np.asarray(reference["identity_0"]),
        "ntuple_file_index": np.asarray(reference["identity_1"]),
        "ntuple_entry": np.asarray(reference["identity_2"]),
    }
    reference_identity = structured_identity(reference_source)
    if len(np.unique(row_identity)) != len(row_identity) or len(np.unique(reference_identity)) != len(reference_identity):
        raise RuntimeError("validation reference identity is not unique")
    order = np.argsort(reference_identity, order=reference_identity.dtype.names, kind="stable")
    sorted_identity = reference_identity[order]
    positions = np.searchsorted(sorted_identity, row_identity)
    if np.any(positions >= len(sorted_identity)) or not np.array_equal(sorted_identity[positions], row_identity):
        raise RuntimeError("validation reference identity does not match the canonical row map")
    return np.asarray(reference["full_reco_scores"], dtype=np.float64)[order[positions]]


def load_models(
    repo: Path,
    processed_dir: Path,
    checkpoint_root: Path,
    device: torch.device,
) -> tuple[list[nn.Module], dict[str, str]]:
    sys.path.insert(0, str(repo / "NN"))
    from hpo_utils import create_model  # pylint: disable=import-outside-toplevel

    metadata = json.loads((processed_dir / "metadata.json").read_text())
    models: list[nn.Module] = []
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        checkpoint_path = checkpoint_root / f"reco-full_reco-seed{seed}" / "current_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("arm") != "full_reco" or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError(f"checkpoint identity mismatch for seed {seed}")
        base, _ = create_model(metadata, "small", 0.1, device)
        model = TokenBlockModel(base).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        hashes[f"seed{seed}"] = sha256_file(checkpoint_path)
    return models, hashes


def infer_split(
    processed_dir: Path,
    split: str,
    row_map: Mapping[str, np.ndarray],
    models: list[nn.Module],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    from hpo_utils import create_streaming_loader, move_batch, shutdown_loader_workers  # pylint: disable=import-outside-toplevel

    split_mask = np.asarray(row_map["split_id"]) == SPLIT_IDS[split]
    expected_labels = np.asarray(1 - row_map["sample_id"][split_mask], dtype=np.uint8)
    expected_events = np.asarray(row_map["source_event_number"][split_mask], dtype=np.uint64)
    _, loader = create_streaming_loader(
        processed_dir,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=2,
        shuffle=False,
        balanced=False,
        seed=42,
        worker_partition="shard",
    )
    label_chunks: list[np.ndarray] = []
    event_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for cpu_batch in loader:
                label_chunks.append(cpu_batch["labels"].numpy().astype(np.uint8))
                event_chunks.append(cpu_batch["event_numbers"].numpy().astype(np.uint64))
                batch = move_batch(cpu_batch, device)
                seed_scores = [torch.sigmoid(model(batch)).detach().cpu().numpy() for model in models]
                score_chunks.append(np.mean(np.stack(seed_scores), axis=0, dtype=np.float64))
    finally:
        shutdown_loader_workers(loader)
    labels = np.concatenate(label_chunks)
    events = np.concatenate(event_chunks)
    scores = np.concatenate(score_chunks).astype(np.float64)
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"{split}: processed labels do not match the canonical row map")
    if not np.array_equal(events, expected_events):
        raise RuntimeError(f"{split}: processed event numbers do not match the canonical row map")
    if len(scores) != int(split_mask.sum()) or not np.isfinite(scores).all():
        raise RuntimeError(f"{split}: score count or finiteness check failed")
    return {"labels": labels, "events": events, "scores": scores, "split_mask": split_mask}


def load_processed_split(processed_dir: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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
        for record in metadata["shards"][split][sample]:
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


def compact_geometry(
    processed_dir: Path,
    split: str,
    inference: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    data, specs = load_processed_split(processed_dir, split)
    if not np.array_equal(data["labels"].astype(np.uint8), inference["labels"]):
        raise RuntimeError(f"{split}: geometry and inference labels differ")
    if not np.array_equal(data["event_numbers"].astype(np.uint64), inference["events"]):
        raise RuntimeError(f"{split}: geometry and inference event order differs")
    tracks = select_tracks(data, specs)
    reco = reconstruct(data, specs, tracks)
    one_prong_id = int(specs["metadata"]["tau_decay_mode_to_id"]["0"])
    mode = np.all(data["tau_decay_mode"] == one_prong_id, axis=1)
    unique = mode & tracks["unique_pair"]
    track = unique & tracks["charge_pair"]
    collinear = track & reco["valid"]
    finite = collinear & reco["angle_finite"]
    physical = finite & reco["angle_range"]
    compact = {
        "labels": data["labels"].astype(np.uint8),
        "scores": np.asarray(inference["scores"], dtype=np.float64),
        "track": track,
        "physical": physical,
        "cosines": reco["raw_cos"].astype(np.float64),
        "split_id": np.full(len(data["labels"]), SPLIT_IDS[split], dtype=np.uint8),
    }
    counts = {
        "events": int(len(mode)),
        "mode_1p0n_x_1p0n": int(mode.sum()),
        "unique_core_selector_track_pair": int(unique.sum()),
        "charge_ordered_track_pair": int(track.sum()),
        "collinear_valid": int(collinear.sum()),
        "finite_raw_angles": int(finite.sum()),
        "both_raw_angles_in_range": int(physical.sum()),
    }
    del data, tracks, reco
    gc.collect()
    return compact, counts


def concatenate(records: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([np.asarray(record[key]) for record in records]) for key in records[0]}


def histogram2d(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    values, _, _ = np.histogram2d(x, y, bins=bins, range=(MAP_RANGE, MAP_RANGE))
    return values.astype(np.float64)


def validate_histogram(histogram: np.ndarray, context: str) -> int:
    if not np.isfinite(histogram).all():
        raise RuntimeError(f"{context}: histogram contains NaN or infinity")
    return int(np.sum(histogram == 0))


def decorate(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.5)
    axis.set_xlabel(r"reco $\cos\theta_{-}$")
    axis.set_ylabel(r"reco $\cos\theta_{+}$")
    axis.set_aspect("equal")


def draw(axis: plt.Axes, values: np.ndarray, bins: int, cmap: Any, norm: Normalize) -> Any:
    edges = np.linspace(MAP_RANGE[0], MAP_RANGE[1], bins + 1)
    return axis.pcolormesh(edges, edges, values.T, cmap=cmap, norm=norm, shading="flat")


def save_png(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def footer(population: str, bins: int) -> str:
    return (
        f"Simulation samples • {population} • fixed 3-seed full-reco classifier • "
        f"unit weight • nominal • {bins}×{bins} bins • reco collinear charged-track angle proxy"
    )


def plot_inclusive(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    histograms = {}
    densities = {}
    for label in (1, 0):
        mask = data["physical"] & (data["labels"] == label)
        histogram = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
        empty_bins = validate_histogram(histogram, f"{population} inclusive {SAMPLE_NAMES[label]}")
        density = histogram / histogram.sum()
        if not np.isfinite(density).all() or not np.isclose(density.sum(), 1.0):
            raise RuntimeError(f"{population} inclusive {SAMPLE_NAMES[label]} normalization failed")
        histograms[label] = histogram
        densities[label] = density
    vmax = max(float(np.max(value)) for value in densities.values())
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        image = draw(axis, densities[label], bins, "viridis", Normalize(0.0, vmax))
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={int(histograms[label].sum())}")
    assert image is not None
    figure.colorbar(image, ax=axes, label="class-normalized bin probability")
    figure.suptitle(r"Reco 1p0n $\times$ 1p0n: $(\cos\theta_{-},\,\cos\theta_{+})$", fontsize=13)
    figure.text(0.5, -0.015, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            "histogram": histograms[label],
            "empty_bins": int(np.sum(histograms[label] == 0)),
        }
        for label in (1, 0)
    }


def assign_tertiles(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    levels = np.full(len(data["labels"]), -1, dtype=np.int8)
    edges = {}
    for label in (1, 0):
        cohort = data["track"] & (data["labels"] == label)
        class_levels, class_edges = stable_tertiles(data["scores"][cohort])
        levels[cohort] = class_levels
        edges[label] = class_edges
    return levels, edges


def plot_tertiles(
    data: Mapping[str, np.ndarray],
    population: str,
    bins: int,
    levels: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    histograms: dict[int, dict[int, np.ndarray]] = {1: {}, 0: {}}
    densities = []
    for label in (1, 0):
        for level in range(3):
            mask = data["physical"] & (data["labels"] == label) & (levels == level)
            histogram = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
            validate_histogram(histogram, f"{population} {SAMPLE_NAMES[label]} tertile {level}")
            density = histogram / histogram.sum()
            if not np.isfinite(density).all() or not np.isclose(density.sum(), 1.0):
                raise RuntimeError(f"{population} {SAMPLE_NAMES[label]} tertile {level} normalization failed")
            histograms[label][level] = histogram
            densities.append(density)
    vmax = max(float(np.max(value)) for value in densities)
    figure, axes = plt.subplots(2, 3, figsize=(11.3, 7.1), constrained_layout=True, sharex=True, sharey=True)
    names = ("low", "mid", "high")
    image = None
    for row, label in enumerate((1, 0)):
        for level in range(3):
            histogram = histograms[label][level]
            density = histogram / histogram.sum()
            image = draw(axes[row, level], density, bins, "viridis", Normalize(0.0, vmax))
            decorate(axes[row, level], f"{SAMPLE_NAMES[label]} • {names[level]} within-class score • N={int(histogram.sum())}")
    assert image is not None
    figure.colorbar(image, ax=axes, label="panel-normalized bin probability")
    figure.suptitle("Fixed classifier score tertiles within each truth class", fontsize=13)
    figure.text(0.5, -0.01, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            str(level): {
                "histogram": histograms[label][level],
                "empty_bins": int(np.sum(histograms[label][level] == 0)),
            }
            for level in range(3)
        }
        for label in (1, 0)
    }


def plot_mean_score(data: Mapping[str, np.ndarray], population: str, bins: int, path: Path) -> dict[str, Any]:
    means = {}
    counts = {}
    for label in (1, 0):
        mask = data["physical"] & (data["labels"] == label)
        count = histogram2d(data["cosines"][mask, 0], data["cosines"][mask, 1], bins)
        validate_histogram(count, f"{population} mean score {SAMPLE_NAMES[label]}")
        summed, _, _ = np.histogram2d(
            data["cosines"][mask, 0],
            data["cosines"][mask, 1],
            bins=bins,
            range=(MAP_RANGE, MAP_RANGE),
            weights=data["scores"][mask],
        )
        mean = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
        if not np.isfinite(mean).all():
            raise RuntimeError(f"{population} mean score {SAMPLE_NAMES[label]} contains NaN or infinity")
        means[label] = mean
        counts[label] = count
    occupied_means = [means[label][counts[label] > 0] for label in (1, 0)]
    vmin = min(float(np.min(value)) for value in occupied_means)
    vmax = max(float(np.max(value)) for value in occupied_means)
    score_cmap = matplotlib.colormaps["magma"].copy()
    score_cmap.set_bad("#d9d9d9")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True)
    image = None
    for axis, label in zip(axes, (1, 0), strict=True):
        visible_mean = np.ma.masked_where(counts[label] == 0, means[label])
        image = draw(axis, visible_mean, bins, score_cmap, Normalize(vmin, vmax))
        decorate(
            axis,
            f"{SAMPLE_NAMES[label]} truth class • occupancy {int(counts[label].min())}–{int(counts[label].max())} • gray=empty",
        )
    assert image is not None
    figure.colorbar(image, ax=axes, label="mean fixed-classifier H-like score")
    figure.suptitle("Mean fixed-classifier score per angular bin", fontsize=13)
    figure.text(0.5, -0.015, footer(population, bins), ha="center", fontsize=7.8)
    save_png(figure, path)
    return {
        SAMPLE_NAMES[label]: {
            "mean": means[label],
            "count": counts[label],
            "empty_bins": int(np.sum(counts[label] == 0)),
            "nan_bins": int(np.sum(~np.isfinite(means[label]))),
        }
        for label in (1, 0)
    }


def plot_scatter(data: Mapping[str, np.ndarray], population: str, path: Path) -> dict[str, int]:
    colors = {1: "#D55E00", 0: "#0072B2"}
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.25), constrained_layout=True, sharex=True, sharey=True)
    counts = {}
    for axis, label in zip(axes, (1, 0), strict=True):
        mask = data["physical"] & (data["labels"] == label)
        values = data["cosines"][mask]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{population} scatter {SAMPLE_NAMES[label]} contains NaN or infinity")
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=5,
            alpha=0.22,
            linewidths=0,
            color=colors[label],
            rasterized=True,
            label=f"{SAMPLE_NAMES[label]} truth class",
        )
        decorate(axis, f"{SAMPLE_NAMES[label]} truth class • N={len(values)}")
        axis.set_xlim(MAP_RANGE)
        axis.set_ylim(MAP_RANGE)
        axis.legend(loc="upper right", frameon=True, fontsize=8, markerscale=2.2)
        counts[SAMPLE_NAMES[label]] = int(len(values))
    figure.suptitle(r"Reco 1p0n $\times$ 1p0n angular scatter", fontsize=13)
    figure.text(
        0.5,
        -0.015,
        f"Simulation samples • {population} • all selected physical-map events • reco collinear charged-track angle proxy",
        ha="center",
        fontsize=7.8,
    )
    save_png(figure, path)
    return counts


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
    if args.bins <= 6:
        raise ValueError("--bins must be finer than the previous 6x6 binning")
    repo = args.repo.resolve()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    device = choose_device(args.device)
    with np.load(args.row_map, allow_pickle=False) as source:
        row_map = {key: np.asarray(source[key]) for key in source.files}
    models, checkpoint_hashes = load_models(repo, processed_dir, args.checkpoint_root.resolve(), device)

    compact_records = {}
    split_counts = {}
    validation_parity = None
    for split in ("train", "validation", "test"):
        inference = infer_split(processed_dir, split, row_map, models, device, args.batch_size)
        if split == "validation":
            split_identity = structured_identity(
                {key: row_map[key][inference["split_mask"]] for key in ("sample_id", "ntuple_file_index", "ntuple_entry")}
            )
            with np.load(args.validation_ensemble, allow_pickle=False) as source:
                reference = {key: np.asarray(source[key]) for key in source.files}
            reference_scores = join_reference_scores(split_identity, reference)
            absolute = np.abs(inference["scores"] - reference_scores)
            validation_parity = {
                "max_abs_difference": float(np.max(absolute)),
                "mean_abs_difference": float(np.mean(absolute)),
                "allclose_atol_5e-4": bool(np.allclose(inference["scores"], reference_scores, rtol=0.0, atol=5.0e-4)),
            }
            if not validation_parity["allclose_atol_5e-4"]:
                raise RuntimeError(f"frozen validation score parity failed: {validation_parity}")
        compact, split_counts[split] = compact_geometry(processed_dir, split, inference)
        compact_records[split] = compact

    del models
    gc.collect()
    populations = {
        "all-data": concatenate([compact_records["train"], compact_records["validation"], compact_records["test"]]),
        "validation-test": concatenate([compact_records["validation"], compact_records["test"]]),
    }
    population_labels = {
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
        "split_counts": split_counts,
        "populations": {},
        "definitions": {
            "selection": "both tau decay-mode raw code 0; exactly one core && passTrkSelector charged track per tau; charge ordered tau- then tau+",
            "angle_proxy": "unchanged collinear charged-track proxy from reco_1p1p_score_maps.py; raw values are not clipped; maps require both values within [-1,1]",
            "score": "arithmetic mean of frozen full-reco seeds 42, 43, and 44 terminal checkpoints; inference only",
            "tertiles": "unit-weight score tertiles defined separately within each truth class and population on the charge-ordered track cohort before collinear validity",
            "normalization": "inclusive and tertile panels are normalized independently to unit sum; mean maps show the raw mean H-like classifier score",
        },
    }
    for key, data in populations.items():
        label = population_labels[key]
        prefix = f"reco-1p0n-x-1p0n-{key}"
        levels, edges = assign_tertiles(data)
        inclusive = plot_inclusive(data, label, args.bins, output_dir / f"{prefix}-inclusive.png")
        tertiles = plot_tertiles(data, label, args.bins, levels, output_dir / f"{prefix}-score-tertiles.png")
        mean_score = plot_mean_score(data, label, args.bins, output_dir / f"{prefix}-mean-score.png")
        scatter = plot_scatter(data, label, output_dir / f"{prefix}-scatter.png")
        manifest["populations"][key] = {
            "events": int(len(data["labels"])),
            "track_cohort": {SAMPLE_NAMES[label_id]: int(np.sum(data["track"] & (data["labels"] == label_id))) for label_id in (1, 0)},
            "physical_map": {SAMPLE_NAMES[label_id]: int(np.sum(data["physical"] & (data["labels"] == label_id))) for label_id in (1, 0)},
            "tertile_edges": {SAMPLE_NAMES[label_id]: edges[label_id] for label_id in (1, 0)},
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
    (output_dir / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "bins_per_axis": args.bins,
        "validation_inference_parity": validation_parity,
        "split_counts": split_counts,
        "population_counts": {
            key: manifest["populations"][key]["physical_map"] for key in populations
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
