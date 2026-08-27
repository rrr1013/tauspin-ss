#!/usr/bin/env python3
"""Decay-mode-pair resolved discrimination of the frozen tauspin classifiers.

The population is the held-out report subset of the fixed-v3 pT-matched
validation surface.  Nothing is trained, calibrated, or re-selected here: the
script joins the frozen adaptive-v1 three-seed ensembles (full reco, tau object
only, event only) to the processed tensors by the exact ROOT identity triple,
splits the events into unordered reconstructed decay-mode pair cells, and
measures the within-cell H-versus-Z AUC of each arm.

The scientific question is whether the within-cell discrimination ordering
tracks the textbook energy-fraction spin analyzing power of the reconstructed
decay modes, which production kinematics has no reason to reproduce.  The
event-only arm is the same-event comparator that carries no per-tau decay
structure.

All design constants below were fixed before any AUC was computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Prespecified design constants.
# ---------------------------------------------------------------------------

SAMPLE_IDS = {"H": 0, "Z": 1}
LABEL_NAMES = {1: "H", 0: "Z"}

MODE_NAMES = {0: "1p0n", 1: "1p1n", 2: "1pXn", 3: "3p0n", 4: "3pXn"}

# Energy-fraction (single-observable) spin analyzing power used only as an
# ordering proxy for the cells.  pi: 1, rho: ~0.46, a1: ~0.02.  1pXn is a
# rho/a1 mixture and is placed with rho in the primary assignment; the
# sensitivity assignment moves it towards a1.
ANALYZING_POWER_PRIMARY = {0: 1.00, 1: 0.46, 2: 0.46, 3: 0.02, 4: 0.02}
ANALYZING_POWER_SENSITIVITY = {0: 1.00, 1: 0.46, 2: 0.20, 3: 0.02, 4: 0.02}

# Post-hoc, added after the primary figures were inspected: how the two tau
# analyzing powers should be combined is itself a physics question.  A pure
# spin-correlation term scales with the product, while a single-tau
# polarisation term scales with the sum or with the stronger side alone.
LINK_FUNCTIONS = {
    "product": (lambda a, b: a * b, "P(a) × P(b)   — spin-correlation-like"),
    "mean": (lambda a, b: 0.5 * (a + b), "[P(a) + P(b)] / 2   — polarisation-like"),
    "max": (lambda a, b: max(a, b), "max[P(a), P(b)]   — stronger side only"),
}
POST_HOC_LINK_NOTE = (
    "Post-hoc: the link-function comparison was added after the prespecified product panel was seen. "
    "It is a description of the observed cell pattern, not a prespecified test."
)

ARMS = ("full_reco", "tau_object_only", "event_only")
ARM_LABELS = {
    "full_reco": "full reco",
    "tau_object_only": "tau objects only",
    "event_only": "event only",
}
ARM_STYLE = {
    "full_reco": {"color": "#1f4e9c", "marker": "o", "linestyle": "-"},
    "tau_object_only": {"color": "#c8641e", "marker": "s", "linestyle": "--"},
    "event_only": {"color": "#3f7d3f", "marker": "^", "linestyle": ":"},
}
KINEMATIC_ARM = "met_only"
KINEMATIC_LABEL = "reco MET (single variable)"

MIN_CLASS_OCCUPANCY = 200
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260828
PERMUTATION_REPLICATES = 1000
PERMUTATION_SEED = 20260829
FIGURE_DPI = 240

COMMON_FOOTER = (
    "Simulation samples • held-out report subset of fixed-v3 pT-matched validation • "
    "unit weight • nominal only • frozen adaptive-v1 three-seed ensembles"
)
CLAIM_BOUNDARY = (
    "Within-cell association between frozen classifier scores and reconstructed decay-mode "
    "categories in selected simulated events. Not a spin fraction, not a causal statement that "
    "the classifier uses spin, and not an information decomposition. Reconstructed decay mode "
    "is not truth decay mode."
)


# ---------------------------------------------------------------------------
# Small numerical helpers.
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average (tie-corrected) ranks, 1-based, without a scipy dependency."""
    values = np.asarray(values)
    sorter = np.argsort(values, kind="stable")
    inverse = np.empty(len(values), dtype=np.intp)
    inverse[sorter] = np.arange(len(values), dtype=np.intp)
    ordered = values[sorter]
    is_new = np.empty(len(values), dtype=bool)
    is_new[0] = True
    if len(values) > 1:
        np.not_equal(ordered[1:], ordered[:-1], out=is_new[1:])
    dense = np.cumsum(is_new)[inverse]
    boundaries = np.append(np.flatnonzero(is_new), len(values))
    return 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1)


def auc_from_scores(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """P(score(H) > score(Z)) + 0.5 P(tie), the Mann-Whitney AUC."""
    positives = int(np.count_nonzero(is_positive))
    negatives = int(len(is_positive) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[is_positive].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def restore(values: np.ndarray, stats_block: dict[str, Any], index: int) -> np.ndarray:
    result = values[..., index].astype(np.float64)
    if bool(stats_block["standardize"][index]):
        result = result * float(stats_block["std"][index]) + float(stats_block["mean"][index])
    return result


# ---------------------------------------------------------------------------
# Frozen-input loading and identity join (contract shared with the 1p1p run).
# ---------------------------------------------------------------------------


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
    """Right-row indices for every left key; any ambiguity fails closed."""
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
    if not np.array_equal(sorted_right[position], left):
        raise RuntimeError(f"{description}: key sets differ")
    return order[position]


def join_frozen_inputs(
    sample_ids: np.ndarray,
    event_numbers: np.ndarray,
    labels: np.ndarray,
    identity_reference: dict[str, np.ndarray],
    ensemble: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Audit the frozen row-map sidecar, then join every arm by ROOT identity.

    The processed shards keep eventNumber only as a diagnostic and it is not
    unique, so it is never a join key.  The shard builder writes validation rows
    in the frozen validation row-map order and the trainer exports that row map
    as this sidecar; every available row diagnostic is checked before the
    sidecar identities are attached.  The score join itself is by the complete
    ROOT identity triple and does not depend on ensemble row order.
    """
    expected_length = len(labels)
    for key in ("labels", "sample_id", "ntuple_file_index", "ntuple_entry", "source_file_index", "source_event_number"):
        if key not in identity_reference:
            raise RuntimeError(f"identity sidecar is missing {key}")
        if len(identity_reference[key]) != expected_length:
            raise RuntimeError("processed and identity-sidecar row counts differ")
    if not np.array_equal(sample_ids.astype(np.uint8), identity_reference["sample_id"].astype(np.uint8)):
        raise RuntimeError("processed sample blocks disagree with frozen identity-sidecar order")
    if not np.array_equal(
        event_numbers.astype(np.uint64), identity_reference["source_event_number"].astype(np.uint64)
    ):
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
    rows = exact_join_indices(reference_keys, ensemble_keys, "identity-sidecar-to-ensemble ROOT join")
    if not np.array_equal(labels.astype(np.uint8), ensemble["labels"][rows].astype(np.uint8)):
        raise RuntimeError("processed labels disagree with ensemble after exact ROOT identity join")
    joined: dict[str, np.ndarray] = {
        "report_mask": ensemble["report_mask"][rows].astype(bool),
        "source_file_index": identity_reference["source_file_index"].astype(np.int64),
    }
    for arm in ARMS:
        joined[arm] = ensemble[f"{arm}_scores"][rows].astype(np.float64)
    return joined


def load_processed(processed_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only the validation event-level blocks needed by this run."""
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    stats = json.loads((processed_dir / "stats.json").read_text())
    fields = ("event_features", "tau_features", "tau_decay_mode", "labels", "event_numbers")
    accumulators: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    sample_ids: list[np.ndarray] = []
    shard_paths: list[Path] = []
    for sample in ("H", "Z"):
        for record in metadata["shards"]["validation"][sample]:
            path = processed_dir / record["path"]
            shard_paths.append(path)
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for key in fields:
                accumulators[key].append(shard[key].numpy())
            sample_ids.append(np.full(len(shard["labels"]), SAMPLE_IDS[sample], dtype=np.uint8))
    data = {key: np.concatenate(values) for key, values in accumulators.items()}
    data["sample_ids"] = np.concatenate(sample_ids)
    return data, {"metadata": metadata, "stats": stats, "shard_paths": shard_paths}


# ---------------------------------------------------------------------------
# Decay-mode cells.
# ---------------------------------------------------------------------------


def raw_decay_modes(decay_mode_ids: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Map the processed categorical ids back to raw reco decay-mode codes.

    Unknown ids become -1 and are excluded from the primary population.
    """
    id_to_raw = {int(value): int(key) for key, value in metadata["tau_decay_mode_to_id"].items()}
    unknown_id = int(metadata["tau_decay_unknown_id"])
    if unknown_id in id_to_raw:
        raise RuntimeError("the unknown decay-mode id collides with a real mode id")
    lookup = np.full(int(metadata["tau_decay_num_embeddings"]), -1, dtype=np.int64)
    for value, raw in id_to_raw.items():
        lookup[value] = raw
    if np.any(decay_mode_ids < 0) or np.any(decay_mode_ids >= len(lookup)):
        raise RuntimeError("decay-mode id outside the declared embedding range")
    return lookup[decay_mode_ids]


def cell_indices(raw_modes: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Unordered mode-pair cell id per event, plus the ordered cell key list."""
    keys = [(a, b) for a in sorted(MODE_NAMES) for b in sorted(MODE_NAMES) if a <= b]
    lookup = {key: index for index, key in enumerate(keys)}
    low = np.minimum(raw_modes[:, 0], raw_modes[:, 1])
    high = np.maximum(raw_modes[:, 0], raw_modes[:, 1])
    cells = np.full(len(raw_modes), -1, dtype=np.int64)
    known = (low >= 0) & (high >= 0)
    for key, index in lookup.items():
        cells[known & (low == key[0]) & (high == key[1])] = index
    if np.any(known & (cells < 0)):
        raise RuntimeError("a known mode pair did not map onto a cell")
    return cells, keys


def cell_name(key: tuple[int, int]) -> str:
    return f"{MODE_NAMES[key[0]]} x {MODE_NAMES[key[1]]}"


def power_product(key: tuple[int, int], table: dict[int, float]) -> float:
    return table[key[0]] * table[key[1]]


def power_link(key: tuple[int, int], table: dict[int, float], link: str) -> float:
    combine, _ = LINK_FUNCTIONS[link]
    return float(combine(table[key[0]], table[key[1]]))


# ---------------------------------------------------------------------------
# Cluster bootstrap machinery.
# ---------------------------------------------------------------------------


class ClusterResampler:
    """Resample source-ROOT-file clusters with replacement, within class."""

    def __init__(self, cluster_ids: np.ndarray, is_positive: np.ndarray) -> None:
        self.blocks = []
        for positive in (True, False):
            rows = np.flatnonzero(is_positive == positive)
            order = rows[np.argsort(cluster_ids[rows], kind="stable")]
            clusters = cluster_ids[order]
            starts = np.flatnonzero(np.append(True, clusters[1:] != clusters[:-1]))
            lengths = np.diff(np.append(starts, len(order)))
            self.blocks.append((order, starts.astype(np.int64), lengths.astype(np.int64)))

    def draw(self, generator: np.random.Generator) -> np.ndarray:
        picked = []
        for order, starts, lengths in self.blocks:
            chosen = generator.integers(0, len(starts), size=len(starts))
            chosen_starts = starts[chosen]
            chosen_lengths = lengths[chosen]
            total = int(chosen_lengths.sum())
            base = np.repeat(chosen_starts - np.append(0, np.cumsum(chosen_lengths)[:-1]), chosen_lengths)
            picked.append(order[np.arange(total, dtype=np.int64) + base])
        return np.concatenate(picked)


def per_cell_auc(
    cells: np.ndarray,
    is_positive: np.ndarray,
    arm_scores: dict[str, np.ndarray],
    n_cells: int,
) -> dict[str, np.ndarray]:
    """AUC per cell per arm on one (possibly resampled) event set."""
    order = np.argsort(cells, kind="stable")
    ordered_cells = cells[order]
    starts = np.flatnonzero(np.append(True, ordered_cells[1:] != ordered_cells[:-1]))
    bounds = np.append(starts, len(order))
    result = {arm: np.full(n_cells, np.nan) for arm in arm_scores}
    for index, start in enumerate(starts):
        stop = bounds[index + 1]
        cell = int(ordered_cells[start])
        rows = order[start:stop]
        positive = is_positive[rows]
        for arm, scores in arm_scores.items():
            result[arm][cell] = auc_from_scores(scores[rows], positive)
    return result


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    figure.text(0.5, 0.005, COMMON_FOOTER, ha="center", va="bottom", fontsize=7, color="#444444")
    paths = {}
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        paths[suffix] = path.name
    plt.close(figure)
    return paths


def annotate_points(axis: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: list[str]) -> None:
    """Label a scatter without letting neighbouring cells overwrite each other."""
    order = np.argsort(xs)
    offsets = ((7, 6), (7, -11), (-7, 6), (-7, -11))
    alignments = ("left", "left", "right", "right")
    for rank, index in enumerate(order):
        slot = rank % len(offsets)
        axis.annotate(
            labels[index],
            (xs[index], ys[index]),
            textcoords="offset points",
            xytext=offsets[slot],
            ha=alignments[slot],
            fontsize=7,
            color="#333333",
        )


def plot_occupancy(output_dir: Path, counts: dict[str, np.ndarray], keys: list[tuple[int, int]]) -> dict[str, str]:
    modes = sorted(MODE_NAMES)
    grid = {name: np.full((len(modes), len(modes)), np.nan) for name in ("H", "Z")}
    for index, key in enumerate(keys):
        for name in ("H", "Z"):
            grid[name][key[0], key[1]] = counts[name][index]
            grid[name][key[1], key[0]] = counts[name][index]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for axis, name in zip(axes[:2], ("H", "Z"), strict=True):
        image = axis.imshow(grid[name], origin="lower", cmap="viridis")
        axis.set_title(f"{name} events per mode-pair cell")
        figure.colorbar(image, ax=axis, fraction=0.046, label="events")
        for i in modes:
            for j in modes:
                value = grid[name][i, j]
                if np.isfinite(value):
                    axis.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=7, color="w")
    ratio = np.where(grid["Z"] > 0, grid["H"] / np.maximum(grid["Z"], 1), np.nan)
    image = axes[2].imshow(ratio, origin="lower", cmap="coolwarm", vmin=0.6, vmax=1.4)
    axes[2].set_title("H / Z occupancy ratio")
    figure.colorbar(image, ax=axes[2], fraction=0.046, label="ratio")
    for i in modes:
        for j in modes:
            if np.isfinite(ratio[i, j]):
                axes[2].text(j, i, f"{ratio[i, j]:.2f}", ha="center", va="center", fontsize=7)
    for axis in axes:
        axis.set_xticks(modes, [MODE_NAMES[m] for m in modes], rotation=45)
        axis.set_yticks(modes, [MODE_NAMES[m] for m in modes])
        axis.set_xlabel("reco decay mode, tau b")
        axis.set_ylabel("reco decay mode, tau a")
    figure.suptitle("Reconstructed decay-mode pair occupancy (unordered cells, symmetrised for display)")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return save_figure(figure, output_dir, "modepair-occupancy")


def plot_forest(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    shown = [row for row in rows if not row["masked"]]
    shown = sorted(shown, key=lambda row: row["power_product_primary"])
    positions = np.arange(len(shown))
    figure, axis = plt.subplots(figsize=(9.5, 0.62 * len(shown) + 3.0))
    offsets = {"full_reco": 0.24, "tau_object_only": 0.0, "event_only": -0.24}
    for arm in ARMS:
        style = ARM_STYLE[arm]
        centres = np.array([row["auc"][arm]["estimate"] for row in shown])
        low = np.array([row["auc"][arm]["ci"][0] for row in shown])
        high = np.array([row["auc"][arm]["ci"][1] for row in shown])
        axis.errorbar(
            centres,
            positions + offsets[arm],
            xerr=np.vstack([centres - low, high - centres]),
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            linestyle="none",
            label=ARM_LABELS[arm],
        )
    kinematic = np.array([row["auc"][KINEMATIC_ARM]["estimate"] for row in shown])
    axis.plot(kinematic, positions, "x", color="#7a7a7a", markersize=7, label=KINEMATIC_LABEL)
    axis.axvline(0.5, color="k", linewidth=0.9, linestyle="-")
    axis.set_yticks(
        positions,
        [f"{row['cell']}  (P·P = {row['power_product_primary']:.3g}, N = {row['n_total']})" for row in shown],
    )
    axis.set_xlabel("within-cell AUC, H versus Z  (score ordering, unit weight)")
    axis.set_ylabel("reco decay-mode pair cell, ordered by analyzing-power product")
    axis.set_title("Within-cell discrimination of the frozen arms, by reconstructed decay-mode pair")
    axis.grid(axis="x", alpha=0.3)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(figure, output_dir, "modepair-auc-forest")


def plot_delta_vs_power(output_dir: Path, rows: list[dict[str, Any]], spearman: dict[str, Any]) -> dict[str, str]:
    shown = [row for row in rows if not row["masked"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for axis, table_key, title in zip(
        axes,
        ("power_product_primary", "power_product_sensitivity"),
        ("primary analyzing-power assignment", "sensitivity: 1pXn moved towards a1"),
        strict=True,
    ):
        x = np.array([row[table_key] for row in shown])
        y = np.array([row["delta_full_minus_event"]["estimate"] for row in shown])
        low = np.array([row["delta_full_minus_event"]["ci"][0] for row in shown])
        high = np.array([row["delta_full_minus_event"]["ci"][1] for row in shown])
        sizes = np.array([row["n_total"] for row in shown], dtype=float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="none",
            ecolor="#888888",
            elinewidth=1.2,
            capsize=3,
            zorder=1,
        )
        scatter = axis.scatter(
            x, y, s=18 + 130 * sizes / sizes.max(), c=np.log10(sizes), cmap="cividis", zorder=2, edgecolor="k",
            linewidth=0.4,
        )
        annotate_points(axis, x, y, [row["cell"] for row in shown])
        axis.axhline(0.0, color="k", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_xlabel("analyzing-power product  P(mode a) × P(mode b)   [log scale]")
        axis.set_ylabel("ΔAUC = full reco − event only  (paired, same cell)")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        figure.colorbar(scatter, ax=axis, fraction=0.046, label="log10 cell events")
    primary = spearman["primary"]
    axes[0].text(
        0.03,
        0.96,
        f"Spearman ρ = {primary['estimate']:.3f}\n95% cluster bootstrap [{primary['ci'][0]:.3f}, {primary['ci'][1]:.3f}]",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )
    sensitivity = spearman["sensitivity"]
    axes[1].text(
        0.03,
        0.96,
        f"Spearman ρ = {sensitivity['estimate']:.3f}\n95% cluster bootstrap "
        f"[{sensitivity['ci'][0]:.3f}, {sensitivity['ci'][1]:.3f}]",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )
    figure.suptitle("Per-cell gain of the decay-aware arm over the event-only arm against analyzing power")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_figure(figure, output_dir, "modepair-delta-vs-power")


def plot_link_functions(
    output_dir: Path, rows: list[dict[str, Any]], link_spearman: dict[str, dict[str, Any]]
) -> dict[str, str]:
    shown = [row for row in rows if not row["masked"]]
    figure, axes = plt.subplots(1, len(LINK_FUNCTIONS), figsize=(5.4 * len(LINK_FUNCTIONS), 5.2), sharey=True)
    for axis, link in zip(axes, LINK_FUNCTIONS, strict=True):
        _, description = LINK_FUNCTIONS[link]
        x = np.array([row["power_link"][link] for row in shown])
        y = np.array([row["delta_full_minus_event"]["estimate"] for row in shown])
        low = np.array([row["delta_full_minus_event"]["ci"][0] for row in shown])
        high = np.array([row["delta_full_minus_event"]["ci"][1] for row in shown])
        sizes = np.array([row["n_total"] for row in shown], dtype=float)
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", ecolor="#999999", capsize=3, zorder=1)
        axis.scatter(
            x, y, s=18 + 130 * sizes / sizes.max(), c="#1f4e9c", zorder=2, edgecolor="k", linewidth=0.4, alpha=0.85
        )
        annotate_points(axis, x, y, [row["cell"] for row in shown])
        axis.axhline(0.0, color="k", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_xlabel(f"combined analyzing power\n{description}")
        axis.grid(alpha=0.3)
        entry = link_spearman[link]
        axis.set_title(
            f"Spearman ρ = {entry['estimate']:.3f}\n95% cluster bootstrap "
            f"[{entry['ci'][0]:.3f}, {entry['ci'][1]:.3f}]",
            fontsize=10,
        )
    axes[0].set_ylabel("ΔAUC = full reco − event only  (paired, same cell)")
    figure.suptitle("How the two tau analyzing powers combine: product against sum-like alternatives")
    figure.text(0.5, 0.028, POST_HOC_LINK_NOTE, ha="center", fontsize=7.5, color="#8a4b12")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    return save_figure(figure, output_dir, "modepair-link-functions")


def plot_score_distributions(
    output_dir: Path,
    cells: np.ndarray,
    is_positive: np.ndarray,
    scores: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    shown = [row for row in rows if not row["masked"]]
    ordered = sorted(shown, key=lambda row: -row["power_product_primary"])
    selected = ordered[:3] + ordered[-1:]
    edges = np.linspace(0.0, 1.0, 41)
    figure, axes = plt.subplots(2, len(selected), figsize=(4.1 * len(selected), 7.2), sharex=True)
    for column, row in enumerate(selected):
        mask = cells == row["cell_index"]
        for axis_row, arm in enumerate(("full_reco", "event_only")):
            axis = axes[axis_row, column]
            for label_value, name, colour, style in ((True, "H", "#1f4e9c", "-"), (False, "Z", "#c8641e", "--")):
                values = scores[arm][mask & (is_positive == label_value)]
                axis.hist(
                    values,
                    bins=edges,
                    histtype="step",
                    density=True,
                    color=colour,
                    linestyle=style,
                    linewidth=1.6,
                    label=f"{name} (N = {len(values)})",
                )
            axis.set_title(
                f"{row['cell']}\n{ARM_LABELS[arm]}, AUC = {row['auc'][arm]['estimate']:.4f}", fontsize=9
            )
            axis.legend(fontsize=7)
            axis.grid(alpha=0.3)
            if column == 0:
                axis.set_ylabel("normalised events / bin")
            if axis_row == 1:
                axis.set_xlabel("H-like ensemble score")
    figure.suptitle("Within-cell score shapes: three highest analyzing-power cells and the lowest")
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_figure(figure, output_dir, "modepair-score-distributions")


def plot_confounds(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    shown = [row for row in rows if not row["masked"]]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    panels = (
        ("mean_tau_pt_gev", "mean reconstructed tau pT [GeV]"),
        ("mean_met_gev", "mean reconstructed MET [GeV]"),
        ("auc_met_only", "within-cell AUC of reco MET alone"),
    )
    for axis, (key, xlabel) in zip(axes, panels, strict=True):
        x = np.array([row[key] for row in shown])
        y = np.array([row["auc"]["full_reco"]["estimate"] for row in shown])
        low = np.array([row["auc"]["full_reco"]["ci"][0] for row in shown])
        high = np.array([row["auc"]["full_reco"]["ci"][1] for row in shown])
        colours = np.log10([row["power_product_primary"] for row in shown])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="none", ecolor="#999999", capsize=3, zorder=1)
        scatter = axis.scatter(x, y, c=colours, cmap="plasma", s=60, edgecolor="k", linewidth=0.4, zorder=2)
        annotate_points(axis, x, y, [row["cell"] for row in shown])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("within-cell full-reco AUC")
        axis.grid(alpha=0.3)
        figure.colorbar(scatter, ax=axis, fraction=0.046, label="log10 P·P")
    figure.suptitle("Competing explanations for the cell ordering: cell kinematics against full-reco AUC")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return save_figure(figure, output_dir, "modepair-confounds")


def plot_spread_null(output_dir: Path, null: dict[str, Any]) -> dict[str, str]:
    figure, axes = plt.subplots(1, len(ARMS), figsize=(5.0 * len(ARMS), 4.4), sharey=True)
    for axis, arm in zip(axes, ARMS, strict=True):
        samples = np.asarray(null["samples"][arm])
        observed = float(null["observed"][arm])
        axis.hist(samples, bins=40, color="#b8c4d9", edgecolor="#5a6b8c")
        axis.axvline(observed, color="#c0392b", linewidth=2.0, label=f"observed = {observed:.4f}")
        quantile = float(null["p_ge_observed"][arm])
        axis.set_title(f"{ARM_LABELS[arm]}\nfraction of null ≥ observed = {quantile:.3f}", fontsize=10)
        axis.set_xlabel("occupancy-weighted spread of cell AUC")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(f"permutation replicates ({null['replicates']})")
    figure.suptitle("Is the cell-to-cell AUC spread larger than reshuffling the cell labels within class?")
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_figure(figure, output_dir, "modepair-spread-null")


# ---------------------------------------------------------------------------
# Statistics assembly.
# ---------------------------------------------------------------------------


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = average_ranks(np.asarray(x, dtype=float))
    ry = average_ranks(np.asarray(y, dtype=float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denominator)


def weighted_spread(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & (weights > 0)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    v = values[finite]
    w = weights[finite].astype(float)
    mean = float((w * v).sum() / w.sum())
    return float(np.sqrt((w * (v - mean) ** 2).sum() / w.sum()))


def percentile_interval(samples: np.ndarray) -> list[float]:
    finite = samples[np.isfinite(samples)]
    if len(finite) < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    if sha256_file(ensemble_path) != comparison["ensemble_artifact"]["sha256"]:
        raise RuntimeError("ensemble hash disagrees with frozen comparison summary")
    report = joined["report_mask"]
    if int(report.sum()) != int(comparison["report_partition"]["events"]):
        raise RuntimeError("joined report mask count disagrees with frozen comparison summary")
    for label_text, expected in comparison["report_partition"]["class_counts"].items():
        label = int(label_text)
        if int(np.sum(report & (data["labels"] == label))) != int(expected):
            raise RuntimeError(f"report-mask class count for label {label} disagrees with the frozen summary")

    # Closure check: the whole-report AUC of each arm must reproduce the frozen
    # comparison summary.  This is the strongest available end-to-end check that
    # the identity join, the report mask, and the score columns line up.
    is_positive_report = data["labels"] == 1
    recomputed_global = {
        arm: float(auc_from_scores(joined[arm][report], is_positive_report[report])) for arm in ARMS
    }
    for arm, frozen_value in comparison["terminal_ensemble_auc"].items():
        if abs(recomputed_global[arm] - float(frozen_value)) > 1e-9:
            raise RuntimeError(
                f"recomputed report AUC for {arm} ({recomputed_global[arm]!r}) disagrees with the frozen summary"
            )

    raw_modes = raw_decay_modes(data["tau_decay_mode"], specs["metadata"])
    cells_all, keys = cell_indices(raw_modes)
    known_modes = cells_all >= 0
    primary = report & known_modes
    is_positive_all = data["labels"] == 1

    counts = {
        "report": {name: int(np.sum(report & (data["labels"] == label))) for label, name in LABEL_NAMES.items()},
        "unknown_mode_dropped": {
            name: int(np.sum(report & ~known_modes & (data["labels"] == label)))
            for label, name in LABEL_NAMES.items()
        },
        "primary": {name: int(np.sum(primary & (data["labels"] == label))) for label, name in LABEL_NAMES.items()},
        "per_cell": {
            cell_name(key): {
                name: int(np.sum(primary & (cells_all == index) & (data["labels"] == label)))
                for label, name in LABEL_NAMES.items()
            }
            for index, key in enumerate(keys)
        },
    }
    if args.counts_only:
        print(json.dumps({"min_class_occupancy": MIN_CLASS_OCCUPANCY, "counts": counts}, indent=2))
        return 0

    if args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --counts-only is used")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rows_index = np.flatnonzero(primary)
    cells = cells_all[rows_index]
    is_positive = is_positive_all[rows_index]
    clusters = joined["source_file_index"][rows_index]
    met_gev = np.expm1(restore(data["event_features"], specs["stats"]["event"], 0))[rows_index]
    tau_pt_gev = np.expm1(restore(data["tau_features"], specs["stats"]["tau"], 0))[rows_index]

    arm_scores = {arm: joined[arm][rows_index] for arm in ARMS}
    arm_scores[KINEMATIC_ARM] = met_gev
    n_cells = len(keys)

    observed = per_cell_auc(cells, is_positive, arm_scores, n_cells)

    cell_counts = {
        "H": np.array([int(np.sum((cells == index) & is_positive)) for index in range(n_cells)]),
        "Z": np.array([int(np.sum((cells == index) & ~is_positive)) for index in range(n_cells)]),
    }
    masked = (cell_counts["H"] < MIN_CLASS_OCCUPANCY) | (cell_counts["Z"] < MIN_CLASS_OCCUPANCY)
    unmasked_index = np.flatnonzero(~masked)

    resampler = ClusterResampler(clusters, is_positive)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = int(args.bootstrap_replicates)
    boot = {arm: np.full((replicates, n_cells), np.nan) for arm in arm_scores}
    boot_delta = np.full((replicates, n_cells), np.nan)
    boot_spearman = {"primary": np.full(replicates, np.nan), "sensitivity": np.full(replicates, np.nan)}
    power_primary = np.array([power_product(key, ANALYZING_POWER_PRIMARY) for key in keys])
    power_sensitivity = np.array([power_product(key, ANALYZING_POWER_SENSITIVITY) for key in keys])
    power_links = {
        link: np.array([power_link(key, ANALYZING_POWER_PRIMARY, link) for key in keys]) for link in LINK_FUNCTIONS
    }
    boot_link_spearman = {link: np.full(replicates, np.nan) for link in LINK_FUNCTIONS}
    for replicate in range(replicates):
        picked = resampler.draw(generator)
        values = per_cell_auc(
            cells[picked], is_positive[picked], {arm: scores[picked] for arm, scores in arm_scores.items()}, n_cells
        )
        for arm in arm_scores:
            boot[arm][replicate] = values[arm]
        delta = values["full_reco"] - values["event_only"]
        boot_delta[replicate] = delta
        usable = unmasked_index[np.isfinite(delta[unmasked_index])]
        boot_spearman["primary"][replicate] = spearman_rho(power_primary[usable], delta[usable])
        boot_spearman["sensitivity"][replicate] = spearman_rho(power_sensitivity[usable], delta[usable])
        for link, values_link in power_links.items():
            boot_link_spearman[link][replicate] = spearman_rho(values_link[usable], delta[usable])

    observed_delta = observed["full_reco"] - observed["event_only"]
    observed_spearman = {
        "primary": spearman_rho(power_primary[unmasked_index], observed_delta[unmasked_index]),
        "sensitivity": spearman_rho(power_sensitivity[unmasked_index], observed_delta[unmasked_index]),
    }
    spearman = {
        key: {"estimate": observed_spearman[key], "ci": percentile_interval(boot_spearman[key])}
        for key in observed_spearman
    }
    link_spearman = {
        link: {
            "estimate": spearman_rho(power_links[link][unmasked_index], observed_delta[unmasked_index]),
            "ci": percentile_interval(boot_link_spearman[link]),
            "description": LINK_FUNCTIONS[link][1],
        }
        for link in LINK_FUNCTIONS
    }

    # Permutation null on the spread of cell AUC: shuffle the cell labels within
    # truth class, keeping the scores and the class labels fixed.
    null_generator = np.random.default_rng(PERMUTATION_SEED)
    null_samples = {arm: np.full(int(args.permutation_replicates), np.nan) for arm in ARMS}
    weights = (cell_counts["H"] + cell_counts["Z"]).astype(float)
    weights_masked = np.where(masked, 0.0, weights)
    observed_spread = {
        arm: weighted_spread(observed[arm], weights_masked) for arm in ARMS
    }
    positive_rows = np.flatnonzero(is_positive)
    negative_rows = np.flatnonzero(~is_positive)
    for replicate in range(int(args.permutation_replicates)):
        shuffled = cells.copy()
        shuffled[positive_rows] = cells[null_generator.permutation(positive_rows)]
        shuffled[negative_rows] = cells[null_generator.permutation(negative_rows)]
        values = per_cell_auc(shuffled, is_positive, {arm: arm_scores[arm] for arm in ARMS}, n_cells)
        for arm in ARMS:
            null_samples[arm][replicate] = weighted_spread(values[arm], weights_masked)
    null = {
        "replicates": int(args.permutation_replicates),
        "seed": PERMUTATION_SEED,
        "statistic": "occupancy-weighted standard deviation of within-cell AUC over unmasked cells",
        "observed": observed_spread,
        "samples": {arm: null_samples[arm].tolist() for arm in ARMS},
        "p_ge_observed": {
            arm: float((1.0 + np.count_nonzero(null_samples[arm] >= observed_spread[arm])) / (len(null_samples[arm]) + 1.0))
            for arm in ARMS
        },
    }

    rows: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        mask = cells == index
        entry: dict[str, Any] = {
            "cell_index": index,
            "cell": cell_name(key),
            "mode_a": MODE_NAMES[key[0]],
            "mode_b": MODE_NAMES[key[1]],
            "n_H": int(cell_counts["H"][index]),
            "n_Z": int(cell_counts["Z"][index]),
            "n_total": int(cell_counts["H"][index] + cell_counts["Z"][index]),
            "masked": bool(masked[index]),
            "power_product_primary": float(power_primary[index]),
            "power_product_sensitivity": float(power_sensitivity[index]),
            "power_link": {link: float(power_links[link][index]) for link in LINK_FUNCTIONS},
            "mean_tau_pt_gev": float(np.mean(tau_pt_gev[mask])) if np.any(mask) else float("nan"),
            "mean_met_gev": float(np.mean(met_gev[mask])) if np.any(mask) else float("nan"),
            "auc": {
                arm: {
                    "estimate": float(observed[arm][index]),
                    "ci": percentile_interval(boot[arm][:, index]),
                }
                for arm in arm_scores
            },
            "delta_full_minus_event": {
                "estimate": float(observed_delta[index]),
                "ci": percentile_interval(boot_delta[:, index]),
            },
        }
        entry["auc_met_only"] = entry["auc"][KINEMATIC_ARM]["estimate"]
        rows.append(entry)

    figures = {
        "occupancy": plot_occupancy(output_dir, cell_counts, keys),
        "auc_forest": plot_forest(output_dir, rows),
        "delta_vs_power": plot_delta_vs_power(output_dir, rows, spearman),
        "link_functions": plot_link_functions(output_dir, rows, link_spearman),
        "score_distributions": plot_score_distributions(output_dir, cells, is_positive, arm_scores, rows),
        "confounds": plot_confounds(output_dir, rows),
        "spread_null": plot_spread_null(output_dir, null),
    }

    flat_rows = []
    for entry in rows:
        flat: dict[str, Any] = {
            key: entry[key]
            for key in (
                "cell",
                "mode_a",
                "mode_b",
                "n_H",
                "n_Z",
                "n_total",
                "masked",
                "power_product_primary",
                "power_product_sensitivity",
                "mean_tau_pt_gev",
                "mean_met_gev",
            )
        }
        for link in LINK_FUNCTIONS:
            flat[f"power_link_{link}"] = entry["power_link"][link]
        for arm in arm_scores:
            flat[f"auc_{arm}"] = entry["auc"][arm]["estimate"]
            flat[f"auc_{arm}_lo"] = entry["auc"][arm]["ci"][0]
            flat[f"auc_{arm}_hi"] = entry["auc"][arm]["ci"][1]
        flat["delta_full_minus_event"] = entry["delta_full_minus_event"]["estimate"]
        flat["delta_full_minus_event_lo"] = entry["delta_full_minus_event"]["ci"][0]
        flat["delta_full_minus_event_hi"] = entry["delta_full_minus_event"]["ci"][1]
        flat_rows.append(flat)
    write_csv(output_dir / "cell-summary.csv", flat_rows)
    np.savez_compressed(
        output_dir / "cell-data.npz",
        cell_index=np.arange(n_cells),
        n_H=cell_counts["H"],
        n_Z=cell_counts["Z"],
        masked=masked,
        power_primary=power_primary,
        power_sensitivity=power_sensitivity,
        **{f"auc_{arm}": observed[arm] for arm in arm_scores},
        **{f"boot_auc_{arm}": boot[arm] for arm in arm_scores},
        boot_delta=boot_delta,
        boot_spearman_primary=boot_spearman["primary"],
        boot_spearman_sensitivity=boot_spearman["sensitivity"],
    )

    input_hashes = {
        "metadata.json": sha256_file(processed_dir / "metadata.json"),
        "stats.json": sha256_file(processed_dir / "stats.json"),
        "terminal-ensembles.npz": sha256_file(ensemble_path),
        "identity-reference.npz": sha256_file(identity_path),
        "comparison-summary.json": sha256_file(comparison_path),
    }
    for path in specs["shard_paths"]:
        input_hashes[str(path.relative_to(processed_dir))] = sha256_file(path)

    manifest = {
        "format_version": 1,
        "status": "completed",
        "primary_estimand": (
            "within-cell unit-weight H-versus-Z AUC of each frozen arm on the held-out report subset "
            "of the fixed-v3 pT-matched validation surface"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "population": {
            "partition": "held-out report subset of validation",
            "counts": counts,
            "weights": "unit weight; no parent-overlap weight",
        },
        "definition": {
            "cell": "unordered pair of reconstructed tau decay modes; tau-side ordering is not retained",
            "modes": MODE_NAMES,
            "analyzing_power_primary": ANALYZING_POWER_PRIMARY,
            "analyzing_power_sensitivity": ANALYZING_POWER_SENSITIVITY,
            "analyzing_power_note": (
                "energy-fraction single-observable analyzing power used only as an ordering proxy; "
                "a network that sees tracks and PFOs is not bounded by it"
            ),
            "auc": "Mann-Whitney AUC with tie-corrected average ranks; H is the positive class",
            "min_class_occupancy": MIN_CLASS_OCCUPANCY,
            "kinematic_reference": "within-cell AUC of reconstructed MET as a single variable",
        },
        "identity_join": {
            "ensemble_key": ["sample_id", "ntuple_file_index", "ntuple_entry"],
            "event_number_caveat": "source_event_number is diagnostic and non-unique, so it is never a join key",
            "semantics": (
                "missing, duplicate, or label-mismatched rows fail closed; ensemble row order is not used"
            ),
        },
        "uncertainty": {
            "bootstrap": {
                "replicates": replicates,
                "seed": BOOTSTRAP_SEED,
                "unit": "source_file_index cluster, resampled with replacement independently within truth class",
                "pairing": "all arms are recomputed on the same resampled events, so arm differences are paired",
                "interval": "percentile 2.5 / 97.5 of the replicate distribution",
            },
            "permutation_null": {
                key: null[key] for key in ("replicates", "seed", "statistic", "observed", "p_ge_observed")
            },
        },
        "spearman_delta_versus_power": spearman,
        "spearman_delta_versus_link_function": {"note": POST_HOC_LINK_NOTE, **link_spearman},
        "cells": rows,
        "figures": figures,
        "tables": ["cell-summary.csv", "cell-data.npz"],
        "limitations": [
            "Reconstructed decay mode is not truth decay mode; 1p1n and 1pXn migration is not corrected.",
            "The event-only arm is a same-schedule comparator, not a sufficient statistic for production kinematics.",
            "Cells are unordered mode pairs, so a tau-charge asymmetry would be averaged away.",
            "The analyzing-power values are textbook energy-fraction values, not a fit to this sample.",
            "Fifteen cells are inspected; the intervals are not adjusted for multiplicity.",
            "No parent pT-|eta| overlap weight is applied and the H/Z mass supports remain non-overlapping.",
        ],
        "lineage": {
            "ensemble_artifact_sha256": comparison["ensemble_artifact"]["sha256"],
            "comparison_protocol_sha256": comparison["protocol_sha256"],
            "comparison_claim_boundary": comparison["claim_boundary"],
            "frozen_terminal_ensemble_auc": comparison["terminal_ensemble_auc"],
            "recomputed_terminal_ensemble_auc": recomputed_global,
            "recomputation_note": (
                "the whole-report-subset AUC of every arm is recomputed here and required to agree with the "
                "frozen comparison summary, which closes the identity join and the report mask"
            ),
        },
        "input_sha256": input_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifact_hashes = {
        path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.name != "artifact-sha256.json"
    }
    (output_dir / "artifact-sha256.json").write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "cells": len(rows), "masked": int(masked.sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
