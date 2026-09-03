#!/usr/bin/env python3
"""Diagnose pi0 energy/direction effects on thesis-defined 1p1n observables.

The calculation is tau-level and deliberately does not train or evaluate a
classifier.  It uses the canonical validation rows, truth 1p1n tau objects,
reco tau decayMode==1, and the frozen same-tau nearest isPi0-PFO matching
convention used by the preceding pi0 matching audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


FORMAT_VERSION = 1
VALIDATION_SPLIT_ID = 1
PRIMARY_DR = 0.05
COS_TOLERANCE = 5.0e-5
MAP_RANGE = (-1.0, 1.0)
MAP_BINS = 40
FIGURE_DPI = 240

# Constants used by the existing thesis-definition implementations.
TAU_MASS_GEV = 1.77686
RHO_MASS_GEV = 0.778
CHARGED_PION_MASS_GEV = 0.13957039

# This is used only to make the two hybrid pi0 four-vectors physical.  It is
# not a replacement observable definition or a truth/reco mass measurement.
NEUTRAL_PION_MASS_GEV = 0.1349768

ROLE_CHARGED_PION = 1
ROLE_NEUTRAL_PION = 2
ROLE_TAU_NEUTRINO = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=MAP_BINS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def delta_phi(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    return np.arctan2(np.sin(np.asarray(left) - np.asarray(right)), np.cos(np.asarray(left) - np.asarray(right)))


def thesis_cospsi(charged_four: np.ndarray, neutral_four: np.ndarray) -> np.ndarray:
    """Oshihara thesis Eq. 3.5, as used by the existing implementation."""
    denominator_momentum = np.linalg.norm(
        charged_four[..., :3] + neutral_four[..., :3], axis=-1
    )
    mass_factor_squared = RHO_MASS_GEV**2 - 4.0 * CHARGED_PION_MASS_GEV**2
    factor = RHO_MASS_GEV / math.sqrt(mass_factor_squared)
    return factor * (charged_four[..., 3] - neutral_four[..., 3]) / denominator_momentum


def thesis_costheta(tau_four: np.ndarray, rho_four: np.ndarray) -> np.ndarray:
    """Oshihara thesis Eq. 3.4 for tau -> rho nu, as already implemented."""
    tau_energy = tau_four[..., 3]
    rho_energy = rho_four[..., 3]
    x = rho_energy / tau_energy
    a_squared = (RHO_MASS_GEV / TAU_MASS_GEV) ** 2
    beta = np.sqrt(1.0 - (TAU_MASS_GEV / tau_energy) ** 2)
    return (2.0 * x - 1.0 - a_squared) / (beta * (1.0 - a_squared))


def validate_thesis_formulas() -> None:
    """Run the same synthetic consistency checks used by the existing maps."""
    beta_tau = 0.82
    gamma_tau = 1.0 / math.sqrt(1.0 - beta_tau**2)
    rho_energy_star = (TAU_MASS_GEV**2 + RHO_MASS_GEV**2) / (2.0 * TAU_MASS_GEV)
    rho_momentum_star = (TAU_MASS_GEV**2 - RHO_MASS_GEV**2) / (2.0 * TAU_MASS_GEV)
    expected_theta = np.asarray([-1.0, -0.45, 0.0, 0.3, 1.0])
    tau_four = np.zeros((len(expected_theta), 4), dtype=np.float64)
    tau_four[:, 2] = gamma_tau * beta_tau * TAU_MASS_GEV
    tau_four[:, 3] = gamma_tau * TAU_MASS_GEV
    rho_four = np.zeros_like(tau_four)
    rho_four[:, 0] = rho_momentum_star * np.sqrt(1.0 - expected_theta**2)
    rho_four[:, 2] = gamma_tau * (
        rho_momentum_star * expected_theta + beta_tau * rho_energy_star
    )
    rho_four[:, 3] = gamma_tau * (
        rho_energy_star + beta_tau * rho_momentum_star * expected_theta
    )
    if float(np.max(np.abs(thesis_costheta(tau_four, rho_four) - expected_theta))) > 1.0e-12:
        raise RuntimeError("Eq. 3.4 synthetic validation failed")

    p_star = math.sqrt(RHO_MASS_GEV**2 / 4.0 - CHARGED_PION_MASS_GEV**2)
    energy_star = RHO_MASS_GEV / 2.0
    beta = 0.8
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    expected_psi = np.asarray([-1.0, -0.4, 0.0, 0.35, 1.0])
    charged = np.zeros((len(expected_psi), 4), dtype=np.float64)
    neutral = np.zeros_like(charged)
    charged[:, 0] = p_star * np.sqrt(1.0 - expected_psi**2)
    charged[:, 2] = gamma * (p_star * expected_psi + beta * energy_star)
    charged[:, 3] = gamma * (energy_star + beta * p_star * expected_psi)
    neutral[:, 0] = -charged[:, 0]
    neutral[:, 2] = gamma * (-p_star * expected_psi + beta * energy_star)
    neutral[:, 3] = gamma * (energy_star - beta * p_star * expected_psi)
    if float(np.max(np.abs(thesis_cospsi(charged, neutral) - expected_psi))) > 1.0e-12:
        raise RuntimeError("Eq. 3.5 synthetic validation failed")


def as_numpy(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(ak.to_numpy(value), dtype=dtype)


def four_to_eta_phi(four: np.ndarray) -> tuple[float, float, np.ndarray]:
    spatial = np.asarray(four[:3], dtype=np.float64)
    momentum = float(np.linalg.norm(spatial))
    if not np.isfinite(momentum) or momentum <= 0.0:
        raise ValueError("zero or non-finite truth pi0 momentum")
    pt = float(np.hypot(spatial[0], spatial[1]))
    eta = float(np.arcsinh(spatial[2] / pt)) if pt > 0.0 else math.copysign(float("inf"), spatial[2])
    phi = float(np.arctan2(spatial[1], spatial[0]))
    if not np.isfinite(eta) or not np.isfinite(phi):
        raise ValueError("non-finite pi0 direction")
    return eta, phi, spatial / momentum


def reco_direction(eta: float, phi: float) -> np.ndarray:
    if not np.isfinite(eta) or not np.isfinite(phi):
        raise ValueError("non-finite reco pi0 direction")
    cosh_eta = math.cosh(eta)
    direction = np.asarray(
        [math.cos(phi) / cosh_eta, math.sin(phi) / cosh_eta, math.tanh(eta)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid reco pi0 direction")
    return direction / norm


def on_shell_pi0(energy: float, direction: np.ndarray) -> np.ndarray:
    """Construct (px,py,pz,E) with the supplied E/direction and m_pi0."""
    if not np.isfinite(energy) or energy < NEUTRAL_PION_MASS_GEV:
        raise ValueError("pi0 energy is below the physical on-shell threshold")
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid pi0 direction")
    direction = direction / norm
    momentum = math.sqrt(max(energy * energy - NEUTRAL_PION_MASS_GEV**2, 0.0))
    return np.r_[momentum * direction, energy]


def find_object(
    data: dict[str, np.ndarray], event_index: int, tau_slot: int, role: int
) -> int:
    mask = (
        (data["object_event_local_index"] == event_index)
        & (data["object_boson_tau_slot"] == tau_slot)
        & (data["object_role"] == role)
    )
    indices = np.flatnonzero(mask)
    if len(indices) != 1:
        raise RuntimeError(
            f"event={event_index}, tau_slot={tau_slot}: expected one object role={role}, found {len(indices)}"
        )
    return int(indices[0])


def validate_row_identity(
    truth: dict[str, np.ndarray], row_map: dict[str, np.ndarray], seen: np.ndarray, path: Path
) -> None:
    rows = truth["event_row_global_index"].astype(np.int64)
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(seen)):
        raise RuntimeError(f"{path.name}: invalid truth global rows")
    if np.any(seen[rows]):
        raise RuntimeError(f"{path.name}: truth rows overlap a previous shard")
    for truth_key, row_key in (
        ("event_sample_id", "sample_id"),
        ("event_split_id", "split_id"),
        ("event_ntuple_entry", "ntuple_entry"),
        ("event_source_file_index", "source_file_index"),
        ("event_source_event_number", "source_event_number"),
    ):
        if not np.array_equal(truth[truth_key], row_map[row_key][rows]):
            raise RuntimeError(f"{path.name}: {truth_key} disagrees with canonical row map")
    seen[rows] = True


def ntuple_paths(audit: dict[str, Any]) -> dict[tuple[str, int], Path]:
    output: dict[tuple[str, int], Path] = {}
    for record in audit["ntuple"]["files"]:
        match = re.search(r"_chunk_(\d+)\.root$", Path(record["path"]).name)
        if match is None:
            raise RuntimeError(f"cannot infer chunk index from {record['path']}")
        key = (str(record["sample"]), int(match.group(1)))
        if key in output:
            raise RuntimeError(f"duplicate ntuple path for {key}")
        output[key] = Path(record["path"])
    return output


def scalar_branch(
    reco: dict[str, Any], name: str, entry: int, dtype: Any
) -> np.ndarray:
    return as_numpy(reco[name][entry], dtype)


def make_stats(
    panel: str,
    observable: str,
    hybrid: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    if len(x) == 0:
        raise RuntimeError(f"{panel}: empty panel population")
    delta = y - x
    if len(x) >= 2 and np.std(x) > 0.0 and np.std(y) > 0.0:
        pearson = float(np.corrcoef(x, y)[0, 1])
    else:
        pearson = float("nan")
    return {
        "panel": panel,
        "observable": observable,
        "hybrid": hybrid,
        "event_count": int(len(np.unique(rows))),
        "tau_count": int(len(x)),
        "H_tau_count": int(np.sum(sample_ids == 1)),
        "Z_tau_count": int(np.sum(sample_ids == 0)),
        "pearson": pearson,
        "rmse_to_diagonal": float(np.sqrt(np.mean(delta**2))),
        "mean_delta_y_minus_x": float(np.mean(delta)),
        "median_abs_delta": float(np.median(np.abs(delta))),
    }


def histogram_density(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, x_edges, y_edges = np.histogram2d(
        x, y, bins=bins, range=(MAP_RANGE, MAP_RANGE), density=False
    )
    bin_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    density = values / (len(x) * bin_area)
    return density, x_edges, y_edges


def make_figure(
    output_path: Path,
    panel_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    bins: int,
) -> list[dict[str, Any]]:
    panel_specs = (
        ("upper_left", "cosθ: energy-reco", "cos_theta", "energy_reco"),
        ("upper_right", "cosθ: direction-reco", "cos_theta", "direction_reco"),
        ("lower_left", "cosψ: energy-reco", "cos_psi", "energy_reco"),
        ("lower_right", "cosψ: direction-reco", "cos_psi", "direction_reco"),
    )
    densities: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    edges: tuple[np.ndarray, np.ndarray] | None = None
    for panel, _, observable, hybrid in panel_specs:
        x, y, rows, sample_ids = panel_data[panel]
        stats.append(make_stats(panel, observable, hybrid, x, y, rows, sample_ids))
        density, x_edges, y_edges = histogram_density(x, y, bins)
        densities.append(density)
        if edges is None:
            edges = (x_edges, y_edges)
        elif not (np.array_equal(edges[0], x_edges) and np.array_equal(edges[1], y_edges)):
            raise RuntimeError("panel binning is not shared")

    vmax = max(float(np.max(density)) for density in densities)
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise RuntimeError("all panel densities are empty")
    norm = Normalize(vmin=0.0, vmax=vmax)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), constrained_layout=True)
    mesh = None
    for axis, density, spec, panel_stats in zip(axes.flat, densities, panel_specs, stats):
        panel, title, observable, hybrid = spec
        mesh = axis.pcolormesh(
            edges[0], edges[1], density.T, cmap="viridis", norm=norm, shading="auto"
        )
        axis.plot(MAP_RANGE, MAP_RANGE, color="white", linewidth=1.1, linestyle="--")
        axis.set_xlim(MAP_RANGE)
        axis.set_ylim(MAP_RANGE)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(
            r"truth $\cos\theta$" if observable == "cos_theta" else r"truth $\cos\psi$"
        )
        axis.set_ylabel(
            ("energy-reco " if hybrid == "energy_reco" else "direction-reco ")
            + (r"$\cos\theta$" if observable == "cos_theta" else r"$\cos\psi$")
        )
        axis.set_title(title)
        axis.text(
            0.04,
            0.96,
            f"Nτ={panel_stats['tau_count']}\nr={panel_stats['pearson']:.4f}\nRMSE={panel_stats['rmse_to_diagonal']:.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=8.5,
            bbox={"facecolor": "black", "alpha": 0.52, "pad": 3.0, "edgecolor": "none"},
        )
    if mesh is None:
        raise RuntimeError("no figure panels were created")
    figure.colorbar(mesh, ax=axes.ravel().tolist(), label="2D density")
    figure.suptitle(
        r"Matched 1p1n $\tau$: $\pi^0$ reco-error diagnostic (validation; H+Z)",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return stats


def write_stats_csv(path: Path, stats: list[dict[str, Any]]) -> None:
    fields = [
        "panel",
        "observable",
        "hybrid",
        "event_count",
        "tau_count",
        "H_tau_count",
        "Z_tau_count",
        "pearson",
        "rmse_to_diagonal",
        "mean_delta_y_minus_x",
        "median_abs_delta",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in stats)


def main() -> int:
    args = parse_args()
    validate_thesis_formulas()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    run_root = args.run_root.resolve()
    row_map_path = args.row_map.resolve()
    truth_dir = args.truth_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    audit_path = run_root / "artifacts/analysis-surface-audit.json"
    truth_manifest_path = truth_dir / "manifest.json"
    audit = json.loads(audit_path.read_text())
    truth_manifest = json.loads(truth_manifest_path.read_text())
    row_map_archive = np.load(row_map_path, allow_pickle=False)
    row_map = {key: np.asarray(row_map_archive[key]) for key in row_map_archive.files}
    required_row_keys = {
        "sample_id",
        "ntuple_entry",
        "source_file_index",
        "source_event_number",
        "split_id",
    }
    if not required_row_keys.issubset(row_map):
        raise RuntimeError("canonical row map is missing required identity fields")

    ntuples = ntuple_paths(audit)
    seen_rows = np.zeros(len(row_map["sample_id"]), dtype=bool)
    target_by_file: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    flow_by_sample: dict[str, Counter[str]] = {"H": Counter(), "Z": Counter()}
    target_event_rows: dict[str, set[int]] = {"H": set(), "Z": set()}

    truth_paths = sorted(truth_dir.glob("[HZ]-*.npz"))
    if not truth_paths:
        raise FileNotFoundError(f"no truth shards in {truth_dir}")
    expected_manifest_shards = {(item["sample"], int(item["chunk_index"])): item for item in truth_manifest["shards"]}

    for truth_path in truth_paths:
        match = re.match(r"([HZ])-(\d+)\.npz$", truth_path.name)
        if match is None:
            raise RuntimeError(f"unexpected truth shard name {truth_path.name}")
        sample = match.group(1)
        chunk = int(match.group(2))
        shard_meta = expected_manifest_shards.get((sample, chunk))
        if shard_meta is None:
            raise RuntimeError(f"{truth_path.name}: missing manifest entry")
        if sha256_file(truth_path) != shard_meta["npz_sha256"]:
            raise RuntimeError(f"{truth_path.name}: truth shard hash mismatch")
        with np.load(truth_path, allow_pickle=False) as source:
            truth = {key: np.asarray(source[key]) for key in source.files}
        validate_row_identity(truth, row_map, seen_rows, truth_path)
        event_rows = truth["event_row_global_index"].astype(np.int64)
        validation_events = truth["event_split_id"] == VALIDATION_SPLIT_ID
        tau_event = truth["tau_event_local_index"].astype(np.int64)
        tau_pdg = truth["tau_pdg_id"].astype(np.int16)
        tau_mode = truth["tau_mode_code"].astype(np.int16)
        tau_exact = (
            (tau_mode == 1)
            & (truth["tau_n_charged_pion"] == 1)
            & (truth["tau_n_neutral_pion"] == 1)
            & (truth["tau_n_tau_neutrino"] == 1)
            & (truth["tau_n_other_neutrino"] == 0)
            & (truth["tau_n_photon"] == 0)
            & (truth["tau_object_count"] == 3)
            & (truth["tau_decay_vector_match"] == 1)
            & (np.abs(tau_pdg) == 15)
        )
        for tau_index in np.flatnonzero(tau_exact):
            event_index = int(tau_event[tau_index])
            if not validation_events[event_index]:
                continue
            flow_by_sample[sample]["truth_1p1n_validation_candidates"] += 1
            tau_slot = int(truth["tau_boson_slot"][tau_index])
            try:
                charged_index = find_object(truth, event_index, tau_slot, ROLE_CHARGED_PION)
                neutral_index = find_object(truth, event_index, tau_slot, ROLE_NEUTRAL_PION)
                neutrino_index = find_object(truth, event_index, tau_slot, ROLE_TAU_NEUTRINO)
            except RuntimeError:
                flow_by_sample[sample]["truth_object_contract_failed"] += 1
                continue
            charged_pdg = int(truth["object_pdg_id"][charged_index])
            neutral_pdg = int(truth["object_pdg_id"][neutral_index])
            neutrino_pdg = int(truth["object_pdg_id"][neutrino_index])
            expected_charged_pdg = -211 if int(tau_pdg[tau_index]) == 15 else 211
            expected_neutrino_pdg = 16 if int(tau_pdg[tau_index]) == 15 else -16
            if (charged_pdg, neutral_pdg, neutrino_pdg) != (
                expected_charged_pdg,
                111,
                expected_neutrino_pdg,
            ):
                flow_by_sample[sample]["truth_daughter_pdg_failed"] += 1
                continue
            tau_four = truth["tau_px_py_pz_e"][tau_index].astype(np.float64)
            charged_four = truth["object_px_py_pz_e"][charged_index].astype(np.float64)
            neutral_four = truth["object_px_py_pz_e"][neutral_index].astype(np.float64)
            neutrino_four = truth["object_px_py_pz_e"][neutrino_index].astype(np.float64)
            closure = float(
                np.max(np.abs(tau_four - charged_four - neutral_four - neutrino_four))
                / max(float(tau_four[3]), 1.0)
            )
            if not np.isfinite(closure) or closure > 1.0e-4:
                flow_by_sample[sample]["truth_three_daughter_closure_failed"] += 1
                continue

            reco_to_slot = truth["event_reco_to_boson_tau_slot"][event_index].astype(np.int16)
            reco_indices = np.flatnonzero(reco_to_slot == tau_slot)
            if len(reco_indices) != 1:
                flow_by_sample[sample]["reco_to_truth_tau_mapping_failed"] += 1
                continue
            flow_by_sample[sample]["truth_tau_ready_for_reco_selection"] += 1
            row = int(event_rows[event_index])
            target_event_rows[sample].add(row)
            target_by_file[(sample, chunk)].append(
                {
                    "event_index": event_index,
                    "row": row,
                    "ntuple_entry": int(truth["event_ntuple_entry"][event_index]),
                    "truth_slot": tau_slot,
                    "reco_tau_index": int(reco_indices[0]),
                    "tau_four": tau_four,
                    "charged_four": charged_four,
                    "truth_pi0_four": neutral_four,
                    "sample_id": int(truth["event_sample_id"][event_index]),
                }
            )

    if not np.array_equal(seen_rows, np.ones_like(seen_rows, dtype=bool)):
        raise RuntimeError(f"truth shards do not cover all canonical rows: {int(np.sum(~seen_rows))} missing")

    record_lists: dict[str, list[Any]] = {
        "row": [],
        "sample_id": [],
        "reco_tau_index": [],
        "truth_slot": [],
        "cos_theta_truth": [],
        "cos_theta_energy_reco": [],
        "cos_theta_direction_reco": [],
        "cos_psi_truth": [],
        "cos_psi_energy_reco": [],
        "cos_psi_direction_reco": [],
    }

    branches = [
        "tau_decayMode",
        "pfo_tauIndex",
        "pfo_pt",
        "pfo_eta",
        "pfo_phi",
        "pfo_e",
        "pfo_isPi0",
    ]
    flow_global = Counter()
    flow_by_sample_final: dict[str, Counter[str]] = {"H": Counter(), "Z": Counter()}
    for key, targets in sorted(target_by_file.items()):
        sample, chunk = key
        if not targets:
            continue
        ntuple_path = ntuples.get(key)
        if ntuple_path is None:
            raise RuntimeError(f"no ntuple for truth shard {sample}-{chunk:03d}")
        with uproot.open(ntuple_path) as source:
            tree = source["tauspin"]
            reco = tree.arrays(branches, library="ak")
        for target in targets:
            sample_flow = flow_by_sample_final[sample]
            event_index = target["event_index"]
            entry = target["ntuple_entry"]
            tau_modes = scalar_branch(reco, "tau_decayMode", entry, np.int16)
            if len(tau_modes) != 2:
                sample_flow["reco_tau_count_failed"] += 1
                continue
            reco_tau_index = target["reco_tau_index"]
            if int(tau_modes[reco_tau_index]) != 1:
                sample_flow["reco_tau_not_1p1n"] += 1
                continue
            sample_flow["truth_and_reco_1p1n"] += 1
            pfo_tau = scalar_branch(reco, "pfo_tauIndex", entry, np.int16)
            pfo_pt = scalar_branch(reco, "pfo_pt", entry, np.float64)
            pfo_eta = scalar_branch(reco, "pfo_eta", entry, np.float64)
            pfo_phi = scalar_branch(reco, "pfo_phi", entry, np.float64)
            pfo_energy = scalar_branch(reco, "pfo_e", entry, np.float64)
            pfo_ispi0 = scalar_branch(reco, "pfo_isPi0", entry, np.int8)
            if not all(len(value) == len(pfo_tau) for value in (pfo_pt, pfo_eta, pfo_phi, pfo_energy, pfo_ispi0)):
                raise RuntimeError(f"{sample}-{chunk:03d} entry={entry}: PFO branch length mismatch")
            valid_pfo = (
                np.isfinite(pfo_pt)
                & np.isfinite(pfo_eta)
                & np.isfinite(pfo_phi)
                & np.isfinite(pfo_energy)
                & (pfo_pt >= 0.0)
            )
            candidate_indices = np.flatnonzero(
                (pfo_tau == reco_tau_index) & (pfo_ispi0 == 1) & valid_pfo
            )
            if len(candidate_indices) == 0:
                sample_flow["no_valid_reco_pi0_candidate"] += 1
                continue
            truth_eta, truth_phi, truth_direction = four_to_eta_phi(target["truth_pi0_four"])
            candidate_deta = truth_eta - pfo_eta[candidate_indices]
            candidate_dphi = delta_phi(truth_phi, pfo_phi[candidate_indices])
            candidate_dr = np.hypot(candidate_deta, candidate_dphi)
            nearest_position = int(np.argmin(candidate_dr))
            nearest_dr = float(candidate_dr[nearest_position])
            if not np.isfinite(nearest_dr) or nearest_dr >= PRIMARY_DR:
                sample_flow["truth_reco_pi0_match_failed"] += 1
                continue
            # This is the existing deterministic one-to-one nearest assignment
            # for one truth pi0 per 1p1n tau. Equal-distance ties are rejected.
            if int(np.sum(np.isclose(candidate_dr, nearest_dr, rtol=0.0, atol=1.0e-12))) != 1:
                sample_flow["truth_reco_pi0_nearest_tie"] += 1
                continue
            chosen = int(candidate_indices[nearest_position])
            sample_flow["unique_truth_reco_pi0_match"] += 1
            reco_dir = reco_direction(float(pfo_eta[chosen]), float(pfo_phi[chosen]))
            try:
                energy_pi0 = on_shell_pi0(float(pfo_energy[chosen]), truth_direction)
                direction_pi0 = on_shell_pi0(float(target["truth_pi0_four"][3]), reco_dir)
            except ValueError:
                sample_flow["physical_hybrid_four_vector_failed"] += 1
                continue
            rho_truth = target["charged_four"] + target["truth_pi0_four"]
            rho_energy = target["charged_four"] + energy_pi0
            rho_direction = target["charged_four"] + direction_pi0
            try:
                observables = {
                    "cos_theta_truth": float(thesis_costheta(target["tau_four"], rho_truth)),
                    "cos_theta_energy_reco": float(thesis_costheta(target["tau_four"], rho_energy)),
                    "cos_theta_direction_reco": float(thesis_costheta(target["tau_four"], rho_direction)),
                    "cos_psi_truth": float(thesis_cospsi(target["charged_four"], target["truth_pi0_four"])),
                    "cos_psi_energy_reco": float(thesis_cospsi(target["charged_four"], energy_pi0)),
                    "cos_psi_direction_reco": float(thesis_cospsi(target["charged_four"], direction_pi0)),
                }
            except (FloatingPointError, ValueError, ZeroDivisionError):
                sample_flow["observable_evaluation_failed"] += 1
                continue
            if not all(np.isfinite(value) for value in observables.values()):
                sample_flow["observable_nonfinite"] += 1
                continue
            for field, value in observables.items():
                record_lists[field].append(value)
            record_lists["row"].append(target["row"])
            record_lists["sample_id"].append(target["sample_id"])
            record_lists["reco_tau_index"].append(reco_tau_index)
            record_lists["truth_slot"].append(target["truth_slot"])
            sample_flow["physical_matched_observable_records"] += 1

    values_all = {key: np.asarray(value, dtype=np.float64) for key, value in record_lists.items() if key.startswith("cos_")}
    rows_all = np.asarray(record_lists["row"], dtype=np.int64)
    sample_ids_all = np.asarray(record_lists["sample_id"], dtype=np.int8)
    if len(rows_all) == 0:
        raise RuntimeError("no matched physical observable records survived")

    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for observable in ("cos_theta", "cos_psi"):
        fields = [
            f"{observable}_truth",
            f"{observable}_energy_reco",
            f"{observable}_direction_reco",
        ]
        matrix = np.column_stack([values_all[field] for field in fields])
        valid = np.all(np.isfinite(matrix), axis=1) & np.all(
            np.abs(matrix) <= 1.0 + COS_TOLERANCE, axis=1
        )
        masks[observable] = valid
        for field in fields:
            values[field] = np.clip(values_all[field][valid], -1.0, 1.0)
        flow_global[f"{observable}_valid_records"] = int(np.sum(valid))
        flow_global[f"{observable}_out_of_range_records"] = int(np.sum(~valid))

    # Direct comparison within each observable uses the same tau mask for its
    # energy and direction panels. This avoids changing the population between
    # the two reconstruction-error hypotheses.
    theta_mask = masks["cos_theta"]
    psi_mask = masks["cos_psi"]
    theta_rows = rows_all[theta_mask]
    theta_samples = sample_ids_all[theta_mask]
    psi_rows = rows_all[psi_mask]
    psi_samples = sample_ids_all[psi_mask]

    # Theta and psi can have different physical-validity masks.  Keep the
    # energy/direction pair on the same tau population within each observable.
    panel_data = {
        "upper_left": (
            values["cos_theta_truth"],
            values["cos_theta_energy_reco"],
            theta_rows,
            theta_samples,
        ),
        "upper_right": (
            values["cos_theta_truth"],
            values["cos_theta_direction_reco"],
            theta_rows,
            theta_samples,
        ),
        "lower_left": (
            values["cos_psi_truth"],
            values["cos_psi_energy_reco"],
            psi_rows,
            psi_samples,
        ),
        "lower_right": (
            values["cos_psi_truth"],
            values["cos_psi_direction_reco"],
            psi_rows,
            psi_samples,
        ),
    }
    stats = make_figure(
        output_dir / "pi0-observable-resolution-2x2.png",
        panel_data,
        args.bins,
    )
    write_stats_csv(output_dir / "panel-stats.csv", stats)

    np.savez_compressed(
        output_dir / "matched-tau-values.npz",
        row_global_index_theta=theta_rows,
        sample_id_theta=theta_samples,
        cos_theta_truth=values["cos_theta_truth"],
        cos_theta_energy_reco=values["cos_theta_energy_reco"],
        cos_theta_direction_reco=values["cos_theta_direction_reco"],
        row_global_index_psi=psi_rows,
        sample_id_psi=psi_samples,
        cos_psi_truth=values["cos_psi_truth"],
        cos_psi_energy_reco=values["cos_psi_energy_reco"],
        cos_psi_direction_reco=values["cos_psi_direction_reco"],
    )

    merged_flow_by_sample: dict[str, Counter[str]] = {}
    for sample in ("H", "Z"):
        merged = Counter(flow_by_sample[sample])
        merged.update(flow_by_sample_final[sample])
        merged_flow_by_sample[sample] = merged
    per_sample = {
        sample: {key: int(value) for key, value in sorted(counter.items())}
        for sample, counter in merged_flow_by_sample.items()
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "completed",
        "scope": "tau-level matched 1p1n pi0 energy/direction observable diagnostic only",
        "split": "validation only; test rows were not selected or used",
        "samples": "H and Z processed with the same selection and combined in the figure",
        "selection": {
            "truth_tau": "truth tau_mode_code==1 with exactly one charged pion, one neutral pion, one tau neutrino, no other neutrino/photon, tau decay-vector match, and three-daughter closure <=1e-4",
            "reco_tau": "reco tau_decayMode==1 for the reco tau mapped to the target truth tau",
            "pi0_match": "same-tau pfo_isPi0==1 candidate, valid pT/eta/phi/E, nearest DeltaR assignment, primary DeltaR<0.05; equal-distance ties rejected",
            "opposite_tau": "unrestricted",
            "event_selection": "canonical analysis-surface rows; no additional event-level requirement on the opposite tau truth decay mode",
        },
        "definitions": {
            "cos_theta": "Existing thesis Eq. 3.4 implementation: x=E_rho/E_tau, a=m_rho/m_tau, beta=sqrt(1-m_tau^2/E_tau^2), cos(theta)=(2*x-1-a^2)/(beta*(1-a^2)); rho=charged pion+pi0",
            "cos_psi": "Existing thesis Eq. 3.5 implementation: [m_rho/sqrt(m_rho^2-4*m_pi^2)]*(E_charged-E_pi0)/|p_charged+p_pi0|",
            "m_tau_GeV": TAU_MASS_GEV,
            "m_rho_GeV": RHO_MASS_GEV,
            "m_charged_pion_GeV": CHARGED_PION_MASS_GEV,
            "hybrid_pi0_mass_GeV": NEUTRAL_PION_MASS_GEV,
            "hybrid_four_vector": "on-shell pi0 p=sqrt(E^2-m_pi0^2) times the supplied unit direction; energy-reco uses reco E/truth direction and direction-reco uses truth E/reco eta-phi",
            "range": list(MAP_RANGE),
            "bins_per_axis": int(args.bins),
            "out_of_range_handling": "raw values within 5e-5 of [-1,1] are clipped at the plotting boundary; larger excursions are omitted from both energy/direction panels for that observable",
        },
        "counts": {
            "canonical_rows": int(len(row_map["sample_id"])),
            "canonical_validation_rows": int(np.sum(row_map["split_id"] == VALIDATION_SPLIT_ID)),
            "truth_candidate_events_by_sample": {sample: int(len(rows)) for sample, rows in target_event_rows.items()},
            "matched_physical_records": int(len(rows_all)),
            **flow_global,
        },
        "flow_by_sample": per_sample,
        "panels": stats,
        "artifacts": {
            "figure": "pi0-observable-resolution-2x2.png",
            "table": "panel-stats.csv",
            "values": "matched-tau-values.npz",
        },
        "inputs": {
            "run_root": str(run_root),
            "audit_sha256": sha256_file(audit_path),
            "truth_manifest_sha256": sha256_file(truth_manifest_path),
            "row_map_sha256": sha256_file(row_map_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "interpretation_boundary": "This is a matched 1p1n tau observable-smearing diagnostic for pi0 reco energy/direction errors. It is not a detector-level counterfactual or a causal attribution of all reconstruction effects.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
