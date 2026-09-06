#!/usr/bin/env python3
"""Extract event-level production and decay observables for the non-spin information run.

Reads the reco-truth-common-v2 ntuple, keeps only rows that sit on the canonical
fixed-v3 analysis surface, and writes one compact per-event table.  Nothing here
depends on the classifier and the test partition is never read.

The table is deliberately raw: block definitions and every estimator live in
``analysis/nonspin_information.py`` so that the extraction can be re-used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = 0, 1, 2

BRANCHES = [
    "eventNumber",
    "averageInteractionsPerCrossing",
    "met_et",
    "met_phi",
    "met_sumet",
    "truth_boson_pt",
    "truth_boson_eta",
    "truth_boson_phi",
    "truth_boson_m",
    "truth_met_et",
    "truth_met_phi",
    "truth_met_sumet",
    "truth_tau_count",
    "truth_decay_complete",
    "tau_pt",
    "tau_eta",
    "tau_phi",
    "tau_m",
    "tau_charge",
    "tau_nTracks",
    "tau_nIsolatedTracks",
    "tau_decayMode",
    "tau_rnnJetScore",
    "tau_truthMatched",
    "tau_truthDecayMode",
]

# Per-event scalars copied straight through.
SCALARS = [
    "averageInteractionsPerCrossing",
    "met_et",
    "met_phi",
    "met_sumet",
    "truth_boson_pt",
    "truth_boson_eta",
    "truth_boson_phi",
    "truth_boson_m",
    "truth_met_et",
    "truth_met_phi",
    "truth_met_sumet",
]

# Per-tau quantities, stored as ``<name>_minus`` (slot 0) and ``<name>_plus`` (slot 1).
PER_TAU = [
    "tau_pt",
    "tau_eta",
    "tau_phi",
    "tau_m",
    "tau_charge",
    "tau_nTracks",
    "tau_nIsolatedTracks",
    "tau_decayMode",
    "tau_rnnJetScore",
    "tau_truthMatched",
    "tau_truthDecayMode",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntuple-dir", type=Path, required=True)
    p.add_argument("--row-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step-size", type=int, default=50000)
    p.add_argument("--max-chunks", type=int, default=0, help="smoke-test limit; 0 = all")
    p.add_argument(
        "--splits",
        default="validation",
        help="canonical splits to keep; the test partition is always refused",
    )
    return p.parse_args()


def delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed phi difference wrapped into (-pi, pi]."""
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def four_vector(pt, eta, phi, mass):
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return px, py, pz, energy


def main() -> None:
    args = parse_args()
    names = {"train": SPLIT_TRAIN, "validation": SPLIT_VALIDATION, "test": SPLIT_TEST}
    wanted = sorted(names[n.strip()] for n in args.splits.split(",") if n.strip())
    if SPLIT_TEST in wanted:
        raise SystemExit("refusing to read the test partition in this run")
    wanted = np.array(wanted, dtype=np.int64)

    with np.load(args.row_map, allow_pickle=False) as source:
        row = {key: np.asarray(source[key]) for key in source.files}
    n_rows = len(row["sample_id"])

    ident = (
        row["sample_id"].astype(np.int64) * (1 << 40)
        + row["ntuple_file_index"].astype(np.int64) * (1 << 24)
        + row["ntuple_entry"].astype(np.int64)
    )
    if len(np.unique(ident)) != n_rows:
        raise RuntimeError("canonical identity triple is not unique in the row map")
    order = np.argsort(ident, kind="stable")
    ident_sorted = ident[order]

    columns = (
        ["row_index", "sample_id", "split_id", "source_file_index"]
        + SCALARS
        + [f"{name}_{side}" for name in PER_TAU for side in ("minus", "plus")]
        + [
            "vis_mass", "vis_pt", "vis_eta", "vis_rapidity",
            "delta_r_tt", "delta_phi_tt", "delta_eta_tt",
            "total_pt", "dphi_met_minus", "dphi_met_plus", "dphi_met_visible",
            "mt_total", "met_centrality_x", "met_centrality_y",
        ]
    )
    out: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    audit = {
        "entries_read": 0,
        "identity_checked": 0,
        "rows_on_surface": 0,
        "rows_kept": 0,
        "charge_ordering_violations": 0,
        "nonfinite_dropped": 0,
    }
    seen = np.zeros(n_rows, dtype=bool)

    for sample_name, sample_id in (("H", 0), ("Z", 1)):
        files = sorted((args.ntuple_dir / sample_name).glob(f"{sample_name}_chunk_*.root"))
        if not files:
            raise FileNotFoundError(f"no chunks for {sample_name}")
        if args.max_chunks:
            files = files[: args.max_chunks]
        for file_index, path in enumerate(files):
            expected = f"{sample_name}_chunk_{file_index:03d}.root"
            if path.name != expected:
                raise RuntimeError(f"chunk ordering broken: {path.name} != {expected}")
            tree = uproot.open(path)["tauspin"]
            start = 0
            for arrays in tree.iterate(BRANCHES, step_size=args.step_size, library="ak"):
                n = len(arrays)
                entries = np.arange(start, start + n, dtype=np.int64)
                start += n
                audit["entries_read"] += n

                key = sample_id * (1 << 40) + file_index * (1 << 24) + entries
                pos = np.clip(np.searchsorted(ident_sorted, key), 0, n_rows - 1)
                on_surface = ident_sorted[pos] == key
                rows = order[pos]

                event_number = ak.to_numpy(arrays.eventNumber)
                if not np.array_equal(
                    event_number[on_surface], row["source_event_number"][rows[on_surface]]
                ):
                    raise RuntimeError(f"{path.name}: eventNumber mismatch against the row map")
                audit["identity_checked"] += int(on_surface.sum())
                audit["rows_on_surface"] += int(on_surface.sum())

                keep = on_surface & np.isin(row["split_id"][rows], wanted)
                if not keep.any():
                    continue
                sub = arrays[keep]
                rows_keep = rows[keep]
                if seen[rows_keep].any():
                    raise RuntimeError(f"{path.name}: duplicate global rows")
                seen[rows_keep] = True

                per_tau = {name: ak.to_numpy(sub[name]) for name in PER_TAU}
                charge = per_tau["tau_charge"]
                audit["charge_ordering_violations"] += int(
                    ((charge[:, 0] >= 0) | (charge[:, 1] <= 0)).sum()
                )

                pt = per_tau["tau_pt"].astype(np.float64)
                eta = per_tau["tau_eta"].astype(np.float64)
                phi = per_tau["tau_phi"].astype(np.float64)
                mass = per_tau["tau_m"].astype(np.float64)
                px0, py0, pz0, e0 = four_vector(pt[:, 0], eta[:, 0], phi[:, 0], mass[:, 0])
                px1, py1, pz1, e1 = four_vector(pt[:, 1], eta[:, 1], phi[:, 1], mass[:, 1])
                sx, sy, sz, se = px0 + px1, py0 + py1, pz0 + pz1, e0 + e1
                vis_mass2 = se * se - (sx * sx + sy * sy + sz * sz)
                vis_mass = np.sqrt(np.maximum(vis_mass2, 0.0))
                vis_pt = np.hypot(sx, sy)
                vis_eta = np.arcsinh(sz / np.maximum(vis_pt, 1e-12))
                vis_rapidity = 0.5 * np.log(
                    np.maximum(se + sz, 1e-12) / np.maximum(se - sz, 1e-12)
                )

                dphi_tt = delta_phi(phi[:, 0], phi[:, 1])
                deta_tt = eta[:, 0] - eta[:, 1]
                dr_tt = np.hypot(deta_tt, dphi_tt)

                met = ak.to_numpy(sub["met_et"]).astype(np.float64)
                met_phi = ak.to_numpy(sub["met_phi"]).astype(np.float64)
                mx, my = met * np.cos(met_phi), met * np.sin(met_phi)
                total_pt = np.hypot(sx + mx, sy + my)
                dphi_met_minus = delta_phi(met_phi, phi[:, 0])
                dphi_met_plus = delta_phi(met_phi, phi[:, 1])
                dphi_met_visible = delta_phi(met_phi, np.arctan2(sy, sx))
                mt_total2 = np.maximum(
                    (np.hypot(sx, sy) + met) ** 2 - ((sx + mx) ** 2 + (sy + my) ** 2), 0.0
                )
                # MET centrality in the transverse tau-tau basis (bisector projection).
                ux0, uy0 = np.cos(phi[:, 0]), np.sin(phi[:, 0])
                ux1, uy1 = np.cos(phi[:, 1]), np.sin(phi[:, 1])
                det = ux0 * uy1 - uy0 * ux1
                safe = np.where(np.abs(det) < 1e-9, np.nan, det)
                cx = (mx * uy1 - my * ux1) / safe
                cy = (my * ux0 - mx * uy0) / safe

                block = {
                    "row_index": rows_keep.astype(np.int64),
                    "sample_id": np.full(len(rows_keep), sample_id, dtype=np.int8),
                    "split_id": row["split_id"][rows_keep].astype(np.uint8),
                    "source_file_index": row["source_file_index"][rows_keep].astype(np.int32),
                    "vis_mass": vis_mass,
                    "vis_pt": vis_pt,
                    "vis_eta": vis_eta,
                    "vis_rapidity": vis_rapidity,
                    "delta_r_tt": dr_tt,
                    "delta_phi_tt": dphi_tt,
                    "delta_eta_tt": deta_tt,
                    "total_pt": total_pt,
                    "dphi_met_minus": dphi_met_minus,
                    "dphi_met_plus": dphi_met_plus,
                    "dphi_met_visible": dphi_met_visible,
                    "mt_total": np.sqrt(mt_total2),
                    "met_centrality_x": cx,
                    "met_centrality_y": cy,
                }
                for name in SCALARS:
                    block[name] = ak.to_numpy(sub[name]).astype(np.float64)
                for name in PER_TAU:
                    values = per_tau[name]
                    block[f"{name}_minus"] = values[:, 0]
                    block[f"{name}_plus"] = values[:, 1]

                for name in columns:
                    out[name].append(np.asarray(block[name]))
                audit["rows_kept"] += len(rows_keep)

    table = {name: np.concatenate(parts) for name, parts in out.items()}
    order_out = np.argsort(table["row_index"], kind="stable")
    table = {name: values[order_out] for name, values in table.items()}
    if len(np.unique(table["row_index"])) != len(table["row_index"]):
        raise RuntimeError("duplicate global rows in the extracted table")

    finite = np.ones(len(table["row_index"]), dtype=bool)
    for name, values in table.items():
        if values.dtype.kind == "f":
            finite &= np.isfinite(values)
    audit["nonfinite_dropped"] = int((~finite).sum())
    if audit["nonfinite_dropped"]:
        table = {name: values[finite] for name, values in table.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **table)
    audit["rows_written"] = int(len(table["row_index"]))
    audit["columns"] = list(columns)
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in audit.items() if k != "columns"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
