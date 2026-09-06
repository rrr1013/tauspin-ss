#!/usr/bin/env python3
"""Extract per-tau reconstructed substructure summaries for the substructure run.

Substructure means: quantities built only from the core tracks and the neutral
PFOs that belong to one tau.  Whole-tau scalars that the 2026-09-06 non-spin run
already measured as ``D-reco`` (tau pT, eta, decay mode, track count, RNN score)
are deliberately not re-derived here, so that the substructure result reads as an
increment over that block.

Identity and population follow ``extract_nonspin_features.py`` exactly: the
canonical analysis surface, the validation split only, the test partition never
read, and ``eventNumber`` checked fail-closed against the row map.
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
    "tau_pt",
    "tau_m",
    "tau_charge",
    "tau_nTracks",
    "tau_nIsolatedTracks",
    "track_tauIndex",
    "track_pt",
    "track_eta",
    "track_phi",
    "track_dEta",
    "track_dPhi",
    "track_ptFraction",
    "track_d0",
    "track_z0SinTheta",
    "track_isFake",
    "track_passTrkSelector",
    "track_numberOfPixelHits",
    "track_numberOfSCTHits",
    "track_numberOfTRTHits",
    "pfo_tauIndex",
    "pfo_pt",
    "pfo_eta",
    "pfo_phi",
    "pfo_dEta",
    "pfo_dPhi",
    "pfo_ptFraction",
    "pfo_isPi0",
    "pfo_bdtPi0Score",
    "pfo_nPi0Proto",
]

# Per-side column names, in block order.  Written as ``<name>_minus`` / ``<name>_plus``.
PER_SIDE = [
    # S1 energy sharing
    "f_charged", "f_pi0", "ups_pi0", "lead_pfo_ptfrac", "sum_pfo_ptfrac",
    # S2 track structure
    "lead_trk_ptfrac", "subl_trk_ptfrac", "trk_pt_asym", "trk_width", "trk_dr_max",
    "sum_trk_ptfrac", "max_abs_d0", "max_abs_z0sin",
    # S3 neutral structure
    "n_pfo", "n_pi0", "max_npi0proto", "max_bdtpi0", "ptmean_bdtpi0", "pfo_width",
    # S4 shape
    "all_width", "m_const", "const_over_taupt", "tau_m",
    # S5 detector-only
    "frac_fake", "frac_passsel", "mean_pix", "mean_sct", "mean_trt", "n_iso",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntuple-dir", type=Path, required=True)
    p.add_argument("--row-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step-size", type=int, default=50000)
    p.add_argument("--max-chunks", type=int, default=0, help="smoke-test limit; 0 = all")
    p.add_argument("--splits", default="validation")
    return p.parse_args()


def safe_ratio(numerator, denominator, fill: float):
    """Elementwise ratio with a fixed value wherever the denominator vanishes."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(numerator.shape, float(fill), dtype=np.float64)
    good = denominator > 0.0
    out[good] = numerator[good] / denominator[good]
    return out


def sorted_column(values, index: int, fill: float) -> np.ndarray:
    """``index``-th largest entry of a jagged array, ``fill`` when it does not exist."""
    ordered = ak.sort(values, axis=1, ascending=False)
    padded = ak.pad_none(ordered, index + 1, axis=1, clip=True)
    return ak.to_numpy(ak.fill_none(padded[:, index], fill)).astype(np.float64)


def jagged_max(values, fill: float) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.max(values, axis=1), fill)).astype(np.float64)


def jagged_sum(values) -> np.ndarray:
    return ak.to_numpy(ak.sum(values, axis=1)).astype(np.float64)


def jagged_count(values) -> np.ndarray:
    return ak.to_numpy(ak.num(values, axis=1)).astype(np.float64)


def weighted_mean(weight, values, fill: float) -> np.ndarray:
    return safe_ratio(jagged_sum(weight * values), jagged_sum(weight), fill)


def side_features(tracks, pfos, tau_pt, tau_m, n_iso) -> dict[str, np.ndarray]:
    """All substructure columns for one tau side.

    ``tracks`` and ``pfos`` are already restricted to this tau.  Constituents are
    treated as massless four-vectors built from (pt, eta, phi); the tau selection
    guarantees at least one core track, so no charged quantity needs a fallback.
    """
    trk_pt = tracks["pt"]
    pfo_pt = pfos["pt"]
    pt_ch = jagged_sum(trk_pt)
    pt_neu = jagged_sum(pfo_pt)
    if not np.all(pt_ch > 0.0):
        raise RuntimeError("a selected tau has no core track momentum")
    pt_pi0 = jagged_sum(pfo_pt * pfos["isPi0"])
    lead_trk_pt = sorted_column(trk_pt, 0, 0.0)

    trk_dr = np.hypot(tracks["dEta"], tracks["dPhi"])
    pfo_dr = np.hypot(pfos["dEta"], pfos["dPhi"])
    lead_frac = sorted_column(tracks["ptFraction"], 0, 0.0)
    subl_frac = sorted_column(tracks["ptFraction"], 1, 0.0)

    # Massless constituent system: tracks plus neutral PFOs.
    px = jagged_sum(trk_pt * np.cos(tracks["phi"])) + jagged_sum(pfo_pt * np.cos(pfos["phi"]))
    py = jagged_sum(trk_pt * np.sin(tracks["phi"])) + jagged_sum(pfo_pt * np.sin(pfos["phi"]))
    pz = jagged_sum(trk_pt * np.sinh(tracks["eta"])) + jagged_sum(pfo_pt * np.sinh(pfos["eta"]))
    energy = jagged_sum(trk_pt * np.cosh(tracks["eta"])) + jagged_sum(pfo_pt * np.cosh(pfos["eta"]))
    m2 = energy * energy - (px * px + py * py + pz * pz)

    all_pt_dr = jagged_sum(trk_pt * trk_dr) + jagged_sum(pfo_pt * pfo_dr)

    n_trk = jagged_count(trk_pt)
    return {
        "f_charged": pt_ch / (pt_ch + pt_neu),
        "f_pi0": pt_pi0 / (pt_ch + pt_neu),
        "ups_pi0": safe_ratio(lead_trk_pt - pt_pi0, lead_trk_pt + pt_pi0, 1.0),
        "lead_pfo_ptfrac": sorted_column(pfos["ptFraction"], 0, 0.0),
        "sum_pfo_ptfrac": jagged_sum(pfos["ptFraction"]),
        "lead_trk_ptfrac": lead_frac,
        "subl_trk_ptfrac": subl_frac,
        "trk_pt_asym": safe_ratio(lead_frac - subl_frac, lead_frac + subl_frac, 1.0),
        "trk_width": weighted_mean(trk_pt, trk_dr, 0.0),
        "trk_dr_max": jagged_max(trk_dr, 0.0),
        "sum_trk_ptfrac": jagged_sum(tracks["ptFraction"]),
        "max_abs_d0": jagged_max(np.abs(tracks["d0"]), 0.0),
        "max_abs_z0sin": jagged_max(np.abs(tracks["z0SinTheta"]), 0.0),
        "n_pfo": jagged_count(pfo_pt),
        "n_pi0": jagged_sum(pfos["isPi0"]),
        "max_npi0proto": jagged_max(pfos["nPi0Proto"], 0.0),
        "max_bdtpi0": jagged_max(pfos["bdtPi0Score"], 0.0),
        "ptmean_bdtpi0": weighted_mean(pfo_pt, pfos["bdtPi0Score"], 0.0),
        "pfo_width": weighted_mean(pfo_pt, pfo_dr, 0.0),
        "all_width": safe_ratio(all_pt_dr, pt_ch + pt_neu, 0.0),
        "m_const": np.sqrt(np.maximum(m2, 0.0)),
        "const_over_taupt": safe_ratio(pt_ch + pt_neu, tau_pt, 0.0),
        "tau_m": tau_m.astype(np.float64),
        "frac_fake": safe_ratio(jagged_sum(tracks["isFake"]), n_trk, 0.0),
        "frac_passsel": safe_ratio(jagged_sum(tracks["passTrkSelector"]), n_trk, 0.0),
        "mean_pix": safe_ratio(jagged_sum(tracks["numberOfPixelHits"]), n_trk, 0.0),
        "mean_sct": safe_ratio(jagged_sum(tracks["numberOfSCTHits"]), n_trk, 0.0),
        "mean_trt": safe_ratio(jagged_sum(tracks["numberOfTRTHits"]), n_trk, 0.0),
        "n_iso": n_iso.astype(np.float64),
    }


TRACK_FIELDS = ("pt", "eta", "phi", "dEta", "dPhi", "ptFraction", "d0", "z0SinTheta",
                "isFake", "passTrkSelector", "numberOfPixelHits", "numberOfSCTHits",
                "numberOfTRTHits")
PFO_FIELDS = ("pt", "eta", "phi", "dEta", "dPhi", "ptFraction", "isPi0", "bdtPi0Score",
              "nPi0Proto")


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
        + [f"{name}_{side}" for name in PER_SIDE for side in ("minus", "plus")]
    )
    out: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    audit = {
        "entries_read": 0,
        "identity_checked": 0,
        "rows_on_surface": 0,
        "rows_kept": 0,
        "charge_ordering_violations": 0,
        "tau_index_violations": 0,
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

                charge = ak.to_numpy(sub["tau_charge"])
                audit["charge_ordering_violations"] += int(
                    ((charge[:, 0] >= 0) | (charge[:, 1] <= 0)).sum()
                )
                # Slot 0 is tau-, slot 1 is tau+; the token index refers to those slots.
                for name in ("track_tauIndex", "pfo_tauIndex"):
                    flat = ak.to_numpy(ak.flatten(sub[name]))
                    audit["tau_index_violations"] += int(((flat < 0) | (flat > 1)).sum())

                tau_pt = ak.to_numpy(sub["tau_pt"]).astype(np.float64)
                tau_m = ak.to_numpy(sub["tau_m"])
                n_iso = ak.to_numpy(sub["tau_nIsolatedTracks"])
                n_tracks = ak.to_numpy(sub["tau_nTracks"])

                block = {
                    "row_index": rows_keep.astype(np.int64),
                    "sample_id": np.full(len(rows_keep), sample_id, dtype=np.int8),
                    "split_id": row["split_id"][rows_keep].astype(np.uint8),
                    "source_file_index": row["source_file_index"][rows_keep].astype(np.int32),
                }
                for slot, side in ((0, "minus"), (1, "plus")):
                    trk_mask = sub["track_tauIndex"] == slot
                    pfo_mask = sub["pfo_tauIndex"] == slot
                    tracks = {f: sub[f"track_{f}"][trk_mask] for f in TRACK_FIELDS}
                    pfos = {f: sub[f"pfo_{f}"][pfo_mask] for f in PFO_FIELDS}
                    counted = ak.to_numpy(ak.num(tracks["pt"], axis=1))
                    if not np.array_equal(counted, n_tracks[:, slot]):
                        raise RuntimeError("track_tauIndex does not reproduce tau_nTracks")
                    values = side_features(tracks, pfos, tau_pt[:, slot], tau_m[:, slot],
                                           n_iso[:, slot])
                    for name, column in values.items():
                        block[f"{name}_{side}"] = column

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
        raise RuntimeError("non-finite substructure values; every column has a defined fallback")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **table)
    audit["rows_written"] = int(len(table["row_index"]))
    audit["columns"] = list(columns)
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in audit.items() if k != "columns"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
