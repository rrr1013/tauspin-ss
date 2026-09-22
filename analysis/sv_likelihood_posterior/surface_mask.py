"""Build the operational exact reco-surface valid/invalid mask.

This deliberately matches WP-A: reconstructed visible tau four-vectors,
reconstructed MET, a 64x64 deterministic proposal lattice, and up to 128
retained transverse proposals.  The mask is a mechanism-check cohort only;
the continuous posterior analysis itself does not require an exact surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


TAU_MASS_GEV = 1.7769


def invariant_mass(four: np.ndarray) -> np.ndarray:
    mass2 = four[..., 3] ** 2 - np.sum(four[..., :3] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-lattice", type=int, default=64)
    parser.add_argument("--keep", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    sys.path.insert(0, str(args.module_dir))
    import metspace as ms  # type: ignore

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    with np.load(args.inputs, allow_pickle=False) as data:
        global_indices = np.asarray(data["global_indices"], dtype=np.int64)
        visible = np.asarray(data["reco_visible_tau_lab4"], dtype=np.float64)
        met = np.asarray(data["reco_met_xy"], dtype=np.float64)

    mass = invariant_mass(visible)
    algebraic = (
        np.isfinite(visible).all(axis=(1, 2))
        & np.isfinite(met).all(axis=1)
        & (visible[..., 3] > 0.0).all(axis=1)
        & (mass < TAU_MASS_GEV).all(axis=1)
    )
    retained = np.zeros(len(visible), dtype=np.int16)
    sheet_present = np.zeros((len(visible), 4), dtype=bool)
    rng = np.random.default_rng(args.seed)
    eligible = np.flatnonzero(algebraic)
    for start in range(0, len(eligible), args.row_chunk):
        rows = eligible[start : start + args.row_chunk]
        unit = ms.lattice(args.proposal_lattice, len(rows), rng).transpose(1, 0, 2)
        drawn = ms.sample_region(visible[rows], met[rows], unit)
        picked = ms.compact(drawn["accepted"], args.keep, rng)
        retained[rows] = np.sum(picked >= 0, axis=0).astype(np.int16)
        nu_t = np.take_along_axis(
            drawn["nu_t"], np.maximum(picked, 0)[..., None], axis=0
        )
        solved = ms.solution_sheets(visible[rows], met[rows], nu_t)
        valid = solved["valid"] & (picked >= 0)[:, None, :]
        sheet_present[rows] = np.any(valid, axis=0).T
        print(json.dumps({"processed": int(min(start + args.row_chunk, len(eligible))),
                          "eligible": int(len(eligible))}), flush=True)
    surface_valid = sheet_present.any(axis=1)
    np.savez_compressed(
        args.output / "surface_mask.npz",
        global_indices=global_indices,
        algebraically_possible=algebraic,
        surface_valid=surface_valid,
        sheet_present=sheet_present,
        retained=retained,
        reco_visible_mass=mass,
    )
    report = {
        "events": int(len(visible)),
        "algebraically_possible": int(algebraic.sum()),
        "surface_valid": int(surface_valid.sum()),
        "surface_invalid": int((~surface_valid).sum()),
        "surface_valid_fraction": float(surface_valid.mean()),
        "sheet_multiplicity": {
            str(value): int((sheet_present.sum(axis=1) == value).sum())
            for value in range(5)
        },
        "contract": {
            "visible": "reco_visible_tau_lab4",
            "met": "reco_met_xy",
            "tau_mass_GeV": TAU_MASS_GEV,
            "proposal_lattice": args.proposal_lattice,
            "keep": args.keep,
            "seed": args.seed,
            "purpose": "mechanism-check cohort only; not a selection on the main result",
            "test_loaded": False,
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

