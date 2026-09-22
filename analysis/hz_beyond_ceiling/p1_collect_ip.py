"""Collect PV-referenced track impact-parameter vectors for the canonical cohort.

The ntuple stores the xAOD perigee parameters ``d0`` and ``z0``.  Both are
expressed relative to the beam-spot reference point, not the primary vertex.
``z0`` is therefore dominated by the primary-vertex z position (tens of mm),
while the tau impact parameter is O(50 um).  Earlier geometry studies built a
"directional IP" from ``(-d0 sin phi, d0 cos phi, z0 sin theta)`` without the
PV subtraction.  This script rebuilds the 3-D impact-parameter vector relative
to the reconstructed primary vertex for every core track and stores it next to
the row-aligned truth tau direction, so the azimuthal information carried by
the impact parameter can be audited directly.

Row order is identical to ``geometry_audit.npz`` of the 2026-09-20 IP/SV run:
samples H then Z, ROOT files sorted, selected entries ascending.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

BRANCHES = (
    "eventNumber", "tau_eta", "tau_phi", "tau_nTracks",
    "track_tauIndex", "track_isCore", "track_pt", "track_eta", "track_phi",
    "track_theta", "track_d0", "track_z0", "track_charge",
    "track_numberOfPixelHits",
    "primaryVertex_x", "primaryVertex_y", "primaryVertex_z",
)
MAX_CORE = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    root = Path("/home/rbaba/tauspin-ip-sv-common-v2/MakeNtuple/outputs/tauspin-ip-sv-20260920")
    p.add_argument("--root-h", type=Path, default=root / "H")
    p.add_argument("--root-z", type=Path, default=root / "Z")
    p.add_argument("--selection-manifest", type=Path, default=Path(
        "/home/rbaba/tauspin-ss/NN/outputs/data-preparation/"
        "fixed-partial-v3-20260730-2100-ptmatched20/pt_matching_manifest.json"))
    p.add_argument("--geometry-audit", type=Path, default=Path(
        "/home/rbaba/tauspin-ip-sv-analysis-corrected-20260920/geometry_audit.npz"))
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    manifest = json.loads(args.selection_manifest.read_text())["selected_entries"]
    with np.load(args.geometry_audit) as g:
        audit_events = np.asarray(g["event_numbers"])
        audit_pv = np.asarray(g["pv"])
        n_rows = len(audit_events)

    out = {
        "event_numbers": np.zeros(n_rows, np.int64),
        "pv": np.zeros((n_rows, 3)),
        "tau_eta": np.zeros((n_rows, 2)), "tau_phi": np.zeros((n_rows, 2)),
        "n_core": np.zeros((n_rows, 2), np.int8),
        # per side, per core track (pt ordered): pt, theta, phi, charge, d0, z0, npix
        "trk": np.full((n_rows, 2, MAX_CORE, 7), np.nan),
    }
    row = 0
    for sample in ("H", "Z"):
        root_dir = args.root_h if sample == "H" else args.root_z
        for path in sorted(root_dir.glob("*.root")):
            allowed = sorted(int(i) for i in manifest[sample].get(path.name, []))
            if not allowed:
                continue
            with uproot.open(path) as f:
                a = f["tauspin"].arrays(list(BRANCHES), library="ak")
            a = a[np.asarray(allowed)]
            n = len(a)
            sl = slice(row, row + n)
            out["event_numbers"][sl] = ak.to_numpy(a["eventNumber"])
            out["pv"][sl] = np.column_stack([ak.to_numpy(a[f"primaryVertex_{c}"]) for c in "xyz"])
            out["tau_eta"][sl] = ak.to_numpy(a["tau_eta"][:, :2])
            out["tau_phi"][sl] = ak.to_numpy(a["tau_phi"][:, :2])
            for side in (0, 1):
                m = (a["track_tauIndex"] == side) & (a["track_isCore"] == 1)
                fields = [a[name][m] for name in (
                    "track_pt", "track_theta", "track_phi", "track_charge",
                    "track_d0", "track_z0", "track_numberOfPixelHits")]
                order = ak.argsort(fields[0], ascending=False)
                fields = [x[order] for x in fields]
                out["n_core"][sl, side] = np.minimum(ak.to_numpy(ak.num(fields[0])), 127)
                for k, x in enumerate(fields):
                    padded = ak.to_numpy(ak.fill_none(ak.pad_none(x, MAX_CORE, clip=True), np.nan))
                    out["trk"][sl, side, :, k] = padded
            row += n
            print(f"{sample} {path.name} rows={row} {time.time()-start:.0f}s", flush=True)
    if row != n_rows:
        raise RuntimeError(f"row count {row} != audit {n_rows}")
    if not np.array_equal(out["event_numbers"], audit_events):
        raise RuntimeError("event-number order differs from geometry audit")
    if np.max(np.abs(out["pv"] - audit_pv)) > 1e-6:
        raise RuntimeError("PV differs from geometry audit")
    np.savez_compressed(args.output, **out)
    print("done", row, f"{time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
