"""Full per-tau detector geometry in the local reco (n, r, k) frame.

v2 (--with-track-offset, 22 channels): after the parallax finding of the
2026-09-23 rho IP frame-tilt run, every track block also carries the offset of
the track direction from the visible axis, (t.n, t.r)/10 mrad.  The IP only
fixes the tau azimuth around its own track; for rho and 3-prong the track is
displaced from k by O(5 mrad), comparable to the tau-track angle, so the
network needs t_perp to place the tau direction in the visible frame.
Layout v2: track j in 0..2 -> 6j..6j+5 = (u.n, u.r, log|IP|, avail, t.n/0.01, t.r/0.01);
18..21 = SV block.

v1 docstring: Full per-tau detector geometry in the local reco (n, r, k) frame (16 channels).

Per tau side (k = reco visible direction):
  0-11  PV-referenced 3-D IP of up to three core tracks (pt ordered), each
        (u.n, u.r, log(|IP|/50um), available)
  12-15 SV: offsets of the PV->SV direction from k, (s.n, s.r)/10 mrad
        (clipped to +-20), log(L/1mm), available

Variants written for train and validation:
  full        all 16 channels
  ip3         IP channels only (SV channels zero)
  sv          SV channels only (IP channels zero)
  full_shuffle the 16-vector permuted among same-split sides with the same
              reco mode and core-track multiplicity (event link broken)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LADDER = Path("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1")
AUDIT = Path("/home/rbaba/tauspin-ip-sv-analysis-corrected-20260920/geometry_audit.npz")


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--with-track-offset", action="store_true")
    p.add_argument("--ideal-ip", action="store_true",
                   help="oracle: replace each measured IP direction by the truth tau direction perpendicular to that track")
    args = p.parse_args()
    trk = np.load(args.tracks)
    pv, t, n_core = trk["pv"], trk["trk"], trk["n_core"]
    lead = t[:, :, 0]
    ok = np.isfinite(lead[..., 0])
    beam = np.array([*np.median(pv[:, :2], axis=0),
                     -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))])
    theta, phi, d0, z0 = (t[..., i] for i in (1, 2, 4, 5))  # rows, side, track
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip = pca - pv[:, None, None, :]
    ip = ip - np.sum(ip * tdir, -1, keepdims=True) * tdir
    ip_ok = np.isfinite(d0)
    with np.load(AUDIT) as g:
        sv_av = np.asarray(g["sv_available"]) > 0
        sv_dir = np.asarray(g["sv_direction"])
        sv_len = np.asarray(g["sv_length"])
    sv_ok = sv_av & (sv_len > 1e-6) & np.isfinite(sv_dir).all(-1)

    out, report = {}, {"beam_reference_mm": beam.tolist()}
    rng = np.random.default_rng(20260924)
    for split in ("train", "validation"):
        with np.load(LADDER / f"{split}.npz") as d:
            ids = np.asarray(d["global_indices"])
            basis = np.asarray(d["reco_basis"])  # N, side, (n,r,k), xyz
            reco_mode = np.asarray(d["reco_h_mode"])
            tau_true = unit(np.asarray(d["truth_visible_tau_lab4"])[..., :3] + np.asarray(d["truth_neutrino_lab4"])[..., :3])
        W = 6 if args.with_track_offset else 4
        G = 3 * W + 4
        f = np.zeros(ids.shape + (2, G), np.float32)
        for j in range(3):
            v = ip[ids, :, j]
            u = unit(v)
            if args.ideal_ip:
                td0 = tdir[ids, :, j]
                u = unit(tau_true - np.sum(tau_true * td0, -1, keepdims=True) * td0)
            good = ip_ok[ids, :, j] & np.isfinite(u).all(-1)
            cols = [np.sum(u * basis[:, :, 0], -1), np.sum(u * basis[:, :, 1], -1),
                    np.clip(np.log(np.maximum(np.linalg.norm(v, axis=-1), 1e-9) / 0.05), -4, 4),
                    np.ones(ids.shape + (2,))]
            if args.with_track_offset:
                td = tdir[ids, :, j]
                cols += [np.clip(np.sum(td * basis[:, :, 0], -1) / 0.01, -20, 20),
                         np.clip(np.sum(td * basis[:, :, 1], -1) / 0.01, -20, 20)]
            block = np.stack(cols, -1)
            f[..., W * j:W * j + W] = np.where(good[..., None], block, 0.0)
        s = sv_dir[ids]
        good = sv_ok[ids]
        block = np.stack([np.clip(np.sum(s * basis[:, :, 0], -1) / 0.01, -20, 20),
                          np.clip(np.sum(s * basis[:, :, 1], -1) / 0.01, -20, 20),
                          np.clip(np.log(np.maximum(sv_len[ids], 1e-6) / 1.0), -6, 6),
                          np.ones(ids.shape + (2,))], -1)
        f[..., G - 4:G] = np.where(good[..., None], block, 0.0)
        f = np.nan_to_num(f)
        ip3 = f.copy(); ip3[..., G - 4:] = 0
        svo = f.copy(); svo[..., :G - 4] = 0
        sh = f.reshape(-1, G).copy()
        key = reco_mode.reshape(-1).astype(np.int64) * 10 + n_core[ids].reshape(-1).astype(np.int64)
        for k in np.unique(key):
            idx = np.flatnonzero(key == k)
            sh[idx] = sh[rng.permutation(idx)]
        out[f"{split}_global_indices"] = ids
        for name, arr in (("full", f), ("ip3", ip3), ("sv", svo), ("full_shuffle", sh.reshape(f.shape))):
            out[f"{split}_{name}"] = arr
        report[split] = {"rows": int(len(ids)),
                         "ip_track_available_fraction": f[..., [3, 3 + W, 3 + 2 * W]].mean(axis=(0, 1)).tolist(),
                         "sv_available_fraction": float(f[..., G - 1].mean()),
                         "std": f.reshape(-1, G).std(0).round(3).tolist()}
    np.savez_compressed(args.output, **out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
