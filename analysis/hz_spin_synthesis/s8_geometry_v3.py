"""v3 geometry: v2 plus the measured IP error ellipse (from s6, TRAIN-fitted).

The full IP likelihood of this run (s3 'ipg') showed that the IP magnitude and
the strongly anisotropic transverse/longitudinal errors carry information the
azimuth alone does not.  v3 gives the network, per core track (W = 9):
  (b.n, b.r)/50um            IP vector in the local frame, magnitude kept (clip +-10)
  sigma_T/50um, sigma_L/50um  per-track error along the track-frame axes
  cos 2g, sin 2g             orientation of that error ellipse in the (n, r) plane
  available, (t.n, t.r)/10mrad  as v2
then the SV block (4).  GEO_DIM = 31.

p10 docstring follows.
Full per-tau detector geometry in the local reco (n, r, k) frame.

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

RESOLUTION = Path("/home/rbaba/hz-spin-synthesis-20260923/artifacts/s6_ip_resolution.json")

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
        table = json.loads(RESOLUTION.read_text())
        pe, ee = np.array(table["bins_pt"]), np.array(table["bins_eta"])
        sT_tab, sL_tab = np.array(table["sigma_T_um"], float), np.array(table["sigma_L_um"], float)
        W = 9
        G = 3 * W + 4
        f = np.zeros(ids.shape + (2, G), np.float32)
        for j in range(3):
            v = ip[ids, :, j]
            td = tdir[ids, :, j]
            good = ip_ok[ids, :, j] & np.isfinite(v).all(-1)
            eT = unit(np.cross(np.broadcast_to([0.0, 0.0, 1.0], td.shape), td))
            pt = t[ids, :, j, 0]
            eta = np.abs(-np.log(np.tan(np.clip(theta[ids, :, j], 1e-6, np.pi - 1e-6) / 2)))
            ib = np.clip(np.digitize(np.nan_to_num(pt), pe) - 1, 0, len(pe) - 2)
            kb = np.clip(np.digitize(np.nan_to_num(eta), ee) - 1, 0, len(ee) - 2)
            g = np.arctan2(np.sum(eT * basis[:, :, 1], -1), np.sum(eT * basis[:, :, 0], -1))
            cols = [np.clip(np.sum(v * basis[:, :, 0], -1) / 0.05, -10, 10),
                    np.clip(np.sum(v * basis[:, :, 1], -1) / 0.05, -10, 10),
                    sT_tab[ib, kb] / 50.0, sL_tab[ib, kb] / 50.0, np.cos(2 * g), np.sin(2 * g),
                    np.ones(ids.shape + (2,)),
                    np.clip(np.sum(td * basis[:, :, 0], -1) / 0.01, -20, 20),
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
                         "ip_track_available_fraction": f[..., [6, 6 + W, 6 + 2 * W]].mean(axis=(0, 1)).tolist(),
                         "sv_available_fraction": float(f[..., G - 1].mean()),
                         "std": f.reshape(-1, G).std(0).round(3).tolist()}
    np.savez_compressed(args.output, **out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
