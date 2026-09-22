"""Build per-tau local-frame geometry features for the point-h ablation.

All features are expressed in the reconstructed (n, r, k) basis of each tau
(k = reco visible direction), so the network receives O(1) numbers instead of
lab-frame unit vectors whose informative part is a few mrad.

Channels (4 per tau side, written into tau-token slots 6..9):
  ip_pv       : (u.n, u.r, log(|IP|/50um), available) with u the PV-referenced
                3-D impact-parameter direction of the leading core track
  ip_legacy   : same representation of the beam-spot-z legacy IP vector
  ip_shuffle  : ip_pv permuted among same-split sides with the same reco mode
                and track multiplicity (breaks the event link, keeps marginals)
  oracle      : truth tau direction offsets (tau.n, tau.r)/10 mrad, opening
                angle/10 mrad, 1  -- a representation test, not a measurement
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LADDER = Path("/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1")


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def perp(v, axis):
    return v - np.sum(v * axis, -1, keepdims=True) * axis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    trk = np.load(args.tracks)
    pv, t, n_core = trk["pv"], trk["trk"], trk["n_core"]
    lead = t[:, :, 0]
    bxy = np.median(pv[:, :2], axis=0)
    ok = np.isfinite(lead[..., 0])
    bz = -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))
    beam = np.array([bxy[0], bxy[1], bz])
    theta, phi, d0, z0 = (lead[..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + beam
    ip_pv = perp(pca - pv[:, None, :], tdir)
    ip_leg = perp(np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0 * np.sin(theta)], -1), tdir)

    out = {}
    report = {"beam_reference_mm": beam.tolist()}
    rng = np.random.default_rng(20260923)
    for split in ("train", "validation"):
        with np.load(LADDER / f"{split}.npz") as d:
            ids = np.asarray(d["global_indices"])
            basis = np.asarray(d["reco_basis"])  # [N, side, (n,r,k), xyz]
            vis = np.asarray(d["truth_visible_tau_lab4"])[..., :3]
            nu = np.asarray(d["truth_neutrino_lab4"])[..., :3]
            reco_mode = np.asarray(d["reco_h_mode"])
        tau_hat = unit(vis + nu)

        def local(vec, available, scale_log):
            u = unit(vec[ids])
            f = np.zeros(ids.shape + (2, 4), np.float32)
            f[..., 0] = np.sum(u * basis[:, :, 0], -1)
            f[..., 1] = np.sum(u * basis[:, :, 1], -1)
            mag = np.linalg.norm(vec[ids], axis=-1)
            f[..., 2] = np.clip(np.log(np.maximum(mag, 1e-9) / scale_log), -4, 4)
            f[..., 3] = 1.0
            bad = ~available[ids] | ~np.isfinite(f).all(-1)
            f[bad] = 0.0
            return f

        avail = np.isfinite(d0) & (n_core > 0)
        f_pv = local(ip_pv, avail, 0.05)
        f_leg = local(ip_leg, avail, 0.05)
        # shuffle among sides with same reco mode and n_core within this split
        f_sh = f_pv.copy()
        flat = f_sh.reshape(-1, 4)
        key = (reco_mode.reshape(-1).astype(np.int64) * 10
               + n_core[ids].reshape(-1).astype(np.int64))
        for k in np.unique(key):
            idx = np.flatnonzero(key == k)
            flat[idx] = flat[rng.permutation(idx)]
        f_sh = flat.reshape(f_pv.shape)
        f_or = np.zeros_like(f_pv)
        f_or[..., 0] = np.sum(tau_hat * basis[:, :, 0], -1) / 0.01
        f_or[..., 1] = np.sum(tau_hat * basis[:, :, 1], -1) / 0.01
        f_or[..., 2] = np.arccos(np.clip(np.sum(tau_hat * basis[:, :, 2], -1), -1, 1)) / 0.01
        f_or[..., 3] = 1.0
        f_or = np.clip(f_or, -20, 20)
        for name, f in (("ip_pv", f_pv), ("ip_legacy", f_leg), ("ip_shuffle", f_sh), ("oracle", f_or)):
            out[f"{split}_{name}"] = f
        out[f"{split}_global_indices"] = ids
        report[split] = {
            "rows": int(len(ids)),
            "ip_available_side_fraction": float(f_pv[..., 3].mean()),
            "ip_pv_feature_mean": f_pv.reshape(-1, 4).mean(0).tolist(),
            "ip_pv_feature_std": f_pv.reshape(-1, 4).std(0).tolist(),
            "oracle_feature_std": f_or.reshape(-1, 4).std(0).tolist(),
        }
    np.savez_compressed(args.output, **out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
