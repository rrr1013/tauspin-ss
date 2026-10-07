"""The acoplanarity angle phi*_CP of the LHC H -> tau tau CP analyses, from the reco files.

Standard construction (Berge, Bernreuther, Kirchner et al.; ATLAS arXiv:2212.05833,
CMS arXiv:2110.04836): for each tau a pair (q, lambda) of four-vectors,
  * 1 prong, no pi0 (and 3 prongs): q = leading charged track, lambda = (0, impact-parameter
    vector of that track)                                      [impact-parameter method]
  * 1 prong + pi0(s): q = charged pion, lambda = sum of pi0s, y = (E_pi - E_pi0) / (E_pi + E_pi0)
                                                               [rho decay-plane method]
Both are boosted to the zero-momentum frame of q- + q+, lambda is projected perpendicular to
q, phi* = arccos(l-perp . l+perp), mapped to [0, 2 pi) with the sign of q-hat- . (l+perp x l-perp),
and shifted by pi when the product of the y of the rho-method sides is negative.
Output keyed by the uid of build_dataset.py (side 0 = tau- by reconstructed charge).
"""
import argparse
import glob

import numpy as np


def boost(p, b):
    """Boost four-vectors p (..., 4) [px, py, pz, E] by velocity b (..., 3)."""
    b2 = np.maximum((b * b).sum(-1, keepdims=True), 1e-16)
    g = 1 / np.sqrt(np.maximum(1 - b2, 1e-12))
    bp = (b * p[..., :3]).sum(-1, keepdims=True)
    p3 = p[..., :3] + ((g - 1) * bp / b2 - g * p[..., 3:]) * b
    e = g * (p[..., 3:] - bp)
    return np.concatenate([p3, e], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recodir", required=True)
    ap.add_argument("--proc", default="trainHU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.recodir}/{args.proc}_run_*.npz"))
    uids, phis, sgns, ys, lmag = [], [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        swap = d["q"][:, 0] > 0
        S = lambda a: np.where(swap.reshape((-1,) + (1,) * (a.ndim - 1)), a[:, ::-1], a)
        ch, pi0, ip, rmode = S(d["ch"]), S(d["pi0"]), S(d["ip"]), S(d["rmode"])
        n = len(rmode)
        q = ch[:, :, 0]                                   # leading / charged track (N, 2, 4)
        pi0s = pi0.sum(2)
        rho = (rmode == 1) | (rmode == 2)
        lam_ip = np.concatenate([ip[:, :, 0], np.zeros((n, 2, 1))], -1)
        lam = np.where(rho[..., None], pi0s, lam_ip)
        y = np.where(rho, (q[..., 3] - pi0s[..., 3]) / np.maximum(q[..., 3] + pi0s[..., 3], 1e-9), 1.0)
        zmf = q.sum(1)
        b = zmf[:, :3] / np.maximum(zmf[:, 3:], 1e-9)
        qz = boost(q, b[:, None, :])
        # the impact-parameter vector is a direction: boost it as a 4-vector with zero time component
        lz = boost(lam, b[:, None, :])
        qh = qz[..., :3] / np.maximum(np.linalg.norm(qz[..., :3], axis=-1, keepdims=True), 1e-12)
        lp = lz[..., :3] - (lz[..., :3] * qh).sum(-1, keepdims=True) * qh
        mag = np.linalg.norm(lp, axis=-1)
        lp = lp / np.maximum(mag[..., None], 1e-12)
        c = np.clip((lp[:, 0] * lp[:, 1]).sum(-1), -1, 1)
        phi = np.arccos(c)
        ostar = (qh[:, 0] * np.cross(lp[:, 1], lp[:, 0])).sum(-1)
        phi = np.where(ostar < 0, 2 * np.pi - phi, phi)
        sgn = np.sign(np.where(rho[:, 0], y[:, 0], 1.0) * np.where(rho[:, 1], y[:, 1], 1.0))
        phi = np.where(sgn < 0, np.mod(phi + np.pi, 2 * np.pi), phi)
        valid = (rmode >= 0).all(1) & (mag > 0).all(1)
        uids.append(i * 1_000_000 + d["event"])
        phis.append(np.where(valid, phi, np.nan))
        sgns.append(sgn)
        ys.append(y)
        lmag.append(mag)
    np.savez_compressed(args.out, uid=np.concatenate(uids), phi=np.concatenate(phis), sgn=np.concatenate(sgns),
                        y=np.concatenate(ys), lmag=np.concatenate(lmag))
    print("phi* computed for", len(np.concatenate(uids)), "events;", files[0], "...")


if __name__ == "__main__":
    main()
