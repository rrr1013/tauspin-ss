"""Per-event Z -> tau tau transverse spin state on the spin-flat H(125)+jet events, and
the statistical power of a Z-based calibration of the transverse response kappa.

Z state.  The tau-pair transverse correlation of a spin-1 source is fixed by the rank-2
(tensor) polarisation of the boson.  With the dilepton-type angular distribution in the
Collins-Soper (CS) frame,
    I(k) = 1 + cos^2 th + A0/2 (1 - 3 cos^2 th) + A1 sin 2th cos ph + A2/2 sin^2 th cos 2ph
         = alpha + k^T N k,   N = zz - 3/2 A0 zz + A1 (zx + xz) + A2/2 (xx - yy),
the transverse block in the plane orthogonal to the tau axis k is
    C_perp = -2 sqrt(1 - A_tau^2) [N_perp - tr(N_perp)/2] / I(k)
(reduces to C_nn = -C_rr = 0.989 sin^2 th / (1 + cos^2 th) for A_i = 0).  A0 = A2 (Lam-Tung)
with A0(qT) = qT^2 / (qT^2 + 65^2) (rough fit to the 8 TeV dilepton measurement), A1 = 0:
a statistical-power model, not a precision prediction.  The (n, r) axes are the project's
canonical ones (lab beam, not boosted), so C_perp is rotated per event into that basis.

kappa = data / MC multiplier of the transverse spin response.  Moment estimator with
statistic O(x):  sigma(kappa) = sqrt(Var_flat O) / (sqrt(N_Z) |Cov_flat(O, S_Z)|),
S_Z = sum_ij C_Z,ij h-_i h+_i (transverse part only, truth h).  Statistics compared:
  exact     O = S_Z(h)                          (perfect polarimetry, upper bound)
  reco/true O = sum C_Z,ij(true CS frame) hh_pred_ij   (tauspin estimate, frame known)
  reco/reco O = same with the CS frame and tau axis from reco visible + MET
  canonical O = hh_pred_nn - hh_pred_rr                (no frame information)
Run on ICEPP: reads the reco files of the spin-flat sample to get truth tau and reco
visible / MET four-vectors, joined to the entanglement dataset by uid.
"""
import argparse
import glob
import json

import numpy as np

A_TAU = 0.1469
MZ = 91.1876


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def boost(p, b):
    """Boost four-vectors p (..., 4: px py pz E) into the frame moving with velocity b."""
    b2 = np.sum(b * b, -1, keepdims=True)
    g = 1 / np.sqrt(np.maximum(1 - b2, 1e-12))
    bp = np.sum(b * p[..., :3], -1, keepdims=True)
    g2 = np.where(b2 > 0, (g - 1) / np.maximum(b2, 1e-30), 0)
    sp = p[..., :3] + g2 * bp * b - g * b * p[..., 3:4]
    return np.concatenate([sp, g * (p[..., 3:4] - bp)], -1)


def frame_quantities(tau_m, tau_p):
    """Canonical (n, r, k) and CS axes in the pair frame; tau_m/tau_p lab four-vectors (N, 4)."""
    pair = tau_m + tau_p
    beta = pair[:, :3] / pair[:, 3:4]
    k = unit(boost(tau_p, beta)[:, :3])
    beam = np.array([0., 0., -1.])
    n = unit(np.cross(beam, k))
    r = unit(beam - (k @ beam)[:, None] * k)
    E = 7000.0
    p1 = boost(np.tile([0, 0, E, E], (len(k), 1)), beta)[:, :3]
    p2 = boost(np.tile([0, 0, -E, E], (len(k), 1)), beta)[:, :3]
    z = unit(unit(p1) - unit(p2))
    x = unit(-(unit(p1) + unit(p2)))
    x = unit(x - (x * z).sum(1, keepdims=True) * z)
    y = np.cross(z, x)
    qt = np.hypot(pair[:, 0], pair[:, 1])
    return dict(k=k, n=n, r=r, z=z, x=x, y=y, qt=qt)


def z_transverse(fq):
    """C_perp (N, 2, 2) in the canonical (n, r) basis."""
    k, n, r, z, x, y, qt = (fq[a] for a in ("k", "n", "r", "z", "x", "y", "qt"))
    A0 = qt ** 2 / (qt ** 2 + 65.0 ** 2)
    A2 = A0
    zz = np.einsum("ni,nj->nij", z, z)
    N = zz - 1.5 * A0[:, None, None] * zz + 0.5 * A2[:, None, None] * (np.einsum("ni,nj->nij", x, x) - np.einsum("ni,nj->nij", y, y))
    alpha = 1 + 0.5 * A0
    I = alpha + np.einsum("ni,nij,nj->n", k, N, k)
    B = np.stack([n, r], 1)  # (N, 2, 3)
    Np = np.einsum("nai,nij,nbj->nab", B, N, B)
    tr = 0.5 * (Np[:, 0, 0] + Np[:, 1, 1])
    Np[:, 0, 0] -= tr
    Np[:, 1, 1] -= tr
    return -2 * np.sqrt(1 - A_TAU ** 2) * Np / I[:, None, None], I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recodir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hpred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.recodir}/trainHU_run_*.npz"))
    uid, taus, vis, met, hx, qs = [], [], [], [], [], []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        uid.append(i * 1_000_000 + z["event"])
        taus.append(z["tau_p4"])
        vis.append(z["vis"])
        met.append(z["met"])
        hx.append(z["h_exact"])
        qs.append(z["q"])
    uid, taus, vis, met, hx, qs = map(np.concatenate, (uid, taus, vis, met, hx, qs))
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    pos = {u: j for j, u in enumerate(uid)}
    idx = np.array([pos[u] for u in d["uid"]])
    assert np.array_equal(hp["uid"], d["uid"])
    taus, vis, met, hx, qs = taus[idx], vis[idx], met[idx], hx[idx], qs[idx]
    # h (and hh_pred) are ordered tau-, tau+; the four-vectors follow the reco side order
    swap = qs[:, 0] > 0
    taus = np.where(swap[:, None, None], taus[:, ::-1], taus)
    vis = np.where(swap[:, None, None], vis[:, ::-1], vis)
    print("side swaps:", swap.mean(), " charge-sum zero:", (qs.sum(1) == 0).mean(), flush=True)
    h = d["h_exact"]
    ok = np.isfinite(h).all((1, 2))
    same = np.nanmax(np.abs(hx[ok] - h[ok]))
    print("max |h_reco_file - h_dataset| on matched rows:", same, flush=True)
    tm, tp = taus[:, 0], taus[:, 1]
    fq = frame_quantities(tm, tp)
    C, I = z_transverse(fq)
    # reco frame: visible taus + MET split collinearly (fractions from the transverse system)
    v0, v1 = vis[:, 0], vis[:, 1]
    M = np.stack([v0[:, :2], v1[:, :2]], 2)  # (N, 2, 2) columns = visible pT vectors
    with np.errstate(all="ignore"):
        a = np.linalg.solve(M, met[:, :, None])[:, :, 0]  # MET = a0 v0T + a1 v1T
    a = np.clip(np.nan_to_num(a, nan=0.0), 0, 5)
    t0 = v0 * (1 + a[:, :1])
    t1 = v1 * (1 + a[:, 1:])
    fr = frame_quantities(t0, t1)
    Cr, _ = z_transverse(fr)
    hh = hp["hh_pred"][:, :2, :2]  # predicted h-_i h+_j, canonical (n, r) block
    S = np.einsum("nab,nab->n", C, np.where(ok[:, None, None], h[:, 0, :2, None] * h[:, 1, None, :2], 0.0))
    stats = {
        "exact": S,
        "reco_trueframe": np.einsum("nab,nab->n", C, hh),
        "reco_recoframe": np.einsum("nab,nab->n", Cr, hh),
        "canonical_Q1": hh[:, 0, 0] - hh[:, 1, 1],
    }
    w = d["w"]
    sel = ok
    res = {"events": int(sel.sum()),
           "C_nn_mean_true_frame": float(np.average(C[sel, 0, 0], weights=w[sel])),
           "abs_Cperp_mean": float(np.average(np.abs(C[sel, 0, 0]), weights=w[sel])),
           "frame_agreement_cos2": float(np.average(np.sign(C[sel, 0, 0]) * np.sign(Cr[sel, 0, 0]), weights=w[sel])),
           "per_event": {}}
    ptt = d["kin_pt_tautau"]
    for name, O in stats.items():
        row = {}
        for lab, m in (("all", sel), ("ptt60-120", sel & (ptt >= 60) & (ptt < 120)),
                       ("ptt120-200", sel & (ptt >= 120) & (ptt < 200)), ("ptt>200", sel & (ptt >= 200))):
            ww = w[m] / w[m].sum()
            Oc = O[m] - ww @ O[m]
            cov = float(ww @ (Oc * S[m]))
            var = float(ww @ Oc ** 2)
            # sigma(kappa) * sqrt(N): per-event resolution of the calibration
            row[lab] = {"n": int(m.sum()), "cov": cov, "var": var,
                        "sigma_kappa_sqrtN": float(np.sqrt(var) / abs(cov)) if cov != 0 else None}
        res["per_event"][name] = row
        print(name, {k: (v["n"], round(v["sigma_kappa_sqrtN"], 2)) for k, v in row.items()}, flush=True)
    print("mean C_nn (true frame):", res["C_nn_mean_true_frame"], " mean |C_nn|:", res["abs_Cperp_mean"])
    np.savez_compressed(args.out.replace(".json", "_arrays.npz"), uid=d["uid"], C=C, Cr=Cr, qt=fq["qt"])
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
