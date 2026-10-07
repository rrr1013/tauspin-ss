"""How many reconstructed H -> tau_had tau_had events are needed to establish
spin entanglement, with tauspin-level polarimetry?

Spin model.  The ditau spin density of the Higgs is embedded in the Werner family
    rho_p(h-, h+) = 1 + p (h-_n h+_n + h-_r h+_r - h-_k h+_k),
p = 1 for the CP-even Higgs (a pure Bell state) and p <= 1/3 separable (Peres-Horodecki).
Because rho_p is linear in p, the reco-level density is p0(x) (1 + p g(x)) with
g(x) = E_flat[s | x], s = h-_n h+_n + h-_r h+_r - h-_k h+_k, so g is the optimal
statistic.  It is learned by regression on a spin-flat H(125)+jet sample (unpolarised,
uncorrelated tau decays), whose generator spin density is 1, so every hypothesis is
an exact, bounded reweighting of the same events.  Events without an implemented
polarimeter on both sides keep weight 1 (no spin information credited).

Background.  Z -> tau tau in the Higgs mass window, with the Z spin state
(C_kk = +1, P_tau = -0.147, optionally C_nn = +0.49, C_rr = -0.46), modelled on the
same kinematics (the reweighted spin-flat events).

Test.  Asimov data = S T_1 + B T_Z (binned in g).  Null: p = 1/3 (separability
boundary).  Signal and background yields float in both fits (shape-only spin test),
with an optional Gaussian constraint on B.  Z_ent = sqrt(q).  The script scans S at
fixed S/B and reports the S needed for 3 and 5 sigma.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

HYPS = ("H", "W13", "Z", "U")


def spin_s(h):
    hm, hp = h[:, 0], h[:, 1]
    return hm[:, 0] * hp[:, 0] + hm[:, 1] * hp[:, 1] - hm[:, 2] * hp[:, 2]


def z_target(h, z_transverse):
    """rho_Z - 1 with one-side marginals; 0 when no polarimeter."""
    one = np.isfinite(h).all(-1)
    hz = np.where(one[..., None], h, 0.0)
    hm, hp = hz[:, 0], hz[:, 1]
    t = -0.147 * (hm[:, 2] + hp[:, 2]) + np.where(one.all(1), hm[:, 2] * hp[:, 2], 0.0)
    if z_transverse:
        t = t + np.where(one.all(1), 0.49 * hm[:, 0] * hp[:, 0] - 0.46 * hm[:, 1] * hp[:, 1], 0.0)
    return t


def weights(h, z_transverse):
    ok = np.isfinite(h).all((1, 2))
    hz = np.where(ok[:, None, None], h, 0.0)
    s = spin_s(hz)
    hm, hp = hz[:, 0], hz[:, 1]
    kk = hm[:, 2] * hp[:, 2]
    z = 1 - 0.147 * (hm[:, 2] + hp[:, 2]) + kk
    if z_transverse:
        z = z + 0.49 * hm[:, 0] * hp[:, 0] - 0.46 * hm[:, 1] * hp[:, 1]
    one = np.isfinite(h).all(-1)
    only = one[:, 0] ^ one[:, 1]
    k1 = np.where(one[:, 0], hz[:, 0, 2], hz[:, 1, 2])
    W = {"H": np.where(ok, 1 + s, 1.0), "W13": np.where(ok, 1 + s / 3, 1.0),
         "Z": np.where(ok, z, np.where(only, 1 - 0.147 * k1, 1.0)), "U": np.ones(len(h))}
    return W, s, ok


def nll_fit(n, T_sig, T_bkg, S0, B0, b_unc):
    """Profile S and B (B optionally constrained) for fixed signal template."""
    def f(par):
        S, B = np.exp(par)
        lam = np.maximum(S * T_sig + B * T_bkg, 1e-12)
        pen = 0.0 if b_unc is None else 0.5 * ((B - B0) / (b_unc * B0)) ** 2
        return float((lam - n * np.log(lam)).sum() + pen)
    r = minimize(f, np.log([S0, B0]), method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=8000))
    return r.fun


def z_ent(T, S, B, b_unc=None):
    n = S * T["H"] + B * T["Z"]
    l1 = nll_fit(n, T["H"], T["Z"], S, B, b_unc)
    l0 = nll_fit(n, T["W13"], T["Z"], S, B, b_unc)
    return float(np.sqrt(max(2 * (l0 - l1), 0.0)))


def templates(G, W, nq=8):
    """2D quantile binning of (g_s, g_z)."""
    e1 = np.quantile(G[:, 0], np.linspace(0, 1, nq + 1)[1:-1])
    b1 = np.searchsorted(e1, G[:, 0])
    b = b1 * nq
    for i in range(nq):
        m = b1 == i
        e2 = np.quantile(G[m, 1], np.linspace(0, 1, nq + 1)[1:-1]) if m.sum() > nq else np.array([])
        b[m] += np.searchsorted(e2, G[m, 1])
    nb = nq * nq
    return {k: np.bincount(b, weights=W[k], minlength=nb) / W[k].sum() for k in HYPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset of the spin-flat H sample (build_dataset.py)")
    ap.add_argument("--hpred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-transverse", action="store_true")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    h = d["h_exact"]
    W, s, ok = weights(h, args.z_transverse)
    rmode = np.stack([d[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)
    feats = {
        "tauspin": np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode], 1),
        "textbook": np.concatenate([np.stack([d[f"spin_{v}_{q}"] for q in (0, 1)
                                               for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")], 1), rmode], 1),
    }
    fold = (np.arange(len(h)) % 2)
    res = {"n_events": int(len(h)), "frac_both_h": float(ok.mean()), "z_transverse": args.z_transverse,
           "levels": {}}
    zt = z_target(h, args.z_transverse)
    gs = {"exact": np.stack([np.where(ok, s, 0.0), zt], 1)}
    for name, X in feats.items():
        G = np.zeros((len(h), 2))
        for j, target in enumerate((np.where(ok, s, 0.0), zt)):
            for f in (0, 1):
                reg = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                                    l2_regularization=1.0, early_stopping=True, random_state=0)
                reg.fit(X[fold != f], target[fold != f])
                G[fold == f, j] = reg.predict(X[fold == f])
        gs[name] = G
    for name, G in gs.items():
        T = templates(G, W)
        g = G[:, 0]
        info = float(np.average(g ** 2 / np.maximum(1 + g, 1e-3) ** 2, weights=W["H"]))
        scan = {}
        for sb, bunc in ((None, None), (1.0, 0.02), (0.3, 0.02), (0.1, 0.02), (0.3, 0.05), (0.1, 0.05)):
            rows = []
            for S in np.geomspace(30, 3e6, 70):
                if sb is None:
                    n = S * T["H"]
                    def fit(Tsig):
                        r = minimize(lambda a: float((np.exp(a[0]) * Tsig - n * np.log(np.maximum(np.exp(a[0]) * Tsig, 1e-12))).sum()),
                                     [np.log(S)], method="Nelder-Mead")
                        return r.fun
                    z = float(np.sqrt(max(2 * (fit(T["W13"]) - fit(T["H"])), 0)))
                else:
                    z = z_ent(T, S, S / sb, bunc)
                rows.append((S, z))
            rows = np.array(rows)
            s3 = float(np.interp(3, rows[:, 1], rows[:, 0])) if rows[:, 1].max() > 3 else None
            s5 = float(np.interp(5, rows[:, 1], rows[:, 0])) if rows[:, 1].max() > 5 else None
            key = "inf" if sb is None else f"SB{sb}_dB{bunc}"
            scan[key] = {"S_for_3sigma": s3, "S_for_5sigma": s5, "curve": rows.tolist()}
        res["levels"][name] = {"fisher_info_per_event": info, "scan": scan,
                               "corr_gs_exact": float(np.corrcoef(G[:, 0], gs["exact"][:, 0])[0, 1])}
        print(name, "I =", round(info, 4), {k: (None if v["S_for_3sigma"] is None else round(v["S_for_3sigma"]),
                                                None if v["S_for_5sigma"] is None else round(v["S_for_5sigma"])) for k, v in scan.items()}, flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
