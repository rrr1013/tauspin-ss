"""Spin gain versus the background composition of the signal-like bin.

Templates: hold-out events matched to the HH tau kinematics above the K cut at
20 % signal efficiency, reweighted to the spin hypotheses (as matched_rw.py),
exact-h analytic likelihood ratio and the tauspin regressor classifier.
For each composition the discriminant D = p_H / sum_b f_b p_b is rebuilt and
the stat-only Asimov significance gain R of the spin-binned bin is computed at
the S/B of the ATLAS top bin (S = 40.8, B = 155 at 3 ab^-1).
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from classify import asimov_z
from factorized_rw import HYPS
from matched_rw import MATCH, rho_all, rmode_onehot

S_TOT, B_TOT = 40.8, 155.0
GH = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}


def gain(probs, W, frac, nbins=20):
    p = probs / probs.sum(1, keepdims=True)
    den = sum(f * p[:, HYPS.index(GH[G])] for G, f in frac.items())
    D = np.log(p[:, 0] + 1e-12) - np.log(den + 1e-12)
    wb = sum(f * W[GH[G]] for G, f in frac.items())
    o = np.argsort(D)
    c = np.cumsum(wb[o]) / wb.sum()
    e = np.interp(np.linspace(0, 1, nbins + 1)[1:-1], c, D[o])
    b = np.searchsorted(e, D)
    T = {X: np.bincount(b, weights=W[X], minlength=nbins) / W[X].sum() for X in HYPS}
    s = S_TOT * T["H"]
    bb = sum(B_TOT * f * T[GH[G]] for G, f in frac.items())
    return asimov_z(s, bb) / asimov_z(np.array([S_TOT]), np.array([B_TOT]))


def main():
    ho = np.load("outputs/v1/train_holdout_slim.npz", allow_pickle=True)
    hp = np.load("outputs/v1/hpred_partial_holdout.npz", allow_pickle=True)
    d = np.load("outputs/v1/dataset.npz", allow_pickle=True)
    ks = np.load("outputs/v1/classify_s0_scores.npz", allow_pickle=True)
    h = ho["h_exact"]
    ok = np.isfinite(h).all((1, 2))
    isH = ho["proc"] == "trainH"
    hz = h[ok & ~isH]
    C_z = 9 * np.einsum("ni,nj->ij", hz[:, 0], hz[:, 1]) / len(hz)
    B_z = 3 * (np.mean(h[ok & ~isH][:, :, 2]) - np.mean(h[ok & isH][:, :, 2]))
    rho, zgen = rho_all(np.where(ok[:, None, None], h, 0.0), C_z, B_z)
    nH, nZ = (ok & isH).sum(), (ok & ~isH).sum()
    base = (nH * rho["H"] + nZ * zgen) / (nH + nZ)
    # kinematic matching to HH above the 20 % signal-efficiency K cut
    hh = d["proc"] == "hh"
    K = ks["K"]
    o = np.argsort(K[hh])
    cdf = np.cumsum(d["w"][hh][o]) / d["w"][hh].sum()
    tgt = hh & (K >= np.interp(0.8, cdf, K[hh][o]))
    Xm_ho = np.concatenate([np.stack([ho[c] for c in MATCH], 1), rmode_onehot(ho)], 1)
    Xm_hh = np.concatenate([np.stack([d[c] for c in MATCH], 1), rmode_onehot(d)], 1)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                         early_stopping=True, random_state=0)
    clf.fit(np.concatenate([Xm_hh[tgt], Xm_ho]), np.r_[np.ones(tgt.sum()), np.zeros(len(Xm_ho))],
            sample_weight=np.r_[np.full(tgt.sum(), len(Xm_ho) / tgt.sum()), np.ones(len(Xm_ho))])
    pr = clf.predict_proba(Xm_ho)[:, 1]
    r = pr / np.maximum(1 - pr, 1e-6)
    wm = np.clip(r, 0, np.quantile(r, 0.999))
    wm /= wm.mean()
    W = {X: wm * np.where(ok, rho[X] / np.maximum(base, 1e-6), 1.0) for X in HYPS}
    dens = np.stack([np.where(ok, rho[X], 1.0) for X in HYPS], 1)
    X = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode_onehot(ho)], 1)
    probs = np.zeros((len(X), 4))
    for f in (0, 1):
        fold = np.arange(len(X)) % 2 == f
        Xt = np.concatenate([X[~fold]] * 4)
        yt = np.repeat(np.arange(4), (~fold).sum())
        wt = np.concatenate([W[Y][~fold] / W[Y][~fold].sum() for Y in HYPS]) * (~fold).sum()
        c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=True, random_state=0)
        c.fit(Xt, yt, sample_weight=wt)
        probs[fold] = c.predict_proba(X[fold])
    out = {"pure": {}, "scan_top": [], "table5": {}}
    t5 = {"Z+HF": 2.7 * 1.18, "top": 0.97 * 1.18, "fakes": 0.76 * 1.18, "singleH": 1.7 * 1.15}
    t5 = {k: v / sum(t5.values()) for k, v in t5.items()}
    for name, P in (("exact", dens), ("tauspin", probs)):
        out["table5"][name] = gain(P, W, t5)
        out["pure"][name] = {G: gain(P, W, {G: 1.0}) for G in GH}
    # replace the Z+HF share of the table-5 mix progressively by top
    for x in np.linspace(0, 1, 11):
        fr = dict(t5)
        moved = x * t5["Z+HF"]
        fr["Z+HF"] -= moved
        fr["top"] += moved
        out["scan_top"].append({"top_fraction": fr["top"], "exact": gain(dens, W, fr), "tauspin": gain(probs, W, fr)})
    print(json.dumps({k: out[k] for k in ("table5", "pure")}, indent=1))
    json.dump(out, open("outputs/v1/scan_composition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
