"""Closure of the spin reweighting on the same HH events.

The spin-flat HH sample reweighted by rho_H must reproduce the real
(spin-correlated) HH sample made from the same LHE files: compare exact-h
moments, the distributions of the regressed h components, and the template of
the spin-only discriminant (classifier trained on the spin-flat sample,
applied to both).  Weighted chi2 per bin with sumw2 errors.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from spin_gain_U import HYPS, TOPBIN, UNC, densities, disc, edges_for, hyp_weights


def chi2(a, wa, b, wb, edges):
    ha, _ = np.histogram(a, edges, weights=wa)
    va, _ = np.histogram(a, edges, weights=wa ** 2)
    hb, _ = np.histogram(b, edges, weights=wb)
    vb, _ = np.histogram(b, edges, weights=wb ** 2)
    na, nb = wa.sum(), wb.sum()
    m = (va + vb) > 0
    c = ((ha / na - hb / nb) ** 2 / (va / na ** 2 + vb / nb ** 2))[m]
    return float(c.sum()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/v2")
    args = ap.parse_args()
    R = args.run
    u = np.load(f"{R}/dataset_hhU.npz", allow_pickle=True)
    uhp = np.load(f"{R}/hpred_hhU.npz", allow_pickle=True)
    r = np.load(f"{R}/dataset_hhreal.npz", allow_pickle=True)
    rhp = np.load(f"{R}/hpred_hhreal.npz", allow_pickle=True)
    hu, hr = u["h_exact"], r["h_exact"]
    oku, okr = np.isfinite(hu).all((1, 2)), np.isfinite(hr).all((1, 2))
    W, rho, _ = hyp_weights(hu, False)
    out = {}
    Cu = 9 * np.einsum("ni,nj->ij", hu[oku, 0] * W["H"][oku, None], hu[oku, 1]) / W["H"][oku].sum()
    Cr = 9 * np.einsum("ni,nj->ij", hr[okr, 0], hr[okr, 1]) / okr.sum()
    out["C_flat_to_H"] = np.diag(Cu).tolist()
    out["C_real_H"] = np.diag(Cr).tolist()
    out["n_flat"], out["n_real"] = int(len(hu)), int(len(hr))
    out["frac_selected_weighted_flat_to_H"] = float(W["H"].mean())
    # regressed h components
    e = np.linspace(-1.2, 1.2, 25)
    comp = {}
    for s in (0, 1):
        for j, c in enumerate("nrk"):
            comp[f"side{s}_{c}"] = chi2(uhp["h_pred"][:, s, j], W["H"], rhp["h_pred"][:, s, j], np.ones(len(hr)), e)
    out["chi2_hpred_components"] = comp
    # spin discriminant: train on the flat sample (all events), apply to both
    rm = lambda d: np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)
    Xu = np.concatenate([uhp["h_pred"].reshape(-1, 6), uhp["hh_pred"].reshape(-1, 9), rm(u)], 1)
    Xr = np.concatenate([rhp["h_pred"].reshape(-1, 6), rhp["hh_pred"].reshape(-1, 9), rm(r)], 1)
    Xt = np.concatenate([Xu] * len(HYPS))
    yt = np.repeat(np.arange(len(HYPS)), len(Xu))
    wt = np.concatenate([W[Y] / W[Y].sum() for Y in HYPS]) * len(Xu)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                                         early_stopping=True, random_state=0).fit(Xt, yt, sample_weight=wt)
    tot = sum(v for k, v in TOPBIN.items() if k != "signal")
    frac = {G: TOPBIN[G] / tot for G in UNC}
    fmap = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}
    Du = disc(clf.predict_proba(Xu), frac, fmap)
    Dr = disc(clf.predict_proba(Xr), frac, fmap)
    ed = np.r_[-np.inf, edges_for(Du, W, np.ones(len(Du), bool), frac, fmap), np.inf]
    out["chi2_D"] = chi2(Du, W["H"], Dr, np.ones(len(Dr)), ed)
    # out-of-fold closure inside the K region at 20 % signal efficiency
    Ku = np.load(f"{R}/dataset_hhU_K.npz", allow_pickle=True)["K"]
    Kr = np.load(f"{R}/dataset_hhreal_K.npz", allow_pickle=True)["K"]
    o = np.argsort(Ku)
    cut = np.interp(0.8, np.cumsum(W["H"][o]) / W["H"].sum(), Ku[o])
    ru, rr = Ku >= cut, Kr >= cut
    fold = np.zeros(len(Xu), int)
    fold[np.flatnonzero(ru)] = np.arange(ru.sum()) % 2
    tr = ru & (fold == 0)
    c2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=100, early_stopping=True,
                                        random_state=1).fit(np.concatenate([Xu[tr]] * len(HYPS)),
                                                            np.repeat(np.arange(len(HYPS)), tr.sum()),
                                                            sample_weight=np.concatenate([W[Y][tr] / W[Y][tr].sum() for Y in HYPS]) * tr.sum())
    te = ru & (fold == 1)
    Du2 = disc(c2.predict_proba(Xu[te]), frac, fmap)
    Dr2 = disc(c2.predict_proba(Xr[rr]), frac, fmap)
    Wte = {Y: W[Y][te] for Y in HYPS}
    ed2 = np.r_[-np.inf, edges_for(Du2, Wte, np.ones(te.sum(), bool), frac, fmap, nbins=15), np.inf]
    out["region20_oof_chi2_D"] = chi2(Du2, W["H"][te], Dr2, np.ones(rr.sum()), ed2)
    out["region20_n_flat_test"], out["region20_n_real"] = int(te.sum()), int(rr.sum())
    out["region20_real_eff"] = float(rr.mean())
    np.savez_compressed(f"{R}/closure_U_D.npz", Du=Du, Dr=Dr, wH=W["H"], wZ=W["Z"], wW=W["W"], wU=W["U"],
                        hpu=uhp["h_pred"], hpr=rhp["h_pred"])
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{R}/closure_U.json", "w"), indent=1)


if __name__ == "__main__":
    main()
