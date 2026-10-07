"""Final HL-LHC projection for observing spin entanglement in H -> tau_had tau_had.

Ingredients (see entanglement_feasibility.py, ent_hllhc.py, ent_closure_syst.py):
  * spin-flat H(125)+jet sample, cross-fitted regressors of (g_s, g_z) for each
    reconstruction level: exact h, tauspin regressor (nominal), tauspin regressor
    without IP/SV inputs, textbook observables;
  * per-category templates restricted to the category's pT(tautau) range (the
    information per event falls with the boost): boosted bins use their pT(H)
    range, VBF / VH / ttH use 60-200 GeV;
  * ATLAS Run-2 tau_had tau_had yields (arXiv:2201.08269), 3 ab^-1 at 14 TeV, split in
    nine m_tautau bins with acceptances from our simulation;
  * Asimov test p = 1 vs p = 1/3 (separability boundary of the Werner family, i.e. the
    witness C_nn + C_rr - C_kk > 1), signal free, background normalisation +-2 %,
    optional response-scale nuisance (10 %, 20 %);
  * background spin template: spin-flat reweighted to Z (default) or the real
    Z(125)+jet hold-out (variant, tauspin level only).
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ent_closure_syst import z_ent_shape
from ent_hllhc import CATS, LUMI
from entanglement_feasibility import weights, z_target

PT_RANGE = {"VBF_0": (60, 200), "VBF_1": (60, 200), "VH_0": (60, 200), "VH_1": (60, 200),
            "ttH_0": (60, 200), "ttH_1": (60, 200),
            "boost_100_120_1j": (100, 120), "boost_100_120_2j": (100, 120),
            "boost_120_200_1j": (120, 200), "boost_120_200_2j": (120, 200),
            "boost_200_300": (200, 300), "boost_300_inf": (300, 5000)}


def binning(G, nq=6):
    e1 = np.quantile(G[:, 0], np.linspace(0, 1, nq + 1)[1:-1])
    b1 = np.searchsorted(e1, G[:, 0])
    e2 = [np.quantile(G[b1 == i, 1], np.linspace(0, 1, nq + 1)[1:-1]) for i in range(nq)]

    def idx(X):
        a = np.searchsorted(e1, X[:, 0])
        out = a * nq
        for i in range(nq):
            m = a == i
            out[m] += np.searchsorted(e2[i], X[m, 1])
        return out
    return idx, nq * nq


def crossfit(X, targets, fold):
    G = np.zeros((len(X), len(targets)))
    for j, t in enumerate(targets):
        for f in (0, 1):
            r = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                              l2_regularization=1.0, early_stopping=True, random_state=0)
            r.fit(X[fold != f], t[fold != f])
            G[fold == f, j] = r.predict(X[fold == f])
    return G


def main():
    d = np.load("outputs/ent/dataset_HU.npz", allow_pickle=True)
    hp = np.load("outputs/ent/hpred_HU.npz", allow_pickle=True)
    hn = np.load("outputs/ent/hpred_noipsv.npz", allow_pickle=True)
    h = d["h_exact"]
    W, s, ok = weights(h, False)
    zt = z_target(h, False)
    targets = (np.where(ok, s, 0.0), zt)
    rm = np.stack([d[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)
    fold = np.arange(len(h)) % 2
    levels = {"exact": np.stack(targets, 1),
              "tauspin": crossfit(np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm], 1), targets, fold),
              "tauspin_noIPSV": crossfit(np.concatenate([hn["h_pred"].reshape(-1, 6), hn["hh_pred"].reshape(-1, 9), rm], 1), targets, fold),
              "textbook": crossfit(np.concatenate([np.stack([d[f"spin_{v}_{q}"] for q in (0, 1)
                                                             for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")], 1), rm], 1), targets, fold)}
    ptt = d["kin_pt_tautau"]
    acc = json.load(open("outputs/ent/mass_acc.json"))
    res = {"levels": {}, "info_vs_pt": {}}
    for name, G in levels.items():
        idx, nb = binning(G)
        b = idx(G)
        g = G[:, 0]
        res["info_vs_pt"][name] = {}
        for c, (lo, hi) in PT_RANGE.items():
            m = (ptt >= lo) & (ptt < hi)
            res["info_vs_pt"][name][c] = float(np.average(g[m] ** 2 / np.maximum(1 + g[m], 1e-3) ** 2, weights=W["H"][m]))
        row = {}
        for sig in (0.0, 0.1, 0.2):
            per = {}
            for c, (s_, zz, f, t, o) in CATS.items():
                lo, hi = PT_RANGE[c]
                m = (ptt >= lo) & (ptt < hi)
                T = {k: np.bincount(b[m], weights=W[k][m], minlength=nb) / W[k][m].sum() for k in ("H", "W13", "Z", "U")}
                z2 = 0.0
                for i in range(len(acc["S"])):
                    S = s_ * LUMI * 1.13 * acc["S"][i]
                    B = (zz * acc["Z"][i] + f * acc["F"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15
                    z2 += z_ent_shape(T, S, B, 0.02, sig) ** 2
                per[c] = float(np.sqrt(z2))
            row[f"resp{sig}"] = {"per_category": per, "combined": float(np.sqrt(sum(v ** 2 for v in per.values())))}
        res["levels"][name] = row
        print(name, {k: round(v["combined"], 2) for k, v in row.items()}, flush=True)
    json.dump(res, open("outputs/ent/ent_final.json", "w"), indent=1)
    np.savez_compressed("outputs/ent/ent_final_G.npz", **{k: v for k, v in levels.items()}, ptt=ptt,
                        **{f"w_{k}": W[k] for k in ("H", "W13", "Z", "U")})


if __name__ == "__main__":
    main()
