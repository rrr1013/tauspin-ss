"""Factorised spin gain on the phase-2 MC.

The direct K vs K+spin classifier comparison is limited by background MC
statistics in the signal-like region (ttbar needs ~1e-4 rejection; the
evaluation split keeps ~0-2 MC events there).  Here the kinematic part is
taken from ATLAS (arXiv:2209.10910 Table 5, most signal-like tau_had tau_had
bin, scaled to 3 ab^-1 at 14 TeV) and only the spin dimension comes from the
MC:

  1. a spin-only multiclass classifier (signal / Z+HF / top / fakes / single H)
     on spin inputs only, trained on split 1 with equal class weights;
  2. per-group templates of D = p_sig / sum_g f_g p_g on split 2;
  3. Asimov / profiled-Fisher significance of the spin-binned top bin, R.
Factorisation is tested by comparing each group's D distribution across
slices of the kinematic (K) classifier score where MC statistics allow.

Spin input sets: exact (oracle; fakes get an isotropic h), tauspin (regressed
h, h-h+ products, reco modes), obs (Upsilon, x, hybrid h, reco modes),
modes (reco decay modes only: the null reference for decay-mode composition).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from classify import asimov_z, fisher_z

GROUPS = ["signal", "Z+HF", "top", "fakes", "singleH"]
PROC_GROUP = {"hh": "signal", "zbb": "Z+HF", "ttll": "top", "ttlj": "fakes", "zh": "singleH", "tth": "singleH"}
UNC = {"Z+HF": 0.10, "top": 0.10, "fakes": 0.20, "singleH": 0.15}
LUMI = 3000 / 139
# Table 5 tau_had tau_had most signal-like bin x 14 TeV factor: S and B per group
TOPBIN = {"signal": (1.58 + 0.0227) * 1.18, "Z+HF": (2.3 + 0.4) * 1.18, "top": (0.5 + 0.47) * 1.18,
          "fakes": (0.47 + 0.29) * 1.18, "singleH": 1.7 * 1.15}


def spin_inputs(d, hp, which):
    rmode = np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)
    if which == "modes":
        return rmode
    if which == "tauspin":
        return np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode], 1)
    if which == "obs":
        cols = [f"spin_{v}_{s}" for s in (0, 1) for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")]
        return np.concatenate([np.stack([d[c] for c in cols], 1), rmode], 1)
    if which == "exact":
        h = d["h_exact"].copy()
        rng = np.random.default_rng(5)
        iso = rng.standard_normal(h.shape)
        iso /= np.linalg.norm(iso, axis=-1, keepdims=True)
        h = np.where(np.isfinite(h), h, iso)
        return np.concatenate([h.reshape(-1, 6), np.einsum("ni,nj->nij", h[:, 0], h[:, 1]).reshape(-1, 9), rmode], 1)
    raise ValueError(which)


def fit_probs(X, g, w, tr, seed):
    ws = np.zeros(len(w))
    for k in range(len(GROUPS)):
        m = tr & (g == k)
        ws[m] = w[m] / w[m].sum() * tr.sum() / len(GROUPS)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                         n_iter_no_change=30, random_state=seed)
    clf.fit(X[tr], g[tr], sample_weight=ws[tr])
    return clf.predict_proba(X)


def discriminant(p, frac):
    den = sum(frac[G] * p[:, k] for k, G in enumerate(GROUPS) if G != "signal")
    return np.log(p[:, 0] + 1e-12) - np.log(den + 1e-12)


def templates(D, g, w, sel, edges):
    b = np.searchsorted(edges, D)
    T = {}
    for k, G in enumerate(GROUPS):
        m = sel & (g == k)
        T[G] = np.bincount(b[m], weights=w[m], minlength=len(edges) + 1) / w[m].sum()
    return T


def significance(T):
    s = TOPBIN["signal"] * LUMI * T["signal"]
    bg = {G: TOPBIN[G] * LUMI * T[G] for G in UNC}
    one = {G: np.array([TOPBIN[G] * LUMI]) for G in UNC}
    s1 = np.array([TOPBIN["signal"] * LUMI])
    z = {"Z_stat_1bin": asimov_z(s1, sum(one.values())), "Z_stat": asimov_z(s, sum(bg.values())),
         "Z_syst_1bin": fisher_z(s1, one), "Z_syst": fisher_z(s, bg)}
    z["R_stat"] = z["Z_stat"] / z["Z_stat_1bin"]
    z["R_syst"] = z["Z_syst"] / z["Z_syst_1bin"]
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nboot", type=int, default=200)
    args = ap.parse_args()
    R = Path(args.run)
    d = np.load(R / "dataset.npz", allow_pickle=True)
    hp = np.load(R / "hpred.npz", allow_pickle=True)
    ks = np.load(R / "classify_s0_scores.npz", allow_pickle=True)
    assert (hp["uid"] == d["uid"]).all() and (ks["uid"] == d["uid"]).all()
    g = np.array([GROUPS.index(PROC_GROUP[p]) for p in d["proc"]])
    w = d["w"].astype(float)
    split = d["uid"] % 3
    tr, ev = split == 1, split == 2
    tot = sum(v for k, v in TOPBIN.items() if k != "signal")
    frac = {G: TOPBIN[G] / tot for G in UNC}
    # kinematic-score slices defined by signal efficiency of the K classifier
    K = ks["K"]
    sig_ev = ev & (g == 0)
    o = np.argsort(K[sig_ev])
    cdf = np.cumsum(w[sig_ev][o]) / w[sig_ev].sum()
    cut = np.interp([0.5, 0.8, 0.95], cdf, K[sig_ev][o])
    kslice = np.searchsorted(cut, K)       # 0: <50 %, 1: 50-80 %, 2: 80-95 %, 3: top 5 % of signal
    out = {"composition_fractions": frac, "kslice_cuts": cut.tolist(), "sets": {}}
    rng = np.random.default_rng(1)
    store = {}
    for which in ("modes", "obs", "tauspin", "exact"):
        X = spin_inputs(d, hp, which)
        res = {"seeds": []}
        for seed in range(args.seeds):
            p = fit_probs(X, g, w, tr, seed)
            D = discriminant(p, frac)
            # bins: 20 quantiles of the composition-weighted background on the evaluation split
            wb = sum(frac[G] * np.where(g == GROUPS.index(G), w / w[ev & (g == GROUPS.index(G))].sum(), 0) for G in UNC)
            m = ev & (g > 0)
            oo = np.argsort(D[m])
            c = np.cumsum(wb[m][oo]) / wb[m].sum()
            edges = np.interp(np.linspace(0, 1, 21)[1:-1], c, D[m][oo])
            T = templates(D, g, w, ev, edges)
            z = significance(T)
            aucs = {G: float(roc_auc_score(np.r_[np.ones((ev & (g == 0)).sum()), np.zeros((ev & (g == k)).sum())],
                                           np.r_[D[ev & (g == 0)], D[ev & (g == k)]],
                                           sample_weight=np.r_[w[ev & (g == 0)], w[ev & (g == k)]]))
                    for k, G in enumerate(GROUPS) if G != "signal"}
            boots = []
            idx_ev = np.flatnonzero(ev)
            for _ in range(args.nboot if seed == 0 else 0):
                bi = rng.choice(idx_ev, len(idx_ev))
                mask = np.zeros(len(w))
                np.add.at(mask, bi, 1)
                Tb = templates(D, g, w * mask, mask > 0, edges)
                boots.append(significance(Tb)["R_syst"])
            # factorisation test: D templates by K slice, per group
            slices = {}
            for k, G in enumerate(GROUPS):
                rows = []
                for s_ in range(4):
                    mm = ev & (g == k) & (kslice == s_)
                    if mm.sum() < 30:
                        rows.append({"n_mc": int(mm.sum())})
                        continue
                    Ts = templates(D, g, w, mm, edges)[G]
                    rows.append({"n_mc": int(mm.sum()), "mean_D": float(np.average(D[mm], weights=w[mm])),
                                 "template": Ts.tolist()})
                slices[G] = rows
            # R recomputed with the templates of the top-two K slices where available
            T_hi = {}
            for k, G in enumerate(GROUPS):
                mm = ev & (g == k) & (kslice >= 2)
                T_hi[G] = templates(D, g, w, mm, edges)[G] if mm.sum() >= 100 else T[G]
            z_hi = significance(T_hi)
            res["seeds"].append({"seed": seed, **z, "auc_vs_signal": aucs, "R_syst_hiK": z_hi["R_syst"],
                                 "hiK_groups_used": [G for k, G in enumerate(GROUPS) if (ev & (g == k) & (kslice >= 2)).sum() >= 100],
                                 "R_syst_boot_sd": float(np.std(boots)) if boots else None,
                                 "kslices": slices, "templates": {G: T[G].tolist() for G in GROUPS}})
            if seed == 0:
                store[which] = D
            print(which, seed, f"R_syst {z['R_syst']:.4f} (hiK {z_hi['R_syst']:.4f})",
                  {G: round(a, 3) for G, a in aucs.items()}, flush=True)
        rs = np.array([s["R_syst"] for s in res["seeds"]])
        res["R_syst_mean"], res["R_syst_seed_sd"] = float(rs.mean()), float(rs.std(ddof=1))
        out["sets"][which] = res
    json.dump(out, open(R / "factorized.json", "w"), indent=1)
    np.savez_compressed(R / "factorized_D.npz", uid=d["uid"], group=g, kslice=kslice, **store)


if __name__ == "__main__":
    main()
