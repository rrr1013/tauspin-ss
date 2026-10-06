"""Event classification and expected significance with and without tau-spin
information, HH -> bb tau_had tau_had, 3 ab^-1, 14 TeV.

Feature sets (all share the ATLAS-like kinematic set K):
  K          ATLAS Run-2 style kinematics (m_HH, m_tautau(MMC), m_bb, dR, pT, eta,
             m_T, MET, MET centrality, ...)
  K+obs      K + textbook spin observables without neutrino regression
             (Upsilon, x = E_vis/E_tau(MMC), hybrid h from reco visibles + MMC nu)
  K+low      K + every low-level spin input of the h regressor (no h bottleneck)
  K+h        K + tauspin-style regressed h (6) and h-h+ products (9)
  K+exact    K + exact h and products (oracle; a fake tau gets an isotropic h)
  K[mtt x%]  K with m_tautau(MMC) replaced by the true m_tautau smeared by a
             Gaussian of x % (requirement curve for neutrino reconstruction)
Classifier: XGBoost, signal (ggF HH) vs the sum of backgrounds weighted to
their expected yields and rescaled to the signal total; trained on split 1,
evaluated on split 2 (weights x3).

Statistics: the classifier score is binned in 20 quantiles of the signal;
significance of mu_HH = 1 from the Asimov formula (stat only) and from the
profiled Fisher information with Gaussian normalisation nuisances per
background group (Z+HF 10 %, top 10 %, fakes 20 %, single H 15 %).
ATLAS calibration: per-group normalisation factors make the composition of the
most signal-like region of the K classifier (top 30 % of the signal) equal to
arXiv:2209.10910 Table 5 (tau_had tau_had most signal-like bin); the same
factors are applied to every feature set.  R = Z(K+X) / Z(K).
"""
import argparse
import json

import numpy as np

GROUPS = {"zbb": "Z+HF", "ttll": "top", "ttlj": "fakes", "zh": "singleH", "tth": "singleH"}
UNC = {"Z+HF": 0.10, "top": 0.10, "fakes": 0.20, "singleH": 0.15}
# arXiv:2209.10910 Table 5, tau_had tau_had most signal-like bin (139 fb^-1): B_group / S
ATLAS_TOPBIN = {"Z+HF": (2.3 + 0.4) / 1.60, "top": (0.5 + 0.47) / 1.60, "fakes": (0.47 + 0.29) / 1.60,
                "singleH": 1.7 / 1.60}


def asimov_z(s, b):
    m = b > 0
    s, b = s[m], b[m]
    return float(np.sqrt(np.sum(2 * ((s + b) * np.log1p(s / b) - s))))


def fisher_z(s, bg):
    nu = s + sum(bg.values())
    m = nu > 0
    derivs = [s[m]] + [bg[g][m] * UNC[g] for g in bg]
    n = len(derivs)
    F = np.array([[np.sum(derivs[i] * derivs[j] / nu[m]) for j in range(n)] for i in range(n)])
    F[1:, 1:] += np.eye(n - 1)
    return float(1 / np.sqrt(np.linalg.inv(F)[0, 0]))


def feature_sets(d, hp):
    kin = sorted(k for k in d.files if k.startswith("kin_"))
    K = np.stack([d[k] for k in kin], 1)
    spin_cols = sorted(k for k in d.files if k.startswith("spin_"))
    low = np.stack([d[k] for k in spin_cols], 1)
    obs_cols = [k for k in spin_cols if any(t in k for t in ("upsilon", "_x_", "hhyb", "rmode"))]
    obs = np.stack([d[k] for k in obs_cols], 1)
    H = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9)], 1)
    h = d["h_exact"].copy()
    rng = np.random.default_rng(5)
    iso = rng.standard_normal(h.shape)
    iso /= np.linalg.norm(iso, axis=-1, keepdims=True)
    h = np.where(np.isfinite(h), h, iso)          # fakes / undefined modes: no spin information
    E = np.concatenate([h.reshape(-1, 6), np.einsum("ni,nj->nij", h[:, 0], h[:, 1]).reshape(-1, 9)], 1)
    sets = {}
    mtt_true = d["m_tautau_true"]
    i_mtt = kin.index("kin_m_tautau")
    rng2 = np.random.default_rng(11)
    for res in (0.15, 0.10, 0.05):
        Km = K.copy()
        smeared = mtt_true * (1 + res * rng2.standard_normal(len(mtt_true)))
        # a fake tau has no true ditau mass: keep the MMC value there
        Km[:, i_mtt] = np.where(d["is_true"].all(1), smeared, K[:, i_mtt])
        sets[f"K[mtt {int(res * 100)}%]"] = Km
    sets["K[mtt 10%]+h"] = np.concatenate([sets["K[mtt 10%]"], H], 1)
    sets["K[mtt 10%]+exact"] = np.concatenate([sets["K[mtt 10%]"], E], 1)
    return {**sets, "K": K, "K+obs": np.concatenate([K, obs], 1), "K+low": np.concatenate([K, low], 1),
            "K+h": np.concatenate([K, H], 1), "K+exact": np.concatenate([K, E], 1)}


def train_score(X, y, w, tr, ev, seed=0):
    import xgboost as xgb
    ws = w.copy()
    ws[tr & (y == 1)] /= ws[tr & (y == 1)].sum()
    ws[tr & (y == 0)] /= ws[tr & (y == 0)].sum()
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(tr)
    rng.shuffle(idx)
    nv = len(idx) // 5
    dtr = xgb.DMatrix(X[idx[nv:]], label=y[idx[nv:]], weight=ws[idx[nv:]] * len(idx))
    dva = xgb.DMatrix(X[idx[:nv]], label=y[idx[:nv]], weight=ws[idx[:nv]] * len(idx))
    params = dict(objective="binary:logistic", eval_metric="logloss", eta=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  tree_method="hist", seed=seed, nthread=16)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1)), bst.best_iteration


def binned(score, y, w, groups, ev, nbins=20):
    s_ev = ev & (y == 1)
    o = np.argsort(score[s_ev])
    cdf = np.cumsum(w[s_ev][o]) / w[s_ev].sum()
    qs = np.interp(np.linspace(0, 1, nbins + 1)[1:-1], cdf, score[s_ev][o])
    b = np.searchsorted(qs, score)
    S = np.bincount(b[s_ev], weights=w[s_ev], minlength=nbins)
    BG = {}
    for g in sorted(set(groups) - {"signal"}):
        m = ev & (groups == g)
        BG[g] = np.bincount(b[m], weights=w[m], minlength=nbins)
    return S, BG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hpred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    assert (hp["uid"] == d["uid"]).all() and (hp["proc"] == d["proc"]).all()
    proc = d["proc"]
    y = (proc == "hh").astype(int)
    groups = np.array([GROUPS.get(p, "signal") for p in proc])
    split = d["uid"] % 3
    w = d["w"].astype(float).copy()
    tr, ev = split == 1, split == 2
    w_ev = np.where(ev, w * 3, 0.0)
    sets = feature_sets(d, hp)
    res = {"yields_eval": {g: float(w_ev[groups == g].sum()) for g in set(groups)}, "sets": {}}
    scores = {}
    for name, X in sets.items():
        sc, it = train_score(X.astype(np.float32), y, w, tr, ev, args.seed)
        scores[name] = sc
        print(name, "trees", it, flush=True)
    # ATLAS calibration factors from the K classifier's most signal-like region (top 30 % of signal)
    sk = scores["K"]
    s_ev = ev & (y == 1)
    cut = np.interp(0.70, np.cumsum(w_ev[s_ev][np.argsort(sk[s_ev])]) / w_ev[s_ev].sum(), np.sort(sk[s_ev]))
    top = ev & (sk >= cut)
    S_top = w_ev[top & (y == 1)].sum()
    calib = {}
    for g in UNC:
        ours = w_ev[top & (groups == g)].sum() / S_top
        calib[g] = ATLAS_TOPBIN[g] / ours if ours > 0 else 1.0
    res["calibration"] = {"cut_K": float(cut), "S_top": float(S_top),
                          "ours_B_over_S": {g: float(w_ev[top & (groups == g)].sum() / S_top) for g in UNC},
                          "factor": calib}
    for name, sc in scores.items():
        out = {}
        for tag, scale in (("raw", {g: 1.0 for g in UNC}), ("atlas_calibrated", calib)):
            ww = w_ev * np.array([scale.get(g, 1.0) for g in groups])
            S, BG = binned(sc, y, ww, groups, ev)
            B = sum(BG.values())
            out[tag] = {"Z_stat": asimov_z(S, B), "Z_syst": fisher_z(S, BG),
                        "S": float(S.sum()), "B": float(B.sum()),
                        "top_bin": {"S": float(S[-1]), **{g: float(BG[g][-1]) for g in BG}}}
        res["sets"][name] = out
    for tag in ("raw", "atlas_calibrated"):
        zk = res["sets"]["K"][tag]
        for name in sets:
            o = res["sets"][name][tag]
            o["R_stat"] = o["Z_stat"] / zk["Z_stat"]
            o["R_syst"] = o["Z_syst"] / zk["Z_syst"]
    np.savez_compressed(args.out.replace(".json", "_scores.npz"), uid=d["uid"], proc=proc,
                        **{k.replace("+", "_"): v for k, v in scores.items()})
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    for name in sets:
        a = res["sets"][name]["atlas_calibrated"]
        print(f"{name:8s} Z_stat {a['Z_stat']:.3f} Z_syst {a['Z_syst']:.3f} R_stat {a['R_stat']:.4f} R_syst {a['R_syst']:.4f}")


if __name__ == "__main__":
    main()
