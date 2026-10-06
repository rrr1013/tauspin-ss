"""Spin-only gain at signal-like kinematics, by spin reweighting of the H -> tau tau
events (ggF HH, ZH, ttH) of the phase-2 MC.

The direct spin classifier on real background MC (factorized.py) mixes in
kinematic information: its discriminant on the signal shifts with the
kinematic (K) score, while the exact-h one does not.  Here kinematics are held
fixed by construction: the same H -> tau tau events are reweighted to each
background spin hypothesis (rho_X / rho_H on the exact h; events whose decay
modes have no polarimeter keep weight 1), so a classifier can only use spin.
Hypotheses: H (signal, single H), Z (C_kk = +1, P_tau = -0.147),
W (tau-_L tau+_R product, true-tau top), U (no spin information; fakes).
Region: the events of split 2 above a cut on the K classifier score (signal
efficiency 100 %, 50 %, 20 %), i.e. the kinematics the ATLAS top bin selects.
Composition: arXiv:2209.10910 Table 5, as in factorized.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from classify import asimov_z, fisher_z
from factorized import LUMI, TOPBIN, UNC

HYPS = ("H", "Z", "W", "U")
GROUP_HYP = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}


def densities(h):
    hm, hp = h[:, 0], h[:, 1]
    nn, rr, kk = hm[:, 0] * hp[:, 0], hm[:, 1] * hp[:, 1], hm[:, 2] * hp[:, 2]
    km, kp = hm[:, 2], hp[:, 2]
    return {"H": 1 + nn + rr - kk, "Z": 1 - 0.147 * (km + kp) + kk, "W": (1 - km) * (1 - kp),
            "U": np.ones_like(kk)}


def features(d, hp, which):
    rmode = np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)
    if which == "tauspin":
        return np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode], 1)
    if which == "obs":
        cols = [f"spin_{v}_{s}" for s in (0, 1) for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")]
        return np.concatenate([np.stack([d[c] for c in cols], 1), rmode], 1)
    raise ValueError(which)


def significance(T):
    s = TOPBIN["signal"] * LUMI * T["H"]
    bg = {G: TOPBIN[G] * LUMI * T[GROUP_HYP[G]] for G in UNC}
    one = {G: np.array([TOPBIN[G] * LUMI]) for G in UNC}
    s1 = np.array([TOPBIN["signal"] * LUMI])
    z = {"Z_syst_1bin": fisher_z(s1, one), "Z_syst": fisher_z(s, bg),
         "Z_stat_1bin": asimov_z(s1, sum(one.values())), "Z_stat": asimov_z(s, sum(bg.values()))}
    z["R_syst"] = z["Z_syst"] / z["Z_syst_1bin"]
    z["R_stat"] = z["Z_stat"] / z["Z_stat_1bin"]
    return z


def evaluate(probs, W, sel, frac, nbins=20):
    p = probs / probs.sum(1, keepdims=True)
    den = sum(frac[G] * p[:, HYPS.index(GROUP_HYP[G])] for G in UNC)
    D = np.log(p[:, 0] + 1e-12) - np.log(den + 1e-12)
    wb = sum(frac[G] * W[GROUP_HYP[G]] for G in UNC)
    o = np.argsort(D[sel])
    c = np.cumsum(wb[sel][o]) / wb[sel].sum()
    edges = np.interp(np.linspace(0, 1, nbins + 1)[1:-1], c, D[sel][o])
    b = np.searchsorted(edges, D)
    T = {X: np.bincount(b[sel], weights=W[X][sel], minlength=nbins) / W[X][sel].sum() for X in HYPS}
    return significance(T), D


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
    keep = np.isin(d["proc"], ["hh", "zh", "tth"])
    K = ks["K"][keep]
    split = (d["uid"] % 3)[keep]
    w_evt = d["w"][keep].astype(float)
    h = d["h_exact"][keep]
    ok = np.isfinite(h).all((1, 2))
    rho = densities(np.where(ok[:, None, None], h, 0.0))
    W = {X: np.where(ok, rho[X] / np.maximum(rho["H"], 1e-6), 1.0) for X in HYPS}
    # signal-efficiency cuts on K from HH events of split 2
    hh = (d["proc"][keep] == "hh") & (split == 2)
    o = np.argsort(K[hh])
    cdf = np.cumsum(w_evt[hh][o]) / w_evt[hh].sum()
    tot = sum(v for k, v in TOPBIN.items() if k != "signal")
    frac = {G: TOPBIN[G] / tot for G in UNC}
    out = {"n_events": int(keep.sum()), "frac_exact_h_defined": float(ok.mean()), "regions": {}}
    rng = np.random.default_rng(3)
    for eff in (1.0, 0.5, 0.2):
        kcut = -np.inf if eff == 1.0 else float(np.interp(1 - eff, cdf, K[hh][o]))
        region = K >= kcut
        res = {"k_cut": kcut, "n_eval": int((region & (split == 2)).sum())}
        ev = region & (split == 2)
        ess = {X: float(W[X][ev].sum() ** 2 / (W[X][ev] ** 2).sum()) for X in HYPS}
        res["ess"] = ess
        # analytic exact-h likelihood ratio
        dens = np.stack([np.where(ok, rho[X], 1.0) for X in HYPS], 1)
        res["exact"] = evaluate(dens, W, ev, frac)[0]
        for which in ("obs", "tauspin"):
            X = features(d, hp, which)[keep]
            rs = []
            for seed in range(args.seeds):
                probs = np.zeros((len(X), len(HYPS)))
                for f in (0, 1):                  # 2-fold within the region, all splits used
                    fold = (np.arange(len(X)) % 2 == f)
                    trn = region & ~fold
                    Xs = np.concatenate([X[trn]] * len(HYPS))
                    ys = np.repeat(np.arange(len(HYPS)), trn.sum())
                    ws = np.concatenate([W[Y][trn] / W[Y][trn].sum() for Y in HYPS]) * trn.sum()
                    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=100,
                                                         l2_regularization=1.0, early_stopping=True,
                                                         validation_fraction=0.2, n_iter_no_change=30,
                                                         random_state=seed)
                    clf.fit(Xs, ys, sample_weight=ws)
                    probs[region & fold] = clf.predict_proba(X[region & fold])
                z, D = evaluate(probs, W, region, frac)
                if seed == 0 and args.nboot:
                    idx = np.flatnonzero(region)
                    bs = []
                    for _ in range(args.nboot):
                        bi = rng.choice(idx, len(idx))
                        m = np.zeros(len(X), bool)
                        cnt = np.bincount(bi, minlength=len(X)).astype(float)
                        Wb = {Y: W[Y] * cnt for Y in HYPS}
                        bs.append(evaluate(probs, Wb, cnt > 0, frac)[0]["R_syst"])
                    z["R_syst_boot_sd"] = float(np.std(bs))
                rs.append(z)
            res[which] = {"seeds": rs, "R_syst_mean": float(np.mean([z["R_syst"] for z in rs])),
                          "R_syst_seed_sd": float(np.std([z["R_syst"] for z in rs], ddof=1))}
        out["regions"][f"sig_eff_{int(eff * 100)}"] = res
        print(f"eff {eff}: n_eval {res['n_eval']} ESS {({k: round(v) for k, v in ess.items()})} "
              f"exact R {res['exact']['R_syst']:.4f}  obs R {res['obs']['R_syst_mean']:.4f}  "
              f"tauspin R {res['tauspin']['R_syst_mean']:.4f} ± {res['tauspin']['R_syst_seed_sd']:.4f}", flush=True)
    json.dump(out, open(R / "factorized_rw.json", "w"), indent=1)


if __name__ == "__main__":
    main()
