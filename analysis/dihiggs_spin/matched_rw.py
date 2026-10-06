"""Spin-only gain at HH signal-like tau kinematics with the realistic
(closure-validated) reconstruction, using the regressor hold-out events.

Hold-out events (equal-mass H(125)+j / Z(125)+j, never used to train the
regressor) are
  1. matched to the tau kinematics of the HH events above a K-score cut
     (signal efficiency 100 / 50 / 20 %) with a classifier density ratio on
     pT, eta of both visible taus, m_vis, MET, dR(tau, tau) and reco modes;
  2. reweighted to the spin hypotheses H, Z, W, U with the pooled generator
     density base = (n_H rho_H + n_Z rho_Zgen) / n, which stays bounded where
     rho_H -> 0 (the importance-weight blow-up of factorized_rw.py).
rho_H is the scalar CP-even density; rho_Zgen is measured on the Z hold-out
(C = 9 <h- h+^T>, B from the Z - H shift of <h_k>), since Pythia's Z+jet
carries production-dependent transverse correlations.
Then a spin classifier on (h_pred, h-h+ products, reco modes), 2-fold, gives
templates; significance with the arXiv:2209.10910 Table 5 composition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from factorized import TOPBIN
from factorized_rw import HYPS, evaluate

MATCH = ["kin_pt_tau1", "kin_pt_tau2", "kin_eta_tau1", "kin_eta_tau2", "kin_m_vis", "kin_met", "kin_dr_tautau"]


def rho_all(h, C_z, B_z):
    hm, hp = h[:, 0], h[:, 1]
    nn, rr, kk = hm[:, 0] * hp[:, 0], hm[:, 1] * hp[:, 1], hm[:, 2] * hp[:, 2]
    km, kp = hm[:, 2], hp[:, 2]
    zgen = 1 + B_z * (km + kp) + np.einsum("ni,ij,nj->n", hm, C_z, hp)
    return {"H": 1 + nn + rr - kk, "Z": 1 - 0.147 * (km + kp) + kk, "W": (1 - km) * (1 - kp),
            "U": np.ones_like(kk)}, np.maximum(zgen, 0.0)


def rmode_onehot(d):
    return np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/v1")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    R = Path(args.run)
    ho = np.load(R / "train_holdout_slim.npz", allow_pickle=True)
    hp = np.load(R / "hpred_partial_holdout.npz", allow_pickle=True)
    d = np.load(R / "dataset.npz", allow_pickle=True)
    ks = np.load(R / "classify_s0_scores.npz", allow_pickle=True)
    h = ho["h_exact"]
    ok = np.isfinite(h).all((1, 2))
    isH = ho["proc"] == "trainH"
    hz = h[ok & ~isH]
    C_z = 9 * np.einsum("ni,nj->ij", hz[:, 0], hz[:, 1]) / len(hz)
    B_z = 3 * (np.mean(h[ok & ~isH][:, :, 2]) - np.mean(h[ok & isH][:, :, 2]))
    rho, zgen = rho_all(np.where(ok[:, None, None], h, 0.0), C_z, B_z)
    nH, nZ = (ok & isH).sum(), (ok & ~isH).sum()
    base = (nH * rho["H"] + nZ * zgen) / (nH + nZ)
    Wspin = {X: np.where(ok, rho[X] / np.maximum(base, 1e-6), 1.0) for X in HYPS}
    out = {"C_zgen": C_z.tolist(), "B_zgen": float(B_z), "regions": {}}
    print("C_zgen diag", np.diag(C_z).round(3), "B_z", round(B_z, 3))
    # HH target kinematics
    hh = d["proc"] == "hh"
    K = ks["K"]
    o = np.argsort(K[hh])
    cdf = np.cumsum(d["w"][hh][o]) / d["w"][hh].sum()
    Xm_ho = np.concatenate([np.stack([ho[c] for c in MATCH], 1), rmode_onehot(ho)], 1)
    Xm_hh = np.concatenate([np.stack([d[c] for c in MATCH], 1), rmode_onehot(d)], 1)
    Xs = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode_onehot(ho)], 1)
    cols_obs = [f"spin_{v}_{s}" for s in (0, 1) for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")]
    Xo = np.concatenate([np.stack([ho[c] for c in cols_obs], 1), rmode_onehot(ho)], 1)
    tot = sum(v for k, v in TOPBIN.items() if k != "signal")
    frac = {G: TOPBIN[G] / tot for G in ("Z+HF", "top", "fakes", "singleH")}
    for eff in (None, 1.0, 0.5, 0.2):
        if eff is None:
            wm = np.ones(len(Xm_ho))
            tag = "unmatched"
        else:
            kcut = -np.inf if eff == 1.0 else float(np.interp(1 - eff, cdf, K[hh][o]))
            tgt = hh & (K >= kcut)
            Xc = np.concatenate([Xm_hh[tgt], Xm_ho])
            yc = np.r_[np.ones(tgt.sum()), np.zeros(len(Xm_ho))]
            wc = np.r_[np.full(tgt.sum(), len(Xm_ho) / tgt.sum()), np.ones(len(Xm_ho))]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                                 early_stopping=True, random_state=0)
            clf.fit(Xc, yc, sample_weight=wc)
            p = clf.predict_proba(Xm_ho)[:, 1]
            wm = np.clip(p / np.maximum(1 - p, 1e-6), 0, np.quantile(p / np.maximum(1 - p, 1e-6), 0.999))
            tag = f"sig_eff_{int(eff * 100)}"
        wm = wm / wm.mean()
        W = {X: wm * Wspin[X] for X in HYPS}
        allrows = np.ones(len(wm), bool)
        ess = {X: float(W[X].sum() ** 2 / (W[X] ** 2).sum()) for X in HYPS}
        dens = np.stack([np.where(ok, rho[X], 1.0) for X in HYPS], 1)
        res = {"ess": ess, "exact": evaluate(dens, W, allrows, frac)[0]}
        for name, X in (("tauspin", Xs), ("obs", Xo)):
            zs = []
            for seed in range(args.seeds):
                probs = np.zeros((len(X), len(HYPS)))
                for f in (0, 1):
                    fold = np.arange(len(X)) % 2 == f
                    tr = ~fold
                    Xt = np.concatenate([X[tr]] * len(HYPS))
                    yt = np.repeat(np.arange(len(HYPS)), tr.sum())
                    wt = np.concatenate([W[Y][tr] / W[Y][tr].sum() for Y in HYPS]) * tr.sum()
                    c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=200,
                                                       l2_regularization=1.0, early_stopping=True,
                                                       validation_fraction=0.2, n_iter_no_change=30,
                                                       random_state=seed)
                    c.fit(Xt, yt, sample_weight=wt)
                    probs[fold] = c.predict_proba(X[fold])
                zs.append(evaluate(probs, W, allrows, frac)[0])
            res[name] = {"seeds": zs, "R_syst_mean": float(np.mean([z["R_syst"] for z in zs])),
                         "R_syst_seed_sd": float(np.std([z["R_syst"] for z in zs], ddof=1)),
                         "R_stat_mean": float(np.mean([z["R_stat"] for z in zs]))}
        out["regions"][tag] = res
        print(f"{tag:12s} ESS {({k: round(v) for k, v in ess.items()})}  exact R {res['exact']['R_syst']:.4f}  "
              f"obs R {res['obs']['R_syst_mean']:.4f}  tauspin R {res['tauspin']['R_syst_mean']:.4f} "
              f"± {res['tauspin']['R_syst_seed_sd']:.4f}", flush=True)
    json.dump(out, open(R / "matched_rw.json", "w"), indent=1)


if __name__ == "__main__":
    main()
