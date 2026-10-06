"""Why the phase-2 spin gain is smaller than phase 1: decay-mode coverage.
Pairwise spin-only AUCs and R on the hold-out (unmatched), for all events and for
events whose reco modes are 1p0n / 1p1n / 3p0n on both sides (the phase-1 cohort)."""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from factorized import TOPBIN
from factorized_rw import HYPS, evaluate
from matched_rw import rho_all, rmode_onehot

ho = np.load("outputs/v1/train_holdout_slim.npz", allow_pickle=True)
hp = np.load("outputs/v1/hpred_partial_holdout.npz", allow_pickle=True)
h = ho["h_exact"]; ok = np.isfinite(h).all((1, 2)); isH = ho["proc"] == "trainH"
hz = h[ok & ~isH]
C_z = 9 * np.einsum("ni,nj->ij", hz[:, 0], hz[:, 1]) / len(hz)
B_z = 3 * (np.mean(h[ok & ~isH][:, :, 2]) - np.mean(h[ok & isH][:, :, 2]))
rho, zgen = rho_all(np.where(ok[:, None, None], h, 0.0), C_z, B_z)
nH, nZ = (ok & isH).sum(), (ok & ~isH).sum()
base = (nH * rho["H"] + nZ * zgen) / (nH + nZ)
W = {X: np.where(ok, rho[X] / np.maximum(base, 1e-6), 1.0) for X in HYPS}
rm = rmode_onehot(ho)
rmode = np.stack([rm[:, :5].argmax(1), rm[:, 5:].argmax(1)], 1)
good = np.isin(rmode, [0, 1, 3]).all(1)
tot = sum(v for k, v in TOPBIN.items() if k != "signal")
frac = {G: TOPBIN[G] / tot for G in ("Z+HF", "top", "fakes", "singleH")}
X = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm], 1)
res = {"frac_good_modes": float(good.mean()), "frac_exact_defined": float(ok.mean())}
for tag, sel in (("all", np.ones(len(X), bool)), ("pi_rho_3p_both", good)):
    Xs = X[sel]; Ws = {k: v[sel] for k, v in W.items()}
    probs = np.zeros((len(Xs), 4))
    for f in (0, 1):
        fold = np.arange(len(Xs)) % 2 == f
        Xt = np.concatenate([Xs[~fold]] * 4); yt = np.repeat(np.arange(4), (~fold).sum())
        wt = np.concatenate([Ws[Y][~fold] / Ws[Y][~fold].sum() for Y in HYPS]) * (~fold).sum()
        c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=200, early_stopping=True,
                                           random_state=0).fit(Xt, yt, sample_weight=wt)
        probs[fold] = c.predict_proba(Xs[fold])
    aucs = {}
    for i, Y in enumerate(HYPS[1:], 1):
        sc = np.log(probs[:, 0] + 1e-12) - np.log(probs[:, i] + 1e-12)
        aucs[Y] = float(roc_auc_score(np.r_[np.ones(len(sc)), np.zeros(len(sc))], np.r_[sc, sc],
                                      sample_weight=np.r_[Ws["H"], Ws[Y]]))
    z = evaluate(probs, Ws, np.ones(len(Xs), bool), frac)[0]
    dens = np.stack([np.where(ok[sel], [rho[Y][sel] for Y in HYPS][i], 1.0) for i in range(4)], 1)
    ze = evaluate(dens, Ws, np.ones(len(Xs), bool), frac)[0]
    res[tag] = {"auc_H_vs": aucs, "R_syst_tauspin": z["R_syst"], "R_syst_exact": ze["R_syst"]}
    print(tag, json.dumps(res[tag]))
print(res["frac_good_modes"], res["frac_exact_defined"])
json.dump(res, open("outputs/v1/diag_modes.json", "w"), indent=1)
