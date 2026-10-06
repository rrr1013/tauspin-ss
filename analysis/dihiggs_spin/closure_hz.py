"""Closure of the parametric detector + regressor against the tauspin ATLAS
full-simulation results, on the held-out equal-mass H(125)+j / Z(125)+j events.

Compares (i) the per-component correlation of the regressed h with the exact h
and (ii) the H/Z AUC of a fixed readout trained on (h_pred, h-h+ products,
reco modes), 2-fold, with the exact-h likelihood-ratio ceiling.  tauspin
references (ATLAS full sim, 91 GeV H/Z+j): corr(h_k) 0.765, transverse 0.62 / 0.58;
H/Z weighted AUC: point h without IP/SV 0.618, with 3 IP + SV 0.626, exact h 0.719-0.729.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def main(path, out):
    d = np.load(path, allow_pickle=True)
    y = (d["proc"] == "trainH").astype(int)
    h, hp, hh = d["h_exact"], d["h_pred"], d["hh_pred"]
    ok = np.isfinite(h).all((1, 2))
    res = {"n": int(ok.sum()), "n_H": int(y[ok].sum())}
    res["corr"] = {f"side{s}_{c}": float(np.corrcoef(h[ok, s, j], hp[ok, s, j])[0, 1])
                   for s in (0, 1) for j, c in enumerate("nrk")}
    # exact-h likelihood ratio (H: C = diag(1,1,-1); Z: C_kk = +1, P_tau = -0.147)
    nn = h[:, 0, 0] * h[:, 1, 0] + h[:, 0, 1] * h[:, 1, 1]
    kk = h[:, 0, 2] * h[:, 1, 2]
    lr = np.log(1 + nn - kk) - np.log(1 - 0.147 * (h[:, 0, 2] + h[:, 1, 2]) + kk)
    res["auc_exact_LR"] = float(roc_auc_score(y[ok], lr[ok]))
    X = np.concatenate([hp.reshape(-1, 6), hh.reshape(-1, 9),
                        np.eye(5)[d["rmode"]], np.eye(5)[d["rmode1"]]], 1)
    fold = np.arange(len(y)) % 2
    score = np.zeros(len(y))
    for f in (0, 1):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                             early_stopping=True, random_state=0)
        clf.fit(X[fold != f], y[fold != f])
        score[fold == f] = clf.predict_proba(X[fold == f])[:, 1]
    res["auc_readout_hpred"] = float(roc_auc_score(y, score))
    res["auc_readout_hpred_valid_h"] = float(roc_auc_score(y[ok], score[ok]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
