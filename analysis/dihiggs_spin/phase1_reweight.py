"""Phase 1: spin-only separation of HH signal from bb tautau backgrounds,
measured on the ATLAS full-simulation validation cohort (59,390 events) by
TauSpinner-style reweighting.

Every event carries the exact polarimeter h (generator current for 3pi) and the
learned point-h prediction h_pred of a fixed network.  The ditau spin density in
the common (n, r, k) basis is
    rho_X(h-, h+) ∝ 1 + B_X^- . h- + B_X^+ . h+ + h-^T C_X h+ ,
so the same events, reweighted by rho_X / rho_pool, describe any spin hypothesis
X at fixed kinematics.  Hypotheses (sign convention checked on the data:
<h_k>_Z - <h_k>_H < 0 on both sides):
    H   (signal, single Higgs) : C = diag(1, 1, -1), B = 0
    Z   (Z+HF, ZZ)             : C = diag(0, 0, 1),  B^± = P_tau k, P_tau = -0.147
    W   (true-tau top)         : (1 - h-_k)(1 - h+_k)        [W- -> tau-_L, W+ -> tau+_R]
    U   (multijet fakes proxy) : 1                            [no spin information]
    WU  (ttbar fake proxy)     : 1/2[(1 - h-_k) + (1 - h+_k)] [one W tau, one unpolarised]
The pooled H/Z cohort was generated with rho_H (label 1) and rho_Zgen (label 0,
tree-level sin^2 theta_W -> P_tau = -0.22), so rho_pool = (rho_H + rho_Zgen)/2.
A per-sample importance weight (rho_X / rho_own) is kept as a cross-check.

A weighted multiclass gradient-boosting classifier on (h_pred, reco modes) is
trained with 2-fold cross-fitting; templates of the composition-optimal
discriminant D = p_H / sum_b f_b p_b are then evaluated out of fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path(__file__).resolve().parents[1] / "cp_mixing" / "data"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"

HYPS = ("H", "Z", "W", "U", "WU")
P_TAU_Z_PHYS = -0.147
P_TAU_Z_GEN = -0.2196

ARMS = {
    "exact": None,                       # analytic likelihood ratio on exact h
    "reco_noIPSV": "gen3pi_base_s43.npz",
    "reco_IPSV": "gen3pi_full22_s42.npz",
    "reco_idealIP": "gen3pi_idealip22_s42.npz",
}


def densities(h):
    hm, hp = h[:, 0], h[:, 1]
    nn = hm[:, 0] * hp[:, 0]
    rr = hm[:, 1] * hp[:, 1]
    kk = hm[:, 2] * hp[:, 2]
    km, kp = hm[:, 2], hp[:, 2]
    rho = {
        "H": 1 + nn + rr - kk,
        "Z": 1 + P_TAU_Z_PHYS * (km + kp) + kk,
        "W": (1 - km) * (1 - kp),
        "U": np.ones_like(kk),
        "WU": 0.5 * ((1 - km) + (1 - kp)),
    }
    zgen = 1 + P_TAU_Z_GEN * (km + kp) + kk
    return rho, zgen


def hyp_weights(h, labels, scheme):
    rho, zgen = densities(h)
    if scheme == "pool":
        base = 0.5 * (rho["H"] + zgen)
    else:  # per-sample importance weights
        base = np.where(labels == 1, rho["H"], zgen)
        base = np.maximum(base, 1e-3)
    w = {k: v / base for k, v in rho.items()}
    for k in w:
        w[k] = w[k] / w[k].mean()
    return w


def features(h_pred, reco_modes):
    oh = np.concatenate([(reco_modes[:, s:s + 1] == m).astype(np.float32)
                         for s in (0, 1) for m in (0, 1, 3)], axis=1)
    return np.concatenate([h_pred.reshape(len(h_pred), 6), oh], axis=1)


def crossfit_probs(X, w, fold, seed=0):
    """Out-of-fold class probabilities under equal class priors."""
    probs = np.zeros((len(X), len(HYPS)))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        Xs = np.concatenate([X[tr]] * len(HYPS))
        ys = np.concatenate([np.full(tr.sum(), i) for i in range(len(HYPS))])
        ws = np.concatenate([w[k][tr] / w[k][tr].sum() for k in HYPS]) * tr.sum()
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            random_state=seed)
        clf.fit(Xs, ys, sample_weight=ws)
        probs[te] = clf.predict_proba(X[te])
    return probs


def analytic_probs(h):
    rho, _ = densities(h)
    p = np.stack([rho[k] for k in HYPS], 1)
    return p / p.sum(1, keepdims=True)


def weighted_auc(score, w_sig, w_bkg):
    o = np.argsort(score)
    s, ws, wb = score[o], w_sig[o], w_bkg[o]
    cb = np.cumsum(wb) - 0.5 * wb
    # ties are negligible for continuous scores
    return float((ws * cb).sum() / (ws.sum() * wb.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="pool", choices=["pool", "own"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    lad = np.load(DATA / "ladder_validation.npz")
    ref = np.load(DATA / "gen3pi_full22_s42.npz")
    h = ref["h"].astype(np.float64)
    labels = ref["labels"].astype(int)
    reco_modes = lad["reco_h_mode"].astype(int)
    gidx = ref["global_indices"]
    assert (lad["global_indices"] == gidx).all()
    fold = (gidx % 2).astype(int)
    w = hyp_weights(h, labels, args.scheme)

    result = {"scheme": args.scheme, "n_events": int(len(h)), "arms": {}}
    store = {"labels": labels, "truth_modes": ref["truth_modes"],
             "reco_modes": reco_modes, "fold": fold}
    for k in HYPS:
        store[f"w_{k}"] = w[k]
    for arm, fname in ARMS.items():
        if fname is None:
            probs = analytic_probs(h)
        else:
            d = np.load(DATA / fname)
            assert (d["global_indices"] == gidx).all()
            probs = crossfit_probs(features(d["h_pred"].astype(np.float64), reco_modes), w, fold)
        store[f"probs_{arm}"] = probs
        aucs = {}
        for i, k in enumerate(HYPS[1:], start=1):
            score = np.log(probs[:, 0] + 1e-12) - np.log(probs[:, i] + 1e-12)
            aucs[f"H_vs_{k}"] = weighted_auc(score, w["H"], w[k])
        result["arms"][arm] = {"pairwise_auc": aucs}
        print(arm, {k: round(v, 4) for k, v in aucs.items()}, flush=True)

    np.savez_compressed(OUT / f"probs_{args.scheme}.npz", **store)
    (OUT / f"pairwise_auc_{args.scheme}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
