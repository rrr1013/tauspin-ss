"""Spin-hypothesis classifiers on the spin-flat HH cohort (out-of-fold probabilities).

The cohort (analysis/dihiggs_spin/outputs/v2, 58,805 events, 14 TeV MG5+Pythia8, tau
decays unpolarised and uncorrelated) carries the exact polarimeter h of each tau and
the TauSpin point regression h_pred whose detector response was calibrated on the ATLAS
full-simulation H/Z(125) cohort (fixed-readout H/Z AUC 0.627 vs 0.626 in ATLAS sim).
Reweighting every event by the physical spin density rho_X(h_exact) of hypothesis X
gives the same events under any spin state at fixed kinematics.

Conventions follow dihiggs_spin/latest_spin.py (canonical h = minus physical h on both
sides; a side without an implemented polarimeter contributes its marginal):
  ditau     H  : 1 + nn + rr - kk                          (signal, single H)
            Z  : 1 - 0.147 (k0 + k1) + kk + cT (nn - rr)   (Z/gamma* -> tautau + HF)
            W  : (1 - k0)(1 - k1)                          (true-tau top: W -> tau nu)
            WU : [(1 - k0) + (1 - k1)] / 2                 (ttbar with one fake tau)
            U  : 1                                         (fakes, other)
  single    H : 1, Z : 1 - 0.147 k, W : 1 - k, U : 1        (lep-had: hadronic side only)

Arms (feature sets seen by the classifier):
  tauspin  : h_pred (6) + regressed joint moments hh_pred (9) + decay-mode flags
  textbook : per-side Upsilon, x, visible mass and hybrid h (n, r, k) + mode flags
             (the observables used in LHC tau-polarisation/CP analyses, no IP/SV)
  exact    : analytic likelihood rho_X(h_exact) (perfect polarimetry, upper bound)

Classifiers (XGBoost, settings of latest_spin.py) are cross-fitted: fold f = uid % 3 is predicted by a model trained on the
other two folds (one of them used for early stopping), so every event carries an
out-of-fold probability.  Nothing here uses ATLAS yields.
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

V2 = Path(os.environ.get("HH_SPINFLAT", Path.home() / "dihiggs-spin-20261006/runs/v2"))
OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs")) / "spin"

DI_HYPS = ("H", "Z", "W", "WU", "U")
SI_HYPS = ("H", "Z", "W")          # single-side U equals H (both flat)
P_Z = -0.147


def ditau_density(h, c_t=0.5):
    valid = np.isfinite(h).all(-1)
    a = np.where(valid[..., None], h, 0.0)
    m, p = a[:, 0], a[:, 1]
    nn, rr, kk = m[:, 0] * p[:, 0], m[:, 1] * p[:, 1], m[:, 2] * p[:, 2]
    rho = np.stack([
        1 + nn + rr - kk,
        1 + P_Z * (m[:, 2] + p[:, 2]) + kk + c_t * (nn - rr),
        (1 - m[:, 2]) * (1 - p[:, 2]),
        0.5 * ((1 - m[:, 2]) + (1 - p[:, 2])),
        np.ones(len(h)),
    ], 1)
    if rho.min() < -1e-9:
        raise ValueError(f"negative spin density {rho.min()}")
    return np.maximum(rho, 0.0), valid


def single_density(h1):
    valid = np.isfinite(h1).all(-1)
    k = np.where(valid, h1[:, 2], 0.0)
    return np.stack([np.ones(len(h1)), 1 + P_Z * k, 1 - k], 1), valid


def load():
    d = np.load(V2 / "dataset_hhU.npz", allow_pickle=True)
    p = np.load(V2 / "hpred_hhU.npz", allow_pickle=True)
    k = np.load(V2 / "dataset_hhU_K.npz", allow_pickle=True)
    assert np.array_equal(d["uid"], p["uid"]) and np.array_equal(d["uid"], k["uid"])
    assert len(np.unique(d["uid"])) == len(d["uid"])
    return d, p, k["K"]


def modes(d, s):
    return np.stack([d[f"spin_rmode{j}_{s}"] for j in range(5)], 1)


def features(d, p):
    m = np.c_[modes(d, 0), modes(d, 1)]
    hp = p["h_pred"].reshape(-1, 6)
    tb = [np.stack([d[f"spin_{v}_{s}"] for v in ("upsilon", "x", "mvis", "hhyb_n", "hhyb_r", "hhyb_k")], 1)
          for s in (0, 1)]
    di = {"tauspin": np.c_[hp, p["hh_pred"].reshape(-1, 9), m],
          "textbook": np.c_[tb[0], tb[1], m]}
    si = {"tauspin": [np.c_[p["h_pred"][:, s], modes(d, s)] for s in (0, 1)],
          "textbook": [np.c_[tb[s], modes(d, s)] for s in (0, 1)]}
    return di, si


def crossfit(X, W, fold, seed=0):
    """W: (N, C) per-hypothesis event weights.  Returns out-of-fold class posteriors
    (equal class priors) and per-fold iteration counts."""
    n, c = W.shape
    out = np.zeros((n, c))
    iters = []
    for f in range(3):
        tr = fold != f
        ww = W[tr] / W[tr].sum(0, keepdims=True) * tr.sum()
        Xt = np.tile(X[tr], (c, 1))
        yt = np.repeat(np.arange(c), tr.sum())
        wt = ww.T.reshape(-1)
        es_fold = (f + 1) % 3
        val = np.tile(fold[tr] == es_fold, c)
        # Early stopping on a fixed held-out training fold (event-level split before
        # duplication into hypotheses, so no event is in both); same booster settings
        # as dihiggs_spin/latest_spin.py.
        dtr = xgb.DMatrix(Xt[~val], label=yt[~val], weight=wt[~val])
        dva = xgb.DMatrix(Xt[val], label=yt[val], weight=wt[val])
        model = xgb.train(dict(objective="multi:softprob", num_class=c, eval_metric="mlogloss",
                               tree_method="hist", max_depth=4, eta=0.04, min_child_weight=100,
                               reg_lambda=2, nthread=8, subsample=0.8, colsample_bytree=0.9,
                               seed=seed),
                          dtr, num_boost_round=3000, evals=[(dva, "validation")],
                          early_stopping_rounds=50, verbose_eval=False)
        if model.best_iteration >= 2949:
            raise RuntimeError("training ceiling reached")
        out[fold == f] = model.predict(xgb.DMatrix(X[fold == f]),
                                       iteration_range=(0, model.best_iteration + 1))
        iters.append(int(model.best_iteration))
    return out, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c_t", type=float, default=0.5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, p, K = load()
    fold = d["uid"] % 3
    w0 = d["w"].astype(float)
    he = d["h_exact"]
    rho, valid = ditau_density(he, args.c_t)
    W = rho * w0[:, None]
    di, si = features(d, p)
    res = {"N": int(len(w0)), "valid_side_fraction": valid.mean(0).tolist(), "c_t": args.c_t,
           "hyps_ditau": DI_HYPS, "hyps_single": SI_HYPS, "iters": {}}
    save = {"uid": d["uid"], "fold": fold, "w": w0, "rho": rho, "K": K,
            "m_hh": d["kin_m_hh"], "pt_tau": np.c_[d["kin_pt_tau1"], d["kin_pt_tau2"]]}
    # exact: analytic posteriors with equal priors
    norm = W.sum(0) / w0.sum()
    save["di_exact"] = (rho / norm) / (rho / norm).sum(1, keepdims=True)
    for arm, X in di.items():
        assert np.isfinite(X).all(), arm
        pr, it = crossfit(X.astype(np.float32), W, fold, args.seed)
        save[f"di_{arm}"] = pr
        res["iters"][f"di_{arm}"] = it
        print(arm, "ditau iters", it, flush=True)
    # single-tau (lep-had hadronic side): stack both sides as independent rows
    rs, vs = zip(*(single_density(he[:, s]) for s in (0, 1)))
    rs = np.concatenate(rs)
    Ws = rs * np.r_[w0, w0][:, None]
    fold_s = np.r_[fold, fold]
    save["si_rho"] = rs
    save["si_w"] = np.r_[w0, w0]
    save["si_fold"] = fold_s
    save["si_pt"] = np.r_[d["kin_pt_tau1"], d["kin_pt_tau2"]]
    norm_s = Ws.sum(0) / save["si_w"].sum()
    save["si_exact"] = (rs / norm_s) / (rs / norm_s).sum(1, keepdims=True)
    for arm, Xs in si.items():
        X = np.concatenate(Xs).astype(np.float32)
        assert np.isfinite(X).all(), arm
        pr, it = crossfit(X, Ws, fold_s, args.seed)
        save[f"si_{arm}"] = pr
        res["iters"][f"si_{arm}"] = it
        print(arm, "single iters", it, flush=True)
    tag = f"s{args.seed}_ct{args.c_t:g}"
    np.savez_compressed(OUT / f"spin_probs_{tag}.npz", **save)
    json.dump(res, open(OUT / f"spin_probs_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
