"""tau_lep tau_had HH significance with and without the lepton impact parameter (the
tauspin-style lifetime information) and the hadronic-side tauspin inputs.

Cross-fitted gradient-boosted classifiers (three folds by event id; every event is scored
by a model that did not see it), HH signal against the weighted background mixture.
Feature sets:
  K       ATLAS lep-had BDT inputs (arXiv:2209.10910 Table 3) and the remaining event
          kinematics (generous to the baseline);
  K+d0    K plus |d0|/sigma only (a plain lifetime cut variable);
  K+IP    K plus the lepton impact vector in the local basis, its lifetime sign along the
          MMC tau direction, z0 significance, flavour, resolution;
  K+IP+H  K+IP plus the hadronic-side tauspin inputs (Upsilon, x, mode, IP/SV in the local
          basis).
Selections: m_bb < 150 GeV, m_tautau(MMC) > 60 GeV; scenario `ttva` additionally applies the
standard track-to-vertex association (|d0/sigma| < 3 mu / 5 e) to the lepton.
Significance per trigger category (SLT, LTT) from bins in the classifier score, each bin
merged from the signal-like end until its background has at least `min_ess` effective MC
events; Asimov discovery q0 with the background normalisations of five groups profiled
(top 10 %, single top 15 %, Z+HF 10 %, single Higgs 15 %, fakes 20 %), statistically only
as a variant.  Categories combined in quadrature.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

GROUPS = {"top": ("ttll", "ttlt"), "stop": ("tWll",), "zhf": ("zbb",), "sh": ("zh", "tth"),
          "fake": ("ttlj", "ttljp", "tWlj")}
PRIOR = {"top": 0.10, "stop": 0.15, "zhf": 0.10, "sh": 0.15, "fake": 0.20}
SIGNAL = "hhC"


def feature_sets(d, lip="lip_"):
    kin = sorted(k for k in d.files if k.startswith("kin_") and k not in ("kin_mmc_valid",))
    ipk = [lip + s for s in ("d0sig", "absd0", "z0sig", "n", "r", "k", "life", "sd0", "is_e", "x")]
    had = sorted(k for k in d.files if k.startswith("had_"))
    hspin = [k for k in had if not (k.startswith("had_tip_") or k.startswith("had_tsv_"))]
    return {"K": kin, "K+d0": kin + [lip + "d0sig", lip + "sd0", lip + "is_e"], "K+IP": kin + ipk, "K+IP+H": kin + ipk + had,
            "K+IP+Hspin": kin + ipk + hspin, "K+Hspin": kin + hspin}


def crossfit(X, y, w, fold, seed=0):
    s = np.zeros(len(y))
    for f in range(3):
        tr = fold != f
        wt = w[tr].copy()
        wt[y[tr] == 1] *= w[tr][y[tr] == 0].sum() / w[tr][y[tr] == 1].sum()
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=100, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.15, random_state=seed)
        clf.fit(X[tr], y[tr], sample_weight=wt)
        s[fold == f] = clf.predict_proba(X[fold == f])[:, 1]
    return s


def binning(score, y, w, min_ess, nmax=25):
    """Edges from the signal-like end: equal signal fractions, merged until bkg ESS >= min_ess."""
    sig = y == 1
    qs = np.quantile(score[sig], np.linspace(0, 1, nmax + 1))
    edges = [np.inf]
    lo_idx = nmax
    while lo_idx > 0:
        for j in range(lo_idx - 1, -1, -1):
            m = (score >= qs[j]) & (score < edges[-1]) & ~sig
            ess = w[m].sum() ** 2 / max((w[m] ** 2).sum(), 1e-30)
            if ess >= min_ess or j == 0:
                edges.append(qs[j] if j > 0 else -np.inf)
                lo_idx = j
                break
    return np.array(edges[::-1])


def q0(s, B, prior, stat_only=False):
    """Asimov discovery q0; s (nbins,), B (ngroups, nbins), priors per group."""
    n = s + B.sum(0)
    if stat_only:
        b = B.sum(0)
        return float(2 * (n * np.log(n / b) - s).sum())
    def nll(th):
        b = (B * (1 + prior[:, None] * th[:, None])).sum(0)
        b = np.maximum(b, 1e-9)
        return float((b - n * np.log(b)).sum() + 0.5 * (th ** 2).sum())
    r = minimize(nll, np.zeros(len(prior)), method="L-BFGS-B")
    l0 = r.fun
    l1 = float((n - n * np.log(n)).sum())
    return max(2 * (l0 - l1), 0.0)


def evaluate(score, y, w, proc, trig, min_ess):
    out = {}
    for tname, tv in (("SLT", 0), ("LTT", 1)):
        m = trig == tv
        e = binning(score[m], y[m], w[m], min_ess)
        b = np.clip(np.digitize(score[m], e) - 1, 0, len(e) - 2)
        nb = len(e) - 1
        s = np.bincount(b[y[m] == 1], weights=w[m][y[m] == 1], minlength=nb)
        B = np.stack([np.bincount(b[np.isin(proc[m], g)], weights=w[m][np.isin(proc[m], g)], minlength=nb)
                      for g in GROUPS.values()])
        B = np.maximum(B, 0) + 1e-6
        prior = np.array([PRIOR[g] for g in GROUPS])
        z = np.sqrt(q0(s, B, prior))
        zs = np.sqrt(q0(s, B, prior, stat_only=True))
        top = nb - 1
        comp = {g: float(B[i, top] / B[:, top].sum()) for i, g in enumerate(GROUPS)}
        out[tname] = {"Z": z, "Z_stat": zs, "nbins": int(nb), "S": float(s.sum()), "B": float(B.sum()),
                      "top_bin": {"S": float(s[top]), "B": float(B[:, top].sum()), "composition": comp}}
    out["combined"] = float(np.hypot(out["SLT"]["Z"], out["LTT"]["Z"]))
    out["combined_stat"] = float(np.hypot(out["SLT"]["Z_stat"], out["LTT"]["Z_stat"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ess", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="nominal,ip_res_x1.5,ip_res_x0.8")
    ap.add_argument("--sets", default="K,K+d0,K+IP,K+IP+H")
    ap.add_argument("--scen", default="no_ttva,ttva")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    proc, w, uid = d["proc"], d["w"], d["uid"]
    base = (d["kin_m_bb"] < 150) & (d["kin_m_tautau"] > 60)
    y = (proc == SIGNAL).astype(int)
    pidx = np.searchsorted(np.unique(proc), proc)
    res = {"config": vars(args), "yields": {}, "scenarios": {}}
    for p in np.unique(proc):
        m = base & (proc == p)
        res["yields"][str(p)] = {"events": int(m.sum()), "yield": float(w[m].sum()),
                                 "prompt_lepton_frac": float(np.average(~d["lep_from_tau"][m], weights=np.abs(w[m]) + 1e-12)) if m.any() else None}
    variants = [v for v in (("nominal", "lip_"), ("ip_res_x1.5", "lip1.5_"), ("ip_res_x0.8", "lip0.8_"))
                if v[0] in args.variants.split(",")]
    for scen in args.scen.split(","):
        for vname, lip in variants:
            if lip + "d0sig" not in d.files:
                continue
            sel = base & (d[lip + "ttva"] > 0 if scen == "ttva" else True)
            fs = {k: v for k, v in feature_sets(d, lip).items() if k in args.sets.split(",")}
            if vname != "nominal":
                fs = {k: v for k, v in fs.items() if k in ("K", "K+IP", "K+IP+H", "K+IP+Hspin")}
            key = f"{scen}/{vname}"
            res["scenarios"][key] = {}
            for name, cols in fs.items():
                X = np.stack([d[c] for c in cols], 1)[sel]
                zs = []
                for seed in range(args.seeds):
                    fold = ((uid // 3 + seed) * 7 + pidx * 13 + uid) % 3      # fold assignment varies with the seed
                    sc = crossfit(X, y[sel], w[sel], fold[sel], seed)
                    zs.append(evaluate(sc, y[sel], w[sel], proc[sel], d["trigger"][sel], args.min_ess))
                comb = np.array([z["combined"] for z in zs])
                res["scenarios"][key][name] = {"mean": float(comb.mean()), "sd": float(comb.std(ddof=1)) if len(comb) > 1 else 0.0,
                                               "mean_stat": float(np.mean([z["combined_stat"] for z in zs])),
                                               "SLT": float(np.mean([z["SLT"]["Z"] for z in zs])),
                                               "LTT": float(np.mean([z["LTT"]["Z"] for z in zs])), "runs": zs}
                r = res["scenarios"][key][name]
                print(key, name, round(r["mean"], 3), "+-", round(r["sd"], 3), "stat", round(r["mean_stat"], 3),
                      "SLT", round(r["SLT"], 3), "LTT", round(r["LTT"], 3), flush=True)
                json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
