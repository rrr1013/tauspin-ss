"""Does a solution-set likelihood add to the learned point-h readout?  Quick probe.

5-fold cross-fitted logistic regression on the development validation rows,
inputs = learner logit (+ optional physics log-LRs with a validity flag).  Both
arms use identical folds, so the AUC difference measures complementary
information, not readout capacity.  Weighted AUC with parent-overlap weights;
paired bootstrap over events of the out-of-fold scores.  This is a probe to
decide whether a proper train-split stacking is worth running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from s2_auc import TARGETS, wauc


def load(path, ids):
    d = np.load(path)
    pos = {int(v): i for i, v in enumerate(d["global_indices"])}
    rows = np.array([pos.get(int(v), -1) for v in ids])
    out = np.full(len(ids), np.nan)
    out[rows >= 0] = np.asarray(d["scores"], float)[rows[rows >= 0]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--learner", required=True)
    p.add_argument("--physics", nargs="+", required=True, help="name=path")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    t = np.load(TARGETS)
    v = t["h_valid"].astype(bool)
    ids, y, w, modes = t["global_indices"][v], t["labels"][v].astype(int), t["weights"][v].astype(float), t["modes"][v]
    s = np.clip(load(a.learner, ids), 1e-6, 1 - 1e-6)
    learner = np.log(s / (1 - s))
    phys = {}
    for spec in a.physics:
        name, path = spec.split("=", 1)
        phys[name] = load(path, ids)
    keep = np.isfinite(learner)
    for x in phys.values():
        keep &= ~np.isnan(x) | True
    # physics rows outside its cohort -> 0 with flag
    ids, y, w, modes, learner = ids[keep], y[keep], w[keep], modes[keep], learner[keep]
    cols = {"learner": learner}
    for name, x in phys.items():
        x = x[keep]
        cols[name] = np.where(np.isfinite(x), x, 0.0)
        cols[name + "_ok"] = np.isfinite(x).astype(float)
    rng = np.random.default_rng(20260923)
    fold = rng.integers(0, 5, len(ids))
    arms = {"learner_only": ["learner"],
            "physics_only": [c for c in cols if c != "learner"],
            "learner+physics": list(cols)}
    for name in phys:
        arms[f"learner+{name}"] = ["learner", name, name + "_ok"]
    oof = {}
    for arm, feats in arms.items():
        X = np.stack([cols[c] for c in feats], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        score = np.zeros(len(ids))
        for k in range(5):
            tr, te = fold != k, fold == k
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
            score[te] = clf.decision_function(X[te])
        oof[arm] = score
    out = {"events": int(len(ids)), "auc": {k: wauc(s_, y, w) for k, s_ in oof.items()}, "delta": {}}
    base = oof["learner_only"]
    for arm, sc in oof.items():
        if arm == "learner_only":
            continue
        d = []
        for _ in range(300):
            b = rng.integers(0, len(ids), len(ids))
            d.append(wauc(sc[b], y[b], w[b]) - wauc(base[b], y[b], w[b]))
        out["delta"][arm] = [out["auc"][arm] - out["auc"]["learner_only"], *np.quantile(d, [0.025, 0.975]).tolist()]
    pairs = {"pi x pi": (0, 0), "pi x rho": (0, 1), "rho x rho": (1, 1), "pi x 3pi": (0, 3), "rho x 3pi": (1, 3), "3pi x 3pi": (3, 3)}
    out["by_pair"] = {}
    for pn, (m0, m1) in pairs.items():
        sel = ((modes[:, 0] == m0) & (modes[:, 1] == m1)) | ((modes[:, 0] == m1) & (modes[:, 1] == m0))
        out["by_pair"][pn] = {k: wauc(sc[sel], y[sel], w[sel]) for k, sc in oof.items()}
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
