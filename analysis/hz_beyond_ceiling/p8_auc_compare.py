"""Weighted H/Z AUCs for the fixed h readout (p4 arms) and direct classifiers (p7).

Weights are the parent-overlap weights of the canonical targets; the validation
cohort is the h-valid validation split (test never loaded).  Differences to the
baseline point-h readout use a paired event bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGETS = Path("/home/rbaba/reco-h-supervision-20260906-targets/validation.npz")
PAIRS = {"1p0n x 1p0n": (0, 0), "1p0n x rho": (0, 1), "rho x rho": (1, 1),
         "1p0n x 3p0n": (0, 3), "rho x 3p0n": (1, 3), "3p0n x 3p0n": (3, 3)}


def wauc(score, y, w):
    order = np.argsort(score, kind="mergesort")
    s, y, w = score[order], y[order], w[order]
    wp = np.where(y == 1, w, 0.0)
    wn = np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp = np.add.reduceat(wp, first)
    gn = np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--readout-dir", type=Path, required=True)
    p.add_argument("--direct-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", default="readout:baseline")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    t = np.load(TARGETS)
    valid = t["h_valid"].astype(bool)
    ids, y, w, modes = (t["global_indices"][valid], t["labels"][valid].astype(int),
                        t["weights"][valid].astype(float), t["modes"][valid])
    scores = {}
    for f in sorted(a.readout_dir.glob("readout_scores_*.npz")):
        d = np.load(f)
        assert np.array_equal(d["global_indices"], ids), f
        scores["readout:" + f.stem.replace("readout_scores_", "")] = d["scores"]
    if a.direct_dir:
        for f in sorted(a.direct_dir.glob("*/validation_predictions.npz")):
            d = np.load(f)
            pos = {int(v): i for i, v in enumerate(d["global_indices"])}
            rows = np.asarray([pos[int(v)] for v in ids])
            scores["direct:" + f.parent.name] = d["scores"][rows]
    # exact truth-h spin likelihood ratio as ceiling reference
    h = t["h"][valid]
    prod = h[:, 0] * h[:, 1]
    scores["ceiling:exact_h_LR"] = np.log(1 + prod @ np.array([1, 1, -1.0])) - np.log(1 + prod[:, 2])
    out = {"events": int(len(ids)), "weighted_auc": {}, "by_pair": {}, "paired_delta": {}}
    for k, v in scores.items():
        out["weighted_auc"][k] = wauc(v, y, w)
    for name, (m0, m1) in PAIRS.items():
        sel = ((modes[:, 0] == m0) & (modes[:, 1] == m1)) | ((modes[:, 0] == m1) & (modes[:, 1] == m0))
        out["by_pair"][name] = {"events": int(sel.sum()),
                                **{k: wauc(v[sel], y[sel], w[sel]) for k, v in scores.items()}}
    rng = np.random.default_rng(20260923)
    base = scores[a.baseline]
    boots = {k: [] for k in scores if k != a.baseline}
    for _ in range(a.boot):
        b = rng.integers(0, len(ids), len(ids))
        ab = wauc(base[b], y[b], w[b])
        for k in boots:
            boots[k].append(wauc(scores[k][b], y[b], w[b]) - ab)
    for k, v in boots.items():
        out["paired_delta"][k] = {"delta": out["weighted_auc"][k] - out["weighted_auc"][a.baseline],
                                  "ci95": np.quantile(v, [0.025, 0.975]).tolist()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["weighted_auc"], indent=1))
    print(json.dumps(out["paired_delta"], indent=1))


if __name__ == "__main__":
    main()
