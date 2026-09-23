"""Weighted H/Z AUCs of arbitrary per-event scores on a common validation cohort.

Every input is an npz with ``global_indices`` and ``scores``.  The cohort is the
h-valid validation split intersected with the rows of every score file (so a
score defined on a subset, e.g. the strict reco-polarimeter cohort, restricts
all arms to the same rows).  Weights are the parent-overlap weights, pairs use
the truth decay modes.  Paired event bootstrap against ``--baseline``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGETS = Path("/home/rbaba/reco-h-supervision-20260906-targets/validation.npz")
PAIRS = {"pi x pi": (0, 0), "pi x rho": (0, 1), "rho x rho": (1, 1),
         "pi x 3pi": (0, 3), "rho x 3pi": (1, 3), "3pi x 3pi": (3, 3)}


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
    p.add_argument("--scores", nargs="+", required=True, help="name=path.npz")
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--unweighted", action="store_true")
    p.add_argument("--fill-nan", type=float, default=None,
                   help="replace non-finite scores by this value instead of dropping the row")
    a = p.parse_args()
    t = np.load(TARGETS)
    valid = t["h_valid"].astype(bool)
    ids, y, w, modes = (t["global_indices"][valid], t["labels"][valid].astype(int),
                        t["weights"][valid].astype(float), t["modes"][valid])
    h = t["h"][valid].astype(float)
    raw = {}
    keep = np.ones(len(ids), bool)
    for spec in a.scores:
        name, path = spec.split("=", 1)
        d = np.load(path)
        pos = {int(v): i for i, v in enumerate(d["global_indices"])}
        rows = np.array([pos.get(int(v), -1) for v in ids])
        sc = np.full(len(ids), np.nan)
        sc[rows >= 0] = np.asarray(d["scores"], float)[rows[rows >= 0]]
        keep &= rows >= 0
        if a.fill_nan is None:
            keep &= np.isfinite(sc)
        else:
            sc[(rows >= 0) & ~np.isfinite(sc)] = a.fill_nan
        raw[name] = sc
    prod = h[:, 0] * h[:, 1]
    raw["exact_h_LR"] = np.log(1 + prod @ np.array([1, 1, -1.0])) - np.log(1 + prod[:, 2])
    ids, y, w, modes = ids[keep], y[keep], w[keep], modes[keep]
    if a.unweighted:
        w = np.ones_like(w)
    scores = {k: v[keep] for k, v in raw.items()}
    out = {"events": int(len(ids)), "weighted": not a.unweighted, "auc": {}, "by_pair": {},
           "paired_delta": {}}
    for k, v in scores.items():
        out["auc"][k] = wauc(v, y, w)
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
        out["paired_delta"][k] = {"delta": out["auc"][k] - out["auc"][a.baseline],
                                  "ci95": np.quantile(v, [0.025, 0.975]).tolist()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"events": out["events"], "auc": out["auc"]}, indent=1))
    print(json.dumps(out["paired_delta"], indent=1))


if __name__ == "__main__":
    main()
