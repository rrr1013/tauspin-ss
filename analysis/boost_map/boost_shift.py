"""Stress scenario for m_H = 125 GeV: shift the tau-pair boost by a factor k
(k = 125.0/91.19 = 1.37 is the extreme case where the whole tau momentum
scales with the parent mass) by reweighting events in ln(beta*gamma_gm).
Within a beta*gamma slice the x mix, mode mix and everything else stay as in
this sample.  Weight = f(ln bg - ln k) / f(ln bg) from a 30-bin histogram,
applied on top of the overlap weights; events whose shifted source bin is
empty get weight 0 (reported as lost support).
"""
import json
import os

import numpy as np

from boost_map import wauc

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, 'scores.npz'))
y, w = S['y'], S['w']
lb = np.log(np.sqrt(S['bg'][:, 0] * S['bg'][:, 1]))
arms = ['exact', 'base', 'ipsv', 'idealip']
edges = np.quantile(lb, np.linspace(0, 1, 31))
f, _ = np.histogram(lb, edges, weights=w, density=True)
cent_idx = np.clip(np.digitize(lb, edges) - 1, 0, 29)
out = {}
rng = np.random.default_rng(3)
for k in (1.0, 1.15, 1.37):
    src = lb - np.log(k)
    si = np.digitize(src, edges) - 1
    inside = (si >= 0) & (si < 30)
    num = np.where(inside, f[np.clip(si, 0, 29)], 0.0)
    rw = w * num / f[cent_idx]
    ess = rw.sum() ** 2 / np.sum(rw ** 2)
    res = {'ess': float(ess), 'zero_weight_frac': float(1 - np.sum(w * inside) / np.sum(w)), 'unrepresented_target_frac': float(np.sum(w * (lb > edges[-1] - np.log(k))) / np.sum(w)),
           'median_bg_gm': float(np.exp(np.quantile(lb, 0.5))) if k == 1 else None,
           'auc': {a: wauc(S[f's_{a}'], y, rw) for a in arms}}
    B = {a: [] for a in arms}
    for _ in range(100):
        i = rng.integers(0, len(y), len(y))
        for a in arms:
            B[a].append(wauc(S[f's_{a}'][i], y[i], rw[i]))
    res['auc_se'] = {a: float(np.std(B[a])) for a in arms}
    out[str(k)] = res
base = out['1.0']['auc']
for k, r in out.items():
    print(k, 'ESS', round(r['ess']), 'unrepresented', round(r['unrepresented_target_frac'], 4),
          {a: f"{r['auc'][a]:.4f} ({r['auc'][a]-base[a]:+.4f})" for a in arms})
json.dump(out, open(os.path.join(HERE, 'results_shift.json'), 'w'), indent=1)
