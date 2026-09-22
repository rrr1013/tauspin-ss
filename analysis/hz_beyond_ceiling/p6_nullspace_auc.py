"""H/Z AUCs from the IP-weighted null-space moments (p5), with paired bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CH = np.diag([1.0, 1.0, -1.0])
CZ = np.diag([0.0, 0.0, 1.0])
MEASURES = ('flat', 'ip', 'ip_shuffle', 'ip_legacy', 'mass', 'ip_mass')
PAIRS = {'1p0n x 1p0n': (0, 0), 'rho x rho': (1, 1), '3p0n x 3p0n': (3, 3)}


def auc(score, label, weight=None):
    w = np.ones_like(score) if weight is None else weight
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], label[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    # tie-aware: group equal scores
    _, start = np.unique(s, return_index=True)
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    total = 0.0
    bounds = np.append(start, len(s))
    for a, b in zip(bounds[:-1], bounds[1:]):
        total += wp[a:b].sum() * (cum_neg[a] + 0.5 * wn[a:b].sum())
    return total / (wp.sum() * wn.sum())


def fast_auc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = (label == 1).sum()
    n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def lr(second):
    return np.log(np.maximum(1 + np.einsum('ij,nij->n', CH, second), 1e-9)) - \
        np.log(np.maximum(1 + np.einsum('ij,nij->n', CZ, second), 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--moments', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--boot', type=int, default=1000)
    a = p.parse_args()
    d = np.load(a.moments)
    y = d['labels'].astype(int)
    modes = d['modes']
    valid = d['count'] > 0
    th = d['truth_h_reference']
    truth_second = np.einsum('ni,nj->nij', th[:, 0], th[:, 1])
    scores = {'truth_h_LR': lr(truth_second)}
    for m in MEASURES:
        scores[f'{m}_LR'] = lr(d[f'second_{m}'])
        M = d[f'second_{m}']
        scores[f'{m}_ET'] = M[:, 0, 0] + M[:, 1, 1] - M[:, 2, 2]
    out = {'events': int(valid.sum()), 'unweighted': {}, 'weighted': {}, 'by_pair': {}, 'ess': {}}
    for k, v in scores.items():
        out['unweighted'][k] = float(fast_auc(v[valid], y[valid]))
        out['weighted'][k] = float(auc(v[valid], y[valid], d['weights'][valid]))
    for name, (m0, m1) in PAIRS.items():
        sel = valid & (modes[:, 0] == m0) & (modes[:, 1] == m1)
        out['by_pair'][name] = {'events': int(sel.sum()), **{
            k: float(fast_auc(scores[k][sel], y[sel])) for k in
            ('truth_h_LR', 'flat_LR', 'ip_LR', 'ip_shuffle_LR', 'ip_legacy_LR', 'mass_LR', 'ip_mass_LR')}}
    for m in MEASURES:
        out['ess'][m] = np.quantile(d[f'ess_{m}'][valid], [0.05, 0.5, 0.95]).tolist()
    rng = np.random.default_rng(20260923)
    idx = np.flatnonzero(valid)
    diffs = {k: [] for k in ('ip_LR', 'ip_shuffle_LR', 'ip_legacy_LR', 'mass_LR', 'ip_mass_LR')}
    for _ in range(a.boot):
        b = rng.choice(idx, len(idx))
        base = fast_auc(scores['flat_LR'][b], y[b])
        for k in diffs:
            diffs[k].append(fast_auc(scores[k][b], y[b]) - base)
    out['paired_delta_vs_flat_LR'] = {
        k: {'delta': out['unweighted'][k] - out['unweighted']['flat_LR'],
            'ci95': np.quantile(v, [0.025, 0.975]).tolist()} for k, v in diffs.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
