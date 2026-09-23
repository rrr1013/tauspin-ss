"""Null-space likelihood-ratio AUCs, CLEO vs generator 3pi current, same draws.

Both moment files come from p12 with identical arguments and seed, so the
hypotheses and their IP/SV/mass weights are identical event by event; only the
polarimeter evaluated on the 3pi sides differs.  AUCs are unweighted as the
p12/p6 headline numbers; deltas are generator - CLEO with a paired event
bootstrap.
"""
import argparse
import json
from pathlib import Path

import numpy as np

CH, CZ = np.diag([1.0, 1.0, -1.0]), np.diag([0.0, 0.0, 1.0])
PAIRS = {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1),
         'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}


def auc(s, y):
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = (y == 1).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def lr(second):
    return (np.log(np.maximum(1 + np.einsum('ij,nij->n', CH, second), 1e-9))
            - np.log(np.maximum(1 + np.einsum('ij,nij->n', CZ, second), 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cleo', type=Path, required=True)
    ap.add_argument('--gen', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=500)
    a = ap.parse_args()
    C, G = np.load(a.cleo), np.load(a.gen)
    for k in ('labels', 'modes', 'global_indices', 'count', 'retained'):
        if not np.array_equal(C[k], G[k]):
            raise RuntimeError(f'{k} differs: the two runs did not use the same draws')
    for k in [f for f in C.files if f.startswith('ess_') or f.startswith('sum_w_')]:
        if not np.allclose(C[k], G[k], equal_nan=True):
            raise RuntimeError(f'{k} differs: the weights are not identical')
    y, modes = C['labels'].astype(int), C['modes']
    valid = C['count'] > 0
    measures = [k[len('second_'):] for k in C.files if k.startswith('second_')]
    scores = {}
    for tag, d in (('cleo', C), ('gen', G)):
        th = d['truth_h_reference']
        scores[f'{tag}:truth_h'] = lr(np.einsum('ni,nj->nij', th[:, 0], th[:, 1]))
        for m in measures:
            scores[f'{tag}:{m}'] = lr(d[f'second_{m}'])
    sel = {'overall': valid}
    for p, (m0, m1) in PAIRS.items():
        sel[p] = valid & (((modes[:, 0] == m0) & (modes[:, 1] == m1)) | ((modes[:, 0] == m1) & (modes[:, 1] == m0)))
    out = {'events': int(valid.sum()), 'auc': {}, 'delta_gen_minus_cleo': {}}
    for k, v in scores.items():
        out['auc'][k] = {p: auc(v[m], y[m]) for p, m in sel.items()}
    rng = np.random.default_rng(20260924)
    names = ['truth_h'] + measures
    boots = {n: {p: [] for p in sel} for n in names}
    for _ in range(a.boot):
        for p, m in sel.items():
            idx = np.flatnonzero(m)
            b = rng.choice(idx, len(idx))
            for n in names:
                boots[n][p].append(auc(scores[f'gen:{n}'][b], y[b]) - auc(scores[f'cleo:{n}'][b], y[b]))
    for n in names:
        out['delta_gen_minus_cleo'][n] = {
            p: {'delta': out['auc'][f'gen:{n}'][p] - out['auc'][f'cleo:{n}'][p],
                'ci95': np.quantile(boots[n][p], [0.025, 0.975]).tolist()} for p in sel}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: round(v['overall'], 4) for k, v in out['auc'].items()}, indent=1))


if __name__ == '__main__':
    main()
