"""Classical-vs-classical check on the TRAIN holdout (event_number % 5 == 0, excluded from every response table).

The learner was trained on these events, so it is not compared here.  Output: AUC by truth pair, paired bootstrap of
the requested arm differences, and the per-side correlation with the exact h on truth pi & reco pi / rho / 3pi sides.
usage: eval_holdout.py --arm NAME=DIR ... --delta 'A - B' ... --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN, pair_masks, wauc  # noqa: E402
from hquality import wcorr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', nargs='+', required=True)
    ap.add_argument('--delta', nargs='*', default=[])
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    P = {k: np.load(Path(v) / 'posterior.npz') for k, v in (s.split('=', 1) for s in a.arm)}
    ref = P[next(iter(P))]
    ids, y, w = ref['global_indices'], ref['labels'].astype(int), ref['weights'].astype(float)
    assert np.all(ref['event_numbers'] % 5 == 0)
    for d in P.values():
        assert np.array_equal(d['global_indices'], ids)
    t = np.load(GEN / 'targets' / 'train.npz')
    v = t['h_valid'].astype(bool)
    look = {int(g): i for i, g in enumerate(t['global_indices'][v])}
    hx = t['h'][v][np.array([look[int(g)] for g in ids])].astype(float)
    modes, rm, av = ref['modes'], ref['reco_h_mode'].astype(int), ref['reco_h_available_side']
    sel = pair_masks(modes)
    scores = {f'{k}:ip3_sv_mass': d['ip3_sv_mass_lr'] for k, d in P.items()}
    res = {'events': int(len(ids)), 'events_truth_pairs': {p: int(m.sum()) for p, m in sel.items()},
           'auc': {k: {p: wauc(s[m], y[m], w[m]) for p, m in sel.items()} for k, s in scores.items()}}
    W = np.broadcast_to(w[:, None], av.shape)
    res['side_quality'] = {}
    for k, d in P.items():
        h = d['ip3_sv_mass_Eh']
        o = {}
        for g, (tm, r) in {'pi': (0, 0), 'rho': (1, 1), '3pi': (3, 3)}.items():
            m = av & (rm == r) & (modes == tm) & np.isfinite(h).all(-1)
            o[g] = {c: round(wcorr(h[..., i][m], hx[..., i][m], W[m]), 4) for i, c in enumerate('nrk')}
            o[g]['sides'] = int(m.sum())
        res['side_quality'][k] = o
    rng = np.random.default_rng(2)
    res['paired_delta'] = {}
    for spec in a.delta:
        p, q = (x.strip() + ':ip3_sv_mass' for x in spec.split(' - '))
        out = {}
        for pair in ('overall', 'pi x pi', 'pi x rho', 'rho x rho'):
            idx = np.flatnonzero(sel[pair])
            d0 = wauc(scores[p][idx], y[idx], w[idx]) - wauc(scores[q][idx], y[idx], w[idx])
            bs = []
            for _ in range(a.boot if pair == 'overall' else a.boot // 2):
                b = idx[rng.integers(0, len(idx), len(idx))]
                bs.append(wauc(scores[p][b], y[b], w[b]) - wauc(scores[q][b], y[b], w[b]))
            out[pair] = {'delta': round(d0, 5), 'ci95': np.round(np.quantile(bs, [0.025, 0.975]), 5).tolist()}
        res['paired_delta'][spec] = out
    a.output.write_text(json.dumps(res, indent=1))
    for k in scores:
        print(k, {p: round(res['auc'][k][p], 4) for p in ('overall', 'pi x pi', 'pi x rho', 'rho x rho')})
    for k, o in res['side_quality'].items():
        print(k, {g: (o[g]['n'], o[g]['r'], o[g]['k']) for g in o})
    for k, o in res['paired_delta'].items():
        print(k, {p: (x['delta'], x['ci95']) for p, x in o.items()})


if __name__ == '__main__':
    main()
