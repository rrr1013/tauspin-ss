"""Old (CLEO 3pi) vs new (generator 3pi) teacher: fixed-readout AUCs and h quality.

Same validation cohort, parent-overlap weights and weighted AUC as
hz_beyond_ceiling/p8_auc_compare.py.  Deltas are new - old for the same arm and
seed, paired event bootstrap.  h quality is measured against the generator h
(the polarimeter of the sample) and the canonical CLEO h.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OLD_T = Path('/home/rbaba/reco-h-supervision-20260906-targets/validation.npz')
PAIRS = {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1),
         'pi x 3pi': (0, 3), 'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3)}
ARMS = ('base_s43', 'full22_s42', 'full22_s43', 'ens_full22', 'idealip22_s42')
CH, CZ = np.array([1, 1, -1.0]), np.array([0, 0, 1.0])


def wauc(score, y, w):
    order = np.argsort(score, kind='mergesort')
    s, y, w = score[order], y[order], w[order]
    wp, wn = np.where(y == 1, w, 0.0), np.where(y == 0, w, 0.0)
    _, first = np.unique(s, return_index=True)
    bounds = np.append(first, len(s))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def lr(h):
    p = h[:, 0] * h[:, 1]
    return np.log(1 + p @ CH) - np.log(1 + p @ CZ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', type=Path, required=True, help='hz_beyond_ceiling artifacts dir')
    ap.add_argument('--new', type=Path, required=True, help='gen3pi artifacts dir')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    t = np.load(OLD_T)
    valid = t['h_valid'].astype(bool)
    ids, y, w, modes = t['global_indices'][valid], t['labels'][valid].astype(int), \
        t['weights'][valid].astype(float), t['modes'][valid]
    h_cleo = t['h'][valid].astype(float)
    h_gen = np.load(a.new / 'targets' / 'validation.npz')['h'][valid].astype(float)
    sel = {k: ((modes[:, 0] == m0) & (modes[:, 1] == m1)) | ((modes[:, 0] == m1) & (modes[:, 1] == m0))
           for k, (m0, m1) in PAIRS.items()}
    sel['overall'] = np.ones(len(y), bool)

    scores = {'exact_h_LR:cleo': lr(h_cleo), 'exact_h_LR:gen': lr(h_gen)}
    for tag, root in (('old', a.old), ('new', a.new)):
        for arm in ARMS:
            f = root / 'p11_readout' / f'readout_scores_{arm}.npz'
            if f.exists():
                d = np.load(f)
                assert np.array_equal(d['global_indices'], ids), f
                scores[f'{tag}:{arm}'] = d['scores']
    out = {'events': int(len(y)), 'auc': {}, 'delta_new_minus_old': {}, 'h_quality': {}}
    for k, v in scores.items():
        out['auc'][k] = {p: wauc(v[m], y[m], w[m]) for p, m in sel.items()}
    rng = np.random.default_rng(20260924)
    pairs = [(f'old:{arm}', f'new:{arm}') for arm in ARMS if f'old:{arm}' in scores and f'new:{arm}' in scores]
    pairs.append(('exact_h_LR:cleo', 'exact_h_LR:gen'))
    boots = {n: {p: [] for p in sel} for _, n in pairs}
    idx_by = {p: np.flatnonzero(m) for p, m in sel.items()}
    for _ in range(a.boot):
        for p, idx in idx_by.items():
            b = rng.choice(idx, len(idx))
            for o, n in pairs:
                boots[n][p].append(wauc(scores[n][b], y[b], w[b]) - wauc(scores[o][b], y[b], w[b]))
    for o, n in pairs:
        out['delta_new_minus_old'][n.split(':', 1)[1] if n.startswith('new') else 'exact_h_LR'] = {
            p: {'delta': out['auc'][n][p] - out['auc'][o][p],
                'ci95': np.quantile(boots[n][p], [0.025, 0.975]).tolist()} for p in sel}

    for tag, root in (('old', a.old), ('new', a.new)):
        for arm in ('base_s43', 'full22_s42', 'idealip22_s42'):
            f = root / 'p11' / arm / 'validation_predictions.npz'
            if not f.exists():
                continue
            d = np.load(f)
            pos = {int(v): i for i, v in enumerate(d['global_indices'])}
            hp = d['h_pred'][[pos[int(v)] for v in ids]].reshape(len(ids), 2, 3).astype(float)
            q = {}
            for md, name in ((0, 'pi'), (1, 'rho'), (3, '3pi')):
                m = modes == md
                p, g, c = hp[m], h_gen[m], h_cleo[m]
                q[name] = {
                    'sides': int(m.sum()),
                    'corr_vs_gen_nrk': [float(np.corrcoef(p[:, i], g[:, i])[0, 1]) for i in range(3)],
                    'corr_vs_cleo_nrk': [float(np.corrcoef(p[:, i], c[:, i])[0, 1]) for i in range(3)],
                    'mse_vs_gen': float(np.mean((p - g) ** 2)), 'mse_vs_cleo': float(np.mean((p - c) ** 2)),
                    'mean_dot_vs_gen': float(np.mean(np.sum(p * g, -1))),
                    'mean_norm': float(np.mean(np.linalg.norm(p, axis=-1)))}
            out['h_quality'][f'{tag}:{arm}'] = q
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: round(v['overall'], 4) for k, v in out['auc'].items()}, indent=1))
    print(json.dumps({k: {p: round(x['delta'], 4) for p, x in v.items()} for k, v in out['delta_new_minus_old'].items()}, indent=1))


if __name__ == '__main__':
    main()
