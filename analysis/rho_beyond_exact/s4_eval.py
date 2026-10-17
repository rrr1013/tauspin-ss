"""H/Z AUC and rho-side quality of the denoised reference E[h | reco rho, truth nu] (s4), next to the formula and the learner.

Readouts as s2 (formula; fixed-recipe MLP trained on the TRAIN cross-fitted denoised h).  Per-side weighted correlation
with the exact h on the s3 rho sides (truth rho, reco rho).  Paired event bootstrap for the AUC deltas.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_readouts import GEN, formula, mlp_readout, pair_masks, wauc  # noqa: E402


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=400)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tt = np.load(GEN / 'targets' / 'train.npz')
    vt = np.load(GEN / 'targets' / 'validation.npz')
    tv, vv = tt['h_valid'].astype(bool), vt['h_valid'].astype(bool)
    y_tr, y_va = tt['labels'][tv].astype(int), vt['labels'][vv].astype(int)
    w_va, hx, modes = vt['weights'][vv].astype(float), vt['h'][vv].astype(float), vt['modes'][vv]
    D = np.load(a.run_dir / 'out' / 's4' / 'denoised.npz')
    assert np.array_equal(D['train_global_indices'], tt['global_indices'][tv])
    assert np.array_equal(D['validation_global_indices'], vt['global_indices'][vv])
    S2 = np.load(a.run_dir / 'out' / 's2_readouts.scores.npz')
    names = list(S2['names'])
    scores = {n: S2['values'][i] for i, n in enumerate(names) if n in (
        'reco+truth nu formula', 'reco+truth nu mlp', 'ladder pi0dir formula', 'ladder pi0 formula',
        'learner ens_full22 readout', 'exact formula')}
    scores['denoised formula'] = formula(D['h_validation'])
    scores['denoised mlp'] = mlp_readout(D['h_train'], y_tr, D['h_validation'], device)
    masks = pair_masks(modes)
    res = {'auc': {n: {k: wauc(s[m], y_va[m], w_va[m]) for k, m in masks.items()} for n, s in scores.items()}, 'delta': {}}
    comps = [('denoised mlp', 'learner ens_full22 readout'), ('denoised mlp', 'reco+truth nu mlp'),
             ('denoised formula', 'ladder pi0dir formula')]
    rng = np.random.default_rng(20261018)
    for key in ('rho x rho', 'pi x rho', 'rho x 3pi', 'overall'):
        m = np.flatnonzero(masks[key])
        bs = {c: [] for c in comps}
        for _ in range(a.boot):
            b = m[rng.integers(0, len(m), len(m))]
            for c in comps:
                bs[c].append(wauc(scores[c[0]][b], y_va[b], w_va[b]) - wauc(scores[c[1]][b], y_va[b], w_va[b]))
        res['delta'][key] = {f'{c[0]} - {c[1]}': [res['auc'][c[0]][key] - res['auc'][c[1]][key],
                                                  float(np.quantile(bs[c], .025)), float(np.quantile(bs[c], .975))] for c in comps}
    L = np.load(a.run_dir / 'out' / 's1' / 'ladder_validation.npz')
    q = L['rho_side'] & np.isfinite(hx).all(-1) & np.isfinite(D['h_validation']).all(-1) & np.isfinite(L['h_reco']).all(-1)
    W = np.broadcast_to(w_va[:, None], q.shape)[q]
    res['quality'] = {n: {c: wcorr(h[..., i][q], hx[..., i][q], W) for i, c in enumerate('nrk')}
                      for n, h in (('formula', L['h_reco']), ('denoised', D['h_validation']))}
    res['quality']['sides'] = int(q.sum())
    a.output.write_text(json.dumps(res, indent=1))
    keys = ('pi x pi', 'pi x rho', 'rho x rho', 'rho x 3pi', '3pi x 3pi', 'overall')
    print(f"{'':30s}" + ''.join(f'{k:>11s}' for k in keys))
    for n in scores:
        print(f'{n:30s}' + ''.join(f"{res['auc'][n][k]:11.4f}" for k in keys))
    for key, d in res['delta'].items():
        print(key)
        for c, (v, lo, hi) in d.items():
            print(f'   {c:55s} {v:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    print('quality', json.dumps(res['quality']))


if __name__ == '__main__':
    main()
