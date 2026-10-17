"""Classical posterior (10/17) with the reco pi0 vs the truth pi0 / truth pi0 direction on truth-rho reco-rho sides.

Baseline: 10/17 'unpol' run (iso MET, |M|^2_unpol, rho shell cap).  Oracles: this run's out/cp_{truth,truthdir}, same
flags and seed.  Weighted H/Z AUC by truth pair of the classical LR (ip3_sv, ip3_sv_mass) and the rho-side weighted
correlation of the posterior mean E[h] with the exact h.  Test never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_readouts import GEN, pair_masks, wauc  # noqa: E402

BASE = Path('/home/rbaba/atlas-classical-posterior-20261017/unpol/posterior.npz')


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    runs = {'reco pi0 (10/17)': BASE, 'truth pi0': a.run_dir / 'out' / 'cp_truth' / 'posterior.npz',
            'truth pi0 direction': a.run_dir / 'out' / 'cp_truthdir' / 'posterior.npz'}
    vt = np.load(GEN / 'targets' / 'validation.npz')
    vv = vt['h_valid'].astype(bool)
    hx = vt['h'][vv].astype(float)
    res = {'auc': {}, 'quality': {}}
    for name, path in runs.items():
        P = np.load(path)
        assert np.array_equal(P['global_indices'], vt['global_indices'][vv])
        y, w, modes = P['labels'].astype(int), P['weights'].astype(float), P['modes']
        masks = pair_masks(modes)
        for m in ('ip3_sv', 'ip3_sv_mass'):
            res['auc'][f'{name} {m}'] = {k: wauc(P[f'{m}_lr'][s], y[s], w[s]) for k, s in masks.items()}
        q = P['reco_h_available_side'] & (modes == 1) & (P['reco_h_mode'] == 1) & np.isfinite(P['ip3_sv_Eh']).all(-1)
        W = np.broadcast_to(w[:, None], q.shape)[q]
        res['quality'][name] = {c: wcorr(P['ip3_sv_Eh'][..., i][q], hx[..., i][q], W) for i, c in enumerate('nrk')}
    a.output.write_text(json.dumps(res, indent=1))
    keys = ('pi x rho', 'rho x rho', 'rho x 3pi', 'overall')
    for n, d in res['auc'].items():
        print(f'{n:36s}', ' '.join(f'{d[k]:.4f}' for k in keys))
    for n, d in res['quality'].items():
        print(f'{n:36s} rho side E[h] corr n/r/k', ' '.join(f'{d[c]:.3f}' for c in 'nrk'))


if __name__ == '__main__':
    main()
