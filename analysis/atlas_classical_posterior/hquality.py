"""Per-side h quality: weighted correlation with the exact (generator-current) h, by reco mode.

Rows: classical posterior mean E[h] (own pair basis of each retained pair, averaged),
learner h_pred (gen3pi full22 s42 / base s43), and the reco visible system with the
truth neutrinos.  Sides whose reco polarimeter is unavailable are skipped.  Also the
pair statistic E[T] = sum_i s_i E[h-_i h+_i] (s = (1,1,-1)) vs the exact T, by truth pair.
"""
import argparse
import json
from pathlib import Path

import numpy as np

GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
S = np.array([1.0, 1.0, -1.0])


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posterior', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    P0 = np.load(a.posterior[0])
    ids, w = P0['global_indices'], P0['weights'].astype(float)
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    assert np.array_equal(t['global_indices'][v], ids)
    hx = t['h'][v].astype(float)
    rows = {}
    for arm in ('full22_s42', 'base_s43'):
        d = np.load(GEN / 'p11' / arm / 'validation_predictions.npz')
        assert np.array_equal(d['global_indices'], ids)
        rows[f'learner {arm}'] = (d['h_pred'].astype(float), None)
    rows['reco visible + truth nu'] = (P0['truthnu_h'], None)
    for path in a.posterior:
        P = np.load(path)
        assert np.array_equal(P['global_indices'], ids)
        tag = path.parent.name
        for m in ('met', 'ip3_sv', 'ip3_sv_mass'):
            rows[f'classical[{tag}] {m}'] = (P[f'{m}_Eh'], P[f'{m}_Ehh'])
    rmode = P0['reco_h_mode'].astype(int)
    av = P0['reco_h_available_side']
    tmodes = P0['modes']
    res = {'side': {}, 'pair_T': {}}
    for name, (h, hh) in rows.items():
        out = {}
        for md, lab in ((0, 'pi'), (1, 'rho'), (3, '3pi')):
            m = av & (rmode == md) & np.isfinite(h).all(-1) & np.isfinite(hx).all(-1)
            ww = np.broadcast_to(w[:, None], m.shape)[m]
            out[lab] = {c: wcorr(h[..., i][m], hx[..., i][m], ww) for i, c in enumerate('nrk')}
            out[lab]['sides'] = int(m.sum())
        res['side'][name] = out
        T = (hh.diagonal(axis1=1, axis2=2) @ S) if hh is not None else (h[:, 0] * h[:, 1]) @ S
        Tx = (hx[:, 0] * hx[:, 1]) @ S
        pt = {}
        for lab, (p, q) in {'pi x pi': (0, 0), 'pi x rho': (0, 1), 'rho x rho': (1, 1), 'pi x 3pi': (0, 3),
                            'rho x 3pi': (1, 3), '3pi x 3pi': (3, 3), 'overall': (None, None)}.items():
            m = np.isfinite(T) & np.isfinite(Tx)
            if p is not None:
                m &= ((tmodes[:, 0] == p) & (tmodes[:, 1] == q)) | ((tmodes[:, 0] == q) & (tmodes[:, 1] == p))
            pt[lab] = wcorr(T[m], Tx[m], w[m])
        res['pair_T'][name] = pt
    a.output.write_text(json.dumps(res, indent=1))
    for name, out in res['side'].items():
        print(f'{name:40s}', ' '.join(f"{lab}:" + ','.join(f"{out[lab][c]:.2f}" for c in 'nrk') for lab in ('pi', 'rho', '3pi')),
              f" T:{res['pair_T'][name]['overall']:.3f}")


if __name__ == '__main__':
    main()
