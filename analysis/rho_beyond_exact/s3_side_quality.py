"""Per-side h quality on rho sides vs the size of the reco error, and cross-side dependence of the learner.

Sides: truth rho AND reco rho (reco_h_mode 1, available), validation.
Errors (truth from the gen3pi truth surface):
  dtheta_pi0   angle between reco pi0 (PFO sum) and truth pi0 momenta [mrad]
  rE_pi0       E_reco / E_truth of the pi0
  dy           reco - truth energy asymmetry y = (E_pi - E_pi0) / (E_pi + E_pi0) in the lab
  m_vis        reco visible mass [GeV]
  raw          reco_mode_raw (1 = 1p1n, 2 = 1p2n)
Rows: exact formula on reco visible + truth nu (s1 'reco') and its replacements, learner h_pred (full22 s42+s43
mean, base s43).  Weighted correlation with the exact (gen3pi) h, per component, in quantile bins.

Cross-side: within H and within Z separately, weighted least squares
  h_row[side, c] = a * h_exact[side, c] + b * h_exact[other, c] + const
on rho sides whose partner has a valid exact h.  b != 0 means the row's own-side h carries the other tau's
spin.  Event bootstrap for b.  Test never loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')
SURFACE = GEN / 'targets' / 'truth_surface_gen3pi.npz'


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw, Xw.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bins', type=int, default=6)
    ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args()
    V = np.load(LADDER / 'validation.npz')
    S = np.load(SURFACE)
    vt = np.load(GEN / 'targets' / 'validation.npz')
    vv = vt['h_valid'].astype(bool)
    ids = vt['global_indices'][vv]
    assert np.array_equal(V['global_indices'], ids) and np.array_equal(S['global_indices'], ids)
    hx = vt['h'][vv].astype(float)
    w = vt['weights'][vv].astype(float)
    y = vt['labels'][vv].astype(int)
    L = np.load(a.ladder_dir / 'ladder_validation.npz')
    rows = {f'formula {k}': L[f'h_{k}'] for k in ('reco', 'pi0dir', 'pi0E', 'pi0', 'pich')}
    hp = [np.load(GEN / 'p11' / arm / 'validation_predictions.npz') for arm in ('full22_s42', 'full22_s43', 'base_s43')]
    for d in hp:
        assert np.array_equal(d['global_indices'], ids)
    rows['learner full22'] = 0.5 * (hp[0]['h_pred'] + hp[1]['h_pred']).astype(float)
    rows['learner base'] = hp[2]['h_pred'].astype(float)

    rho = L['rho_side']
    rn, tn = V['reco_h_neutral_lab4'], S['pi0']
    rp, tp = V['reco_h_pions_lab4'][..., 0, :], S['pions'][..., 0, :]
    cosang = np.sum(rn[..., :3] * tn[..., :3], -1) / np.maximum(
        np.linalg.norm(rn[..., :3], axis=-1) * np.linalg.norm(tn[..., :3], axis=-1), 1e-300)
    vis = V['reco_visible_tau_lab4']
    var = {
        'dtheta_pi0': 1e3 * np.arccos(np.clip(cosang, -1, 1)),
        'rE_pi0': rn[..., 3] / np.maximum(tn[..., 3], 1e-12),
        'dy': (rp[..., 3] - rn[..., 3]) / (rp[..., 3] + rn[..., 3]) - (tp[..., 3] - tn[..., 3]) / (tp[..., 3] + tn[..., 3]),
        'm_vis': np.sqrt(np.maximum(vis[..., 3] ** 2 - np.sum(vis[..., :3] ** 2, -1), 0)),
        'raw': V['reco_mode_raw'].astype(float),
    }
    W = np.broadcast_to(w[:, None], rho.shape)
    good = rho & np.isfinite(hx).all(-1)
    for h in rows.values():
        good &= np.isfinite(h).all(-1)
    res = {'n_sides': int(good.sum()), 'overall': {}, 'binned': {}, 'cross': {}}
    for name, h in rows.items():
        res['overall'][name] = {c: wcorr(h[..., i][good], hx[..., i][good], W[good]) for i, c in enumerate('nrk')}
    for vn, x in var.items():
        xs = x[good]
        if vn == 'raw':
            edges = np.array([0.5, 1.5, 2.5])
        elif vn == 'dy':
            edges = np.quantile(np.abs(xs), np.linspace(0, 1, a.bins + 1))
            xs = np.abs(xs)
        else:
            edges = np.quantile(xs, np.linspace(0, 1, a.bins + 1))
        edges[0], edges[-1] = edges[0] - 1e-9, edges[-1] + 1e-9
        b = np.digitize(xs, edges) - 1
        out = {'edges': edges.tolist(), 'frac': [], 'rows': {k: {c: [] for c in 'nrk'} for k in rows}}
        for i in range(len(edges) - 1):
            s = b == i
            out['frac'].append(float(W[good][s].sum() / W[good].sum()))
            for name, h in rows.items():
                for j, c in enumerate('nrk'):
                    out['rows'][name][c].append(wcorr(h[..., j][good][s], hx[..., j][good][s], W[good][s]))
        res['binned'][vn] = out

    # cross-side dependence, per label
    rng = np.random.default_rng(20261018)
    other_ok = np.isfinite(hx).all(-1)[:, ::-1]
    for name, h in rows.items():
        res['cross'][name] = {}
        for lab, ln in ((1, 'H'), (0, 'Z')):
            ev, side = np.nonzero(good & other_ok & (y[:, None] == lab))
            res['cross'][name][ln] = {}
            for j, c in enumerate('nrk'):
                X = np.stack([hx[ev, side, j], hx[ev, 1 - side, j], np.ones(len(ev))], 1)
                t = h[ev, side, j]
                ww = w[ev]
                coef = wls(X, t, ww)
                bs = []
                uniq = np.unique(ev)
                for _ in range(a.boot):
                    pick = rng.choice(uniq, len(uniq))
                    cnt = np.bincount(pick, minlength=len(y))[ev]
                    bs.append(wls(X, t, ww * cnt)[1])
                res['cross'][name][ln][c] = [float(coef[0]), float(coef[1]),
                                             float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    a.output.write_text(json.dumps(res, indent=1))
    print('sides', res['n_sides'])
    for name, d in res['overall'].items():
        print(f'{name:18s}', ' '.join(f'{c}:{d[c]:.3f}' for c in 'nrk'))
    for vn, out in res['binned'].items():
        print(vn, 'edges', np.round(out['edges'], 3).tolist())
        for name in rows:
            print(f'  {name:18s} k', ' '.join(f'{v:.2f}' for v in out['rows'][name]['k']),
                  ' | n', ' '.join(f'{v:.2f}' for v in out['rows'][name]['n']))
    print('cross-side b (coef of the other side exact h), [CI]')
    for name, d in res['cross'].items():
        print(f'  {name:18s}', ' '.join(f"{ln}{c}:{d[ln][c][1]:+.3f}[{d[ln][c][2]:+.3f},{d[ln][c][3]:+.3f}]"
                                         for ln in ('H', 'Z') for c in 'nrk'))


if __name__ == '__main__':
    main()
