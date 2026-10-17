"""Training-composition control: does the learner's own-side h carry the other tau's spin, and through which prior?

Arms (validation predictions of the point-h network, all base16 = geometry hidden, seed 43 unless noted):
  mix       existing gen3pi p11 base_s43 (full mixed TRAIN)
  mixhalf   random half of TRAIN (both labels)              } this run, s5
  Honly     TRAIN H only                                    }
  Zonly     TRAIN Z only                                    }
  full22    existing gen3pi p11 full22 s42/s43 mean (+IP/SV), for reference
  exact     the exact h (control)

Per arm, on validation (test never loaded):
  cross     within H and within Z, WLS  h_pred[side,c] = a h_exact[side,c] + b h_exact[other,c] + const, on sides of the
            given reco mode whose partner has a valid exact h; event bootstrap CI for b.
  plugin    diag of 9 <h-_i h+_i> (the plug-in spin-correlation estimate) for H and Z, by truth pair (rho x rho, all)
  dphi      static first harmonic of 10/01: m = mean exp(i(phi- - phi+)), phi = atan2(h_r, h_n), unit weight,
            truth rho x rho; amplitude 2|m| and its real part, for H and Z
  quality   weighted correlation of each component with the exact h on rho sides (truth rho, reco rho)
  decomp    (review M1) on truth rho x rho, per label and component: own-side fit p = a h + const pooled over both sides,
            residual r = p - a h - const; the plug-in 9<p- p+> split into 9<(a h- + c)(a h+ + c)> (own-side exact part)
            and the rest; 9<r- r+> is the part carried by neither side's own exact h
  auc       weighted H/Z AUC of the exact-LR formula applied to the arm's h, by truth pair (no trained readout: the
            single-class arms have no two-class TRAIN predictions to train one on)
"""
import argparse
import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_readouts import formula, pair_masks, wauc  # noqa: E402

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts')


def wcorr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float(np.average(a * b, weights=w) / np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w)))


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw, Xw.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--suffix', default='', help="e.g. _s44: evaluate this run's seed-replication arms")
    a = ap.parse_args()
    vt = np.load(GEN / 'targets' / 'validation.npz')
    vv = vt['h_valid'].astype(bool)
    ids = vt['global_indices'][vv]
    hx, w, y, modes = vt['h'][vv].astype(float), vt['weights'][vv].astype(float), vt['labels'][vv].astype(int), vt['modes'][vv]
    V = np.load(LADDER / 'validation.npz')
    assert np.array_equal(V['global_indices'], ids)
    rmode, avail = V['reco_h_mode'].astype(int), V['reco_h_available_side']

    def load(path):
        d = np.load(path)
        assert np.array_equal(d['global_indices'], ids)
        return d['h_pred'].astype(float)
    arms = {'exact': hx}
    if a.suffix:   # seed replication: every arm from this run's out/p11/<name><suffix>
        for name in ('mix', 'Honly', 'Zonly', 'flatw'):
            p = a.run_dir / 'out' / 'p11' / f'{name}{a.suffix}' / 'validation_predictions.npz'
            if p.exists():
                arms[name] = load(p)
    else:
        arms['mix'] = load(GEN / 'p11' / 'base_s43' / 'validation_predictions.npz')
        for name in ('mixhalf', 'Honly', 'Zonly', 'flatw'):
            p = a.run_dir / 'out' / 'p11' / name / 'validation_predictions.npz'
            if p.exists():
                arms[name] = load(p)
    arms['full22'] = 0.5 * (load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')
                            + load(GEN / 'p11' / 'full22_s43' / 'validation_predictions.npz'))

    rng = np.random.default_rng(20261018)
    okx = np.isfinite(hx).all(-1)
    rhorho = (modes == 1).all(1)
    res = {}
    for name, h in arms.items():
        r = {'cross': {}, 'plugin': {}, 'dphi': {}, 'quality': {}}
        for side_sel, sname in (((rmode == 1) & (modes == 1), 'rho'), ((rmode == 0) & (modes == 0), 'pi'),
                                (avail & okx, 'all')):
            sel = side_sel & avail & okx & okx[:, ::-1]
            r['cross'][sname] = {}
            for lab, ln in ((1, 'H'), (0, 'Z')):
                ev, side = np.nonzero(sel & (y[:, None] == lab))
                uniq = np.unique(ev)
                r['cross'][sname][ln] = {}
                for j, c in enumerate('nrk'):
                    X = np.stack([hx[ev, side, j], hx[ev, 1 - side, j], np.ones(len(ev))], 1)
                    t, ww = h[ev, side, j], w[ev]
                    coef = wls(X, t, ww)
                    bs = []
                    for _ in range(a.boot):
                        cnt = np.bincount(rng.choice(uniq, len(uniq)), minlength=len(y))[ev]
                        bs.append(wls(X, t, ww * cnt)[1])
                    r['cross'][sname][ln][c] = [float(coef[0]), float(coef[1]),
                                                float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        for pm, pname in ((rhorho, 'rho x rho'), (np.ones(len(y), bool), 'all')):
            for lab, ln in ((1, 'H'), (0, 'Z')):
                m = pm & (y == lab) & okx.all(1)
                r['plugin'][f'{pname} {ln}'] = [float(9 * np.average(h[m, 0, i] * h[m, 1, i], weights=w[m])) for i in range(3)]
                phi = np.arctan2(h[m, :, 1], h[m, :, 0])
                mm = np.mean(np.exp(1j * (phi[:, 0] - phi[:, 1])))
                if pname == 'rho x rho':
                    r['dphi'][ln] = [float(2 * abs(mm)), float(2 * mm.real), int(m.sum())]
        q = (rmode == 1) & (modes == 1) & avail & okx
        Wq = np.broadcast_to(w[:, None], q.shape)[q]
        r['quality'] = {c: wcorr(h[..., i][q], hx[..., i][q], Wq) for i, c in enumerate('nrk')}
        r['decomp'] = {}
        for lab, ln in ((1, 'H'), (0, 'Z')):
            m = rhorho & (y == lab) & okx.all(1) & np.isfinite(h).all((1, 2))
            ww = w[m]
            for j, c in enumerate('nrk'):
                P, X = h[m][..., j], hx[m][..., j]
                Xs = np.stack([X.reshape(-1), np.ones(X.size)], 1)
                coef = wls(Xs, P.reshape(-1), np.repeat(ww, 2))
                fit = coef[0] * X + coef[1]
                res_ = P - fit
                tot = 9 * np.average(P[:, 0] * P[:, 1], weights=ww)
                own = 9 * np.average(fit[:, 0] * fit[:, 1], weights=ww)
                rr = 9 * np.average(res_[:, 0] * res_[:, 1], weights=ww)
                r['decomp'][f'{ln} {c}'] = {'a': float(coef[0]), 'total': float(tot), 'own_exact_part': float(own),
                                            'rest': float(tot - own), 'resid_resid': float(rr)}
        lr = formula(h)
        r['auc'] = {k: wauc(lr[m], y[m], w[m]) for k, m in pair_masks(modes).items()}
        res[name] = r
    a.output.write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(f'== {name}  quality rho n/r/k ' + '/'.join(f"{r['quality'][c]:.3f}" for c in 'nrk'))
        for sname in ('rho', 'pi', 'all'):
            print(f'   cross b [{sname:3s}] ' + '  '.join(
                f"{ln}{c}:{r['cross'][sname][ln][c][1]:+.3f}[{r['cross'][sname][ln][c][2]:+.3f},{r['cross'][sname][ln][c][3]:+.3f}]"
                for ln in ('H', 'Z') for c in 'nrk'))
        print('   plugin 9<hh> ' + '  '.join(f"{k}:" + ','.join(f'{v:+.3f}' for v in vals) for k, vals in r['plugin'].items()))
        print('   decomp rho x rho (total = own-exact part + rest; resid-resid) ' + '  '.join(
            f"{k}:{v['total']:+.2f}={v['own_exact_part']:+.2f}{v['rest']:+.2f}(rr{v['resid_resid']:+.2f})" for k, v in r['decomp'].items()))
        print('   auc formula ' + '  '.join(f"{k}:{v:.4f}" for k, v in r['auc'].items()))
        print('   dphi rho x rho  ' + '  '.join(f"{ln}: A={v[0]:.3f} Re={v[1]:+.3f} n={v[2]}" for ln, v in r['dphi'].items()))


if __name__ == '__main__':
    main()
