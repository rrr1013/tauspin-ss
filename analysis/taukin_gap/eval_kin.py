"""Evaluate posterior_kin.py arms: H/Z AUC by truth pair, per-side h quality and tau-energy quality, PIT, ESS, side swap.

  --arm NAME=DIR ...   posterior_kin outputs (DIR/posterior.npz); all must cover the same leading validation rows
  --boot N             paired event bootstrap for --delta pairs 'A:meas - B:meas' (B may be 'learner')
Same cohort, weights and weighted AUC as ../atlas_classical_posterior/evaluate.py; learner = gen3pi p11 full22
(fixed readout scores of the 2-seed ensemble for AUC, h_pred of s42 for side quality).  Test is never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN, lr_of, pair_masks, wauc  # noqa: E402
from hquality import wcorr  # noqa: E402

MEAS = ('met', 'ip3_sv', 'ip3_sv_mass')


def side_groups(modes, rmode, av):
    return {'pi': av & (rmode == 0) & (modes == 0), 'rho': av & (rmode == 1) & (modes == 1),
            '3pi': av & (rmode == 3) & (modes == 3), 'reco pi, truth rho': av & (rmode == 0) & (modes == 1),
            'reco rho, truth pi': av & (rmode == 1) & (modes == 0)}


def point_lr(h):
    a, b = h[:, 0], h[:, 1]
    T = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2]
    L = a[:, 2] * b[:, 2]
    return np.log(np.clip(1 + T, 1e-3, None)) - np.log(np.clip(1 + L, 1e-3, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=0)
    ap.add_argument('--delta', nargs='*', default=[])
    ap.add_argument('--swap', default='', help='arm name: side-swap attribution with the learner (point readout)')
    a = ap.parse_args()
    arms = dict(s.split('=', 1) for s in a.arm)
    P = {k: np.load(Path(v) / 'posterior.npz') for k, v in arms.items()}
    ref = P[next(iter(P))]
    n = len(ref['global_indices'])
    ids, y, w = ref['global_indices'], ref['labels'].astype(int), ref['weights'].astype(float)
    for k, d in P.items():
        assert np.array_equal(d['global_indices'], ids), k
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    assert np.array_equal(t['global_indices'][v][:n], ids)
    hx = t['h'][v][:n].astype(float)
    modes, rmode, av = ref['modes'], ref['reco_h_mode'].astype(int), ref['reco_h_available_side']
    etrue = ref['etau_true']
    dd = np.load(GEN / 'p11_readout' / 'readout_scores_ens_full22.npz')
    assert np.array_equal(dd['global_indices'][:n], ids)
    scores = {'exact h': lr_of(hx), 'learner': dd['scores'][:n]}
    for k, d in P.items():
        for m in MEAS:
            scores[f'{k}:{m}'] = d[f'{m}_lr']
    sel = pair_masks(modes)
    res = {'rows': int(n), 'arms': arms, 'events_truth_pairs': {p: int(m.sum()) for p, m in sel.items()}}
    res['auc'] = {k: {p: wauc(s[m], y[m], w[m]) for p, m in sel.items()} for k, s in scores.items()}
    # per-side quality
    hl = {'learner full22 s42': np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')['h_pred'][:n].astype(float),
          'learner base s43': np.load(GEN / 'p11' / 'base_s43' / 'validation_predictions.npz')['h_pred'][:n].astype(float)}
    rows = dict(hl)
    for k, d in P.items():
        for m in MEAS:
            rows[f'{k}:{m}'] = d[f'{m}_Eh']
    grp = side_groups(modes, rmode, av)
    W = np.broadcast_to(w[:, None], av.shape)
    hq = {}
    for name, h in rows.items():
        o = {}
        for lab, msk in grp.items():
            msk = msk & np.isfinite(h).all(-1)
            o[lab] = {c: round(wcorr(h[..., i][msk], hx[..., i][msk], W[msk]), 4) for i, c in enumerate('nrk')}
            o[lab]['sides'] = int(msk.sum())
        hq[name] = o
    res['side_quality'] = hq
    # tau energy: posterior mean vs truth, PIT
    te = {}
    for k, d in P.items():
        for m in MEAS:
            Et = np.stack([d[f'{m}_meanEt_{s}'] for s in (0, 1)], 1)
            pit = np.stack([d[f'{m}_pitEt_{s}'] for s in (0, 1)], 1)
            o = {}
            for lab, msk in grp.items():
                r = Et[msk] / etrue[msk] - 1
                hist, _ = np.histogram(pit[msk], np.linspace(0, 1, 11))
                o[lab] = {'corr_logE': round(wcorr(np.log(Et[msk]), np.log(etrue[msk]), W[msk]), 4),
                          'rel_q16_50_84': np.quantile(r, [.16, .5, .84]).round(4).tolist(),
                          'pit_hist10': (hist / hist.sum()).round(4).tolist(),
                          'pit_max_cdf_dev': round(float(np.max(np.abs(np.cumsum(hist) / hist.sum() - np.linspace(.1, 1, 10)))), 4)}
            te[f'{k}:{m}'] = o
    res['tau_energy'] = te
    res['ess'] = {f'{k}:{m}': {'pair_q05': float(np.quantile(d[f'{m}_ess_pair'], .05)),
                               'pair_median': float(np.median(d[f'{m}_ess_pair'])),
                               'no_valid': float(np.mean(d[f'{m}_nvalid'] == 0))} for k, d in P.items() for m in MEAS}
    if a.swap:
        hc = np.where(np.isfinite(P[a.swap]['ip3_sv_mass_Eh']), P[a.swap]['ip3_sv_mass_Eh'], 0.0)
        hL = hl['learner full22 s42']
        sw = {'classical point': point_lr(hc), 'learner point': point_lr(hL)}
        for T, nm in ((0, 'pi'), (1, 'rho'), (3, '3pi')):
            msk = (rmode == T)[..., None]
            sw[f'classical, learner on reco {nm}'] = point_lr(np.where(msk, hL, hc))
            sw[f'learner, classical on reco {nm}'] = point_lr(np.where(msk, hc, hL))
        res['swap'] = {k: {p: wauc(s[m], y[m], w[m]) for p, m in sel.items()} for k, s in sw.items()}
        scores.update({f'swap:{k}': s for k, s in sw.items()})
    if a.boot:
        rng = np.random.default_rng(1)
        res['paired_delta'] = {}
        for spec in a.delta:
            p, q = (x.strip() for x in spec.split(' - '))
            out = {}
            for pair in ('overall', 'pi x pi', 'pi x rho', 'rho x rho', 'rho x 3pi', '3pi x 3pi'):
                idx = np.flatnonzero(sel[pair])
                d0 = wauc(scores[p][idx], y[idx], w[idx]) - wauc(scores[q][idx], y[idx], w[idx])
                bs = []
                for _ in range(a.boot if pair == 'overall' else a.boot // 2):
                    b = idx[rng.integers(0, len(idx), len(idx))]
                    bs.append(wauc(scores[p][b], y[b], w[b]) - wauc(scores[q][b], y[b], w[b]))
                out[pair] = {'delta': round(d0, 5), 'ci95': np.round(np.quantile(bs, [0.025, 0.975]), 5).tolist()}
            res['paired_delta'][spec] = out
    np.savez_compressed(a.output.with_suffix('.scores.npz'), **{k.replace(' ', '_'): v for k, v in scores.items()},
                        labels=y, weights=w, modes=modes)
    a.output.write_text(json.dumps(res, indent=1))
    print(f"{'score':45s} overall  pixpi  pixrho rhoxrho rhox3pi 3pix3pi")
    for k in scores:
        r = res['auc'].get(k) or res['swap'].get(k.replace('swap:', ''))
        print(f"{k:45s} " + ' '.join(f"{r[p]:.4f}" for p in ('overall', 'pi x pi', 'pi x rho', 'rho x rho', 'rho x 3pi', '3pi x 3pi')))
    print('side quality (n/r/k)')
    for k, o in hq.items():
        print(f"{k:35s} " + '  '.join(f"{g}:{o[g]['n']:.3f}/{o[g]['r']:.3f}/{o[g]['k']:.3f}" for g in ('pi', 'rho', '3pi')))
    print('tau energy (pi sides): corr logE, rel q16/50/84, PIT max dev')
    for k, o in te.items():
        print(f"{k:35s} " + '  '.join(f"{g}: {o[g]['corr_logE']:.3f} {o[g]['rel_q16_50_84']} {o[g]['pit_max_cdf_dev']:.3f}" for g in ('pi', 'rho')))
    for k, o in res['ess'].items():
        print(k, o)


if __name__ == '__main__':
    main()
