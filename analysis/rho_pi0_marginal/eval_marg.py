"""Evaluate the pi0-marginalised classical posterior against the fixed-pi0 posterior, the truth-pi0 oracle and the learner.

  --pit FILE ...   PIT / pi0-direction calibration on truth-rho marginalised sides (TRAIN or VALIDATION outputs)
  --val DIR        validation arms DIR/val_<model>_<prior>/posterior.npz -> AUCs by truth pair, paired bootstrap deltas,
                   cross-fit combination with the learner, rho-side h quality
Same cohort, weights and weighted AUC as ../atlas_classical_posterior/evaluate.py.  Test is never loaded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN, crossfit, lr_of, pair_masks, wauc  # noqa: E402
from hquality import wcorr  # noqa: E402

OLD = {'10/17 final (fixed, uniform x)': Path('/home/rbaba/atlas-classical-posterior-20261017/unpol/posterior.npz'),
       '10/18 truth pi0 oracle (uniform x)': Path('/home/rbaba/rho-beyond-exact-20261018/out/cp_truth/posterior.npz')}
ARMS = ('fixed_invE', 'fixed_train', 'marg_invE', 'marg_train', 'truth_train', 'cond_invE')
MEAS = ('met', 'ip3_sv', 'ip3_sv_mass', 'ip3_shuffle')


def pit_summary(path):
    P = np.load(path)
    out = {}
    side = P['marg_side'] & P['truth_rho_side']
    tr = P['truth_E2_mv_Etau']
    for m in ('met', 'ip3_sv_mass'):
        o = {'sides': int(side.sum())}
        for q in ('E2', 'MV', 'ET'):
            v = np.concatenate([P[f'{m}_pit{q}_{s}'][side[:, s]] for s in (0, 1)])
            h, _ = np.histogram(v, np.linspace(0, 1, 11))
            o[q] = {'hist10': (h / h.sum()).round(4).tolist(), 'mean': float(v.mean()), 'sd': float(v.std()),
                    'frac_outside_90': float(np.mean((v < 0.05) | (v > 0.95))),
                    'max_cdf_dev': float(np.max(np.abs(np.cumsum(h) / h.sum() - np.linspace(0.1, 1, 10))))}
        ang = np.concatenate([P[f'{m}_ang_{s}'][side[:, s]] for s in (0, 1)])
        o['pi0_dir_err_posterior_mean_mrad_q25_50_75'] = (1e3 * np.quantile(ang, [.25, .5, .75])).round(3).tolist()
        e2 = np.concatenate([P[f'{m}_meanE2_{s}'][side[:, s]] for s in (0, 1)])
        t2 = np.concatenate([tr[side[:, s], s, 0] for s in (0, 1)])
        o['pi0_E_rel_err_posterior_mean_q16_50_84'] = np.quantile(e2 / t2 - 1, [.16, .5, .84]).round(4).tolist()
        out[m] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pit', type=Path, nargs='*', default=[])
    ap.add_argument('--val', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=400)
    a = ap.parse_args()
    res = {'pit': {str(p): pit_summary(p) for p in a.pit}}
    if a.val:
        P = {k: np.load(a.val / f'val_{k}' / 'posterior.npz') for k in ARMS if (a.val / f'val_{k}' / 'posterior.npz').exists()}
        P.update({k: np.load(v) for k, v in OLD.items()})
        ref = P[next(iter(P))]
        ids, y, w = ref['global_indices'], ref['labels'].astype(int), ref['weights'].astype(float)
        for k, d in P.items():
            assert np.array_equal(d['global_indices'], ids), k
        modes = ref['modes']
        t = np.load(GEN / 'targets' / 'validation.npz')
        v = t['h_valid'].astype(bool)
        assert np.array_equal(t['global_indices'][v], ids)
        hx = t['h'][v].astype(float)
        scores = {'exact h': lr_of(hx)}
        for k, d in P.items():
            for m in MEAS:
                scores[f'{k}:{m}'] = d[f'{m}_lr']
        learner = {}
        for arm in ('base_s43', 'ens_full22'):
            dd = np.load(GEN / 'p11_readout' / f'readout_scores_{arm}.npz')
            assert np.array_equal(dd['global_indices'], ids)
            learner[arm] = scores[f'learner:{arm}'] = dd['scores']
        fold = (ref['event_numbers'] % 2).astype(int)
        L = learner['ens_full22'][:, None]
        for k in ARMS:
            if k in P:
                scores[f'combo learner+{k}:ip3_sv_mass'] = crossfit(np.c_[L, P[k]['ip3_sv_mass_lr']], y, w, fold, 'logistic')
                scores[f'combo learner+{k}:ip3_shuffle'] = crossfit(np.c_[L, P[k]['ip3_shuffle_lr']], y, w, fold, 'logistic')
        sel = pair_masks(modes)
        res['events_truth_pairs'] = {p: int(m.sum()) for p, m in sel.items()}
        res['auc'] = {k: {p: wauc(s[m], y[m], w[m]) for p, m in sel.items()} for k, s in scores.items()}
        rng = np.random.default_rng(1)
        deltas = []
        for e in ('train', 'invE'):
            for m in ('ip3_sv_mass', 'met', 'ip3_sv'):
                deltas.append((f'marg_{e}:{m}', f'fixed_{e}:{m}'))
            deltas.append((f'marg_{e}:ip3_sv_mass', 'learner:ens_full22'))
            deltas.append((f'fixed_{e}:ip3_sv_mass', 'learner:ens_full22'))
            deltas.append((f'combo learner+marg_{e}:ip3_sv_mass', 'learner:ens_full22'))
            deltas.append((f'combo learner+marg_{e}:ip3_shuffle', 'learner:ens_full22'))
            deltas.append((f'combo learner+fixed_{e}:ip3_sv_mass', 'learner:ens_full22'))
            deltas.append((f'marg_{e}:ip3_sv_mass', 'truth_train:ip3_sv_mass'))
        deltas.append(('fixed_train:ip3_sv_mass', 'fixed_invE:ip3_sv_mass'))
        for m in ('ip3_sv_mass', 'met'):
            deltas.append((f'cond_invE:{m}', f'marg_invE:{m}'))
            deltas.append((f'cond_invE:{m}', f'fixed_invE:{m}'))
        deltas.append(('cond_invE:ip3_sv_mass', 'learner:ens_full22'))
        deltas.append(('cond_invE:ip3_sv_mass', 'truth_train:ip3_sv_mass'))
        deltas.append(('combo learner+cond_invE:ip3_sv_mass', 'learner:ens_full22'))
        deltas.append(('combo learner+cond_invE:ip3_shuffle', 'learner:ens_full22'))
        deltas.append(('fixed_invE:ip3_sv_mass', '10/17 final (fixed, uniform x):ip3_sv_mass'))
        res['paired_delta'] = {}
        for p, q in deltas:
            if p not in scores or q not in scores:
                continue
            out = {}
            for pair in ('overall', 'rho x rho', 'pi x rho', 'rho x 3pi'):
                idx = np.flatnonzero(sel[pair])
                d0 = wauc(scores[p][idx], y[idx], w[idx]) - wauc(scores[q][idx], y[idx], w[idx])
                bs = []
                for _ in range(a.boot if pair == 'overall' else a.boot // 2):
                    b = idx[rng.integers(0, len(idx), len(idx))]
                    bs.append(wauc(scores[p][b], y[b], w[b]) - wauc(scores[q][b], y[b], w[b]))
                out[pair] = {'delta': d0, 'ci95': np.quantile(bs, [0.025, 0.975]).tolist()}
            res['paired_delta'][f'{p} - {q}'] = out
        # rho-side quality: truth rho AND reco rho, and all reco rho
        rmode, av = ref['reco_h_mode'].astype(int), ref['reco_h_available_side']
        hq = {}
        hp = {arm: np.load(GEN / 'p11' / arm / 'validation_predictions.npz')['h_pred'].astype(float)
              for arm in ('full22_s42', 'base_s43')}
        rows = {f'learner {k}': v for k, v in hp.items()}
        for k, d in P.items():
            for m in ('met', 'ip3_sv_mass'):
                rows[f'{k}:{m}'] = d[f'{m}_Eh']
        for name, h in rows.items():
            o = {}
            for lab, msk in (('truth rho & reco rho', av & (rmode == 1) & (modes == 1)), ('reco rho, truth pi', av & (rmode == 1) & (modes == 0)),
                             ('reco pi', av & (rmode == 0)), ('reco 3pi', av & (rmode == 3))):
                msk = msk & np.isfinite(h).all(-1)
                ww = np.broadcast_to(w[:, None], msk.shape)[msk]
                o[lab] = {c: wcorr(h[..., i][msk], hx[..., i][msk], ww) for i, c in enumerate('nrk')}
                o[lab]['sides'] = int(msk.sum())
            hq[name] = o
        res['side_quality'] = hq
        res['ess'] = {k: {m: {'side_q05': float(np.quantile(np.r_[P[k][f'{m}_ess0'], P[k][f'{m}_ess1']], .05)),
                              'pair_q05': float(np.quantile(P[k][f'{m}_ess_pair'], .05)),
                              'no_valid': float(np.mean(P[k][f'{m}_nvalid'] == 0))} for m in ('met', 'ip3_sv_mass')}
                      for k in ARMS if k in P}
        np.savez_compressed(a.output.with_suffix('.scores.npz'), **{k.replace(' ', '_').replace('/', '-'): v for k, v in scores.items()},
                            labels=y, weights=w, modes=modes)
        for k in scores:
            print(f"{k:60s} all {res['auc'][k]['overall']:.4f}  rhorho {res['auc'][k]['rho x rho']:.4f}")
    a.output.write_text(json.dumps(res, indent=1))
    if a.pit:
        print(json.dumps(res['pit'], indent=1))


if __name__ == '__main__':
    main()
