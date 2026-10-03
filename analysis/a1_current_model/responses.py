"""Spin responses (calibration factors) of each estimator in both worlds.

For an estimator hhat and a world (weights ww, true polarimeter hs of that world):
  s_k   = dE_Z[sum_{3pi sides} hhat_k] / dP_tau          (Z, events with a 3pi side)
  R_kk  = dE_Z[hhat-_k hhat+_k] / dC_kk                  (Z)
  R_perp= dE_H[(hhat-_n hhat+_n + hhat-_r hhat+_r)/2] / dc,  C_H = diag(c, c, -1)   (H)
each = weighted Cov(observable, d log f / d parameter) with f built from hs.
A_perp has no acceptance offset (azimuthal symmetry), so R_perp(CLEO)/R_perp(gen) is
directly the bias of a gen-calibrated transverse-correlation (entanglement) measurement.
Paired bootstrap: the same resampled events in both worlds.
Also: world effect on the AUC in bins of m(3pi) for events with exactly one 3pi side.
"""
import json

import numpy as np

import analyze as an
import reweight as rw

BOOT = 400


def fH(h, c=1.0):
    p = h[:, 0] * h[:, 1]
    return 1 + c * (p[:, 0] + p[:, 1]) - p[:, 2]


def fZ(h):
    return 1 + rw.P_TAU_Z * (h[:, 0, 2] + h[:, 1, 2]) + h[:, 0, 2] * h[:, 1, 2]


def wcov(x, s, w):
    mx = np.sum(w * x) / w.sum()
    return float(np.sum(w * (x - mx) * s) / w.sum())


def build(d, est, scope='any 3pi'):
    y, is3 = d['y'], d['is3']
    any3 = is3.any(1) if scope == 'any 3pi' else np.ones(len(y), bool)
    mz, mh = any3 & (y == 0), any3 & (y == 1)
    side = is3 if scope == 'any 3pi' else np.ones_like(is3)
    pre = {}
    for world, hs in (('gen', d['h_gen']), ('cleo', d['h_cleo'])):
        pre[world] = dict(
            sP=(hs[:, 0, 2] + hs[:, 1, 2]) / fZ(hs),
            skk=hs[:, 0, 2] * hs[:, 1, 2] / fZ(hs),
            sperp=(hs[:, 0, 0] * hs[:, 1, 0] + hs[:, 0, 1] * hs[:, 1, 1]) / fH(hs))
    obs = {}
    for name, h in est.items():
        obs[name] = dict(x=np.where(side, h[..., 2], 0.0).sum(1), kk=h[:, 0, 2] * h[:, 1, 2],
                         perp=0.5 * (h[:, 0, 0] * h[:, 1, 0] + h[:, 0, 1] * h[:, 1, 1]))
    return mz, mh, pre, obs


def responses_on(idx_z, idx_h, d, pre, obs):
    w0, w = d['w0'], d['w']
    out = {}
    for name, o in obs.items():
        r = {}
        for world, ww in (('gen', w0), ('cleo', w0 * w)):
            p = pre[world]
            r[world] = (wcov(o['x'][idx_z], p['sP'][idx_z], ww[idx_z]),
                        wcov(o['kk'][idx_z], p['skk'][idx_z], ww[idx_z]),
                        wcov(o['perp'][idx_h], p['sperp'][idx_h], ww[idx_h]))
        out[name] = r
    return out


def main():
    d, nets = an.load_all()
    est = {'exact h_gen': d['h_gen'], 'exact h_CLEO': d['h_cleo']}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v:
            est[f'reco {arm} / {teacher} teacher'] = v['h_pred']
    rng = np.random.default_rng(7)
    names = ('s_k (P_tau)', 'R_kk', 'R_perp')
    out = {}
    for scope in ('any 3pi', 'all events'):
        mz, mh, pre, obs = build(d, est, scope)
        iz, ih = np.flatnonzero(mz), np.flatnonzero(mh)
        point = responses_on(iz, ih, d, pre, obs)
        boots = {n: [] for n in est}
        for _ in range(BOOT):
            bz, bh = rng.choice(iz, len(iz)), rng.choice(ih, len(ih))
            r = responses_on(bz, bh, d, pre, obs)
            for n in est:
                boots[n].append(np.array(r[n]['cleo']) / np.array(r[n]['gen']))
        out[scope] = {}
        for n in est:
            b = np.array(boots[n])
            out[scope][n] = {q: {'gen': point[n]['gen'][i], 'cleo': point[n]['cleo'][i],
                                 'ratio': point[n]['cleo'][i] / point[n]['gen'][i],
                                 'ratio_ci68': np.quantile(b[:, i], [0.16, 0.84]).tolist(),
                                 'ratio_ci95': np.quantile(b[:, i], [0.025, 0.975]).tolist()} for i, q in enumerate(names)}
    # AUC world effect vs m3pi (exactly one 3pi side)
    y, w0, w, is3 = d['y'], d['w0'], d['w'], d['is3']
    one = is3.sum(1) == 1
    m3 = np.where(is3, d['m3'], -1.0).max(1)
    edges = np.array([0.55, 1.0, 1.1, 1.2, 1.3, 1.777])
    scores = {'exact LR h_gen': an.lr(d['h_gen']), 'exact LR h_CLEO': an.lr(d['h_cleo']),
              'gen:full22_s42': nets[('gen', 'full22_s42')]['score'], 'cleo:full22_s42': nets[('cleo', 'full22_s42')]['score'],
              'gen:base_s43': nets[('gen', 'base_s43')]['score'], 'cleo:base_s43': nets[('cleo', 'base_s43')]['score']}
    mb = {}
    for i in range(len(edges) - 1):
        idx = np.flatnonzero(one & (m3 >= edges[i]) & (m3 < edges[i + 1]))
        row = {'events': int(len(idx))}
        for sn, s in scores.items():
            pt = an.wauc(s[idx], y[idx], (w0 * w)[idx]) - an.wauc(s[idx], y[idx], w0[idx])
            bb = []
            for _ in range(200):
                b = rng.choice(idx, len(idx))
                bb.append(an.wauc(s[b], y[b], (w0 * w)[b]) - an.wauc(s[b], y[b], w0[b]))
            row[sn] = {'delta': pt, 'ci68': np.quantile(bb, [0.16, 0.84]).tolist()}
        mb[f'{edges[i]:.2f}-{edges[i + 1]:.2f}'] = row
    res = {'responses': out, 'auc_world_effect_vs_m3pi': mb, 'm3pi_edges': edges.tolist()}
    (rw.HERE / 'results' / 'responses.json').write_text(json.dumps(res, indent=2))
    for n, v in out['all events'].items():
        print(f'{n:34s}', '  '.join(f"{q}: {x['ratio']:.3f} [{x['ratio_ci95'][0]:.3f},{x['ratio_ci95'][1]:.3f}]" for q, x in v.items()))
    for k, v in mb.items():
        print(k, v['events'], ' '.join(f"{sn}:{x['delta']:+.4f}" for sn, x in v.items() if sn != 'events'))


if __name__ == '__main__':
    main()
