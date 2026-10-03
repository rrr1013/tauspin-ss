"""Review follow-up: split the CLEO-world weight into its Dalitz part (u only) and
its spin part (f_CLEO/f_gen only), and report the offsets next to the responses.

Worlds: gen (w0), u-only (w0 * prod u), spin-only (w0 * f_C/f_g), full (w0 * w).
Offsets: A_perp of H reweighted to c = 0 (f_H(c=0)/f_H(c=1), built from the world's
true h), E_Z[sum_{3pi} h_k] at P_tau = 0, and the H null E_H[sum_{3pi} h_k].
Paired bootstrap over events (same draws in every world).
"""
import json

import numpy as np

import analyze as an
import reweight as rw
from responses import fH, fZ, wcov

BOOT = 300


def main():
    d, nets = an.load_all()
    y, w0, is3 = d['y'], d['w0'], d['is3']
    any3 = is3.any(1)
    u = np.prod(d['u'], axis=1)
    spin = rw.spin_factor(d['h_cleo'], y) / rw.spin_factor(d['h_gen'], y)
    worlds = {'gen': (w0, d['h_gen']), 'u_only': (w0 * u, d['h_gen']),
              'spin_only': (w0 * spin, d['h_cleo']), 'full': (w0 * d['w'], d['h_cleo'])}
    # in the u-only world the true polarimeter is still h_gen (spin structure unchanged)
    scores = {'exact LR h_gen': an.lr(d['h_gen'])}
    est = {'exact h_gen': d['h_gen'], 'exact h_CLEO': d['h_cleo']}
    for teacher in ('gen', 'cleo'):
        for arm in ('base_s43', 'full22_s42'):
            scores[f'{teacher}:{arm}'] = nets[(teacher, arm)]['score']
            est[f'reco {arm} / {teacher} teacher'] = nets[(teacher, arm)]['h_pred']
    sel = {'any 3pi': any3, '3pi x 3pi': is3.all(1)}
    rng = np.random.default_rng(11)
    out = {'auc_delta_vs_gen': {}, 'responses': {}, 'offsets': {}}
    for p, m in sel.items():
        idx = np.flatnonzero(m)
        draws = [rng.choice(idx, len(idx)) for _ in range(BOOT)]
        for sn, s in scores.items():
            base = an.wauc(s[idx], y[idx], w0[idx])
            row = {}
            for wn in ('u_only', 'spin_only', 'full'):
                ww = worlds[wn][0]
                pt = an.wauc(s[idx], y[idx], ww[idx]) - base
                bb = [an.wauc(s[b], y[b], ww[b]) - an.wauc(s[b], y[b], w0[b]) for b in draws]
                row[wn] = {'delta': pt, 'ci95': np.quantile(bb, [0.025, 0.975]).tolist()}
            out['auc_delta_vs_gen'].setdefault(sn, {})[p] = row
    # responses and offsets on events with a 3pi side
    mz, mh = any3 & (y == 0), any3 & (y == 1)
    iz, ih = np.flatnonzero(mz), np.flatnonzero(mh)

    def stats(bz, bh):
        res = {}
        for wn, (ww, hs) in worlds.items():
            sP = (hs[:, 0, 2] + hs[:, 1, 2]) / fZ(hs)
            sperp = (hs[:, 0, 0] * hs[:, 1, 0] + hs[:, 0, 1] * hs[:, 1, 1]) / fH(hs)
            to_p0 = 1 / fZ(hs) * (1 + hs[:, 0, 2] * hs[:, 1, 2])     # f_Z(P=0)/f_Z(P)
            to_c0 = fH(hs, 0.0) / fH(hs)                              # f_H(c=0)/f_H(c=1)
            for name, h in est.items():
                x = np.where(is3, h[..., 2], 0.0).sum(1)
                a = 0.5 * (h[:, 0, 0] * h[:, 1, 0] + h[:, 0, 1] * h[:, 1, 1])
                r = res.setdefault(name, {})
                r[wn] = [wcov(x[bz], sP[bz], ww[bz]),
                         wcov(a[bh], sperp[bh], ww[bh]),
                         np.sum((ww * to_c0)[bh] * a[bh]) / np.sum((ww * to_c0)[bh]),
                         np.sum((ww * to_p0)[bz] * x[bz]) / np.sum((ww * to_p0)[bz]),
                         np.sum(ww[bh] * x[bh]) / np.sum(ww[bh])]
        return res
    point = stats(iz, ih)
    boots = [stats(rng.choice(iz, len(iz)), rng.choice(ih, len(ih))) for _ in range(150)]
    names = ('s_k', 'R_perp', 'A_perp_at_c0', 'E_Z_x_at_P0', 'E_H_x')
    for n in est:
        out['responses'][n] = {}
        for wn in worlds:
            out['responses'][n][wn] = {}
            for i, q in enumerate(names):
                v = point[n][wn][i]
                if i < 2:
                    r = np.array([b[n][wn][i] / b[n]['gen'][i] for b in boots])
                    out['responses'][n][wn][q + '_ratio_to_gen'] = {
                        'value': v / point[n]['gen'][i], 'ci95': np.quantile(r, [0.025, 0.975]).tolist()}
                else:
                    dlt = np.array([b[n][wn][i] - b[n]['gen'][i] for b in boots])
                    se = np.std([b[n][wn][i] for b in boots])
                    out['responses'][n][wn][q] = {'value': v, 'se': float(se),
                                                  'shift_vs_gen_ci95': np.quantile(dlt, [0.025, 0.975]).tolist()}
    (rw.HERE / 'results' / 'decomposition.json').write_text(json.dumps(out, indent=2))
    for sn, v in out['auc_delta_vs_gen'].items():
        print(f'{sn:16s}', '  '.join(f"{p}: " + ' '.join(f"{wn}={x['delta']:+.4f}[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" for wn, x in r.items()) for p, r in v.items()))
    for n, v in out['responses'].items():
        print(n)
        for wn, r in v.items():
            print(f'   {wn:9s}', ' '.join(f"{q}={x['value']:+.4f}" for q, x in r.items()))


if __name__ == '__main__':
    main()
