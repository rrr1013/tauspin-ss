"""Per-side transverse h quality vs beta*gamma, per truth decay mode, with the
visible-energy fraction x reweighted to the mode's inclusive x distribution in
every beta*gamma bin (so the curves show boost at fixed x mix).

Also the per-side quantities that set the physics scales: opening angle
between visible and tau direction (~1/beta*gamma) against the reco visible
axis error, per mode.
"""
import json
import os

import numpy as np

from boost_map import load, wcorr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'cp_mixing', 'data')
MODES = {0: 'pi', 1: 'rho', 3: '3pi'}


def angle(a, b):
    c = np.einsum('ni,ni->n', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.arccos(np.clip(c, -1, 1))


def main():
    D = load()
    L = np.load(os.path.join(DATA, 'ladder_validation.npz'))
    w = np.repeat(D['w'], 2)
    bg, x, mode = D['bg'].reshape(-1), D['x_vis'].reshape(-1), D['modes'].reshape(-1)
    th = D['theta_vis_tau'].reshape(-1)
    tv = L['truth_visible_tau_lab4'][..., :3].reshape(-1, 3)
    rv = L['reco_visible_tau_lab4'][..., :3].reshape(-1, 3)
    axis_err = angle(tv, rv)
    R = D['h']['exact'].reshape(-1, 3)
    P = {a: D['h'][a].reshape(-1, 3) for a in ['base', 'ipsv', 'idealip']}
    xb = np.linspace(0, 1, 11)
    out = {}
    for mcode, mname in MODES.items():
        mm = mode == mcode
        e = np.quantile(bg[mm], np.linspace(0, 1, 7))
        hx_all, _ = np.histogram(x[mm], xb, weights=w[mm])
        hx_all = hx_all / hx_all.sum()
        rows = []
        for i in range(6):
            m = mm & (bg >= e[i]) & (bg <= e[i + 1])
            hx, _ = np.histogram(x[m], xb, weights=w[m])
            hx = hx / hx.sum()
            ratio = np.where(hx > 0, hx_all / np.maximum(hx, 1e-12), 0)
            rw = w[m] * ratio[np.clip(np.digitize(x[m], xb) - 1, 0, 9)]
            row = {'bg_med': float(np.median(bg[m])), 'n': int(m.sum()),
                   'x_med': float(np.median(x[m])),
                   'theta_vis_tau_mrad_med': float(1e3 * np.median(th[m])),
                   'axis_err_mrad_med': float(1e3 * np.median(axis_err[m])),
                   'axis_err_over_theta_med': float(np.median(axis_err[m] / th[m]))}
            for a in P:
                for tag, ww in (('raw', w[m]), ('xrw', rw)):
                    row[f'{a}_T_{tag}'] = float(np.mean([wcorr(P[a][m, c], R[m, c], ww) for c in (0, 1)]))
                    row[f'{a}_k_{tag}'] = wcorr(P[a][m, 2], R[m, 2], ww)
            # bootstrap se for base and ipsv-base (xrw)
            rng = np.random.default_rng(i + 10 * mcode)
            ii = np.flatnonzero(m)
            rwf = dict(zip(ii, rw))
            bb, dd = [], []
            for _ in range(60):
                k = rng.choice(len(ii), len(ii))
                kk = ii[k]
                ww = rw[k]
                tb = np.mean([wcorr(P['base'][kk, c], R[kk, c], ww) for c in (0, 1)])
                ti = np.mean([wcorr(P['ipsv'][kk, c], R[kk, c], ww) for c in (0, 1)])
                bb.append(tb)
                dd.append(ti - tb)
            row['base_T_xrw_se'] = float(np.std(bb))
            row['ipsv_minus_base_T_xrw_se'] = float(np.std(dd))
            rows.append(row)
        out[mname] = rows
        print(mname)
        for r in rows:
            print(f"  bg {r['bg_med']:6.0f} x {r['x_med']:.2f} th {r['theta_vis_tau_mrad_med']:5.2f} axErr {r['axis_err_mrad_med']:.3f} ratio {r['axis_err_over_theta_med']:.3f} | "
                  f"base T {r['base_T_xrw']:.3f}±{r['base_T_xrw_se']:.3f} ipsv {r['ipsv_T_xrw']:.3f} (+{r['ipsv_T_xrw']-r['base_T_xrw']:.3f}±{r['ipsv_minus_base_T_xrw_se']:.3f}) ideal {r['idealip_T_xrw']:.3f} | k base {r['base_k_xrw']:.3f}")
    with open(os.path.join(HERE, 'results_mode.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
