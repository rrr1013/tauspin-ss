"""Separate the tau boost from the neutrino energy share.

At low beta*gamma the visible-pT selection keeps mostly high visible-energy
fraction x (soft neutrino), so a 1D beta*gamma binning mixes the two.  Here:

  * per-side h quality (corr of predicted vs exact transverse / longitudinal
    component) on a (beta*gamma, x) grid, both sides pooled;
  * event-level AUC on a (geometric-mean beta*gamma, mean x) grid, using the
    cross-fitted per-event scores written by boost_map.py;
  * AUC in bins of the collinear-approximation x from reco visible + MET, a
    quantity available in data;
  * weighted straight-line fits of the AUC differences vs log(beta*gamma)
    within x terciles.
"""
import json
import os

import numpy as np

from boost_map import load, wauc, wcorr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'cp_mixing', 'data')
ARMS = ['exact', 'base', 'ipsv', 'idealip', 'cleo_ipsv', 'cleo_shuffle']


def x_collinear(L):
    vis = L['reco_visible_tau_lab4']
    met = L['reco_met_xy']
    a = np.stack([vis[:, 0, :2], vis[:, 1, :2]], axis=2)          # (N, 2 xy, 2 sides)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    c0 = (met[:, 0] * a[:, 1, 1] - met[:, 1] * a[:, 0, 1]) / det
    c1 = (a[:, 0, 0] * met[:, 1] - a[:, 1, 0] * met[:, 0]) / det
    r = np.stack([c0, c1], 1)                                     # nu_T = r * vis_T
    return 1.0 / (1.0 + r)                                        # x = E_vis / E_tau (collinear)


def boot_auc(s, y, w, ii, arms, pairs, nboot, rng):
    out = {'n': int(len(ii)), 'auc': {a: wauc(s[a][ii], y[ii], w[ii]) for a in arms}}
    B = {a: [] for a in arms}
    for _ in range(nboot):
        k = rng.choice(ii, len(ii))
        for a in arms:
            B[a].append(wauc(s[a][k], y[k], w[k]))
    B = {a: np.array(v) for a, v in B.items()}
    out['auc_se'] = {a: float(B[a].std()) for a in arms}
    out['diff'] = {f'{a}-{b}': out['auc'][a] - out['auc'][b] for a, b in pairs}
    out['diff_se'] = {f'{a}-{b}': float((B[a] - B[b]).std()) for a, b in pairs}
    return out


def wls_slope(x, y, se):
    wt = 1 / np.asarray(se) ** 2
    X = np.stack([np.ones_like(x), x], 1)
    cov = np.linalg.inv((X * wt[:, None]).T @ X)
    b = cov @ (X * wt[:, None]).T @ y
    chi2 = float(np.sum(wt * (y - X @ b) ** 2))
    return {'slope': float(b[1]), 'slope_se': float(np.sqrt(cov[1, 1])), 'chi2': chi2, 'ndf': len(x) - 2}


def main():
    D = load()
    L = np.load(os.path.join(DATA, 'ladder_validation.npz'))
    S = np.load(os.path.join(HERE, 'scores.npz'))
    y, w = S['y'], S['w']
    s = {a: S[f's_{a}'] for a in ARMS}
    rng = np.random.default_rng(7)
    pairs = [('ipsv', 'base'), ('idealip', 'base'), ('exact', 'base'), ('cleo_ipsv', 'cleo_shuffle')]
    res = {}

    # ---- per-side h quality on (bg, x) grid ----
    bg_s, x_s = D['bg'].reshape(-1), D['x_vis'].reshape(-1)
    ws = np.repeat(w, 2)
    bg_e = np.quantile(bg_s, np.linspace(0, 1, 7))
    x_e = np.quantile(x_s, np.linspace(0, 1, 6))
    hq = {}
    for a in ['base', 'ipsv', 'idealip']:
        P = D['h'][a].reshape(-1, 3)
        R = D['h']['exact'].reshape(-1, 3)
        grid = np.full((6, 5, 2), np.nan)
        for i in range(6):
            for j in range(5):
                m = (bg_s >= bg_e[i]) & (bg_s <= bg_e[i + 1]) & (x_s >= x_e[j]) & (x_s <= x_e[j + 1])
                if m.sum() < 300:
                    continue
                grid[i, j, 0] = np.mean([wcorr(P[m, c], R[m, c], ws[m]) for c in (0, 1)])
                grid[i, j, 1] = wcorr(P[m, 2], R[m, 2], ws[m])
        hq[a] = grid.tolist()
    res['side_grid'] = {'bg_edges': bg_e.tolist(), 'x_edges': x_e.tolist(), 'hq': hq,
                        'counts': [[int(((bg_s >= bg_e[i]) & (bg_s <= bg_e[i + 1]) & (x_s >= x_e[j]) & (x_s <= x_e[j + 1])).sum())
                                    for j in range(5)] for i in range(6)]}
    # also 1D per-side profiles vs bg in x terciles
    x_t = np.quantile(x_s, [0, 1 / 3, 2 / 3, 1])
    prof = {}
    for j in range(3):
        mj = (x_s >= x_t[j]) & (x_s <= x_t[j + 1])
        e = np.quantile(bg_s[mj], np.linspace(0, 1, 7))
        rows = []
        for i in range(6):
            m = mj & (bg_s >= e[i]) & (bg_s <= e[i + 1])
            row = {'bg_med': float(np.median(bg_s[m])), 'n': int(m.sum())}
            for a in ['base', 'ipsv', 'idealip']:
                P = D['h'][a].reshape(-1, 3)
                R = D['h']['exact'].reshape(-1, 3)
                row[a] = {'T': float(np.mean([wcorr(P[m, c], R[m, c], ws[m]) for c in (0, 1)])),
                          'k': wcorr(P[m, 2], R[m, 2], ws[m])}
            rows.append(row)
        prof[f'x_tercile_{j}'] = {'x_lo': float(x_t[j]), 'x_hi': float(x_t[j + 1]), 'rows': rows}
    res['side_profile'] = prof

    # ---- event-level AUC on (bg_gm, x_mean) grid: 3 x-terciles x 4 bg quartiles ----
    bg_gm = np.sqrt(D['bg'][:, 0] * D['bg'][:, 1])
    x_mean = D['x_vis'].mean(1)
    xt = np.quantile(x_mean, [0, 1 / 3, 2 / 3, 1])
    grid = []
    for j in range(3):
        mj = (x_mean >= xt[j]) & (x_mean <= xt[j + 1])
        e = np.quantile(bg_gm[mj], np.linspace(0, 1, 5))
        row = []
        for i in range(4):
            ii = np.flatnonzero(mj & (bg_gm >= e[i]) & (bg_gm <= e[i + 1]))
            st = boot_auc(s, y, w, ii, ARMS, pairs, 150, rng)
            st['bg_med'] = float(np.median(bg_gm[ii]))
            st['x_med'] = float(np.median(x_mean[ii]))
            row.append(st)
        fits = {}
        lb = np.log([r['bg_med'] for r in row])
        for key in ['exact', 'base', 'ipsv', 'idealip']:
            fits[key] = wls_slope(lb, np.array([r['auc'][key] for r in row]), [r['auc_se'][key] for r in row])
        for key in ['ipsv-base', 'idealip-base', 'cleo_ipsv-cleo_shuffle']:
            fits[key] = wls_slope(lb, np.array([r['diff'][key] for r in row]), [r['diff_se'][key] for r in row])
        grid.append({'x_lo': float(xt[j]), 'x_hi': float(xt[j + 1]), 'bins': row, 'fits_vs_log_bg': fits})
        print(f'x tercile {j} [{xt[j]:.2f},{xt[j+1]:.2f}]',
              [f"{r['bg_med']:.0f}: {r['auc']['exact']:.3f}/{r['auc']['base']:.3f}/{r['auc']['ipsv']:.3f}/{r['auc']['idealip']:.3f}" for r in row])
        print('   slopes/ln(bg):', {k: f"{v['slope']:+.4f}±{v['slope_se']:.4f}" for k, v in fits.items()})
    res['event_grid'] = grid

    # ---- x bins alone (truth mean x) and collinear x (reco observable) ----
    xc = x_collinear(L)
    ok = np.all((xc > 0) & (xc < 1), axis=1)
    res['x_coll_valid_frac'] = float(np.average(ok, weights=w))
    xc_mean = np.where(ok, xc.mean(1), np.nan)
    for name, v, mask in [('x_mean_truth', x_mean, np.ones(len(y), bool)), ('x_coll_mean', xc_mean, ok)]:
        e = np.quantile(v[mask], np.linspace(0, 1, 7))
        rows = []
        for i in range(6):
            ii = np.flatnonzero(mask & (v >= e[i]) & (v <= e[i + 1]))
            st = boot_auc(s, y, w, ii, ARMS, pairs, 150, rng)
            st['med'] = float(np.median(v[ii]))
            st['bg_gm_med'] = float(np.median(bg_gm[ii]))
            rows.append(st)
        res[name] = rows
        print(name, [f"{r['med']:.2f}: {r['auc']['exact']:.3f}/{r['auc']['base']:.3f}/{r['auc']['ipsv']:.3f}/{r['auc']['idealip']:.3f}" for r in rows])
    ii = np.flatnonzero(~ok)
    st = boot_auc(s, y, w, ii, ARMS, pairs, 150, rng)
    res['x_coll_invalid'] = st
    print('x_coll invalid', res['x_coll_valid_frac'], {a: round(v, 3) for a, v in st['auc'].items()})
    with open(os.path.join(HERE, 'results_grid.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
