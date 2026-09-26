"""q8: the numerical claims of the two independent reviews, checked here.

The OpenAI gpt-5.6-sol review (review/validity_sol.md) says the corrected
transport bias -0.018, and therefore the 9.2e3 crossover, are not established:
the six-dimensional histogram ratio is sparse, it uses truth decay modes rather
than the reconstructed ones, and refitting it inside a bootstrap gives an
uncertainty larger than the residual itself.  It also says the two-fold visible
readouts carry fold-to-fold variability far larger than their quoted bootstrap
errors, and that p_cut is fitted and evaluated on the same sample.

Everything below is a check of those statements, not new physics.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
import pol_visible as V

P0 = PT.P0_SAMPLE
S = PD.load_surface()
lab, h, modes, ow = S['labels'], S['h_gen'], S['modes'], S['weights']
z = ~lab
L = PD.load_ladder()
rvis = np.array(L['reco_visible_tau_lab4'], float)
rmode = np.array(L['reco_h_mode'], int)
pt_vis = V.pt(rvis)
eta_vis = np.arcsinh(rvis[:, :, V.PZ] / np.maximum(pt_vis, 1e-9))
T = h[:, 0, 2] + h[:, 1, 2]
out = {}

pt_edges = np.concatenate([[0], np.percentile(pt_vis[z], [10, 25, 40, 55, 70, 85, 95]),
                           [1e9]])
eta_edges = np.array([-3.0, -1.5, -0.7, 0.0, 0.7, 1.5, 3.0])
mode_edges = np.array([-0.5, 0.5, 2.0, 3.5])


def ratio_weights(fH, fZ, bins):
    hH, edges = np.histogramdd(fH, bins=bins)
    hZ, _ = np.histogramdd(fZ, bins=edges)
    ratio = np.where(hH > 0, hZ / np.maximum(hH, 1), 0.0)
    ratio *= hH.sum() / max((ratio * hH).sum(), 1e-12)
    ix = [np.clip(np.digitize(fH[:, d], edges[d][1:-1]), 0, len(edges[d]) - 2)
          for d in range(fH.shape[1])]
    iz = [np.clip(np.digitize(fZ[:, d], edges[d][1:-1]), 0, len(edges[d]) - 2)
          for d in range(fZ.shape[1])]
    unsupported = float((hH[tuple(iz)] == 0).mean())
    return ratio[tuple(ix)], unsupported


def feats(mode_array, rows):
    return np.concatenate([pt_vis[rows], mode_array[rows].astype(float),
                           eta_vis[rows]], -1)


def measure(w_H):
    m = PT.control_moments(h[lab], T[lab], PT.C_H, PT.C_Z, w=w_H)
    return PT.invert_mean(m, float(T[z].mean()))


print('--- (1) truth decay mode vs reconstructed decay mode in the reweighting ---')
binsets = [pt_edges, pt_edges, mode_edges, mode_edges, eta_edges, eta_edges]
res = {}
for tag, ma in (('truth mode', modes), ('reco mode', rmode)):
    w, uns = ratio_weights(feats(ma, lab), feats(ma, z), binsets)
    ph = measure(w)
    res[tag] = {'P_hat': ph, 'bias': ph - P0, 'z_outside_H_support': uns}
    print(f'  {tag:11s} P_hat {ph:+.5f}  bias {ph - P0:+.5f}'
          f'   Z rows in cells H never fills: {uns*100:.2f}%')
out['mode_choice'] = res

print('\n--- (2) bootstrap that refits the histogram ratio in every replica ---')
rng = np.random.default_rng(3)
iH, iZ = np.flatnonzero(lab), np.flatnonzero(z)
boot = {}
for tag, ma in (('truth mode', modes), ('reco mode', rmode)):
    vals = []
    for _ in range(250):
        kH = rng.choice(iH, len(iH))
        kZ = rng.choice(iZ, len(iZ))
        w, _u = ratio_weights(feats(ma, kH), feats(ma, kZ), binsets)
        m = PT.control_moments(h[kH], T[kH], PT.C_H, PT.C_Z, w=w)
        vals.append(PT.invert_mean(m, float(T[kZ].mean())))
    vals = np.array(vals)
    boot[tag] = {'mean': float(vals.mean()), 'sd': float(vals.std()),
                 'p16': float(np.percentile(vals, 16)),
                 'p84': float(np.percentile(vals, 84))}
    print(f'  {tag:11s} mean {vals.mean():+.5f}  sd {vals.std():.5f}'
          f'   -> bias {vals.mean() - P0:+.5f} +- {vals.std():.5f}')
out['reweighted_bootstrap'] = boot

print('\n--- (3) how significant is the uncorrected transport difference? ---')
sig = {}
vals, e0d = [], []
for _ in range(400):
    kH, kZ = rng.choice(iH, len(iH)), rng.choice(iZ, len(iZ))
    m = PT.control_moments(h[kH], T[kH], PT.C_H, PT.C_Z)
    vals.append(PT.invert_mean(m, float(T[kZ].mean())))
    uH = 1 / PT.f_density(h[kH], 0.0, PT.C_H)
    uZ = 1 / PT.f_density(h[kZ], P0, PT.C_Z)
    e0d.append(float(np.sum(uH[:, None] * h[kH][:, :, 2]) / uH.sum()
                     - np.sum(uZ[:, None] * h[kZ][:, :, 2]) / uZ.sum()))
vals, e0d = np.array(vals), np.array(e0d)
sig['P_hat'] = {'mean': float(vals.mean()), 'sd': float(vals.std()),
                'bias_sigma': float(abs(vals.mean() - P0) / vals.std())}
sig['E0_sum_difference_H_minus_Z'] = {
    'mean': float(e0d.mean()), 'sd': float(e0d.std()),
    'sigma': float(abs(e0d.mean()) / e0d.std())}
print(f'  uncorrected bias {vals.mean()-P0:+.5f} +- {vals.std():.5f}'
      f'  ({sig["P_hat"]["bias_sigma"]:.2f} sigma)')
print(f'  E_0[h_k] sum, H - Z: {e0d.mean():+.5f} +- {e0d.std():.5f}'
      f'  ({sig["E0_sum_difference_H_minus_Z"]["sigma"]:.2f} sigma)')
out['significance'] = sig

print('\n--- (4) fold stability of the cross-fitted visible readouts ---')
ups, _ = V.upsilon(np.array(L['reco_h_pions_lab4'], float),
                   np.array(L['reco_h_counts'], int),
                   np.array(L['reco_h_neutral_lab4'], float))
cts = V.cos_theta_star(V.leading_charged(np.array(L['reco_h_pions_lab4'], float),
                                         np.array(L['reco_h_counts'], int)), rvis)
m_vis = V.mass(rvis)
x_coll, coll_ok = V.collinear_fractions(rvis, np.array(L['reco_met_xy'], float))
met = np.array(L['reco_met_xy'], float)
met_mag = np.hypot(met[:, 0], met[:, 1])
vh = rvis[:, :, :2] / np.maximum(pt_vis, 1e-9)[:, :, None]
met_par = np.sum(met[:, None, :] * vh, -1) / np.maximum(pt_vis, 1e-9)
b = [ups, cts, m_vis, np.log(np.maximum(pt_vis, 1.0)), ups ** 2, cts ** 2, ups * cts]
cols = []
for md in (0, 1, 3):
    sel = rmode == md
    cols.append(sel.astype(float))
    cols += [np.where(sel, np.nan_to_num(x), 0.0) for x in b]
X_vis = np.stack([c[z].sum(1) for c in cols], -1)
X_met = np.concatenate([X_vis, np.stack([c[z].sum(1) for c in [
    np.where(coll_ok[:, None], np.nan_to_num(2 * x_coll - 1), 0.0),
    np.repeat(coll_ok[:, None], 2, 1).astype(float), met_par,
    np.log(np.maximum(pt_vis, 1.0)) - np.log(np.maximum(met_mag[:, None], 1.0))]], -1)], 1)
Sz, _ = PT.score(h[z])


def crossfit(X, Sv, folds, seed):
    r = np.random.default_rng(seed)
    parts = np.array_split(r.permutation(len(Sv)), folds)
    Tt = np.empty(len(Sv))
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        Xc = X[tr] - X[tr].mean(0)
        Sc = Sv[tr] - Sv[tr].mean()
        cv = np.cov(Xc, rowvar=False)
        a = np.linalg.solve(cv + 1e-6 * np.trace(cv) / len(cv) * np.eye(len(cv)),
                            (Xc * Sc[:, None]).mean(0))
        Tt[te] = X[te] @ a
    return Tt


fold = {}
for name, X in (('visible only', X_vis), ('visible + MET', X_met)):
    fold[name] = {}
    for k in (2, 5, 10):
        v = [PT.sensitivity(crossfit(X, Sz, k, s), Sz, 100_000)['sigma']
             for s in range(20)]
        fold[name][f'{k}-fold'] = {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                                   'min': float(np.min(v)), 'max': float(np.max(v))}
        print(f'  {name:14s} {k:2d}-fold  {np.mean(v):.5f} +- {np.std(v):.5f}'
              f'   [{np.min(v):.5f}, {np.max(v):.5f}]  (20 partitions)')
out['fold_stability'] = fold

print('\n--- (5) p_cut fitted on H, evaluated on Z ---')
tv = np.array(L['truth_visible_tau_lab4'], float)
tn = np.array(L['truth_neutrino_lab4'], float)
p3 = tv[:, :, :3] + tn[:, :, :3]
pt_tau = np.hypot(p3[:, :, 0], p3[:, :, 1])
fits = {}
for tag, mask, C, P in (('H', lab, PT.C_H, 0.0), ('Z', z, PT.C_Z, P0)):
    u0 = 1 / PT.f_density(h[mask], P, C)
    w = np.where(modes[mask] == 0, np.repeat(u0[:, None], 2, 1), 0.0).ravel()
    hk, x = h[mask][:, :, 2].ravel(), pt_tau[mask].ravel()
    m = w > 0
    inv = 1 / x[m]
    pc = float(np.sum(w[m] * hk[m] * inv) / np.sum(w[m] * inv ** 2))
    bs = []
    for _ in range(200):
        k = rng.integers(0, m.sum(), m.sum())
        ww, hh, ii = w[m][k], hk[m][k], inv[k]
        bs.append(np.sum(ww * hh * ii) / np.sum(ww * ii ** 2))
    fits[tag] = {'p_cut': pc, 'p_cut_sd': float(np.std(bs))}
    print(f'  fitted on {tag}: p_cut = {pc:.2f} +- {np.std(bs):.2f} GeV')
# apply the H fit to the Z rows
u0 = 1 / PT.f_density(h[z], P0, PT.C_Z)
w = np.where(modes[z] == 0, np.repeat(u0[:, None], 2, 1), 0.0).ravel()
hk, x = h[z][:, :, 2].ravel(), pt_tau[z].ravel()
m = w > 0
edges = np.array([0, 30, 40, 50, 60, 70, 85, 100, 120, 150, 200, 1e9])
i = np.clip(np.digitize(x[m], edges[1:-1]), 0, len(edges) - 2)
worst = 0.0
for k in range(len(edges) - 1):
    s = i == k
    if s.sum() < 150:
        continue
    ww = w[m][s]
    mu = np.sum(ww * hk[m][s]) / ww.sum()
    pred = fits['H']['p_cut'] * np.sum(ww / x[m][s]) / ww.sum()
    worst = max(worst, abs(mu - pred))
fits['H_fit_applied_to_Z_worst_residual'] = float(worst)
print(f'  H fit applied to the Z rows: worst binned residual {worst:.3f}')
out['p_cut_holdout'] = fits

PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q8_review.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q8_review.json')
