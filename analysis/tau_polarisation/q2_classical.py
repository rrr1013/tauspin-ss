"""q2: what non-network observables achieve on the same Z events.

Everything is priced by the same linear response as q1, so no method is refitted
against a different sample and the comparison is event-by-event identical.
Only the *shape* of each observable is used (the covariance is mean-subtracted),
so the P-dependence of the total selected rate is deliberately not counted.

tiers
  x_truth        E_vis / E_tau with the truth tau energy.  The LEP observable:
                 for tau -> pi nu, dGamma/dx = 1 + P(2x-1), analysing power 1.
                 It needs E_tau, so it is a reference, not a hadron-collider method.
  visible only   Upsilon, cos(theta*) of the leading charged pion in the visible
                 rest frame, m_vis, log pT_vis -- with quadratic terms, and a
                 separate readout per *reco* decay mode.  Cross-fitted on halves.
  visible + MET  the same plus the collinear-approximation energy fraction and
                 two MET projections.
  network        h_pred_k of the point-h regression arms.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
import pol_visible as V

N_REF = 100_000
BOOT = 300
FOLDS = 2

S_surface = PD.load_surface()
lab = S_surface['labels']
z = ~lab
h = S_surface['h_gen']
modes = S_surface['modes']
L = PD.load_ladder()
assert np.array_equal(L['global_indices'], S_surface['global_indices'])
assert np.array_equal(L['modes'], modes)

tvis = np.array(L['truth_visible_tau_lab4'], float)
tnu = np.array(L['truth_neutrino_lab4'], float)
rvis = np.array(L['reco_visible_tau_lab4'], float)
rpi = np.array(L['reco_h_pions_lab4'], float)
rcnt = np.array(L['reco_h_counts'], int)
rn = np.array(L['reco_h_neutral_lab4'], float)
rmode = np.array(L['reco_h_mode'], int)
met = np.array(L['reco_met_xy'], float)

x_truth = tvis[:, :, V.E] / (tvis[:, :, V.E] + tnu[:, :, V.E])
ups, _ = V.upsilon(rpi, rcnt, rn)
cts = V.cos_theta_star(V.leading_charged(rpi, rcnt), rvis)
m_vis = V.mass(rvis)
pt_vis = V.pt(rvis)
x_coll, coll_ok = V.collinear_fractions(rvis, met)
met_mag = np.hypot(met[:, 0], met[:, 1])
vhat = rvis[:, :, :2] / np.maximum(pt_vis, 1e-9)[:, :, None]
met_par = np.sum(met[:, None, :] * vhat, -1) / np.maximum(pt_vis, 1e-9)

S, f = PT.score(h[z])
mz = modes[z]
out = {'n_ref': N_REF, 'collinear_valid_fraction_Z': float(coll_ok[z].mean()),
       'score_mean': float(S.mean()),
       'note_score_mean': 'non-zero because the selected rate itself depends on '
                          'P; only mean-subtracted covariances are used below'}


def vis_features(side_mask):
    """(n_z, n_feat) event-summed visible-only features, restricted to sides."""
    base = [ups, cts, m_vis, np.log(np.maximum(pt_vis, 1.0)),
            ups ** 2, cts ** 2, ups * cts]
    cols = []
    for md in (0, 1, 3):                      # separate readout per *reco* mode
        sel = side_mask & (rmode == md)
        cols.append(sel.astype(float))
        for b in base:
            cols.append(np.where(sel, np.nan_to_num(b), 0.0))
    return np.stack([c[z].sum(1) for c in cols], -1)


def met_features(side_mask):
    xc = np.where(coll_ok[:, None] & side_mask, np.nan_to_num(2 * x_coll - 1), 0.0)
    cols = [xc, (coll_ok[:, None] & side_mask).astype(float),
            np.where(side_mask, met_par, 0.0),
            np.where(side_mask, np.log(np.maximum(pt_vis, 1.0))
                     - np.log(np.maximum(met_mag[:, None], 1.0)), 0.0)]
    return np.stack([c[z].sum(1) for c in cols], -1)


def crossfit(X, S, folds=FOLDS, seed=0):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(S)), folds)
    T = np.empty(len(S))
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        Xc = X[tr] - X[tr].mean(0)
        Sc = S[tr] - S[tr].mean()
        cov = np.cov(Xc, rowvar=False)
        a = np.linalg.solve(cov + 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov)),
                            (Xc * Sc[:, None]).mean(0))
        T[te] = X[te] @ a
    return T


rows = {}


def price(key, T, label, sub=None):
    t = np.asarray(T, float)
    ok = np.isfinite(t) if sub is None else (np.isfinite(t) & sub)
    r = PT.sensitivity(t[ok], S[ok], N_REF, boot=BOOT)
    r.update({'label': label, 'n_used': int(ok.sum())})
    rows[key] = r
    print(f'{label:56s} n {ok.sum():6d}  sigma(P) {r["sigma"]:.5f}'
          f' +- {r.get("sigma_se",0):.5f}   response {r["response"]:+.5f}')
    return r


def summed(v, side_mask):
    return np.nansum(np.where(side_mask, v, 0.0), axis=1)[z]


arms = {'base_s43': 'network h_pred_k (no geometry)',
        'full22_s42': 'network h_pred_k + 3 IP + SV'}
hp = {a_: PD.load_arm(a_, S_surface)['h_pred'][z] for a_ in arms}

for scope, side_mask, tag in [('all', np.ones_like(rmode, bool), 'all modes')] + \
        [(nm, modes == md, f'truth mode {nm}') for md, nm in PD.MODES.items()]:
    print(f'\n--- {tag} ---')
    pre = '' if scope == 'all' else scope + '_'
    price(pre + 'exact_h', summed(h[:, :, 2], side_mask), 'exact h, h_k sum')
    price(pre + 'x_truth', summed(2 * x_truth - 1, side_mask),
          'x_truth (needs E_tau; LEP reference)')
    price(pre + 'upsilon', summed(ups, side_mask), 'Upsilon alone (visible only)')
    price(pre + 'costheta', summed(cts, side_mask),
          'cos(theta*) alone (visible only)')
    Xv = vis_features(side_mask)
    price(pre + 'vis_best', crossfit(Xv, S), 'best visible-only readout (cross-fit)')
    Xm = np.concatenate([Xv, met_features(side_mask)], 1)
    price(pre + 'vismet_best', crossfit(Xm, S), 'best visible + MET readout (cross-fit)')
    for a_, d in arms.items():
        price(pre + a_, np.nansum(np.where(side_mask[z], hp[a_][:, :, 2], 0.0), 1), d)

out['rows'] = rows
PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q2_classical.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q2_classical.json')
