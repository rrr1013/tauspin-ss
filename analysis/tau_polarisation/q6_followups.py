"""q6: two checks the main result invites.

(a) is h_pred a sufficient statistic for P_tau?  Cross-fit a linear readout of
    h_pred alone, of the visible+MET features alone, and of both together.  If
    the union beats h_pred, the regression is discarding polarisation information
    that the raw reco event still has.

(b) is the +0.049 first moment of p_0 really the visible-pT selection?  The cut
    can only bite when the tau itself is soft, so E_0[h_k] must fall towards zero
    at high truth tau pT.  Measured in bins of the truth tau pT (visible + its
    neutrino), which is not a function of h_k by construction.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
import pol_visible as V

N_REF, BOOT = 100_000, 300
S_surface = PD.load_surface()
lab, h, modes = S_surface['labels'], S_surface['h_gen'], S_surface['modes']
z = ~lab
L = PD.load_ladder()
tvis = np.array(L['truth_visible_tau_lab4'], float)
tnu = np.array(L['truth_neutrino_lab4'], float)
rvis = np.array(L['reco_visible_tau_lab4'], float)
rpi = np.array(L['reco_h_pions_lab4'], float)
rcnt = np.array(L['reco_h_counts'], int)
rn = np.array(L['reco_h_neutral_lab4'], float)
rmode = np.array(L['reco_h_mode'], int)
met = np.array(L['reco_met_xy'], float)

S, _ = PT.score(h[z])
out = {'n_ref': N_REF}


def crossfit(X, S, folds=10, seed=0):
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


# ---- (a) sufficiency ---------------------------------------------------------
ups, _ = V.upsilon(rpi, rcnt, rn)
cts = V.cos_theta_star(V.leading_charged(rpi, rcnt), rvis)
m_vis, pt_vis = V.mass(rvis), V.pt(rvis)
x_coll, coll_ok = V.collinear_fractions(rvis, met)
met_mag = np.hypot(met[:, 0], met[:, 1])
vhat = rvis[:, :, :2] / np.maximum(pt_vis, 1e-9)[:, :, None]
met_par = np.sum(met[:, None, :] * vhat, -1) / np.maximum(pt_vis, 1e-9)

base = [ups, cts, m_vis, np.log(np.maximum(pt_vis, 1.0)), ups ** 2, cts ** 2, ups * cts]
cols = []
for md in (0, 1, 3):
    sel = rmode == md
    cols.append(sel.astype(float))
    for b in base:
        cols.append(np.where(sel, np.nan_to_num(b), 0.0))
cols += [np.where(coll_ok[:, None], np.nan_to_num(2 * x_coll - 1), 0.0),
         np.repeat(coll_ok[:, None], 2, 1).astype(float), met_par,
         np.log(np.maximum(pt_vis, 1.0)) - np.log(np.maximum(met_mag[:, None], 1.0))]
X_vis = np.stack([c[z].sum(1) for c in cols], -1)

print('--- (a) is h_pred a sufficient statistic for P_tau? ---')
suff = {}
for arm, desc in (('base_s43', 'reco, no geometry'), ('full22_s42', 'reco + 3 IP + SV')):
    hp = PD.load_arm(arm, S_surface)['h_pred'][z]
    X_h = hp.reshape(len(hp), 6)
    rows = {
        'h_pred only': PT.sensitivity(crossfit(X_h, S), S, N_REF, boot=BOOT),
        'visible + MET only': PT.sensitivity(crossfit(X_vis, S), S, N_REF, boot=BOOT),
        'both': PT.sensitivity(crossfit(np.concatenate([X_h, X_vis], 1), S), S,
                               N_REF, boot=BOOT)}
    suff[arm] = {'label': desc, **{k: v for k, v in rows.items()}}
    g = rows['h_pred only']['sigma'] / rows['both']['sigma']
    suff[arm]['gain_from_adding_visible'] = float(g)
    print(f'  {desc:22s} h_pred {rows["h_pred only"]["sigma"]:.5f}'
          f'   vis+MET {rows["visible + MET only"]["sigma"]:.5f}'
          f'   both {rows["both"]["sigma"]:.5f}   gain {g:.3f}')
out['sufficiency'] = suff

# ---- (b) the mechanism -------------------------------------------------------
print('\n--- (b) E_0[h_k] against the truth tau pT ---')
p_tau = tvis[:, :, :3] + tnu[:, :, :3]
pt_tau = np.hypot(p_tau[:, :, 0], p_tau[:, :, 1])
edges = np.array([0, 30, 40, 50, 60, 70, 85, 100, 120, 150, 200, 1e9])
mech = {}
for tag, mask, C, P in (('H', lab, PT.C_H, 0.0), ('Z', z, PT.C_Z, PT.P0_SAMPLE)):
    u0 = 1.0 / PT.f_density(h[mask], P, C)
    w = np.repeat(u0[:, None], 2, 1).ravel()
    hk = h[mask][:, :, 2].ravel()
    ptt = pt_tau[mask].ravel()
    i = np.clip(np.digitize(ptt, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = i == k
        if m.sum() < 200:
            rows.append(None)
            continue
        ww = w[m]
        mu = float(np.sum(ww * hk[m]) / ww.sum())
        se = float(np.sqrt(np.sum(ww ** 2 * (hk[m] - mu) ** 2)) / ww.sum())
        rows.append({'lo': float(edges[k]), 'hi': float(edges[k + 1]),
                     'n_sides': int(m.sum()), 'E0_hk': mu, 'E0_hk_se': se,
                     'min_vis_pt_that_can_fail': None})
    mech[tag] = rows
    print(f'  {tag}: ' + '  '.join(
        f'[{r["lo"]:.0f},{r["hi"]:.0f}) {r["E0_hk"]:+.3f}' for r in rows if r))
out['mechanism_vs_truth_tau_pt'] = {'edges': edges.tolist(), 'rows': mech}

# the same restricted to pi sides, where the effect is largest
print('\n  pi sides only:')
pi_rows = {}
for tag, mask, C, P in (('H', lab, PT.C_H, 0.0), ('Z', z, PT.C_Z, PT.P0_SAMPLE)):
    u0 = 1.0 / PT.f_density(h[mask], P, C)
    w = np.where(modes[mask] == 0, np.repeat(u0[:, None], 2, 1), 0.0).ravel()
    hk = h[mask][:, :, 2].ravel()
    ptt = pt_tau[mask].ravel()
    i = np.clip(np.digitize(ptt, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = (i == k) & (w > 0)
        if m.sum() < 150:
            rows.append(None)
            continue
        ww = w[m]
        mu = float(np.sum(ww * hk[m]) / ww.sum())
        se = float(np.sqrt(np.sum(ww ** 2 * (hk[m] - mu) ** 2)) / ww.sum())
        rows.append({'lo': float(edges[k]), 'hi': float(edges[k + 1]),
                     'n_sides': int(m.sum()), 'E0_hk': mu, 'E0_hk_se': se})
    pi_rows[tag] = rows
    print(f'  {tag}: ' + '  '.join(
        f'[{r["lo"]:.0f},{r["hi"]:.0f}) {r["E0_hk"]:+.3f}±{r["E0_hk_se"]:.3f}'
        for r in rows if r))
out['mechanism_pi_only'] = pi_rows

# the selection threshold actually in the ntuple
print('\n  reco visible pT: min %.2f GeV, 1st percentile %.2f GeV'
      % (pt_vis.min(), np.percentile(pt_vis, 1)))
out['reco_vis_pt_min'] = float(pt_vis.min())

PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q6_followups.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q6_followups.json')


# ---- (c) a one-parameter model of the mechanism ------------------------------
# For tau -> pi nu the visible pT is x pT_tau with x = (1 + h_k)/2 uniform under
# p_0, so a cut at p_cut removes h_k < 2 p_cut/pT_tau - 1 and leaves
#     E_0[h_k] = p_cut / pT_tau .
print('\n--- (c) one-parameter acceptance model for the pi mode ---')
u0 = 1.0 / PT.f_density(h[lab], 0.0, PT.C_H)
w = np.where(modes[lab] == 0, np.repeat(u0[:, None], 2, 1), 0.0).ravel()
hk_f = h[lab][:, :, 2].ravel()
ptt_f = pt_tau[lab].ravel()
m = w > 0
inv = 1.0 / ptt_f[m]
p_cut = float(np.sum(w[m] * hk_f[m] * inv) / np.sum(w[m] * inv ** 2))
i = np.clip(np.digitize(ptt_f[m], edges[1:-1]), 0, len(edges) - 2)
model = []
for k in range(len(edges) - 1):
    s = i == k
    if s.sum() < 150:
        continue
    ww = w[m][s]
    mu = float(np.sum(ww * hk_f[m][s]) / ww.sum())
    pred = float(p_cut * np.sum(ww / ptt_f[m][s]) / ww.sum())
    model.append({'pt_tau': float(np.sum(ww * ptt_f[m][s]) / ww.sum()),
                  'measured': mu, 'model': pred})
resid = max(abs(r['measured'] - r['model']) for r in model)
out['acceptance_model'] = {'p_cut_fitted_GeV': p_cut, 'rows': model,
                           'max_residual': resid,
                           'reco_vis_pt_1st_percentile': float(np.percentile(pt_vis, 1))}
print(f'  fitted p_cut = {p_cut:.2f} GeV (ntuple reco visible pT 1st percentile'
      f' {np.percentile(pt_vis, 1):.2f} GeV); worst residual {resid:.3f}')
(PD.RESULTS / 'q6_followups.json').write_text(json.dumps(out, indent=1))
print('rewrote', PD.RESULTS / 'q6_followups.json')
