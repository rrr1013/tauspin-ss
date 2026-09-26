"""q4: where the H-control bias comes from, and how well p_0 must be known.

Three things are separated.

  estimator     split one sample into disjoint halves, take p_0 from one half and
                measure the other with that same sample's own C.  Truth is P = 0
                for H and P = P_0 for Z.  Any residual here is the 1/f ratio
                estimator on finite statistics, not transport.
  transport     H+jet and Z+jet have different tau kinematics, so the visible-pT
                selection shapes the surviving unpolarised measure differently.
                The H control is reweighted to the Z per-side kinematics and P is
                remeasured.
  requirement   how accurately E_0[h_k] has to be known for the bias to stay
                below the statistical error of a given number of Z events.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
import pol_visible as V

P0 = PT.P0_SAMPLE
S_surface = PD.load_surface()
lab, h, modes, ow = (S_surface['labels'], S_surface['h_gen'],
                     S_surface['modes'], S_surface['weights'])
z = ~lab
L = PD.load_ladder()
rvis = np.array(L['reco_visible_tau_lab4'], float)
pt_vis = V.pt(rvis)
eta_vis = np.arcsinh(rvis[:, :, V.PZ] / np.maximum(pt_vis, 1e-9))
T_exact = h[:, 0, 2] + h[:, 1, 2]
out = {'P0': P0}

# ---- 1. is the estimator itself unbiased? -----------------------------------
print('--- estimator closure on disjoint halves of the same sample ---')
rng = np.random.default_rng(11)
est = {}
for tag, mask, C, P_true in (('H -> H  (truth P = 0)', lab, PT.C_H, 0.0),
                             ('Z -> Z  (truth P = P_0)', z, PT.C_Z, P0)):
    idx = np.flatnonzero(mask)
    vals = []
    for _ in range(40):
        p = rng.permutation(idx)
        a, b = p[:len(p) // 2], p[len(p) // 2:]
        m = PT.control_moments(h[a], T_exact[a], C, C, P_ctrl=P_true)
        vals.append(PT.invert_mean(m, float(T_exact[b].mean())))
    vals = np.array(vals)
    est[tag] = {'truth': P_true, 'mean': float(vals.mean()), 'sd': float(vals.std()),
                'bias': float(vals.mean() - P_true)}
    print(f'  {tag:26s} recovered {vals.mean():+.5f} +- {vals.std():.5f}'
          f'   bias {vals.mean() - P_true:+.5f}')
out['estimator_closure'] = est

# ---- 2. transport ------------------------------------------------------------
print('\n--- reweighting the H control to the Z per-side kinematics ---')


def binned_weights(feat_H, feat_Z, bins):
    hH, edges = np.histogramdd(feat_H, bins=bins)
    hZ, _ = np.histogramdd(feat_Z, bins=edges)
    ratio = np.where(hH > 0, hZ / np.maximum(hH, 1), 0.0)
    ratio *= hH.sum() / max((ratio * hH).sum(), 1e-12)
    ix = [np.clip(np.digitize(feat_H[:, d], edges[d][1:-1]), 0, len(edges[d]) - 2)
          for d in range(feat_H.shape[1])]
    return ratio[tuple(ix)]


pt_edges = np.concatenate([[0], np.percentile(pt_vis[z], [10, 25, 40, 55, 70, 85, 95]),
                           [1e9]])
eta_edges = np.array([-3.0, -1.5, -0.7, 0.0, 0.7, 1.5, 3.0])
mode_edges = np.array([-0.5, 0.5, 2.0, 3.5])
feat_all = np.stack([pt_vis[:, 0], pt_vis[:, 1],
                     modes[:, 0].astype(float), modes[:, 1].astype(float)], -1)
variants = {
    'none': None,
    'overlap weights (existing)': ow[lab],
    'per-side pT_vis': binned_weights(
        np.stack([pt_vis[lab, 0], pt_vis[lab, 1]], -1),
        np.stack([pt_vis[z, 0], pt_vis[z, 1]], -1), [pt_edges, pt_edges]),
    'per-side pT_vis + mode': binned_weights(
        feat_all[lab], feat_all[z], [pt_edges, pt_edges, mode_edges, mode_edges]),
    'per-side pT_vis + eta + mode': binned_weights(
        np.concatenate([feat_all[lab], eta_vis[lab]], -1),
        np.concatenate([feat_all[z], eta_vis[z]], -1),
        [pt_edges, pt_edges, mode_edges, mode_edges, eta_edges, eta_edges]),
}
tr = {}
for tag, w in variants.items():
    m = PT.control_moments(h[lab], T_exact[lab], PT.C_H, PT.C_Z, w=w)
    P_hat = PT.invert_mean(m, float(T_exact[z].mean()))
    tr[tag] = {'P_hat': P_hat, 'bias': P_hat - P0, 'ess_fraction': m['ess_fraction']}
    print(f'  {tag:30s} P_hat {P_hat:+.5f}  bias {P_hat - P0:+.5f}'
          f'   ESS {m["ess_fraction"]:.3f}')
out['transport'] = tr

print('\n  p_0 first moment E_0[h_k] per decay mode, H vs Z:')
diag = {}
for md, mname in PD.MODES.items():
    row = {}
    for tag, mask, C, P in (('H', lab, PT.C_H, 0.0), ('Z', z, PT.C_Z, P0)):
        u0 = 1.0 / PT.f_density(h[mask], P, C)
        side = modes[mask] == md
        row[tag] = float(np.sum(u0[:, None] * np.where(side, h[mask][:, :, 2], 0.0))
                         / np.sum(u0[:, None] * side))
    row['H_minus_Z'] = row['H'] - row['Z']
    row['side_fraction_H'] = float((modes[lab] == md).mean())
    row['side_fraction_Z'] = float((modes[z] == md).mean())
    diag[mname] = row
    print(f'    {mname:4s} H {row["H"]:+.4f}  Z {row["Z"]:+.4f}  diff {row["H_minus_Z"]:+.4f}'
          f'   side fraction H {row["side_fraction_H"]:.4f} / Z {row["side_fraction_Z"]:.4f}')
out['p0_diagnostic'] = diag
out['pt_vis_median'] = {'H': float(np.median(pt_vis[lab])), 'Z': float(np.median(pt_vis[z]))}
print('\n  median reco visible pT per side: H %.2f GeV, Z %.2f GeV'
      % (np.median(pt_vis[lab]), np.median(pt_vis[z])))

# ---- 3. how well must E_0[h_k] be known? ------------------------------------
print('\n--- requirement on the unpolarised first moment ---')
m0 = PT.control_moments(h[lab], T_exact[lab], PT.C_H, PT.C_Z)
base = PT.invert_mean(m0, float(T_exact[z].mean()))
slopes = {}
for delta in (0.001, 0.005):
    m1 = dict(m0)
    m1['T'] = m0['T'] + 2 * delta          # both sides shifted by delta
    m1['u'] = m0['u'] + 2 * delta
    slopes[delta] = PT.invert_mean(m1, float(T_exact[z].mean())) - base
    print(f'  shifting E_0[h_k] by {delta:+.4f} on both sides moves P_hat by'
          f' {slopes[delta]:+.5f}')
dP_dE0 = slopes[0.001] / 0.001
out['dP_hat_dE0_hk'] = float(dP_dE0)
try:
    q1 = json.loads((PD.RESULTS / 'q1_sensitivity.json').read_text())
    sig = q1['rows']['full22_s42_hk']['sigma']
    for N in (1e5, 1e6, 1e7, 1e8):
        s_N = sig * np.sqrt(1e5 / N)
        print(f'  N = {N:.0e} Z events: sigma(P) = {s_N:.5f}'
              f'  -> E_0[h_k] must be known to {abs(s_N / dP_dE0):.2e}')
    out['requirement'] = {'sigma_per_1e5': sig, 'dP_dE0': float(dP_dE0),
                          'needed_E0_accuracy': {f'{N:.0e}': float(abs(sig * np.sqrt(1e5 / N) / dP_dE0))
                                                 for N in (1e5, 1e6, 1e7, 1e8)}}
    bias_c = abs(tr['per-side pT_vis + eta + mode']['bias'])
    bias_u = abs(tr['none']['bias'])
    out['crossover_events'] = {'uncorrected': 1e5 * (sig / bias_u) ** 2,
                               'kinematically_reweighted': 1e5 * (sig / bias_c) ** 2}
    print(f'  bias = statistical error at {out["crossover_events"]["uncorrected"]:.3g}'
          f' Z events (uncorrected) / {out["crossover_events"]["kinematically_reweighted"]:.3g}'
          f' after the kinematic reweighting')
except FileNotFoundError:
    pass

PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q4_transport.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q4_transport.json')
