"""q0: what the raw first moment of h is made of, and does the density model close?

The H rows have B = 0 exactly, so any non-zero <h_k> there is entirely the
post-selection anisotropy of the unpolarised measure p_0 plus the C_kk = -1
correlation term.  The Z rows add the physics we want.  Everything here uses the
generator-current exact h; no network output enters.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
from pol_density import higgs_state, z_state, p_tau_of_sin2w, dp_dsin2w

out = {}
S = PD.load_surface()
h, lab, modes, w = S['h_gen'], S['labels'], S['modes'], S['weights']
print(f'rows {len(lab)}  H {lab.sum()}  Z {(~lab).sum()}')

BmH, BpH, CH = higgs_state()
BmZ, BpZ, CZ = z_state()
out['analytic'] = {'B_H_physical': BmH.tolist(), 'C_H': CH.tolist(),
                   'B_Z_physical': BmZ.tolist(),
                   'B_Z_canonical': (-BmZ).tolist(), 'C_Z': CZ.tolist(),
                   'P_tau': p_tau_of_sin2w(0.23152), 'dP_dsin2w': dp_dsin2w()}
print('analytic  B_H_phys =', np.round(BmH, 9), ' B_Z_phys =', np.round(BmZ, 5),
      ' P_tau =', round(out['analytic']['P_tau'], 6))
assert np.abs(BmH).max() < 1e-9 and np.abs(BpH).max() < 1e-9
assert np.allclose(CH, PT.C_H, atol=2e-5) and np.allclose(CZ, PT.C_Z, atol=2e-5)

# ---- raw first moments -------------------------------------------------------
raw = {}
for name, m in (('H', lab), ('Z', ~lab)):
    raw[name] = {'n': int(m.sum()),
                 'h_minus': h[m, 0].mean(0).tolist(),
                 'h_plus': h[m, 1].mean(0).tolist(),
                 'hk_sum': float((h[m, 0, 2] + h[m, 1, 2]).mean()),
                 'hk_sum_se': float((h[m, 0, 2] + h[m, 1, 2]).std(ddof=1) / np.sqrt(m.sum()))}
    print(f'{name}  <h-> {np.round(h[m,0].mean(0),4)}  <h+> {np.round(h[m,1].mean(0),4)}'
          f'  <h-_k+h+_k> {raw[name]["hk_sum"]:+.4f} +- {raw[name]["hk_sum_se"]:.4f}')
out['raw_first_moment'] = raw
out['raw_Z_minus_H_hk_sum'] = raw['Z']['hk_sum'] - raw['H']['hk_sum']
print(f'Z - H in <h-_k+h+_k>: {out["raw_Z_minus_H_hk_sum"]:+.4f}'
      f'   (P_tau alone would give {2*PT.P0_SAMPLE/3:+.4f} with an isotropic p0)')

# ---- p0 moments from each sample --------------------------------------------
p0 = {}
for name, m, P, C in (('H', lab, 0.0, PT.C_H), ('Z', ~lab, PT.P0_SAMPLE, PT.C_Z)):
    mm, MM = PT.p0_moments(h[m], P, C)
    p0[name] = {'m': mm.tolist(), 'M_diag': [np.diag(MM[0]).tolist(), np.diag(MM[1]).tolist()]}
    print(f'p0 from {name}: m- {np.round(mm[0],4)} m+ {np.round(mm[1],4)}'
          f'  diag M- {np.round(np.diag(MM[0]),4)} diag M+ {np.round(np.diag(MM[1]),4)}')
out['p0_moments'] = p0

# per-mode p0 first moment along k (the quantity that fakes a polarisation)
per_mode = {}
for md, mname in PD.MODES.items():
    row = {}
    for name, m, P, C in (('H', lab, 0.0, PT.C_H), ('Z', ~lab, PT.P0_SAMPLE, PT.C_Z)):
        sel = m & ((modes[:, 0] == md) | (modes[:, 1] == md))
        if sel.sum() < 200:
            continue
        f = PT.f_density(h[sel], P, C)
        u = 1.0 / f
        side = modes[sel] == md
        hk = np.where(side, h[sel, :, 2], np.nan)
        num = np.nansum(u[:, None] * np.where(side, h[sel, :, 2], 0.0))
        den = np.sum(u[:, None] * side)
        num2 = np.nansum(u[:, None] * np.where(side, h[sel, :, 2] ** 2, 0.0))
        row[name] = {'n_sides': int(side.sum()), 'E0_hk': float(num / den),
                     'E0_hk2': float(num2 / den),
                     'raw_hk': float(np.nanmean(hk))}
    per_mode[mname] = row
    if 'H' in row and 'Z' in row:
        print(f'  mode {mname:4s}: E0[h_k] H {row["H"]["E0_hk"]:+.4f} / Z {row["Z"]["E0_hk"]:+.4f}'
              f'   E0[h_k^2] H {row["H"]["E0_hk2"]:.4f} / Z {row["Z"]["E0_hk2"]:.4f}')
out['per_mode_p0'] = per_mode

# ---- closure of the product-p0 model ----------------------------------------
clo = {}
for name, m, P, C in (('H', lab, 0.0, PT.C_H), ('Z', ~lab, PT.P0_SAMPLE, PT.C_Z)):
    mm, MM = PT.p0_moments(h[m], P, C)
    pred = PT.predict_first_moment(P, C, mm, MM)
    obs = np.array([h[m, 0, 2].mean(), h[m, 1, 2].mean()])
    clo[name] = {'pred': pred.tolist(), 'obs': obs.tolist(),
                 'residual': (obs - pred).tolist()}
    print(f'closure {name}: model {np.round(pred,5)} vs observed {np.round(obs,5)}'
          f'  residual {np.round(obs-pred,5)}')
out['self_closure'] = clo
PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q0_structure.json').write_text(json.dumps(out, indent=1))
print('wrote', PD.RESULTS / 'q0_structure.json')
