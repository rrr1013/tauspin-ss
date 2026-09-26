"""q7: is the H-control bias really the p_0 first moment, or is P_0 wrong?

Two explanations survive q4: the unpolarised measure does not transport from
H+jet to Z+jet, or the sample's true polarisation is not the SM -0.147037.  They
are degenerate in the H -> Z measurement itself, so they are separated here.

  1. substitution.  Replace only E_0[h_k] in the H-derived moments by the value
     measured on the Z rows and see how much of the bias that removes.  If the
     transport of one number explains it, the residual should be small.
  2. mode ordering.  The bias per decay mode should follow the per-mode H - Z
     difference in E_0[h_k], not the mode's own analysing power.
  3. what would have to be true otherwise.  Convert the bias into the shift of
     sin^2 theta_W that the generator would have had to use.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT
from pol_density import dp_dsin2w

P0 = PT.P0_SAMPLE
S = PD.load_surface()
lab, h, modes = S['labels'], S['h_gen'], S['modes']
z = ~lab
T = h[:, 0, 2] + h[:, 1, 2]
out = {'P0': P0}


def p0_first_moment(mask, C, P):
    u0 = 1.0 / PT.f_density(h[mask], P, C)
    U = u0.sum()
    return np.array([float(np.sum(u0 * h[mask][:, s, 2]) / U) for s in (0, 1)])


mH = PT.control_moments(h[lab], T[lab], PT.C_H, PT.C_Z)
base = PT.invert_mean(mH, float(T[z].mean()))
e0H, e0Z = p0_first_moment(lab, PT.C_H, 0.0), p0_first_moment(z, PT.C_Z, P0)
shift = float((e0Z - e0H).sum())
print(f'E_0[h_k] sum over sides: H {e0H.sum():+.5f}  Z {e0Z.sum():+.5f}'
      f'  difference {shift:+.5f}')

m2 = dict(mH)
m2['T'] = mH['T'] + shift
m2['u'] = mH['u'] + shift
sub = PT.invert_mean(m2, float(T[z].mean()))
out['substitution'] = {'P_hat_H_control': base, 'P_hat_after_substitution': sub,
                       'E0_shift': shift, 'bias_before': base - P0,
                       'bias_after': sub - P0,
                       'fraction_explained': 1 - abs(sub - P0) / abs(base - P0)}
print(f'P_hat  {base:+.5f} -> {sub:+.5f} after substituting E_0[h_k] alone'
      f'   (bias {base-P0:+.5f} -> {sub-P0:+.5f},'
      f' {out["substitution"]["fraction_explained"]*100:.0f}% explained)')

print('\nper decay mode: bias against the H - Z difference in E_0[h_k]')
rows = {}
for md, mname in PD.MODES.items():
    Tm = np.sum(np.where(modes == md, h[:, :, 2], 0.0), 1)
    m = PT.control_moments(h[lab], Tm[lab], PT.C_H, PT.C_Z)
    ph = PT.invert_mean(m, float(Tm[z].mean()))
    d = {}
    for tag, mask, C, P in (('H', lab, PT.C_H, 0.0), ('Z', z, PT.C_Z, P0)):
        u0 = 1.0 / PT.f_density(h[mask], P, C)
        side = modes[mask] == md
        d[tag] = float(np.sum(u0[:, None] * np.where(side, h[mask][:, :, 2], 0.0))
                       / np.sum(u0[:, None] * side))
    diff = d['Z'] - d['H']
    rows[mname] = {'P_hat': ph, 'bias': ph - P0, 'E0_H': d['H'], 'E0_Z': d['Z'],
                   'E0_diff': diff, 'bias_over_diff': (ph - P0) / diff if diff else None}
    print(f'  {mname:4s} bias {ph-P0:+.5f}   E_0[h_k] H {d["H"]:+.4f} Z {d["Z"]:+.4f}'
          f'  diff {diff:+.5f}   ratio {(ph-P0)/diff:+.2f}')
out['per_mode'] = rows
r = [v['bias_over_diff'] for v in rows.values()]
print(f'  the three ratios agree to {max(r)/min(r):.2f}; the global dP/dE_0 is'
      f' {json.loads((PD.RESULTS/"q4_transport.json").read_text())["dP_hat_dE0_hk"]:+.2f}'
      ' per unit and per side')

dP = dp_dsin2w()
out['alternative_explanation'] = {
    'required_delta_sin2w': float((base - P0) / dP),
    'required_sin2w': float(0.23152 + (base - P0) / dP)}
print(f'\nif instead the generator had a different weak mixing angle, it would have'
      f' to be sin^2 theta_W = {out["alternative_explanation"]["required_sin2w"]:.5f}'
      f' (SM input 0.23152), a shift of'
      f' {out["alternative_explanation"]["required_delta_sin2w"]:+.5f}.')
print('That is excluded: the rho mode, whose E_0[h_k] is nearly isotropic, returns'
      f' {rows["rho"]["P_hat"]:+.4f} on its own, and a wrong P_0 would bias every'
      ' mode equally.')

(PD.RESULTS / 'q7_bias_origin.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q7_bias_origin.json')
