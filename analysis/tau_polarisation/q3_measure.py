"""q3: measure P_tau on the Z rows with the H rows as the control region.

H -> tau tau has B = 0 *exactly*, so the H rows give the unpolarised (post-
selection) measure p_0 with no polarisation assumption: for any quantity g,

    E_0[g] = E[g / f_H] / E[1 / f_H] ,     f_H = 1 + h^- C_H h^+ .

With those joint moments (no side-independence assumed anywhere) the Z mean of
any observable T is predicted exactly as a function of P,

    E_Z[T](P) = (E_0[T] + P E_0[T u] + E_0[T v]) / (1 + P E_0[u] + E_0[v]) ,
    u = h^-_k + h^+_k ,   v = h^- C_Z h^+ ,

and inverting against the observed Z mean gives P_hat.  The only assumption is
that p_0 transports from H+jet to Z+jet events, so the bias of P_hat *is* that
systematic.  Using the Z rows' own p_0 instead returns P_0 algebraically and is
shown only as a reference.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT

BOOT = 400
P0 = PT.P0_SAMPLE

S_surface = PD.load_surface()
lab, h, modes = S_surface['labels'], S_surface['h_gen'], S_surface['weights'] * 0
modes = S_surface['modes']
ow = S_surface['weights']
z = ~lab


def joint_moments(hh, C_control, T, w=None, P_ctrl=0.0, C_target=PT.C_Z):
    return PT.control_moments(hh, T, C_control, C_target, P_ctrl=P_ctrl, w=w)


predict = PT.predict_mean
invert = PT.invert_mean


def measure(T_all, control_mask, C_control, w=None, boot=BOOT, seed=0):
    m = joint_moments(h[control_mask], C_control, T_all[control_mask],
                      None if w is None else w[control_mask])
    obs = float(np.average(T_all[z], weights=None if w is None else w[z]))
    P_hat = invert(m, obs)
    res = {'P_hat': P_hat, 'bias': P_hat - P0, 'obs_Z': obs,
           'pred_Z_at_P0': predict(m, P0),
           'ess_fraction': m['ess_fraction'],
           'max_weight_fraction': m['max_weight_fraction']}
    if boot:
        rng = np.random.default_rng(seed)
        ic = np.flatnonzero(control_mask)
        iz = np.flatnonzero(z)
        vals = []
        for _ in range(boot):
            kc = rng.choice(ic, len(ic))
            kz = rng.choice(iz, len(iz))
            mm = joint_moments(h[kc], C_control, T_all[kc],
                               None if w is None else w[kc])
            ob = float(np.average(T_all[kz], weights=None if w is None else w[kz]))
            vals.append(invert(mm, ob))
        res['P_hat_sd'] = float(np.nanstd(vals))
        res['P_hat_sd_Z_only'] = None
    return res


out = {'P0': P0}
T_exact = h[:, 0, 2] + h[:, 1, 2]

print('--- exact h, all modes ---')
for tag, cm, C, w in (('H control (unit weight)', lab, PT.C_H, None),
                      ('H control (overlap weights)', lab, PT.C_H, ow),
                      ('Z control = circular reference', z, PT.C_Z, None)):
    if tag.startswith('Z control'):
        m = joint_moments(h[z], PT.C_Z, T_exact[z], P_ctrl=P0)
        r = {'P_hat': invert(m, float(T_exact[z].mean()))}
        r['bias'] = r['P_hat'] - P0
    else:
        r = measure(T_exact, cm, C, w)
    out[f'exact_{tag}'] = r
    print(f'{tag:32s} P_hat {r["P_hat"]:+.5f}  bias {r["bias"]:+.5f}'
          + (f'  (bootstrap sd {r["P_hat_sd"]:.5f}, ESS {r["ess_fraction"]:.3f},'
             f' max w {r["max_weight_fraction"]:.2e})' if 'P_hat_sd' in r else ''))

# ---- injection closure: reweight Z to a different P, remeasure ---------------
print('\n--- injection closure (p_0 still from the H rows, unit weight) ---')
inj = {}
m_H = joint_moments(h[lab], PT.C_H, T_exact[lab])
for P_inj in (-0.30, -0.20, -0.147037, -0.05, 0.0, +0.15):
    w_inj = PT.reweight(h[z], P_inj)
    obs = float(np.average(T_exact[z], weights=w_inj))
    P_hat = invert(m_H, obs)
    inj[f'{P_inj:+.6f}'] = {'P_hat': P_hat, 'bias': P_hat - P_inj}
    print(f'  injected {P_inj:+.5f} -> recovered {P_hat:+.5f}  bias {P_hat - P_inj:+.5f}')
out['injection'] = inj

# ---- per decay mode ---------------------------------------------------------
print('\n--- exact h, by truth decay mode of the side ---')
per_mode = {}
for md, mname in PD.MODES.items():
    sel = modes == md
    T = np.sum(np.where(sel, h[:, :, 2], 0.0), 1)
    r = measure(T, lab, PT.C_H)
    per_mode[mname] = r
    print(f'  {mname:4s} P_hat {r["P_hat"]:+.5f}  bias {r["bias"]:+.5f}'
          f'  (bootstrap sd {r["P_hat_sd"]:.5f})')
out['per_mode_exact'] = per_mode

# ---- the same with the network output ---------------------------------------
print('\n--- network h_pred_k sum, H control ---')
net = {}
for arm, desc in (('base_s43', 'reco, no geometry'),
                  ('full22_s42', 'reco + 3 IP + SV')):
    hp = PD.load_arm(arm, S_surface)['h_pred']
    T = hp[:, 0, 2] + hp[:, 1, 2]
    r = measure(T, lab, PT.C_H)
    net[arm] = dict(r, label=desc)
    print(f'  {desc:22s} P_hat {r["P_hat"]:+.5f}  bias {r["bias"]:+.5f}'
          f'  (bootstrap sd {r["P_hat_sd"]:.5f})')
out['network'] = net

PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q3_measure.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q3_measure.json')
