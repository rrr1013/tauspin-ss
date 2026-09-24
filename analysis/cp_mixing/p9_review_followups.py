"""Checks prompted by the 2026-09-25 independent validity review.

1. Is the Fisher information for phi_tau finite?  Claimed E[S^2] = 4/3 for
   unit isotropic polarimeters with no selection, with E[S^4] log-divergent.
2. Does the classical rho method change if y is taken in the tau rest frame
   (the ideal definition) rather than in the lab (the realisable one)?
3. Does the geometry comparison survive a same-seed base vs IP+SV pair?
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import bilinear, score, sensitivity, triple  # noqa: E402
from p3_classical import boost_to, phistar_cp, unit  # noqa: E402
from p6_geometry_value import auc, llr, ppf  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402

import math

out = {}
C0 = c_matrix(0.0)

# ---- 1. Fisher information for isotropic unit polarimeters, no selection ----
rng = np.random.default_rng(11)
rows = []
for n in (10**5, 10**6, 10**7):
    got = []
    need = n
    while need > 0:
        m = min(need, 2_000_000)
        a = rng.normal(size=(4 * m, 3)); a /= np.linalg.norm(a, axis=-1)[:, None]
        b = rng.normal(size=(4 * m, 3)); b /= np.linalg.norm(b, axis=-1)[:, None]
        f = 1 + np.einsum('ni,ij,nj->n', a, C0, b)
        keep = rng.random(4 * m) < f / 2
        h = np.stack([a[keep], b[keep]], 1)[:m]
        got.append(h)
        need -= len(h)
    h = np.concatenate(got)
    S, f = score(h, C0)
    rows.append({'n': int(len(h)), 'E_S2': float(np.mean(S**2)), 'E_S4': float(np.mean(S**4)),
                 'f_min': float(f.min())})
out['fisher_isotropic'] = {'rows': rows, 'analytic_E_S2': 4 / 3,
                           'sigma_deg_10k_from_4_3': float(np.rad2deg(1 / np.sqrt(1e4 * 4 / 3)))}

# ---- 2. classical rho method with the rest-frame y ----
D = np.load(HERE / 'data' / 'truth_surface_cleo.npz')
L = np.load(HERE / 'data' / 'ladder_validation.npz')
T = np.load(HERE / 'data' / 'ip_tracks.npz')
y = D['labels'].astype(bool)
tmode = D['modes']
fr = frames(D['pions'], D['pi0'], D['nu4'])
h_gen = np.array(D['h_ref'], float)
for md in (0, 1, 3):
    for s in (0, 1):
        sel, hh = canonical_h(fr, tmode, D['pion_charges'], s, md)
        h_gen[sel, s] = hh
S_all, _ = score(h_gen, C0)

pions_t = D['pions'][:, :, 0]
pi0_t = D['pi0']
tau4 = D['pions'].sum(2) + D['pi0'] + D['nu4']
tau_dir = unit(tau4[..., :3])
pdir = unit(pions_t[..., :3])
ip_t = unit(tau_dir - np.sum(tau_dir * pdir, -1, keepdims=True) * pdir)
ana_t = np.concatenate([np.where((tmode == 1)[..., None], pi0_t[..., :3], ip_t),
                        np.where((tmode == 1)[..., None], pi0_t[..., 3:4], 0.0)], -1)
y_lab = np.where(tmode == 1, (pions_t[..., 3] - pi0_t[..., 3])
                 / np.maximum(pions_t[..., 3] + pi0_t[..., 3], 1e-9), 0.0)
pr = boost_to(pions_t, tau4)
zr = boost_to(pi0_t, tau4)
y_rest = np.where(tmode == 1, (pr[..., 3] - zr[..., 3]) / np.maximum(pr[..., 3] + zr[..., 3], 1e-9), 0.0)

# reco-side availability, identical cohort to p4
pv, t = T['pv'], T['trk']
lead = t[:, :, 0]
ok = np.isfinite(lead[..., 0])
beam = np.array([*np.median(pv[:, :2], axis=0),
                 -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))])
theta, phi_t, d0, z0 = (t[..., i] for i in (1, 2, 4, 5))
tdir = np.stack([np.sin(theta) * np.cos(phi_t), np.sin(theta) * np.sin(phi_t), np.cos(theta)], -1)
pca = np.stack([-d0 * np.sin(phi_t), d0 * np.cos(phi_t), z0], -1) + beam
ipv = pca - pv[:, None, None, :]
ipv = ipv - np.sum(ipv * tdir, -1, keepdims=True) * tdir
ids = L['global_indices']
ip_lead = ipv[ids, :, 0]
ip_ok = np.isfinite(d0)[ids, :, 0] & np.isfinite(ip_lead).all(-1) & (np.linalg.norm(ip_lead, axis=-1) > 1e-9)
pions_r = L['reco_h_pions_lab4'][:, :, 0]
pi0_r = L['reco_h_neutral_lab4']
rmode = L['reco_h_mode']
side_ok_r = np.where(rmode == 1, (L['reco_h_neutral_counts'] == 1) & np.isfinite(pi0_r).all(-1)
                     & (pi0_r[..., 3] > 0), ip_ok)
base_ok = y & L['reco_h_available_side'].all(1) & (rmode == tmode).all(1) \
    & np.isfinite(pions_r).all((1, 2)) & side_ok_r.all(1)

P = np.load(HERE / 'data' / 'gen3pi_full22_s42.npz')['h_pred'].astype(float)
chans = {'pi-pi': (tmode[:, 0] == 0) & (tmode[:, 1] == 0),
         'rho-rho': (tmode[:, 0] == 1) & (tmode[:, 1] == 1),
         'pi-rho': ((tmode[:, 0] == 0) & (tmode[:, 1] == 1)) | ((tmode[:, 0] == 1) & (tmode[:, 1] == 0))}
rows = {}
for name, chan in chans.items():
    k = base_ok & chan
    Sk = S_all[k]
    res = {'n': int(k.sum())}
    for tag, yv in (('lab_y', y_lab), ('rest_frame_y', y_rest)):
        yf = np.where(tmode[k] == 1, np.sign(yv[k]), 1.0).prod(1) < 0
        pcp, _ = phistar_cp(pions_t[k, 0], pions_t[k, 1], ana_t[k, 0], ana_t[k, 1], yf)
        res[f'classical_truth_{tag}'] = sensitivity(np.sin(pcp), Sk, 10000, boot=200)['sigma_deg']
    fl = np.where(tmode[k] == 1, np.sign(y_lab[k]), 1.0).prod(1) < 0
    fr_ = np.where(tmode[k] == 1, np.sign(y_rest[k]), 1.0).prod(1) < 0
    res['y_sign_disagreement'] = float(np.mean(fl != fr_))
    res['learned_reco_h_Tp'] = sensitivity(triple(P[k, 0], P[k, 1]), Sk, 10000, boot=200)['sigma_deg']
    rows[name] = res
out['classical_rest_frame_y'] = rows

# ---- 3. same-seed geometry comparison ----
G = np.load(HERE / 'data' / 'gen3pi_full22_s42.npz')
no3p = (tmode != 3).all(1)
kH = no3p & y
geo = {}
for tag, arm in (('base (seed 43)', 'gen3pi_base_s43'), ('IP+SV v2 (seed 42)', 'gen3pi_full22_s42'),
                 ('IP+SV v2 (seed 43)', 'gen3pi_full22_s43'), ('ideal-IP (seed 42)', 'gen3pi_idealip22_s42')):
    hp = np.load(HERE / 'data' / f'{arm}.npz')['h_pred'].astype(float)
    geo[tag] = {'cp_sigma_deg': sensitivity(triple(hp[kH, 0], hp[kH, 1]), S_all[kH], 10000, boot=200)['sigma_deg'],
                'hz_auc': auc(llr(hp[no3p]), y[no3p])}
b = geo['base (seed 43)']
dp = math.sqrt(2) * ppf(b['hz_auc'])
for tag, r in geo.items():
    r['cp_effective_lumi'] = (b['cp_sigma_deg'] / r['cp_sigma_deg'])**2
    r['hz_effective_lumi'] = (math.sqrt(2) * ppf(r['hz_auc']) / dp)**2
out['same_seed_geometry'] = {'selection': 'no 3-prong side, generator-current teacher throughout',
                             'arms': geo}

(HERE / 'results' / 'review_followups.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
