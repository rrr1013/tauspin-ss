"""Does the classical phi*_CP peak move by 2 phi_tau, and does it have a
spin-independent component of its own?

The review asked why fig1 shows the classical peak moving by about -75 deg
rather than 90 deg.  Here the phase and its bootstrap error are measured for
phi_tau = 0 and 45 deg, and the same observable is built for Z events, which
carry no transverse spin correlation.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import weights  # noqa: E402
from p3_classical import boost_to, phistar_cp, unit  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402


def harmonic(x, w=None, bins=16):
    n, e = np.histogram(x, bins=bins, range=(0, 2 * np.pi), weights=w)
    c = 0.5 * (e[1:] + e[:-1])
    d = n / n.sum()
    M = np.stack([np.ones_like(c), np.cos(c), np.sin(c)], 1)
    b = np.linalg.lstsq(M, d, rcond=None)[0]
    return math.hypot(b[1], b[2]) / b[0], math.atan2(-b[2], b[1])


C0 = c_matrix(0.0)
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
ip_lead, ip_ok = ipv[ids, :, 0], np.isfinite(d0)[ids, :, 0]
pions_r = L['reco_h_pions_lab4'][:, :, 0]
pi0_r = L['reco_h_neutral_lab4']
rmode = L['reco_h_mode']
ana_r = np.concatenate([np.where((rmode == 1)[..., None], pi0_r[..., :3], unit(ip_lead)),
                        np.where((rmode == 1)[..., None], pi0_r[..., 3:4], 0.0)], -1)
y_r = np.where(np.isfinite(pi0_r[..., 3]) & (pi0_r[..., 3] > 0),
               (pions_r[..., 3] - pi0_r[..., 3]) / np.maximum(pions_r[..., 3] + pi0_r[..., 3], 1e-9), 0.0)
side_ok = np.where(rmode == 1, (L['reco_h_neutral_counts'] == 1) & np.isfinite(pi0_r).all(-1)
                   & (pi0_r[..., 3] > 0), ip_ok & np.isfinite(ip_lead).all(-1))
rr = (tmode[:, 0] == 1) & (tmode[:, 1] == 1)
base = L['reco_h_available_side'].all(1) & (rmode == tmode).all(1) \
    & np.isfinite(pions_r).all((1, 2)) & side_ok.all(1) & rr

out = {}
rng = np.random.default_rng(13)
for tag, k in (('H', base & y), ('Z', base & ~y)):
    yf = np.where(rmode[k] == 1, np.sign(y_r[k]), 1.0).prod(1) < 0
    pcp, _ = phistar_cp(pions_r[k, 0], pions_r[k, 1], ana_r[k, 0], ana_r[k, 1], yf)
    row = {'n': int(k.sum())}
    sets = [('phi0', None)]
    if tag == 'H':
        sets.append(('phi45', weights(h_gen[k], np.deg2rad(45.), C0)))
    ph = {}
    for name, w in sets:
        a, p = harmonic(pcp, w)
        boot = [harmonic(pcp[i], None if w is None else w[i])
                for i in (rng.integers(0, len(pcp), len(pcp)) for _ in range(400))]
        row[name] = {'amplitude': a, 'amplitude_se': float(np.std([b[0] for b in boot])),
                     'phase_deg': math.degrees(p),
                     'phase_se_deg': float(np.rad2deg(np.std(np.angle(
                         np.exp(1j * (np.array([b[1] for b in boot]) - p))))))}
        ph[name] = np.array([b[1] for b in boot])
    if tag == 'H':
        d = np.angle(np.exp(1j * (ph['phi45'] - ph['phi0'])))
        row['shift_deg'] = float(np.rad2deg(np.angle(np.exp(1j * (
            np.deg2rad(row['phi45']['phase_deg'] - row['phi0']['phase_deg']))))))
        row['shift_se_deg'] = float(np.rad2deg(np.std(d)))
    out[tag] = row
(HERE / 'results' / 'classical_phase.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
