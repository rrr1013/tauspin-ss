"""Kinematic solution set with reco visible momenta and reco MET, and the H/Z
spin likelihood ratio on it, with the IP / SV geometry as likelihoods.

This extends the truth-level null space of the 09-19 / 09-23 runs to the reco
level so that the loss from truth to reco can be split by stage:

  visible  : per tau side, the polarimeter constituents (pions, pi0) and the
             kinematics come either from truth or from reco.  --visible
             truth | reco | reco_pi | reco_rho | reco_3pi   (reco_X: only sides
             whose truth mode is X use reco, the rest stay truth)
  MET      : --met exact  sum of the two truth tau-neutrino pT, taken as exact
                          (sigma 0) or with a small Gaussian tolerance
             --met reco   reco MET with the TRAIN-fitted resolution (isotropic
                          two-component Gaussian mixture of reco MET minus the
                          truth tau-neutrino pT sum)
  geometry : measures flat, ip1, ip3, sv, ip3_sv, ip3_sv_mass, ip3_shuffle, as in
             p12 (IP azimuth response per track rank and |IP| bin, SV angle
             density, both TRAIN-fitted against the truth tau direction), and
             ipg*: the full IP likelihood -- every core track's true IP is
             L * perp(tau direction) with one decay length L ~ Exp(beta gamma c tau)
             shared by the tracks of a tau; the measured transverse / longitudinal
             IP components carry Gaussian errors fitted on TRAIN (s6) with a 5 %
             x5 tail; L is integrated analytically.  This uses the anisotropic
             errors and the IP magnitude, which the azimuth response ignores.

Sampler.  Side i's allowed transverse neutrino momenta form an ellipse
nu_i = c_i + s_i A_i + t_i B_i with s^2 + t^2 <= 1 (A major, B minor axis).  The
target measure is flat on both ellipses times the MET likelihood
N(nu_0 + nu_1 - MET).  Draw t_0, t_1 ~ U(-1, 1) and the MET error delta from the
fitted mixture, and solve the 2x2 linear system s_0 A_0 + s_1 A_1 = MET + delta -
c_0 - c_1 - t_0 B_0 - t_1 B_1 for (s_0, s_1).  All maps are linear with
event-constant Jacobians, so accepted draws (both points inside their disks)
follow the target exactly with unit weight; with sigma = 0 this is the flat
measure on the exact-MET solution set of the earlier runs.

Reco sides whose visible mass exceeds m_tau have no mass-shell solution; they
use m_tau,eff = max(m_tau, 1.02 m_vis) (the fraction is reported), and the
polarimeter is evaluated off shell there.

Output per event: log likelihood ratio log(E[w_H] / E[w_Z]) per measure, with
w_X = 1 + h-^T C_X h+ evaluated in each hypothesis' own n,r,k basis, the
measure-averaged h in a common reference basis, ESS and acceptance.  Test is
never loaded; only the TRAIN split is used for response fits.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metspace as ms  # noqa: E402
import nullspace as ns  # noqa: E402

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
SURFACE = Path('/home/rbaba/azimuth-nullspace-20260918/artifacts/truth_surface.npz')
TRACKS = Path('/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz')
AUDIT = Path('/home/rbaba/tauspin-ip-sv-analysis-corrected-20260920/geometry_audit.npz')
MEASURES = ('flat', 'ip1', 'ip3', 'sv', 'ip3_sv', 'ip3_sv_mass', 'ip3_shuffle',
            'ipg', 'ipg_sv', 'ipg_sv_mass', 'ipg_shuffle')
RESOLUTION = Path('/home/rbaba/hz-spin-synthesis-20260923/artifacts/s6_ip_resolution.json')
CTAU_UM = 87.03
TAIL_FRACTION = 0.05
TAIL_SCALE = 5.0
SV_EDGES = np.geomspace(1e-5, np.pi, 61)
IP_EDGES_UM = np.array([0, 10, 20, 40, 80, 160, 320, 1e9])
PHI_BINS = 48
PARENT_MASS = 91.1874
TAU_MASS = ns.UPSTREAM_TAU_MASS
SHEETS = ((0, 0), (0, 1), (1, 0), (1, 1))
C_H = np.array([1.0, 1.0, -1.0])
C_Z = np.array([0.0, 0.0, 1.0])
MODE_CODE = {'reco_pi': 0, 'reco_rho': 1, 'reco_3pi': 3}


# ----------------------------------------------------------------- geometry
def track_frame(t):
    e1 = np.cross(np.broadcast_to(np.array([0.0, 0.0, 1.0]), t.shape), t)
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-300)
    return e1, np.cross(t, e1)


def azimuth_in_frame(v, e1, e2):
    return np.arctan2(np.sum(v * e2, -1), np.sum(v * e1, -1))


def ip_measurements(tracks, ids, j):
    pv, t, n_core = tracks['pv'], tracks['trk'], tracks['n_core']
    ok = np.isfinite(t[:, :, 0, 0])
    lead = t[:, :, 0]
    bxy = np.median(pv[:, :2], axis=0)
    bz = -float(np.median((lead[..., 5] - pv[:, None, 2])[ok]))
    trj = t[:, :, j]
    theta, phi, d0, z0 = (trj[ids][..., i] for i in (1, 2, 4, 5))
    tdir = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    pca = np.stack([-d0 * np.sin(phi), d0 * np.cos(phi), z0], -1) + np.array([bxy[0], bxy[1], bz])
    ip = pca - pv[ids][:, None, :]
    ip -= np.sum(ip * tdir, -1, keepdims=True) * tdir
    e1, e2 = track_frame(tdir)
    pt = trj[ids][..., 0]
    return {'e1': e1, 'e2': e2, 'psi': azimuth_in_frame(ip, e1, e2),
            'mag_um': np.linalg.norm(ip, axis=-1) * 1e3, 'n_core': n_core[ids],
            'available': np.isfinite(d0) & (n_core[ids] > j),
            'tdir': tdir, 'bT': np.sum(ip * e1, -1) * 1e3, 'bL': np.sum(ip * e2, -1) * 1e3,
            'pt': pt, 'abseta': np.abs(-np.log(np.tan(np.clip(theta, 1e-6, np.pi - 1e-6) / 2)))}


def attach_resolution(meas):
    """Per-track transverse / longitudinal IP error from the TRAIN fit of s6."""
    table = json.loads(RESOLUTION.read_text())
    pe, ee = np.array(table['bins_pt']), np.array(table['bins_eta'])
    sT, sL = np.array(table['sigma_T_um'], float), np.array(table['sigma_L_um'], float)
    i = np.clip(np.digitize(np.nan_to_num(meas['pt']), pe) - 1, 0, len(pe) - 2)
    k = np.clip(np.digitize(np.nan_to_num(meas['abseta']), ee) - 1, 0, len(ee) - 2)
    meas['sT'], meas['sL'] = sT[i, k], sL[i, k]


def ipg_log_weight(tau_dir, p_tau, metas, block, b_keys=('bT', 'bL')):
    """Full IP likelihood: all core tracks of a tau share one decay length L ~ Exp(bg c tau);
    the true IP of track j is L * perp_j(tau_dir); measured (bT, bL) has Gaussian errors
    (sT, sL) with a 5% x5 tail.  L is integrated analytically.  Returns (...,R) summed over taus."""
    from scipy.special import log_ndtr
    lam = p_tau / 1.77686 * CTAU_UM                                  # (...,R,2) um
    comps = []
    for scale, logw in ((1.0, np.log(1 - TAIL_FRACTION)), (TAIL_SCALE, np.log(TAIL_FRACTION))):
        A = 0.0; B = 0.0; C = 0.0; norm = 0.0
        for m in metas:
            av = m['available'][block]
            t = m['tdir'][block]
            perp = tau_dir - np.sum(tau_dir * t, -1, keepdims=True) * t
            aT = np.sum(perp * m['e1'][block], -1)
            aL = np.sum(perp * m['e2'][block], -1)
            vT = (scale * m['sT'][block]) ** 2
            vL = (scale * m['sL'][block]) ** 2
            bT = np.nan_to_num(m[b_keys[0]][block]); bL = np.nan_to_num(m[b_keys[1]][block])
            A = A + np.where(av, aT * aT / vT + aL * aL / vL, 0.0)
            B = B + np.where(av, aT * bT / vT + aL * bL / vL, 0.0)
            C = C + np.where(av, bT * bT / vT + bL * bL / vL, 0.0)
            norm = norm + np.where(av, -np.log(2 * np.pi) - 0.5 * np.log(vT * vL), 0.0)
        A = np.maximum(A, 1e-30)
        mu = (B - 1.0 / lam) / A
        logI = (-np.log(lam) + norm + 0.5 * A * mu * mu - 0.5 * C + 0.5 * np.log(2 * np.pi / A)
                + log_ndtr(mu * np.sqrt(A)))
        comps.append(logI + logw)
    any_av = np.zeros(np.shape(comps[0]), bool)
    for m in metas:
        any_av = any_av | m['available'][block]
    total = np.where(any_av, np.logaddexp(comps[0], comps[1]), 0.0)
    return total.sum(-1)


def fit_ip_response(meas, truth_dir):
    psi_true = azimuth_in_frame(truth_dir, meas['e1'], meas['e2'])
    delta = np.angle(np.exp(1j * (psi_true - meas['psi'])))
    cls = (meas['n_core'] >= 2).astype(int)
    b = np.clip(np.digitize(meas['mag_um'], IP_EDGES_UM) - 1, 0, len(IP_EDGES_UM) - 2)
    table = np.ones((2, len(IP_EDGES_UM) - 1, PHI_BINS))
    good = meas['available'] & np.isfinite(delta)
    idx = np.floor((delta[good] + np.pi) / (2 * np.pi) * PHI_BINS).astype(int) % PHI_BINS
    np.add.at(table, (cls[good], b[good], idx), 1.0)
    density = table / table.sum(-1, keepdims=True) * PHI_BINS / (2 * np.pi)
    return np.log(density * 2 * np.pi)


def ip_log_weight(tau_dir, meas, block, response, psi_key='psi'):
    e1, e2 = meas['e1'][block], meas['e2'][block]
    delta = np.angle(np.exp(1j * (azimuth_in_frame(tau_dir, e1, e2) - meas[psi_key][block])))
    idx = np.floor((np.nan_to_num(delta) + np.pi) / (2 * np.pi) * PHI_BINS).astype(int) % PHI_BINS
    cls = (meas['n_core'][block] >= 2).astype(int)
    b = np.clip(np.digitize(meas['mag_um'][block], IP_EDGES_UM) - 1, 0, len(IP_EDGES_UM) - 2)
    return np.where(meas['available'][block], response[cls, b, idx], 0.0).sum(-1)


def fit_sv_response(truth_dir, sv_dir, avail):
    a = np.arccos(np.clip(np.sum(truth_dir * sv_dir, -1), -1, 1))[avail]
    counts = np.histogram(np.clip(a, SV_EDGES[0], SV_EDGES[-1] * 0.999999), SV_EDGES)[0] + 1.0
    solid = 2 * np.pi * (np.cos(SV_EDGES[:-1]) - np.cos(SV_EDGES[1:]))
    return np.log(counts / counts.sum() / solid)


def sv_log_weight(tau_dir, sv_dir, avail, response):
    a = np.arccos(np.clip(np.sum(tau_dir * sv_dir, -1), -1, 1))
    idx = np.clip(np.digitize(a, SV_EDGES) - 1, 0, len(SV_EDGES) - 2)
    return np.where(avail, response[idx], 0.0).sum(-1)


# ---------------------------------------------------------------- MET model
def fit_met_mixture(residual, iterations=200):
    """Isotropic two-component 2-D Gaussian mixture of the MET residual (TRAIN)."""
    r2 = np.sum((residual - np.median(residual, 0)) ** 2, -1)
    pi, s2 = np.array([0.8, 0.2]), np.array([400.0, 2500.0])
    for _ in range(iterations):
        dens = pi / (2 * np.pi * s2) * np.exp(-0.5 * r2[:, None] / s2)
        resp = dens / dens.sum(1, keepdims=True)
        pi = resp.mean(0)
        s2 = (resp * r2[:, None]).sum(0) / (2 * resp.sum(0))
    return {'weights': pi.tolist(), 'sigma_GeV': np.sqrt(s2).tolist()}


def draw_met_error(model, shape, rng):
    if model is None:
        return np.zeros(shape + (2,))
    comp = rng.random(shape) < model['weights'][0]
    sigma = np.where(comp, model['sigma_GeV'][0], model['sigma_GeV'][1])
    return rng.normal(size=shape + (2,)) * sigma[..., None]


# ------------------------------------------------------------------ inputs
def load_inputs(split, visible_mode):
    lad = dict(np.load(LADDER / f'{split}.npz'))
    rows = np.flatnonzero(lad['reco_h_available'])
    out = {'global_indices': lad['global_indices'][rows], 'labels': lad['labels'][rows],
           'weights': lad['weights'][rows].astype(float), 'modes': lad['modes'][rows],
           'truth_nu': lad['truth_neutrino_lab4'][rows][..., :3],
           'reco_met': lad['reco_met_xy'][rows],
           'h_ref': lad['h'][rows].astype(float)}
    reco = {'pions': lad['reco_h_pions_lab4'][rows],
            'pion_charges': lad['reco_h_charges'][rows],
            'pion_counts': lad['reco_h_counts'][rows],
            'pi0': lad['reco_h_neutral_lab4'][rows],
            'pi0_counts': lad['reco_h_neutral_counts'][rows],
            'modes': lad['reco_h_mode'][rows].astype(np.int64)}
    if visible_mode == 'reco':
        use_reco = np.ones((len(rows), 2), bool)
    else:
        if split != 'validation':
            raise RuntimeError('truth constituents only cached for validation')
        with np.load(SURFACE) as s:
            if not np.array_equal(s['global_indices'], lad['global_indices']):
                raise RuntimeError('surface / ladder row order differ')
            truth = {k: s[k][rows] for k in ('pions', 'pion_charges', 'pion_counts', 'pi0',
                                             'pi0_counts', 'modes')}
        if visible_mode == 'truth':
            use_reco = np.zeros((len(rows), 2), bool)
        else:
            use_reco = out['modes'] == MODE_CODE[visible_mode]
        for key, value in truth.items():
            sel = use_reco.reshape(use_reco.shape + (1,) * (value.ndim - 2))
            reco[key] = np.where(sel, reco[key].astype(value.dtype), value)
    surf = dict(reco)
    surf['visible'] = (np.where((np.arange(3)[None, None, :] < surf['pion_counts'][..., None])[..., None],
                                surf['pions'], 0.0).sum(2) + surf['pi0'])
    truth_vis = lad['truth_visible_tau_lab4'][rows]
    surf['tau'] = truth_vis + np.concatenate(
        (out['truth_nu'], np.linalg.norm(out['truth_nu'], axis=-1, keepdims=True)), -1)
    surf['h_ref'] = out['h_ref']
    out['surface'] = surf
    out['use_reco'] = use_reco
    return out, lad


# ------------------------------------------------------------------ sampler
def sample(vis, m_eff, met, n_draw, keep, met_model, rng, rounds=3):
    """Accepted transverse pairs (K,R,2,2) with -1 padding mask, and acceptance."""
    rows = len(vis)
    f0 = ms.ellipse_frame(vis[:, 0], m_eff[:, 0])
    f1 = ms.ellipse_frame(vis[:, 1], m_eff[:, 1])
    G = np.stack((f0['axis_major'], f1['axis_major']), -1)            # (R,2,2) columns A0,A1
    det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    inv = np.stack((np.stack((G[:, 1, 1], -G[:, 0, 1]), -1),
                    np.stack((-G[:, 1, 0], G[:, 0, 0]), -1)), -2) / np.where(
        np.abs(det) > 1e-300, det, 1e-300)[:, None, None]
    bank = np.zeros((0, rows, 2, 2))
    ok_bank = np.zeros((0, rows), bool)
    tried = np.zeros(rows)
    for _ in range(rounds):
        t = 2.0 * rng.random((n_draw, rows, 2)) - 1.0
        delta = draw_met_error(met_model, (n_draw, rows), rng)
        rhs = (met[None] + delta - f0['centre'][None] - f1['centre'][None]
               - t[..., 0, None] * f0['axis_minor'][None] - t[..., 1, None] * f1['axis_minor'][None])
        s = np.einsum('rij,nrj->nri', inv, rhs)
        acc = (s[..., 0] ** 2 + t[..., 0] ** 2 <= 1.0) & (s[..., 1] ** 2 + t[..., 1] ** 2 <= 1.0)
        acc &= np.isfinite(s).all(-1) & (np.abs(det) > 1e-300)[None]
        nu0 = f0['centre'][None] + s[..., 0, None] * f0['axis_major'][None] + t[..., 0, None] * f0['axis_minor'][None]
        nu1 = f1['centre'][None] + s[..., 1, None] * f1['axis_major'][None] + t[..., 1, None] * f1['axis_minor'][None]
        bank = np.concatenate((bank, np.stack((nu0, nu1), -2)), 0)
        ok_bank = np.concatenate((ok_bank, acc), 0)
        tried += n_draw
        if (ok_bank.sum(0) >= keep).all():
            break
    index = ms.compact(ok_bank, keep, rng)
    pairs = np.take_along_axis(bank, np.maximum(index, 0)[..., None, None], axis=0)
    return pairs, index >= 0, ok_bank.sum(0) / tried, (f0, f1)


def build_sheets(vis, m_eff, nu_t_pair):
    solved = ns.solve_nu_z(vis[None], nu_t_pair, m_eff[None])
    nz, valid = solved['nz'], solved['valid']
    nu = np.stack([np.concatenate((nu_t_pair, np.stack((nz[..., 0, i], nz[..., 1, j]), -1)[..., None]), -1)
                   for i, j in SHEETS], axis=1)                              # (K,4,R,2,3)
    ok = np.stack([valid[..., 0, i] & valid[..., 1, j] for i, j in SHEETS], axis=1)
    return nu, ok


# ------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='validation', choices=('validation', 'train'))
    p.add_argument('--visible', default='reco', choices=('truth', 'reco', 'reco_pi', 'reco_rho', 'reco_3pi'))
    p.add_argument('--met', default='reco', choices=('exact', 'reco'))
    p.add_argument('--met-sigma', type=float, default=0.0, help='Gaussian tolerance for --met exact')
    p.add_argument('--draws', type=int, default=4096)
    p.add_argument('--keep', type=int, default=384)
    p.add_argument('--row-start', type=int, default=0)
    p.add_argument('--row-stop', type=int, default=0)
    p.add_argument('--row-chunk', type=int, default=400)
    p.add_argument('--hypothesis-chunk', type=int, default=64)
    p.add_argument('--mass-sigma', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--ip-sim-scale', type=float, default=0.0,
                   help='>0: replace the measured IPs by a simulation from the truth tau direction, '
                        'L ~ Exp(beta gamma c tau) shared by the tracks of a tau, Gaussian errors '
                        '(sigma_T, sigma_L) x scale; the likelihood uses the same scaled errors')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(a.seed + 7919 * a.row_start)

    data, lad = load_inputs(a.split, a.visible)
    n_all = len(data['labels'])
    stop = a.row_stop or n_all
    sel = np.arange(a.row_start, min(stop, n_all))
    surf = {k: v[sel] for k, v in data['surface'].items()}
    ids = data['global_indices'][sel].astype(np.int64)
    truth_nu = data['truth_nu'][sel]
    vis_all = surf['visible']
    m_vis = np.sqrt(np.maximum(ns.minkowski_dot(vis_all, vis_all), 0.0))
    m_eff = np.maximum(TAU_MASS, 1.02 * m_vis)
    polarimeter, _ = ns.load_polarimeter()

    # TRAIN-only response fits (truth tau direction vs reco geometry / MET)
    with np.load(LADDER / 'train.npz') as tr:
        tr_ids = np.asarray(tr['global_indices'])
        tr_dir = tr['truth_visible_tau_lab4'][..., :3] + tr['truth_neutrino_lab4'][..., :3]
        tr_res = tr['reco_met_xy'] - tr['truth_neutrino_lab4'][:, :, :2].sum(1)
    tr_dir /= np.linalg.norm(tr_dir, axis=-1, keepdims=True)
    tracks = dict(np.load(TRACKS))
    responses, metas = [], []
    for j in range(3):
        responses.append(fit_ip_response(ip_measurements(tracks, tr_ids, j), tr_dir))
        metas.append(ip_measurements(tracks, ids, j))
    shuffle_rng = np.random.default_rng(a.seed + 7)
    key = (np.minimum(metas[0]['n_core'], 3) * 10
           + np.clip(np.digitize(metas[0]['mag_um'], IP_EDGES_UM) - 1, 0, 6)).reshape(-1)
    perm = np.arange(key.size)
    for k in np.unique(key):
        where = np.flatnonzero(key == k)
        perm[where] = where[shuffle_rng.permutation(len(where))]
    for m in metas:
        m['psi_shuffle'] = m['psi'].reshape(-1)[perm].reshape(m['psi'].shape)
        attach_resolution(m)
        # shuffled measurement: another event's (bT, bL, errors) placed in this track's own frame
        m['bT_shuffle'] = m['bT'].reshape(-1)[perm].reshape(m['bT'].shape)
        m['bL_shuffle'] = m['bL'].reshape(-1)[perm].reshape(m['bL'].shape)
    if a.ip_sim_scale > 0:
        sim_rng = np.random.default_rng(a.seed + 99)
        tau_true = surf['tau'][:, :, :3]
        p_true = np.linalg.norm(tau_true, axis=-1)
        d_true = tau_true / p_true[..., None]
        L = sim_rng.exponential(p_true / 1.77686 * CTAU_UM)            # (R,2) um, shared by tracks
        for m in metas:
            t = m['tdir']
            perp = d_true - np.sum(d_true * t, -1, keepdims=True) * t
            m['sT'] = m['sT'] * a.ip_sim_scale
            m['sL'] = m['sL'] * a.ip_sim_scale
            m['bT'] = L * np.sum(perp * m['e1'], -1) + sim_rng.normal(size=L.shape) * m['sT']
            m['bL'] = L * np.sum(perp * m['e2'], -1) + sim_rng.normal(size=L.shape) * m['sL']
            m['bT_shuffle'] = m['bT'].reshape(-1)[perm].reshape(m['bT'].shape)
            m['bL_shuffle'] = m['bL'].reshape(-1)[perm].reshape(m['bL'].shape)
    with np.load(AUDIT) as g:
        sv_av = np.asarray(g['sv_available']) > 0
        sv_dir_all = np.asarray(g['sv_direction'])
        sv_len = np.asarray(g['sv_length'])
    sv_ok_all = sv_av & (sv_len > 1e-6) & np.isfinite(sv_dir_all).all(-1)
    sv_response = fit_sv_response(tr_dir, sv_dir_all[tr_ids], sv_ok_all[tr_ids])
    sv_dir, sv_ok = sv_dir_all[ids], sv_ok_all[ids]

    if a.met == 'reco':
        met_model = fit_met_mixture(tr_res)
        met = data['reco_met'][sel]
    else:
        met_model = None if a.met_sigma <= 0 else {'weights': [1.0, 0.0], 'sigma_GeV': [a.met_sigma, a.met_sigma]}
        met = truth_nu[:, 0, :2] + truth_nu[:, 1, :2]
    print(json.dumps({'rows': len(sel), 'visible': a.visible, 'met': a.met, 'met_model': met_model,
                      'm_vis_gt_mtau_side_fraction': float((m_vis > TAU_MASS).mean()),
                      'reco_side_fraction': float(data['use_reco'][sel].mean())}), flush=True)

    R = len(sel)
    sum_w = {m: np.zeros(R) for m in MEASURES}
    sum_w2 = {m: np.zeros(R) for m in MEASURES}
    sum_wh = {m: np.zeros(R) for m in MEASURES}
    sum_wz = {m: np.zeros(R) for m in MEASURES}
    first = {m: np.zeros((R, 2, 3)) for m in MEASURES}
    count = np.zeros(R, np.int64)
    acceptance = np.zeros(R)
    retained = np.zeros(R, np.int64)
    truth_closure = np.full(R, np.nan)

    for start in range(0, R, a.row_chunk):
        block = np.arange(start, min(start + a.row_chunk, R))
        rows = len(block)
        vis, meff = vis_all[block], m_eff[block]
        pairs, keep_mask, acc, _ = sample(vis, meff, met[block], a.draws, a.keep, met_model, rng)
        acceptance[block] = acc
        retained[block] = keep_mask.sum(0)
        nu, ok = build_sheets(vis, meff, pairs)
        ok &= keep_mask[:, None, :]
        count[block] = ok.sum((0, 1))
        tau_h = vis[None, None, :, :, :3] + nu
        tau_dir = tau_h / np.maximum(np.linalg.norm(tau_h, axis=-1, keepdims=True), 1e-300)
        log_ip = [ip_log_weight(tau_dir, metas[j], block, responses[j]) for j in range(3)]
        log_sh = sum(ip_log_weight(tau_dir, metas[j], block, responses[j], 'psi_shuffle') for j in range(3))
        log_ip3 = sum(log_ip)
        log_sv = sv_log_weight(tau_dir, sv_dir[block], sv_ok[block], sv_response)
        p_tau = np.linalg.norm(tau_h, axis=-1)
        log_ipg = ipg_log_weight(tau_dir, p_tau, metas, block)
        log_ipg_sh = ipg_log_weight(tau_dir, p_tau, metas, block, ('bT_shuffle', 'bL_shuffle'))
        e_nu = np.linalg.norm(nu, axis=-1)
        pair4 = np.concatenate((tau_h.sum(-2), (vis[None, None, :, :, 3] + e_nu).sum(-1)[..., None]), -1)
        m_h = np.sqrt(np.maximum(pair4[..., 3] ** 2 - np.sum(pair4[..., :3] ** 2, -1), 0.0))
        log_m = -0.5 * ((m_h - PARENT_MASS) / a.mass_sigma) ** 2

        def bank(log_w):
            log_w = np.where(ok, log_w, -np.inf)
            peak = np.max(log_w.reshape(-1, rows), axis=0)
            peak = np.where(np.isfinite(peak), peak, 0.0)
            return np.where(ok, np.exp(log_w - peak[None, None]), 0.0).reshape(-1, rows)
        banks = {'flat': ok.astype(float).reshape(-1, rows), 'ip1': bank(log_ip[0]),
                 'ip3': bank(log_ip3), 'sv': bank(log_sv), 'ip3_sv': bank(log_ip3 + log_sv),
                 'ip3_sv_mass': bank(log_ip3 + log_sv + log_m), 'ip3_shuffle': bank(log_sh),
                 'ipg': bank(log_ipg), 'ipg_sv': bank(log_ipg + log_sv),
                 'ipg_sv_mass': bank(log_ipg + log_sv + log_m), 'ipg_shuffle': bank(log_ipg_sh)}

        tau_sum = np.zeros((rows, 2, 4))
        for sh in range(4):
            nu4 = np.concatenate((nu[:, sh], np.linalg.norm(nu[:, sh], axis=-1)[..., None]), -1)
            tau_sum += np.sum(np.where(ok[:, sh][..., None, None], vis[None] + nu4, 0.0), 0)
        basis = ns.pair_basis(tau_sum / np.maximum(count[block], 1)[:, None, None])
        bad = ~np.isfinite(basis).all((-1, -2))
        basis[bad] = np.eye(3)

        sub = {k: v[block] for k, v in surf.items()}
        sub['tau'] = surf['tau'][block]
        if a.visible == 'truth':
            tr_res_h = ns.evaluate_h(polarimeter, sub, truth_nu[block][None], np.arange(rows))
            truth_closure[block] = np.abs(tr_res_h['h'][0] - sub['h_ref']).max((-1, -2))

        flat_nu = nu.reshape(-1, rows, 2, 3)
        flat_ok = ok.reshape(-1, rows)
        flat_nu = np.where(flat_ok[..., None, None], flat_nu, np.broadcast_to(truth_nu[block][None], flat_nu.shape))
        for h0 in range(0, flat_nu.shape[0], a.hypothesis_chunk):
            h1 = min(h0 + a.hypothesis_chunk, flat_nu.shape[0])
            res = ns.evaluate_h(polarimeter, sub, flat_nu[h0:h1], np.arange(rows))
            good = flat_ok[h0:h1] & res['valid'] & np.isfinite(res['h']).all((-1, -2))
            h_own = np.nan_to_num(res['h'])
            prod = h_own[..., 0, :] * h_own[..., 1, :]
            w_h = 1.0 + prod @ C_H
            w_z = 1.0 + prod @ C_Z
            h_ref = ms.to_reference_basis(h_own, res['hypothesis_basis'], basis[None])
            h_ref = np.nan_to_num(h_ref)
            for m, wts in banks.items():
                wv = np.where(good, wts[h0:h1], 0.0)
                sum_w[m][block] += wv.sum(0)
                sum_w2[m][block] += (wv ** 2).sum(0)
                sum_wh[m][block] += (wv * w_h).sum(0)
                sum_wz[m][block] += (wv * w_z).sum(0)
                first[m][block] += np.einsum('hr,hrsi->rsi', wv, h_ref)
        print(json.dumps({'rows_done': int(block[-1] + 1), 'elapsed_s': round(time.time() - started, 1)}), flush=True)

    payload = {'global_indices': ids, 'labels': data['labels'][sel], 'weights': data['weights'][sel],
               'modes': data['modes'][sel], 'use_reco': data['use_reco'][sel], 'count': count,
               'acceptance': acceptance, 'retained': retained, 'm_eff': m_eff, 'm_vis': m_vis}
    for m in MEASURES:
        norm = np.maximum(sum_w[m], 1e-300)
        valid = sum_w[m] > 0
        payload[f'log_lr_{m}'] = np.where(valid, np.log(np.maximum(sum_wh[m], 1e-300)) - np.log(np.maximum(sum_wz[m], 1e-300)), np.nan)
        payload[f'mean_h_{m}'] = first[m] / norm[:, None, None]
        payload[f'ess_{m}'] = np.where(sum_w2[m] > 0, sum_w[m] ** 2 / np.maximum(sum_w2[m], 1e-300), 0.0)
    if a.visible == 'truth':
        payload['truth_closure'] = truth_closure
    np.savez_compressed(a.output, **payload)
    summary = {'rows': int(R), 'visible': a.visible, 'met': a.met, 'met_sigma': a.met_sigma,
               'met_model': met_model, 'keep': a.keep, 'draws': a.draws,
               'no_valid_hypothesis': int((count == 0).sum()),
               'acceptance_quantiles': np.quantile(acceptance, [0.05, 0.5, 0.95]).tolist(),
               'retained_quantiles': np.quantile(retained, [0.05, 0.5, 0.95]).tolist(),
               'ess_median': {m: float(np.median(payload[f'ess_{m}'])) for m in MEASURES},
               'truth_closure_max': float(np.nanmax(truth_closure)) if a.visible == 'truth' else None,
               'elapsed_s': time.time() - started, 'host': platform.node()}
    Path(str(a.output).replace('.npz', '_summary.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
