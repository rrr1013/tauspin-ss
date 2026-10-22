"""Classical posterior polarimeter with tau-kinematics options (extends ../rho_pi0_marginal/posterior_marg.py).

The base is the 10/22 primary: reco-rho pi0 marginalised (stage 1), invE tau-energy prior, IP (3 core tracks, azimuth)
+ SV (direction) + isotropic MET + |M|^2_unpol, parent-mass term N(91.1876, 3 GeV) in 'ip3_sv_mass'.  Options:

  --vis track       pi and 3pi sides: the visible system of the tau hypothesis (mass shell, nu = P t - p_vis, parent
                    mass) is the one the polarimeter uses (sum of the reco pions + neutral PFO sum), not the ATLAS tau
                    four-vector reco_visible_tau_lab4.  rho sides are unchanged (identical already, or marginalised).
  --mass-prior gauss|train
                    gauss: N(91.1876, --mass-sigma); train: TRAIN truth m(tau tau) histogram convolved with a Gaussian of
                    width --mass-sigma (a smoothing width so that the finite pair sample can resolve it)
  --met-model iso|ptbin|exact
                    ptbin: per bin of the visible ditau pT (TRAIN): median bias and RMS along, RMS across the visible pT;
                    exact: oracle, MET = sum of truth nu_T with --exact-sigma
  --decay-length    each side draw x p(|IP_lead| | lam = (P/m_tau) c tau sin(angle(tau, track)), prong class) (TRAIN
                    table); 3-prong sides with a usable SV use p(L_SV | (P/m_tau) c tau) instead of the IP magnitude
                    (the three IPs and the SV share one decay length); applied to the geometry measures only
  --ip-joint        1-prong sides (fewer than 2 core tracks): the leading-track IP enters through one TRAIN table of the
                    joint p(azimuth difference, log|IP| | log lam, track pT bin, |eta| bin), lam as for --decay-length,
                    which replaces both the p12 azimuth response and the separate |IP| term; 3-prong sides as
                    --decay-length (p12 azimuths, SV direction and length)
  --ip-fine         the IP azimuth response (p12: per prong class and |IP| bin) also conditioned on the track pT
                    (0-20-40-80-inf GeV) and |eta| (0-1-1.7-inf); TRAIN without event_number % 5 == 0
  --ip-orient       as --ip-fine, also conditioned on the signed orientation of the measured IP: the measured azimuth is
                    pulled toward the z0 axis (z0 resolution worse than d0), so p(psi_true - psi_meas) depends on psi_meas.
                    With u = psi_meas mod pi, the reflection u -> pi - u flips the sign of the azimuth difference; the
                    table is over min(u, pi - u) (6 bins on [0, pi/2]) and sign x (psi_true - psi_meas).  Since psi_true
                    is uniform on TRAIN, this conditional is proportional to the likelihood in psi_true
  --ip-aniso        as --ip-fine, also conditioned on the orientation of the measured IP in the plane transverse to the
                    track (|psi_meas| folded into [0, pi/2], 4 bins: along the transverse d0 axis ... along the z0 axis),
                    so that the anisotropic IP error (z0 vs d0, PV z vs beam spot) enters the azimuth response

Measures: met (prior + MET), ip3_sv, ip3_sv_mass.  Pairs: M per side by systematic resampling on the side weight, M x M
pairs weighted by MET (and mass), K resampled.  Output adds, per side, the posterior mean tau energy and the PIT of the
truth tau energy over the K pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rho_pi0_marginal'))
import posterior_marg as PM  # noqa: E402

CP = PM.CP
G = CP.G
MEASURES = ('met', 'ip3_sv', 'ip3_sv_mass')
CTAU_UM = 87.03


def interp_rows(tab, edges, x):
    """tab (..., nrow, ny) rows at the centres of edges; linear interpolation in x (clipped) -> (..., ny) via index/frac."""
    c = 0.5 * (edges[1:] + edges[:-1])
    xc = np.clip(x, c[0], c[-1])
    j = np.clip(np.searchsorted(c, xc) - 1, 0, len(c) - 2)
    return j, (xc - c[j]) / (c[j + 1] - c[j])


def dl_logw_side(d, rows, s):
    """Decay-length log weight (R,S) for one side's draws."""
    T = G['kin']
    P = np.sqrt(np.maximum(d['Et'] ** 2 - CP.M_TAU ** 2, 1e-12))
    lb = P / CP.M_TAU * CTAU_UM
    m = G['meas'][0]
    # IP magnitude of the leading track
    tdir = m['tdir'][rows, s]
    sina = np.linalg.norm(np.cross(d['t'], tdir[:, None]), axis=-1)
    lam = np.log10(np.maximum(lb * sina, 1e-3))
    cls = (m['n_core'][rows, s] >= 2).astype(int)
    ye = T['IP_EDGES']
    iy = np.clip(np.searchsorted(ye, np.log10(np.maximum(m['mag_um'][rows, s], 1e-3))) - 1, 0, len(ye) - 2)
    j, f = interp_rows(None, T['LAM_EDGES'], lam)
    tab = T['ip_logp'][cls, :, iy]                                         # (R,nrow)
    v_ip = (1 - f) * np.take_along_axis(tab, j, 1) + f * np.take_along_axis(tab, j + 1, 1)
    use_ip = m['available'][rows, s] & (m['mag_um'][rows, s] > 0)
    # SV length on 3-prong sides with a usable SV
    sv_ok = G['sv_ok'][rows, s]
    ye = T['SV_EDGES']
    iy = np.clip(np.searchsorted(ye, np.log10(np.maximum(G['sv_len_um'][rows, s], 1e-3))) - 1, 0, len(ye) - 2)
    j, f = interp_rows(None, T['LB_EDGES'], np.log10(lb))
    tab = T['sv_logp'][:, iy].T                                            # (R,nrow)
    v_sv = (1 - f) * np.take_along_axis(tab, j, 1) + f * np.take_along_axis(tab, j + 1, 1)
    out = np.where(sv_ok[:, None], v_sv, np.where(use_ip[:, None], v_ip, 0.0))
    return out


PT_EDGES = np.array([0.0, 20.0, 40.0, 80.0, np.inf])
ETA_EDGES = np.array([0.0, 1.0, 1.7, np.inf])


N_ORI = 4
N_SIGNED = 6


def fine_bins(meas, trk, j):
    pt, th = trk[..., j, 0], trk[..., j, 1]
    eta = np.abs(-np.log(np.tan(np.clip(th, 1e-6, np.pi - 1e-6) / 2)))
    pb = np.clip(np.searchsorted(PT_EDGES, np.nan_to_num(pt)) - 1, 0, len(PT_EDGES) - 2)
    eb = np.clip(np.searchsorted(ETA_EDGES, np.nan_to_num(eta)) - 1, 0, len(ETA_EDGES) - 2)
    cls = (meas['n_core'] >= 2).astype(int)
    b = np.clip(np.digitize(meas['mag_um'], CP.P12.IP_EDGES_UM) - 1, 0, len(CP.P12.IP_EDGES_UM) - 2)
    if G.get('ip_orient'):
        u = np.mod(meas['psi'], np.pi)
        uf = np.where(u > np.pi / 2, np.pi - u, u)
        ob = np.clip((uf / (np.pi / 2) * N_SIGNED).astype(int), 0, N_SIGNED - 1)
        eb = eb * N_SIGNED + ob
    elif G.get('ip_aniso'):
        fold = np.abs(np.angle(np.exp(1j * np.where(np.abs(meas['psi']) > np.pi / 2, meas['psi'] - np.pi, meas['psi']))))
        ob = np.clip((fold / (np.pi / 2) * N_ORI).astype(int), 0, N_ORI - 1)
        eb = eb * N_ORI + ob                                   # |eta| bin x orientation bin
    return cls, b, pb, eb


def orient_sign(psi):
    if not G.get('ip_orient'):
        return 1.0
    return np.where(np.mod(psi, np.pi) > np.pi / 2, -1.0, 1.0)


def fit_fine(meas, trk, j, truth_dir, keep):
    P12 = CP.P12
    psi_true = P12.azimuth_in_frame(truth_dir, meas['e1'], meas['e2'])
    delta = np.angle(np.exp(1j * (psi_true - meas['psi']))) * orient_sign(meas['psi'])
    cls, b, pb, eb = fine_bins(meas, trk, j)
    nori = N_SIGNED if G.get('ip_orient') else (N_ORI if G.get('ip_aniso') else 1)
    table = np.ones((2, len(P12.IP_EDGES_UM) - 1, len(PT_EDGES) - 1, (len(ETA_EDGES) - 1) * nori, P12.PHI_BINS))
    good = meas['available'] & np.isfinite(delta) & keep[:, None]
    idx = np.floor((delta[good] + np.pi) / (2 * np.pi) * P12.PHI_BINS).astype(int) % P12.PHI_BINS
    np.add.at(table, (cls[good], b[good], pb[good], eb[good], idx), 1.0)
    return np.log(table / table.sum(-1, keepdims=True) * P12.PHI_BINS)


def ip_fine_logw_side(tdir, rows, s, j):
    m = G['meas'][j]
    e1, e2 = m['e1'][rows, s], m['e2'][rows, s]
    psi_h = np.arctan2(np.sum(tdir * e2[:, None], -1), np.sum(tdir * e1[:, None], -1))
    delta = np.angle(np.exp(1j * (psi_h - m['psi'][rows, s][:, None])))
    delta = delta * np.asarray(orient_sign(m['psi'][rows, s]))[..., None] if G.get('ip_orient') else delta
    idx = np.floor((delta + np.pi) / (2 * np.pi) * CP.P12.PHI_BINS).astype(int) % CP.P12.PHI_BINS
    cls, b, pb, eb = (x[rows, s][:, None] for x in G['fine_bins'][j])
    val = G['ip_fine'][j][cls, b, pb, eb, idx]
    return np.where(m['available'][rows, s][:, None], val, 0.0)


J_PSI, J_IP = 16, np.linspace(0.0, 3.5, 13)          # azimuth bins; log10 |IP| [um] edges (clipped)
J_LAM = np.linspace(0.5, 3.5, 11)                       # log10 lam [um] edges (clipped); rows at the centres
J_PT, J_ETA = np.array([0.0, 30.0, 70.0, np.inf]), np.array([0.0, 1.2, np.inf])


def joint_bins(meas, trk):
    pt, th = trk[..., 0, 0], trk[..., 0, 1]
    eta = np.abs(-np.log(np.tan(np.clip(th, 1e-6, np.pi - 1e-6) / 2)))
    pb = np.clip(np.searchsorted(J_PT, np.nan_to_num(pt)) - 1, 0, len(J_PT) - 2)
    eb = np.clip(np.searchsorted(J_ETA, np.nan_to_num(eta)) - 1, 0, len(J_ETA) - 2)
    im = np.clip(np.searchsorted(J_IP, np.log10(np.maximum(meas['mag_um'], 1e-3))) - 1, 0, len(J_IP) - 2)
    return pb, eb, im


def fit_joint(meas, trk, tdir_true, p_true, keep):
    P12 = CP.P12
    psi_true = P12.azimuth_in_frame(tdir_true, meas['e1'], meas['e2'])
    dpsi = np.angle(np.exp(1j * (psi_true - meas['psi'])))
    sina = np.linalg.norm(np.cross(tdir_true, meas['tdir']), axis=-1)
    lam = np.log10(np.maximum(p_true / CP.M_TAU * CTAU_UM * sina, 1e-3))
    pb, eb, im = joint_bins(meas, trk)
    il = np.clip(np.searchsorted(J_LAM, lam) - 1, 0, len(J_LAM) - 2)
    ip_ = np.floor((dpsi + np.pi) / (2 * np.pi) * J_PSI).astype(int) % J_PSI
    good = meas['available'] & (meas['n_core'] < 2) & np.isfinite(dpsi) & keep[:, None]
    H = np.zeros((len(J_PT) - 1, len(J_ETA) - 1, len(J_LAM) - 1, J_PSI, len(J_IP) - 1))
    np.add.at(H, (pb[good], eb[good], il[good], ip_[good], im[good]), 1.0)
    Hs = H.copy()
    Hs[:, :, 1:-1] = 0.25 * H[:, :, :-2] + 0.5 * H[:, :, 1:-1] + 0.25 * H[:, :, 2:]
    Hs = Hs + 0.5
    return np.log(Hs / Hs.sum((-1, -2), keepdims=True)), int(good.sum())


def ip_joint_logw_side(d, rows, s):
    m = G['meas'][0]
    P = np.sqrt(np.maximum(d['Et'] ** 2 - CP.M_TAU ** 2, 1e-12))
    tdir = m['tdir'][rows, s]
    sina = np.linalg.norm(np.cross(d['t'], tdir[:, None]), axis=-1)
    lam = np.log10(np.maximum(P / CP.M_TAU * CTAU_UM * sina, 1e-3))
    e1, e2 = m['e1'][rows, s], m['e2'][rows, s]
    psi_h = np.arctan2(np.sum(d['t'] * e2[:, None], -1), np.sum(d['t'] * e1[:, None], -1))
    dpsi = np.angle(np.exp(1j * (psi_h - m['psi'][rows, s][:, None])))
    ip_ = np.floor((dpsi + np.pi) / (2 * np.pi) * J_PSI).astype(int) % J_PSI
    c = 0.5 * (J_LAM[1:] + J_LAM[:-1])
    lc = np.clip(lam, c[0], c[-1])
    j = np.clip(np.searchsorted(c, lc) - 1, 0, len(c) - 2)
    f = (lc - c[j]) / (c[j + 1] - c[j])
    pb, eb, im = (x[rows, s][:, None] for x in G['joint_bins'])
    T = G['ip_joint']
    val = (1 - f) * T[pb, eb, j, ip_, im] + f * T[pb, eb, j + 1, ip_, im]
    return np.where(m['available'][rows, s][:, None], val, 0.0)


def ip_vm_logw_side(tdir, rows, s):
    """Diagnostic: leading track only, von Mises in the azimuth difference with fixed kappa, around the measured IP azimuth
    (or the truth tau azimuth with --ip-truth-psi)."""
    m = G['meas'][0]
    e1, e2 = m['e1'][rows, s], m['e2'][rows, s]
    psi_h = np.arctan2(np.sum(tdir * e2[:, None], -1), np.sum(tdir * e1[:, None], -1))
    ref = G['psi_truth'][rows, s] if G['ip_truth_psi'] else m['psi'][rows, s]
    val = G['ip_vm'] * (np.cos(psi_h - ref[:, None]) - 1.0)
    return np.where(m['available'][rows, s][:, None], val, 0.0)


def mass_logw(mh):
    if G['mass_prior'] == 'gauss':
        return -0.5 * ((mh - CP.PARENT_MASS) / G['mass_sigma']) ** 2
    T = G['kin']
    e = T['m_edges']
    c, w = 0.5 * (e[1:] + e[:-1]), np.exp(T['m_logdens']) * np.diff(e)
    sel = w > 1e-5
    c, w, wid = c[sel], w[sel] / w[sel].sum(), np.diff(e)[sel]
    s2 = G['mass_sigma'] ** 2 + wid ** 2 / 12
    flat = mh.reshape(-1)
    out = np.zeros(flat.shape)
    for a in range(0, len(flat), 1 << 18):
        x = flat[a:a + (1 << 18), None]
        out[a:a + (1 << 18)] = np.log(np.sum(w / np.sqrt(2 * np.pi * s2) * np.exp(-0.5 * (x - c) ** 2 / s2), -1) + 1e-300)
    return out.reshape(mh.shape)


def met_logw(r, rows):
    if G['met_model2'] != 'ptbin':
        return -0.5 * np.sum(r ** 2, -1) / (G['met_sigma'] * G['met_scale']) ** 2
    T = G['kin']
    u = G['met_axis'][rows]
    b = np.clip(np.searchsorted(T['met_pt_edges'], G['vis_pt'][rows]) - 1, 0, len(T['met_bias_par']) - 1)
    shp = (len(rows),) + (1,) * (r.ndim - 2)
    bias, sp, sq = (T[k][b].reshape(shp) for k in ('met_bias_par', 'met_sig_par', 'met_sig_perp'))
    ux, uy = u[:, 0].reshape(shp), u[:, 1].reshape(shp)
    rpar = r[..., 0] * ux + r[..., 1] * uy - bias
    rperp = r[..., 0] * uy - r[..., 1] * ux
    return -0.5 * (rpar ** 2 / sp ** 2 + rperp ** 2 / sq ** 2)


def run_block(args):
    start, stop, S, M, K, seed = args
    rng = np.random.default_rng(seed + start)
    rows = np.arange(start, stop)
    R = len(rows)
    D, logs = [], {}
    for s in (0, 1):
        d, base = PM.side_draws(rows, s, S, rng)
        D.append(d)
        if G['ip_vm'] is not None:
            ip = ip_vm_logw_side(d['t'], rows, s)
        elif G['ip_fine'] is not None:
            ip = sum(ip_fine_logw_side(d['t'], rows, s, j) for j in range(3))
        else:
            ip = sum(CP.ip_logw_side(d['t'], G['meas'][j], rows, s, G['ip_resp'][j], 'psi') for j in range(3))
        sv = CP.sv_logw_side(d['t'], rows, s)
        dl = dl_logw_side(d, rows, s) if G['decay_length'] else 0.0
        if G['ip_joint'] is not None:
            one = (G['meas'][0]['n_core'][rows, s] < 2)[:, None]
            ip = np.where(one, ip_joint_logw_side(d, rows, s), ip)
            dl = np.where(one, 0.0, dl_logw_side(d, rows, s))
        geo = base + ip + sv + dl
        logs[s] = {'met': base, 'ip3_sv': geo, 'ip3_sv_mass': geo}
    met = G['met'][rows]
    avail = G['avail_side']
    rep = lambda a: np.repeat(a[rows][:, None], K, axis=1)
    etau_true = G['etau_true'][rows]
    res = {}
    for m in MEASURES:
        idx, ess_side = zip(*(CP.systematic(logs[s][m], M, rng) for s in (0, 1)))
        good_row = (idx[0][:, 0] >= 0) & (idx[1][:, 0] >= 0)
        ii = [np.maximum(idx[s], 0) for s in (0, 1)]
        g = lambda s, k: np.take_along_axis(D[s][k], ii[s][..., None] if D[s][k].ndim == 3 else ii[s], 1)
        nu0, nu1 = g(0, 'nu'), g(1, 'nu')
        r = met[:, None, None] - (nu0[:, :, None, :2] + nu1[:, None, :, :2])
        lw = met_logw(r, rows)
        if m == 'ip3_sv_mass':
            v0, v1 = g(0, 'vis'), g(1, 'vis')
            e0 = v0[..., 3] + np.linalg.norm(nu0, axis=-1)
            e1 = v1[..., 3] + np.linalg.norm(nu1, axis=-1)
            px = (v0[:, :, None, :3] + nu0[:, :, None]) + (v1[:, None, :, :3] + nu1[:, None, :])
            et = e0[:, :, None] + e1[:, None, :]
            mh = np.sqrt(np.maximum(et ** 2 - np.sum(px ** 2, -1), 0))
            lw = lw + mass_logw(mh)
        pidx, ess_pair = CP.systematic(lw.reshape(R, M * M), K, rng)
        pidx = np.maximum(pidx, 0)
        a, b = pidx // M, pidx % M
        pick = lambda arr, i: np.take_along_axis(arr, i[..., None], 1) if arr.ndim == 3 else np.take_along_axis(arr, i, 1)
        nu_pair = np.stack([pick(nu0, a), pick(nu1, b)], axis=2)
        neu_pair = np.stack([pick(g(0, 'neu'), a), pick(g(1, 'neu'), b)], axis=2)
        out = G['pol'](rep(G['pions']), rep(G['charges']), rep(G['counts']), neu_pair, rep(G['ncount']),
                       rep(G['modes']), nu_pair, availability=rep(avail))
        h, valid = out['h'], out['valid'] & good_row[:, None]
        wt = valid.astype(float)
        hh = h[:, :, 0, :, None] * h[:, :, 1, None, :]
        diag = np.einsum('rkii->rki', hh)
        sw = wt.sum(1)
        mean = lambda x: np.where(sw > 0, np.sum(np.where(valid, x * wt, 0.0), 1) / np.maximum(sw, 1e-300), np.nan)
        eH, eZ = mean(1 + diag @ CP.CH), mean(1 + diag @ CP.CZ)
        vm = valid[..., None, None]
        Eh = np.where(vm, h * wt[..., None, None], 0.0).sum(1) / np.maximum(sw, 1e-300)[:, None, None]
        Ehh = np.where(vm, hh * wt[..., None, None], 0.0).sum(1) / np.maximum(sw, 1e-300)[:, None, None]
        rr = {'lr': np.where(sw > 0, np.log(eH) - np.log(eZ), 0.0), 'Eh': Eh, 'Ehh': Ehh, 'nvalid': valid.sum(1),
              'ess0': ess_side[0], 'ess1': ess_side[1], 'ess_pair': ess_pair}
        ab = (a, b)
        for s in (0, 1):
            Et = pick(g(s, 'Et'), ab[s])
            rr[f'meanEt_{s}'] = Et.mean(1)
            rr[f'pitEt_{s}'] = np.mean(Et < etau_true[:, s:s + 1], 1)
        res[m] = rr
    return start, res


def init(a):
    PM.init(a)
    V = G['V']
    G['kin'] = dict(np.load(a.tables))
    G['decay_length'] = a.decay_length
    G['mass_prior'] = a.mass_prior
    G['met_model2'] = a.met_model
    G['met_scale'] = a.met_scale
    cnt = V['reco_h_counts']
    hv = (V['reco_h_pions_lab4'] * (np.arange(3)[None, None, :, None] < cnt[..., None, None])).sum(2) + V['reco_h_neutral_lab4']
    G['vis_tau'] = V['reco_visible_tau_lab4']
    if a.vis == 'track':
        sel = V['reco_h_available_side'] & np.isin(V['reco_h_mode'], (0, 3))
        vis = G['vis'].copy()
        vis[sel] = hv[sel]
        G['vis'] = vis
        G['vis_track_sides'] = int(sel.sum())
    vv = V['reco_visible_tau_lab4'][:, :, :2].sum(1)
    G['vis_pt'] = np.linalg.norm(vv, axis=1)
    with np.load(CP.P12.AUDIT) as g:
        G['sv_len_um'] = g['sv_length'][V['global_indices']] * 1e3
    G['etau_true'] = V['truth_visible_tau_lab4'][..., 3] + V['truth_neutrino_lab4'][..., 3]
    G['ip_joint'] = None
    if a.ip_joint:
        tracks = dict(np.load(CP.TRACKS))
        with np.load(CP.LADDER / 'train.npz') as d:
            tr_ids, tr_ev = d['global_indices'], d['event_numbers']
            tt = d['truth_visible_tau_lab4'] + d['truth_neutrino_lab4']
        mt = CP.P12.ip_measurements(tracks, tr_ids, 0)
        G['ip_joint'], nj = fit_joint(mt, tracks['trk'][tr_ids], CP.unit(tt[..., :3]), np.linalg.norm(tt[..., :3], axis=-1),
                                      tr_ev % 5 != 0)
        G['joint_bins'] = joint_bins(G['meas'][0], tracks['trk'][V['global_indices']])
        print(json.dumps({'ip_joint_train_sides': nj}), flush=True)
    G['ip_vm'], G['ip_truth_psi'] = a.ip_vm, a.ip_truth_psi
    td = CP.unit(V['truth_visible_tau_lab4'][..., :3] + V['truth_neutrino_lab4'][..., :3])
    m0 = G['meas'][0]
    G['psi_truth'] = CP.P12.azimuth_in_frame(td, m0['e1'], m0['e2'])
    G['ip_fine'] = None
    G['ip_aniso'] = a.ip_aniso
    G['ip_orient'] = a.ip_orient
    if a.ip_fine or a.ip_aniso or a.ip_orient:
        tracks = dict(np.load(CP.TRACKS))
        with np.load(CP.LADDER / 'train.npz') as d:
            tr_ids, tr_ev = d['global_indices'], d['event_numbers']
            tr_truth = CP.unit(d['truth_visible_tau_lab4'][..., :3] + d['truth_neutrino_lab4'][..., :3])
        G['ip_fine'], G['fine_bins'] = [], []
        for j in range(3):
            mt = CP.P12.ip_measurements(tracks, tr_ids, j)
            G['ip_fine'].append(fit_fine(mt, tracks['trk'][tr_ids], j, tr_truth, tr_ev % 5 != 0))
            G['fine_bins'].append(fine_bins(G['meas'][j], tracks['trk'][V['global_indices']], j))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tables', type=Path, required=True, help='kin_tables.py output')
    ap.add_argument('--response', type=Path, required=True, help='rho_pi0_marginal response.py output')
    ap.add_argument('--pi0-model', choices=('fixed', 'marg'), default='marg')
    ap.add_argument('--vis', choices=('tau', 'track'), default='tau')
    ap.add_argument('--mass-prior', choices=('gauss', 'train'), default='gauss')
    ap.add_argument('--mass-sigma', type=float, default=3.0)
    ap.add_argument('--met-model', choices=('iso', 'ptbin', 'exact'), default='iso')
    ap.add_argument('--exact-sigma', type=float, default=2.0)
    ap.add_argument('--met-scale', type=float, default=1.0, help='diagnostic: isotropic MET width x this factor')
    ap.add_argument('--decay-length', action='store_true')
    ap.add_argument('--ip-fine', action='store_true')
    ap.add_argument('--ip-aniso', action='store_true')
    ap.add_argument('--ip-orient', action='store_true')
    ap.add_argument('--ip-vm', type=float, default=None, help='diagnostic: von Mises IP azimuth response, this kappa')
    ap.add_argument('--ip-truth-psi', action='store_true', help='oracle (with --ip-vm): truth tau azimuth as the measurement')
    ap.add_argument('--ip-joint', action='store_true')
    ap.add_argument('--split', choices=('validation', 'train'), default='validation')
    ap.add_argument('--truth-train', type=Path, default=Path('/home/rbaba/rho-pi0-marginal-20261022/out/train_truth_rho.npz'),
                    help='needed by posterior_marg.init on --split train (its rho PIT truth; unused here)')
    ap.add_argument('--S', type=int, default=16384)
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--K', type=int, default=512)
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1, help='validation: every stride-th event (pilots)')
    ap.add_argument('--block', type=int, default=50)
    ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--seed', type=int, default=20261017)
    a = ap.parse_args()
    a.response_cond, a.psi, a.e_prior = None, False, 'invE'
    a.x_prior, a.unpol_weight, a.rho_shell_cap = 'uniform', True, True
    a.pi0 = 'reco'
    # CP.init_globals reads a.met_model: 'exact' replaces MET by sum truth nu_T; ptbin is handled here (iso in CP)
    met_model = a.met_model
    a.met_model = 'exact' if met_model == 'exact' else 'iso'
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    init(a)
    a.met_model = met_model
    G['met_model2'] = met_model
    V = G['V']
    nall = len(V['event_numbers'])
    if a.split == 'train':
        sub = np.flatnonzero(V['event_numbers'] % 5 == 0)
        sub = sub[:a.rows] if a.rows else sub
    elif a.stride > 1:
        sub = np.arange(0, nall, a.stride)          # the validation file is ordered by label: pilots need a stride
    else:
        sub = np.arange(a.rows or nall)
    if len(sub) < nall:
        for k, v in list(G.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (nall,):
                G[k] = v[sub]
        for mm in G['meas']:
            for k in list(mm):
                if isinstance(mm[k], np.ndarray) and mm[k].shape[:1] == (nall,):
                    mm[k] = mm[k][sub]
        if G['ip_fine'] is not None:
            G['fine_bins'] = [tuple(x[sub] for x in fb) for fb in G['fine_bins']]
        if G['ip_joint'] is not None:
            G['joint_bins'] = tuple(x[sub] for x in G['joint_bins'])
        V = {k: v[sub] for k, v in V.items()}
        G['V'] = V
        G['ids'] = V['global_indices']
    n = len(sub)
    print(json.dumps({'setup_s': time.time() - t0, 'rows': n, 'marg_sides': int(G['marg'][:n].sum()),
                      'vis_track_sides': G.get('vis_track_sides', 0)}), flush=True)
    jobs = [(s, min(s + a.block, n), a.S, a.M, a.K, a.seed) for s in range(0, n, a.block)]
    out = {}
    with Pool(a.workers) as pool:
        for k, (start, res) in enumerate(pool.imap_unordered(run_block, jobs)):
            out[start] = res
            if k % 50 == 0:
                print(json.dumps({'done': k + 1, 'of': len(jobs), 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    starts = sorted(out)
    payload = {}
    for m in MEASURES:
        for f in out[starts[0]][m]:
            payload[f'{m}_{f}'] = np.concatenate([out[s][m][f] for s in starts])
    for k in ('global_indices', 'labels', 'weights', 'modes', 'reco_h_mode', 'reco_h_available_side', 'event_numbers',
              'h', 'reco_mode_raw'):
        payload[k] = V[k][:n]
    payload['etau_true'] = G['etau_true'][:n]
    np.savez_compressed(a.output / 'posterior.npz', **payload)
    summary = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    summary.update({'rows': n, 'elapsed_s': time.time() - t0, 'host': os.uname().nodename,
                    'rho_cap_GeV': G['rho_cap'], 'met_sigma_GeV': G['met_sigma']})
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
