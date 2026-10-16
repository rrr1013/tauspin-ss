"""Classical posterior polarimeter on the ATLAS full-reco H/Z cohort.

No learned component.  For every event the tau pair is sampled from the
spin-flat (unpolarised) decay prior around the *reconstructed* visible systems
and weighted by detector likelihoods; on the retained pairs the exact
polarimeter of the reconstructed visible system is evaluated with the generator
(TauDecay UFO) 3pi current, as for the gen3pi teacher.

  prior   per side, x = E_vis/E_tau uniform on [m_vis^2/m_tau^2, 1] and the
          azimuth of the tau around the visible axis uniform (mass shell exact;
          m_vis capped at 1.6 GeV for the shell only), as analysis/ip_tau_reco/ipmmc2.py
  IP      PV-referenced 3-D IP of up to three core tracks: the azimuth of the
          hypothesis tau around each track vs the measured IP azimuth, through the
          response density of hz_beyond_ceiling/p12 (fitted on TRAIN, truth tau
          direction, per prong class and |IP| bin)
  SV      angle between hypothesis tau and PV->SV direction, p12 response (TRAIN)
  MET     isotropic Gaussian on reco MET - sum nu_T, sigma from TRAIN (default); or
          anisotropic (along / across the visible ditau pT, median bias, TRAIN); or
          the oracle 'exact' (MET = sum of truth nu_T, narrow Gaussian)
  x prior optional (--x-prior empirical): TRAIN histogram of E_vis(reco)/E_tau(truth) per
          reco mode and reco E_vis decile, as an importance weight on the uniform proposal
  |M|^2   optional (--unpol-weight): each side draw weighted by its spin-summed decay rate (rho, 3pi);
          without it the posterior is phase-space only, which is the spin-flat posterior only for pi
  mass    optional m_tautau ~ N(91.1876, 3 GeV), common to H and Z (as p12)

Sampling per measure: S prior draws per side, M per side kept by systematic
resampling on the side's geometry weight, the M x M pairs weighted by MET (and
mass) and K pairs resampled.  Because the prior is spin-flat, the H/Z likelihood
ratio of the event is E[1 + h-^T C_H h+] / E[1 + h-^T C_Z h+] over the retained
pairs, each h in its own pair basis (C_H = diag(1,1,-1), C_Z = diag(0,0,1)).

Measures: met (prior + MET only), ip3, sv, ip3_sv, ip3_sv_mass, ip3_shuffle
(IP azimuth permuted among sides with the same prong class and |IP| bin).
Also written: the reco-visible polarimeter with the truth neutrinos ('truthnu'),
and the mean log response at the truth tau direction (closure of the weights).

Split: responses and MET sigma from TRAIN only; the output is VALIDATION.
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

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
TRACKS = Path('/home/rbaba/hz-beyond-ceiling-20260923/artifacts/ip_tracks.npz')
P12_DIR = Path('/home/rbaba/met-nullspace-20260919/scripts')
M_TAU = 1.77686
M_VIS_CAP = 1.6
PARENT_MASS = 91.1876
CH, CZ = np.array([1.0, 1.0, -1.0]), np.array([0.0, 0.0, 1.0])
MEASURES = ('met', 'ip3', 'sv', 'ip3_sv', 'ip3_sv_mass', 'ip3_shuffle')

sys.path.insert(0, str(P12_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
import p12_geometry_nullspace as P12  # noqa: E402
import nullspace as ns  # noqa: E402

G = {}
XP_MODES = np.array([-1, 0, 1, 3])
XP_EBINS, XP_XBINS = 10, 40


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def perp_basis(a):
    z = np.zeros_like(a)
    z[..., 2] = 1.0
    u1 = unit(np.cross(a, z))
    return u1, np.cross(a, u1)


def systematic(logw, m, rng):
    """logw (R,S) -> indices (R,m) by systematic resampling; rows with no finite weight -> -1."""
    peak = np.max(logw, axis=1, keepdims=True)
    w = np.exp(logw - np.where(np.isfinite(peak), peak, 0.0))
    w = np.where(np.isfinite(logw), w, 0.0)
    tot = w.sum(1, keepdims=True)
    c = np.cumsum(w / np.maximum(tot, 1e-300), axis=1)
    u = (rng.random((len(w), 1)) + np.arange(m)[None]) / m
    idx = np.minimum(np.array([np.searchsorted(c[r], u[r]) for r in range(len(w))]), w.shape[1] - 1)
    ess = tot[:, 0] ** 2 / np.maximum((w ** 2).sum(1), 1e-300)
    return np.where(tot > 0, idx, -1), ess


def xprior_logw(x, xmin, evis, mode):
    """log of the TRAIN empirical p(x | E_vis bin, reco mode) relative to the uniform proposal."""
    tab, eb, xb = G['xprior']
    c = np.searchsorted(XP_MODES, mode)
    e = np.clip(np.array([np.searchsorted(eb[ci], ei) for ci, ei in zip(c, evis)]) - 1, 0, eb.shape[1] - 2)
    i = np.clip(np.floor(x / xb).astype(int), 0, tab.shape[-1] - 1)
    return tab[c[:, None], e[:, None], i] + np.log(1 - xmin)[:, None]


def sample_side(vis, S, rng, cap=None):
    """vis (R,4) -> tau direction (R,S,3), nu (R,S,3), ok (R,S), x (R,S), xmin (R,)."""
    p = vis[:, :3]
    pm = np.linalg.norm(p, axis=-1)
    cap = np.full(len(p), M_VIS_CAP) if cap is None else cap
    mv2 = np.clip(vis[:, 3] ** 2 - pm ** 2, 0.0, cap ** 2)
    ev = np.sqrt(pm ** 2 + mv2)                     # energy consistent with the capped mass
    xmin = np.minimum(mv2 / M_TAU ** 2, 0.999)
    x = np.maximum(xmin[:, None] + (1 - xmin[:, None]) * rng.random((len(p), S)), 1e-4)
    E = ev[:, None] / x
    P = np.sqrt(np.maximum(E ** 2 - M_TAU ** 2, 1e-12))
    cth = (2 * E * ev[:, None] - M_TAU ** 2 - mv2[:, None]) / (2 * P * pm[:, None])
    ok = np.abs(cth) <= 1.0
    cth = np.clip(cth, -1, 1)
    sth = np.sqrt(1 - cth ** 2)
    phi = (rng.random((len(p), S)) * 2 - 1) * np.pi
    v = unit(p)
    u1, u2 = perp_basis(v)
    t = (cth[..., None] * v[:, None] + sth[..., None] * (np.cos(phi)[..., None] * u1[:, None]
                                                         + np.sin(phi)[..., None] * u2[:, None]))
    nu = P[..., None] * t - p[:, None]
    return t, nu, ok, x, xmin


def ip_logw_side(tdir, meas, rows, side, response, psi_key):
    """tdir (R,S,3) for one side -> log response (R,S), one track's measurement dict."""
    e1, e2 = meas['e1'][rows, side], meas['e2'][rows, side]
    psi_h = np.arctan2(np.sum(tdir * e2[:, None], -1), np.sum(tdir * e1[:, None], -1))
    delta = np.angle(np.exp(1j * (psi_h - meas[psi_key][rows, side][:, None])))
    idx = np.floor((delta + np.pi) / (2 * np.pi) * P12.PHI_BINS).astype(int) % P12.PHI_BINS
    cls = (meas['n_core'][rows, side] >= 2).astype(int)
    b = np.clip(np.digitize(meas['mag_um'][rows, side], P12.IP_EDGES_UM) - 1, 0, len(P12.IP_EDGES_UM) - 2)
    val = response[cls[:, None], b[:, None], idx]
    return np.where(meas['available'][rows, side][:, None], val, 0.0)


def sv_logw_side(tdir, rows, side):
    d, ok = G['sv_dir'][rows, side], G['sv_ok'][rows, side]
    a = np.arccos(np.clip(np.sum(tdir * d[:, None], -1), -1, 1))
    idx = np.clip(np.digitize(a, P12.SV_EDGES) - 1, 0, len(P12.SV_EDGES) - 2)
    return np.where(ok[:, None], G['sv_resp'][idx], 0.0)


def polar(rows, nu_pair, avail):
    """nu_pair (R,K,2,3) -> h (R,K,2,3) own pair basis, valid (R,K)."""
    R, K = nu_pair.shape[:2]
    rep = lambda a: np.repeat(a[rows][:, None], K, axis=1)
    out = G['pol'](rep(G['pions']), rep(G['charges']), rep(G['counts']), rep(G['neutral']),
                   rep(G['ncount']), rep(G['modes']), nu_pair, availability=rep(avail))
    return out['h'], out['valid']


def lboost(v, P):
    """Four-vectors v (...,4) (px,py,pz,E) into the rest frame of P (...,4)."""
    b = P[..., :3] / P[..., 3:4]
    b2 = np.maximum(np.sum(b * b, -1, keepdims=True), 1e-300)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.sum(b * v[..., :3], -1, keepdims=True)
    e = g * (v[..., 3:4] - bp)
    return np.concatenate([v[..., :3] + ((g - 1.0) * bp / b2 - g * v[..., 3:4]) * b, e], -1)


def unpol_logw_side(rows, s, nu):
    """log of the spin-summed decay rate |M|^2 of side s for draws nu (R,S,3); 0 for pi.

    Given the observed visible configuration, the spin-flat posterior over tau hypotheses is
    (x, azimuth uniform) x |M|^2_unpol; the pion configuration inside the visible system is fixed
    by the measurement.  rho: 2(q.N)q0 - q^2 N0 in the tau rest frame (q = pi - pi0);
    3pi: the trace of the generator (TauDecay UFO) current, as r0 in mode_pair_auc_origin/polarimeter.
    Factors depending only on the visible invariant masses are common to all draws and dropped.
    Non-positive rates (unphysical reco visible systems) -> -inf.
    """
    from polarimeter import current_taudecay, slash, V_MINUS_A, G0
    md4 = lambda a, b: a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], -1)
    R, S = nu.shape[:2]
    out = np.zeros((R, S))
    pions, cnt, neu = G['pions'][rows, s], G['counts'][rows, s], G['neutral'][rows, s]
    mode, q = G['modes'][rows, s], G['charges'][rows, s]
    act = np.arange(3)[None, :] < cnt[:, None]
    vis = np.where(act[..., None], pions, 0.0).sum(1) + neu
    nu4 = np.concatenate([nu, np.linalg.norm(nu, axis=-1)[..., None]], -1)
    tau = vis[:, None] + nu4
    N = lboost(nu4, tau)
    rate = np.ones((R, S))
    rho = (mode == 1) & G['avail_side'][rows, s]
    if rho.any():
        qv = lboost(pions[rho, None, 0], tau[rho]) - lboost(neu[rho, None], tau[rho])
        rate[rho] = 2 * md4(qv, N[rho]) * qv[..., 3] - md4(qv, qv) * N[rho, :, 3]
    tri = np.flatnonzero((mode == 3) & G['avail_side'][rows, s])
    ch = -1 if s == 0 else 1
    sgn = 1.0 if ch < 0 else -1.0
    for a in range(0, len(tri), 8):
        t = tri[a:a + 8]
        order = np.argsort(q[t] != ch, axis=-1, kind='stable')
        pp = np.take_along_axis(pions[t], order[..., None], axis=1)
        rest = lboost(pp[:, None], tau[t][:, :, None])
        e = np.concatenate([rest[..., 3:4], sgn * rest[..., :3]], -1)
        n4 = np.concatenate([N[t][..., 3:4], sgn * N[t][..., :3]], -1).astype(complex)
        J = current_taudecay(e[..., 0, :], e[..., 1, :], e[..., 2, :])
        A = np.einsum('...ab,bc->...ac', slash(J), V_MINUS_A)
        Abar = np.einsum('ab,...cb,cd->...ad', G0, np.conj(A), G0)
        rate[t] = np.real(np.einsum('...ii->...', Abar @ slash(n4) @ A))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.isfinite(rate) & (rate > 0), np.log(np.where(rate > 0, rate, 1.0)), -np.inf)
    return out


def run_block(args):
    start, stop, S, M, K, seed = args
    rng = np.random.default_rng(seed + start)
    rows = np.arange(start, stop)
    R = len(rows)
    vis = G['vis'][rows]
    draws = [sample_side(vis[:, s], S, rng, G['shell_cap'][rows, s]) for s in (0, 1)]
    logs = {}
    for s in (0, 1):
        tdir, nu, ok, x, xmin = draws[s]
        base = np.where(ok, 0.0, -np.inf)
        if G['xprior'] is not None:
            base = base + xprior_logw(x, xmin, vis[:, s, 3], G['xp_mode'][rows, s])
        if G['unpol']:
            base = base + unpol_logw_side(rows, s, nu)
        ip = sum(ip_logw_side(tdir, G['meas'][j], rows, s, G['ip_resp'][j], 'psi') for j in range(3))
        ipsh = sum(ip_logw_side(tdir, G['meas'][j], rows, s, G['ip_resp'][j], 'psi_shuffle')
                   for j in range(3))
        sv = sv_logw_side(tdir, rows, s)
        logs[s] = {'met': base, 'ip3': base + ip, 'sv': base + sv, 'ip3_sv': base + ip + sv,
                   'ip3_sv_mass': base + ip + sv, 'ip3_shuffle': base + ipsh}
    met = G['met'][rows]
    sig = G['met_sigma']
    u = G['met_axis'][rows]                                            # (R,2) unit, visible ditau pT
    avail = G['avail_side']
    res = {}
    for m in MEASURES:
        idx, ess_side = zip(*(systematic(logs[s][m], M, rng) for s in (0, 1)))
        good_row = (idx[0][:, 0] >= 0) & (idx[1][:, 0] >= 0)
        i0, i1 = np.maximum(idx[0], 0), np.maximum(idx[1], 0)
        nu0 = np.take_along_axis(draws[0][1], i0[..., None], 1)        # (R,M,3)
        nu1 = np.take_along_axis(draws[1][1], i1[..., None], 1)
        sum_t = nu0[:, :, None, :2] + nu1[:, None, :, :2]               # (R,M,M,2)
        r = met[:, None, None] - sum_t
        if G['met_model'] == 'aniso':
            rpar = r[..., 0] * u[:, None, None, 0] + r[..., 1] * u[:, None, None, 1] - G['met_bias_par']
            rperp = r[..., 0] * u[:, None, None, 1] - r[..., 1] * u[:, None, None, 0]
            lw = -0.5 * (rpar ** 2 / G['met_sig_par'] ** 2 + rperp ** 2 / G['met_sig_perp'] ** 2)
        else:
            lw = -0.5 * np.sum(r ** 2, -1) / sig ** 2
        if m == 'ip3_sv_mass':
            v0, v1 = vis[:, 0], vis[:, 1]
            e0 = v0[:, None, 3] + np.linalg.norm(nu0, axis=-1)          # tau energies (massless nu)
            e1 = v1[:, None, 3] + np.linalg.norm(nu1, axis=-1)
            px = (v0[:, None, None, :3] + nu0[:, :, None]) + (v1[:, None, None, :3] + nu1[:, None, :])
            et = e0[:, :, None] + e1[:, None, :]
            mh = np.sqrt(np.maximum(et ** 2 - np.sum(px ** 2, -1), 0))
            lw = lw - 0.5 * ((mh - PARENT_MASS) / 3.0) ** 2
        flat = lw.reshape(R, M * M)
        pidx, ess_pair = systematic(flat, K, rng)
        pidx = np.maximum(pidx, 0)
        a, b = pidx // M, pidx % M
        nu_pair = np.stack([np.take_along_axis(nu0, a[..., None], 1),
                            np.take_along_axis(nu1, b[..., None], 1)], axis=2)  # (R,K,2,3)
        h, valid = polar(rows, nu_pair, avail)
        valid &= good_row[:, None]
        wt = valid.astype(float)
        hh = h[:, :, 0, :, None] * h[:, :, 1, None, :]                   # (R,K,3,3)
        diag = np.einsum('rkii->rki', hh)
        wH = 1 + diag @ CH
        wZ = 1 + diag @ CZ
        n = valid.sum(1)
        sw = wt.sum(1)
        mean = lambda x: np.where(sw > 0, np.sum(np.where(valid, x * wt, 0.0), 1) / np.maximum(sw, 1e-300), np.nan)
        eH, eZ = mean(wH), mean(wZ)
        vm = valid[..., None, None]
        Eh = np.where(vm, h * wt[..., None, None], 0.0).sum(1) / np.maximum(sw, 1e-300)[:, None, None]
        Ehh = np.where(vm, hh * wt[..., None, None], 0.0).sum(1) / np.maximum(sw, 1e-300)[:, None, None]
        ess_w = sw ** 2 / np.maximum((wt ** 2).sum(1), 1e-300)
        res[m] = {'lr': np.where(sw > 0, np.log(eH) - np.log(eZ), 0.0), 'Eh': Eh, 'Ehh': Ehh,
                  'nvalid': n, 'ess0': ess_side[0], 'ess1': ess_side[1], 'ess_pair': ess_pair, 'ess_unpol': ess_w}
    # truth-neutrino polarimeter of the reco visible system (reference row)
    h_t, v_t = polar(rows, G['truth_nu'][rows][:, None], avail)
    hh = h_t[:, 0, 0] * h_t[:, 0, 1]
    res['truthnu'] = {'lr': np.where(v_t[:, 0], np.log(1 + hh @ CH) - np.log(1 + hh @ CZ), 0.0),
                      'h': h_t[:, 0]}
    return start, res


def init_globals(a):
    t0 = time.time()
    with np.load(LADDER / 'validation.npz') as d:
        V = {k: d[k] for k in d.files}
    with np.load(LADDER / 'train.npz') as d:
        tr_ids = d['global_indices']
        tr_truth = unit(d['truth_visible_tau_lab4'][..., :3] + d['truth_neutrino_lab4'][..., :3])
        tr_met_res = d['reco_met_xy'] - d['truth_neutrino_lab4'][:, :, :2].sum(1)
    tracks = dict(np.load(TRACKS))
    ids = V['global_indices']
    G['ids'] = ids
    G['ip_resp'], G['meas'] = [], []
    for j in range(3):
        G['ip_resp'].append(P12.fit_response(P12.ip_measurements(tracks, tr_ids, j), tr_truth, 'psi', 'mag_um'))
        G['meas'].append(P12.ip_measurements(tracks, ids, j))
    srng = np.random.default_rng(a.seed + 7)
    m0 = G['meas'][0]
    key = (np.minimum(m0['n_core'], 3) * 10
           + np.clip(np.digitize(m0['mag_um'], P12.IP_EDGES_UM) - 1, 0, 6)).reshape(-1)
    perm = np.arange(key.size)
    for k in np.unique(key):
        w = np.flatnonzero(key == k)
        perm[w] = w[srng.permutation(len(w))]
    for m in G['meas']:
        m['psi_shuffle'] = m['psi'].reshape(-1)[perm].reshape(m['psi'].shape)
    with np.load(P12.AUDIT) as g:
        sv_av, sv_dir, sv_len = g['sv_available'] > 0, g['sv_direction'], g['sv_length']
    sv_ok = sv_av & (sv_len > 1e-6) & np.isfinite(sv_dir).all(-1)
    G['sv_resp'] = P12.fit_sv_response(tr_truth, sv_dir[tr_ids], sv_ok[tr_ids])
    G['sv_dir'], G['sv_ok'] = sv_dir[ids], sv_ok[ids]
    G['met_sigma'] = float(np.sqrt(np.mean(tr_met_res ** 2)))   # per component, isotropic
    G['met_model'] = a.met_model
    G['xprior'] = None
    if a.x_prior == 'empirical':
        # x_r = E_vis(reco) / E_tau(truth) on TRAIN, per reco mode and reco E_vis decile.
        # Single-tau polarisation is zero for H and for this Z, so the marginal is spin-flat.
        with np.load(LADDER / 'train.npz') as d:
            ev = d['reco_visible_tau_lab4'][..., 3]
            et = d['truth_visible_tau_lab4'][..., 3] + d['truth_neutrino_lab4'][..., 3]
            md = np.where(d['reco_h_available_side'], d['reco_h_mode'], -1)
        xr = ev / et
        tab = np.zeros((len(XP_MODES), XP_EBINS, XP_XBINS))
        eb = np.zeros((len(XP_MODES), XP_EBINS + 1))
        for c, m in enumerate(XP_MODES):
            sel = md == m
            q = np.quantile(ev[sel], np.linspace(0, 1, XP_EBINS + 1))
            q[0], q[-1] = 0.0, np.inf
            eb[c] = q
            ei = np.clip(np.searchsorted(q, ev[sel]) - 1, 0, XP_EBINS - 1)
            xi = np.clip(np.floor(xr[sel] * XP_XBINS).astype(int), 0, XP_XBINS - 1)
            cnt = np.ones((XP_EBINS, XP_XBINS))
            np.add.at(cnt, (ei, xi), 1.0)
            tab[c] = np.log(cnt / cnt.sum(1, keepdims=True) * XP_XBINS)   # log density on [0,1]
        G['xprior'] = (tab, eb, 1.0 / XP_XBINS)
    with np.load(LADDER / 'train.npz') as d:
        vt = d['reco_visible_tau_lab4'][:, :, :2].sum(1)
    ut = vt / np.linalg.norm(vt, axis=1, keepdims=True)
    rpar = np.sum(tr_met_res * ut, 1)
    rperp = tr_met_res[:, 0] * ut[:, 1] - tr_met_res[:, 1] * ut[:, 0]
    half68 = lambda x: float(np.diff(np.quantile(x, [0.16, 0.84]))[0] / 2)
    G['met_bias_par'], G['met_sig_par'], G['met_sig_perp'] = float(np.median(rpar)), half68(rpar), half68(rperp)
    G['vis'] = V['reco_visible_tau_lab4']
    G['met'] = V['reco_met_xy']
    if a.met_model == 'exact':            # oracle: MET replaced by sum of truth nu_T
        G['met'] = V['truth_neutrino_lab4'][:, :, :2].sum(1)
        G['met_sigma'] = a.exact_sigma
    vv = V['reco_visible_tau_lab4'][:, :, :2].sum(1)
    G['met_axis'] = vv / np.linalg.norm(vv, axis=1, keepdims=True)
    G['pions'], G['charges'] = V['reco_h_pions_lab4'], V['reco_h_charges']
    G['counts'], G['neutral'] = V['reco_h_counts'], V['reco_h_neutral_lab4']
    G['ncount'], G['modes'] = V['reco_h_neutral_counts'], V['reco_h_mode'].astype(np.int64)
    G['avail_side'] = V['reco_h_available_side']
    G['unpol'] = a.unpol_weight
    G['shell_cap'] = np.full(V['reco_h_mode'].shape, M_VIS_CAP)
    G['rho_cap'] = None
    if a.rho_shell_cap:
        # reco rho sides: shell mass capped at the TRAIN 95% quantile of the truth visible mass of truth-rho sides,
        # so that an over-measured reco m_vis does not push x_min above the true x
        with np.load(LADDER / 'train.npz') as d:
            tv, tm = d['truth_visible_tau_lab4'], d['modes']
        mt = np.sqrt(np.maximum(tv[..., 3] ** 2 - np.sum(tv[..., :3] ** 2, -1), 0))[tm == 1]
        G['rho_cap'] = float(np.quantile(mt, 0.95))
        G['shell_cap'] = np.where(V['reco_h_mode'] == 1, G['rho_cap'], M_VIS_CAP)
    G['xp_mode'] = np.where(V['reco_h_available_side'], V['reco_h_mode'], -1).astype(int)
    G['truth_nu'] = V['truth_neutrino_lab4'][..., :3]
    truth_dir = unit(V['truth_visible_tau_lab4'][..., :3] + V['truth_neutrino_lab4'][..., :3])
    G['truth_dir'] = truth_dir
    from generator_current import install
    pol, _ = ns.load_polarimeter()
    G['pol'] = install(pol)
    G['V'] = V
    G['setup_s'] = time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--S', type=int, default=4096)
    ap.add_argument('--M', type=int, default=64)
    ap.add_argument('--K', type=int, default=128)
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--block', type=int, default=100)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--met-model', choices=('iso', 'aniso', 'exact'), default='iso',
                    help='iso: isotropic Gaussian (TRAIN rms); aniso: TRAIN median bias and 68%% half-widths '
                         'along / across the visible ditau pT; exact: oracle MET = sum truth nu_T')
    ap.add_argument('--exact-sigma', type=float, default=2.0)
    ap.add_argument('--unpol-weight', action='store_true',
                    help='weight each side draw by its spin-summed decay rate |M|^2 (rho, 3pi)')
    ap.add_argument('--rho-shell-cap', action='store_true',
                    help='reco rho sides: shell mass capped at the TRAIN 95%% quantile of truth rho visible mass')
    ap.add_argument('--x-prior', choices=('uniform', 'empirical'), default='uniform',
                    help='empirical: TRAIN p(E_vis/E_tau | reco E_vis decile, reco mode) instead of uniform x')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    init_globals(a)
    V = G['V']
    n = a.rows or len(G['ids'])
    # closure of the weights at the truth tau direction (mean log response per side)
    rows = np.arange(n)
    clos = {}
    for s in (0, 1):
        td = G['truth_dir'][rows, s][:, None]
        clos[f'ip3_side{s}'] = float(np.mean(sum(ip_logw_side(td, G['meas'][j], rows, s, G['ip_resp'][j], 'psi')
                                                 for j in range(3))))
        clos[f'sv_side{s}'] = float(np.mean(sv_logw_side(td, rows, s)))
    print(json.dumps({'setup_s': G['setup_s'], 'met_sigma': G['met_sigma'], 'closure': clos}), flush=True)
    jobs = [(s, min(s + a.block, n), a.S, a.M, a.K, a.seed) for s in range(0, n, a.block)]
    out = {}
    with Pool(a.workers) as pool:
        for k, (start, res) in enumerate(pool.imap_unordered(run_block, jobs)):
            out[start] = res
            if k % 20 == 0:
                print(json.dumps({'done': k + 1, 'of': len(jobs), 'elapsed_s': round(time.time() - t0, 1)}),
                      flush=True)
    starts = sorted(out)
    payload = {}
    for m in MEASURES:
        for f in ('lr', 'Eh', 'Ehh', 'nvalid', 'ess0', 'ess1', 'ess_pair', 'ess_unpol'):
            payload[f'{m}_{f}'] = np.concatenate([out[s][m][f] for s in starts])
    payload['truthnu_lr'] = np.concatenate([out[s]['truthnu']['lr'] for s in starts])
    payload['truthnu_h'] = np.concatenate([out[s]['truthnu']['h'] for s in starts])
    for k in ('global_indices', 'labels', 'weights', 'modes', 'reco_h_mode', 'reco_h_available_side',
              'event_numbers', 'h'):
        payload[k] = V[k][:n]
    np.savez_compressed(a.output / 'posterior.npz', **payload)
    summary = {'rows': n, 'S': a.S, 'M': a.M, 'K': a.K, 'seed': a.seed, 'met_sigma_GeV': G['met_sigma'],
               'met_model': a.met_model, 'x_prior': a.x_prior, 'rho_shell_cap_GeV': G['rho_cap'], 'unpol_weight': a.unpol_weight, 'met_aniso': [G['met_bias_par'], G['met_sig_par'], G['met_sig_perp']],
               'closure_mean_log_response_truth_dir': clos, 'elapsed_s': time.time() - t0,
               'host': os.uname().nodename, 'workers': a.workers}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
