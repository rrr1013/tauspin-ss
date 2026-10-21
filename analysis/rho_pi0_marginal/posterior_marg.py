"""Classical posterior polarimeter with the reco-rho pi0 marginalised (extends ../atlas_classical_posterior/posterior.py).

On every reco-rho side (reco_h_mode 1, available) the true pi0 momentum p2 is a latent variable.  For the observed
charged pion p1 the spin-flat joint density of (p2, x, phi) -- x = E_vis/E_tau, phi the tau azimuth around the
visible axis, visible V = p1 + p2 -- is, per d^3p2 dx dphi,

    rho_E(E_tau) E_V / (P pV x^2)      tau-momentum prior f(P) d^3P = rho_E(E) dE dOmega / 4 pi, nu on its shell
  x |B_rho(s)|^2 T(p1, p2, nu)          spin-summed tau -> pi pi0 nu rate, T = 2(q.N)q0 - q^2 N0 (tau rest frame),
                                        B the Kuehn-Santamaria rho+rho' form factor of the generator
  x 1 / E2                              pi0 phase space d^3p2 / 2E2

times the pi0 likelihood L(u, l, raw | E2) (response.py, TRAIN), u = log(E_reco/E2), l = log angle(reco, p2).
Draws: (u, l) from the TRAIN proposal table at the reco pi0 energy, psi uniform, then x uniform on [m_V^2/m_tau^2, 1]
and phi uniform.  With d^3p2 = |p2| E2 dE2 dOmega and the proposal density q(u, l) / (E2 delta sin delta), the
per-draw log weight is

    log rho_E(E_tau) + log E_V - log P - log pV - 2 log x + log(1 - x_min)
  + log |B(s)|^2 + log T + log |p2| + log E2 + log L(u, l, raw | E2) - log q(u, l)

and then the IP / SV / MET (/ parent mass) likelihoods exactly as in posterior.py.  The polarimeter of every retained
pair is evaluated with that pair's own pi0.  Sides that are not reco rho are sampled as in posterior.py (reco visible
fixed, x and phi uniform, |M|^2_unpol); with --e-prior train they also get log rho_E(E_tau) - log P - 2 log x.

--pi0-model fixed : reco pi0 fixed (posterior.py), only the tau-energy prior may change
--pi0-model marg  : as above, pi0 marginalised on reco-rho sides
--pi0-model truth : oracle, truth pi0 on truth-rho reco-rho sides (posterior.py --pi0 truth)
--e-prior invE    : rho_E ~ 1/E (what uniform x implies for a fixed visible system; posterior.py default)
--e-prior train   : TRAIN truth tau-energy density (post-selection)

PIT: on truth-rho reco-rho sides, the posterior CDF at the truth pi0 energy, visible mass and tau energy, from the
final K pairs, for the 'met' and 'ip3_sv_mass' measures; also the angle of the posterior-mean pi0 direction to the
truth pi0.  --split train runs on TRAIN rows with event_number % 5 == 0 (held out of the response tables).
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
sys.path.insert(0, str(HERE.parent / 'atlas_classical_posterior'))
import posterior as CP  # noqa: E402
from polarimeter import _b_rho0  # noqa: E402  (mode_pair_auc_origin, on the path via posterior.py)

G = CP.G
M_TAU, M_PI0 = CP.M_TAU, 0.1349768
PIT_MEASURES = ('met', 'ip3_sv_mass')


def interp_logrho(E):
    lc, lr = G['etau_logcent'], G['etau_logrho']
    if G['e_prior'] == 'invE':
        return -np.log(E)
    return np.interp(np.log(E), lc, lr, left=lr[0] - 5.0, right=lr[-1] - 5.0)


def sample_given_vis(p, mv2, ev, rng):
    """p (R,S,3), mv2, ev (R,S) visible momentum / mass^2 / energy -> t, nu, ok, x, xmin, E_tau, P (all (R,S[,3]))."""
    pm = np.linalg.norm(p, axis=-1)
    xmin = np.minimum(mv2 / M_TAU ** 2, 0.999)
    x = np.maximum(xmin + (1 - xmin) * rng.random(xmin.shape), 1e-4)
    E = ev / x
    P = np.sqrt(np.maximum(E ** 2 - M_TAU ** 2, 1e-12))
    cth = (2 * E * ev - M_TAU ** 2 - mv2) / (2 * P * pm)
    ok = (np.abs(cth) <= 1.0) & (mv2 < M_TAU ** 2)
    cth = np.clip(cth, -1, 1)
    sth = np.sqrt(1 - cth ** 2)
    phi = (rng.random(xmin.shape) * 2 - 1) * np.pi
    v = CP.unit(p)
    u1, u2 = CP.perp_basis(v)
    t = cth[..., None] * v + sth[..., None] * (np.cos(phi)[..., None] * u1 + np.sin(phi)[..., None] * u2)
    nu = P[..., None] * t - p
    return t, nu, ok, x, xmin, E, P


def rho_rate_log(p1, p2, nu):
    """log(|B(s)|^2 T) for per-draw p1 (R,S,4) or (R,1,4), p2 (R,S,4), nu (R,S,3)."""
    md4 = lambda a, b: a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], -1)
    nu4 = np.concatenate([nu, np.linalg.norm(nu, axis=-1)[..., None]], -1)
    p1 = np.broadcast_to(p1, p2.shape)
    tau = p1 + p2 + nu4
    N = CP.lboost(nu4, tau)
    q = CP.lboost(p1, tau) - CP.lboost(p2, tau)
    T = 2 * md4(q, N) * q[..., 3] - md4(q, q) * N[..., 3]
    vis = p1 + p2
    s = md4(vis, vis)
    B2 = np.abs(_b_rho0(np.maximum(s, 1e-6))) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((T > 0) & np.isfinite(T), np.log(np.where(T > 0, T, 1.0)) + np.log(B2), -np.inf)


def draw_pi0(rows, s, S, rng):
    """pi0 hypotheses for reco-rho sides: p2 (R,S,4), log(L/q) + log|p2| + log E2 (R,S), u (R,S)."""
    R = len(rows)
    raw = G['raw'][rows, s] - 1
    Er = G['neutral0'][rows, s, 3]
    nhat = CP.unit(G['neutral0'][rows, s, :3])
    rb = np.clip(np.searchsorted(G['r_edges'], Er) - 1, 0, G['pq'].shape[1] - 1)
    nU, nL = G['pq'].shape[2:]
    du = G['u_edges'][1] - G['u_edges'][0]
    dl = G['l_edges'][1] - G['l_edges'][0]
    cells = np.empty((R, S), dtype=np.int64)
    for r in range(R):
        c = np.cumsum(G['pq'][raw[r], rb[r]].ravel())
        cells[r] = np.minimum(np.searchsorted(c, rng.random(S) * c[-1]), nU * nL - 1)
    iu, il = cells // nL, cells % nL
    logq = np.log(G['pq'][raw[:, None], rb[:, None], iu, il]) - np.log(du * dl)
    u = G['u_edges'][0] + (iu + rng.random((R, S))) * du
    l = G['l_edges'][0] + (il + rng.random((R, S))) * dl
    E2 = Er[:, None] * np.exp(-u)
    good = E2 > M_PI0 * 1.0001
    E2 = np.where(good, E2, 1.0)
    p2m = np.sqrt(E2 ** 2 - M_PI0 ** 2)
    d = np.exp(l)
    psi = rng.random((R, S)) * 2 * np.pi
    e1, e2 = CP.perp_basis(nhat)
    dirn = (np.cos(d)[..., None] * nhat[:, None] + np.sin(d)[..., None]
            * (np.cos(psi)[..., None] * e1[:, None] + np.sin(psi)[..., None] * e2[:, None]))
    p2 = np.concatenate([p2m[..., None] * dirn, E2[..., None]], -1)
    # likelihood: log density per (u, l) cell, linear in log E2 between the TRAIN true-energy bin centres
    lc = G['e_logcent']
    le = np.clip(np.log(E2), lc[0], lc[-1])
    j = np.clip(np.searchsorted(lc, le) - 1, 0, len(lc) - 2)
    f = (le - lc[j]) / (lc[j + 1] - lc[j])
    tab = G['logL'][raw[:, None], :, iu, il]                       # (R,S,nE)
    logL = ((1 - f) * np.take_along_axis(tab, j[..., None], -1)[..., 0]
            + f * np.take_along_axis(tab, (j + 1)[..., None], -1)[..., 0])
    lw = np.where(good, logL - logq + np.log(p2m) + np.log(E2), -np.inf)
    return p2, lw


def draw_pi0_cond(rows, s, S, rng):
    """As draw_pi0, with the stage-2 response (response_cond.py): likelihood conditioned on the hypothesis pi+-/pi0
    separation and the reco PFO-mass class, proposal on the reco separation; u and l drawn independently."""
    C = G['cond']
    R = len(rows)
    raw = G['raw'][rows, s] - 1
    rn = G['neutral0'][rows, s]
    Er = rn[:, 3]
    nhat = CP.unit(rn[:, :3])
    p1 = CP.unit(G['pions'][rows, s, 0, :3])
    mc = (np.sqrt(np.maximum(Er ** 2 - np.sum(rn[:, :3] ** 2, -1), 0)) > C['m_cut']).astype(int)
    bins = lambda x, e: np.clip(np.searchsorted(e, x) - 1, 0, len(e) - 2)
    rb = bins(Er, C['r_edges'])
    sb = bins(np.arccos(np.clip(np.sum(p1 * nhat, -1), -1, 1)) * Er, C['sr_edges'])
    du = C['u_edges'][1] - C['u_edges'][0]
    dl = C['l_edges'][1] - C['l_edges'][0]

    def draw(logq):                                   # logq (R, nb) log density -> bin index (R,S)
        c = np.cumsum(np.exp(logq), -1)
        return np.stack([np.minimum(np.searchsorted(c[r], rng.random(S) * c[r, -1]), logq.shape[1] - 1) for r in range(R)])
    qu, ql = C['logqu'][raw, mc, rb, sb], C['logql'][raw, mc, rb, sb]
    iu, il = draw(qu), draw(ql)
    logq = np.take_along_axis(qu, iu, 1) + np.take_along_axis(ql, il, 1)
    u = C['u_edges'][0] + (iu + rng.random((R, S))) * du
    l = C['l_edges'][0] + (il + rng.random((R, S))) * dl
    E2 = Er[:, None] * np.exp(-u)
    good = E2 > M_PI0 * 1.0001
    E2 = np.where(good, E2, 1.0)
    p2m = np.sqrt(E2 ** 2 - M_PI0 ** 2)
    d = np.exp(l)
    psi = rng.random((R, S)) * 2 * np.pi
    e1, e2 = CP.perp_basis(nhat)
    dirn = (np.cos(d)[..., None] * nhat[:, None] + np.sin(d)[..., None]
            * (np.cos(psi)[..., None] * e1[:, None] + np.sin(psi)[..., None] * e2[:, None]))
    p2 = np.concatenate([p2m[..., None] * dirn, E2[..., None]], -1)
    sep = np.arccos(np.clip(np.sum(dirn * p1[:, None], -1), -1, 1)) * E2
    def corner(lc, x):                                # bilinear interpolation in log E2 and log sep
        xc = np.clip(x, lc[0], lc[-1])
        j = np.clip(np.searchsorted(lc, xc) - 1, 0, len(lc) - 2)
        return j, (xc - lc[j]) / (lc[j + 1] - lc[j])
    je, fe = corner(C['e_logcent'], np.log(E2))
    js, fs = corner(C['s_logcent'], np.log(np.maximum(sep, 1e-9)))
    logL = 0.0
    rr, mm = raw[:, None], mc[:, None]
    if G['psi']:
        # stage 3: azimuth of the reco pi0 around the hypothesis pi0, from the axis track -> hypothesis pi0
        ea = CP.unit((dirn - p1[:, None]) - np.sum((dirn - p1[:, None]) * dirn, -1, keepdims=True) * dirn)
        eb = np.cross(dirn, ea)
        dv = nhat[:, None] - np.sum(nhat[:, None] * dirn, -1, keepdims=True) * dirn
        psiL = np.arctan2(np.sum(dv * eb, -1), np.sum(dv * ea, -1))
        npsi = C['logpsi'].shape[-1]
        ip = np.clip(np.floor((psiL + np.pi) / (2 * np.pi) * npsi).astype(int), 0, npsi - 1)
        le = C['l_psi_edges']
        lb = np.clip(np.searchsorted(le, np.clip(l, le[0], le[-1] - 1e-9)) - 1, 0, len(le) - 2)
    for a_, wa in ((je, 1 - fe), (je + 1, fe)):
        for b_, wb in ((js, 1 - fs), (js + 1, fs)):
            t_ = C['logcat'][rr, mm, a_, b_] + C['logpu'][rr, mm, a_, b_, iu] + C['logpl'][rr, mm, a_, b_, il]
            if G['psi']:
                t_ = t_ + C['logpsi'][rr, mm, a_, b_, lb, ip]
            logL = logL + wa * wb * t_
    lw = np.where(good, logL - logq + np.log(p2m) + np.log(E2), -np.inf)
    return p2, lw


def side_draws(rows, s, S, rng):
    """All rows of one side -> dict of per-draw arrays (R,S,...) and the base log weight (R,S)."""
    R = len(rows)
    vis = G['vis'][rows, s]
    pm = np.linalg.norm(vis[:, :3], axis=-1)
    cap = G['shell_cap'][rows, s]
    mv2 = np.clip(vis[:, 3] ** 2 - pm ** 2, 0.0, cap ** 2)
    ev = np.sqrt(pm ** 2 + mv2)
    pS = np.broadcast_to(vis[:, None, :3], (R, S, 3))
    t, nu, ok, x, xmin, Et, P = sample_given_vis(pS, np.broadcast_to(mv2[:, None], (R, S)),
                                                  np.broadcast_to(ev[:, None], (R, S)), rng)
    base = np.where(ok, 0.0, -np.inf)
    if G['e_prior'] == 'train':
        base = base + interp_logrho(Et) - np.log(P) - 2 * np.log(x)
    if G['unpol']:
        base = base + CP.unpol_logw_side(rows, s, nu)
    neu = np.broadcast_to(G['neutral'][rows, s][:, None], (R, S, 4)).copy()
    visd = np.broadcast_to(vis[:, None], (R, S, 4)).copy()
    mr = np.flatnonzero(G['marg'][rows, s])
    if len(mr):
        p1 = G['pions'][rows[mr], s, 0][:, None]                       # (Rm,1,4)
        p2, lw2 = (draw_pi0_cond if G['cond'] is not None else draw_pi0)(rows[mr], s, S, rng)
        V = p1 + p2
        pv = np.linalg.norm(V[..., :3], axis=-1)
        mv2m = np.maximum(V[..., 3] ** 2 - pv ** 2, 0.0)
        tm, num, okm, xm, xminm, Etm, Pm = sample_given_vis(V[..., :3], mv2m, V[..., 3], rng)
        with np.errstate(divide='ignore', invalid='ignore'):
            lw = (lw2 + interp_logrho(Etm) + np.log(V[..., 3]) - np.log(Pm) - np.log(pv) - 2 * np.log(xm)
                  + np.log(1 - xminm) + rho_rate_log(p1, p2, num))
        lw = np.where(okm & np.isfinite(lw), lw, -np.inf)
        t[mr], nu[mr], base[mr] = tm, num, lw
        x[mr], Et[mr] = xm, Etm
        neu[mr], visd[mr] = p2, V
    return {'t': t, 'nu': nu, 'neu': neu, 'vis': visd, 'Et': Et}, base


def run_block(args):
    start, stop, S, M, K, seed = args
    rng = np.random.default_rng(seed + start)
    rows = np.arange(start, stop)
    R = len(rows)
    D, logs = [], {}
    for s in (0, 1):
        d, base = side_draws(rows, s, S, rng)
        D.append(d)
        ip = sum(CP.ip_logw_side(d['t'], G['meas'][j], rows, s, G['ip_resp'][j], 'psi') for j in range(3))
        ipsh = sum(CP.ip_logw_side(d['t'], G['meas'][j], rows, s, G['ip_resp'][j], 'psi_shuffle') for j in range(3))
        sv = CP.sv_logw_side(d['t'], rows, s)
        logs[s] = {'met': base, 'ip3': base + ip, 'sv': base + sv, 'ip3_sv': base + ip + sv,
                   'ip3_sv_mass': base + ip + sv, 'ip3_shuffle': base + ipsh}
    met, sig = G['met'][rows], G['met_sigma']
    avail = G['avail_side']
    rep = lambda a: np.repeat(a[rows][:, None], K, axis=1)
    res = {}
    for m in CP.MEASURES:
        idx, ess_side = zip(*(CP.systematic(logs[s][m], M, rng) for s in (0, 1)))
        good_row = (idx[0][:, 0] >= 0) & (idx[1][:, 0] >= 0)
        ii = [np.maximum(idx[s], 0) for s in (0, 1)]
        g = lambda s, k: np.take_along_axis(D[s][k], ii[s][..., None] if D[s][k].ndim == 3 else ii[s], 1)
        nu0, nu1 = g(0, 'nu'), g(1, 'nu')
        sum_t = nu0[:, :, None, :2] + nu1[:, None, :, :2]
        lw = -0.5 * np.sum((met[:, None, None] - sum_t) ** 2, -1) / sig ** 2
        if m == 'ip3_sv_mass':
            v0, v1 = g(0, 'vis'), g(1, 'vis')
            e0 = v0[..., 3] + np.linalg.norm(nu0, axis=-1)
            e1 = v1[..., 3] + np.linalg.norm(nu1, axis=-1)
            px = (v0[:, :, None, :3] + nu0[:, :, None]) + (v1[:, None, :, :3] + nu1[:, None, :])
            et = e0[:, :, None] + e1[:, None, :]
            mh = np.sqrt(np.maximum(et ** 2 - np.sum(px ** 2, -1), 0))
            lw = lw - 0.5 * ((mh - CP.PARENT_MASS) / G['mass_sigma']) ** 2
        pidx, ess_pair = CP.systematic(lw.reshape(R, M * M), K, rng)
        pidx = np.maximum(pidx, 0)
        a, b = pidx // M, pidx % M
        pick = lambda arr, i: np.take_along_axis(arr, i[..., None], 1) if arr.ndim == 3 else np.take_along_axis(arr, i, 1)
        nu_pair = np.stack([pick(nu0, a), pick(nu1, b)], axis=2)                       # (R,K,2,3)
        neu_pair = np.stack([pick(g(0, 'neu'), a), pick(g(1, 'neu'), b)], axis=2)        # (R,K,2,4)
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
        r = {'lr': np.where(sw > 0, np.log(eH) - np.log(eZ), 0.0), 'Eh': Eh, 'Ehh': Ehh, 'nvalid': valid.sum(1),
             'ess0': ess_side[0], 'ess1': ess_side[1], 'ess_pair': ess_pair}
        if m in PIT_MEASURES:
            # posterior CDF at the truth, per side, over the K retained pairs (no validity requirement)
            ab = (a, b)
            for s in (0, 1):
                E2 = pick(g(s, 'neu')[..., 3], ab[s])
                vv = pick(g(s, 'vis'), ab[s])
                mv = np.sqrt(np.maximum(vv[..., 3] ** 2 - np.sum(vv[..., :3] ** 2, -1), 0))
                Et = pick(g(s, 'Et'), ab[s])
                tr = G['truth'][rows, s]                                                  # (R,3) E2, m_vis, E_tau
                r[f'pitE2_{s}'] = np.mean(E2 < tr[:, 0:1], 1) + 0.5 * np.mean(E2 == tr[:, 0:1], 1)
                r[f'pitMV_{s}'] = np.mean(mv < tr[:, 1:2], 1)
                r[f'pitET_{s}'] = np.mean(Et < tr[:, 2:3], 1)
                n2 = CP.unit(pick(g(s, 'neu'), ab[s])[..., :3]).mean(1)
                r[f'ang_{s}'] = np.arccos(np.clip(np.sum(CP.unit(n2) * G['truth_pi0dir'][rows, s], -1), -1, 1))
                r[f'meanE2_{s}'] = E2.mean(1)
        res[m] = r
    return start, res


def init(a):
    CP.init_globals(a)
    V = G['V']
    Rz = np.load(a.response)
    for k in ('u_edges', 'l_edges', 'e_logcent', 'logL', 'r_edges', 'pq', 'etau_logcent', 'etau_logrho'):
        G[k] = Rz[k]
    G['e_prior'] = a.e_prior
    G['mass_sigma'] = a.mass_sigma
    G['psi'] = a.psi
    G['cond'] = dict(np.load(a.response_cond)) if a.response_cond else None
    G['raw'] = V['reco_mode_raw']
    G['neutral0'] = V['reco_h_neutral_lab4']
    rr = V['reco_h_available_side'] & (V['reco_h_mode'] == 1)
    G['marg'] = rr & ((G['raw'] == 1) | (G['raw'] == 2)) if a.pi0_model == 'marg' else np.zeros_like(rr)
    # truth for the PIT (truth-rho sides only; others NaN)
    ids = V['global_indices']
    if a.split == 'validation':
        with np.load(CP.SURFACE) as d:
            assert np.array_equal(d['global_indices'], ids)
            tp0, tpi = d['pi0'], d['pions'][:, :, 0]
    else:
        T = np.load(a.truth_train)
        pos = {int(gi): i for i, gi in enumerate(T['global_indices'])}
        tp0, tpi = np.zeros(V['modes'].shape + (4,)), np.zeros(V['modes'].shape + (4,))
        have = np.array([int(gi) in pos for gi in ids])
        j = np.array([pos[int(gi)] for gi in ids[have]], dtype=int)
        tp0[have], tpi[have] = T['pi0'][j], T['pi_lead'][j]
    tv = tpi + tp0
    mv = np.sqrt(np.maximum(tv[..., 3] ** 2 - np.sum(tv[..., :3] ** 2, -1), 0))
    Etau = V['truth_visible_tau_lab4'][..., 3] + V['truth_neutrino_lab4'][..., 3]
    isr = (V['modes'] == 1) & (tp0[..., 3] > 0)
    G['truth'] = np.where(isr[..., None], np.stack([tp0[..., 3], mv, Etau], -1), np.nan)
    G['truth_pi0dir'] = CP.unit(np.where(isr[..., None], tp0[..., :3], 1.0))
    G['truth_rho_side'] = isr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--response', type=Path, required=True)
    ap.add_argument('--pi0-model', choices=('fixed', 'marg', 'truth'), required=True)
    ap.add_argument('--response-cond', type=Path, help='stage 2: response_cond.py output (marg only)')
    ap.add_argument('--e-prior', choices=('invE', 'train'), required=True)
    ap.add_argument('--split', choices=('validation', 'train'), default='validation')
    ap.add_argument('--truth-train', type=Path, help='p0_train_truth.py output (for --split train)')
    ap.add_argument('--psi', action='store_true', help='stage 3: anisotropic pi0 direction response (needs --response-cond with logpsi)')
    ap.add_argument('--mass-sigma', type=float, default=3.0, help='width of the parent-mass term [GeV] (10/17: 3)')
    ap.add_argument('--S', type=int, default=16384)
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--K', type=int, default=512)
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--block', type=int, default=50)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=20261017)
    a = ap.parse_args()
    # fixed settings of the 10/17 final classical posterior
    a.met_model, a.exact_sigma, a.x_prior, a.unpol_weight, a.rho_shell_cap = 'iso', 2.0, 'uniform', True, True
    a.pi0 = 'truth' if a.pi0_model == 'truth' else 'reco'
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    init(a)
    V = G['V']
    if a.split == 'train':
        sub = np.flatnonzero(V['event_numbers'] % 5 == 0)
        sub = sub[:a.rows] if a.rows else sub
        for k, v in list(G.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (len(V['event_numbers']),) and k not in ('V',):
                G[k] = v[sub]
        for mm in G['meas']:
            for k in list(mm):
                if isinstance(mm[k], np.ndarray) and mm[k].shape[:1] == (len(V['event_numbers']),):
                    mm[k] = mm[k][sub]
        V = {k: v[sub] for k, v in V.items()}
        G['V'] = V
        n = len(sub)
    else:
        n = a.rows or len(G['ids'])
    print(json.dumps({'setup_s': time.time() - t0, 'rows': n, 'marg_sides': int(G['marg'][:n].sum())}), flush=True)
    jobs = [(s, min(s + a.block, n), a.S, a.M, a.K, a.seed) for s in range(0, n, a.block)]
    out = {}
    with Pool(a.workers) as pool:
        for k, (start, res) in enumerate(pool.imap_unordered(run_block, jobs)):
            out[start] = res
            if k % 50 == 0:
                print(json.dumps({'done': k + 1, 'of': len(jobs), 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    starts = sorted(out)
    payload = {}
    for m in CP.MEASURES:
        for f in out[starts[0]][m]:
            payload[f'{m}_{f}'] = np.concatenate([out[s][m][f] for s in starts])
    for k in ('global_indices', 'labels', 'weights', 'modes', 'reco_h_mode', 'reco_h_available_side', 'event_numbers',
              'h', 'reco_mode_raw'):
        payload[k] = V[k][:n]
    payload['marg_side'] = G['marg'][:n]
    payload['truth_rho_side'] = G['truth_rho_side'][:n]
    payload['truth_E2_mv_Etau'] = G['truth'][:n]
    np.savez_compressed(a.output / 'posterior.npz', **payload)
    summary = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    summary.update({'rows': n, 'elapsed_s': time.time() - t0, 'host': os.uname().nodename,
                    'rho_cap_GeV': G['rho_cap'], 'met_sigma_GeV': G['met_sigma']})
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
