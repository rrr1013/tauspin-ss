"""Classical posterior polarimeter with orthogonal rho oracles (extends taukin_gap/posterior_kin.py).

Enables the 2x2 oracle matrix on reco-rho sides:
  --rho-pi0 {marg, truth, truthdir, truthe, fixed}
      marg      : sample pi0 from detector response likelihood (posterior_marg.py default)
      truth     : exact truth pi0 4-momentum on truth-rho reco-rho sides
      truthdir  : truth pi0 direction + reco pi0 energy
      truthe    : reco pi0 direction + truth pi0 energy
      fixed     : reco pi0 fixed (no marginalisation)
  --rho-psi {reco, truth}
      reco      : reconstructed IP orientation-conditioned response (--ip-orient)
      truth     : exact truth tau azimuth around track (von Mises kappa=2000 on truth azimuth)
  --rho-psi-scope {rho_only, all}
      rho_only  : truth azimuth applied only to truth-rho reco-rho sides (default)
      all       : truth azimuth applied to all available sides

Always includes:
  --vis track       (reco pions + neutral PFO on pi/3pi)
  --decay-length    (decay length log likelihood)
  --ip-orient       (signed measured IP orientation response on reco sides)
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
sys.path.insert(0, str(HERE.parent / 'taukin_gap'))
sys.path.insert(0, str(HERE.parent / 'rho_pi0_marginal'))
sys.path.insert(0, str(HERE.parent / 'atlas_classical_posterior'))

import posterior_kin as PK  # noqa: E402
import posterior_marg as PM  # noqa: E402
import posterior as CP  # noqa: E402

G = CP.G
MEASURES = PK.MEASURES
M_TAU, M_PI0 = CP.M_TAU, 0.1349768


def rho_oracle_side_draws(rows, s, S, rng):
    """Generates hypothesis draws, supporting modular pi0 oracles on reco-rho sides."""
    R = len(rows)
    vis = G['vis'][rows, s]
    pm = np.linalg.norm(vis[:, :3], axis=-1)
    cap = G['shell_cap'][rows, s]
    mv2 = np.clip(vis[:, 3] ** 2 - pm ** 2, 0.0, cap ** 2)
    ev = np.sqrt(pm ** 2 + mv2)
    pS = np.broadcast_to(vis[:, None, :3], (R, S, 3))
    t, nu, ok, x, xmin, Et, P = PM.sample_given_vis(pS, np.broadcast_to(mv2[:, None], (R, S)),
                                                  np.broadcast_to(ev[:, None], (R, S)), rng)
    base = np.where(ok, 0.0, -np.inf)
    if G['e_prior'] == 'train':
        base = base + PM.interp_logrho(Et) - np.log(P) - 2 * np.log(x)
    if G['unpol']:
        base = base + CP.unpol_logw_side(rows, s, nu)
    neu = np.broadcast_to(G['neutral'][rows, s][:, None], (R, S, 4)).copy()
    visd = np.broadcast_to(vis[:, None], (R, S, 4)).copy()

    # Reco rho sides with marginalization or oracle
    mr = np.flatnonzero(G['marg'][rows, s])
    if len(mr):
        p1 = G['pions'][rows[mr], s, 0][:, None]  # (Rm,1,4)
        is_truth_rho = G['truth_rho_side'][rows[mr], s]

        # Determine pi0 momentum p2 and prior/proposal log weight lw2
        pi0_mode = G['rho_pi0_mode']
        if pi0_mode == 'truth':
            # Exact truth pi0 on truth-rho sides; fallback to marg for non-truth-rho
            p2_truth = G['truth_pi0'][rows[mr], s][:, None, :].copy()
            p2_marg, lw2_marg = (PM.draw_pi0_cond if G['cond'] is not None else PM.draw_pi0)(rows[mr], s, S, rng)
            p2 = np.where(is_truth_rho[:, None, None], p2_truth, p2_marg)
            lw2 = np.where(is_truth_rho[:, None], 0.0, lw2_marg)
        elif pi0_mode == 'truthdir':
            # Truth pi0 direction + reco pi0 energy
            p2_reco = G['neutral0'][rows[mr], s]
            p2_dir = CP.unit(G['truth_pi0'][rows[mr], s][..., :3])
            p2_mag = np.linalg.norm(p2_reco[..., :3], axis=-1, keepdims=True)
            p2_td = np.concatenate([p2_dir * p2_mag, p2_reco[..., 3:]], -1)[:, None, :].copy()
            p2_marg, lw2_marg = (PM.draw_pi0_cond if G['cond'] is not None else PM.draw_pi0)(rows[mr], s, S, rng)
            p2 = np.where(is_truth_rho[:, None, None], p2_td, p2_marg)
            lw2 = np.where(is_truth_rho[:, None], 0.0, lw2_marg)
        elif pi0_mode == 'truthe':
            # Reco pi0 direction + truth pi0 energy
            p2_reco = G['neutral0'][rows[mr], s]
            p2_dir = CP.unit(p2_reco[..., :3])
            E_true = G['truth_pi0'][rows[mr], s][..., 3:]
            p2_mag = np.sqrt(np.maximum(E_true ** 2 - M_PI0 ** 2, 0.0))
            p2_te = np.concatenate([p2_dir * p2_mag, E_true], -1)[:, None, :].copy()
            p2_marg, lw2_marg = (PM.draw_pi0_cond if G['cond'] is not None else PM.draw_pi0)(rows[mr], s, S, rng)
            p2 = np.where(is_truth_rho[:, None, None], p2_te, p2_marg)
            lw2 = np.where(is_truth_rho[:, None], 0.0, lw2_marg)
        elif pi0_mode == 'fixed':
            p2 = np.broadcast_to(G['neutral0'][rows[mr], s][:, None, :], (len(mr), S, 4))
            lw2 = np.zeros((len(mr), S))
        else:  # 'marg'
            p2, lw2 = (PM.draw_pi0_cond if G['cond'] is not None else PM.draw_pi0)(rows[mr], s, S, rng)

        V = p1 + p2
        pv = np.linalg.norm(V[..., :3], axis=-1)
        mv2m = np.maximum(V[..., 3] ** 2 - pv ** 2, 0.0)
        tm, num, okm, xm, xminm, Etm, Pm = PM.sample_given_vis(V[..., :3], mv2m, V[..., 3], rng)
        with np.errstate(divide='ignore', invalid='ignore'):
            lw = (lw2 + PM.interp_logrho(Etm) + np.log(V[..., 3]) - np.log(Pm) - np.log(pv) - 2 * np.log(xm)
                  + np.log(1 - xminm) + PM.rho_rate_log(p1, p2, num))
        lw = np.where(okm & np.isfinite(lw), lw, -np.inf)
        t[mr], nu[mr], base[mr] = tm, num, lw
        x[mr], Et[mr] = xm, Etm
        neu[mr], visd[mr] = p2, V
    return {'t': t, 'nu': nu, 'neu': neu, 'vis': visd, 'Et': Et}, base


def ip_truth_psi_logw_side(tdir, rows, s, kappa=2000.0):
    """Von Mises around the truth tau azimuth."""
    m = G['meas'][0]
    e1, e2 = m['e1'][rows, s], m['e2'][rows, s]
    psi_h = np.arctan2(np.sum(tdir * e2[:, None], -1), np.sum(tdir * e1[:, None], -1))
    ref = G['psi_truth'][rows, s]
    val = kappa * (np.cos(psi_h - ref[:, None]) - 1.0)
    return np.where(m['available'][rows, s][:, None], val, 0.0)


def run_block(args):
    start, stop, S, M, K, seed = args
    rng = np.random.default_rng(seed + start)
    rows = np.arange(start, stop)
    R = len(rows)
    D, logs = [], {}
    for s in (0, 1):
        d, base = rho_oracle_side_draws(rows, s, S, rng)
        D.append(d)

        # IP likelihood calculation
        # Baseline: fine / orient response
        ip_reco = sum(PK.ip_fine_logw_side(d['t'], rows, s, j) for j in range(3))
        if G['rho_psi_mode'] == 'truth':
            if G['rho_psi_scope'] == 'rho_only':
                target = G['truth_rho_side'][rows, s] & (G['modes'][rows, s] == 1)
            else:
                target = np.ones(R, dtype=bool)
            ip_truth = ip_truth_psi_logw_side(d['t'], rows, s, kappa=2000.0)
            ip = np.where(target[:, None], ip_truth, ip_reco)
        else:
            ip = ip_reco

        sv = CP.sv_logw_side(d['t'], rows, s)
        dl = PK.dl_logw_side(d, rows, s) if G['decay_length'] else 0.0
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
        sum_t = nu0[:, :, None, :2] + nu1[:, None, :, :2]
        lw = PK.met_logw(met[:, None, None] - sum_t, rows)
        if m == 'ip3_sv_mass':
            v0, v1 = g(0, 'vis'), g(1, 'vis')
            p0 = v0[:, :, None, :3] + nu0[:, :, None]
            p1 = v1[:, None, :, :3] + nu1[:, None, :]
            E0 = v0[:, :, None, 3] + np.linalg.norm(nu0[:, :, None], axis=-1)
            E1 = v1[:, None, :, 3] + np.linalg.norm(nu1[:, None, :], axis=-1)
            p_tot = p0 + p1
            E_tot = E0 + E1
            mh = np.sqrt(np.maximum(E_tot ** 2 - np.sum(p_tot ** 2, -1), 0.0))
            lw = lw + PK.mass_logw(mh)
        idx_pair, ess_pair = CP.systematic(lw.reshape(R, M * M), K, rng)
        a, b = idx_pair // M, idx_pair % M
        pick = lambda arr, p: np.take_along_axis(arr, p[..., None] if arr.ndim == 3 else p, 1)
        kneu0, kneu1 = pick(nu0, a), pick(nu1, b)
        nu_pair = np.stack([kneu0, kneu1], axis=2)
        h, valid = CP.polar(rows, nu_pair, avail)
        good = good_row[:, None] & valid & (idx_pair >= 0)
        hh = np.where(good[..., None], h[:, :, 0] * h[:, :, 1], 0.0)
        w_H = 1 + np.sum(hh * CP.CH, -1)
        w_Z = 1 + np.sum(hh * CP.CZ, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lr_pair = np.log(np.maximum(w_H, 1e-300)) - np.log(np.maximum(w_Z, 1e-300))
            lr = np.log(np.maximum(np.mean(np.where(good, w_H, 0.0), 1), 1e-300)) - \
                 np.log(np.maximum(np.mean(np.where(good, w_Z, 0.0), 1), 1e-300))
        rr = {'lr': np.where(good_row, lr, 0.0),
              'pair_lr': np.where(good, lr_pair, 0.0).mean(1),
              'Eh': np.where(good[..., None, None], h, 0.0).mean(1),
              'ess0': ess_side[0], 'ess1': ess_side[1], 'ess_pair': ess_pair}
        ab = (a, b)
        for s in (0, 1):
            Et = pick(g(s, 'Et'), ab[s])
            rr[f'meanEt_{s}'] = Et.mean(1)
            rr[f'pitEt_{s}'] = np.mean(Et < etau_true[:, s:s + 1], 1)
        res[m] = rr
    return start, res


def init(a):
    a.pi0 = 'reco'
    PK.init(a)
    V = G['V']
    ids = V['global_indices']
    # Load truth surface for truth pi0
    with np.load(CP.SURFACE) as d:
        assert np.array_equal(d['global_indices'], ids)
        tp0 = d['pi0']
    G['truth_pi0'] = tp0
    isr = (V['modes'] == 1) & (tp0[..., 3] > 0)
    G['truth_rho_side'] = isr
    G['rho_pi0_mode'] = a.rho_pi0
    G['rho_psi_mode'] = a.rho_psi
    G['rho_psi_scope'] = a.rho_psi_scope
    print(json.dumps({
        'rho_pi0_mode': a.rho_pi0,
        'rho_psi_mode': a.rho_psi,
        'rho_psi_scope': a.rho_psi_scope,
        'truth_rho_sides': int(isr.sum())
    }), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tables', type=Path, required=True, help='kin_tables.py output')
    ap.add_argument('--response', type=Path, required=True, help='rho_pi0_marginal response.py output')
    ap.add_argument('--rho-pi0', choices=('marg', 'truth', 'truthdir', 'truthe', 'fixed'), default='marg')
    ap.add_argument('--rho-psi', choices=('reco', 'truth'), default='reco')
    ap.add_argument('--rho-psi-scope', choices=('rho_only', 'all'), default='rho_only')
    ap.add_argument('--vis', choices=('tau', 'track'), default='track')
    ap.add_argument('--mass-prior', choices=('gauss', 'train'), default='gauss')
    ap.add_argument('--mass-sigma', type=float, default=3.0)
    ap.add_argument('--met-model', choices=('iso', 'ptbin', 'exact'), default='iso')
    ap.add_argument('--exact-sigma', type=float, default=2.0)
    ap.add_argument('--met-scale', type=float, default=1.0)
    ap.add_argument('--decay-length', action='store_true', default=True)
    ap.add_argument('--ip-orient', action='store_true', default=True)
    ap.add_argument('--ip-fine', action='store_true', default=True)
    ap.add_argument('--ip-aniso', action='store_true', default=False)
    ap.add_argument('--ip-vm', type=float, default=None)
    ap.add_argument('--ip-truth-psi', action='store_true', default=False)
    ap.add_argument('--ip-joint', action='store_true', default=False)
    ap.add_argument('--split', choices=('validation', 'train'), default='validation')
    ap.add_argument('--truth-train', type=Path, default=Path('/home/rbaba/rho-pi0-marginal-20261022/out/train_truth_rho.npz'))
    ap.add_argument('--S', type=int, default=16384)
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--K', type=int, default=512)
    ap.add_argument('--rows', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--block', type=int, default=50)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--seed', type=int, default=20261024)
    ap.add_argument('--response-cond', type=Path, default=None)
    ap.add_argument('--psi', action='store_true', default=False)
    ap.add_argument('--e-prior', choices=('invE', 'train'), default='invE')
    ap.add_argument('--x-prior', choices=('uniform', 'empirical'), default='uniform')
    ap.add_argument('--unpol-weight', action='store_true', default=True)
    ap.add_argument('--rho-shell-cap', action='store_true', default=True)
    ap.add_argument('--pi0-model', choices=('fixed', 'marg', 'truth'), default='marg')

    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    init(a)

    V = G['V']
    nall = len(V['event_numbers'])
    if a.split == 'train':
        sub = np.flatnonzero(V['event_numbers'] % 5 == 0)[::a.stride]
        sub = sub[:a.rows] if a.rows else sub
    elif a.stride > 1:
        sub = np.arange(0, nall, a.stride)
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
