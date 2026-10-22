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
        return -0.5 * np.sum(r ** 2, -1) / G['met_sigma'] ** 2
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
        ip = sum(CP.ip_logw_side(d['t'], G['meas'][j], rows, s, G['ip_resp'][j], 'psi') for j in range(3))
        sv = CP.sv_logw_side(d['t'], rows, s)
        geo = base + ip + sv + (dl_logw_side(d, rows, s) if G['decay_length'] else 0.0)
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
    ap.add_argument('--decay-length', action='store_true')
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
