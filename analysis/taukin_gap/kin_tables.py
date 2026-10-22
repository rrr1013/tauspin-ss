"""TRAIN tables for the tau-kinematics arms of the classical posterior (posterior_kin.py).

Events with event_number % 5 == 0 are held out of every table (they are the TRAIN PIT subset, as in rho_pi0_marginal).

  decay length, IP   p(log10 |IP| | log10 lam, prong class) of the leading core track, lam = (P/m_tau) c tau sin(alpha),
                     alpha = angle(truth tau, track); |IP| is the PV-referenced 3-D IP (p12 ip_measurements)
  decay length, SV   p(log10 L_SV | log10 (P/m_tau) c tau) for sides with a usable SV
  MET                per bin of the visible ditau pT (5 TRAIN quantile bins): median and RMS of reco MET - sum nu_T
                     along and across the visible ditau pT
  parent mass        TRAIN histogram of the truth m(tau tau) (truth visible + truth nu)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
import posterior as CP  # noqa: E402
P12 = CP.P12

CTAU_MM = 0.08703
LAM_EDGES = np.linspace(-1.0, 4.0, 41)      # log10 lam [um]
IP_EDGES = np.linspace(-1.0, 4.5, 56)       # log10 |IP| [um]
LB_EDGES = np.linspace(0.0, 5.0, 41)        # log10 (beta gamma c tau) [um]
SV_EDGES = np.linspace(-1.0, 5.5, 66)       # log10 L_SV [um]
M_EDGES = np.concatenate([np.arange(40, 90.5, 0.5), np.arange(90.5, 92.0, 0.02), np.arange(92.0, 140.5, 0.5)])


def cond_table(x, y, xe, ye, alpha=0.5):
    """log p(y | x) as (nx, ny) log density in y (per unit y), add-alpha smoothing, 1-2-1 smoothing along x."""
    H, _, _ = np.histogram2d(np.clip(x, xe[0], xe[-1] - 1e-9), np.clip(y, ye[0], ye[-1] - 1e-9), [xe, ye])
    n = H.sum(1)
    Hs = H.copy()
    Hs[1:-1] = 0.25 * H[:-2] + 0.5 * H[1:-1] + 0.25 * H[2:]
    Hs = Hs + alpha
    dens = Hs / Hs.sum(1, keepdims=True) / np.diff(ye)[None]
    return np.log(dens), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    with np.load(CP.LADDER / 'train.npz') as d:
        D = {k: d[k] for k in ('global_indices', 'event_numbers', 'truth_visible_tau_lab4', 'truth_neutrino_lab4',
                               'reco_visible_tau_lab4', 'reco_met_xy', 'reco_h_available_side', 'reco_h_mode', 'modes')}
    keep = D['event_numbers'] % 5 != 0
    ids = D['global_indices']
    tau = D['truth_visible_tau_lab4'] + D['truth_neutrino_lab4']
    P = np.linalg.norm(tau[..., :3], axis=-1)
    tdir = CP.unit(tau[..., :3])
    lb_um = P / CP.M_TAU * CTAU_MM * 1e3
    tracks = dict(np.load(CP.TRACKS))
    m = P12.ip_measurements(tracks, ids, 0)
    sina = np.linalg.norm(np.cross(tdir, m['tdir']), axis=-1)
    lam = lb_um * sina
    cls = (m['n_core'] >= 2).astype(int)
    ok = m['available'] & keep[:, None] & (m['mag_um'] > 0) & (lam > 0)
    out, info = {}, {}
    tabs = []
    for c in (0, 1):
        s = ok & (cls == c)
        t, n = cond_table(np.log10(lam[s]), np.log10(m['mag_um'][s]), LAM_EDGES, IP_EDGES)
        tabs.append(t)
        info[f'ip_sides_class{c}'] = int(s.sum())
        info[f'ip_rows_min_count_class{c}'] = int(n.min())
    out['ip_logp'] = np.stack(tabs)
    with np.load(P12.AUDIT) as g:
        sv_av, sv_len, sv_dir = g['sv_available'] > 0, g['sv_length'], g['sv_direction']
    sv_ok = (sv_av & (sv_len > 1e-6) & np.isfinite(sv_dir).all(-1))[ids] & keep[:, None]
    L = sv_len[ids] * 1e3
    t, n = cond_table(np.log10(lb_um[sv_ok]), np.log10(np.maximum(L[sv_ok], 1e-3)), LB_EDGES, SV_EDGES)
    out['sv_logp'] = t
    info['sv_sides'] = int(sv_ok.sum())
    # MET per visible ditau pT bin
    res = D['reco_met_xy'] - D['truth_neutrino_lab4'][:, :, :2].sum(1)
    vv = D['reco_visible_tau_lab4'][:, :, :2].sum(1)
    pt = np.linalg.norm(vv, axis=1)
    u = vv / pt[:, None]
    rpar = np.sum(res * u, 1)
    rperp = res[:, 0] * u[:, 1] - res[:, 1] * u[:, 0]
    k = keep
    pe = np.quantile(pt[k], np.linspace(0, 1, 6))
    pe[0], pe[-1] = 0.0, np.inf
    b = np.clip(np.searchsorted(pe, pt) - 1, 0, 4)
    med, spar, sperp = (np.array([f(r[k & (b == i)]) for i in range(5)]) for f, r in
                        ((np.median, rpar), (lambda x: np.sqrt(np.mean((x - np.median(x)) ** 2)), rpar),
                         (lambda x: np.sqrt(np.mean(x ** 2)), rperp)))
    out['met_pt_edges'], out['met_bias_par'], out['met_sig_par'], out['met_sig_perp'] = pe, med, spar, sperp
    info['met'] = {'pt_edges': pe.tolist(), 'bias_par': med.tolist(), 'sig_par': spar.tolist(), 'sig_perp': sperp.tolist(),
                   'iso_rms': float(np.sqrt(np.mean(res[k] ** 2)))}
    # truth m(tau tau)
    tt = tau.sum(1)
    mtt = np.sqrt(np.maximum(tt[:, 3] ** 2 - np.sum(tt[:, :3] ** 2, -1), 0))[k]
    h, _ = np.histogram(np.clip(mtt, M_EDGES[0], M_EDGES[-1] - 1e-9), M_EDGES)
    out['m_edges'], out['m_logdens'] = M_EDGES, np.log((h + 0.5) / (h + 0.5).sum() / np.diff(M_EDGES))
    info['mass_frac_within_0p02'] = float(np.mean(np.abs(mtt - 91.1876) < 0.02))
    info['mass_frac_within_1'] = float(np.mean(np.abs(mtt - 91.1876) < 1))
    for kk, v in (('LAM_EDGES', LAM_EDGES), ('IP_EDGES', IP_EDGES), ('LB_EDGES', LB_EDGES), ('SV_EDGES', SV_EDGES)):
        out[kk] = v
    np.savez_compressed(a.output, **out)
    a.output.with_suffix('.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
