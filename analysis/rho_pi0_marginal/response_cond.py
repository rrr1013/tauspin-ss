"""TRAIN pi0 response conditioned on the true pi+- / pi0 geometry and the reco PFO mass (stage 2).

The energy and angular response of the reco pi0 depends strongly on how close the pi0 is to the charged pion
(shower overlap) and on the reco PFO mass (merged clusters).  The likelihood of the observation
(raw mode, PFO-mass class, u, l) given the TRUE pi0 is factorised as

    P(raw, mclass | E2, sep) * p(u | raw, mclass, E2, sep) * p(l | raw, mclass, E2, sep)

with sep = angle(pi+-, pi0) * E2 [GeV rad] evaluated on the true pi0 (pi+- direction from truth; at use time the reco
track), u = log(E_reco/E2), l = log(angle(reco, true)).  P(...) is normalised to ALL truth-rho sides of the (E2, sep)
cell, so the probability to be reconstructed as rho with this raw mode / mass class is included.  u-l correlation
is |r| < 0.1 (response.py), hence the factorisation.  1-D densities are shrunk toward the shape pooled over sep and
mass class (pseudo-count 20) and 1-2-1 smoothed.

Proposal: p(u | raw, mclass, E_reco bin, sep_reco bin) * p(l | same), sep_reco from the reco pi0 and reco track,
built the same way (independent u and l draws).

Stage 3 (--psi): the azimuth psi of the reco pi0 around the TRUE pi0 direction, measured from the axis pointing from
the charged track to the true pi0 (psi = 0: reco displaced away from the track, pi: toward it), is not uniform: the
reco pi0 is pulled toward the track when the two overlap.  p(psi | raw, mclass, E2, sep, l-bin) in 8 bins (l in 4 bins)
is stored as logpsi (log of 2 pi x density, i.e. 0 for isotropy); the sampler keeps psi uniform in the proposal and
adds logpsi to the weight.

Rows with event_number % 5 == 0 are held out (TRAIN PIT subset), as in response.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
U_EDGES = np.linspace(-2.5, 2.0, 46)
L_EDGES = np.linspace(np.log(1e-4), np.log(0.3), 33)
NE, NS = 8, 5
M_CUT = 0.3
K_SHRINK = 20.0
N_PSI = 8
L_PSI_EDGES = np.log(np.array([1e-4, 3e-3, 8e-3, 2e-2, 0.3]))


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def smooth1(c):
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode='edge'), [0.25, 0.5, 0.25], mode='valid'), -1, c)


def dens1d(x, edges, keys, shape, pooled_keys, pooled_shape):
    """log density tables (shape + (nbins,)) of x with shrinkage to the pooled shape."""
    nb = len(edges) - 1
    i = np.clip(np.floor((x - edges[0]) / (edges[1] - edges[0])).astype(int), 0, nb - 1)
    c = np.zeros(shape + (nb,))
    np.add.at(c, keys + (i,), 1.0)
    cp = np.zeros(pooled_shape + (nb,))
    np.add.at(cp, pooled_keys + (i,), 1.0)
    cp = smooth1(cp) + 0.5
    shape_p = cp / cp.sum(-1, keepdims=True)
    # broadcast pooled (raw, Ebin) shape onto (raw, mclass, Ebin, sbin)
    sp = shape_p[:, None, :, None, :]
    c = smooth1(c) + K_SHRINK * sp
    return np.log(c / c.sum(-1, keepdims=True) / (edges[1] - edges[0]))


def bins(x, edges):
    return np.clip(np.searchsorted(edges, x) - 1, 0, len(edges) - 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    T = np.load(a.truth)
    with np.load(LADDER / 'train.npz') as d:
        L = {k: d[k] for k in ('global_indices', 'event_numbers', 'reco_h_neutral_lab4', 'reco_h_mode', 'reco_h_pions_lab4',
                               'reco_h_available_side', 'reco_mode_raw', 'modes')}
    pos = {int(g): i for i, g in enumerate(L['global_indices'])}
    lr = np.array([pos[int(g)] for g in T['global_indices']])
    side = (T['modes'] == 1) & ((L['event_numbers'][lr] % 5) != 0)[:, None]
    p2, p1 = T['pi0'][side], T['pi_lead'][side]
    E2 = p2[:, 3]
    sep = np.arccos(np.clip(np.sum(unit(p1[:, :3]) * unit(p2[:, :3]), -1), -1, 1)) * E2
    rr = (L['reco_h_available_side'][lr] & (L['reco_h_mode'][lr] == 1))[side]
    raw = L['reco_mode_raw'][lr][side]
    rr &= (raw == 1) | (raw == 2)
    rn = L['reco_h_neutral_lab4'][lr][side]
    p1r = L['reco_h_pions_lab4'][lr][side][:, 0]
    mpfo = np.sqrt(np.maximum(rn[:, 3] ** 2 - np.sum(rn[:, :3] ** 2, -1), 0))
    mc = (mpfo > M_CUT).astype(int)
    u = np.log(np.maximum(rn[:, 3], 1e-6) / E2)
    l = np.log(np.maximum(np.arccos(np.clip(np.sum(unit(rn[:, :3]) * unit(p2[:, :3]), -1), -1, 1)), 1e-12))
    eedges = np.quantile(E2, np.linspace(0, 1, NE + 1))
    sedges = np.quantile(sep, np.linspace(0, 1, NS + 1))
    eb, sb = bins(E2, eedges), bins(sep, sedges)
    ecent = np.array([np.median(np.log(E2[eb == i])) for i in range(NE)])
    scent = np.array([np.median(np.log(sep[sb == i])) for i in range(NS)])
    # categorical: P(raw, mclass | Ebin, sbin), over all truth-rho sides
    N = np.zeros((NE, NS))
    np.add.at(N, (eb, sb), 1.0)
    cat = np.zeros((2, 2, NE, NS))
    r = rr
    np.add.at(cat, (raw[r] - 1, mc[r], eb[r], sb[r]), 1.0)
    logcat = np.log((cat + 0.5) / (N[None, None] + 2.0))
    inr = r & (u >= U_EDGES[0]) & (u < U_EDGES[-1]) & (l >= L_EDGES[0]) & (l < L_EDGES[-1])
    keys = (raw[inr] - 1, mc[inr], eb[inr], sb[inr])
    pk = (raw[inr] - 1, eb[inr])
    logpu = dens1d(u[inr], U_EDGES, keys, (2, 2, NE, NS), pk, (2, NE))
    logpl = dens1d(l[inr], L_EDGES, keys, (2, 2, NE, NS), pk, (2, NE))
    # stage 3: psi of the reco pi0 around the true pi0, axis from the (truth) track to the true pi0
    t, n, c = unit(p2[:, :3]), unit(rn[:, :3]), unit(p1[:, :3])
    ax_a = unit((t - c) - np.sum((t - c) * t, -1, keepdims=True) * t)
    ax_b = np.cross(t, ax_a)
    dv = n - np.sum(n * t, -1, keepdims=True) * t
    psi = np.arctan2(np.sum(dv * ax_b, -1), np.sum(dv * ax_a, -1))
    ip = np.clip(np.floor((psi + np.pi) / (2 * np.pi) * N_PSI).astype(int), 0, N_PSI - 1)
    lb = bins(np.clip(l, L_PSI_EDGES[0], L_PSI_EDGES[-1] - 1e-9), L_PSI_EDGES)
    cps = np.zeros((2, 2, NE, NS, len(L_PSI_EDGES) - 1, N_PSI))
    np.add.at(cps, (raw[inr] - 1, mc[inr], eb[inr], sb[inr], lb[inr], ip[inr]), 1.0)
    cps = cps + 5.0 / N_PSI                                    # shrink toward isotropy (pseudo-count 5)
    logpsi = np.log(cps / cps.sum(-1, keepdims=True) * N_PSI)
    # proposal on reco quantities
    Er = rn[:, 3]
    sepr = np.arccos(np.clip(np.sum(unit(p1r[:, :3]) * unit(rn[:, :3]), -1), -1, 1)) * Er
    redges = np.quantile(Er[inr], np.linspace(0, 1, NE + 1))
    redges[0], redges[-1] = 0.0, np.inf
    sredges = np.quantile(sepr[inr], np.linspace(0, 1, NS + 1))
    sredges[0], sredges[-1] = 0.0, np.inf
    qk = (raw[inr] - 1, mc[inr], bins(Er[inr], redges), bins(sepr[inr], sredges))
    qpk = (raw[inr] - 1, bins(Er[inr], redges))
    logqu = dens1d(u[inr], U_EDGES, qk, (2, 2, NE, NS), qpk, (2, NE))
    logql = dens1d(l[inr], L_EDGES, qk, (2, 2, NE, NS), qpk, (2, NE))
    np.savez(a.output, u_edges=U_EDGES, l_edges=L_EDGES, e_logcent=ecent, s_logcent=scent, logcat=logcat,
             logpu=logpu, logpl=logpl, r_edges=redges, sr_edges=sredges, logqu=logqu, logql=logql, m_cut=M_CUT,
             logpsi=logpsi, l_psi_edges=L_PSI_EDGES)
    hw = lambda x: float(np.diff(np.quantile(x, [.16, .84]))[0] / 2)
    diag = {'sides': int(side.sum()), 'e_edges': eedges.round(2).tolist(), 'sep_edges': sedges.round(3).tolist(),
            'reco_rho_eff_by_E_sep': (cat.sum((0, 1)) / N).round(3).tolist(),
            'mclass1_frac': float(mc[r].mean()),
            'u_half68_by_sep_in_Ebin3': [hw(u[r & (eb == 3) & (sb == j)]) for j in range(NS)],
            'psi_frac_8bins_all (bin0,7 = toward track)': (np.bincount(ip[inr], minlength=N_PSI) / inr.sum()).round(3).tolist(),
            'psi_toward_track_frac_by_sep (|psi| > 3pi/4)': [float(np.mean(np.abs(psi[inr & (sb == j)]) > 0.75 * np.pi)) for j in range(NS)],
            'ang_median_mrad_by_sep_in_Ebin3': [float(1e3 * np.exp(np.median(l[r & (eb == 3) & (sb == j)]))) for j in range(NS)]}
    Path(str(a.output) + '.json').write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))


if __name__ == '__main__':
    main()
