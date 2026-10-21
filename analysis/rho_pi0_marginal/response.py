"""TRAIN-only ingredients of the pi0-marginalised classical posterior.

1. pi0 likelihood  L(reco pi0, reco raw mode | true pi0), as a density in
       u = log(E_reco / E_true),   l = log(angle(reco pi0, true pi0)),
   tabulated per reco raw mode (1 = 1p1n, 2 = 1p2n) in bins of the TRUE pi0 energy, and normalised to ALL truth-rho
   sides of that true-energy bin (so P(reco rho with this raw mode | E_true) is included).
2. proposal  q(u, l | reco raw mode, reco pi0 energy bin): the same histogram binned by the RECO energy, normalised
   per (raw, reco bin) to a probability over cells.  Draws are uniform inside a cell.
3. tau energy prior: density of the truth tau energy (all sides, all modes) per unit energy, from a histogram in
   log E.  This is the post-selection TRAIN distribution (see the run note for what that implies).

The rows with event_number % 5 == 0 are held out of every table (they are the TRAIN PIT subset).
"""
import argparse
import json
from pathlib import Path

import numpy as np

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
U_EDGES = np.linspace(-2.5, 2.0, 46)          # 0.1 wide
L_EDGES = np.linspace(np.log(1e-4), np.log(0.3), 33)   # 0.25 wide
N_EBINS = 12
N_ETAU = 80


def smooth(c):
    """1-2-1 smoothing in both cell axes (last two), counts preserved up to the edges."""
    k = np.array([0.25, 0.5, 0.25])
    for ax in (-2, -1):
        c = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode='edge'), k, mode='valid'), ax, c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', type=Path, required=True, help='p0_train_truth.py output')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    T = np.load(a.truth)
    with np.load(LADDER / 'train.npz') as d:
        L = {k: d[k] for k in ('global_indices', 'event_numbers', 'reco_h_neutral_lab4', 'reco_h_mode',
                               'reco_h_available_side', 'reco_mode_raw', 'modes', 'truth_visible_tau_lab4',
                               'truth_neutrino_lab4')}
    pos = {int(g): i for i, g in enumerate(L['global_indices'])}
    lr = np.array([pos[int(g)] for g in T['global_indices']])
    assert np.array_equal(L['modes'][lr], T['modes'])
    keep_ev = (L['event_numbers'][lr] % 5) != 0
    side = (T['modes'] == 1) & keep_ev[:, None]
    p2t = T['pi0'][side]
    Et = p2t[:, 3]
    reco_rho = (L['reco_h_available_side'][lr] & (L['reco_h_mode'][lr] == 1))[side]
    raw = L['reco_mode_raw'][lr][side]
    rn = L['reco_h_neutral_lab4'][lr][side]
    Er = rn[:, 3]
    cosa = np.sum(rn[:, :3] * p2t[:, :3], -1) / np.maximum(np.linalg.norm(rn[:, :3], axis=-1) * np.linalg.norm(p2t[:, :3], axis=-1), 1e-300)
    ang = np.arccos(np.clip(cosa, -1, 1))
    u = np.log(np.maximum(Er, 1e-6) / Et)
    l = np.log(np.maximum(ang, 1e-12))
    eedges = np.quantile(Et, np.linspace(0, 1, N_EBINS + 1))
    eb = np.clip(np.searchsorted(eedges, Et) - 1, 0, N_EBINS - 1)
    ecent = np.array([np.median(np.log(Et[eb == i])) for i in range(N_EBINS)])
    NE = np.bincount(eb, minlength=N_EBINS).astype(float)
    du, dl = np.diff(U_EDGES)[0], np.diff(L_EDGES)[0]
    inr = (u >= U_EDGES[0]) & (u < U_EDGES[-1]) & (l >= L_EDGES[0]) & (l < L_EDGES[-1])
    iu = np.clip(np.floor((u - U_EDGES[0]) / du).astype(int), 0, len(U_EDGES) - 2)
    il = np.clip(np.floor((l - L_EDGES[0]) / dl).astype(int), 0, len(L_EDGES) - 2)
    shape = (2, N_EBINS, len(U_EDGES) - 1, len(L_EDGES) - 1)
    cnt = np.zeros(shape)
    sel = reco_rho & inr & ((raw == 1) | (raw == 2))
    np.add.at(cnt, (raw[sel] - 1, eb[sel], iu[sel], il[sel]), 1.0)
    cs = smooth(cnt) + 0.05                                   # floor: no zero-likelihood cells
    logL = np.log(cs / (NE[None, :, None, None] * du * dl))   # density in (u, l), per true-E bin
    # proposal, binned by reco energy
    sel_r = reco_rho & inr & ((raw == 1) | (raw == 2))
    redges = np.quantile(Er[sel_r], np.linspace(0, 1, N_EBINS + 1))
    redges[0], redges[-1] = 0.0, np.inf
    rb = np.clip(np.searchsorted(redges, Er) - 1, 0, N_EBINS - 1)
    cq = np.zeros(shape)
    np.add.at(cq, (raw[sel_r] - 1, rb[sel_r], iu[sel_r], il[sel_r]), 1.0)
    cq = smooth(cq) + 0.05
    pq = cq / cq.sum((-2, -1), keepdims=True)                 # probability per cell
    # tau energy prior (all sides, all modes, held-out events excluded)
    keep_all = (L['event_numbers'] % 5) != 0
    Etau = (L['truth_visible_tau_lab4'][..., 3] + L['truth_neutrino_lab4'][..., 3])[keep_all].ravel()
    ledges = np.linspace(np.log(Etau.min()) - 0.01, np.log(Etau.max()) + 0.01, N_ETAU + 1)
    h, _ = np.histogram(np.log(Etau), ledges)
    h = np.convolve(np.pad(h.astype(float), 1, mode='edge'), [0.25, 0.5, 0.25], mode='valid') + 0.5
    lc = 0.5 * (ledges[1:] + ledges[:-1])
    logrho = np.log(h / h.sum() / np.diff(ledges)) - lc       # density per unit E = density per log E / E
    np.savez(a.output, u_edges=U_EDGES, l_edges=L_EDGES, e_edges=eedges, e_logcent=ecent, logL=logL,
             r_edges=redges, pq=pq, etau_logcent=lc, etau_logrho=logrho)
    diag = {'truth_rho_sides': int(side.sum()), 'reco_rho_frac': float(reco_rho.mean()),
            'in_range_frac_of_reco_rho': float(inr[reco_rho].mean()),
            'raw2_frac_of_reco_rho': float((raw[reco_rho] == 2).mean()),
            'e_edges': eedges.tolist(), 'r_edges': redges.tolist(),
            'per_Ebin': [{'n': int(NE[i]), 'reco_rho_eff': float(reco_rho[eb == i].mean()),
                          'u_q16_50_84': np.quantile(u[(eb == i) & reco_rho], [.16, .5, .84]).round(3).tolist(),
                          'ang_q16_50_84_mrad': (1e3 * np.quantile(ang[(eb == i) & reco_rho], [.16, .5, .84])).round(2).tolist(),
                          'corr_u_l': float(np.corrcoef(u[(eb == i) & sel], l[(eb == i) & sel])[0, 1])}
                         for i in range(N_EBINS)],
            'etau_quantiles': np.quantile(Etau, [.05, .25, .5, .75, .95]).round(2).tolist()}
    Path(str(a.output) + '.json').write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))


if __name__ == '__main__':
    main()
