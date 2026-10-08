"""MC transfer factor between the m = 0 (Higgs) and m = 2 (Z) transverse responses,
by decay-mode pair and in bins of the true pair pT and the softer tau energy (ATLAS full-reco
validation cohort, generator world).  If R_T / R_Q stays ~1 across kinematics, the Z can be
used in bins matched to the Higgs without a kinematics-dependent structural correction.
Same flat-world construction as transfer_a1.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a1_current_model'))
import analyze as an  # noqa: E402
import reweight as rw  # noqa: E402
from transfer_a1 import comps, flat_weight, wcov_mat  # noqa: E402

BOOT = 200
MODES = {0: 'pi', 1: 'rho', 2: 'rho', 3: '3pi'}  # truth modes in the cohort: 0 pi, 1 rho(1pi0), 2 rho(2pi0)?, 3 3pi


def ratios(X, S, w):
    d = np.diag(wcov_mat(X, S, w))
    q = 0.5 * (d[2] + d[3])
    return np.array([d[0] / q, d[1] / q, d[1] / d[0]])


def main():
    d, nets = an.load_all()
    s = np.load(rw.COHORT / 'truth_surface_cleo.npz')
    tau = s['pions'].sum(2) + s['pi0'] + s['nu4']  # (N, 2, 4) lab (px, py, pz, E)
    pair = tau.sum(1)
    ptt = np.hypot(pair[:, 0], pair[:, 1])
    emin = tau[:, :, 3].min(1)
    wf = flat_weight(d)
    S = comps(d['h_gen'])
    est = {'exact h_gen': S}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v and teacher == 'gen':
            est[f'reco {arm}'] = comps(v['h_pred'])
    modes = d['modes']
    bins = {'all': np.ones(len(wf), bool)}
    for a, b in ((0, 0), (0, 1), (1, 1), (0, 3), (1, 3), (3, 3)):
        bins[f'mode {a}x{b}'] = ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 1e9)):
        bins[f'pT(tautau) {lo}-{hi:g}'] = (ptt >= lo) & (ptt < hi)
    q = np.quantile(emin, [0, .25, .5, .75, 1])
    for i in range(4):
        bins[f'E_min q{i + 1} [{q[i]:.0f},{q[i + 1]:.0f}] GeV'] = (emin >= q[i]) & (emin <= q[i + 1])
    rng = np.random.default_rng(5)
    res = {}
    for bn, m in bins.items():
        idx = np.flatnonzero(m)
        res[bn] = {'events': int(len(idx))}
        for n, X in est.items():
            p = ratios(X[idx], S[idx], wf[idx])
            bb = np.array([ratios(X[b], S[b], wf[b]) for b in (rng.choice(idx, len(idx)) for _ in range(BOOT))])
            ci = np.quantile(bb, [0.16, 0.84], axis=0)
            res[bn][n] = {'T_over_Q': p[0], 'A_over_Q': p[1], 'A_over_T': p[2],
                          'ci68': {'T_over_Q': ci[:, 0].tolist(), 'A_over_Q': ci[:, 1].tolist(), 'A_over_T': ci[:, 2].tolist()}}
        print(f'{bn:32s} n={len(idx):6d} ' + '  '.join(
            f"{n}: T/Q {res[bn][n]['T_over_Q']:.3f}±{np.diff(res[bn][n]['ci68']['T_over_Q'])[0] / 2:.3f} "
            f"A/T {res[bn][n]['A_over_T']:.3f}" for n in est if n != 'exact h_gen'), flush=True)
    (HERE / 'results' / 'transfer_kin.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
