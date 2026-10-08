"""Detector-calibration shifts: does the Z-measurable m = 2 response move like the m = 0
response the Higgs needs?

Input: calibration-shifted predictions of the fixed 9/30 point-h networks (arm none / local)
on the ATLAS validation cohort (analysis/calibration_response/evaluate_remote.py; sagitta
variants dropped), joined to the cohort weights.  For each variant v and structure a:
  r_a(v) = Cov_flat(O_a(hhat_v), S_a(h_exact)),   single ratio r_a(v) / r_a(identity),
  Z-calibrated residual  [r_T(v)/r_T(0)] / [r_Q(v)/r_Q(0)]  (Q = mean of Q1, Q2), same for A.
Predicted by the per-tau factorisation argument: per-object shifts (pi0, track scale) leave
the double ratio at 1; event-level (MET) shifts need not.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a1_current_model'))
import reweight as rw  # noqa: E402
from transfer_a1 import comps, wcov_mat  # noqa: E402

BOOT = 300


def summary(X0, Xv, S, w):
    d0 = np.diag(wcov_mat(X0, S, w))
    dv = np.diag(wcov_mat(Xv, S, w))
    rho = dv / d0
    rq = 0.5 * (dv[2] + dv[3]) / (0.5 * (d0[2] + d0[3]))
    return np.array([rho[0], rho[1], rho[2], rho[3], rho[4], rho[0] / rq, rho[1] / rq])


NAMES = ('single_T', 'single_A', 'single_Q1', 'single_Q2', 'single_kk', 'double_T_over_Q', 'double_A_over_Q')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    args = ap.parse_args()
    s = np.load(rw.COHORT / 'truth_surface_cleo.npz')
    pos = {g: i for i, g in enumerate(s['global_indices'])}
    rng = np.random.default_rng(3)
    out = {}
    for f in args.inputs:
        z = np.load(f)
        arm = Path(f).stem.replace('final_', '')
        idx = np.array([pos[g] for g in z['global_indices']])
        y = z['labels'].astype(int)
        assert np.array_equal(y, s['labels'][idx].astype(int))
        w0 = s['weights'][idx].astype(float)
        h = z['h_exact'].astype(float)
        piH = w0[y == 1].sum() / w0.sum()
        wf = w0 / (piH * rw.spin_factor(h, np.ones_like(y)) + (1 - piH) * rw.spin_factor(h, np.zeros_like(y)))
        S = comps(h)
        H = z['h'].astype(float)  # (n, V, 2, 3)
        kinds, eps, metm = z['variant_kind'], z['variant_eps'], z['variant_met']
        X0 = comps(H[:, 0])
        modes = z['truth_modes']
        scopes = {'all': np.ones(len(y), bool), 'with rho': (modes == 1).any(1), 'with 3pi': (modes == 3).any(1)}
        out[arm] = {}
        for v in range(1, H.shape[1]):
            name = f'{kinds[v]} {eps[v]:+.2f} ({metm[v]})'
            Xv = comps(H[:, v])
            row = {}
            for sc, m in scopes.items():
                ii = np.flatnonzero(m)
                p = summary(X0[ii], Xv[ii], S[ii], wf[ii])
                bb = np.array([summary(X0[b], Xv[b], S[b], wf[b]) for b in (rng.choice(ii, len(ii)) for _ in range(BOOT))])
                ci = np.quantile(bb, [0.025, 0.975], axis=0)
                row[sc] = {k: {'value': float(p[i]), 'ci95': ci[:, i].tolist()} for i, k in enumerate(NAMES)}
            out[arm][name] = row
            a = row['all']
            print(f"{arm:5s} {name:28s} single T {a['single_T']['value']:.4f} Q1 {a['single_Q1']['value']:.4f} "
                  f"Q2 {a['single_Q2']['value']:.4f} kk {a['single_kk']['value']:.4f} | double T/Q "
                  f"{a['double_T_over_Q']['value']:.4f} [{a['double_T_over_Q']['ci95'][0]:.4f},{a['double_T_over_Q']['ci95'][1]:.4f}] "
                  f"A/Q {a['double_A_over_Q']['value']:.4f}", flush=True)
    (HERE / 'results' / 'transfer_calib.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
