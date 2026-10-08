"""Can Z -> tau tau calibrate the transverse spin response that H -> tau tau needs?

Transverse spin correlations decompose under rotations about the tau axis k.  With
z = h_n + i h_r per side:
  m = 0 :  conj(z-) z+  ->  T = h-_n h+_n + h-_r h+_r   (CP-even, the Higgs: C_nn = C_rr = 1)
                            A = h-_n h+_r - h-_r h+_n   (CP-odd, the Higgs CP angle)
  m = 2 :  z- z+        ->  Q1 = h-_n h+_n - h-_r h+_r, Q2 = h-_n h+_r + h-_r h+_n
The Z (and the SM ditau state of any spin-1 source) has a traceless transverse block:
only m = 2.  So Z data can calibrate only the m = 2 response; the Higgs needs m = 0.

Response of an estimator hhat to a spin parameter c_j (f = 1 + c_j S_j + ...):
  r_aj = d E[O_a] / d c_j = Cov_flat(O_a(hhat), S_j(h_true)).
"Flat" = spin-flat world, built from the pooled H + Z (v2) validation cohort with weight
w0 / (pi_H f_H + pi_Z f_Z) (bounded below, unlike 1/f_H alone).  A different 3pi
hadronic current in nature ("CLEO world") multiplies the flat weight by the Dalitz
part u and replaces h_true by h_CLEO on 3pi sides (analysis/a1_current_model).

Question: under the current-model change, does the Z-measurable m = 2 response move by
the same factor as the m = 0 response?  If yes, an in-situ Z calibration absorbs it.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a1_current_model'))
import analyze as an  # noqa: E402
import reweight as rw  # noqa: E402

BOOT = 300
COMP = ('T', 'A', 'Q1', 'Q2')


def comps(h):
    """(N, 4) transverse structures + kk, from (N, 2, 3) canonical h (n, r, k)."""
    a, b = h[:, 0], h[:, 1]
    T = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    A = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    Q1 = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]
    Q2 = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]
    return np.stack([T, A, Q1, Q2, a[:, 2] * b[:, 2]], 1)


def wcov_mat(X, S, w):
    """Cov_w(X_a, S_j) as (a, j) matrix."""
    w = w / w.sum()
    Xc = X - w @ X
    Sc = S - w @ S
    return (Xc * w[:, None]).T @ Sc


def flat_weight(d):
    y, w0, h = d['y'], d['w0'], d['h_gen']
    piH = w0[y == 1].sum() / w0.sum()
    fH = rw.spin_factor(h, np.ones_like(y))
    fZ = rw.spin_factor(h, np.zeros_like(y))
    return w0 / (piH * fH + (1 - piH) * fZ)


def responses(est, S_world, w, idx):
    """r[name][world] = 5x5 Cov(O_a(hhat), S_j(h_world))."""
    out = {}
    for name, X in est.items():
        out[name] = {wn: wcov_mat(X[idx], S_world[wn][idx], w[wn][idx]) for wn in S_world}
    return out


def summarize(r):
    """Diagonal responses, single ratios (cleo/gen) and Z-calibrated double ratios."""
    g, c = r['gen'], r['cleo']
    dg, dc = np.diag(g), np.diag(c)
    rho = dc / dg
    qg = 0.5 * (dg[2] + dg[3])
    qc = 0.5 * (dc[2] + dc[3])
    rq = qc / qg
    return {
        'diag_gen': dict(zip(COMP + ('kk',), dg.tolist())),
        'single_ratio': dict(zip(COMP + ('kk',), rho.tolist())),
        'mc_transfer_T_over_Q': float(dg[0] / qg), 'mc_transfer_A_over_Q': float(dg[1] / qg),
        'double_T_over_Q': float(rho[0] / rq), 'double_A_over_Q': float(rho[1] / rq),
        'double_T_over_Q1': float(rho[0] / rho[2]), 'double_A_over_Q2': float(rho[1] / rho[3]),
        'cross_T_from_Q1_gen': float(g[0, 2] / dg[2]), 'cross_Q1_from_T_gen': float(g[2, 0] / dg[0]),
    }


def main():
    d, nets = an.load_all()
    wf = flat_weight(d)
    # Dalitz-only part of the CLEO world (spin handled by the flat construction)
    w = {'gen': wf, 'cleo': wf * np.prod(d['u'], 1)}
    S = {'gen': comps(d['h_gen']), 'cleo': comps(d['h_cleo'])}
    est = {'exact h_gen': comps(d['h_gen']), 'exact h_CLEO': comps(d['h_cleo'])}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v:
            est[f'reco {arm} / {teacher} teacher'] = comps(v['h_pred'])
    modes = d['modes']
    scopes = {'any 3pi': (modes == 3).any(1), '3pi x 3pi': (modes == 3).all(1),
              'no 3pi (control)': ~(modes == 3).any(1), 'all events': np.ones(len(modes), bool)}
    rng = np.random.default_rng(11)
    keys = ('double_T_over_Q', 'double_A_over_Q', 'double_T_over_Q1', 'double_A_over_Q2',
            'mc_transfer_T_over_Q', 'mc_transfer_A_over_Q')
    res = {'flat_weight': {'min': float(wf.min()), 'max': float(wf.max()),
                           'ess_frac': float(wf.sum() ** 2 / (wf ** 2).sum() / len(wf))}, 'scopes': {}}
    for sc, m in scopes.items():
        idx = np.flatnonzero(m)
        point = responses(est, S, w, idx)
        boots = {n: [] for n in est}
        for _ in range(BOOT):
            b = rng.choice(idx, len(idx))
            rb = responses(est, S, w, b)
            for n in est:
                s = summarize(rb[n])
                boots[n].append([s['single_ratio']['T'], s['single_ratio']['A'], s['single_ratio']['Q1'],
                                 s['single_ratio']['Q2']] + [s[k] for k in keys])
        res['scopes'][sc] = {'events': int(len(idx))}
        for n in est:
            s = summarize(point[n])
            bb = np.array(boots[n])
            names = ['single_T', 'single_A', 'single_Q1', 'single_Q2'] + list(keys)
            s['ci95'] = {k: np.nanquantile(bb[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(names)}
            res['scopes'][sc][n] = s
        print(f'== {sc} ({len(idx)} events)')
        for n in est:
            s = res['scopes'][sc][n]
            sr = s['single_ratio']
            print(f'  {n:34s} single T {sr["T"]:.3f} A {sr["A"]:.3f} Q1 {sr["Q1"]:.3f} Q2 {sr["Q2"]:.3f} kk {sr["kk"]:.3f} | '
                  f'double T/Q {s["double_T_over_Q"]:.3f} {np.round(s["ci95"]["double_T_over_Q"], 3)} '
                  f'A/Q {s["double_A_over_Q"]:.3f} | MC T/Q {s["mc_transfer_T_over_Q"]:.3f} A/Q {s["mc_transfer_A_over_Q"]:.3f}',
                  flush=True)
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    (out / 'transfer_a1.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
