"""Review follow-up: the 3pi-current transfer with the response a Z calibration actually
measures, i.e. the event-dependent Z tensor instead of the equal average of Q1, Q2.

Per event, C_Z (2x2 transverse block in the canonical (n, r) basis) is built from the truth
tau directions with the CS-frame tensor model of z_state.py.  Z-like statistic
  O_Z = sum_ij C_Z,ij hhat-_i hhat+_j,   S_Z = same with h_true,   r_Z = Cov_flat(O_Z, S_Z),
and the Higgs needs r_T as before.  Double ratio rho_T / rho_Z between the CLEO and the
generator 3pi-current worlds; also the nominal ratio r_T / r_Z normalised by the exact-h value.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'a1_current_model'))
import analyze as an  # noqa: E402
import reweight as rw  # noqa: E402
from transfer_a1 import flat_weight  # noqa: E402
from z_state import frame_quantities, z_transverse  # noqa: E402

BOOT = 300


def wcov(x, s, w):
    w = w / w.sum()
    return float(w @ ((x - w @ x) * (s - w @ s)))


def stats(h, Cz):
    pp = h[:, 0, :2, None] * h[:, 1, None, :2]
    return pp[:, 0, 0] + pp[:, 1, 1], np.einsum('nab,nab->n', Cz, pp)


def summary(est, S, Cz, w, idx):
    out = {}
    for name, X in est.items():
        r = {}
        for wn in ('gen', 'cleo'):
            T, Z = stats(X[idx], Cz[idx])
            Ts, Zs = stats(S[wn][idx], Cz[idx])
            r[wn] = (wcov(T, Ts, w[wn][idx]), wcov(Z, Zs, w[wn][idx]))
        out[name] = (r['cleo'][0] / r['gen'][0]) / (r['cleo'][1] / r['gen'][1]), r['gen'][0] / r['gen'][1], r['cleo'][0] / r['gen'][0]
    return out


def main():
    d, nets = an.load_all()
    s = np.load(rw.COHORT / 'truth_surface_cleo.npz')
    tau = s['pions'].sum(2) + s['pi0'] + s['nu4']
    Cz, _ = z_transverse(frame_quantities(tau[:, 0], tau[:, 1]))
    wf = flat_weight(d)
    w = {'gen': wf, 'cleo': wf * np.prod(d['u'], 1)}
    S = {'gen': d['h_gen'], 'cleo': d['h_cleo']}
    est = {'exact h_gen': d['h_gen']}
    for (teacher, arm), v in nets.items():
        if 'h_pred' in v and teacher == 'gen':
            est[f'reco {arm}'] = v['h_pred']
    modes = d['modes']
    rng = np.random.default_rng(13)
    res = {'mean_abs_Cnn': float(np.abs(Cz[:, 0, 0]).mean()), 'scopes': {}}
    for sc, m in (('any 3pi', (modes == 3).any(1)), ('3pi x 3pi', (modes == 3).all(1)), ('all events', np.ones(len(modes), bool))):
        idx = np.flatnonzero(m)
        p = summary(est, S, Cz, w, idx)
        bb = [summary(est, S, Cz, w, rng.choice(idx, len(idx))) for _ in range(BOOT)]
        ex = p['exact h_gen'][1]
        res['scopes'][sc] = {}
        for n in est:
            arr = np.array([b[n] for b in bb])
            res['scopes'][sc][n] = {'double_T_over_Zstate': p[n][0], 'ci95': np.quantile(arr[:, 0], [.025, .975]).tolist(),
                                    'single_T': p[n][2], 'nominal_T_over_Z_rel_exact': p[n][1] / ex,
                                    'nominal_ci68': np.quantile(arr[:, 1], [.16, .84]).tolist()}
            r = res['scopes'][sc][n]
            print(f"{sc:10s} {n:22s} single T {r['single_T']:.3f}  double T/Z-state {r['double_T_over_Zstate']:.3f} "
                  f"{np.round(r['ci95'], 3)}  nominal (r_T/r_Z)/(exact) {r['nominal_T_over_Z_rel_exact']:.3f}", flush=True)
    (HERE / 'results' / 'transfer_zstate.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
