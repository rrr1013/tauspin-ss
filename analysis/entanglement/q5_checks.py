"""Three checks that the run's weaker claims needed.

1. The instrumental correlation without any reweighting.  The generated Z state
   has C_nn = C_rr = 0 to six decimals, so *any* transverse <h- h+> measured on
   Z events is spin-independent by construction: no 1/f weight, no calibration,
   no assumption about p0 enters.  If this agrees with the reweighted number
   obtained on H, the effect is real and not an artefact of the weights.

2. Why cutting on f is not a valid cross-check of the 1/f weighting.  f =
   1 + h-^T C h+ depends on *both* sides, so selecting on it correlates them and
   breaks the factorised measure the estimator assumes.  The number it returns
   is larger, and that is expected, not a failure.

3. Why the pi-pi pair has the worst exact-h sensitivity even though the pion
   carries the full analysing power: its p0 is by far the most anisotropic, so
   the response matrix is the worst conditioned.
"""
import json
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import PAIRS, load_arm, load_surface, pair_mask
from z_density import density as z_density

HERE = Path(__file__).resolve().parent
ARMS = ['base_s43', 'full22_s42', 'full22_s43', 'idealip22_s42']
THETA_H = E.pack(np.zeros(3), np.zeros(3), np.diag([1.0, 1.0, -1.0]))


def main():
    S = load_surface()
    y = S['labels']
    _, Bm, Bp, CZ = z_density()
    THETA_Z = E.pack(-Bm, -Bp, CZ)
    phiH, phiZ = E.features(S['h_gen'][y]), E.features(S['h_gen'][~y])
    out = {'C_Z_transverse': [float(CZ[0, 0]), float(CZ[1, 1])]}

    # ---- 1. weight-free instrumental correlation on Z --------------------
    rows = {}
    rng = np.random.default_rng(1)
    gz = E.features(S['h_gen'][~y])
    rows['exact_h'] = {'Z_raw_nn': float(gz.mean(0)[6]), 'Z_raw_rr': float(gz.mean(0)[10])}
    for arm in ARMS:
        A = load_arm(arm, S)
        gZ, gH = E.features(A['h_pred'][~y]), E.features(A['h_pred'][y])
        sd = float(np.std([gZ[rng.integers(0, len(gZ), len(gZ))].mean(0)[6]
                           for _ in range(400)]))
        rows[arm] = {
            'Z_raw_nn': float(gZ.mean(0)[6]), 'Z_raw_nn_sd': sd,
            'Z_raw_rr': float(gZ.mean(0)[10]),
            'H_spin_off_nn': float(E.calibrate(phiH, gH, THETA_H)['A'][6]),
            'Z_spin_off_nn': float(E.calibrate(phiZ, gZ, THETA_Z)['A'][6])}
    out['instrumental_weight_free'] = rows

    # ---- 2. the f cut is not a valid cross-check -------------------------
    f = E.f_nominal(phiH, THETA_H)
    cut = {}
    for thr in (0.0, 0.3, 0.5, 0.8):
        m = f > thr
        cut[f'f>{thr}'] = {'kept': float(m.mean())}
        for arm in ['base_s43', 'idealip22_s42']:
            A = load_arm(arm, S)
            g = E.features(A['h_pred'][y])
            c = E.calibrate(phiH[m], g[m], THETA_H)
            cut[f'f>{thr}'][arm] = float(c['A'][6])
        # the same cut applied to the exact h, where the truth is ~0
        c = E.calibrate(phiH[m], phiH[m], THETA_H)
        cut[f'f>{thr}']['exact_h'] = float(c['A'][6])
    out['f_cut_is_not_a_control'] = cut

    # ---- 3. pi-pi conditioning ------------------------------------------
    cond = {}
    for a, b, nm in PAIRS:
        m = y & pair_mask(S['modes'], a, b)
        if m.sum() < 800:
            continue
        p = E.features(S['h_gen'][m])
        c = E.calibrate(p, p, THETA_H)
        R = c['R'][6:, 6:]
        cond[nm] = {'n': int(m.sum()), 'cond_R_outer': float(np.linalg.cond(R)),
                    'min_eig_R_outer': float(np.linalg.eigvalsh((R + R.T) / 2).min()),
                    'E0_h_k': float(c['e'][2]),
                    'M_kk_observed': float(np.mean(S['h_gen'][m][:, 0, 2] ** 2))}
    out['mode_pair_conditioning'] = cond

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'checks.json').write_text(json.dumps(out, indent=1))

    print('1. instrumental correlation without any reweighting')
    print(f"   generated C_Z transverse = {out['C_Z_transverse']}, so the raw value on Z"
          ' is instrumental by construction')
    print(f"   {'arm':16s} {'Z raw nn':>12s} {'Z raw rr':>10s} {'H spin-off':>12s} {'Z spin-off':>12s}")
    for k, v in rows.items():
        sd = f"+-{v['Z_raw_nn_sd']:.4f}" if 'Z_raw_nn_sd' in v else ' ' * 7
        print(f"   {k:16s} {v['Z_raw_nn']:+.4f}{sd} {v['Z_raw_rr']:+10.4f}"
              f" {v.get('H_spin_off_nn', float('nan')):+12.4f}"
              f" {v.get('Z_spin_off_nn', float('nan')):+12.4f}")
    print('\n2. cutting on f is not a control (it correlates the two sides)')
    for k, v in cut.items():
        print(f"   {k:8s} kept {v['kept']:.3f}   exact h {v['exact_h']:+.4f}"
              f"   base {v['base_s43']:+.4f}   idealip {v['idealip22_s42']:+.4f}")
    print('   the exact-h column should stay near zero if the cut were harmless; it does not')
    print('\n3. mode-pair response conditioning (exact h)')
    for k, v in cond.items():
        print(f"   {k:10s} n {v['n']:5d}  cond(R) {v['cond_R_outer']:5.2f}"
              f"  min eig {v['min_eig_R_outer']:.4f}  E0[h_k] {v['E0_h_k']:+.4f}"
              f"  <h_k^2> {v['M_kk_observed']:.4f}  (isotropic 0.333)")


if __name__ == '__main__':
    main()
