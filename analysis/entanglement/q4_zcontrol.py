"""Control: does the same reco-level tomography see Z as separable?

The witness is fixed to the one that is optimal for rho_H, so a correct
measurement must return <W> = -1/2 on H and +1/2 on Z, with every separable
state at <W> >= 0.  Running the identical procedure on the Z half of the same
cohort therefore tests the thing that matters: that the method reports
entanglement because the state is entangled, not because the reconstruction
manufactures correlation.

Also reported: the Horodecki criterion m12 = s1^2 + s2^2 > 1, which is the
condition for a CHSH violation to be possible, and the events needed to
establish it.
"""
import json
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import load_arm, load_surface
from q2_reco import THETA_H, reach, sensitivity, witness
from z_density import density as z_density

HERE = Path(__file__).resolve().parent
ARMS = ['base_s43', 'full22_s42', 'idealip22_s42']
N_BOOT = 400


def theta_z():
    _, Bm, Bp, C = z_density()
    return E.pack(-Bm, -Bp, C)


def run(phi, g, th_gen, n_boot=N_BOOT, seed=0):
    """Asimov sensitivity with the response calibrated at the generated state."""
    cal = E.calibrate(phi, g, th_gen)
    r = sensitivity(phi, g, cal, n_boot=n_boot, seed=seed)
    th = E.unfold(cal, g)
    o = E.observables(th)
    r.update({'C_diag': np.diag(E.unpack(th)[2]).tolist(),
              'sd_C_diag': np.diag(np.array(r['sd_C'])).tolist(),
              'neg_eig_pt': o['neg_eig_pt'], 'concurrence': o['concurrence'],
              'B_minus_physical': (-E.unpack(th)[0]).tolist(),
              'B_plus_physical': (-E.unpack(th)[1]).tolist(),
              'calibration_ess': cal['info']['ess_frac']})
    return r


def main():
    S = load_surface()
    y = S['labels']
    TZ = theta_z()
    out = {'theta_Z_generated': TZ.tolist(),
           'witness_on_theory_Z': witness(TZ),
           'witness_on_theory_H': witness(THETA_H),
           'm12_theory': {'H': E.observables(THETA_H)['m12'],
                          'Z': E.observables(TZ)['m12']}}

    for name, mask, th_gen in (('H', y, THETA_H), ('Z', ~y, TZ)):
        phi = E.features(S['h_gen'][mask])
        rows = {'exact_h': run(phi, phi, th_gen)}
        for arm in ARMS:
            A = load_arm(arm, S)
            rows[arm] = run(phi, E.features(A['h_pred'][mask]), th_gen)
        for v in rows.values():
            v['reach_3sigma_entanglement'] = v['reach_3sigma']
            v['reach_5sigma_entanglement'] = v['reach_5sigma']
            # separability side: how many events to put <W> above zero at 3 sigma
            v['reach_3sigma_separable'] = (float(v['n'] * (3 * v['witness_sd']
                                                           / v['witness']) ** 2)
                                           if v['witness'] > 0 else None)
        out[name] = rows

    # ---- the sharpest control: response from the entangled sample, applied
    # to the separable one.  Different cohort, different state, so nothing here
    # is algebraically forced: if the reconstruction were manufacturing the
    # correlation, this would return the H state on Z events.
    cross = {}
    for arm in ['exact_h'] + ARMS:
        if arm == 'exact_h':
            gH = E.features(S['h_gen'][y]); gZ = E.features(S['h_gen'][~y])
        else:
            A = load_arm(arm, S)
            gH = E.features(A['h_pred'][y]); gZ = E.features(A['h_pred'][~y])
        phiH = E.features(S['h_gen'][y]); phiZ = E.features(S['h_gen'][~y])
        calH = E.calibrate(phiH, gH, THETA_H)
        thZ = E.unfold(calH, gZ)
        rng = np.random.default_rng(12)
        sd = np.std(np.array([E.unfold(calH, gZ[k])
                              for k in (rng.integers(0, len(gZ), len(gZ))
                                        for _ in range(400))]), axis=0)
        o = E.observables(thZ)
        op = E.observables(E.project_physical(thZ))
        cross[arm] = {'min_eig_rho': o['min_eig_rho'],
                      'neg_eig_pt_after_projection': op['neg_eig_pt'],
                      'concurrence_after_projection': op['concurrence'],
                      'C_diag': np.diag(E.unpack(thZ)[2]).tolist(),
                      'sd_C_diag': [float(sd[6]), float(sd[10]), float(sd[14])],
                      'witness': witness(thZ),
                      'max_abs_dev_from_theory_Z': float(np.abs(thZ - TZ).max()),
                      'pull_vs_theory_Z': float(np.abs((thZ - TZ)
                                                       / np.maximum(sd, 1e-9)).max()),
                      'neg_eig_pt': o['neg_eig_pt'], 'm12': o['m12']}
    out['cross_cohort_H_response_on_Z'] = cross
    print('--- response calibrated on H, applied to Z events '
          '(theory: C = (0,0,+1), <W> = +0.5)')
    for k, v in cross.items():
        print(f"  {k:16s} C diag "
              + ' '.join(f'{a:+.3f}+-{b:.3f}' for a, b in zip(v['C_diag'], v['sd_C_diag']))
              + f"  <W> {v['witness']:+.4f}  max dev from theory Z "
              f"{v['max_abs_dev_from_theory_Z']:.3f} ({v['pull_vs_theory_Z']:.1f}s)"
              f"  min eig rho {v['min_eig_rho']:+.3f}"
              f"  neg_PT(raw) {v['neg_eig_pt']:+.3f}"
              f"  neg_PT(projected) {v['neg_eig_pt_after_projection']:+.3f}")
    print()

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'zcontrol.json').write_text(json.dumps(out, indent=1))

    print(f"witness built for rho_H: theory gives {out['witness_on_theory_H']:+.3f} on H "
          f"and {out['witness_on_theory_Z']:+.3f} on Z;  separable <=> <W> >= 0")
    print(f"Horodecki m12: theory H {out['m12_theory']['H']:.3f}, Z {out['m12_theory']['Z']:.3f}"
          '  (CHSH violation possible only above 1)\n')
    for name in ('H', 'Z'):
        print(f'--- {name} cohort')
        for k, v in out[name].items():
            r3 = v['reach_3sigma_entanglement']
            r3m = v['reach_3sigma_m12']
            r3s = v['reach_3sigma_separable']
            print(f"  {k:16s} n {v['n']:5d}  C diag "
                  + ' '.join(f'{a:+.3f}+-{b:.3f}' for a, b in zip(v['C_diag'], v['sd_C_diag']))
                  + f"  <W> {v['witness']:+.4f}+-{v['witness_sd']:.4f}"
                  + f"  m12 {v['m12']:.3f}+-{v['m12_sd']:.3f}"
                  + (f"  N3(ent) {r3:,.0f}" if r3 else '  N3(ent) --')
                  + (f"  N3(sep) {r3s:,.0f}" if r3s else '  N3(sep) --')
                  + (f"  N3(m12) {r3m:,.0f}" if r3m else '  N3(m12) --'))
        print()


if __name__ == '__main__':
    main()
