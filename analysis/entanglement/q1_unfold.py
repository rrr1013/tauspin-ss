"""Three estimators of the ditau spin state, and what each one can and cannot do.

Generated states, known analytically from the polarised Dirac trace:

  H:  B- = B+ = 0,          C = diag(1, 1, -1)   (cp_density, exact in beta)
  Z:  B- = B+ = 0.14715 k,  C = diag(0, 0, +1)   (z_density, unpolarised Z)

so rho_H is the pure Bell state (|ud> + |du>)/sqrt2 along the tau flight axis
and rho_Z is the separable classical mixture 0.574 |uu> + 0.426 |dd>.  What the
main line measures as an H/Z AUC is, at truth level, the separation between a
maximally entangled state and a separable one.

Estimators
  naive     B = 3<h>, C = 9<h- h+^T>            -- assumes p0 isotropic
  moment    C = M-^-1 <h- h+^T> M+^-1           -- uses the observed per-side
                                                  second moments, no knowledge
                                                  of the generated state
  calibrated  (R - <g> e^T) theta = <g> - A     -- response from simulation

Identifiability
  The calibrated estimator is exact, but its response cannot be taken from the
  sample being measured: theta = theta_assumed solves the estimating equation
  identically for *any* assumed state (proved in the run note, demonstrated in
  `identifiability` below).  The response has to come from a sample whose
  generated state is known independently.  That is why the exact-h row of this
  script is a closure test, not a measurement, and why every reco number in
  q2/q3 is quoted as a calibrated sensitivity.
"""
import json
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import load_arm, load_surface
from z_density import density as z_density

HERE = Path(__file__).resolve().parent
THETA_H = E.pack(np.zeros(3), np.zeros(3), np.diag([1.0, 1.0, -1.0]))
INJECTIONS = [
    ('theory H', E.pack(np.zeros(3), np.zeros(3), np.diag([1., 1., -1.]))),
    ('singlet', E.pack(np.zeros(3), np.zeros(3), -np.eye(3))),
    ('CP mixed 30deg', E.pack(np.zeros(3), np.zeros(3), np.array(
        [[np.cos(np.pi / 3), np.sin(np.pi / 3), 0],
         [-np.sin(np.pi / 3), np.cos(np.pi / 3), 0], [0, 0, -1.]]))),
    ('separable kk', E.pack(np.zeros(3), np.zeros(3), np.diag([0., 0., 1.]))),
    ('half strength', E.pack(np.zeros(3), np.zeros(3), np.diag([.5, .5, -.5]))),
    ('weakly correlated', E.pack([0.05, -0.05, 0.1], [-0.02, 0.03, -0.1],
                                 np.diag([0.2, 0.2, 0.6]))),
    ('unpolarised', np.zeros(15)),
]


def theta_z():
    _, Bm, Bp, C = z_density()
    return E.pack(-Bm, -Bp, C)          # canonical h flips the single-spin blocks


def report(theta):
    bm, bp, C = E.unpack(theta)
    return {'C': C.tolist(), 'B_minus_physical': (-bm).tolist(),
            'B_plus_physical': (-bp).tolist(),
            **{k: v for k, v in E.observables(theta).items()}}


def boot_sd(fn, n, n_boot, seed=3):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return np.std(np.array(vals), axis=0)


def main():
    S = load_surface()
    h, y = S['h_gen'], S['labels']
    THETA_Z = theta_z()
    out = {'theta_H_generated': THETA_H.tolist(), 'theta_Z_generated': THETA_Z.tolist(),
           'rho_H_theory': report(THETA_H), 'rho_Z_theory': report(THETA_Z)}

    for name, mask, th_gen in (('H', y, THETA_H), ('Z', ~y, THETA_Z)):
        hs = h[mask]
        phi = E.features(hs)
        cal = E.calibrate(phi, phi, th_gen)
        res = {'n': int(mask.sum()), 'calibration_info': cal['info'],
               'E0_h_minus': cal['e'][:3].tolist(), 'E0_h_plus': cal['e'][3:6].tolist(),
               'Z_norm': float(1 + cal['e'] @ th_gen),
               'C_generated': E.unpack(th_gen)[2].tolist()}

        # ---- the three estimators on the nominal sample ----
        th_naive = E.naive_theta(phi)
        C_mom = E.moment_unfold_C(hs)
        th_cal = E.unfold(cal, phi)
        sd_naive = boot_sd(lambda k: E.naive_theta(phi[k])[6:], len(phi), 600)
        sd_mom = boot_sd(lambda k: E.moment_unfold_C(hs[k]).reshape(9), len(phi), 600)
        res['estimators'] = {
            'naive': {'C': E.unpack(th_naive)[2].tolist(), 'sd_C': sd_naive.reshape(3, 3).tolist(),
                      'max_bias': float(np.abs(E.unpack(th_naive)[2] - E.unpack(th_gen)[2]).max()),
                      'observables': E.observables(th_naive)},
            'moment': {'C': C_mom.tolist(), 'sd_C': sd_mom.reshape(3, 3).tolist(),
                       'max_bias': float(np.abs(C_mom - E.unpack(th_gen)[2]).max()),
                       'observables': E.observables(E.pack(np.zeros(3), np.zeros(3), C_mom))},
            'calibrated_same_sample': {
                'C': E.unpack(th_cal)[2].tolist(),
                'max_dev': float(np.abs(th_cal - th_gen).max()),
                'max_dev_if_Z_ignored': float(np.abs(
                    np.linalg.solve(cal['R'], phi.mean(0) - cal['A']) - th_gen).max())},
        }

        # ---- identifiability: calibrate at a deliberately wrong state ----
        ident = []
        for scale in (0.5, 0.8, 1.0):
            bad = th_gen.copy()
            bad[6:] = bad[6:] * scale
            cb = E.calibrate(phi, phi, bad)
            tb = E.unfold(cb, phi)
            ident.append({'assumed_C_scale': scale,
                          'returned_C_diag': np.diag(E.unpack(tb)[2]).tolist(),
                          'max_dev_from_assumption': float(np.abs(tb - bad).max())})
        res['identifiability_same_sample'] = ident

        # ---- injection closure with an independent calibration sample ----
        # Calibration comes from the 472,872-event *training* split, which the
        # validation cohort shares no events with; only the exact h is needed.
        T = load_arm('full22_s42', S, split='train')
        tm = T['labels'] if name == 'H' else ~T['labels']
        phiT = E.features(T['h'][tm])
        calA = E.calibrate(phiT, phiT, th_gen)
        res['calibration_sample'] = {'source': 'train split', 'n': int(tm.sum()),
                                     'ess_frac': calA['info']['ess_frac'],
                                     'E0_h_minus': calA['e'][:3].tolist(),
                                     'E0_h_plus': calA['e'][3:6].tolist()}
        phiB, hB = phi, hs
        inj = []
        for nm, th_t in INJECTIONS:
            w = E.f_nominal(phiB, th_t) / E.f_nominal(phiB, th_gen)
            if w.min() < 0:
                inj.append({'name': nm, 'skipped': 'negative weight'})
                continue
            th_fit = E.unfold(calA, phiB, weights=w)
            sd = boot_sd(lambda k: E.unfold(calA, phiB[k], weights=w[k]), len(phiB), 300)
            # joint bootstrap: resample the calibration sample too, so the pull
            # is against the full uncertainty of the procedure
            rj = np.random.default_rng(5)
            vals = []
            for _ in range(120):
                kc = rj.integers(0, len(phiT), len(phiT))
                km = rj.integers(0, len(phiB), len(phiB))
                vals.append(E.unfold(E.calibrate(phiT[kc], phiT[kc], th_gen),
                                     phiB[km], weights=w[km]))
            sd_joint = np.std(np.array(vals), axis=0)
            dev = th_fit - th_t
            C_m = E.moment_unfold_C(hB, weights=w)
            inj.append({
                'name': nm, 'ess_frac': float(w.sum() ** 2 / (len(w) * (w ** 2).sum())),
                'calibrated': {'C_diag': np.diag(E.unpack(th_fit)[2]).tolist(),
                               'max_abs_dev': float(np.abs(dev).max()),
                               'max_pull': float(np.abs(dev / np.maximum(sd, 1e-9)).max()),
                               'max_pull_joint': float(np.abs(dev / np.maximum(sd_joint, 1e-9)).max()),
                               'sd_C_diag': [float(sd[6]), float(sd[10]), float(sd[14])],
                               'sd_joint_C_diag': [float(sd_joint[6]), float(sd_joint[10]),
                                                   float(sd_joint[14])]},
                'moment': {'C_diag': np.diag(C_m).tolist(),
                           'max_bias_C': float(np.abs(C_m - E.unpack(th_t)[2]).max())},
                'naive': {'C_diag': np.diag(E.unpack(E.naive_theta(phiB, weights=w))[2]).tolist()},
            })
        res['injection_closure'] = {'n_cal': int(tm.sum()), 'n_meas': int(len(phiB)),
                                    'rows': inj}
        out[name] = res

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'unfold.json').write_text(json.dumps(out, indent=1))

    for name in ('H', 'Z'):
        r = out[name]
        gen = np.diag(np.array(r['C_generated']))
        print(f"=== {name}  n={r['n']}  generated C diag {np.round(gen,3)}"
              f"  calib ESS {r['calibration_info']['ess_frac']:.3f}  Z_norm {r['Z_norm']:.4f}")
        print(f"  E0[h-] {np.round(r['E0_h_minus'],4)}   E0[h+] {np.round(r['E0_h_plus'],4)}"
              '   (isotropic p0 would give 0)')
        es = r['estimators']
        for k in ('naive', 'moment'):
            d = np.diag(np.array(es[k]['C'])); s = np.diag(np.array(es[k]['sd_C']))
            print(f"  {k:8s} C diag " + '  '.join(f'{a:+.4f}+-{b:.4f}' for a, b in zip(d, s))
                  + f"   max bias {es[k]['max_bias']:.4f}"
                  + f"   min eig rho {es[k]['observables']['min_eig_rho']:+.4f}")
        print(f"  calibrated (same sample) max dev {es['calibrated_same_sample']['max_dev']:.2e}"
              f"  -- exact by construction; ignoring Z(theta) would cost "
              f"{es['calibrated_same_sample']['max_dev_if_Z_ignored']:.4f}")
        print('  identifiability: calibrating on the measured sample returns the assumption')
        for i in r['identifiability_same_sample']:
            print(f"    assumed C x{i['assumed_C_scale']}  ->  returned C diag "
                  f"{np.round(i['returned_C_diag'],4)}   dev from assumption "
                  f"{i['max_dev_from_assumption']:.2e}")
        print(f"  injection closure, calibration on an independent half "
              f"({r['injection_closure']['n_cal']} events):")
        for i in r['injection_closure']['rows']:
            if 'skipped' in i:
                print(f"    {i['name']:18s} skipped"); continue
            print(f"    {i['name']:18s} ESS {i['ess_frac']:.2f}"
                  f"  calib C {np.round(i['calibrated']['C_diag'],3)}"
                  f"  dev {i['calibrated']['max_abs_dev']:.3f} "
                  f"({i['calibrated']['max_pull']:.1f}s, joint {i['calibrated']['max_pull_joint']:.1f}s)"
                  f" | moment {np.round(i['moment']['C_diag'],3)} bias "
                  f"{i['moment']['max_bias_C']:.3f} | naive {np.round(i['naive']['C_diag'],3)}")
        print()


if __name__ == '__main__':
    main()
