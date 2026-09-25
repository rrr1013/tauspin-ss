"""How much of the ditau quantum state survives reconstruction.

What this script measures is a *sensitivity*, in the Asimov sense: the data are
assumed to follow the generated state, so the central value of every row is the
generated state by construction, and what differs between rows is the
uncertainty.  That is forced by the structure of the estimator.

    theta_hat = theta_cal + (1/S) M^-1 ( <g>_data - <g>_calibration )

so a response calibrated on a sample drawn from the same distribution as the
data returns the calibration's own state; only a *difference* between data and
simulation moves the answer.  The estimator is therefore validated by injecting
states into the measurement sample (`injection_closure` here and in q1), and is
used to quote how precisely a deviation could be seen -- not to rediscover the
input.

Convention used everywhere below:
  - response A, R, e calibrated on the full H validation cohort at the
    generated state theta_H (treated as simulation with negligible statistical
    error; the delta-method and bootstrap errors below hold it fixed);
  - the uncertainty comes from bootstrapping the measured events only;
  - the witness W is the optimal PPT witness of rho_H, fixed a priori:
    <W> = -1/2 for rho_H, >= 0 for every separable state.
"""
import json
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import PAIRS, load_arm, load_surface, pair_mask

HERE = Path(__file__).resolve().parent
THETA_H = E.pack(np.zeros(3), np.zeros(3), np.diag([1.0, 1.0, -1.0]))
WCOEF, WCONST = E.witness_coefficients(THETA_H)
ARMS = [('base_s43', 'reco point-h, no geometry'),
        ('full22_s42', 'reco point-h + 3 IP + SV (seed 42)'),
        ('full22_s43', 'reco point-h + 3 IP + SV (seed 43)'),
        ('idealip22_s42', 'reco point-h + ideal-IP oracle')]
N_BOOT = 2000
INJECTIONS = [
    ('singlet', E.pack(np.zeros(3), np.zeros(3), -np.eye(3))),
    ('CP mixed 30deg', E.pack(np.zeros(3), np.zeros(3), np.array(
        [[np.cos(np.pi / 3), np.sin(np.pi / 3), 0],
         [-np.sin(np.pi / 3), np.cos(np.pi / 3), 0], [0, 0, -1.]]))),
    ('separable kk', E.pack(np.zeros(3), np.zeros(3), np.diag([0., 0., 1.]))),
    ('half strength', E.pack(np.zeros(3), np.zeros(3), np.diag([.5, .5, -.5]))),
    ('unpolarised', np.zeros(15)),
]


def witness(theta):
    return float(WCONST + WCOEF @ theta)


def reach(w_value, w_sd, n_meas, n_sigma):
    if w_value >= 0:
        return None
    return float(n_meas * (n_sigma * w_sd / abs(w_value)) ** 2)


def sensitivity(phi, g, cal, n_boot=N_BOOT, seed=0):
    """Central value + bootstrap spread with the response held fixed."""
    th = E.unfold(cal, g)
    rng = np.random.default_rng(seed)
    n = len(g)
    reps = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        reps.append(E.unfold(cal, g[k]))
    reps = np.array(reps)
    w = np.array([witness(t) for t in reps])
    cov, _ = E.theta_covariance(g, cal)
    sd_delta = float(np.sqrt(WCOEF @ cov @ WCOEF))
    o = E.observables(th)
    m12 = np.array([E.observables(t)['m12'] for t in reps])
    return {'n': int(n),
            'C': E.unpack(th)[2].tolist(),
            'sd_C': reps[:, 6:].reshape(-1, 3, 3).std(0).tolist(),
            'observables': o,
            'witness': witness(th), 'witness_sd': float(w.std()),
            'witness_sd_delta_method': sd_delta,
            'm12': o['m12'], 'm12_sd': float(m12.std()),
            'reach_3sigma': reach(witness(th), float(w.std()), n, 3),
            'reach_5sigma': reach(witness(th), float(w.std()), n, 5),
            'reach_3sigma_m12': (float(n * (3 * m12.std() / (o['m12'] - 1)) ** 2)
                                 if o['m12'] > 1 else None)}


def main():
    S = load_surface()
    y, modes = S['labels'], S['modes']
    phi = E.features(S['h_gen'][y])
    n_H = int(y.sum())
    out = {'witness_const': WCONST, 'witness_coefficients': WCOEF.tolist(),
           'witness_on_theory_H': witness(THETA_H), 'n_validation_H': n_H,
           'convention': 'Asimov: central value is the generated state; rows differ in sigma',
           'arms': {}}

    cal_x = E.calibrate(phi, phi, THETA_H)
    out['exact_h'] = sensitivity(phi, phi, cal_x)
    out['exact_h']['calibration'] = cal_x['info']

    thn = E.naive_theta(phi)
    out['exact_h_naive'] = {'C': E.unpack(thn)[2].tolist(), 'witness': witness(thn),
                            'observables': E.observables(thn),
                            'projected_C': E.unpack(E.project_physical(thn))[2].tolist(),
                            'projected_observables': E.observables(E.project_physical(thn))}
    hv = S['h_gen'][y]
    Cm = E.moment_unfold_C(hv)
    thm = E.pack(np.zeros(3), np.zeros(3), Cm)
    rngm = np.random.default_rng(4)
    wm = [witness(E.pack(np.zeros(3), np.zeros(3),
                         E.moment_unfold_C(hv[rngm.integers(0, len(hv), len(hv))])))
          for _ in range(500)]
    out['exact_h_moment'] = {'C': Cm.tolist(), 'witness': witness(thm),
                             'witness_sd': float(np.std(wm)),
                             'observables': E.observables(thm),
                             'reach_3sigma': reach(witness(thm), float(np.std(wm)), n_H, 3)}

    for arm, desc in ARMS:
        A = load_arm(arm, S)
        g = E.features(A['h_pred'][y])
        cal = E.calibrate(phi, g, THETA_H)
        r = sensitivity(phi, g, cal)
        r['description'] = desc
        r['mean_h_pred_norm'] = float(np.linalg.norm(A['h_pred'][y], axis=-1).mean())
        r['calibration'] = cal['info']
        tn = E.naive_theta(g)
        r['naive'] = {'C': E.unpack(tn)[2].tolist(), 'witness': witness(tn),
                      'observables': E.observables(tn)}
        raw = g.mean(0)[[6, 10, 14]]
        r['instrumental'] = {'raw_outer_diag': raw.tolist(),
                             'unpolarised_outer_diag': cal['A'][[6, 10, 14]].tolist(),
                             'fraction_diag': (cal['A'][[6, 10, 14]] / raw).tolist()}
        # Closure.  Reweighting the *same* events that carry the response makes
        # the solve algebraically exact, so the response is calibrated on one
        # half of validation and the injected state is measured on the other.
        # Both halves are events the network never trained on.
        rng_h = np.random.default_rng(23)
        half = rng_h.random(len(g)) < 0.5
        cal_h = E.calibrate(phi[half], g[half], THETA_H)
        phiB, gB = phi[~half], g[~half]
        inj = []
        for nm, th_t in INJECTIONS:
            w = E.f_nominal(phiB, th_t) / E.f_nominal(phiB, THETA_H)
            if w.min() < 0:
                continue
            th_fit = E.unfold(cal_h, gB, weights=w)
            rng = np.random.default_rng(8)
            sd = np.std(np.array([E.unfold(cal_h, gB[k], weights=w[k])
                                  for k in (rng.integers(0, len(gB), len(gB))
                                            for _ in range(250))]), axis=0)
            inj.append({'name': nm, 'n_cal': int(half.sum()), 'n_meas': int((~half).sum()),
                        'C_diag': np.diag(E.unpack(th_fit)[2]).tolist(),
                        'target_C_diag': np.diag(E.unpack(th_t)[2]).tolist(),
                        'max_abs_dev': float(np.abs(th_fit - th_t).max()),
                        'max_pull': float(np.abs((th_fit - th_t)
                                                 / np.maximum(sd, 1e-9)).max()),
                        'ess_frac': float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))})
        r['injection_closure'] = inj
        out['arms'][arm] = r

    out['instrumental_exact_h'] = {
        'raw_outer_diag': phi.mean(0)[[6, 10, 14]].tolist(),
        'unpolarised_outer_diag': cal_x['A'][[6, 10, 14]].tolist()}

    # ---- mode pairs, best geometry arm ------------------------------------
    A = load_arm('full22_s42', S)
    mp = {}
    for a, b, nm in PAIRS:
        m = y & pair_mask(modes, a, b)
        if m.sum() < 800:
            continue
        pe = E.features(S['h_gen'][m])
        pr = E.features(A['h_pred'][m])
        re_ = sensitivity(pe, pe, E.calibrate(pe, pe, THETA_H), n_boot=600)
        rr = sensitivity(pe, pr, E.calibrate(pe, pr, THETA_H), n_boot=600)
        mp[nm] = {'n': int(m.sum()),
                  'exact': {'witness': re_['witness'], 'witness_sd': re_['witness_sd'],
                            'reach_3sigma': re_['reach_3sigma'],
                            'sd_C_diag': np.diag(np.array(re_['sd_C'])).tolist()},
                  'reco': {'witness': rr['witness'], 'witness_sd': rr['witness_sd'],
                           'reach_3sigma': rr['reach_3sigma'],
                           'sd_C_diag': np.diag(np.array(rr['sd_C'])).tolist()}}
    out['mode_pairs_full22_s42'] = mp

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'reco.json').write_text(json.dumps(out, indent=1))

    def line(tag, r):
        print(f"{tag:32s} sigma(<W>) {r['witness_sd']:.5f}"
              f"  (delta method {r['witness_sd_delta_method']:.5f})"
              f"   sigma(C) diag "
              + ' '.join(f'{v:.4f}' for v in np.diag(np.array(r['sd_C'])))
              + f"   N(3s) {r['reach_3sigma']:,.0f}   N(5s) {r['reach_5sigma']:,.0f}"
              + (f"   N(3s, m12>1) {r['reach_3sigma_m12']:,.0f}"
                 if r['reach_3sigma_m12'] else ''))

    print(f"witness of rho_H: <W> = {witness(THETA_H):+.4f}; separable states have <W> >= 0")
    print(f"H validation events {n_H}; central values are the generated state by "
          f"construction, rows differ in sigma\n")
    line('exact h', out['exact_h'])
    for arm, _ in ARMS:
        line(f"{arm} (|h_pred| {out['arms'][arm]['mean_h_pred_norm']:.3f})", out['arms'][arm])
    print()
    nv = out['exact_h_naive']
    print(f"{'exact h, naive plug-in':32s} C diag "
          + ' '.join(f'{v:+.3f}' for v in np.diag(np.array(nv['C'])))
          + f"   min eig rho {nv['observables']['min_eig_rho']:+.4f}"
          + f"   projected C {np.round(np.diag(np.array(nv['projected_C'])), 3)}")
    m = out['exact_h_moment']
    print(f"{'exact h, moment estimator':32s} C diag "
          + ' '.join(f'{v:+.3f}' for v in np.diag(np.array(m['C'])))
          + f"   <W> {m['witness']:+.4f}+-{m['witness_sd']:.4f}")
    for arm, _ in ARMS:
        r = out['arms'][arm]
        print(f"{arm + ', naive plug-in':32s} C diag "
              + ' '.join(f'{v:+.3f}' for v in np.diag(np.array(r['naive']['C'])))
              + f"   <W> {r['naive']['witness']:+.4f}")
    print('\ninstrumental correlation of the two predicted polarimeters (nn, rr, kk):')
    ix = out['instrumental_exact_h']
    print(f"  {'exact h':16s} raw {np.round(ix['raw_outer_diag'], 4)}"
          f"   unpolarised {np.round(ix['unpolarised_outer_diag'], 4)}")
    for arm, _ in ARMS:
        v = out['arms'][arm]['instrumental']
        print(f"  {arm:16s} raw {np.round(v['raw_outer_diag'], 4)}"
              f"   unpolarised {np.round(v['unpolarised_outer_diag'], 4)}"
              f"   fraction {np.round(v['fraction_diag'], 3)}")
    print('\ninjection closure on the reco arms (response fixed at theta_H):')
    for arm, _ in ARMS:
        for i in out['arms'][arm]['injection_closure']:
            print(f"  {arm:16s} {i['name']:16s} C {np.round(i['C_diag'], 3)}"
                  f" vs {np.round(i['target_C_diag'], 3)}"
                  f"  dev {i['max_abs_dev']:.3f} ({i['max_pull']:.1f}s)  ESS {i['ess_frac']:.2f}")
    print('\nmode pairs, full22_s42:')
    for nm, v in out['mode_pairs_full22_s42'].items():
        print(f"  {nm:10s} n {v['n']:5d}  N3 exact {v['exact']['reach_3sigma']:8,.0f}"
              f"   N3 reco {v['reco']['reach_3sigma']:8,.0f}"
              f"   ratio {v['reco']['reach_3sigma'] / v['exact']['reach_3sigma']:.2f}")


if __name__ == '__main__':
    main()
