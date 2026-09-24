"""Ideal and reco-level sensitivity to the Higgs CP mixing angle phi_tau.

Cohort: the 59,390-event canonical validation surface; only the H rows are
used (Z is a different production and carries no phi_tau).  Exact h is the
generator-current polarimeter; reco h is h_pred of the gen3pi point-h arms.
Nothing is trained here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import (acoplanarity, bilinear, cramer_rao, score,  # noqa: E402
                      sensitivity, triple, weights)
from polarimeter import canonical_h, frames  # noqa: E402

ARMS = [('base_s43', 'reco, no geometry'),
        ('full22_s42', 'reco + 3 IP + SV (seed 42)'),
        ('full22_s43', 'reco + 3 IP + SV (seed 43)'),
        ('idealip22_s42', 'reco + ideal-IP oracle')]
MODES = {0: 'pi', 1: 'rho', 3: '3pi'}
PAIRS = [(0, 0, 'pi x pi'), (1, 1, 'rho x rho'), (3, 3, '3pi x 3pi'),
         (0, 1, 'pi x rho'), (0, 3, 'pi x 3pi'), (1, 3, 'rho x 3pi')]


def pair_mask(modes, a, b):
    return ((modes[:, 0] == a) & (modes[:, 1] == b)) | ((modes[:, 0] == b) & (modes[:, 1] == a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--output', type=Path, default=HERE / 'results')
    ap.add_argument('--n-ref', type=int, default=10000, help='events the quoted sigma refers to')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    D = np.load(args.data / 'truth_surface_cleo.npz')
    y = D['labels'].astype(bool)
    modes = D['modes']
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    h_gen = np.array(D['h_ref'], float)
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, modes, D['pion_charges'], s, md)
            h_gen[sel, s] = h

    C0 = c_matrix(0.0)
    out = {'n_ref': args.n_ref, 'C0': C0.tolist(),
           'cohort': {'rows': int(len(y)), 'H': int(y.sum())}}

    # ---------- closure of the reweighting machinery ----------
    hH = h_gen[y]
    S, f = score(hH, C0)
    out['reweight_closure'] = {'f_min': float(f.min()), 'f_mean': float(f.mean()),
                               'score_mean': float(S.mean()), 'score_sd': float(S.std())}
    probes = {'Tp_exact': triple(hH[:, 0], hH[:, 1]),
              'score': S,
              'cos_aco': np.cos(acoplanarity(hH))}
    fd = {}
    for name, T in probes.items():
        rows = {'cov_response': float(np.cov(np.stack([T, S]))[0, 1])}
        for deg in (2.0, 5.0, 10.0):
            d = np.deg2rad(deg)
            wp, wm = weights(hH, d, C0), weights(hH, -d, C0)
            rows[f'fd_{deg:g}deg'] = float((np.average(T, weights=wp) - np.average(T, weights=wm))
                                           / (2 * d))
        fd[name] = rows
    out['reweight_closure']['response_vs_finite_difference'] = fd
    ess = {}
    for deg in (5.0, 15.0, 45.0, 90.0):
        w = weights(hH, np.deg2rad(deg), C0)
        ess[f'{deg:g}deg'] = {'ess_frac': float(w.sum()**2 / (len(w) * (w**2).sum())),
                              'w_max': float(w.max()), 'w_mean': float(w.mean())}
    out['reweight_closure']['ess'] = ess

    # ---------- optimal (Cramer-Rao) with tail trimming stability ----------
    cr = {'all': cramer_rao(S, args.n_ref)}
    order = np.argsort(f)
    for frac in (0.001, 0.003, 0.01, 0.03):
        keep = np.ones(len(f), bool)
        keep[order[:int(frac * len(f))]] = False
        cr[f'trim_lowest_f_{frac:g}'] = cramer_rao(S[keep], args.n_ref)
    out['cramer_rao'] = cr

    # ---------- observables ----------
    def observables(h, tag):
        hm, hp = h[:, 0].astype(float), h[:, 1].astype(float)
        aco = acoplanarity(h)
        return {f'{tag}:Tp': triple(hm, hp),
                f'{tag}:Tp_over_f': triple(hm, hp) / np.maximum(1 + bilinear(hm, hp, C0), 1e-3),
                f'{tag}:sin_aco': np.sin(aco),
                f'{tag}:cos_aco': np.cos(aco)}

    table = {}
    obs_exact = observables(hH, 'exact')
    for name, T in obs_exact.items():
        table[name] = sensitivity(T, S, args.n_ref, boot=200)
    table['exact:score(optimal)'] = sensitivity(S, S, args.n_ref, boot=200)

    arms = {}
    for arm, desc in ARMS:
        P = np.load(args.data / f'gen3pi_{arm}.npz')
        assert np.array_equal(P['global_indices'], D['global_indices'])
        hp_arm = P['h_pred'][y].astype(float)
        assert np.allclose(P['h'], h_gen, atol=1e-5)
        row = {'description': desc,
               'mean_norm': float(np.linalg.norm(hp_arm, axis=-1).mean()),
               'mean_cos_to_truth': float(np.mean(np.sum(hp_arm * hH, -1)
                                                  / np.linalg.norm(hp_arm, axis=-1)))}
        for name, T in observables(hp_arm, arm).items():
            row[name.split(':')[1]] = sensitivity(T, S, args.n_ref, boot=200)
        arms[arm] = row
    out['sensitivity'] = {'exact': table, 'reco': arms}

    # ---------- shuffle control: reco h paired with the wrong event ----------
    rng = np.random.default_rng(3)
    P = np.load(args.data / 'gen3pi_full22_s42.npz')
    hp_arm = P['h_pred'][y].astype(float)
    perm = rng.permutation(len(hp_arm))
    out['shuffle_control'] = {
        name.split(':')[1]: sensitivity(T, S, args.n_ref, boot=200)
        for name, T in observables(hp_arm[perm], 'shuf').items()}

    # ---------- mode-pair breakdown ----------
    mH = modes[y]
    bym = {}
    for a, b, nm in PAIRS:
        k = pair_mask(mH, a, b)
        if k.sum() < 200:
            continue
        row = {'n': int(k.sum()), 'fraction': float(k.mean()),
               'exact_Tp': sensitivity(triple(hH[k, 0], hH[k, 1]), S[k], args.n_ref),
               'exact_optimal': cramer_rao(S[k], args.n_ref)}
        Pf = np.load(args.data / 'gen3pi_full22_s42.npz')['h_pred'][y][k].astype(float)
        row['full22_Tp'] = sensitivity(triple(Pf[:, 0], Pf[:, 1]), S[k], args.n_ref)
        bym[nm] = row
    out['by_mode_pair'] = bym

    (args.output / 'sensitivity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('cohort', 'reweight_closure', 'cramer_rao')}, indent=1)[:4000])
    print('\n--- sigma(phi_tau) [deg] for', args.n_ref, 'H events ---')
    for k, v in table.items():
        print(f'  {k:28s} {v["sigma_deg"]:8.3f} +- {v.get("sigma_deg_se", float("nan")):.3f}')
    for arm, row in arms.items():
        print(f'  {arm:16s} |h|={row["mean_norm"]:.3f} cos={row["mean_cos_to_truth"]:.3f}')
        for k in ('Tp', 'Tp_over_f', 'sin_aco', 'cos_aco'):
            print(f'    {k:24s} {row[k]["sigma_deg"]:8.3f} +- {row[k].get("sigma_deg_se", float("nan")):.3f}')
    print('  shuffle control Tp:', out['shuffle_control']['Tp']['sigma_deg'])


if __name__ == '__main__':
    main()
