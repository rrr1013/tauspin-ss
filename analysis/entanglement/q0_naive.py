"""Where the textbook estimator goes wrong, and how badly.

Plug-in  B = 3<h>,  C = 9 <h- h+^T>  assumes the unpolarised polarimeter
measure p0 is isotropic.  This prints, for H and Z and for every decay-mode
pair, the plug-in theta, the implied density matrix, and whether that density
matrix is physical.  It also measures the anisotropy of p0 directly, which is
the quantity the textbook formula sets to I/3.
"""
import json
from pathlib import Path

import numpy as np

import ent_tools as E
from ent_data import MODES, PAIRS, load_arm, load_surface, pair_mask

HERE = Path(__file__).resolve().parent
C0 = np.diag([1.0, 1.0, -1.0])


def block(h, mask, label):
    g = E.features(h[mask])
    th = E.naive_theta(g)
    o = E.observables(th)
    bm, bp, C = E.unpack(th)
    return {'label': label, 'n': int(mask.sum()), 'C': C.tolist(),
            'B_minus_canonical_h': bm.tolist(), 'B_plus_canonical_h': bp.tolist(),
            **{k: v for k, v in o.items()}}


def main():
    S = load_surface()
    h, y, modes = S['h_gen'], S['labels'], S['modes']
    out = {'cohort': {'rows': int(len(y)), 'H': int(y.sum()), 'Z': int((~y).sum())},
           'C_H_theory': C0.tolist(),
           'theory_H': E.observables(E.pack(np.zeros(3), np.zeros(3), C0))}

    out['naive'] = {}
    for lab, name in ((y, 'H'), (~y, 'Z')):
        out['naive'][name] = {'all': block(h, lab, f'{name} all modes')}
        for a, b, nm in PAIRS:
            m = lab & pair_mask(modes, a, b)
            if m.sum() > 200:
                out['naive'][name][nm] = block(h, m, f'{name} {nm}')

    # anisotropy of the observed single-side second moments: the textbook
    # formula replaces each of these by I/3.
    aniso = {}
    for lab, name in ((y, 'H'), (~y, 'Z')):
        rows = {}
        for s in (0, 1):
            M = np.einsum('ni,nj->ij', h[lab, s], h[lab, s]) / lab.sum()
            rows[f'side{s}'] = {'M': M.tolist(), 'eig': np.sort(np.linalg.eigvalsh(M)).tolist(),
                                'mean_h': (h[lab, s].mean(0)).tolist()}
        for a, b, nm in PAIRS:
            m = lab & pair_mask(modes, a, b)
            if m.sum() > 200:
                M = np.einsum('ni,nj->ij', h[m, 0], h[m, 0]) / m.sum()
                rows[f'{nm} side0'] = {'eig': np.sort(np.linalg.eigvalsh(M)).tolist(),
                                       'mean_h': h[m, 0].mean(0).tolist()}
        aniso[name] = rows
    out['p_observed_second_moments'] = aniso

    # single-mode marginals: for a given side, split by that side's mode only
    per_mode = {}
    for lab, name in ((y, 'H'), (~y, 'Z')):
        for md, mn in MODES.items():
            vals = []
            for s in (0, 1):
                m = lab & (modes[:, s] == md)
                M = np.einsum('ni,nj->ij', h[m, s], h[m, s]) / m.sum()
                vals.append({'n': int(m.sum()), 'M_diag': np.diag(M).tolist(),
                             'mean_h': h[m, s].mean(0).tolist()})
            per_mode[f'{name} {mn}'] = vals
    out['per_mode_marginals'] = per_mode

    # reco arms, same plug-in
    out['naive_reco'] = {}
    for arm, desc in [('base_s43', ''), ('full22_s42', ''), ('idealip22_s42', '')]:
        A = load_arm(arm, S)
        hp = A['h_pred']
        out['naive_reco'][arm] = {'all_H': block(hp, y, f'{arm} H'),
                                  'mean_norm': float(np.linalg.norm(hp, axis=-1).mean())}

    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'naive.json').write_text(json.dumps(out, indent=1))

    t = out['naive']['H']['all']
    print('H, all modes, plug-in C:')
    print(np.array2string(np.array(t['C']), precision=4))
    print(f"  min eig rho = {t['min_eig_rho']:+.4f}   (negative => not a density matrix)")
    print(f"  neg_eig_PT  = {t['neg_eig_pt']:+.4f}   concurrence = {t['concurrence']:.4f}"
          f"   m12 = {t['m12']:.4f}")
    z = out['naive']['Z']['all']
    print('Z, all modes, plug-in C:')
    print(np.array2string(np.array(z['C']), precision=4))
    print(f"  min eig rho = {z['min_eig_rho']:+.4f}   neg_eig_PT = {z['neg_eig_pt']:+.4f}")
    print()
    for nm in ('side0', 'side1'):
        print(f"H {nm} second moments eig = "
              f"{np.round(out['p_observed_second_moments']['H'][nm]['eig'], 4)}"
              f"  (isotropic would be 1/3 each)")
    for k, v in out['per_mode_marginals'].items():
        print(f"  {k:8s} side0 M_diag {np.round(v[0]['M_diag'],4)} <h> {np.round(v[0]['mean_h'],4)}")
    print()
    for arm, r in out['naive_reco'].items():
        print(f"{arm:16s} |h_pred|={r['mean_norm']:.4f} plug-in C diag "
              f"{np.round(np.diag(np.array(r['all_H']['C'])),4)}  min eig rho "
              f"{r['all_H']['min_eig_rho']:+.4f}")


if __name__ == '__main__':
    main()
