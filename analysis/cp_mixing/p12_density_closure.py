"""The two checks left open by the reviews.

A  External closure of the spin-density model.  Every sigma in this run rests
   on the cohort following p ~ 1 + h-^T C(0) h+.  The naive moment estimator
   9<h- h+^T> is far from C(0) on this cohort (pi x pi gives 1.23/1.36/-0.53),
   which the reviews flagged as unverified.  The 2026-09-24 re-decay toy
   rebuilds each event's decays from rotated rest-frame templates with an
   *ideal* C(0) correlation and then applies the same visible pT/eta
   selection.  If the density model is right, that toy must reproduce the
   observed moments.  Here both sides get bootstrap errors.

B  Cross-fitted combination of the classical phi*_CP and the learned h, so
   the complementarity reported in the run note is not an in-sample number.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import score, triple  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402

C0 = c_matrix(0.0)
NREF = 10000
TOY = json.loads((HERE.parents[0] / 'outputs' / 'mode-pair-auc-origin-20260924'
                  / 'results.json').read_text())
out = {}


def sig(T, S, n=NREF):
    c = np.cov(np.stack([np.asarray(T, float), S]))
    return math.degrees(math.sqrt(c[0, 0] / n) / abs(c[0, 1]))


def cmom(h, B=300, seed=4):
    """9 <h- h+^T> with bootstrap errors."""
    v = 9 * np.einsum('ni,nj->ij', h[:, 0], h[:, 1]) / len(h)
    rng = np.random.default_rng(seed)
    bs = np.stack([9 * np.einsum('ni,nj->ij', h[k, 0], h[k, 1]) / len(h)
                   for k in (rng.integers(0, len(h), len(h)) for _ in range(B))])
    return v, bs.std(0)


D = np.load(HERE / 'data' / 'truth_surface_cleo.npz')
y = D['labels'].astype(bool)
tmode = D['modes']
fr = frames(D['pions'], D['pi0'], D['nu4'])
h_gen = np.array(D['h_ref'], float)
for md in (0, 1, 3):
    for s in (0, 1):
        sel, hh = canonical_h(fr, tmode, D['pion_charges'], s, md)
        h_gen[sel, s] = hh
S_all, _ = score(h_gen, C0)

# ---------------- A ----------------
PAIRS = [(0, 0, 'pi x pi'), (1, 1, 'rho x rho'), (3, 3, '3pi x 3pi'),
         (0, 1, 'pi x rho'), (0, 3, 'pi x 3pi'), (1, 3, 'rho x 3pi')]
rowsA = {}
for a, b, nm in PAIRS:
    k = (((tmode[:, 0] == a) & (tmode[:, 1] == b)) | ((tmode[:, 0] == b) & (tmode[:, 1] == a)))
    for lab, tag in ((1, 'H'), (0, 'Z')):
        kk = k & (y == bool(lab))
        obs, err = cmom(h_gen[kk])
        toy = np.array(TOY['toy'][nm]['with_selection'][f'C_{tag}'])
        d = (np.diag(obs) - np.diag(toy)) / np.maximum(np.diag(err), 1e-9)
        rowsA[f'{nm} ({tag})'] = {
            'n': int(kk.sum()),
            'observed_diag': np.round(np.diag(obs), 3).tolist(),
            'observed_diag_err': np.round(np.diag(err), 3).tolist(),
            'ideal_C0_toy_with_selection_diag': np.round(np.diag(toy), 3).tolist(),
            'pull': np.round(d, 2).tolist(),
            'max_abs_pull': float(np.abs(d).max())}
naive = {'C0_H': np.diag(C0).tolist(), 'C0_Z': [0., 0., 1.]}
out['A_density_closure'] = {
    'what': 'observed 9<h- h+^T> vs a re-decay toy built with an ideal C(0) and the same '
            'visible pT>20 GeV, |eta|<2.5 selection (2026-09-24 run, toy stats ~1.2e5/pair)',
    'naive_unselected_expectation': naive, 'rows': rowsA,
    'worst_pull': max(r['max_abs_pull'] for r in rowsA.values())}

# ---------------- B ----------------
A = np.load(HERE / 'results' / 'classical_arrays.npz')
P42 = np.load(HERE / 'data' / 'gen3pi_full22_s42.npz')['h_pred'].astype(float)
rng = np.random.default_rng(21)
rowsB = {}
for chan in ('pi-pi', 'rho-rho', 'pi-rho'):
    rows = A[f'{chan}__rows']
    Sk = A[f'{chan}__score']
    learned = A[f'{chan}__reco_Tp']
    variants = {'classical_reco': np.sin(A[f'{chan}__phistar_reco']),
                'classical_truth_labY': np.sin(A[f'{chan}__phistar_truth'])}
    r = {'n': int(len(rows)), 'learned_alone': sig(learned, Sk)}
    for name, cl in variants.items():
        half = rng.random(len(Sk)) < 0.5
        pred = np.empty(len(Sk))
        for tr, te in ((half, ~half), (~half, half)):
            X = np.stack([cl[tr], learned[tr]], 1)
            Xc = X - X.mean(0)
            beta = np.linalg.solve(Xc.T @ Xc, Xc.T @ (Sk[tr] - Sk[tr].mean()))
            pred[te] = np.stack([cl[te], learned[te]], 1) @ beta
        r[f'combined_with_{name}'] = {'sigma_deg': sig(pred, Sk),
                                      'classical_alone': sig(cl, Sk)}
    rowsB[chan] = r
out['B_crossfit_combination'] = rowsB

(HERE / 'results' / 'density_closure.json').write_text(json.dumps(out, indent=1))
print('--- A: does an ideal C(0) plus the real selection reproduce the observed moments? ---')
print(f'{"pair":18s} {"n":>6s} {"observed diag":>28s} {"ideal-C0 toy + selection":>28s} {"pull":>20s}')
for nm, r in rowsA.items():
    o = '  '.join(f'{v:+.3f}' for v in r['observed_diag'])
    t = '  '.join(f'{v:+.3f}' for v in r['ideal_C0_toy_with_selection_diag'])
    pl = ' '.join(f'{v:+.1f}' for v in r['pull'])
    print(f'{nm:18s} {r["n"]:6d}   {o}    {t}   {pl}')
print('worst |pull| =', round(out['A_density_closure']['worst_pull'], 2))
print('\n--- B: cross-fitted combination of classical and learned ---')
for chan, r in rowsB.items():
    print(f'{chan:8s} n={r["n"]:5d}  learned {r["learned_alone"]:.3f}', end='')
    for k, v in r.items():
        if k.startswith('combined'):
            print(f'   {k[14:]:22s} alone {v["classical_alone"]:.3f} -> combined {v["sigma_deg"]:.3f}', end='')
    print()
