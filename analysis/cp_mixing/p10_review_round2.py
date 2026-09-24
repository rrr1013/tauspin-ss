"""Follow-ups to the two independent reviews of 2026-09-25.

  A  paired bootstrap for every per-channel comparison (classical vs learned h)
  B  mode-pair dependence of CP and H/Z on one scale (effective statistics),
     with paired tests between pairs
  C  readout study: positive control on exact h, cross-fit on validation,
     degrees up to 6
  D  static Delta-phi modulation across arms (geometry vs learned prior) with
     bootstrap errors, plus the additive-model prediction of the peak shift,
     and the same measurement for the classical phi*_CP
"""
import json
import math
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_density import c_matrix  # noqa: E402
from cp_tools import acoplanarity, score, sensitivity, triple, weights  # noqa: E402
from p5_readout import poly_features  # noqa: E402
from p6_geometry_value import auc, llr, ppf  # noqa: E402
from polarimeter import canonical_h, frames  # noqa: E402

C0 = c_matrix(0.0)
NREF = 10000
out = {}


def sig(T, S, n=NREF):
    c = np.cov(np.stack([np.asarray(T, float), S]))
    return math.degrees(math.sqrt(c[0, 0] / n) / abs(c[0, 1]))


def paired_ratio(Ta, Tb, S, B=2000, seed=5):
    """bootstrap of sigma(Ta)/sigma(Tb) on identical resampled events."""
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        k = rng.integers(0, len(S), len(S))
        v.append(sig(Ta[k], S[k]) / sig(Tb[k], S[k]))
    v = np.asarray(v)
    return {'ratio': float(sig(Ta, S) / sig(Tb, S)),
            'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            'p_ratio_lt_1': float(np.mean(v < 1))}


# ---------------- load ----------------
D = np.load(HERE / 'data' / 'truth_surface_cleo.npz')
A = np.load(HERE / 'results' / 'classical_arrays.npz')
F = np.load(HERE / 'results' / 'review_followups.json') if False else None
y = D['labels'].astype(bool)
tmode = D['modes']
fr = frames(D['pions'], D['pi0'], D['nu4'])
h_gen = np.array(D['h_ref'], float)
for md in (0, 1, 3):
    for s in (0, 1):
        sel, hh = canonical_h(fr, tmode, D['pion_charges'], s, md)
        h_gen[sel, s] = hh
S_all, f_all = score(h_gen, C0)

# rebuild the classical observables including the rest-frame y variant
import p9_review_followups as _  # noqa: F401  (reuses its construction, writes its own json)
RF = json.loads((HERE / 'results' / 'review_followups.json').read_text())

# ---------------- A: paired comparisons per channel ----------------
P42 = np.load(HERE / 'data' / 'gen3pi_full22_s42.npz')['h_pred'].astype(float)
rowsA = {}
for chan in ('pi-pi', 'rho-rho', 'pi-rho'):
    rows = A[f'{chan}__rows']
    Sk = A[f'{chan}__score']
    cl_reco = np.sin(A[f'{chan}__phistar_reco'])
    cl_truth = np.sin(A[f'{chan}__phistar_truth'])
    learned = A[f'{chan}__reco_Tp']
    rowsA[chan] = {
        'n': int(len(rows)),
        'classical_reco_over_learned': paired_ratio(cl_reco, learned, Sk),
        'classical_truth_labY_over_learned': paired_ratio(cl_truth, learned, Sk),
        'combined_classicalTruth_and_learned_insample': None}
    # in-sample optimal linear combination, reported as an optimistic bound
    X = np.stack([cl_truth, learned], 1)
    Xc = X - X.mean(0)
    beta = np.linalg.solve(Xc.T @ Xc, Xc.T @ (Sk - Sk.mean()))
    rowsA[chan]['combined_classicalTruth_and_learned_insample'] = {
        'sigma_deg': sig(X @ beta, Sk), 'learned_alone_sigma_deg': sig(learned, Sk)}
out['A_paired_channel_comparisons'] = rowsA

# ---------------- B: mode-pair dependence on one scale ----------------
PAIRS = [(0, 0, 'pi x pi'), (1, 1, 'rho x rho'), (3, 3, '3pi x 3pi'),
         (0, 1, 'pi x rho'), (0, 3, 'pi x 3pi'), (1, 3, 'rho x 3pi')]
mp = json.loads((HERE.parents[0] / 'outputs' / 'mode-pair-auc-origin-20260924'
                 / 'results.json').read_text())['observed']['generator_current']
rowsB = {}
for a, b, nm in PAIRS:
    k = ((tmode[:, 0] == a) & (tmode[:, 1] == b)) | ((tmode[:, 0] == b) & (tmode[:, 1] == a))
    kH = k & y
    s_cp = sig(triple(h_gen[kH, 0], h_gen[kH, 1]), S_all[kH])
    dprime = math.sqrt(2) * ppf(mp[nm]['auc'])
    rowsB[nm] = {'n_H': int(kH.sum()), 'cp_sigma_deg': s_cp, 'hz_auc': mp[nm]['auc'],
                 'hz_dprime': dprime}
cp_best = min(r['cp_sigma_deg'] for r in rowsB.values())
hz_best = max(r['hz_dprime'] for r in rowsB.values())
for nm, r in rowsB.items():
    r['cp_effective_lumi_vs_best'] = (cp_best / r['cp_sigma_deg'])**2
    r['hz_effective_lumi_vs_best'] = (r['hz_dprime'] / hz_best)**2
out['B_mode_pair_same_scale'] = {
    'rows': rowsB,
    'cp_spread_effective_lumi': max(r['cp_sigma_deg'] for r in rowsB.values())**2 / cp_best**2,
    'hz_spread_effective_lumi': hz_best**2 / min(r['hz_dprime'] for r in rowsB.values())**2}
# paired test of the two extreme CP pairs
rng = np.random.default_rng(7)
k1 = ((tmode[:, 0] == 0) & (tmode[:, 1] == 3)) | ((tmode[:, 0] == 3) & (tmode[:, 1] == 0))
k2 = (tmode[:, 0] == 1) & (tmode[:, 1] == 1)
d = []
for _ in range(2000):
    i1 = rng.integers(0, (k1 & y).sum(), (k1 & y).sum())
    i2 = rng.integers(0, (k2 & y).sum(), (k2 & y).sum())
    h1, s1 = h_gen[k1 & y][i1], S_all[k1 & y][i1]
    h2, s2 = h_gen[k2 & y][i2], S_all[k2 & y][i2]
    d.append(sig(triple(h1[:, 0], h1[:, 1]), s1) - sig(triple(h2[:, 0], h2[:, 1]), s2))
d = np.asarray(d)
out['B_mode_pair_same_scale']['pi3pi_minus_rhorho_cp_sigma'] = {
    'delta_deg': float(d.mean()), 'sd': float(d.std()), 'n_sigma': float(abs(d.mean()) / d.std())}

# ---------------- C: readout, positive control and cross-fit ----------------
rowsC = {}
for tag, arm in (('exact_h (positive control)', None), ('base_s43', 'base_s43'),
                 ('full22_s42', 'full22_s42')):
    if arm is None:
        hp_val = h_gen[y]
    else:
        hp_val = np.load(HERE / 'data' / f'gen3pi_{arm}.npz')['h_pred'].astype(float)[y]
    Sv = S_all[y]
    r = {'plugin_Tp': sig(triple(hp_val[:, 0], hp_val[:, 1]), Sv),
         'optimal_from_exact_score': math.degrees(1 / math.sqrt(NREF * np.var(Sv)))}
    rng2 = np.random.default_rng(3)
    half = rng2.random(len(Sv)) < 0.5
    for deg in (2, 3, 4, 5, 6):
        pred = np.empty(len(Sv))
        for tr, te in ((half, ~half), (~half, half)):
            X = poly_features(hp_val[tr], deg)
            beta, *_ = np.linalg.lstsq(X, Sv[tr], rcond=None)
            pred[te] = poly_features(hp_val[te], deg) @ beta
        r[f'crossfit_deg{deg}'] = {'sigma_deg': sig(pred, Sv),
                                   'corr': float(np.corrcoef(pred, Sv)[0, 1])}
    rowsC[tag] = r
out['C_readout_crossfit'] = rowsC

# ---------------- D: static modulation across arms ----------------
def modulation(x, w=None, bins=16, B=200, seed=2):
    def fit(xx, ww):
        n, e = np.histogram(xx, bins=bins, range=(0, 2 * np.pi), weights=ww)
        c = 0.5 * (e[1:] + e[:-1])
        d = n / n.sum()
        M = np.stack([np.ones_like(c), np.cos(c), np.sin(c)], 1)
        b = np.linalg.lstsq(M, d, rcond=None)[0]
        return math.hypot(b[1], b[2]) / b[0], math.degrees(math.atan2(-b[2], b[1]))
    a, p = fit(x, w)
    rng3 = np.random.default_rng(seed)
    v = [fit(x[k], None if w is None else w[k])[0]
         for k in (rng3.integers(0, len(x), len(x)) for _ in range(B))]
    return {'amplitude': a, 'amplitude_se': float(np.std(v)), 'phase_deg': p, 'n': int(len(x))}

rr = (tmode[:, 0] == 1) & (tmode[:, 1] == 1)
rowsD = {}
for tag, src in (('exact_h', None), ('base_s43', 'cleo_base_s43'),
                 ('full22_s42', 'cleo_full22_s42'),
                 ('full22_shuffle_s42', 'cleo_full22_shuffle_s42'),
                 ('idealip22_s42', 'cleo_idealip22_s42')):
    h = h_gen if src is None else np.load(HERE / 'data' / f'{src}.npz')['h_pred'].astype(float)
    e = {'H_rhorho': modulation(acoplanarity(h[rr & y])),
         'Z_rhorho': modulation(acoplanarity(h[rr & ~y]))}
    w45 = weights(h_gen[rr & y], np.deg2rad(45.), C0)
    e['H_rhorho_phi45'] = modulation(acoplanarity(h[rr & y]), w45)
    ah, az = e['H_rhorho']['amplitude'], e['Z_rhorho']['amplitude']
    spin = max(ah - az, 0.0)
    # additive model: static part fixed at phase 0, spin part rotates by 2 phi
    vx = az + spin * math.cos(math.pi / 2)
    vy = spin * math.sin(math.pi / 2)
    e['additive_model_predicted_shift_deg'] = math.degrees(math.atan2(vy, vx))
    e['observed_shift_deg'] = float(((e['H_rhorho_phi45']['phase_deg']
                                      - e['H_rhorho']['phase_deg'] + 180) % 360) - 180)
    rowsD[tag] = e
# classical phi*_CP on Z
rows = A['rho-rho__rows']
rowsD['classical_phistar'] = {'H_rhorho': modulation(A['rho-rho__phistar_reco'])}
out['D_static_modulation'] = rowsD

(HERE / 'results' / 'review_round2.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
