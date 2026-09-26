"""q1: sigma(P_tau) per Z event at each information level.

Linear response on identical events:  d<T>/dP = Cov(T, S), S from the exact
generator-current h of the Z rows.  T is whatever the method produces, so every
method is priced on the same events with no refitting and no new sample.
1/sqrt(N Var S) is the Cramer-Rao bound with the exact h.

`full22_shuffle_s42` is the arm whose *geometry inputs* were shuffled before
training (the 2026-09-23 control); it is a real prediction from real visible +
MET, so it is NOT a null for this observable.  The null here is an explicit row
permutation of h_pred.
"""
import json

import numpy as np

import pol_data as PD
import pol_tools as PT

N_REF = 100_000          # Z signal events the quoted sigma refers to
BOOT = 300
ARMS = [('base_s43', 'reco, no geometry'),
        ('full22_s42', 'reco + 3 IP + SV (seed 42)'),
        ('full22_s43', 'reco + 3 IP + SV (seed 43)'),
        ('full22_shuffle_s42', 'reco + shuffled geometry'),
        ('idealip22_s42', 'reco + ideal-IP oracle')]

S_surface = PD.load_surface()
lab = S_surface['labels']
z = ~lab
h = S_surface['h_gen']
modes = S_surface['modes']

S, f = PT.score(h[z])
print(f'Z rows {z.sum()}   <S> {S.mean():+.4f}   Var S {S.var():.4f}'
      f'   f range [{f.min():.3f}, {f.max():.3f}]')
out = {'n_ref': N_REF, 'n_z_rows': int(z.sum()),
       'cramer_rao_exact_h': PT.cramer_rao(S, N_REF)}
print('Cramer-Rao, exact h : sigma(P) = %.5f per %d events'
      % (out['cramer_rao_exact_h']['sigma'], N_REF))

rows = {}


def add(key, T, label):
    r = PT.sensitivity(T, S, N_REF, boot=BOOT)
    r['label'] = label
    rows[key] = r
    print(f'{label:44s} sigma(P) {r["sigma"]:.5f} +- {r.get("sigma_se", 0):.5f}'
          f'   response {r["response"]:+.5f}  sd(T) {r["sd_T"]:.4f}')
    return r


def fisher_direction(X, S, folds=2, seed=0):
    """Cross-fitted linear readout a.X maximising |Cov(a.X, S)|/sd(a.X)."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(S)), folds)
    T = np.empty(len(S))
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        Xc = X[tr] - X[tr].mean(0)
        Sc = S[tr] - S[tr].mean()
        a = np.linalg.solve(np.cov(Xc, rowvar=False) + 1e-9 * np.eye(X.shape[1]),
                            (Xc * Sc[:, None]).mean(0))
        T[te] = X[te] @ a
    return T


add('exact_hk', h[z, 0, 2] + h[z, 1, 2], 'exact h, h_k sum')
add('exact_score', S, 'exact h, optimal (score itself)')

for arm, desc in ARMS:
    hp = PD.load_arm(arm, S_surface)['h_pred'][z]
    add(f'{arm}_hk', hp[:, 0, 2] + hp[:, 1, 2], f'{desc}: h_pred_k sum')
    add(f'{arm}_lin', fisher_direction(hp.reshape(len(hp), 6), S),
        f'{desc}: best linear (cross-fit)')

# ---- explicit null: h_pred paired with the wrong event ----------------------
hp = PD.load_arm('full22_s42', S_surface)['h_pred'][z]
rng = np.random.default_rng(7)
nulls = []
for k in range(20):
    p = rng.permutation(len(hp))
    T = hp[p, 0, 2] + hp[p, 1, 2]
    nulls.append(PT.sensitivity(T, S, N_REF)['response'])
out['permutation_null'] = {'responses': nulls, 'mean': float(np.mean(nulls)),
                           'sd': float(np.std(nulls)),
                           'real_response': rows['full22_s42_hk']['response']}
print(f'permutation null response {np.mean(nulls):+.5f} +- {np.std(nulls):.5f}'
      f'   (real arm {rows["full22_s42_hk"]["response"]:+.5f})')
out['rows'] = rows

# ---- per-side decay-mode analysing power ------------------------------------
# One row per tau side.  The leading term of the score for that side is its own
# h_k, so sigma_side = sqrt(Var T)/|Cov(T, h_k)| / sqrt(n_sides) reduces to the
# textbook 1/sqrt(n E_0[h_k^2]) for T = h_k and an isotropic p_0.
print('\nper-tau-side analysing power (sigma(P) per 10,000 tau sides of that mode):')
N_SIDE = 10_000
mode_rows = {}
arm_hp = {a: PD.load_arm(a, S_surface)['h_pred'][z] for a, _ in ARMS}
mz = modes[z]
hz = h[z]
for md, mname in PD.MODES.items():
    sel = (mz == md)                                     # (n, 2)
    hk = hz[:, :, 2][sel]
    row = {'n_sides': int(sel.sum()),
           'E0_hk2': float(np.mean(hk ** 2)),
           'ideal_textbook': float(1 / np.sqrt(N_SIDE * np.mean(hk ** 2)))}
    r = PT.sensitivity(hk, hk, N_SIDE, boot=BOOT)
    row['exact'] = r
    for a, desc in ARMS:
        row[a] = PT.sensitivity(arm_hp[a][:, :, 2][sel], hk, N_SIDE, boot=BOOT)
    mode_rows[mname] = row
    print(f'  {mname:4s} n_sides {row["n_sides"]:6d}  E0[h_k^2] {row["E0_hk2"]:.4f}'
          f'  exact {r["sigma"]:.5f}  base {row["base_s43"]["sigma"]:.5f}'
          f'  +IP/SV {row["full22_s42"]["sigma"]:.5f}'
          f'  idealIP {row["idealip22_s42"]["sigma"]:.5f}'
          f'   attenuation {row["base_s43"]["response"] / r["response"]:.3f}')
out['per_mode_side'] = mode_rows

PD.RESULTS.mkdir(exist_ok=True)
(PD.RESULTS / 'q1_sensitivity.json').write_text(json.dumps(out, indent=1))
print('\nwrote', PD.RESULTS / 'q1_sensitivity.json')
