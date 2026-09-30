"""Figure for the simple-vs-ATLAS IP/SV comparison (evaluate_ipsv.py output).

(a) fixed-readout H/Z AUC per geometry representation, both samples (seed 42 markers, seed 43 open)
(b) AUC gain over 'none' (same sample, same seed), paired 95% CI
(c) correlation of the predicted with the target h per component (n, r, k), seed 42
(d) gain of 'local' and 'lab' over 'none' per truth mode pair, seed 42
usage: python plot_ipsv.py EVAL.json OUT.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E = json.load(open(sys.argv[1]))
R, C = E['runs'], E['vs_none']
KEYS = [('none', 'none'), ('legacy', 'legacy\n(d0, z0)'), ('lab', 'lab\n(PV-IP x,y,z)'),
        ('local', 'local\n(n, r, k)'), ('local_shuffle', 'local,\nshuffled')]
SAMP = {'simple': ('simple smearing (Chen)', 'C1', 'o'), 'atlas': ('ATLAS full reco', 'C0', 's')}
PAIRS = ['pi x pi', 'pi x rho', 'rho x rho', 'pi x 3pi', 'rho x 3pi', '3pi x 3pi']

fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), gridspec_kw={'width_ratios': [1.2, 1.2, 1.1, 1.3]})
x = np.arange(len(KEYS))
for j, (smp, (lab, col, mk)) in enumerate(SAMP.items()):
    off = (j - 0.5) * 0.22
    for seed, fill in (('42', True), ('43', False)):
        ys, xs = [], []
        for i, (k, _) in enumerate(KEYS):
            n = f'{smp}_{k}_s{seed}'
            if n in R:
                xs.append(i + off + (0.06 if seed == '43' else 0)); ys.append(R[n]['auc_unit'])
        ax[0].plot(xs, ys, mk, color=col, mfc=col if fill else 'none', ms=7,
                   label=f'{lab}, seed {seed}' if ys else None)
        ds, dx, lo, hi = [], [], [], []
        for i, (k, _) in enumerate(KEYS):
            n = f'{smp}_{k}_s{seed}'
            if n in C:
                d = C[n]['delta_auc_unit']; ci = C[n]['ci95']
                ds.append(d); dx.append(i + off + (0.06 if seed == '43' else 0)); lo.append(d - ci[0]); hi.append(ci[1] - d)
        if ds:
            ax[1].errorbar(dx, ds, [lo, hi], fmt=mk, color=col, mfc=col if fill else 'none', ms=7, capsize=2,
                           label=f'{lab}, seed {seed}')
for a in ax[:2]:
    a.set_xticks(x, [k[1] for k in KEYS], fontsize=8)
ax[0].set_ylabel('H/Z AUC, fixed h readout (unit weight)')
ax[0].set_title('(a) same learner, same reduced input,\nonly the appended IP/SV block differs')
ax[0].legend(fontsize=7)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_ylabel('ΔAUC vs none (paired 95% CI)')
ax[1].set_title('(b) gain over no IP/SV')
ax[1].legend(fontsize=7)

comp = ['$h_n$', '$h_r$', '$h_k$']
w = 0.08
for j, (smp, (lab, col, mk)) in enumerate(SAMP.items()):
    for i, (k, kl) in enumerate(KEYS[:4]):
        n = f'{smp}_{k}_s42'
        if n not in R:
            continue
        c = R[n]['corr_nrk']
        xs = np.arange(3) + (j * 4 + i - 3.5) * w
        ax[2].bar(xs, c, w, color=col, alpha=0.35 + 0.18 * i, edgecolor='k', lw=0.3,
                  label=f'{lab.split(" (")[0]}: {kl.splitlines()[0]}')
ax[2].set_xticks(range(3), comp)
ax[2].set_ylim(0.25, 0.85)
ax[2].set_ylabel('corr(predicted h, target h)')
ax[2].set_title('(c) per-component h quality, seed 42\n(bars: none, legacy, lab, local)')
ax[2].legend(fontsize=6, ncol=2, loc='upper left')

for j, (smp, (lab, col, mk)) in enumerate(SAMP.items()):
    for k, ls, fill in (('local', '-', True), ('lab', ':', False)):
        n = f'{smp}_{k}_s42'
        if n not in C:
            continue
        d = [C[n]['delta_pairs_unit'][p] for p in PAIRS]
        ax[3].plot(range(len(PAIRS)), d, mk + ls, color=col, mfc=col if fill else 'none', ms=6, label=f'{lab}: {k} − none')
ax[3].axhline(0, color='k', lw=0.6)
ax[3].set_xticks(range(len(PAIRS)), [p.replace(' x ', '×').replace('pi', 'π').replace('rho', 'ρ').replace('3π', '3π') for p in PAIRS])
ax[3].set_ylabel('ΔAUC vs none, per truth mode pair')
ax[3].set_title('(d) where the gain comes from (seed 42)')
ax[3].legend(fontsize=7)
ns = {s: next(R[n]['n'] for n in R if R[n]['sample'] == s) for s in SAMP if any(R[n]['sample'] == s for n in R)}
fig.text(0.5, -0.03, 'Point-h TauSpinTransformer (p11 recipe, 50 epochs), generator-current 3π teacher, common reduced input (tau 6–9 and track 5–14 zeroed in both samples), '
         f"fixed 6→64→64→1 readout on train predictions. Validation h-valid events: simple {ns.get('simple', 0):,}, ATLAS {ns.get('atlas', 0):,}; "
         'SV only for reco 3-prong in both.', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150, bbox_inches='tight')
print('written', sys.argv[2])
