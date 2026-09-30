"""Figure: mass-width control (validation AUC per arm on v2, v3, v2r, v3r)."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

r = json.load(open(sys.argv[1]))
arms = [('truth_nu_transformer', 'truth visible + ν'), ('truth_transformer', 'truth visible + MET'),
        ('reco_transformer', 'reco Transformer'), ('reco_baseline', 'reco baseline')]
tags = [('v2', 'Z v2 (two-step)', 'C0', 'o'), ('v3', 'Z v3 (full ME, BW width)', 'C3', 's'),
        ('v2r', 'Z v2, mass → 91.1872', 'C0', 'D'), ('v3r', 'Z v3, mass → 91.1872', 'C3', '^')]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={'width_ratios': [1.3, 1]})
for i, (arm, lab) in enumerate(arms):
    for j, (t, tl, col, mk) in enumerate(tags):
        v = r[arm][t]
        ax[0].errorbar(i + (j - 1.5) * 0.17, v['auc'], v['boot_std'], fmt=mk, color=col, ms=6,
                       mfc='none' if t.endswith('r') else col, capsize=2, label=tl if i == 0 else None)
ax[0].set_xticks(range(len(arms)), [a[1].replace(' + ', '\n+ ') for a in arms], fontsize=9)
ax[0].set_ylabel('H/Z validation AUC (unit weight)')
ax[0].set_title('(a) each arm on the four Z samples (H identical)')
ax[0].legend(fontsize=8, loc='upper right')
for i, (arm, lab) in enumerate(arms):
    d_mass = r[arm]['v3']['auc'] - r[arm]['v3r']['auc']
    d_rest = r[arm]['v3r']['auc'] - r[arm]['v2r']['auc']
    e = np.hypot(r[arm]['v3r']['boot_std'], r[arm]['v2r']['boot_std'])
    ax[1].barh(i + 0.18, min(d_mass, 0.03), 0.34, color='0.6', label='v3 − v3r (mass width)' if i == 0 else None)
    if d_mass > 0.03:
        ax[1].text(0.0305, i + 0.18, f'{d_mass:+.3f} →', va='center', fontsize=8)
    ax[1].barh(i - 0.18, d_rest, 0.34, xerr=e, color='C3', label='v3r − v2r (not mass: Z spin state)' if i == 0 else None)
ax[1].set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=9)
ax[1].axvline(0, color='k', lw=0.6)
ax[1].set_xlim(-0.01, 0.036)
ax[1].set_xlabel('ΔAUC (bars clipped at 0.03)')
ax[1].set_title('(b) what the v3 change consists of')
ax[1].legend(fontsize=8, loc='lower right')
fig.text(0.5, -0.03, 'Simple-smearing chain (R20 code, seeds, recipe), validation 89,500 events per sample; error bars: event bootstrap std '
         '(single training seed, seed-to-seed spread not included).', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150, bbox_inches='tight')
