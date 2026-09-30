"""Figures for the Z-sample validity check (meeting follow-up 2026-09-30).

fig_z_tomography.png : v3 Z tau-pair spin state measured with exact h vs the prediction from the
                       tau directions alone (EW theory, generator sin^2 theta_W), per |cos theta_CS|.
fig_cp_angles.png    : difference / sum angle of the transverse h and the experimental phi*_CP
                       for H, Z v2, Z v3 (LHE truth).
usage: python plot_zcheck.py TOMO_RESULTS_DIR CPANGLE_DIR OUTDIR
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

tomo_dir, cp_dir, outdir = map(Path, sys.argv[1:4])
outdir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10})

# ---------------- tomography ----------------
r = json.loads((tomo_dir / 'results.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
rows = r['bases']['boosted_beam'][1:]
x = np.array([0.5 * (b['bin'][0] + b['bin'][1]) for b in rows])
for comp, (i, j), col, mk in (('$C_{nn}$', (0, 0), 'C0', 'o'), ('$C_{rr}$', (1, 1), 'C3', 's'), ('$C_{kk}$', (2, 2), 'C2', '^')):
    meas = np.array([b['C_meas'][i][j] for b in rows])
    err = np.array([b['C_err'][i][j] for b in rows])
    pred = np.array([b['C_pred'][i][j] for b in rows])
    axes[0].errorbar(x, meas, err, fmt=mk, color=col, ms=4, capsize=2, label=f'{comp} measured (exact $h$)')
    axes[0].plot(x, pred, '-', color=col, lw=1.3, label=f'{comp} predicted (τ directions)')
axes[0].axhline(0, color='0.6', lw=0.6)
axes[0].set_xlabel(r'$|\cos\theta_{CS}|$ of $\tau^+$')
axes[0].set_ylabel('spin correlation $C_{ij}$')
axes[0].set_title('(a) τ-pair correlation, boosted-beam basis')
axes[0].legend(fontsize=7, ncol=2, loc='lower right')
axes[0].set_ylim(-1.05, 1.25)

for name, col, mk in (('boosted_beam', 'C0', 'o'), ('adaptive', 'C1', 'D'), ('cs_z', 'C4', 'v'), ('canonical_lab_beam', 'C7', 's')):
    rr = r['bases'][name][1:]
    conc = np.array([b['q_meas']['concurrence'] for b in rr])
    ce = np.array([b['q_meas_err']['concurrence'] for b in rr])
    cp = np.array([b['q_pred']['concurrence'] for b in rr])
    lab = {'boosted_beam': 'boosted beam', 'adaptive': 'adaptive (max. transverse)', 'cs_z': 'Collins–Soper $z$',
           'canonical_lab_beam': 'canonical (past runs)'}[name]
    axes[1].errorbar(x, conc, ce, fmt=mk, color=col, ms=4, capsize=2, label=f'{lab}: measured')
    axes[1].plot(x, cp, '-', color=col, lw=1.2)
axes[1].set_xlabel(r'$|\cos\theta_{CS}|$ of $\tau^+$')
axes[1].set_ylabel('concurrence of the bin-averaged state')
axes[1].set_title('(b) entanglement vs basis (lines: prediction)')
axes[1].legend(fontsize=7, loc='upper left')
axes[1].set_ylim(-0.02, 0.95)

allrow = r['bases']['boosted_beam'][0]
lab = ['$B^-_n$', '$B^-_r$', '$B^-_k$', '$B^+_n$', '$B^+_r$', '$B^+_k$']
meas = np.array(allrow['Bm_meas'] + allrow['Bp_meas'])
err = np.array(allrow['Bm_err'] + allrow['Bp_err'])
pred = np.array(allrow['Bm_pred'] + allrow['Bp_pred'])
xx = np.arange(6)
axes[2].errorbar(xx - 0.1, meas, err, fmt='o', color='k', ms=4, capsize=2, label='measured (exact $h$)')
axes[2].plot(xx + 0.1, pred, '_', color='C1', ms=14, mew=2, label=f"predicted, $\\sin^2\\theta_W$ = {r['sin2w_used']:.4f} (generator)")
axes[2].axhline(0.1466, color='C2', ls='--', lw=1, label=r'$-A_\tau$ with $\sin^2\theta_{\rm eff}$ = 0.2315')
axes[2].set_xticks(xx, lab)
axes[2].set_ylabel('single-τ polarisation $B$ (physical sign)')
axes[2].set_title('(c) single-τ polarisation, all events')
axes[2].legend(fontsize=7, loc='upper left')
axes[2].set_ylim(-0.05, 0.32)
fig.text(0.5, -0.02, f"v3 Z (pp→Zj→ττ full ME), raw LHE before filter, test half: {allrow['n']:,} events, unit weight. "
         f"Prediction: Z spin density fitted per (pair $q_T$ × $|y|$ × parton channel) class from τ directions on the other half.",
         ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(outdir / 'fig_z_tomography.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------------- CP-angle distributions ----------------
res = json.loads((cp_dir / 'cp_angle_results.json').read_text())
arr = np.load(cp_dir / 'cp_angle_arrays.npz')
S = {'h_v2': ('H (v2)', 'k', '-', 'o'), 'z_v2': ('Z v2 (two-step)', 'C0', '--', 's'), 'z_v3': ('Z v3 (full ME)', 'C3', '-.', '^')}
edges = np.linspace(0, 2 * np.pi, 25)
cen = 0.5 * (edges[1:] + edges[:-1])
fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
panels = [('boosted_beam', 'dphi', r'$\Delta\phi=\phi^-_h-\phi^+_h$', '(a) difference angle of transverse $h$\n(ideal CP angle; $\\propto C_{nn}+C_{rr}$)'),
          ('boosted_beam', 'sphi', r'$\Sigma\phi=\phi^-_h+\phi^+_h$ (boosted-beam axis)', '(b) sum angle, boosted-beam basis\n($\\propto C_{nn}-C_{rr}$)'),
          ('canonical_lab_beam', 'sphi', r'$\Sigma\phi$ (lab-beam axis, canonical)', '(c) sum angle, canonical basis\n(basis of the past runs)')]
for ax, (basis, var, xl, title) in zip(axes[:3], panels):
    for t, (lab, col, ls, mk) in S.items():
        v = arr[f'{t}_{basis}_{var}']
        h, _ = np.histogram(v, edges)
        d = h / h.mean()
        e = np.sqrt(h) / h.mean()
        amp, aerr = res[t][basis][f'{var}_cos_amp']
        ax.errorbar(cen, d, e, fmt=mk, color=col, ms=3, capsize=0)
        ax.plot(cen, 1 + amp * np.cos(cen), ls, color=col, lw=1.2, label=f'{lab}: $a$ = {amp:+.3f} ± {aerr:.3f}')
    ax.set_xlabel(xl + ' [rad]')
    ax.set_ylabel('events / mean')
    ax.set_title(title)
    ax.set_ylim(0.2, 1.9)
    ax.legend(fontsize=7, loc='upper center')
ax = axes[3]
for t, (lab, col, ls, mk) in S.items():
    for key, fill in (('phistar_rhorho', True), ('phistar_pipi_ip', False)):
        v = arr[f'{t}_{key}']
        v = v[np.isfinite(v)]
        h, _ = np.histogram(v, edges)
        d = h / h.mean()
        amp, aerr = res[t][key]['cos_amp']
        name = 'ρρ' if key == 'phistar_rhorho' else 'ππ (IP)'
        ax.plot(cen, d, ls if fill else ':', color=col, marker=mk if fill else None, ms=3, lw=1.1,
                label=f'{lab} {name}: $a$ = {amp:+.2f} ± {aerr:.2f}')
ax.set_xlabel(r'$\varphi^*_{CP}$ [rad]')
ax.set_ylabel('events / mean')
ax.set_title('(d) experimental $\\varphi^*_{CP}$ (truth 4-vectors)\nρρ: π⁰ decay plane, $y^+y^-$ flip; ππ: ideal IP')
ax.set_ylim(0.2, 2.35)
ax.legend(fontsize=6.5, loc='upper center', ncol=1)
n = {t: res[t]['n_good_h'] for t in S}
fig.text(0.5, -0.03, 'LHE truth after the common filter (both τ vis. $p_T$ > 18 GeV, |η| < 2.6), π/ρ/3π on both sides, unit weight; '
         f"H {n['h_v2']:,}, Z v2 {n['z_v2']:,}, Z v3 {n['z_v3']:,} events. Lines in (a)–(c): $1+a\\cos x$ with $a=2\\langle\\cos x\\rangle$.",
         ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(outdir / 'fig_cp_angles.png', dpi=150, bbox_inches='tight')
print('written', outdir)
