"""Figures for rho_beyond_exact from the result JSONs (run where matplotlib exists; lxgpu02 venv).

fig1_rho_auc_ladder.png      H/Z AUC (formula readout unless noted) of the reco-visible + truth-nu ladder, the denoised
                             reference and the learner, for pi x rho, rho x rho, rho x 3pi
fig2_rho_side_quality.png    rho-side h_k correlation with the exact h vs the shell violation (s7) and vs the pi0 angle
                             error (s3)
fig3_cross_side.png          composition control: cross-side coefficient b and the Z rho x rho plug-in correlation
                             (only if s6_composition.json exists)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e4'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})


def fig1(R, out):
    s2, s4, s7 = (json.loads((R / f).read_text()) for f in ('s2_readouts.json', 's4_eval.json', 's7_shell_refit.json'))
    a2, a4, a7 = s2['auc'], s4['auc'], s7['auc']
    rows = [  # label, source dict, key, group (0 reco+truth nu ladder, 1 refit/denoised, 2 learner, 3 exact)
        ('exact h (truth visible + truth ν)', a2, 'exact formula', 3),
        ('reco ρ + truth ν: truth π⁰ (direction + energy)', a2, 'ladder pi0 formula', 0),
        ('reco ρ + truth ν: denoised E[h | reco ρ, ν] (MLP, train)', a4, 'denoised formula', 1),
        ('reco ρ + truth ν: τ-mass-shell π⁰ refit', a7, 'formula shell refit', 1),
        ('reco ρ + truth ν: truth π⁰ direction', a2, 'ladder pi0dir formula', 0),
        ('reco ρ + truth ν: shell refit, direction only', a7, 'formula shell refit (dir only)', 1),
        ('learner (gen3pi full22, 2 seeds, own readout)', a2, 'learner ens_full22 readout', 2),
        ('reco ρ + truth ν: truth π⁰ energy', a2, 'ladder pi0E formula', 0),
        ('reco ρ + truth ν: truth π±', a2, 'ladder pich formula', 0),
        ('reco ρ + truth ν, trained MLP readout', a2, 'reco+truth nu mlp', 0),
        ('reco ρ + truth ν (10/17 reference)', a2, 'reco+truth nu formula', 0),
    ]
    style = {0: (C[0], 'o'), 1: (C[2], 's'), 2: (C[1], 'D'), 3: (MUTED, '^')}
    pairs = ['pi x rho', 'rho x rho', 'rho x 3pi']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, p in zip(axes, pairs):
        for yi, (lab, src, key, g) in zip(y, rows):
            col, mk = style[g]
            ax.plot(src[key][p], yi, marker=mk, color=col, ms=8, ls='none', mec='white', mew=1.2)
        ref = a2['reco+truth nu formula'][p]
        ax.axvline(ref, color=C[0], lw=1, ls=':', alpha=0.7)
        ax.axvline(a2['learner ens_full22 readout'][p], color=C[1], lw=1, ls='--', alpha=0.7)
        ax.set_title(p.replace(' x ', ' × ').replace('rho', 'ρ').replace('pi', 'π').replace('3π', '3π'), color=INK)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_xlabel('weighted H/Z AUC (validation)')
        ax.set_xlim(0.59, 0.745)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.5)
    handles = [plt.Line2D([], [], color=style[g][0], marker=style[g][1], ls='none', ms=8, label=l) for g, l in
               ((0, 'exact formula on reco ρ + truth ν, constituents replaced by truth'),
                (1, 'π⁰ corrected with truth ν (learned / classical refit)'),
                (2, 'main-line learner (no truth ν)'), (3, 'exact h'))]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle('ATLAS full-reco H/Z validation (59,390 events, parent-overlap weights): what limits the exact formula on reco ρ',
                 color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / 'fig1_rho_auc_ladder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig2(R, out):
    s3, s7 = (json.loads((R / f).read_text()) for f in ('s3_side_quality.json', 's7_shell_refit.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    sb = s7['shell_bins']
    e = np.array(sb['edges_GeV'])
    x = np.arange(len(e) - 1)
    spec7 = [('formula reco', 'exact formula, reco ρ + truth ν', C[0], 'o', '-'),
             ('formula truth pi0 dir', '  same, truth π⁰ direction', C[0], 'o', ':'),
             ('formula shell refit', '  same, τ-mass-shell π⁰ refit', C[2], 's', '-'),
             ('denoised', 'denoised E[h | reco ρ, truth ν]', C[2], 's', '--'),
             ('learner full22', 'learner (no truth ν)', C[1], 'D', '-')]
    for key, lab, col, mk, ls in spec7:
        axes[0].plot(x, sb['rows'][key], color=col, marker=mk, ls=ls, lw=2, ms=7, mec='white', label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{e[i]:.2f}–{e[i + 1]:.2f}' if i < len(e) - 2 else f'>{e[i]:.2f}' for i in x], fontsize=8)
    axes[0].set_xlabel('|m(π± + π⁰ + ν_truth) − m_τ| sextile [GeV]')
    axes[0].set_ylabel('weighted corr(h_k, exact h_k), ρ sides')
    axes[0].set_title('vs τ-mass-shell violation of the reco π⁰', color=INK, fontsize=10)
    d = s3['binned']['dtheta_pi0']
    e2 = np.array(d['edges'])
    for key, lab, col, mk, ls in [('formula reco', 'exact formula, reco ρ + truth ν', C[0], 'o', '-'),
                                  ('formula pi0dir', '  same, truth π⁰ direction', C[0], 'o', ':'),
                                  ('formula pi0E', '  same, truth π⁰ energy', C[3], 'v', ':'),
                                  ('learner full22', 'learner full22 (no truth ν)', C[1], 'D', '-'),
                                  ('learner base', 'learner base (no IP/SV)', C[1], 'D', '--')]:
        axes[1].plot(x, d['rows'][key]['k'], color=col, marker=mk, ls=ls, lw=2, ms=7, mec='white', label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{e2[i]:.1f}–{e2[i + 1]:.1f}' if i < len(e2) - 2 else f'>{e2[i]:.1f}' for i in x], fontsize=8)
    axes[1].set_xlabel('angle(reco π⁰, truth π⁰) sextile [mrad]')
    axes[1].set_title('vs π⁰ direction error', color=INK, fontsize=10)
    for ax in axes:
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
        ax.set_ylim(0.2, 1.0)
    fig.suptitle(f"ρ sides (truth ρ, reco ρ; {s7['quality_sides']:,} sides, validation, parent-overlap weights); each bin holds 1/6 of the sides",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'fig2_rho_side_quality.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig3(R, out):
    p = R / 's6_composition.json'
    if not p.exists():
        return
    s6 = json.loads(p.read_text())
    arms = [a for a in ('Honly', 'mix', 'mixhalf', 'Zonly', 'flatw', 'full22') if a in s6]
    names = {'mix': 'mixed\n(all)', 'mixhalf': 'mixed\n(½)', 'Honly': 'H only\n(½)', 'Zonly': 'Z only\n(½)',
             'flatw': 'spin-flat\nweighted (all)', 'full22': 'main line\n(+IP/SV)'}
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    ax = axes[0]
    for j, (ln, comp, col, mk) in enumerate([('H', 'n', C[0], 'o'), ('Z', 'n', C[1], 'o'),
                                             ('H', 'k', C[0], 's'), ('Z', 'k', C[1], 's')]):
        v = np.array([s6[a]['cross']['rho'][ln][comp] for a in arms])
        off = (j - 1.5) * 0.1
        ax.errorbar(x + off, v[:, 1], yerr=[v[:, 1] - v[:, 2], v[:, 3] - v[:, 1]], color=col, marker=mk, ls='none',
                    ms=7, mec='white', capsize=0, label=f'{ln} test events, h_{comp}')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel('b = coefficient of the other τ’s exact h')
    ax.set_title('ρ sides: h_pred[c] = a·h[c] + b·h_other[c] + const', color=INK, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    for ax, ln, title in ((axes[1], 'Z', 'Z→ττ, truth ρ×ρ'), (axes[2], 'H', 'H→ττ, truth ρ×ρ')):
        ex = s6['exact']['plugin'][f'rho x rho {ln}']
        for i, (comp, mk, col) in enumerate((('n', 'o', C[2]), ('r', '^', C[3]), ('k', 's', C[4]))):
            v = [s6[a]['plugin'][f'rho x rho {ln}'][i] for a in arms]
            ax.plot(x + (i - 1) * 0.12, v, marker=mk, color=col, ls='none', ms=8, mec='white', label=f'C_{comp}{comp} from h_pred')
            ax.axhline(ex[i], color=col, lw=1.2, ls=(0, (4, 3)), alpha=0.9)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel('plug-in 9⟨h⁻_c h⁺_c⟩')
        ax.set_title(f'{title}: pair correlation read off h_pred\n(dashed: same quantity with the exact h)', color=INK, fontsize=9.5)
        ax.set_ylim(-1.05, 1.15)
    axes[1].legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([names[a] for a in arms], fontsize=7.5)
        ax.grid(axis='y', color=GRID, lw=0.8)
    fig.suptitle('Training composition of the point-h network (geometry hidden, seed 43; full22 = main line, 2 seeds) — validation, parent-overlap weights',
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'fig3_cross_side.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    fig1(a.results, a.output)
    fig2(a.results, a.output)
    fig3(a.results, a.output)
