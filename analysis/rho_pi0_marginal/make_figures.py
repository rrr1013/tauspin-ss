"""Figures of the pi0-marginalised classical posterior run (results/ in this directory).

fig1_auc_pairs.png      weighted H/Z AUC by truth mode pair: classical fixed / marginalised / truth-pi0 oracle, learner
fig2_pit_train.png      TRAIN PIT of the truth pi0 energy, visible mass and tau energy (invE prior; isotropic vs anisotropic response)
fig3_rho_hk_bins.png    rho-side h_k quality vs the reco pi0 angle error (truth diagnostic) and the reco visible mass
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'atlas_classical_posterior'))
from evaluate import GEN  # noqa: E402
from hquality import wcorr  # noqa: E402

LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
SURFACE = GEN / 'targets' / 'truth_surface_gen3pi.npz'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
ROWS = [  # (label, score key, color, marker, filled)
    ('classical, reco π⁰ fixed (= 10/17)', 'fixed_invE:ip3_sv_mass', C[1], 'o', False),
    ('classical, π⁰ marginalised (primary)', 'marg_invE:ip3_sv_mass', C[1], 'o', True),
    ('classical, π⁰ marginalised, TRAIN E_τ prior', 'marg_train:ip3_sv_mass', C[3], 's', True),
    ('classical, π⁰ marginalised, separation-dependent response', 'cond_invE:ip3_sv_mass', C[4], 'v', True),
    ('classical, π⁰ marg., + anisotropic direction response', 'psi_invE:ip3_sv_mass', C[5], 'P', True),
    ('classical, truth π⁰ (oracle)', 'truth_train:ip3_sv_mass', C[2], 'D', False),
    ('learner, no IP/SV (base)', 'learner:base_s43', C[0], '^', False),
    ('learner, main line (full22, 2 seeds)', 'learner:ens_full22', C[0], '^', True),
]


def fig1(res, out):
    pairs = ['overall', 'pi x rho', 'rho x rho', 'rho x 3pi']
    names = {'overall': 'all events', 'pi x rho': 'π × ρ', 'rho x rho': 'ρ × ρ', 'rho x 3pi': 'ρ × 3π'}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for j, (lab, key, col, mk, fill) in enumerate(ROWS):
        if key not in res['auc']:
            continue
        yy = np.arange(len(pairs)) + (j - (len(ROWS) - 1) / 2) * 0.12
        xx = [res['auc'][key][p] for p in pairs]
        ax.plot(xx, yy, ls='none', marker=mk, ms=8, mec=col, mfc=col if fill else 'white', mew=1.6, label=lab)
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels([f"{names[p]}\n({res['events_truth_pairs'][p]:,} events)" for p in pairs])
    ax.invert_yaxis()
    ax.set_xlabel('weighted H/Z AUC (validation, parent-overlap weights)')
    ax.grid(axis='x', color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False)
    ax.set_title('ATLAS full-reco H/Z (both 91.19 GeV), by truth decay-mode pair', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fig1_auc_pairs.png', dpi=150)


def fig2(pit_files, out):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    qn = {'E2': 'true π⁰ energy', 'MV': 'true visible mass m(π±π⁰)', 'ET': 'true τ energy'}
    styles = [('invE', 'met', C[1], '-', 'isotropic response: prior + MET'),
              ('invE', 'ip3_sv_mass', C[1], '--', 'isotropic response: + IP, SV, parent mass'),
              ('psi', 'met', C[5], '-', 'anisotropic response: prior + MET'),
              ('psi', 'ip3_sv_mass', C[5], '--', 'anisotropic response: + IP, SV, parent mass')]
    for ax, q in zip(axs, ('E2', 'MV', 'ET')):
        for prior, m, col, ls, lab in styles:
            P = np.load(pit_files[prior])
            side = P['marg_side'] & P['truth_rho_side']
            v = np.concatenate([P[f'{m}_pit{q}_{s}'][side[:, s]] for s in (0, 1)])
            h, e = np.histogram(v, np.linspace(0, 1, 11))
            ax.step(np.r_[e[:-1], 1.0], np.r_[h, h[-1]] / h.sum() * 10, where='post', color=col, ls=ls, lw=2, label=lab)
        ax.axhline(1.0, color='#888888', lw=0.8)
        ax.set_xlabel(f'posterior CDF at the {qn[q]}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, 1.7)
    axs[0].set_ylabel('density (calibrated = 1)')
    axs[0].legend(fontsize=7.5, loc='upper center')
    n = int((np.load(pit_files['invE'])['marg_side'] & np.load(pit_files['invE'])['truth_rho_side']).sum())
    fig.suptitle(f'PIT on TRAIN held-out rows (truth ρ & reco ρ sides, {n:,}); 10 bins', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fig2_pit_train.png', dpi=150)


def fig3(val, out):
    V = np.load(LADDER / 'validation.npz')
    S = np.load(SURFACE)
    t = np.load(GEN / 'targets' / 'validation.npz')
    v = t['h_valid'].astype(bool)
    hx = t['h'][v].astype(float)
    w = V['weights'].astype(float)
    side = V['reco_h_available_side'] & (V['reco_h_mode'] == 1) & (V['modes'] == 1)
    rn, tp = V['reco_h_neutral_lab4'], S['pi0']
    cosa = np.sum(rn[..., :3] * tp[..., :3], -1) / np.maximum(np.linalg.norm(rn[..., :3], axis=-1) * np.linalg.norm(tp[..., :3], axis=-1), 1e-300)
    ang = np.arccos(np.clip(cosa, -1, 1)) * 1e3
    pv = V['reco_visible_tau_lab4']
    mv = np.sqrt(np.maximum(pv[..., 3] ** 2 - np.sum(pv[..., :3] ** 2, -1), 0))
    rows = [('classical, reco π⁰ fixed', np.load(val / 'val_fixed_invE' / 'posterior.npz')['ip3_sv_mass_Eh'], C[1], '-', 'o', False),
            ('classical, π⁰ marginalised', np.load(val / 'val_marg_invE' / 'posterior.npz')['ip3_sv_mass_Eh'], C[1], '-', 'o', True),
            ('classical, π⁰ marg., separation-dependent response', np.load(val / 'val_cond_invE' / 'posterior.npz')['ip3_sv_mass_Eh'], C[4], '-', 'v', True),
            ('classical, π⁰ marg., + anisotropic direction response', np.load(val / 'val_psi_invE' / 'posterior.npz')['ip3_sv_mass_Eh'], C[5], '-', 'P', True),
            ('classical, truth π⁰ (oracle)', np.load(val / 'val_truth_train' / 'posterior.npz')['ip3_sv_mass_Eh'], C[2], '--', 'D', False),
            ('learner full22 s42', np.load(GEN / 'p11' / 'full22_s42' / 'validation_predictions.npz')['h_pred'].astype(float), C[0], '-', '^', True)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, x, xl in ((axs[0], ang, 'reco π⁰ – truth π⁰ angle [mrad] (truth diagnostic)'),
                      (axs[1], mv, 'reco visible mass m(π± + π⁰ PFO) [GeV]')):
        q = np.quantile(x[side], np.linspace(0, 1, 7))
        b = np.clip(np.searchsorted(q, x) - 1, 0, 5)
        cen = [np.median(x[side & (b == i)]) for i in range(6)]
        for lab, h, col, ls, mk, fill in rows:
            ok = side & np.isfinite(h).all(-1)
            ww = np.broadcast_to(w[:, None], side.shape)
            yy = [wcorr(h[..., 2][ok & (b == i)], hx[..., 2][ok & (b == i)], ww[ok & (b == i)]) for i in range(6)]
            ax.plot(cen, yy, color=col, ls=ls, marker=mk, ms=7, mfc=col if fill else 'white', lw=1.8, label=lab)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([round(c, 2 if c < 3 else 0) for c in cen])
        ax.set_xlabel(xl)
        ax.grid(color='#eeeeee', lw=0.6)
    axs[0].set_ylabel('corr(h_k, exact h_k), ρ side')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'Validation truth ρ & reco ρ sides ({int(side.sum()):,}), sextiles of the x variable; IP+SV+MET+mass posterior', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fig3_rho_hk_bins.png', dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = json.loads((a.run_dir / 'out' / 'eval_val.json').read_text())
    fig1(res, a.out)
    fig2({'invE': a.run_dir / 'out' / 'pit_train_marg_invE' / 'posterior.npz',
          'psi': a.run_dir / 'out' / 'pit_train_psi_invE' / 'posterior.npz'}, a.out)
    fig3(a.run_dir / 'out', a.out)


if __name__ == '__main__':
    main()
