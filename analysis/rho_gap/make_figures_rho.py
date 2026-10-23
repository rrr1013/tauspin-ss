"""Generate figures for tauspin-rho-transverse-gap-20261024 run:
1. fig1_learner_rho_intervention.png: Learner sensitivity to IP rotation/flip/shuffle on pi vs rho,
   showing the dual nature of rho transverse spin (visible decay plane baseline + IP boost).
2. fig2_rho_oracle_matrix.png: 2x2 oracle matrix (reco/truth pi0 x reco/truth psi_tau) for rho transverse correlation and AUC.
3. fig3_pi0_decomposition.png: Breakdown of pi0 oracle (reco vs truth direction vs truth energy vs full truth).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'font.family': 'sans-serif'
})


def plot_learner_interventions(intervene_json_path, out_path):
    data = json.load(open(intervene_json_path))
    # Comparison of interventions on pi vs rho
    # Variants to plot: identity, rot90, flip, shuffle_own, pfo_zero_rho, pfo_shuffle_rho
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # 1. pi sides
    pi_variants = [
        ('identity', 'Nominal (full22)'),
        ('rot90_pi', 'IP rot +90°'),
        ('flip_pi', 'IP flip 180°'),
        ('shuffle_own_pi', 'IP shuffle'),
        ('tok_d0z0_zero_pi', 'Track d0,z0=0'),
    ]
    labels_pi = [lbl for _, lbl in pi_variants]
    hn_pi = [data[v]['pi']['hn'] for v, _ in pi_variants]
    hr_pi = [data[v]['pi']['hr'] for v, _ in pi_variants]
    hk_pi = [data[v]['pi']['hk'] for v, _ in pi_variants]

    x = np.arange(len(pi_variants))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, hn_pi, width=w, label=r'$h_n$ (transverse)', color='#1f77b4')
    ax.bar(x, hr_pi, width=w, label=r'$h_r$ (transverse)', color='#aec7e8')
    ax.bar(x + w, hk_pi, width=w, label=r'$h_k$ (longitudinal)', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pi, rotation=35, ha='right')
    ax.set_ylabel(r'Weighted correlation with exact $h$')
    ax.set_title(r'$\tau \to \pi\nu$ (single track, no visible decay plane)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='lower left')

    # 2. rho sides
    rho_variants = [
        ('identity', 'Nominal (full22)'),
        ('rot90_rho', 'IP rot +90°'),
        ('flip_rho', 'IP flip 180°'),
        ('shuffle_own_rho', 'IP shuffle'),
        ('pfo_zero_rho', 'PFO zeroed'),
        ('pfo_shuffle_rho', 'PFO shuffled'),
    ]
    labels_rho = [lbl for _, lbl in rho_variants]
    hn_rho = [data[v]['rho']['hn'] for v, _ in rho_variants]
    hr_rho = [data[v]['rho']['hr'] for v, _ in rho_variants]
    hk_rho = [data[v]['rho']['hk'] for v, _ in rho_variants]

    x_rho = np.arange(len(rho_variants))
    ax = axes[1]
    ax.bar(x_rho - w, hn_rho, width=w, label=r'$h_n$', color='#1f77b4')
    ax.bar(x_rho, hr_rho, width=w, label=r'$h_r$', color='#aec7e8')
    ax.bar(x_rho + w, hk_rho, width=w, label=r'$h_k$', color='#2ca02c')
    ax.set_xticks(x_rho)
    ax.set_xticklabels(labels_rho, rotation=35, ha='right')
    ax.set_title(r'$\tau \to \rho\nu \to \pi^\pm\pi^0\nu$ (visible decay plane + IP)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved {out_path}')


def plot_oracle_matrix(matrix_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Left: rho transverse correlation h_n / h_r
    # Rows: Reco pi0, Truth pi0
    # Cols: Reco psi_tau, Truth psi_tau
    # Data format: dict with keys 'reco_pi0_reco_psi', 'reco_pi0_truth_psi', 'truth_pi0_reco_psi', 'truth_pi0_truth_psi'
    hn_matrix = np.array([
        [matrix_data['reco_pi0_reco_psi']['hn'], matrix_data['reco_pi0_truth_psi']['hn']],
        [matrix_data['truth_pi0_reco_psi']['hn'], matrix_data['truth_pi0_truth_psi']['hn']]
    ])
    hr_matrix = np.array([
        [matrix_data['reco_pi0_reco_psi']['hr'], matrix_data['reco_pi0_truth_psi']['hr']],
        [matrix_data['truth_pi0_reco_psi']['hr'], matrix_data['truth_pi0_truth_psi']['hr']]
    ])
    auc_matrix = np.array([
        [matrix_data['reco_pi0_reco_psi']['auc_rhorho'], matrix_data['reco_pi0_truth_psi']['auc_rhorho']],
        [matrix_data['truth_pi0_reco_psi']['auc_rhorho'], matrix_data['truth_pi0_truth_psi']['auc_rhorho']]
    ])

    im1 = axes[0].imshow(0.5 * (hn_matrix + hr_matrix), cmap='viridis', vmin=0.55, vmax=0.85)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Reco $\\psi_\\tau$ (IP)', 'Truth $\\psi_\\tau$'])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Reco $\\pi^0$ (marg)', 'Truth $\\pi^0$'])
    axes[0].set_title(r'Reco $\rho$ Transverse Spin Correlation $\frac{h_n + h_r}{2}$')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{hn_matrix[i, j]:.3f} / {hr_matrix[i, j]:.3f}\n(mean: {0.5*(hn_matrix[i,j]+hr_matrix[i,j]):.3f})',
                         ha='center', va='center', color='white' if 0.5*(hn_matrix[i,j]+hr_matrix[i,j]) < 0.75 else 'black',
                         fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(auc_matrix, cmap='magma', vmin=0.59, vmax=0.68)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Reco $\\psi_\\tau$ (IP)', 'Truth $\\psi_\\tau$'])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Reco $\\pi^0$ (marg)', 'Truth $\\pi^0$'])
    axes[1].set_title(r'$\rho \times \rho$ Weighted H/Z AUC')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{auc_matrix[i, j]:.4f}',
                         ha='center', va='center', color='white' if auc_matrix[i,j] < 0.65 else 'black',
                         fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved {out_path}')


def plot_pi0_decomposition(decomp_data, out_path):
    # Compares:
    # 1. reco pi0 (marg)
    # 2. truth pi0 energy (reco dir)
    # 3. truth pi0 dir (reco energy)
    # 4. full truth pi0
    # Against learner full22 and learner base
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [
        'Reco $\\pi^0$ (marg)',
        'Truth $\\pi^0$ Energy\n(reco dir)',
        'Truth $\\pi^0$ Direction\n(reco E)',
        'Full Truth $\\pi^0$',
        'Learner base\n(no IP/SV)',
        'Learner full22\n(nominal)'
    ]
    hn = [decomp_data[k]['hn'] for k in ('reco', 'truthe', 'truthdir', 'truth', 'learner_base', 'learner_full22')]
    hr = [decomp_data[k]['hr'] for k in ('reco', 'truthe', 'truthdir', 'truth', 'learner_base', 'learner_full22')]
    hk = [decomp_data[k]['hk'] for k in ('reco', 'truthe', 'truthdir', 'truth', 'learner_base', 'learner_full22')]

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, hn, width=w, label=r'$h_n$ (transverse)', color='#1f77b4')
    ax.bar(x, hr, width=w, label=r'$h_r$ (transverse)', color='#aec7e8')
    ax.bar(x + w, hk, width=w, label=r'$h_k$ (longitudinal)', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Weighted correlation with exact $h$')
    ax.set_title(r'$\tau \to \rho\nu$: Orthogonal Decomposition of $\pi^0$ Direction vs Energy')
    ax.set_ylim(0.4, 0.95)
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='upper left')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--intervene-json', type=Path, default=Path('out/eval_learner_intervene.json'))
    parser.add_argument('--matrix-json', type=Path, default=Path('out/eval_matrix_full.json'))
    parser.add_argument('--figures-dir', type=Path, default=Path('figures'))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.intervene_json.exists():
        plot_learner_interventions(args.intervene_json, args.figures_dir / 'fig1_learner_rho_intervention.png')

    if args.matrix_json.exists():
        data = json.load(open(args.matrix_json))
        # 1. Oracle matrix data
        matrix_data = {}
        arms = ['reco_pi0_reco_psi', 'reco_pi0_truth_psi', 'truth_pi0_reco_psi', 'truth_pi0_truth_psi']
        for arm in arms:
            val_name = f'val_{arm}'
            sq = data['side_quality'][val_name]['rho']
            auc = data['auc'][val_name]['pairs']['rho x rho']
            matrix_data[arm] = {
                'hn': sq['hn'],
                'hr': sq['hr'],
                'hk': sq['hk'],
                'auc_rhorho': auc
            }
        plot_oracle_matrix(matrix_data, args.figures_dir / 'fig2_rho_oracle_matrix.png')

        # 2. Pi0 decomposition data
        sq = data['side_quality']
        decomp_data = {
            'reco': sq['val_reco_pi0_reco_psi']['rho'],
            'truthe': sq['val_truthe_pi0_reco_psi']['rho'],
            'truthdir': sq['val_truthdir_pi0_reco_psi']['rho'],
            'truth': sq['val_truth_pi0_reco_psi']['rho'],
            'learner_base': {'hn': 0.6188, 'hr': 0.6227, 'hk': 0.7410},
            'learner_full22': sq['learner_s42']['rho']
        }
        plot_pi0_decomposition(decomp_data, args.figures_dir / 'fig3_pi0_decomposition.png')


if __name__ == '__main__':
    main()

