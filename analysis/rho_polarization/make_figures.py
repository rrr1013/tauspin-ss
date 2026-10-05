"""Make publication-quality figures for the rho polarization ARIADNE run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    res_dir = Path('analysis/rho_polarization/results')
    fig_dir = Path('analysis/rho_polarization/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    with open(res_dir / 'summary.json') as f:
        summary = json.load(f)
    
    arrays = np.load(res_dir / 'arrays.npz')
    
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14

    # -------------------------------------------------------------
    # Figure 1: Per-side polarimeter quality & detector asymmetry vs y
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    y_bins = summary['per_side_y_bins']
    y_centers = [b['y_center'] for b in y_bins]
    corr_long_full = [b['full22']['corr_longitudinal'] for b in y_bins]
    corr_trans_full = [b['full22']['corr_transverse'] for b in y_bins]
    corr_long_base = [b['base']['corr_longitudinal'] for b in y_bins]
    corr_trans_base = [b['base']['corr_transverse'] for b in y_bins]
    corr_long_ideal = [b['ideal']['corr_longitudinal'] for b in y_bins]
    corr_trans_ideal = [b['ideal']['corr_transverse'] for b in y_bins]
    norm_ratio_full = [b['full22']['norm_ratio'] for b in y_bins]
    norm_ratio_base = [b['base']['norm_ratio'] for b in y_bins]
    unphys_frac = [b['unphysical_mass_fraction'] * 100 for b in y_bins]
    
    # 1(a): Longitudinal correlation vs y
    ax = axes[0, 0]
    ax.plot(y_centers, corr_long_full, 'o-', color='#1f77b4', lw=2, label='Reco +IP/SV ($h_k$)')
    ax.plot(y_centers, corr_long_base, 's--', color='#7f7f7f', lw=1.5, label='Reco Base ($h_k$)')
    ax.plot(y_centers, corr_long_ideal, '^:', color='#2ca02c', lw=1.5, label='Reco Ideal IP ($h_k$)')
    ax.set_xlabel(r'Energy asymmetry $y = (E_{\pi^\pm} - E_{\pi^0})/(E_{\pi^\pm} + E_{\pi^0})$')
    ax.set_ylabel(r'Correlation with exact $h_k$ (Longitudinal)')
    ax.set_ylim(0.1, 0.85)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='upper center')
    ax.set_title(r'(a) Longitudinal polarimeter correlation $r(h_k^\mathrm{pred}, h_k^\mathrm{exact})$')
    
    # 1(b): Transverse correlation vs y
    ax = axes[0, 1]
    ax.plot(y_centers, corr_trans_full, 'o-', color='#d62728', lw=2, label='Reco +IP/SV ($h_\perp$)')
    ax.plot(y_centers, corr_trans_base, 's--', color='#7f7f7f', lw=1.5, label='Reco Base ($h_\perp$)')
    ax.plot(y_centers, corr_trans_ideal, '^:', color='#2ca02c', lw=1.5, label='Reco Ideal IP ($h_\perp$)')
    ax.set_xlabel(r'Energy asymmetry $y = (E_{\pi^\pm} - E_{\pi^0})/(E_{\pi^\pm} + E_{\pi^0})$')
    ax.set_ylabel(r'Correlation with exact $h_\perp$ (Transverse)')
    ax.set_ylim(0.3, 0.9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='lower center')
    ax.set_title(r'(b) Transverse polarimeter correlation $r(h_\perp^\mathrm{pred}, h_\perp^\mathrm{exact})$')

    # 1(c): Predicted norm shrinkage vs y
    ax = axes[1, 0]
    ax.plot(y_centers, norm_ratio_full, 'o-', color='#1f77b4', lw=2, label='Reco +IP/SV')
    ax.plot(y_centers, norm_ratio_base, 's--', color='#7f7f7f', lw=1.5, label='Reco Base')
    ax.axhline(1.0, color='black', ls=':', alpha=0.5, label='Exact truth norm (=1.0)')
    ax.set_xlabel(r'Energy asymmetry $y = (E_{\pi^\pm} - E_{\pi^0})/(E_{\pi^\pm} + E_{\pi^0})$')
    ax.set_ylabel(r'Average norm ratio $\langle |\vec{h}_\mathrm{pred}| \rangle / \langle |\vec{h}_\mathrm{exact}| \rangle$')
    ax.set_ylim(0.55, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='lower center')
    ax.set_title(r'(c) Polarimeter vector norm shrinkage')

    # 1(d): Detector asymmetry: unphysical mass fraction & resolution
    ax = axes[1, 1]
    ax.bar(y_centers, unphys_frac, width=0.16, color='#ff7f0e', alpha=0.7, edgecolor='#d62728', label=r'$m_\mathrm{vis} > m_\tau$ fraction (%)')
    ax.set_xlabel(r'Energy asymmetry $y = (E_{\pi^\pm} - E_{\pi^0})/(E_{\pi^\pm} + E_{\pi^0})$')
    ax.set_ylabel(r'Unphysical fraction $m_\mathrm{vis} > m_\tau$ (%)', color='#d62728')
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax.set_ylim(0, 16)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'(d) Kinematic boundary crossing ($m_\mathrm{vis} > 1.777$ GeV)')

    # Add text box explaining physical significance
    ax.text(-0.85, 13.5, r'$\leftarrow$ Hard $\pi^0$ (clean shower)', color='blue', fontsize=10)
    ax.text(0.15, 13.5, r'Hard $\pi^\pm$ (soft $\pi^0$ error) $\rightarrow$', color='darkred', fontsize=10)

    plt.tight_layout()
    fig1_path = fig_dir / 'fig1_per_side_polarimeter_quality.png'
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: 2D Phase Space and Category AUC
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 2(a): 2D Grid of AUC in (|y0|, |y1|)
    ax = axes[0]
    grid_data = summary['grid_2d']
    matrix_exact = np.zeros((3, 3))
    matrix_full = np.zeros((3, 3))
    counts = np.zeros((3, 3))
    for entry in grid_data:
        i = 0 if entry['bin0'][0] == 0.0 else (1 if entry['bin0'][0] == 0.35 else 2)
        j = 0 if entry['bin1'][0] == 0.0 else (1 if entry['bin1'][0] == 0.35 else 2)
        matrix_exact[i, j] = entry['auc_exact']
        matrix_full[i, j] = entry['auc_full22']
        counts[i, j] = entry['count']
    
    im = ax.imshow(matrix_full, cmap='YlGnBu', vmin=0.55, vmax=0.66, origin='lower')
    ax.set_xticks([0, 1, 2], labels=['[0, 0.35)', '[0.35, 0.65)', '[0.65, 1.0]'])
    ax.set_yticks([0, 1, 2], labels=['[0, 0.35)', '[0.35, 0.65)', '[0.65, 1.0]'])
    ax.set_xlabel(r'Tau 1 $|y|$ bin')
    ax.set_ylabel(r'Tau 0 $|y|$ bin')
    ax.set_title(r'(a) Reco +IP/SV H/Z AUC in $(|y_0|, |y_1|)$ grid')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='H/Z Weighted AUC')
    
    # Annotate cells with values
    for i in range(3):
        for j in range(3):
            val_f = matrix_full[i, j]
            val_e = matrix_exact[i, j]
            cnt = int(counts[i, j])
            color = 'white' if val_f > 0.61 else 'black'
            ax.text(j, i, f'Reco: {val_f:.3f}\nExact: {val_e:.3f}\n(N={cnt})',
                    ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    # 2(b): Category Bar Chart with Error Bars
    ax = axes[1]
    cats = summary['category_results']
    cat_keys = [
        'both_low_asym_symmetric',
        'both_mid_asym',
        'mixed_one_low_one_high',
        'both_high_asym_reco',
        'both_high_asym_truth',
        'both_hard_pich',
        'both_hard_pi0',
        'unphysical_m_vis_at_least_one'
    ]
    labels_clean = [
        r'Both low $|y| < 0.35$' + '\n(Symmetric, N=2207)',
        r'Both mid $|y| \in [0.35, 0.7)$' + '\n(N=2670)',
        'Mixed one low, one high\n(N=3626)',
        r'Reco selected $|y_\mathrm{rec}| \geq 0.7$' + '\n(N=1830)',
        r'Truth $|y| \geq 0.7$' + '\n(Pure long., N=1414)',
        r'Both hard $\pi^\pm$ ($y > 0.5$)' + '\n(N=1270)',
        r'Both hard $\pi^0$ ($y < -0.5$)' + '\n(N=1092)',
        r'Unphys $m_\mathrm{vis} > m_\tau$' + '\n(N=3285)'
    ]
    
    y_pos = np.arange(len(cat_keys))
    auc_f = [cats[k]['auc_full22'] for k in cat_keys]
    err_f_low = [cats[k]['auc_full22'] - cats[k]['auc_full22_ci'][0] for k in cat_keys]
    err_f_high = [cats[k]['auc_full22_ci'][1] - cats[k]['auc_full22'] for k in cat_keys]
    auc_e = [cats[k]['auc_exact'] for k in cat_keys]
    
    ax.barh(y_pos - 0.18, auc_f, height=0.35, xerr=[err_f_low, err_f_high],
            color='#1f77b4', alpha=0.85, label='Reco +IP/SV (95% CI)', capsize=3)
    ax.scatter(auc_e, y_pos + 0.18, color='#d62728', marker='D', s=45, label='Exact truth $h$', zorder=5)
    
    # Global benchmark lines
    ax.axvline(summary['auc_global_rho_rho']['full22'], color='#1f77b4', ls='--', alpha=0.6,
               label=f"Inclusive Reco ({summary['auc_global_rho_rho']['full22']:.3f})")
    ax.axvline(summary['auc_global_rho_rho']['exact'], color='#d62728', ls=':', alpha=0.6,
               label=f"Inclusive Exact ({summary['auc_global_rho_rho']['exact']:.3f})")

    ax.set_yticks(y_pos, labels=labels_clean)
    ax.set_xlabel('H/Z Separation AUC (weighted)')
    ax.set_xlim(0.52, 0.84)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(frameon=True, loc='lower right', fontsize=9)
    ax.set_title(r'(b) H/Z AUC across internal $\rho$ kinematic categories')

    plt.tight_layout()
    fig2_path = fig_dir / 'fig2_rho_rho_auc_categories.png'
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Detector Charge/Neutral Asymmetry & Mixed Modes
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 3(a): Angular & Energy resolution comparison vs y
    ax = axes[0]
    # compute median dtheta_pi0 and dE_rel
    yt_flat = arrays['y_all_truth']
    yr_flat = arrays['y_all_reco']
    # compute bins
    fine_bins = np.linspace(-1.0, 1.0, 11)
    centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])
    y_bias = []
    y_width = []
    for k in range(len(fine_bins) - 1):
        in_b = (yt_flat >= fine_bins[k]) & (yt_flat < fine_bins[k + 1])
        diff = yr_flat[in_b] - yt_flat[in_b]
        y_bias.append(np.median(diff))
        y_width.append(0.5 * (np.percentile(diff, 84) - np.percentile(diff, 16)))
    
    ax.errorbar(centers, y_bias, yerr=y_width, fmt='o-', color='#2ca02c', lw=2, capsize=3,
                label=r'$y_\mathrm{reco} - y_\mathrm{truth}$ (Median $\pm 68\%$)')
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel(r'Truth energy asymmetry $y_\mathrm{truth}$')
    ax.set_ylabel(r'Reconstruction error $y_\mathrm{reco} - y_\mathrm{truth}$')
    ax.set_ylim(-0.35, 0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='upper left')
    ax.set_title(r'(a) Reconstruction resolution and bias of $y$')
    ax.text(-0.9, -0.28, r'Clean $\pi^0$ shower' + '\n' + r'($\Delta y \sim \pm 0.05$)', color='blue', fontsize=10)
    ax.text(0.35, -0.28, r'Degraded soft $\pi^0$' + '\n' + r'($\Delta y \sim \pm 0.18$)', color='darkred', fontsize=10)

    # 3(b): Mixed modes: AUC vs rho |y|
    ax = axes[1]
    mm = summary['mixed_modes']
    mode_names = [r'$\pi \times \rho$', r'$\rho \times \pi$', r'$\rho \times 3\pi$', r'$3\pi \times \rho$']
    keys = ['pi_rho', 'rho_pi', 'rho_3pi', '3pi_rho']
    
    x_pos = np.arange(len(keys))
    auc_low = [mm[k]['low_y_auc_full22'] for k in keys]
    auc_high = [mm[k]['high_y_auc_full22'] for k in keys]
    auc_ov = [mm[k]['overall_auc_full22'] for k in keys]
    
    ax.bar(x_pos - 0.2, auc_low, width=0.35, color='#9467bd', alpha=0.7, label=r'Low $|y_\rho| < 0.4$ (Symmetric)')
    ax.bar(x_pos + 0.2, auc_high, width=0.35, color='#2ca02c', alpha=0.85, label=r'High $|y_\rho| \geq 0.7$ (Asymmetric)')
    ax.plot(x_pos, auc_ov, 'k_', markersize=20, mew=2.5, label='Inclusive mean')
    
    ax.set_xticks(x_pos, labels=mode_names)
    ax.set_ylabel('H/Z Separation AUC (Reco +IP/SV)')
    ax.set_ylim(0.60, 0.69)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(frameon=True, loc='upper right')
    ax.set_title(r'(b) Modulation of mixed modes by $\rho$-side $|y|$')

    plt.tight_layout()
    fig3_path = fig_dir / 'fig3_detector_asymmetry_and_mixed_modes.png'
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Kinematic Threshold Robustness (m_vis > m_tau)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 4(a): Distribution of m_vis_reco
    ax = axes[0]
    m0_rr = arrays['m_vis0_rr']
    m1_rr = arrays['m_vis1_rr']
    m_all = np.concatenate([m0_rr, m1_rr])
    
    counts, edges, _ = ax.hist(m_all, bins=80, range=(0.2, 2.5), density=True,
                               color='#1f77b4', alpha=0.6, edgecolor='blue', label=r'All $\rho$ sides in $\rho\times\rho$')
    ax.axvline(1.777, color='#d62728', lw=2.5, ls='--', label=r'Physical tau mass threshold $m_\tau = 1.777$ GeV')
    ax.axvspan(1.777, 2.5, color='#ff7f0e', alpha=0.25, label=r'Unphysical tail ($9.0\%$ of decays)')
    ax.set_xlabel(r'Reconstructed visible mass $m_\mathrm{vis}^\mathrm{reco}$ [GeV]')
    ax.set_ylabel('Probability density')
    ax.set_xlim(0.2, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='upper right')
    ax.set_title(r'(a) Reconstructed visible mass distribution')

    # 4(b): Score distribution comparison: physical vs unphysical
    ax = axes[1]
    sc_rr = arrays['score_full22_rr']
    lbl_rr = arrays['labels_rr']
    unphys = (m0_rr > 1.777) | (m1_rr > 1.777)
    
    sc_H_phys = sc_rr[(~unphys) & (lbl_rr == 1)]
    sc_Z_phys = sc_rr[(~unphys) & (lbl_rr == 0)]
    sc_H_unphys = sc_rr[unphys & (lbl_rr == 1)]
    sc_Z_unphys = sc_rr[unphys & (lbl_rr == 0)]
    
    ax.hist(sc_H_phys, bins=50, range=(-2.0, 2.0), density=True, histtype='step',
            color='#d62728', lw=2, label='Higgs (Physical, AUC=0.622)')
    ax.hist(sc_Z_phys, bins=50, range=(-2.0, 2.0), density=True, histtype='step',
            color='#1f77b4', lw=2, label='Z boson (Physical)')
    ax.hist(sc_H_unphys, bins=50, range=(-2.0, 2.0), density=True, histtype='step',
            color='#ff7f0e', lw=2, ls='--', label='Higgs (Unphysical, AUC=0.616)')
    ax.hist(sc_Z_unphys, bins=50, range=(-2.0, 2.0), density=True, histtype='step',
            color='#2ca02c', lw=2, ls='--', label='Z boson (Unphysical)')

    ax.set_xlabel(r'Predicted Bilinear Score $\Lambda(h_\mathrm{pred})$')
    ax.set_ylabel('Probability density')
    ax.set_xlim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc='upper left', fontsize=9.5)
    ax.set_title(r'(b) Classifier separation across physical threshold')

    plt.tight_layout()
    fig4_path = fig_dir / 'fig4_kinematic_threshold_robustness.png'
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved {fig4_path}")


if __name__ == '__main__':
    main()
