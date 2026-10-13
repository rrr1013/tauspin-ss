"""Generate publication-quality figures for the background spin shape,
instrumental correlation, mass drift, and transfer systematics study.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path.home() / 'Projects' / 'tauspin'
RESULTS_DIR = REPO / 'analysis' / 'bkg_spin_shape' / 'results'
FIG_DIR = REPO / 'analysis' / 'bkg_spin_shape' / 'figures'
VAULT_FIG_DIR = Path('/Users/ryunosuke/Library/Mobile Documents/iCloud~md~obsidian/Documents/AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-bkg-spin-shape-20261014/Figures')

FIG_DIR.mkdir(parents=True, exist_ok=True)
VAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
})


def load_data():
    with open(RESULTS_DIR / 'bkg_study_results.json') as f:
        return json.load(f)


def make_fig1(data):
    """Fig 1: Reconstructed spin statistics and transverse analyzing power across models and decay modes."""
    p1 = data['part1_instrumental']
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # (a) Reconstructed gT and gL for H, Z, and U across models
    models = ['exact', 'tauspin', 'tauspin_noIPSV', 'textbook']
    labels = ['Exact $h$', 'tauspin\n(+IP/SV)', 'Base\n(no IP/SV)', 'Textbook']
    x = np.arange(len(models))
    width = 0.25
    
    gT_H = [p1[m]['all']['gT']['H'] for m in models]
    gT_Z = [p1[m]['all']['gT']['Z'] for m in models]
    gT_U = [p1[m]['all']['gT']['U'] for m in models]
    
    rects1 = axes[0].bar(x - width, gT_H, width, label='Signal $H \\to \\tau\\tau$ (Scalar)', color='#1f77b4', alpha=0.85, edgecolor='black')
    rects2 = axes[0].bar(x, gT_Z, width, label='Genuine $Z \\to \\tau\\tau$ (Vector)', color='#2ca02c', alpha=0.85, edgecolor='black')
    rects3 = axes[0].bar(x + width, gT_U, width, label='Fake / Unpolarized', color='#d62728', alpha=0.85, edgecolor='black')
    
    axes[0].set_ylabel(r'Mean Reconstructed Statistic $\langle g_T \rangle$')
    axes[0].set_title(r'(a) Reconstructed Transverse Spin Statistic $\langle g_T \rangle$')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].legend(loc='upper right', framealpha=0.9)
    
    # (b) Transverse Signal Separation Delta g_T = gT(H) - gT(Z) by mode pair
    modes = ['pi x pi', 'pi x rho', 'rho x rho', '3pi x 3pi', 'all']
    mode_labels = [r'$\pi \times \pi$', r'$\pi \times \rho$', r'$\rho \times \rho$', r'$3\pi \times 3\pi$', 'Overall']
    xm = np.arange(len(modes))
    wm = 0.25
    
    dgT_tauspin = [p1['tauspin'][m]['gT']['H'] - p1['tauspin'][m]['gT']['Z'] for m in modes]
    dgT_noipsv = [p1['tauspin_noIPSV'][m]['gT']['H'] - p1['tauspin_noIPSV'][m]['gT']['Z'] for m in modes]
    dgT_textbook = [p1['textbook'][m]['gT']['H'] - p1['textbook'][m]['gT']['Z'] for m in modes]
    
    axes[1].bar(xm - wm, dgT_tauspin, wm, label=r'tauspin (+IP/SV)', color='#1f77b4', alpha=0.85, edgecolor='black')
    axes[1].bar(xm, dgT_noipsv, wm, label=r'Base (no IP/SV)', color='#ff7f0e', alpha=0.85, edgecolor='black')
    axes[1].bar(xm + wm, dgT_textbook, wm, label=r'Textbook', color='#2ca02c', alpha=0.85, edgecolor='black')
    
    axes[1].set_ylabel(r'Transverse Separation $\Delta \langle g_T \rangle = \langle g_T^H \rangle - \langle g_T^Z \rangle$')
    axes[1].set_title(r'(b) Transverse Analyzing Power by Decay Mode Pair')
    axes[1].set_xticks(xm)
    axes[1].set_xticklabels(mode_labels)
    axes[1].legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig1_instrumental_correlation.png'
    plt.savefig(fig_path, dpi=200)
    plt.savefig(VAULT_FIG_DIR / 'fig1_instrumental_correlation.png', dpi=200)
    plt.close()
    print("Saved Fig 1 to", fig_path)


def make_fig2(data):
    """Fig 2: Kinematic mass drift across m_tautau bins and sideband divergence."""
    p2 = data['part2_kinematic_drift']['tauspin']
    bins_data = p2['bins']
    
    m_mids = [b['m_mid'] for b in bins_data]
    gT_H = [b['gT']['H'] for b in bins_data]
    gT_Z = [b['gT']['Z'] for b in bins_data]
    gT_U = [b['gT']['U'] for b in bins_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # (a) gT drift across m_tautau
    axes[0].plot(m_mids, gT_H, 'o-', color='#1f77b4', label=r'Signal $H \to \tau\tau$ ($w_H$)', markersize=6)
    axes[0].plot(m_mids, gT_Z, 's--', color='#2ca02c', label=r'Genuine $Z \to \tau\tau$ ($w_Z$)', markersize=6)
    axes[0].plot(m_mids, gT_U, '^-.', color='#d62728', label=r'Fake / Unpolarized ($w_U$)', markersize=6)
    
    # Shading for Sideband vs Signal Region
    axes[0].axvspan(70, 110, color='lightgray', alpha=0.35, label='Low Sideband (SB)')
    axes[0].axvspan(110, 140, color='lightyellow', alpha=0.55, label='Signal Region (SR)')
    axes[0].axvspan(140, 200, color='lightgray', alpha=0.35, label='High Sideband (SB)')
    
    # Linear fit lines
    slope_Z = p2['slope_gT_Z']
    axes[0].set_xlabel(r'Reconstructed Invariant Mass $m_{\tau\tau}$ [GeV]')
    axes[0].set_ylabel(r'Mean Statistic $\langle g_T \rangle$')
    axes[0].set_title(r'(a) Kinematic Drift Across Invariant Mass $m_{\tau\tau}$')
    axes[0].legend(loc='lower left', framealpha=0.9)
    
    # (b) Difference between SR and SB templates across 36 spin bins
    # Load binned histograms for SR and SB
    sh_SR = np.mean([b['hist_Z'] for b in bins_data if 110 <= b['m_mid'] <= 140], axis=0)
    sh_SB = np.mean([b['hist_Z'] for b in bins_data if b['m_mid'] < 110 or b['m_mid'] > 140], axis=0)
    diff = sh_SR - sh_SB
    
    bin_idx = np.arange(len(diff))
    axes[1].bar(bin_idx, diff, color=np.where(diff >= 0, '#1f77b4', '#d62728'), alpha=0.85, edgecolor='black', width=0.8)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xlabel(r'Spin Quantile Bin Index $b \in [0, 35]$ (6$\times$6 $(g_T, g_L)$ Grid)')
    axes[1].set_ylabel(r'Template Distortion $\Delta sh_b = sh_b^{\rm SR} - sh_b^{\rm SB}$')
    axes[1].set_title(r'(b) Sideband-to-Signal Template Residuals ($\rm TVD = 7.7\%$)')
    
    tvd_str = f"Total Variation Distance: {p2['SR_vs_SB']['tvd_Z']*100:.1f}%\n" \
              f"Linear Slope: {slope_Z*1e4:+.2f} $\\times 10^{{-4}}$ /GeV"
    axes[1].text(0.04, 0.92, tvd_str, transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig2_kinematic_mass_drift.png'
    plt.savefig(fig_path, dpi=200)
    plt.savefig(VAULT_FIG_DIR / 'fig2_kinematic_mass_drift.png', dpi=200)
    plt.close()
    print("Saved Fig 2 to", fig_path)


def make_fig3(data):
    """Fig 3: 1D Transfer Systematics Scans for Tau-Embedding and Fake-Factor."""
    p3 = data['part3_transfer_systematics']
    scan_emb = p3['scan_emb']
    scan_fake = p3['scan_fake']
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # (a) Embedding scan
    dg = [x['delta_g'] * 100 for x in scan_emb]
    Z_ent_g = [x['Z_ent'] for x in scan_emb]
    Z_false_g = [x['Z_false'] for x in scan_emb]
    
    ax0_2 = axes[0].twinx()
    l1 = axes[0].plot(dg, Z_ent_g, 'o-', color='#1f77b4', label=r'True Entanglement $Z_{\rm ent}$', markersize=6)
    l2 = ax0_2.plot(dg, Z_false_g, 's--', color='#d62728', label=r'False Significance $Z_{\rm false}$ (Null)', markersize=6)
    
    axes[0].axhline(3.021, color='blue', linestyle=':', label=r'Nominal ($\tau=1.0, \delta=0$, 3.02$\sigma$)')
    axes[0].axhline(1.845, color='gray', linestyle='--', label=r'Sidebands-only Baseline (1.85$\sigma$)')
    axes[0].axhline(2.500, color='darkorange', linestyle='-.', label=r'Acceptance Boundary (2.50$\sigma$)')
    
    axes[0].set_xlabel(r'Tau-Embedding Transfer Distortion $\delta_{\rm emb}$ [%]')
    axes[0].set_ylabel(r'Entanglement Significance $Z_{\rm ent}$ [$\sigma$]', color='#1f77b4')
    ax0_2.set_ylabel(r'False Positive Significance $Z_{\rm false}$ [$\sigma$]', color='#d62728')
    axes[0].tick_params(axis='y', labelcolor='#1f77b4')
    ax0_2.tick_params(axis='y', labelcolor='#d62728')
    axes[0].set_title(r'(a) Tau-Embedding Transfer Systematic Scan')
    
    lines = l1 + l2 + [axes[0].lines[0], axes[0].lines[1], axes[0].lines[2]]
    labels = [l.get_label() for l in lines]
    axes[0].legend(lines, labels, loc='lower left', framealpha=0.9, fontsize=9)
    
    # (b) Fake scan
    df = [x['delta_f'] * 100 for x in scan_fake]
    Z_ent_f = [x['Z_ent'] for x in scan_fake]
    Z_false_f = [x['Z_false'] for x in scan_fake]
    
    ax1_2 = axes[1].twinx()
    l1_f = axes[1].plot(df, Z_ent_f, 'o-', color='#1f77b4', label=r'True Entanglement $Z_{\rm ent}$', markersize=6)
    l2_f = ax1_2.plot(df, Z_false_f, 's--', color='#d62728', label=r'False Significance $Z_{\rm false}$ (Null)', markersize=6)
    
    axes[1].axhline(3.021, color='blue', linestyle=':')
    axes[1].axhline(1.845, color='gray', linestyle='--')
    axes[1].axhline(2.500, color='darkorange', linestyle='-.')
    
    axes[1].set_xlabel(r'Same-Sign Fake-Factor Transfer Distortion $\delta_{\rm fake}$ [%]')
    axes[1].set_ylabel(r'Entanglement Significance $Z_{\rm ent}$ [$\sigma$]', color='#1f77b4')
    ax1_2.set_ylabel(r'False Positive Significance $Z_{\rm false}$ [$\sigma$]', color='#d62728')
    axes[1].tick_params(axis='y', labelcolor='#1f77b4')
    ax1_2.tick_params(axis='y', labelcolor='#d62728')
    axes[1].set_title(r'(b) Same-Sign Fake-Factor Transfer Systematic Scan')
    
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig3_transfer_systematics_scans.png'
    plt.savefig(fig_path, dpi=200)
    plt.savefig(VAULT_FIG_DIR / 'fig3_transfer_systematics_scans.png', dpi=200)
    plt.close()
    print("Saved Fig 3 to", fig_path)


def make_fig4(data):
    """Fig 4: 2D Tolerance Phase Diagram in (delta_emb, delta_fake)."""
    p3 = data['part3_transfer_systematics']
    grid = p3['grid_2d']
    deltas = np.array(p3['grid_deltas']) * 100 # [%]
    
    DG, DF = np.meshgrid(deltas, deltas, indexing='ij')
    Z_grid = np.array([[col['Z_ent'] for col in row] for row in grid])
    Zf_grid = np.array([[col['Z_false'] for col in row] for row in grid])
    
    fig, ax = plt.subplots(figsize=(8.5, 7))
    
    # Contour fill of Z_ent
    levels = np.linspace(1.8, 3.4, 17)
    cp = ax.contourf(DG, DF, Z_grid, levels=levels, cmap='viridis', alpha=0.88)
    cbar = plt.colorbar(cp, ax=ax)
    cbar.set_label(r'Entanglement Significance $Z_{\rm ent}$ [$\sigma$]')
    
    # Specific contour lines
    cs = ax.contour(DG, DF, Z_grid, levels=[1.845, 2.5, 3.0], colors=['white', 'yellow', 'cyan'], linewidths=[2.2, 2.0, 2.2])
    ax.clabel(cs, inline=True, fmt={1.845: '1.85$\\sigma$ (SB baseline)', 2.5: '2.5$\\sigma$', 3.0: '3.0$\\sigma$'}, fontsize=10)
    
    # False positive exclusion zone (Z_false >= 0.3 sigma)
    cf_false = ax.contourf(DG, DF, Zf_grid, levels=[0.3, 10.0], colors=['none', 'none'], hatches=['///'], alpha=0.0)
    cs_false = ax.contour(DG, DF, Zf_grid, levels=[0.3], colors=['red'], linestyles=['--'], linewidths=[2.0])
    ax.clabel(cs_false, inline=True, fmt={0.3: 'False Signal Risk $Z_{\\rm false} > 0.3\\sigma$'}, fontsize=9)
    
    # Experimental Tolerance Box (Safe Region)
    rect = plt.Rectangle((-2.0, -3.0), 5.0, 6.0, fill=False, edgecolor='magenta', linewidth=2.5, linestyle='-', label='ATLAS HL-LHC Tolerance Box\n' + r'($-2\% \leq \delta_{\rm emb} \leq +3\%$, $|\delta_{\rm fake}| \leq 3\%$)')
    ax.add_patch(rect)
    
    # Nominal point
    ax.plot(0, 0, marker='*', markersize=14, color='white', markeredgecolor='black', label=r'Nominal Ideal Point ($\tau=1.0, \delta=0$, 3.02$\sigma$)')
    
    ax.set_xlabel(r'Tau-Embedding Transfer Distortion $\delta_{\rm emb}$ [%]')
    ax.set_ylabel(r'Same-Sign Fake-Factor Transfer Distortion $\delta_{\rm fake}$ [%]')
    ax.set_title(r'2D Tolerance Phase Diagram: Auxiliary Systematic Breakdown Limits')
    ax.legend(loc='lower left', framealpha=0.92, fontsize=10)
    
    plt.tight_layout()
    fig_path = FIG_DIR / 'fig4_tolerance_phase_diagram.png'
    plt.savefig(fig_path, dpi=200)
    plt.savefig(VAULT_FIG_DIR / 'fig4_tolerance_phase_diagram.png', dpi=200)
    plt.close()
    print("Saved Fig 4 to", fig_path)


def main():
    print("Loading data...")
    data = load_data()
    print("Generating figures...")
    make_fig1(data)
    make_fig2(data)
    make_fig3(data)
    make_fig4(data)
    print("All figures successfully generated and copied to Vault Figures!")


if __name__ == '__main__':
    main()
