"""Generate publication-grade figures for the correlated systematics calibration study.

Figures:
- fig1_frame_rotation.png: Frame azimuthal jitter and coherent bias response.
- fig2_met_induced_jitter.png: Kinematic MET resolution driving ditau frame jitter and R_T/Q.
- fig3_pv_resolution_modes.png: Shared Primary Vertex jitter across decay modes.
- fig4_tolerance_envelope.png: 2D tolerance boundary map (sigma_psi vs sigma_PV) with ATLAS operating box.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / 'results'
FIGURES_DIR = HERE / 'figures'
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
})


def make_fig1(data):
    """Figure 1: Frame azimuthal jitter and coherent bias."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Left: Jitter sigma_psi
    jitter = data['study1_frame_jitter']
    s_psi = [v['sigma_psi_rad'] * 1000 for v in jitter.values()]
    
    # Exact theory curve
    th_x = np.linspace(0, 205, 100)
    th_y = np.exp(2.0 * (th_x * 0.001)**2)
    
    # Models
    models = {
        'reco_full22': ('reco +IP/SV (s42)', '#1f77b4', 'o', '-'),
        'reco_base': ('reco base (no IP/SV)', '#ff7f0e', 's', '--'),
        'reco_idealip22': ('reco ideal IP', '#2ca02c', '^', '-.'),
        'exact_h': ('exact h (theory)', '#7f7f7f', 'D', ':'),
    }
    
    for m_key, (m_label, col, mkr, ls) in models.items():
        y_vals = [v['models'][m_key]['double_R'] for v in jitter.values()]
        err_low = [v['models'][m_key]['double_R'] - v['models'][m_key]['ci95'][0] if 'ci95' in v['models'][m_key] else 0 for v in jitter.values()]
        err_high = [v['models'][m_key]['ci95'][1] - v['models'][m_key]['double_R'] if 'ci95' in v['models'][m_key] else 0 for v in jitter.values()]
        if m_key == 'reco_full22':
            ax1.errorbar(s_psi, y_vals, yerr=[err_low, err_high], label=m_label, color=col, marker=mkr, linestyle=ls, capsize=3, zorder=5)
        else:
            ax1.plot(s_psi, y_vals, label=m_label, color=col, marker=mkr, linestyle=ls)
            
    ax1.plot(th_x, th_y, 'k-', linewidth=2.2, alpha=0.8, label=r'Analytic: $\exp(2\sigma_\psi^2)$')
    
    # Reference lines
    ax1.axhline(1.00, color='gray', linestyle=':', alpha=0.7)
    ax1.axhline(1.01, color='red', linestyle='--', alpha=0.5, label=r'$\pm 1\%$ threshold')
    ax1.axhline(1.03, color='crimson', linestyle=':', alpha=0.5, label=r'$\pm 3\%$ threshold')
    
    ax1.set_xlabel(r'Frame Azimuthal Jitter $\sigma_\psi$ [mrad]')
    ax1.set_ylabel(r'Transfer Double Ratio $R_{T/Q} = \rho_T / \rho_Q$')
    ax1.set_title(r'(a) Random Frame Jitter ($\rho_T = 1.000$, $\rho_Q = e^{-2\sigma_\psi^2}$)')
    ax1.set_xlim(-5, 205)
    ax1.set_ylim(0.99, 1.10)
    ax1.grid(True)
    ax1.legend(frameon=True, loc='upper left')

    # Right: Coherent bias psi0
    bias = data['study2_frame_bias']
    psi0 = [v['psi0_rad'] * 1000 for v in bias.values()]
    for m_key, (m_label, col, mkr, ls) in models.items():
        y_vals = [v['models'][m_key]['double_R'] for v in bias.values()]
        ax2.plot(psi0, y_vals, label=m_label, color=col, marker=mkr, linestyle=ls)
        
    ax2.axhline(1.00, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(1.01, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0.99, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(1.03, color='crimson', linestyle=':', alpha=0.5)
    ax2.axhline(0.97, color='crimson', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel(r'Coherent Frame Misalignment $\psi_0$ [mrad]')
    ax2.set_ylabel(r'Transfer Double Ratio $R_{T/Q}$')
    ax2.set_title(r'(b) Coherent Frame Misalignment')
    ax2.set_xlim(-105, 105)
    ax2.set_ylim(0.95, 1.05)
    ax2.grid(True)
    ax2.legend(frameon=True, loc='lower center')
    
    plt.tight_layout()
    p = FIGURES_DIR / 'fig1_frame_rotation.png'
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def make_fig2(data):
    """Figure 2: Kinematic MET Resolution Driving Frame Jitter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    met_data = data['study3_met_resolution']
    sig_met = [v['sigma_met_GeV'] for v in met_data.values()]
    rms_psi = [v['rms_psi_mrad'] for v in met_data.values()]
    
    # Left: MET vs induced frame jitter
    ax1.plot(sig_met, rms_psi, 'o-', color='#1f77b4', linewidth=2.2, label=r'Induced ditau jitter $\sigma_\psi$')
    # Theoretical estimate: sigma_MET / median(qT)
    med_qT = data['meta']['median_qT_GeV']
    theory_psi = [sm / med_qT * 1000 for sm in sig_met]
    ax1.plot(sig_met, theory_psi, 'k--', linewidth=1.8, label=rf'Kinematic scale: $\sigma_{{\rm MET}} / \langle q_T \rangle$ ($q_T = {med_qT:.0f}$ GeV)')
    
    ax1.axvspan(15, 25, color='orange', alpha=0.2, label=r'Typical ATLAS $\sigma_{\rm MET}$ (15-25 GeV)')
    ax1.axhline(70.7, color='red', linestyle=':', alpha=0.7, label=r'1% bias boundary ($\sigma_\psi = 70.7$ mrad)')
    
    ax1.set_xlabel(r'Missing $E_T$ Resolution $\sigma_{\rm MET}$ [GeV]')
    ax1.set_ylabel(r'RMS Ditau Frame Jitter $\sigma_\psi$ [mrad]')
    ax1.set_title(r'(a) Induced Rest-Frame Jitter from Missing Energy')
    ax1.set_xlim(0, 52)
    ax1.set_ylim(0, 230)
    ax1.grid(True)
    ax1.legend(frameon=True, loc='upper left')
    
    # Right: Double ratio vs MET resolution
    models = {
        'reco_full22': ('reco +IP/SV (s42)', '#1f77b4', 'o', '-'),
        'reco_base': ('reco base (no IP/SV)', '#ff7f0e', 's', '--'),
        'exact_h': ('exact h (theory)', '#7f7f7f', 'D', ':'),
    }
    
    for m_key, (m_label, col, mkr, ls) in models.items():
        y_vals = [v['models'][m_key]['double_R'] for v in met_data.values()]
        if m_key == 'reco_full22':
            err_low = [v['models'][m_key]['double_R'] - v['models'][m_key]['ci95'][0] for v in met_data.values()]
            err_high = [v['models'][m_key]['ci95'][1] - v['models'][m_key]['double_R'] for v in met_data.values()]
            ax2.errorbar(sig_met, y_vals, yerr=[err_low, err_high], label=m_label, color=col, marker=mkr, linestyle=ls, capsize=3, zorder=5)
        else:
            ax2.plot(sig_met, y_vals, label=m_label, color=col, marker=mkr, linestyle=ls)
            
    # Theory from rms_psi
    th_R = [v['models']['reco_full22']['theory_R'] for v in met_data.values()]
    ax2.plot(sig_met, th_R, 'k-', alpha=0.8, label=r'Analytic: $\exp(2\sigma_{\psi,{\rm eff}}^2)$')
    
    ax2.axvspan(15, 25, color='orange', alpha=0.2, label=r'Typical ATLAS $\sigma_{\rm MET}$ (15-25 GeV)')
    ax2.axhline(1.00, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(1.01, color='red', linestyle='--', alpha=0.5, label=r'$\pm 1\%$ threshold')
    ax2.axhline(1.03, color='crimson', linestyle=':', alpha=0.5, label=r'$\pm 3\%$ threshold')
    
    ax2.set_xlabel(r'Missing $E_T$ Resolution $\sigma_{\rm MET}$ [GeV]')
    ax2.set_ylabel(r'Transfer Double Ratio $R_{T/Q}$')
    ax2.set_title(r'(b) Calibration Transfer Departure vs $\sigma_{\rm MET}$')
    ax2.set_xlim(0, 52)
    ax2.set_ylim(0.99, 1.10)
    ax2.grid(True)
    ax2.legend(frameon=True, loc='upper left')
    
    plt.tight_layout()
    p = FIGURES_DIR / 'fig2_met_induced_jitter.png'
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def make_fig3(data):
    """Figure 3: Primary Vertex Jitter across decay mode pairs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    pv_data = data['study4_pv_resolution']
    sig_pv = [v['sigma_pv_um'] for v in pv_data.values()]
    
    # Left: Response components rho_T and rho_Q for pipi vs rhorho
    pipi_rT = [v['scopes']['pipi']['rho_T'] for v in pv_data.values()]
    pipi_rQ = [v['scopes']['pipi']['rho_Q'] for v in pv_data.values()]
    rhorho_rT = [v['scopes']['rhorho']['rho_T'] for v in pv_data.values()]
    rhorho_rQ = [v['scopes']['rhorho']['rho_Q'] for v in pv_data.values()]
    ov_rT = [v['scopes']['overall']['rho_T'] for v in pv_data.values()]
    ov_rQ = [v['scopes']['overall']['rho_Q'] for v in pv_data.values()]
    
    ax1.plot(sig_pv, pipi_rT, 'o-', color='#d62728', label=r'$\pi\times\pi$: $\rho_T$ ($m=0$)')
    ax1.plot(sig_pv, pipi_rQ, 's--', color='#d62728', alpha=0.7, label=r'$\pi\times\pi$: $\rho_Q$ ($m=2$)')
    ax1.plot(sig_pv, rhorho_rT, '^-', color='#2ca02c', label=r'$\rho\times\rho$: $\rho_T$')
    ax1.plot(sig_pv, rhorho_rQ, 'v--', color='#2ca02c', alpha=0.7, label=r'$\rho\times\rho$: $\rho_Q$')
    ax1.plot(sig_pv, ov_rT, 'D-', color='#1f77b4', label=r'Overall: $\rho_T$')
    ax1.plot(sig_pv, ov_rQ, 'd--', color='#1f77b4', alpha=0.7, label=r'Overall: $\rho_Q$')
    
    ax1.axvspan(15, 25, color='lightblue', alpha=0.3, label=r'ATLAS PV resolution (15-25 $\mu$m)')
    ax1.set_xlabel(r'Shared PV Resolution $\sigma_{\rm PV}$ [$\mu$m]')
    ax1.set_ylabel(r'Single Response Ratio $\rho = r(\sigma)/r(0)$')
    ax1.set_title(r'(a) Single-Response Suppression by Mode')
    ax1.set_xlim(0, 52)
    ax1.set_ylim(0.55, 1.02)
    ax1.grid(True)
    ax1.legend(frameon=True, loc='lower left')

    # Right: Double ratio R_T/Q across modes
    modes_plot = {
        'pipi': (r'$\pi\times\pi$ (pure IP)', '#d62728', 'o', '-'),
        'pirho': (r'$\pi\times\rho$', '#ff7f0e', 's', '-.'),
        'rhorho': (r'$\rho\times\rho$ (pion-dominated)', '#2ca02c', '^', '--'),
        '3pi3pi': (r'$3\pi\times 3\pi$', '#9467bd', 'v', ':'),
        'overall': ('Overall all events', '#1f77b4', 'D', '-'),
    }
    
    for m_key, (m_label, col, mkr, ls) in modes_plot.items():
        y_vals = [v['scopes'][m_key]['double_R'] for v in pv_data.values()]
        ax2.plot(sig_pv, y_vals, label=m_label, color=col, marker=mkr, linestyle=ls)
        
    ax2.axvspan(15, 25, color='lightblue', alpha=0.3, label=r'ATLAS PV resolution (15-25 $\mu$m)')
    ax2.axhline(1.00, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(0.99, color='red', linestyle='--', alpha=0.5, label=r'$\pm 1\%$ threshold')
    ax2.axhline(0.97, color='crimson', linestyle=':', alpha=0.5, label=r'$\pm 3\%$ threshold')
    
    ax2.set_xlabel(r'Shared PV Resolution $\sigma_{\rm PV}$ [$\mu$m]')
    ax2.set_ylabel(r'Transfer Double Ratio $R_{T/Q} = \rho_T / \rho_Q$')
    ax2.set_title(r'(b) Double Ratio Departure ($R_{T/Q} \leq 1$)')
    ax2.set_xlim(0, 52)
    ax2.set_ylim(0.965, 1.005)
    ax2.grid(True)
    ax2.legend(frameon=True, loc='lower left')
    
    plt.tight_layout()
    p = FIGURES_DIR / 'fig3_pv_resolution_modes.png'
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def make_fig4(data):
    """Figure 4: 2D Tolerance Boundary Phase Diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    
    t_data = data['study5_tolerance_grid']
    grid_psi = np.array(t_data['grid_psi_mrad'])
    grid_pv = np.array(t_data['grid_pv_um'])
    R_mat = np.array(t_data['double_R_matrix'])  # shape (n_psi, n_pv)
    
    # Meshgrid for contour
    X, Y = np.meshgrid(grid_pv, grid_psi)
    
    # Contour levels of departure in percent: (R - 1) * 100
    bias_pct = (R_mat - 1.0) * 100.0
    
    levels = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    cs = ax.contourf(X, Y, bias_pct, levels=np.linspace(-1.5, 5.5, 29), cmap='RdYlBu_r', alpha=0.85)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r'Calibration Transfer Bias $(R_{T/Q} - 1)$ [%]')
    
    # Specific contour lines
    lines = ax.contour(X, Y, bias_pct, levels=[0.0, 1.0, 2.0, 3.0, 5.0], colors=['white', 'black', 'black', 'black', 'black'], linewidths=[2.2, 1.8, 1.4, 1.4, 1.4])
    ax.clabel(lines, inline=True, fmt='%1.1f%%', fontsize=10)
    
    # Highlight ATLAS operating region:
    # sigma_MET = 15-25 GeV -> sigma_psi = 60-100 mrad
    # sigma_PV = 15-25 um
    from matplotlib.patches import Rectangle
    rect = Rectangle((15, 60), 10, 40, linewidth=2.5, edgecolor='magenta', facecolor='none', linestyle='--', label=r'ATLAS Operating Region' + '\n' + r'($\sigma_{\rm PV}=15\text{--}25\,\mu$m, $\sigma_\psi=60\text{--}100$ mrad)', zorder=10)
    ax.add_patch(rect)
    
    # Best-operating point marker
    ax.plot([20], [80], 'm*', markersize=14, markeredgewidth=1.5, markeredgecolor='black', label=r'Nominal ATLAS point ($\approx +1.2\%$ bias)', zorder=11)
    
    ax.set_xlabel(r'Shared PV Resolution $\sigma_{\rm PV}$ [$\mu$m]')
    ax.set_ylabel(r'Ditau Rest-Frame Jitter $\sigma_\psi$ [mrad]')
    ax.set_title(r'2D Tolerance Phase Diagram of $Z\to\tau\tau$ Calibration Transfer')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 150)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='upper left')
    
    plt.tight_layout()
    p = FIGURES_DIR / 'fig4_tolerance_envelope.png'
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def main():
    json_path = RESULTS_DIR / 'correlated_stress_study.json'
    if not json_path.exists():
        print(f"File {json_path} does not exist!")
        return
    data = json.loads(json_path.read_text())
    print(f"Loaded {json_path}")
    make_fig1(data)
    make_fig2(data)
    make_fig3(data)
    make_fig4(data)
    print("All 4 figures generated successfully!")


if __name__ == '__main__':
    main()
