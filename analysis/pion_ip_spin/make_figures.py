"""Plotting script for 1-prong pion IP kinematics and spin headroom."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling for HEP publication-quality plots
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RESULTS_DIR = Path("analysis/pion_ip_spin/results")
FIG_DIR = Path("analysis/pion_ip_spin/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1(data, summary):
    """Fig 1: Kinematic lever arm d0 and angular resolution vs x and d0."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=200)

    res_x = summary["single_pi_resolution_vs_x"]
    x_centers = [0.5 * (r["x_min"] + r["x_max"]) for r in res_x]
    x_widths = [0.5 * (r["x_max"] - r["x_min"]) for r in res_x]
    ip_med = [r["median_ip_um"] for r in res_x]
    theta_med = [r["median_theta_tau_pi_mrad"] for r in res_x]
    cos_mean = [r["mean_cos_dphi"] for r in res_x]
    dphi_med = [r["median_dphi_deg"] for r in res_x]
    dphi_p68 = [r["p68_dphi_deg"] for r in res_x]

    # Panel (a): Lever arm & opening angle vs x
    ax = axes[0]
    color1 = "#1f77b4"
    color2 = "#d62728"
    ax.errorbar(x_centers, ip_med, xerr=x_widths, fmt="o-", color=color1, label=r"Median $d_0$ [$\mu$m]", capsize=3)
    ax.set_xlabel(r"Visible Energy Fraction $x = E_\pi / E_\tau$")
    ax.set_ylabel(r"3D Impact Parameter $d_0$ [$\mu$m]", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(0, 160)

    ax2 = ax.twinx()
    ax2.errorbar(x_centers, theta_med, xerr=x_widths, fmt="s--", color=color2, label=r"Opening angle $\theta_{\tau\pi}$ [mrad]", capsize=3)
    ax2.set_ylabel(r"Opening Angle $\theta_{\tau\pi}$ [mrad]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 18)
    ax.set_title("(a) Lever Arm Collapse vs Visible $x$")

    # Panel (b): Angular resolution vs x
    ax = axes[1]
    ax.errorbar(x_centers, dphi_med, xerr=x_widths, fmt="o-", color="#2ca02c", label=r"Median $|\Delta\phi_{\rm IP}|$", capsize=3)
    ax.errorbar(x_centers, dphi_p68, xerr=x_widths, fmt="^--", color="#17becf", label=r"68% Quantile $|\Delta\phi_{\rm IP}|$", capsize=3)
    ax.set_xlabel(r"Visible Energy Fraction $x = E_\pi / E_\tau$")
    ax.set_ylabel(r"Azimuth Error $|\Delta\phi_{\rm IP}|$ [degrees]")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper left")
    ax.set_title(r"(b) Azimuth Resolution $|\Delta\phi_{\rm IP}|$ vs $x$")

    # Panel (c): cos(dphi) vs physical d0
    ax = axes[2]
    # Compute profile from arrays
    mask_pi = data["mask_pi_single"]
    d0_all = data["ip_norm_um"][mask_pi]
    cos_all = data["cos_dphi"][mask_pi]
    d0_bins = np.linspace(10, 200, 15)
    d0_mids = 0.5 * (d0_bins[:-1] + d0_bins[1:])
    cos_prof = []
    cos_err = []
    for j in range(len(d0_bins) - 1):
        bmask = (d0_all >= d0_bins[j]) & (d0_all < d0_bins[j + 1])
        if bmask.sum() > 20:
            cos_prof.append(np.mean(cos_all[bmask]))
            cos_err.append(np.std(cos_all[bmask]) / np.sqrt(bmask.sum()))
        else:
            cos_prof.append(np.nan)
            cos_err.append(np.nan)

    ax.errorbar(d0_mids, cos_prof, yerr=cos_err, fmt="o-", color="#9467bd", label=r"Measured $\langle\cos\Delta\phi\rangle$")
    ax.axhline(1.0, color="gray", linestyle=":", label="Perfect Alignment")
    ax.axvline(30.0, color="orange", linestyle="--", label=r"Resolution threshold ($\sim 30\,\mu$m)")
    ax.set_xlabel(r"3D Impact Parameter $d_0$ [$\mu$m]")
    ax.set_ylabel(r"Azimuth Alignment $\langle\cos\Delta\phi_{\rm IP}\rangle$")
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(r"(c) Direction Quality vs Physical Lever Arm $d_0$")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_lever_arm_and_resolution.png")
    plt.close()
    print("Saved fig1_lever_arm_and_resolution.png")


def plot_fig2(summary):
    """Fig 2: Neural network polarimeter vector correlation hierarchy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

    corr_x = summary["single_pi_polarimeter_corr_vs_x"]
    x_centers = [0.5 * (r["x_min"] + r["x_max"]) for r in corr_x]
    x_widths = [0.5 * (r["x_max"] - r["x_min"]) for r in corr_x]

    cp_base = [r["corr_perp_Base"] for r in corr_x]
    cp_full = [r["corr_perp_Full22"] for r in corr_x]
    cp_ideal = [r["corr_perp_IdealIP"] for r in corr_x]

    ck_base = [r["corr_k_Base"] for r in corr_x]
    ck_full = [r["corr_k_Full22"] for r in corr_x]
    ck_ideal = [r["corr_k_IdealIP"] for r in corr_x]

    # Panel (a): Transverse correlation (h_n, h_r)
    ax = axes[0]
    ax.errorbar(x_centers, cp_ideal, xerr=x_widths, fmt="s--", color="#2ca02c", label=r"Ideal IP Oracle ($\Delta\phi=0$)", lw=2.2)
    ax.errorbar(x_centers, cp_full, xerr=x_widths, fmt="o-", color="#d62728", label=r"Full22 (Reconstructed IP)", lw=2.2)
    ax.errorbar(x_centers, cp_base, xerr=x_widths, fmt="^-.", color="#1f77b4", label=r"Base (No IP/SV)", lw=1.8)
    ax.set_xlabel(r"Visible Energy Fraction $x = E_\pi / E_\tau$")
    ax.set_ylabel(r"Transverse Correlation $\mathrm{corr}(h_\perp^{\rm pred}, h_\perp^{\rm exact})$")
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower left")
    ax.set_title(r"(a) Transverse Polarimeter $h_\perp$ Response")

    # Panel (b): Delta(Transverse) gain and Longitudinal correlation
    ax = axes[1]
    delta_full = np.array(cp_full) - np.array(cp_base)
    headroom = np.array(cp_ideal) - np.array(cp_full)

    ax.plot(x_centers, delta_full, "o-", color="#d62728", label=r"Recovered Gain: $\mathrm{Full22} - \mathrm{Base}$", lw=2.2)
    ax.plot(x_centers, headroom, "s--", color="#ff7f0e", label=r"Unrecovered Headroom: $\mathrm{Ideal} - \mathrm{Full22}$", lw=2.2)
    ax.set_xlabel(r"Visible Energy Fraction $x = E_\pi / E_\tau$")
    ax.set_ylabel(r"Correlation Difference $\Delta\mathrm{corr}(h_\perp)$")
    ax.set_ylim(0.0, 0.45)
    ax.legend(loc="upper right")
    ax.set_title(r"(b) Recovered Gain vs Lost Headroom")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_polarimeter_correlation_hierarchy.png")
    plt.close()
    print("Saved fig2_polarimeter_correlation_hierarchy.png")


def plot_fig3(summary):
    """Fig 3: pi-pi phase space H/Z AUC across (x1, x2) and min IP."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=200)

    # Panel (a): 2D x1, x2 regions
    ax = axes[0]
    pipi_2d = summary["pipi_2d_x_phase_space"]
    regions = [r["region"] for r in pipi_2d]
    counts = [r["count"] for r in pipi_2d]
    auc_b = [r["auc_base"] for r in pipi_2d]
    auc_f = [r["auc_full"] for r in pipi_2d]
    auc_i = [r["auc_ideal"] for r in pipi_2d]
    auc_e = [r["auc_exact"] for r in pipi_2d]

    y_pos = np.arange(len(regions))
    bar_width = 0.20

    ax.barh(y_pos - 1.5 * bar_width, auc_b, bar_width, label="Base (No IP)", color="#1f77b4", alpha=0.85)
    ax.barh(y_pos - 0.5 * bar_width, auc_f, bar_width, label="Full22 (Reco IP)", color="#d62728", alpha=0.9)
    ax.barh(y_pos + 0.5 * bar_width, auc_i, bar_width, label="IdealIP", color="#2ca02c", alpha=0.85)
    ax.barh(y_pos + 1.5 * bar_width, auc_e, bar_width, label=r"Exact $h$", color="#7f7f7f", alpha=0.6)

    ax.set_yticks(y_pos)
    labels_with_n = [f"{r}\n(N={c})" for r, c in zip(regions, counts)]
    ax.set_yticklabels(labels_with_n, fontsize=9.5)
    ax.set_xlabel("H/Z Separation AUC")
    ax.set_xlim(0.50, 0.85)
    ax.axvline(0.50, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title(r"(a) $\pi\times\pi$ AUC across 2D $(x_1, x_2)$ Phase Space")

    # Panel (b): AUC vs min IP
    ax = axes[1]
    pipi_ip = summary["pipi_by_min_ip"]
    ip_mids = [0.5 * (r["ip_low"] + min(r["ip_high"], 150)) for r in pipi_ip]
    ip_labels = [f"[{r['ip_low']},{r['ip_high']})" for r in pipi_ip]
    counts_ip = [r["count"] for r in pipi_ip]

    auc_b_ip = [r["auc_base"] for r in pipi_ip]
    auc_f_ip = [r["auc_full"] for r in pipi_ip]
    auc_i_ip = [r["auc_ideal"] for r in pipi_ip]
    auc_e_ip = [r["auc_exact"] for r in pipi_ip]

    x_idx = np.arange(len(pipi_ip))
    ax.plot(x_idx, auc_e_ip, "s-", color="#7f7f7f", label=r"Exact $h$ Ceiling", lw=1.8)
    ax.plot(x_idx, auc_i_ip, "^--", color="#2ca02c", label=r"IdealIP Oracle", lw=2.0)
    ax.plot(x_idx, auc_f_ip, "o-", color="#d62728", label=r"Full22 (Reco IP)", lw=2.2)
    ax.plot(x_idx, auc_b_ip, "v-.", color="#1f77b4", label=r"Base (No IP)", lw=1.8)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{lbl}\n(N={c})" for lbl, c in zip(ip_labels, counts_ip)], fontsize=9.5)
    ax.set_xlabel(r"$\min(d_{0,1}, d_{0,2})$ [$\mu$m]")
    ax.set_ylabel("H/Z Separation AUC")
    ax.set_ylim(0.58, 0.77)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title(r"(b) $\pi\times\pi$ AUC vs Minimum Impact Parameter")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_pipi_phase_space_auc.png")
    plt.close()
    print("Saved fig3_pipi_phase_space_auc.png")


def plot_fig4(summary):
    """Fig 4: Mixed modes propagation and experimental selection recipes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)

    # Panel (a): Mixed modes AUC vs IP_pi cut
    ax = axes[0]
    mm = summary["mixed_modes"]
    cuts_rho = mm["pi_x_rho"]["cuts"]
    cuts_3pi = mm["pi_x_3pi"]["cuts"]

    cut_vals = [c["cut_um"] for c in cuts_rho]
    auc_f_rho = [c["auc_full"] for c in cuts_rho]
    auc_b_rho = [c["auc_base"] for c in cuts_rho]
    auc_f_3pi = [c["auc_full"] for c in cuts_3pi]
    auc_b_3pi = [c["auc_base"] for c in cuts_3pi]

    ax.plot(cut_vals, auc_f_rho, "o-", color="#d62728", label=r"$\pi\times\rho$ Full22 (Reco IP)", lw=2.2)
    ax.plot(cut_vals, auc_b_rho, "o--", color="#d62728", alpha=0.5, label=r"$\pi\times\rho$ Base (No IP)", lw=1.5)
    ax.plot(cut_vals, auc_f_3pi, "s-", color="#1f77b4", label=r"$\pi\times 3\pi$ Full22 (Reco IP)", lw=2.2)
    ax.plot(cut_vals, auc_b_3pi, "s--", color="#1f77b4", alpha=0.5, label=r"$\pi\times 3\pi$ Base (No IP)", lw=1.5)

    ax.set_xlabel(r"Pion-Side Impact Parameter Cut: $d_{0,\pi} \geq c$ [$\mu$m]")
    ax.set_ylabel("H/Z Separation AUC")
    ax.set_ylim(0.59, 0.66)
    ax.legend(loc="lower right")
    ax.set_title(r"(a) Mixed Mode AUC vs Pion $d_0$ Selection")

    # Panel (b): Efficiency vs AUC for pipi experimental cuts
    ax = axes[1]
    pipi_cuts = summary["pipi_experimental_cuts"]
    eff = [c["efficiency"] * 100 for c in pipi_cuts]
    auc_f_pi = [c["auc_full22"] for c in pipi_cuts]
    auc_b_pi = [c["auc_base"] for c in pipi_cuts]
    cut_labels = [f"$\\geq {c['min_ip_cut_um']}\\mu$m" for c in pipi_cuts]

    ax.plot(eff, auc_f_pi, "o-", color="#d62728", label="Full22 (Reco IP)", lw=2.2)
    ax.plot(eff, auc_b_pi, "s--", color="#1f77b4", label="Base (No IP)", lw=1.8)

    for i, txt in enumerate(cut_labels):
        ax.annotate(txt, (eff[i], auc_f_pi[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel(r"Event Selection Efficiency [%]")
    ax.set_ylabel(r"$\pi\times\pi$ H/Z Separation AUC")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.60, 0.68)
    ax.legend(loc="lower left")
    ax.set_title(r"(b) Purity-Efficiency Trade-off in $\pi\times\pi$")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_mixed_modes_and_experimental_selection.png")
    plt.close()
    print("Saved fig4_mixed_modes_and_experimental_selection.png")


def main():
    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)
    data = np.load(RESULTS_DIR / "analysis_arrays.npz")

    plot_fig1(data, summary)
    plot_fig2(summary)
    plot_fig3(summary)
    plot_fig4(summary)
    print("All figures successfully generated in", FIG_DIR)


if __name__ == "__main__":
    main()
