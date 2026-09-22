#!/usr/bin/env python3
"""Generate publication-quality figures for the frame-tilt audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("analysis/rho_ip_tilt/results"))
    p.add_argument("--figures-dir", type=Path, default=Path("analysis/rho_ip_tilt/figures"))
    args = p.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    data = np.load(args.results_dir / "audit_arrays.npz")
    with open(args.results_dir / "audit_summary.json") as f:
        summary = json.load(f)

    modes = data["modes"]
    upsilon = data["upsilon"]
    theta_tk_mrad = data["theta_tk_mrad"]
    dphi_track = data["dphi_track"]
    dphi_vis = data["dphi_vis"]
    dphi_corr = data["dphi_corr"]
    dphi_approx = data["dphi_approx"]

    mask_pi = (modes == 0) & np.isfinite(dphi_track)
    mask_rho = (modes == 1) & np.isfinite(dphi_track) & (data["theta_tk_mrad"] > 0)
    mask_3pi = (modes == 3) & np.isfinite(dphi_track)

    # -------------------------------------------------------------
    # Figure 1: Frame tilt distributions & azimuth resolution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: Opening angle theta_tk
    ax = axes[0]
    bins_th = np.linspace(0, 30, 61)
    ax.hist(np.clip(theta_tk_mrad[mask_pi], 0, 30), bins=bins_th, density=True, histtype="step",
            color="C0", lw=2.0, label=r"$\pi$ (1p0n): median 0.0 mrad")
    ax.hist(np.clip(theta_tk_mrad[mask_rho], 0, 30), bins=bins_th, density=True, histtype="step",
            color="C3", lw=2.0, label=r"$\rho$ (1p1n): median 6.1 mrad")
    ax.hist(np.clip(theta_tk_mrad[mask_3pi], 0, 30), bins=bins_th, density=True, histtype="step",
            color="C2", lw=2.0, label=r"$3\pi$ (3p0n lead trk): median 4.3 mrad")
    ax.axvline(1.97, color="0.3", ls="--", lw=1.2, label=r"IP transverse res. $\sim$2.0 mrad")
    ax.set_xlabel(r"Opening angle $\theta(\mathrm{track}, \hat{k}_{\mathrm{vis}})$ [mrad]")
    ax.set_ylabel("Probability density")
    ax.set_title(r"(a) Opening angle between track and visible $\tau$ axis")
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Azimuth error in rho
    ax = axes[1]
    bins_phi = np.linspace(-np.pi, np.pi, 73)
    ax.hist(dphi_vis[mask_rho], bins=bins_phi, density=True, histtype="step",
            color="0.5", ls="--", lw=1.8, label=r"Naive visible frame (09-23): mean $\cos=0.344$")
    ax.hist(dphi_track[mask_rho], bins=bins_phi, density=True, histtype="step",
            color="C3", lw=2.2, label=r"Track frame: mean $\cos=0.691$")
    ax.hist(dphi_corr[mask_rho], bins=bins_phi, density=True, histtype="step",
            color="C0", ls="-.", lw=2.0, label=r"Parallax-corrected: mean $\cos=0.714$")
    ax.axhline(1 / (2 * np.pi), color="k", ls=":", lw=1.0, label="Uniform (no information)")
    ax.set_xlabel(r"Azimuth error $\Delta\phi$ [rad]")
    ax.set_ylabel("Probability density")
    ax.set_title(r"(b) $\rho$ decay (1p1n): IP azimuth resolution")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Panel 3: Azimuth error in 3pi
    ax = axes[2]
    ax.hist(dphi_vis[mask_3pi], bins=bins_phi, density=True, histtype="step",
            color="0.5", ls="--", lw=1.8, label=r"Naive visible frame (09-23): mean $\cos=0.304$")
    ax.hist(dphi_track[mask_3pi], bins=bins_phi, density=True, histtype="step",
            color="C2", lw=2.2, label=r"Track frame: mean $\cos=0.638$")
    ax.hist(dphi_corr[mask_3pi], bins=bins_phi, density=True, histtype="step",
            color="C0", ls="-.", lw=2.0, label=r"Parallax-corrected: mean $\cos=0.615$")
    ax.axhline(1 / (2 * np.pi), color="k", ls=":", lw=1.0, label="Uniform (no information)")
    ax.set_xlabel(r"Azimuth error $\Delta\phi$ [rad]")
    ax.set_ylabel("Probability density")
    ax.set_title(r"(c) $3\pi$ decay (3p0n, lead track): IP azimuth")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig1_frame_tilt_distributions.png", dpi=160)
    plt.close(fig)
    print("Saved fig1_frame_tilt_distributions.png")

    # -------------------------------------------------------------
    # Figure 2: Parallax Mechanism & Upsilon Dependence
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: Diagram of Parallax Effect
    ax = axes[0]
    ax.set_aspect("equal")
    # Draw axes
    ax.axhline(0, color="0.7", lw=1.0)
    ax.axvline(0, color="0.7", lw=1.0)
    # Origin K (visible direction)
    ax.plot(0, 0, "ko", ms=8)
    ax.text(-0.05, -0.15, r"$K$ ($\hat{k}_{\mathrm{vis}}$)", fontsize=11, fontweight="bold")
    # Track T
    theta_demo = 0.6  # mrad demo scale
    t_x, t_y = 0.6, 0.4
    ax.plot(t_x, t_y, "rs", ms=8)
    ax.text(t_x + 0.05, t_y - 0.1, r"$T$ ($\hat{t}_{\pi^\pm}$)", fontsize=11, color="red", fontweight="bold")
    ax.annotate("", xy=(t_x, t_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.0))
    ax.text(0.2, 0.35, r"$\vec{t}_\perp$ ($\sim 6.1$ mrad)", fontsize=10, color="red", rotation=30)
    # True tau
    tau_x, tau_y = 0.8, 1.2
    ax.plot(tau_x, tau_y, "b*", ms=12)
    ax.text(tau_x + 0.05, tau_y + 0.05, r"$\mathcal{T}$ ($\hat{\tau}$)", fontsize=11, color="blue", fontweight="bold")
    # Vector T -> tau along u_ip
    ax.annotate("", xy=(tau_x, tau_y), xytext=(t_x, t_y),
                arrowprops=dict(arrowstyle="->", color="C0", lw=2.0))
    ax.text(t_x + 0.12, (t_y + tau_y)/2, r"$\vec{b} \parallel \hat{u}_{\mathrm{IP}}$", fontsize=10, color="C0")
    # True vector from K -> tau
    ax.annotate("", xy=(tau_x, tau_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, ls="--"))
    ax.text(0.25, 0.8, r"$\hat{\tau}_\perp(\hat{k})$", fontsize=10, color="blue")
    # False vector from K along u_ip (naive)
    u_len = np.sqrt((tau_x - t_x)**2 + (tau_y - t_y)**2)
    u_hat_x, u_hat_y = (tau_x - t_x)/u_len, (tau_y - t_y)/u_len
    ax.annotate("", xy=(u_hat_x * 0.8, u_hat_y * 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.8, ls=":"))
    ax.text(0.05, 0.65, r"$\hat{u}_{\mathrm{IP}}$ (naive)", fontsize=10, color="0.4")
    ax.set_xlim(-0.3, 1.2)
    ax.set_ylim(-0.3, 1.4)
    ax.set_xlabel(r"Transverse basis $\hat{n}$ [arb.]")
    ax.set_ylabel(r"Transverse basis $\hat{r}$ [arb.]")
    ax.set_title(r"(a) Geometric origin of parallax shift")

    # Panel 2: theta_tk vs upsilon
    ax = axes[1]
    u_data = summary["rho_by_upsilon"]
    u_centers = [0.5 * (x["upsilon_range"][0] + x["upsilon_range"][1]) for x in u_data]
    u_theta = [x["theta_tk_mrad_median"] for x in u_data]
    u_counts = [x["n"] for x in u_data]
    ax.plot(u_centers, u_theta, "o-", color="C3", lw=2.2, label=r"Median opening angle $\theta_{tk}$")
    ax.set_xlabel(r"Energy asymmetry $\Upsilon = (E_{\pi^\pm} - E_{\pi^0}) / E_{\mathrm{vis}}$")
    ax.set_ylabel(r"Median $\theta(\mathrm{track}, \hat{k}_{\mathrm{vis}})$ [mrad]")
    ax.set_title(r"(b) Opening angle vs energy asymmetry $\Upsilon$")
    ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.bar(u_centers, u_counts, width=0.15, alpha=0.2, color="0.3", label="Event count")
    ax2.set_ylabel(r"Number of $\rho$ sides")
    ax2.grid(False)

    # Panel 3: Azimuth quality vs upsilon
    ax = axes[2]
    cos_track = [x["track_frame"]["mean_cos"] for x in u_data]
    cos_vis = [x["naive_vis_frame"]["mean_cos"] for x in u_data]
    cos_corr = [x["parallax_corrected"]["mean_cos"] for x in u_data]
    ax.plot(u_centers, cos_track, "s-", color="C3", lw=2.0, label="Track frame (intrinsic)")
    ax.plot(u_centers, cos_vis, "o--", color="0.4", lw=2.0, label="Naive visible frame (09-23)")
    ax.plot(u_centers, cos_corr, "^-.", color="C0", lw=2.0, label="Parallax-corrected")
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel(r"Energy asymmetry $\Upsilon = (E_{\pi^\pm} - E_{\pi^0}) / E_{\mathrm{vis}}$")
    ax.set_ylabel(r"Azimuth correlation $\langle \cos\Delta\phi \rangle$")
    ax.set_title(r"(c) Azimuth quality vs energy asymmetry")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig2_upsilon_and_parallax_mechanism.png", dpi=160)
    plt.close(fig)
    print("Saved fig2_upsilon_and_parallax_mechanism.png")

    # -------------------------------------------------------------
    # Figure 3: Null space vs Learned Point-h Mode Hierarchy
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Data from P5 and P9
    pairs = [
        "1p0n x 1p0n",
        "1p0n x rho",
        "rho x rho",
        "1p0n x 3p0n",
        "rho x 3p0n",
        "3p0n x 3p0n",
    ]
    labels_pair = [
        r"$\pi \times \pi$",
        r"$\pi \times \rho$",
        r"$\rho \times \rho$",
        r"$\pi \times 3\pi$",
        r"$\rho \times 3\pi$",
        r"$3\pi \times 3\pi$",
    ]
    p5_deltas = [0.0180, 0.0278, 0.0158, 0.0327, 0.0169, 0.0043]
    p9_deltas = [0.0072, 0.0072, 0.0030, 0.0171, 0.0071, 0.0061]

    # Panel 1: Bar chart comparison of Delta AUC
    ax = axes[0]
    x = np.arange(len(pairs))
    width = 0.35
    rects1 = ax.bar(x - width/2, p5_deltas, width, label="P5 Null space (Track frame used)", color="navy", alpha=0.85)
    rects2 = ax.bar(x + width/2, p9_deltas, width, label="P9 Learned point-h (Naive vis frame)", color="darkorange", alpha=0.85)

    ax.set_ylabel(r"$\Delta\mathrm{AUC}$ from adding PV-referenced IP")
    ax.set_title(r"(a) H/Z separation gain by decay mode pair")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pair, rotation=25)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 0.038)

    # Highlight rho-containing pairs
    for i in [1, 2, 4]:
        ax.get_xticklabels()[i].set_color("red")
        ax.get_xticklabels()[i].set_fontweight("bold")

    # Panel 2: Ratio of Learned to Nullspace Gain
    ax = axes[1]
    ratios = [p9 / p5 * 100 for p9, p5 in zip(p9_deltas, p5_deltas)]
    colors = ["C0" if i not in [1, 2, 4] else "C3" for i in range(len(pairs))]
    bars = ax.bar(x, ratios, width=0.55, color=colors, alpha=0.85)
    ax.axhline(50, color="0.5", ls="--", lw=1.2, label=r"50% benchmark")
    ax.set_ylabel("Learned gain / Null-space potential [%]")
    ax.set_title(r"(b) Efficiency of learned model in capturing IP headroom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pair, rotation=25)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 150)

    for i in [1, 2, 4]:
        ax.get_xticklabels()[i].set_color("red")
        ax.get_xticklabels()[i].set_fontweight("bold")

    for bar, r in zip(bars, ratios):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig3_nullspace_vs_learned_mode_hierarchy.png", dpi=160)
    plt.close(fig)
    print("Saved fig3_nullspace_vs_learned_mode_hierarchy.png")


if __name__ == "__main__":
    main()
