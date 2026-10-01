"""Make the predeclared distribution-first figure set for the azimuth audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("none", "lab", "local")
COLORS = {"none": "#555555", "lab": "#d95f02", "local": "#1b9e77"}
LABELS = {"none": "no geometry", "lab": "Cartesian geometry", "local": "local engineered geometry"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 180,
        "axes.grid": True, "grid.alpha": 0.2,
    })


def save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = {arm: np.load(getattr(args, arm)) for arm in ARMS}
    event = np.load(args.results / "event_metrics.npz")
    summary = json.loads((args.results / "summary.json").read_text())
    style()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    bins = np.geomspace(1e-5, max(float(np.max(event[f"{a}_d16"])) for a in ARMS) * 1.05, 70)
    for arm in ARMS:
        values = event[f"{arm}_d16"]
        axes[0].hist(values, bins=bins, density=True, histtype="step", lw=1.8,
                     color=COLORS[arm], label=LABELS[arm])
        ordered = np.sort(values)
        axes[1].plot(ordered, np.linspace(0, 1, len(ordered), endpoint=False),
                     color=COLORS[arm], lw=1.8, label=LABELS[arm])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"C16 orbit defect $D_e$"); axes[0].set_ylabel("Normalised density")
    axes[0].set_title("(a) Full defect distribution", loc="left")
    axes[1].set_xscale("log"); axes[1].set_xlabel(r"C16 orbit defect $D_e$")
    axes[1].set_ylabel("Event CDF"); axes[1].set_title("(b) Tail coverage", loc="left")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Fixed point-h pipelines, ATLAS validation (N={summary['rows']:,}, unit weight)")
    save(fig, args.output_dir / "fig1_orbit_defect.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for arm in ARMS:
        h16 = raw[arm]["h_c16"].astype(float)
        h17 = raw[arm]["h_c17"].astype(float)
        h0 = h16[:, 0]
        per16 = np.sqrt(np.mean(np.sum((h16 - h0[:, None]) ** 2, axis=(2, 3)), axis=0))
        per17 = np.sqrt(np.mean(np.sum((h17 - h0[:, None]) ** 2, axis=(2, 3)), axis=0))
        axes[0].plot(raw[arm]["angles_c16"], per16, color=COLORS[arm], marker="o", ms=3,
                     lw=1.3, label=LABELS[arm])
        axes[0].scatter(raw[arm]["angles_c17"], per17, color=COLORS[arm], marker="x", s=16, alpha=.65)
        axes[1].plot(raw[arm]["angles_c16"], summary["arms"][arm]["auc_c16"],
                     color=COLORS[arm], marker="o", ms=3, lw=1.3, label=LABELS[arm])
        axes[1].scatter(raw[arm]["angles_c17"], summary["arms"][arm]["auc_c17"],
                        color=COLORS[arm], marker="x", s=16, alpha=.65)
    axes[0].set_xlabel(r"Global rotation $\alpha$ [rad]"); axes[0].set_ylabel("RMS prediction displacement")
    axes[0].set_title("(a) Mean orbit response\nlines: C16; crosses: off-grid C17", loc="left")
    axes[1].set_xlabel(r"Global rotation $\alpha$ [rad]"); axes[1].set_ylabel("Fixed-readout H/Z AUC")
    axes[1].set_title("(b) Population response", loc="left")
    axes[0].legend(frameon=False)
    save(fig, args.output_dir / "fig2_angle_response.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=True)
    for axis, arm in zip(axes, ARMS):
        density = axis.hexbin(event[f"{arm}_score0"], event[f"{arm}_score_rms"], gridsize=55,
                              bins="log", mincnt=1, cmap="viridis")
        axis.set_xlabel("Unrotated H score")
        axis.set_title(LABELS[arm])
    axes[0].set_ylabel("Event score RMS across C16")
    fig.colorbar(density, ax=axes, label="log10 event count", fraction=.025, pad=.02)
    fig.suptitle("Event-level score migration; fixed arm-specific readout")
    save(fig, args.output_dir / "fig3_score_migration.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for axis, arm in zip(axes, ARMS):
        x, y = event[f"{arm}_error0"], event[f"{arm}_error_avg"]
        high = float(np.quantile(np.r_[x, y], .995))
        density = axis.hexbin(x, y, gridsize=60, bins="log", mincnt=1, cmap="magma", extent=(0, high, 0, high))
        axis.plot([0, high], [0, high], color="white", lw=1, ls="--")
        axis.set_xlim(0, high); axis.set_ylim(0, high)
        axis.set_xlabel("Unrotated event h MSE"); axis.set_title(LABELS[arm])
    axes[0].set_ylabel("C16-average event h MSE")
    fig.colorbar(density, ax=axes, label="log10 event count", fraction=.025, pad=.02)
    fig.suptitle("C16 symmetrisation is a new fixed test-time estimator")
    save(fig, args.output_dir / "fig4_symmetrized_error.png")

    lab_sat = raw["lab"]["lab_saturated_c16"].astype(bool)
    unsaturated = ~lab_sat.any(1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    values = event["lab_d16"]
    bins = np.geomspace(max(float(values[values > 0].min()), 1e-6), float(values.max()) * 1.05, 65)
    axes[0].hist(values, bins=bins, density=True, histtype="step", lw=1.7, color=COLORS["lab"], label="all")
    axes[0].hist(values[unsaturated], bins=bins, density=True, histtype="step", lw=1.7,
                 color="#7570b3", ls="--", label="unsaturated at all C16 angles")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Cartesian-pipeline defect $D_e$"); axes[0].set_ylabel("Normalised density")
    axes[0].set_title(f"(a) Clipping control\nunsaturated N={unsaturated.sum():,}", loc="left")
    axes[0].legend(frameon=False)
    axes[1].plot(raw["lab"]["angles_c16"], lab_sat.mean(0), color=COLORS["lab"], marker="o")
    axes[1].set_xlabel(r"Global rotation $\alpha$ [rad]"); axes[1].set_ylabel("Events with ≥1 clipped geometry component")
    axes[1].set_title("(b) Component-wise clipping occupancy", loc="left")
    save(fig, args.output_dir / "fig5_lab_clipping.png")

    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    for i, arm in enumerate(ARMS):
        for state, marker, ls in (("unrotated", "o", "-"), ("c16_average", "s", "--")):
            moment = summary["arms"][arm][f"z_rhorho_moment_{state}"]
            axis.plot([0, moment["real"]], [0, moment["imag"]], color=COLORS[arm], ls=ls, lw=1.4)
            axis.scatter(moment["real"], moment["imag"], color=COLORS[arm], marker=marker, s=35,
                         label=f"{LABELS[arm]}, {state.replace('_', ' ')}")
    axis.axhline(0, color="black", lw=.7); axis.axvline(0, color="black", lw=.7)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel(r"Re $\langle e^{i(\phi_- - \phi_+)}\rangle$")
    axis.set_ylabel(r"Im $\langle e^{i(\phi_- - \phi_+)}\rangle$")
    axis.set_title(r"Z→ρρ static first harmonic")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, args.output_dir / "fig6_z_rhorho_moment.png")


if __name__ == "__main__":
    main()
