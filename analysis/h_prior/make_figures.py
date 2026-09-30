"""Make the Stage-1 figures from the locked moment decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
COLORS = {
    "exact_h": "#222222",
    "base": "#4C78A8",
    "ip_sv": "#F58518",
    "geometry_shuffle": "#54A24B",
    "ideal_ip": "#B279A2",
}
LABELS = {
    "exact_h": "exact h",
    "base": "point h (base)",
    "ip_sv": "point h (+IP/SV)",
    "geometry_shuffle": "geometry shuffle",
    "ideal_ip": "ideal-IP oracle",
}


def cierr(entry: dict, key: str) -> tuple[float, tuple[float, float]]:
    val = entry[key.split("_")[0]]["amplitude"] if key.endswith("_amplitude") else entry[key]
    lo, hi = entry["bootstrap"][key]["ci95"]
    return val, (val - lo, hi - val)


def main() -> None:
    result = json.loads((RESULTS / "moment_decomposition.json").read_text())
    with np.load(RESULTS / "primary_arrays.npz") as d:
        arrays = {k: np.asarray(d[k]) for k in d.files}
    FIGURES.mkdir(parents=True, exist_ok=True)

    order = ["exact_h", "base", "ip_sv", "geometry_shuffle", "ideal_ip"]
    z = result["populations"]["Z_rhorho"]

    # Figure 1: observed harmonic split into factorisable and connected amplitudes.
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.23
    for shift, comp, hatch in ((-width, "obs", ""), (0, "fact", "//"), (width, "conn", "..")):
        vals, lows, highs = [], [], []
        for name in order:
            val = z[name][comp]["amplitude"]
            lo, hi = z[name]["bootstrap"][f"{comp}_amplitude"]["ci95"]
            vals.append(val); lows.append(val - lo); highs.append(hi - val)
        ax.bar(x + shift, vals, width, label={"obs": "observed", "fact": "factorisable", "conn": "connected"}[comp],
               color=[COLORS[n] for n in order], alpha={"obs": .95, "fact": .55, "conn": .28}[comp],
               edgecolor="black", linewidth=.6, hatch=hatch,
               yerr=np.stack([lows, highs]), capsize=2)
    ax.set_xticks(x, [LABELS[n] for n in order], rotation=15, ha="right")
    ax.set_ylabel(r"first-harmonic amplitude $A=2|m|$")
    ax.set_title(r"Z $\to\rho\rho$: static $\Delta\phi_h$ modulation is decomposed before interpretation", loc="left")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(FIGURES / "fig1_primary_decomposition.png", dpi=180)
    plt.close(fig)

    # Figure 2: complex moment plane. Arrows add m_fact + m_conn = m_obs.
    fig, axes = plt.subplots(1, len(order), figsize=(15, 3.4), sharex=True, sharey=True, constrained_layout=True)
    scale = max(abs(complex(z[n]["obs"]["real"], z[n]["obs"]["imag"])) for n in order) * 1.25
    for ax, name in zip(axes, order):
        mf = complex(z[name]["fact"]["real"], z[name]["fact"]["imag"])
        mc = complex(z[name]["conn"]["real"], z[name]["conn"]["imag"])
        mo = complex(z[name]["obs"]["real"], z[name]["obs"]["imag"])
        ax.arrow(0, 0, mf.real, mf.imag, color="#59A14F", width=scale/130, length_includes_head=True, label="factorisable")
        ax.arrow(mf.real, mf.imag, mc.real, mc.imag, color="#E45756", width=scale/130, length_includes_head=True, label="connected")
        ax.plot([mo.real], [mo.imag], "o", color="#222222", ms=4, label="observed")
        ax.axhline(0, color="0.8", lw=.8); ax.axvline(0, color="0.8", lw=.8)
        ax.set_aspect("equal"); ax.set_title(LABELS[name], fontsize=9)
        ax.set_xlim(-scale, scale); ax.set_ylim(-scale, scale)
    axes[0].set_ylabel(r"Im $m$")
    for ax in axes: ax.set_xlabel(r"Re $m$")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle(r"Complex first moment: vector addition retains phase information (Z $\rho\rho$)", fontsize=11)
    fig.savefig(FIGURES / "fig2_complex_moments.png", dpi=180)
    plt.close(fig)

    # Figure 3: actual Delta-phi density vs independent side pairing closure.
    fig, axes = plt.subplots(1, len(order), figsize=(15, 3.2), sharex=True, sharey=True, constrained_layout=True)
    rng = np.random.default_rng(20261001)
    bins = np.linspace(-np.pi, np.pi, 25)
    for ax, name in zip(axes, order):
        h = arrays[name]
        phi = np.arctan2(h[..., 1], h[..., 0])
        delta = np.angle(np.exp(1j * (phi[:, 0] - phi[:, 1])))
        perm = rng.permutation(len(h))
        delta_ind = np.angle(np.exp(1j * (phi[:, 0] - phi[perm, 1])))
        ax.hist(delta, bins=bins, density=True, histtype="step", lw=1.6, color=COLORS[name], label="same event")
        ax.hist(delta_ind, bins=bins, density=True, histtype="step", lw=1.2, ls="--", color="black", label="independent sides")
        ax.set_title(LABELS[name], fontsize=9); ax.grid(alpha=.2)
    axes[0].set_ylabel("density")
    for ax in axes: ax.set_xlabel(r"$\Delta\phi_h$ [rad]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Same-event structure versus a marginal-preserving independent-pair closure", fontsize=11)
    fig.savefig(FIGURES / "fig3_same_vs_independent.png", dpi=180)
    plt.close(fig)

    # Figure 4: connected fraction by mode pair for the base network, H and Z.
    pairs = list(result["mode_pairs"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    for ax, process in zip(axes, ("H", "Z")):
        vals, lo, hi = [], [], []
        for pair in pairs:
            e = result["mode_pairs"][pair][process]["base"]
            vals.append(e["aligned_conn_fraction"])
            c = e["bootstrap"]["aligned_conn_fraction"]["ci95"]
            lo.append(e["aligned_conn_fraction"] - c[0]); hi.append(c[1] - e["aligned_conn_fraction"])
        ax.errorbar(np.arange(len(pairs)), vals, yerr=np.stack([lo, hi]), fmt="o-", color="#4C78A8", capsize=2)
        ax.axhline(0, color="0.4", lw=.8); ax.axhline(1, color="0.4", lw=.8, ls="--")
        ax.set_xticks(np.arange(len(pairs)), pairs, rotation=30, ha="right")
        ax.set_title(f"{process} sample")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("connected contribution aligned with observed moment")
    fig.suptitle("The decomposition across decay-mode pairs is supportive, not the primary endpoint", fontsize=11)
    fig.savefig(FIGURES / "fig4_mode_pair_connected.png", dpi=180)
    plt.close(fig)

    # Figure 5: train/validation stability for exported arms.
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    xpos = np.arange(len(result["train_validation"]))
    for j, split in enumerate(("train_Z_rhorho", "validation_Z_rhorho")):
        vals = [result["train_validation"][n][split]["obs"]["amplitude"] for n in result["train_validation"]]
        errs = []
        for n in result["train_validation"]:
            e = result["train_validation"][n][split]
            l, h = e["bootstrap"]["obs_amplitude"]["ci95"]
            errs.append((e["obs"]["amplitude"] - l, h - e["obs"]["amplitude"]))
        ax.errorbar(xpos + (j-.5)*.14, vals, yerr=np.array(errs).T, fmt="o", capsize=3,
                    label=split.split("_")[0])
    ax.set_xticks(xpos, [LABELS[n] for n in result["train_validation"]])
    ax.set_ylabel(r"observed amplitude $A$")
    ax.set_title(r"Z $\rho\rho$ static modulation: train/validation stability", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.savefig(FIGURES / "fig5_train_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
