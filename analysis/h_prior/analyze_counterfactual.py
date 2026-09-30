"""Analyse the fixed-network counterfactual outputs and make Stage-2 figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUTS = {"base": HERE / "remote_outputs/base_full.npz", "ip_sv": HERE / "remote_outputs/ipsv_full.npz"}
LABELS = {"base": "point h (base)", "ip_sv": "point h (+IP/SV)"}
COLORS = {"base": "#4C78A8", "ip_sv": "#F58518"}
SEED = 20261001
BOOTSTRAPS = 2000


def q(h: np.ndarray) -> np.ndarray:
    return h[..., 0] + 1j * h[..., 1]


def moments(h: np.ndarray) -> dict[str, complex]:
    phi = np.arctan2(h[..., 1], h[..., 0])
    u = np.exp(1j * phi[:, 0])
    v = np.exp(-1j * phi[:, 1])
    obs = np.mean(u * v)
    fact = np.mean(u) * np.mean(v)
    return {"obs": obs, "fact": fact, "conn": obs - fact}


def moment_json(m: dict[str, complex]) -> dict:
    return {k: {"real": float(v.real), "imag": float(v.imag), "amplitude": float(2 * abs(v)),
                "phase_deg": float(np.degrees(np.angle(v)))} for k, v in m.items()}


def h_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    dot = np.sum(pred * truth, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(truth, axis=-1)
    cos = dot / np.maximum(den, 1e-12)
    return {"mse": float(np.mean((pred - truth) ** 2)), "mean_cosine": float(np.mean(cos))}


def response(original: np.ndarray, opposite: np.ndarray, target: np.ndarray, seed: int) -> dict:
    do = np.abs(q(opposite) - q(original))
    dt = np.abs(q(target) - q(original))

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    ro, rt = rms(do), rms(dt)
    rng = np.random.default_rng(seed)
    ratios = np.empty(BOOTSTRAPS)
    delta_moment = np.empty(BOOTSTRAPS)
    n = len(original)
    for b in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        ratios[b] = rms(do[idx]) / max(rms(dt[idx]), 1e-12)
        delta_moment[b] = 2 * (abs(moments(opposite[idx])["conn"]) - abs(moments(original[idx])["conn"]))
    return {
        "opposite_rms": ro,
        "target_positive_rms": rt,
        "opposite_over_target": ro / max(rt, 1e-12),
        "opposite_over_target_ci95": [float(x) for x in np.percentile(ratios, (2.5, 97.5))],
        "opposite_response_quantiles": {str(p): float(np.percentile(do, p)) for p in (50, 90, 95, 99, 99.9)},
        "target_response_quantiles": {str(p): float(np.percentile(dt, p)) for p in (50, 90, 95, 99, 99.9)},
        "connected_amplitude_change": float(2 * (abs(moments(opposite)["conn"]) - abs(moments(original)["conn"]))),
        "connected_amplitude_change_ci95": [float(x) for x in np.percentile(delta_moment, (2.5, 97.5))],
    }


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def main() -> None:
    data = {name: load(path) for name, path in INPUTS.items()}
    for key in ("global_indices", "labels", "truth_modes", "h_exact"):
        if not np.array_equal(data["base"][key], data["ip_sv"][key]):
            raise RuntimeError(f"cross-arm identity mismatch: {key}")

    result = {"schema": 1, "bootstraps": BOOTSTRAPS, "seed": SEED, "arms": {}}
    for arm_i, (name, d) in enumerate(data.items()):
        original, truth = d["h_original"].astype(float), d["h_exact"].astype(float)
        arm = {
            "n": int(len(original)),
            "original_h_metrics": h_metrics(original, truth),
            "original_moments": moment_json(moments(original)),
            "donor_maps": [],
        }
        for j in range(3):
            opposite = d[f"h_opposite_seed{j}"].astype(float)
            target = d[f"h_target_seed{j}"].astype(float)
            if np.array_equal(d[f"donor_minus_seed{j}"], d["global_indices"]) or np.array_equal(d[f"donor_plus_seed{j}"], d["global_indices"]):
                raise RuntimeError("self donor map")
            row = {
                "map_index": j,
                "opposite_h_metrics": h_metrics(opposite, truth),
                "opposite_moments": moment_json(moments(opposite)),
                "target_positive_moments": moment_json(moments(target)),
                "response": response(original, opposite, target, SEED + 100 * arm_i + j),
            }
            arm["donor_maps"].append(row)
        ratios = [x["response"]["opposite_over_target"] for x in arm["donor_maps"]]
        moment_changes = [x["response"]["connected_amplitude_change"] for x in arm["donor_maps"]]
        arm["donor_map_stability"] = {
            "ratio_min_max": [float(min(ratios)), float(max(ratios))],
            "connected_amplitude_change_min_max": [float(min(moment_changes)), float(max(moment_changes))],
        }
        result["arms"][name] = arm

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    (results_dir / "counterfactual.json").write_text(json.dumps(result, indent=2) + "\n")

    # Figure 6: primary response ratio for all donor maps.
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for i, name in enumerate(data):
        rows = result["arms"][name]["donor_maps"]
        for j, row in enumerate(rows):
            y = row["response"]["opposite_over_target"]
            lo, hi = row["response"]["opposite_over_target_ci95"]
            ax.errorbar(i + (j - 1) * .10, y, yerr=[[y-lo], [hi-y]], fmt="o", capsize=3,
                        color=COLORS[name], alpha=.75 if j != 1 else 1,
                        label=LABELS[name] if j == 1 else None)
    ax.set_xticks(range(len(data)), [LABELS[n] for n in data])
    ax.set_ylabel("opposite-side response / target-side positive response")
    ax.set_title("Fixed-network dependence on the opposite tau's decay constituents", loc="left")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    fig.savefig(figures_dir / "fig6_counterfactual_response.png", dpi=180)
    plt.close(fig)

    # Figure 7: connected amplitude before and after the opposite-side swap.
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    width = .18
    for i, name in enumerate(data):
        original_a = result["arms"][name]["original_moments"]["conn"]["amplitude"]
        ax.bar(i - width, original_a, width, color=COLORS[name], edgecolor="black", label="original" if i == 0 else None)
        swapped = [r["opposite_moments"]["conn"]["amplitude"] for r in result["arms"][name]["donor_maps"]]
        ax.bar(i, np.mean(swapped), width, color=COLORS[name], alpha=.55, hatch="//", edgecolor="black", label="opposite constituents swapped" if i == 0 else None)
        positive = [r["target_positive_moments"]["conn"]["amplitude"] for r in result["arms"][name]["donor_maps"]]
        ax.bar(i + width, np.mean(positive), width, color=COLORS[name], alpha=.28, hatch="..", edgecolor="black", label="target-side positive" if i == 0 else None)
        ax.scatter(np.full(3, i), swapped, color="black", s=15, zorder=3)
    ax.set_xticks(range(len(data)), [LABELS[n] for n in data])
    ax.set_ylabel(r"connected first-harmonic amplitude $2|m_{conn}|$")
    ax.set_title(r"Z $\rho\rho$: what the specified constituent intervention removes", loc="left")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.25)
    fig.savefig(figures_dir / "fig7_counterfactual_moment.png", dpi=180)
    plt.close(fig)

    # Figure 8: event-level response distributions and tails.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bins = np.linspace(-4, 0.5, 55)
    for name, d in data.items():
        original = d["h_original"].astype(float)
        opposite = d["h_opposite_seed1"].astype(float)
        target = d["h_target_seed1"].astype(float)
        do = np.abs(q(opposite) - q(original)).reshape(-1)
        dt = np.abs(q(target) - q(original)).reshape(-1)
        axes[0].hist(np.log10(np.maximum(do, 1e-5)), bins=bins, density=True, histtype="step", lw=1.7, color=COLORS[name], label=LABELS[name])
        axes[1].plot(np.sort(do), 1 - (np.arange(len(do)) + 1) / len(do), color=COLORS[name], lw=1.6, label=f"{LABELS[name]}: opposite")
        axes[1].plot(np.sort(dt), 1 - (np.arange(len(dt)) + 1) / len(dt), color=COLORS[name], lw=1.2, ls="--", label=f"{LABELS[name]}: target +")
    axes[0].set_xlabel(r"$\log_{10}|\Delta(h_n+i h_r)|$"); axes[0].set_ylabel("density")
    axes[0].set_title("opposite-side intervention")
    axes[1].set_xlabel(r"$|\Delta(h_n+i h_r)|$"); axes[1].set_ylabel("survival fraction"); axes[1].set_yscale("log")
    axes[1].set_title("event-level response tail")
    for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
    fig.savefig(figures_dir / "fig8_counterfactual_response_distribution.png", dpi=180)
    plt.close(fig)

    # Figure 9: target reconstruction-quality changes after opposite-side swap.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    for i, name in enumerate(data):
        original = result["arms"][name]["original_h_metrics"]
        swaps = [r["opposite_h_metrics"] for r in result["arms"][name]["donor_maps"]]
        dmse = [x["mse"] - original["mse"] for x in swaps]
        dcos = [x["mean_cosine"] - original["mean_cosine"] for x in swaps]
        offsets = i + np.asarray((-.08, 0, .08))
        axes[0].scatter(offsets, dmse, color=COLORS[name], label=LABELS[name])
        axes[1].scatter(offsets, dcos, color=COLORS[name])
    for ax, ylabel in zip(axes, (r"$\Delta$ h MSE", r"$\Delta$ mean cosine to exact h")):
        ax.axhline(0, color="black", lw=.9)
        ax.set_xticks(range(len(data)), [LABELS[n] for n in data])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False); fig.suptitle("Reconstruction quality is nearly unchanged by the opposite-side intervention")
    fig.savefig(figures_dir / "fig9_counterfactual_quality.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        name: {
            "response_ratios": [r["response"]["opposite_over_target"] for r in result["arms"][name]["donor_maps"]],
            "connected_before": result["arms"][name]["original_moments"]["conn"]["amplitude"],
            "connected_after": [r["opposite_moments"]["conn"]["amplitude"] for r in result["arms"][name]["donor_maps"]],
            "mse_before": result["arms"][name]["original_h_metrics"]["mse"],
            "mse_after": [r["opposite_h_metrics"]["mse"] for r in result["arms"][name]["donor_maps"]],
        } for name in data
    }, indent=2))


if __name__ == "__main__":
    main()
