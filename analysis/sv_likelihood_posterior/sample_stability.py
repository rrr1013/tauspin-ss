"""Check finite-draw stability of the validation posterior h estimates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import jsonable


def metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    residual = prediction[mask] - target[mask]
    return {
        "events": int(mask.sum()),
        "mse": float(np.mean(residual * residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draw-counts", type=int, nargs="+", default=(32, 64, 128, 256))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    with np.load(args.posterior, allow_pickle=False) as posterior:
        samples = np.asarray(posterior["h_samples"], dtype=np.float64)
        target = np.asarray(posterior["h_truth_reco"], dtype=np.float64)
    weights = np.load(args.representations / "weights_sv.npy", mmap_mode="r")
    with np.load(args.representations / "cohorts.npz", allow_pickle=False) as cohorts:
        masks = {
            name.removeprefix("mask_"): np.asarray(cohorts[name], dtype=bool)
            for name in cohorts.files if name.startswith("mask_")
        }
    selected_cohorts = ("inclusive", "any_sv", "threeprong_x_threeprong")
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for count in args.draw_counts:
        if count > samples.shape[1]:
            raise ValueError(f"Requested {count} draws but only {samples.shape[1]} are available")
        uniform_mean = np.mean(samples[:, :count], axis=1)
        prefix_weight = np.asarray(weights[:, :count], dtype=np.float64)
        prefix_weight /= np.sum(prefix_weight, axis=1, keepdims=True)
        weighted_mean = np.sum(samples[:, :count] * prefix_weight[:, :, None, None], axis=1)
        report[str(count)] = {
            cohort: {
                "uniform": metrics(uniform_mean, target, masks[cohort]),
                "sv": metrics(weighted_mean, target, masks[cohort]),
            }
            for cohort in selected_cohorts
        }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    draw_counts = np.asarray(args.draw_counts)
    for axis, cohort in zip(axes, selected_cohorts):
        for method, label, color in (
            ("uniform", "uniform", "0.4"),
            ("sv", "SV weighted", "#4C78A8"),
        ):
            values = [report[str(count)][cohort][method]["mse"] for count in draw_counts]
            axis.plot(draw_counts, values, marker="o", lw=2, label=label, color=color)
        axis.set(xlabel="posterior draws", ylabel="h MSE", title=cohort.replace("_", " "))
        axis.set_xscale("log", base=2)
        axis.set_xticks(draw_counts, [str(value) for value in draw_counts])
        axis.legend()
    fig.suptitle("Finite-draw stability (fixed prefixes; validation)")
    fig.savefig(args.output / "posterior_sample_stability.png", dpi=180)
    plt.close(fig)

    output = {
        "contract": {
            "selection": "fixed prefixes of the saved validation posterior draws",
            "test_loaded": False,
            "purpose": "diagnostic only; no likelihood or model selection",
        },
        "metrics": report,
    }
    (args.output / "report.json").write_text(
        json.dumps(jsonable(output), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
