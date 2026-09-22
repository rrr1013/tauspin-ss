"""Apply frozen SV likelihoods to saved baseline posterior samples.

This script is split-agnostic.  Validation uses the historical 256 paired
draws.  Training uses the inference-only 64-draw regeneration produced by
`resample_train.py`.  It writes compact weighted representations for fixed
readouts and, on validation, the requested raw-distribution figures and h
reconstruction metrics before any H/Z summary is produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import (
    MODE_NAMES,
    candidate_tau_direction,
    gaussian_log_weight,
    h_event_values,
    h_metric_block,
    jsonable,
    matched_direction_shuffle,
    normalized_weights,
    opening_angle,
    sv_log_weight,
    weighted_representations,
)


ARMS = (
    "uniform",
    "sv",
    "sv_raw",
    "global_gaussian",
    "matched_shuffle",
    "oracle_direction",
)
MAIN_TEMPERATURE_KEY = "sv"
GLOBAL_GAUSSIAN_SIGMA_RAD = 0.00790


def load_npz(path: Path, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        selected = data.files if names is None else names
        missing = [name for name in selected if name not in data.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        return {name: np.asarray(data[name]) for name in selected}


def open_output(path: Path, shape: tuple[int, ...], dtype: Any = np.float32) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def response_from_report(report: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], float]:
    response = {
        mode: report["response_models"]["sv"][name]
        for mode, name in MODE_NAMES.items()
    }
    temperature = float(report["temperature_calibration"][MAIN_TEMPERATURE_KEY]["temperature"])
    return response, temperature


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"events": 0}
    return {
        "events": int(len(finite)),
        "q00_q16_q50_q84_q95_q99_q100": np.quantile(
            finite, [0.0, 0.16, 0.5, 0.84, 0.95, 0.99, 1.0]
        ).tolist(),
        "mean": float(np.mean(finite)),
    }


def bootstrap_mean_difference(
    updated: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    difference = (np.asarray(updated) - np.asarray(baseline))[mask]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen = rng.integers(0, len(difference), len(difference))
        samples[index] = float(np.mean(difference[chosen]))
    return {
        "events": int(len(difference)),
        "difference": float(np.mean(difference)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "draws": draws,
        "seed": seed,
    }


def hist_add(target: np.ndarray, values: np.ndarray, bins: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite):
        target += np.histogram(finite, bins=bins)[0]


def density_from_counts(counts: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.sqrt(bins[:-1] * bins[1:]) if np.all(bins > 0) else 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    total = counts.sum()
    density = counts / np.maximum(total * widths, 1.0e-300)
    return centres, density


def draw_geometry_figure(
    output: Path,
    full_bins: np.ndarray,
    full_counts: dict[int, np.ndarray],
    core_bins: np.ndarray,
    core_counts: dict[int, np.ndarray],
    compatible_counts: np.ndarray,
    competing_counts: np.ndarray,
    truth_sv_counts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {0: "#4C78A8", 1: "#F28E2B", 3: "#59A14F"}
    for mode, name in MODE_NAMES.items():
        x, y = density_from_counts(full_counts[mode], full_bins)
        axes[0, 0].plot(x * 1.0e3, y / 1.0e3, lw=2, label=name, color=colors[mode])
    axes[0, 0].set_xscale("log"); axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="candidate tau - observed SV angle [mrad]",
                   ylabel="density [mrad$^{-1}$]", title="All posterior candidates: full range")
    axes[0, 0].legend()
    for mode, name in MODE_NAMES.items():
        x, y = density_from_counts(core_counts[mode], core_bins)
        axes[0, 1].plot(x * 1.0e3, y / 1.0e3, lw=2, label=name, color=colors[mode])
    axes[0, 1].set_xlim(0, 50)
    axes[0, 1].set(xlabel="candidate tau - observed SV angle [mrad]",
                   ylabel="density [mrad$^{-1}$]", title="All posterior candidates: core zoom")
    axes[0, 1].legend()
    for counts, label, color, style in (
        (compatible_counts, "per-event truth-nearest draw", "#4C78A8", "-"),
        (competing_counts, "other posterior draws", "#E45756", "--"),
    ):
        x, y = density_from_counts(counts, full_bins)
        axes[1, 0].plot(x * 1.0e3, y / 1.0e3, lw=2, ls=style, label=label, color=color)
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="candidate tau - observed SV angle [mrad]",
                   ylabel="density [mrad$^{-1}$]",
                   title="Truth-compatible versus competing draws")
    axes[1, 0].legend()
    x, y = density_from_counts(truth_sv_counts, full_bins)
    axes[1, 1].plot(x * 1.0e3, y / 1.0e3, color="#59A14F", lw=2)
    axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="truth tau - observed SV angle [mrad]",
                   ylabel="density [mrad$^{-1}$]",
                   title="Observed SV response on this validation cohort")
    fig.suptitle("SV geometry before summary metrics", fontsize=14)
    fig.savefig(output / "raw_sv_geometry.png", dpi=180)
    plt.close(fig)


def draw_weight_figure(
    output: Path,
    diagnostics: dict[str, dict[str, np.ndarray]],
    sv_weights: np.ndarray,
    strict_any_sv: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    palette = {
        "sv": "#4C78A8", "global_gaussian": "#E45756",
        "matched_shuffle": "#F28E2B", "sv_raw": "#59A14F",
    }
    for arm in ("sv", "sv_raw", "global_gaussian", "matched_shuffle"):
        values = diagnostics[arm]["ess"][strict_any_sv]
        axes[0, 0].hist(values, bins=np.linspace(0, sv_weights.shape[1], 65), density=True,
                        histtype="step", lw=2, label=arm, color=palette[arm])
        values = diagnostics[arm]["max_weight"][strict_any_sv]
        axes[0, 1].hist(values, bins=np.geomspace(1.0 / sv_weights.shape[1], 1.0, 65),
                        density=True, histtype="step", lw=2, label=arm, color=palette[arm])
        values = diagnostics[arm]["normalized_entropy"][strict_any_sv]
        axes[1, 0].hist(values, bins=np.linspace(0, 1, 65), density=True,
                        histtype="step", lw=2, label=arm, color=palette[arm])
    axes[0, 0].set(xlabel="effective sample size", ylabel="density", title="Posterior ESS")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel="maximum normalized weight", ylabel="density", title="Largest draw weight")
    axes[1, 0].set(xlabel="entropy / log(N draws)", ylabel="density", title="Normalized posterior entropy")
    for axis in axes.flat[:3]:
        axis.legend(fontsize=8)
    selected = sv_weights[strict_any_sv]
    rng = np.random.default_rng(20260922)
    flat = selected.reshape(-1)
    if len(flat) > 2_000_000:
        flat = flat[rng.choice(len(flat), 2_000_000, replace=False)]
    scaled = flat * sv_weights.shape[1]
    axes[1, 1].hist(scaled, bins=np.geomspace(1.0e-5, max(10.0, float(np.max(scaled))), 90),
                    density=True, histtype="step", lw=2, color="#4C78A8", label="SV weighted")
    axes[1, 1].axvline(1.0, color="0.25", ls="--", label="uniform weight")
    axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
    axes[1, 1].set(xlabel="normalized weight / uniform weight", ylabel="density",
                   title="Per-draw weight distribution")
    axes[1, 1].legend()
    fig.suptitle("SV posterior weight quality (strict events with observed SV)", fontsize=14)
    fig.savefig(output / "posterior_weight_quality.png", dpi=180)
    plt.close(fig)


def draw_h_figure(
    output: Path,
    baseline: np.ndarray,
    updated: np.ndarray,
    truth: np.ndarray,
    strict: np.ndarray,
    threepi: np.ndarray,
) -> None:
    error_base, cosine_base = h_event_values(baseline, truth)
    error_sv, cosine_sv = h_event_values(updated, truth)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    bins = np.geomspace(1.0e-4, 3.0, 90)
    for values, label, color in (
        (error_base[strict], "unweighted", "0.35"),
        (error_sv[strict], "SV weighted", "#4C78A8"),
        (error_base[strict & threepi], "unweighted 3p x 3p", "#E45756"),
        (error_sv[strict & threepi], "SV weighted 3p x 3p", "#59A14F"),
    ):
        axes[0, 0].hist(values, bins=bins, density=True, histtype="step", lw=2,
                        label=label, color=color)
    axes[0, 0].set_xscale("log"); axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="event mean squared h error", ylabel="density",
                   title="Event-level h error")
    axes[0, 0].legend(fontsize=8)
    for values, label, color in (
        (cosine_base[strict], "unweighted", "0.35"),
        (cosine_sv[strict], "SV weighted", "#4C78A8"),
        (cosine_base[strict & threepi], "unweighted 3p x 3p", "#E45756"),
        (cosine_sv[strict & threepi], "SV weighted 3p x 3p", "#59A14F"),
    ):
        axes[0, 1].hist(values, bins=np.linspace(-1, 1, 90), density=True,
                        histtype="step", lw=2, label=label, color=color)
    axes[0, 1].set(xlabel="mean side-wise h cosine", ylabel="density",
                   title="Event-level cosine")
    axes[0, 1].legend(fontsize=8)
    image = axes[1, 0].hexbin(
        error_base[strict], error_sv[strict], gridsize=65, bins="log",
        mincnt=1, xscale="log", yscale="log", cmap="viridis",
    )
    lo = min(float(np.min(error_base[strict])), float(np.min(error_sv[strict])))
    hi = max(float(np.max(error_base[strict])), float(np.max(error_sv[strict])))
    axes[1, 0].plot([lo, hi], [lo, hi], color="white", ls="--", lw=1.5)
    axes[1, 0].set(xlabel="unweighted event h error", ylabel="SV-weighted event h error",
                   title="Event-by-event comparison")
    fig.colorbar(image, ax=axes[1, 0], label="log event count")
    delta = error_sv - error_base
    axes[1, 1].hist(delta[strict], bins=np.linspace(-0.5, 0.5, 101), density=True,
                    histtype="step", lw=2, color="#4C78A8", label="inclusive")
    axes[1, 1].hist(delta[strict & threepi], bins=np.linspace(-0.5, 0.5, 101), density=True,
                    histtype="step", lw=2, color="#59A14F", label="3p x 3p")
    axes[1, 1].axvline(0.0, color="0.3", ls="--")
    axes[1, 1].set(xlabel="event error(SV) - error(unweighted)", ylabel="density",
                   title="Signed event-level change")
    axes[1, 1].legend()
    fig.suptitle("h reconstruction before H/Z readout", fontsize=14)
    fig.savefig(output / "h_event_distributions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--response-report", type=Path, required=True)
    parser.add_argument("--surface-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    input_names = (
        "global_indices", "labels", "weights", "modes", "reco_h_mode",
        "reco_visible_tau_lab4", "h",
    )
    source = load_npz(args.inputs, input_names)
    posterior_names = (
        "global_indices", "labels", "h_truth_reco", "truth_h_reco_valid",
        "all_draws_valid",
    )
    posterior = load_npz(args.posterior, posterior_names + (() if args.split == "train" else (
        "h_samples", "nu_samples_lab4",
    )))
    if not np.array_equal(source["global_indices"], posterior["global_indices"]):
        raise RuntimeError("Input/posterior global index mismatch")
    if not np.array_equal(source["labels"], posterior["labels"]):
        raise RuntimeError("Input/posterior label mismatch")
    if args.split == "train":
        if args.generated is None:
            raise ValueError("--generated is required for train")
        h_samples = np.load(args.generated / "train_h_samples.npy", mmap_mode="r")
        nu_xyz = np.load(args.generated / "train_nu_samples_xyz.npy", mmap_mode="r")
        generated_valid = np.load(args.generated / "train_draw_valid.npy", mmap_mode="r")
    else:
        h_samples = posterior["h_samples"]
        nu_xyz = posterior["nu_samples_lab4"][..., :3]
        generated_valid = np.isfinite(h_samples).all(axis=(-1, -2))
    if h_samples.shape != nu_xyz.shape:
        raise RuntimeError(f"h/nu paired draw shape mismatch: {h_samples.shape} vs {nu_xyz.shape}")
    events, draws = h_samples.shape[:2]
    if events != len(source["global_indices"]):
        raise RuntimeError("Draw/event count mismatch")

    geometry = load_npz(
        args.geometry,
        (
            "global_indices", "sv_available", "sv_direction", "tau_pt", "tau_eta",
            "tau_phi", "truth_tau_direction",
        ),
    )
    if not np.array_equal(geometry["global_indices"], np.arange(len(geometry["global_indices"]))):
        raise RuntimeError("Geometry global indices are not canonical row indices")
    ids = source["global_indices"].astype(np.int64)
    available = geometry["sv_available"][ids] > 0
    measured = geometry["sv_direction"][ids].astype(np.float64)
    truth_direction = geometry["truth_tau_direction"][ids].astype(np.float64)
    reco_modes = source["reco_h_mode"].astype(np.int8)
    shuffled, shuffle_source = matched_direction_shuffle(
        measured,
        available,
        reco_modes,
        geometry["tau_pt"][ids],
        geometry["tau_eta"][ids],
        geometry["tau_phi"][ids],
        np.random.default_rng(args.seed + (101 if args.split == "train" else 1001)),
    )
    response_report = json.loads(args.response_report.read_text())
    response, temperature = response_from_report(response_report)

    mean_outputs = {
        arm: open_output(args.output / f"mean_{arm}.npy", (events, 2, 3))
        for arm in ARMS
    }
    moments_sv = open_output(args.output / "moments_sv.npy", (events, 15))
    weight_sv = open_output(args.output / "weights_sv.npy", (events, draws))
    diagnostics = {
        arm: {
            name: open_output(args.output / f"{name}_{arm}.npy", (events,))
            for name in (
                "ess", "max_weight", "entropy", "normalized_entropy",
                "effective_populated", "above_uniform_count", "log_weight_span",
            )
        }
        for arm in ARMS if arm != "uniform"
    }

    full_bins = np.geomspace(1.0e-5, np.pi, 120)
    core_bins = np.linspace(0.0, 0.05, 101)
    full_counts = {mode: np.zeros(len(full_bins) - 1, dtype=np.int64) for mode in MODE_NAMES}
    core_counts = {mode: np.zeros(len(core_bins) - 1, dtype=np.int64) for mode in MODE_NAMES}
    compatible_counts = np.zeros(len(full_bins) - 1, dtype=np.int64)
    competing_counts = np.zeros(len(full_bins) - 1, dtype=np.int64)
    truth_sv_counts = np.zeros(len(full_bins) - 1, dtype=np.int64)

    for start in range(0, events, args.chunk):
        stop = min(start + args.chunk, events)
        rows = slice(start, stop)
        tau_direction = candidate_tau_direction(
            source["reco_visible_tau_lab4"][rows], nu_xyz[rows]
        )
        log_weights = {
            "uniform": np.zeros((stop - start, draws), dtype=np.float64),
            "sv": sv_log_weight(
                tau_direction, measured[rows], available[rows], reco_modes[rows],
                response, temperature,
            ),
            "sv_raw": sv_log_weight(
                tau_direction, measured[rows], available[rows], reco_modes[rows],
                response, 1.0,
            ),
            "global_gaussian": gaussian_log_weight(
                tau_direction, measured[rows], available[rows], GLOBAL_GAUSSIAN_SIGMA_RAD,
            ),
            "matched_shuffle": sv_log_weight(
                tau_direction, shuffled[rows], available[rows], reco_modes[rows],
                response, temperature,
            ),
            "oracle_direction": sv_log_weight(
                tau_direction, truth_direction[rows], available[rows], reco_modes[rows],
                response, temperature,
            ),
        }
        for arm, log_weight in log_weights.items():
            weight, diag = normalized_weights(log_weight)
            mean, moments = weighted_representations(h_samples[rows], weight)
            mean_outputs[arm][rows] = mean.astype(np.float32)
            if arm == "sv":
                moments_sv[rows] = moments.astype(np.float32)
                weight_sv[rows] = weight.astype(np.float32)
            if arm != "uniform":
                for name, value in diag.items():
                    diagnostics[arm][name][rows] = value.astype(np.float32)

        if args.split == "validation":
            truth_angle = opening_angle(tau_direction, truth_direction[rows, None])
            best_draw = np.argmin(np.mean(np.square(truth_angle), axis=-1), axis=1)
            local_index = np.arange(stop - start)
            for side in (0, 1):
                side_available = available[rows, side]
                if not side_available.any():
                    continue
                candidate_angle = opening_angle(
                    tau_direction[:, :, side], measured[rows, None, side]
                )
                for mode in MODE_NAMES:
                    selected = side_available & (reco_modes[rows, side] == mode)
                    if selected.any():
                        hist_add(full_counts[mode], candidate_angle[selected], full_bins)
                        hist_add(core_counts[mode], candidate_angle[selected], core_bins)
                chosen = candidate_angle[local_index, best_draw]
                hist_add(compatible_counts, chosen[side_available], full_bins)
                competitor_mask = np.ones_like(candidate_angle, dtype=bool)
                competitor_mask[local_index, best_draw] = False
                hist_add(
                    competing_counts,
                    candidate_angle[side_available][competitor_mask[side_available]],
                    full_bins,
                )
                truth_sv = opening_angle(truth_direction[rows, side], measured[rows, side])
                hist_add(truth_sv_counts, truth_sv[side_available], full_bins)
        print(json.dumps({"split": args.split, "events": stop, "total": events}), flush=True)

    for array in mean_outputs.values():
        array.flush()
    moments_sv.flush(); weight_sv.flush()
    for arm in diagnostics.values():
        for array in arm.values():
            array.flush()

    strict = (
        np.asarray(posterior["all_draws_valid"], dtype=bool)
        & np.asarray(posterior["truth_h_reco_valid"], dtype=bool)
        & np.all(np.asarray(generated_valid, dtype=bool), axis=1)
    )
    truth_modes = source["modes"].astype(np.int8)
    masks: dict[str, np.ndarray] = {
        "inclusive": strict,
        "one_sv": strict & (available.sum(axis=1) == 1),
        "both_sv": strict & available.all(axis=1),
        "any_sv": strict & available.any(axis=1),
        "threeprong_containing": strict & (truth_modes == 3).any(axis=1),
        "threeprong_x_threeprong": strict & (truth_modes == 3).all(axis=1),
    }
    surface_valid = np.zeros(events, dtype=bool)
    if args.surface_mask is not None:
        surface = load_npz(args.surface_mask, ("global_indices", "surface_valid"))
        position = {int(value): index for index, value in enumerate(surface["global_indices"])}
        try:
            order = np.asarray([position[int(value)] for value in ids], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"Surface mask missing global index {error.args[0]}") from error
        surface_valid = surface["surface_valid"][order].astype(bool)
        masks["exact_surface_valid"] = strict & surface_valid
        masks["exact_surface_invalid"] = strict & ~surface_valid
    np.savez_compressed(
        args.output / "cohorts.npz",
        global_indices=ids,
        labels=source["labels"],
        event_weights=source["weights"],
        truth_modes=truth_modes,
        reco_modes=reco_modes,
        sv_available=available,
        strict=strict,
        surface_valid=surface_valid,
        shuffle_source=shuffle_source,
        **{f"mask_{name}": value for name, value in masks.items()},
    )

    report: dict[str, Any] = {
        "contract": {
            "split": args.split,
            "candidate_tau": "reco_visible_tau_lab4 + sampled neutrino lab momentum",
            "primary_response": "train-only decay-mode vMF core + uniform tail, tempered by disjoint-train sheet-posterior calibration",
            "temperature": temperature,
            "global_gaussian_control_sigma_rad": GLOBAL_GAUSSIAN_SIGMA_RAD,
            "unavailable_side": "unit likelihood",
            "test_loaded": False,
        },
        "counts": {
            "events": events,
            "draws": draws,
            "strict_common": int(strict.sum()),
            "any_sv_strict": int((strict & available.any(axis=1)).sum()),
            "both_sv_strict": int((strict & available.all(axis=1)).sum()),
            "matched_shuffle_changed_available_sides": int(
                np.sum(available & (shuffle_source != np.arange(events)[:, None]))
            ),
            "matched_shuffle_available_sides": int(available.sum()),
        },
        "response_audit": {
            "source_contract": response_report["contract"],
            "source_counts": response_report["counts"],
            "source_sv_models": response_report["response_models"]["sv"],
            "source_temperature": response_report["temperature_calibration"]["sv"],
        },
    }

    if args.split == "validation":
        means = {arm: np.load(args.output / f"mean_{arm}.npy", mmap_mode="r") for arm in ARMS}
        target = posterior["h_truth_reco"]
        canonical = source["h"]
        report["h_metrics_reco_visible_truth_nu_functional"] = {
            cohort: {
                arm: h_metric_block(means[arm], target, mask)
                for arm in ARMS
            }
            for cohort, mask in masks.items()
        }
        report["h_metrics_canonical_coordinate_diagnostic"] = {
            cohort: {
                arm: h_metric_block(means[arm], canonical, mask)
                for arm in ("uniform", "sv", "global_gaussian", "matched_shuffle", "oracle_direction")
            }
            for cohort, mask in masks.items()
        }
        report["posterior_quality"] = {}
        for cohort, mask in masks.items():
            report["posterior_quality"][cohort] = {}
            for arm in ("sv", "sv_raw", "global_gaussian", "matched_shuffle", "oracle_direction"):
                diag = {name: np.asarray(value) for name, value in diagnostics[arm].items()}
                selected = mask & available.any(axis=1)
                report["posterior_quality"][cohort][arm] = {
                    name: quantile_summary(value[selected]) for name, value in diag.items()
                }
                report["posterior_quality"][cohort][arm]["collapse_fraction_ess_below_10pct"] = (
                    float(np.mean(diag["ess"][selected] < 0.1 * draws)) if selected.any() else None
                )
                report["posterior_quality"][cohort][arm]["collapse_fraction_max_weight_above_half"] = (
                    float(np.mean(diag["max_weight"][selected] > 0.5)) if selected.any() else None
                )
        base_error, base_cosine = h_event_values(means["uniform"], target)
        sv_error, sv_cosine = h_event_values(means["sv"], target)
        bootstrap_masks = {
            key: masks[key]
            for key in ("inclusive", "any_sv", "threeprong_x_threeprong", "exact_surface_invalid")
            if key in masks and masks[key].any()
        }
        report["paired_bootstrap_h"] = {
            cohort: {
                "delta_event_mse_sv_minus_unweighted": bootstrap_mean_difference(
                    sv_error, base_error, mask, args.bootstrap_draws, args.seed + 10 * index
                ),
                "delta_event_cosine_sv_minus_unweighted": bootstrap_mean_difference(
                    sv_cosine, base_cosine, mask, args.bootstrap_draws, args.seed + 10 * index + 1
                ),
            }
            for index, (cohort, mask) in enumerate(bootstrap_masks.items())
        }
        np.savez_compressed(
            args.output / "event_metrics.npz",
            global_indices=ids,
            baseline_event_mse=base_error,
            sv_event_mse=sv_error,
            delta_event_mse=sv_error - base_error,
            baseline_event_cosine=base_cosine,
            sv_event_cosine=sv_cosine,
            delta_event_cosine=sv_cosine - base_cosine,
            ess_sv=np.asarray(diagnostics["sv"]["ess"]),
            max_weight_sv=np.asarray(diagnostics["sv"]["max_weight"]),
            entropy_sv=np.asarray(diagnostics["sv"]["entropy"]),
            effective_populated_sv=np.asarray(diagnostics["sv"]["effective_populated"]),
            strict=strict,
            threeprong_x_threeprong=masks["threeprong_x_threeprong"],
            any_sv=masks["any_sv"],
            surface_valid=surface_valid,
        )
        draw_geometry_figure(
            args.output, full_bins, full_counts, core_bins, core_counts,
            compatible_counts, competing_counts, truth_sv_counts,
        )
        strict_any_sv = strict & available.any(axis=1)
        draw_weight_figure(args.output, diagnostics, np.asarray(weight_sv), strict_any_sv)
        draw_h_figure(
            args.output, means["uniform"], means["sv"], target, strict,
            masks["threeprong_x_threeprong"],
        )
    (args.output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(jsonable(report["counts"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
