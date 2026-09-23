"""Build non-degenerate SV controls and posterior-support diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_representations import (
    bootstrap_mean_difference,
    load_npz,
    open_output,
    quantile_summary,
    response_from_report,
)
from core import (
    candidate_tau_direction,
    h_event_values,
    h_metric_block,
    jsonable,
    normalized_weights,
    offset_preserving_shuffle,
    opening_angle,
    sv_log_weight,
    unit_vector,
    weighted_representations,
)


CONTROL_ARMS = ("offset_shuffle", "visible_axis")
T_SENSITIVITY = (1.0, 2.0, 4.073573414976854, 8.0)


def draw_validation_figure(
    output: Path,
    measured_offset: np.ndarray,
    shuffled_offset: np.ndarray,
    available: np.ndarray,
    min_truth_angle: np.ndarray,
    means: dict[str, np.ndarray],
    target: np.ndarray,
    masks: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    selected = available
    bins = np.geomspace(1.0e-6, np.pi, 100)
    for values, label, color in (
        (measured_offset[selected], "observed SV offset", "#4C78A8"),
        (shuffled_offset[selected], "matched offset shuffle", "#F28E2B"),
    ):
        axes[0, 0].hist(values, bins=bins, density=True, histtype="step", lw=2,
                        label=label, color=color)
    axes[0, 0].set_xscale("log"); axes[0, 0].set_yscale("log")
    axes[0, 0].set(xlabel="SV - visible-axis angle [rad]", ylabel="density",
                   title="Offset distribution preserved by control")
    axes[0, 0].legend()

    uniform_error, _ = h_event_values(means["uniform"], target)
    for name, label, color in (
        ("sv", "real SV", "#4C78A8"),
        ("offset_shuffle", "offset shuffle", "#F28E2B"),
        ("visible_axis", "visible-axis pseudo-SV", "#59A14F"),
    ):
        error, _ = h_event_values(means[name], target)
        axes[0, 1].hist(
            (error - uniform_error)[masks["any_sv"]], bins=np.linspace(-0.4, 0.4, 101),
            density=True, histtype="step", lw=2, label=label, color=color,
        )
    axes[0, 1].axvline(0.0, color="0.3", ls="--")
    axes[0, 1].set(xlabel="event h error(control) - error(uniform)", ylabel="density",
                   title="Non-degenerate collinearity controls")
    axes[0, 1].legend()

    for mask_name, label, color in (
        ("any_sv", "any SV", "#4C78A8"),
        ("threeprong_x_threeprong", "truth 3p x 3p", "#59A14F"),
        ("reco_threeprong_x_threeprong", "reco 3p x 3p", "#B279A2"),
    ):
        side_mask = masks[mask_name][:, None] & available
        values = np.sort(min_truth_angle[side_mask] * 1.0e3)
        axes[1, 0].plot(values, np.arange(1, len(values) + 1) / len(values),
                        lw=2, label=label, color=color)
    axes[1, 0].axvline(4.551858, color="0.3", ls="--", label="response core sigma")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel="min draw-to-truth tau angle [mrad]", ylabel="side CDF",
                   title="Frozen-posterior directional support")
    axes[1, 0].legend(fontsize=8)

    for name, label, color in (
        ("offset_shuffle", "offset shuffle", "#F28E2B"),
        ("visible_axis", "visible-axis pseudo-SV", "#59A14F"),
    ):
        axes[1, 1].hist(
            diagnostics[name]["ess"][masks["any_sv"]], bins=np.linspace(0, 256, 65),
            density=True, histtype="step", lw=2, label=label, color=color,
        )
    axes[1, 1].set(xlabel="effective sample size", ylabel="density",
                   title="Control posterior leverage")
    axes[1, 1].legend()
    fig.suptitle("Additional SV controls and posterior support", fontsize=14)
    fig.savefig(output / "additional_controls_and_support.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--response-report", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    source = load_npz(
        args.inputs,
        ("global_indices", "reco_h_mode", "reco_visible_tau_lab4", "modes"),
    )
    posterior_names = ("global_indices", "h_truth_reco")
    posterior = load_npz(
        args.posterior,
        posterior_names + (() if args.split == "train" else ("h_samples", "nu_samples_lab4")),
    )
    if not np.array_equal(source["global_indices"], posterior["global_indices"]):
        raise RuntimeError("Input/posterior identity mismatch")
    if args.split == "train":
        if args.generated is None:
            raise ValueError("--generated is required for train")
        h_samples = np.load(args.generated / "train_h_samples.npy", mmap_mode="r")
        nu_xyz = np.load(args.generated / "train_nu_samples_xyz.npy", mmap_mode="r")
    else:
        h_samples = posterior["h_samples"]
        nu_xyz = posterior["nu_samples_lab4"][..., :3]
    events, draws = h_samples.shape[:2]

    geometry = load_npz(
        args.geometry,
        ("global_indices", "sv_available", "sv_direction", "truth_tau_direction"),
    )
    ids = source["global_indices"].astype(np.int64)
    if not np.array_equal(geometry["global_indices"], np.arange(len(geometry["global_indices"]))):
        raise RuntimeError("Geometry rows are not canonical")
    available = geometry["sv_available"][ids] > 0
    measured = geometry["sv_direction"][ids].astype(np.float64)
    truth_direction = geometry["truth_tau_direction"][ids].astype(np.float64)
    visible_axis = unit_vector(source["reco_visible_tau_lab4"][..., :3])
    reco_modes = source["reco_h_mode"].astype(np.int8)

    with np.load(args.representations / "cohorts.npz", allow_pickle=False) as cohort_data:
        if not np.array_equal(cohort_data["global_indices"], ids):
            raise RuntimeError("Representation identity mismatch")
        strict = np.asarray(cohort_data["strict"], dtype=bool)
        surface_valid = np.asarray(cohort_data["surface_valid"], dtype=bool)
        shuffle_source = np.asarray(cohort_data["shuffle_source"], dtype=np.int64)
        masks = {
            name.removeprefix("mask_"): np.asarray(cohort_data[name], dtype=bool)
            for name in cohort_data.files if name.startswith("mask_")
        }
    masks["no_sv"] = strict & ~available.any(axis=1)
    masks["reco_threeprong_containing"] = strict & (reco_modes == 3).any(axis=1)
    masks["reco_threeprong_x_threeprong"] = strict & (reco_modes == 3).all(axis=1)
    masks["any_sv_exact_surface_valid"] = strict & available.any(axis=1) & surface_valid
    masks["any_sv_exact_surface_invalid"] = strict & available.any(axis=1) & ~surface_valid

    offset_measurement = offset_preserving_shuffle(
        measured, visible_axis, shuffle_source, available
    )
    measured_offset = opening_angle(measured, visible_axis)
    shuffled_offset = opening_angle(offset_measurement, visible_axis)
    np.save(args.output / "offset_shuffle_direction.npy", offset_measurement.astype(np.float32))

    response_report = json.loads(args.response_report.read_text())
    response, temperature = response_from_report(response_report)
    sigma_rad = float(response[3]["effective_sigma_mrad"]) * 1.0e-3

    means_out = {
        name: open_output(args.output / f"mean_{name}.npy", (events, 2, 3))
        for name in CONTROL_ARMS
    }
    diagnostics_out = {
        name: {
            field: open_output(args.output / f"{field}_{name}.npy", (events,))
            for field in ("ess", "max_weight", "normalized_entropy", "log_weight_span")
        }
        for name in CONTROL_ARMS
    }
    min_truth_angle = open_output(args.output / "min_truth_angle_side.npy", (events, 2))
    min_truth_angle[:] = np.nan
    train_sensitivity = {
        value: {"squared_error_sum": 0.0, "component_count": 0,
                "any_sv_squared_error_sum": 0.0, "any_sv_component_count": 0}
        for value in T_SENSITIVITY
    }
    target = np.asarray(posterior["h_truth_reco"], dtype=np.float64)

    for start in range(0, events, args.chunk):
        stop = min(start + args.chunk, events)
        rows = slice(start, stop)
        tau_direction = candidate_tau_direction(
            source["reco_visible_tau_lab4"][rows], nu_xyz[rows]
        )
        raw_log_weight = sv_log_weight(
            tau_direction, measured[rows], available[rows], reco_modes[rows], response, 1.0
        )
        control_log_weights = {
            "offset_shuffle": sv_log_weight(
                tau_direction, offset_measurement[rows], available[rows],
                reco_modes[rows], response, temperature,
            ),
            "visible_axis": sv_log_weight(
                tau_direction, visible_axis[rows], available[rows],
                reco_modes[rows], response, temperature,
            ),
        }
        for name, log_weight in control_log_weights.items():
            weight, diagnostics = normalized_weights(log_weight)
            mean, _ = weighted_representations(h_samples[rows], weight)
            means_out[name][rows] = mean.astype(np.float32)
            for field in diagnostics_out[name]:
                diagnostics_out[name][field][rows] = diagnostics[field].astype(np.float32)

        truth_angle = opening_angle(tau_direction, truth_direction[rows, None])
        minimum = np.min(truth_angle, axis=1)
        minimum[~available[rows]] = np.nan
        min_truth_angle[rows] = minimum.astype(np.float32)

        if args.split == "train":
            for value in T_SENSITIVITY:
                weight, _ = normalized_weights(raw_log_weight / value)
                mean, _ = weighted_representations(h_samples[rows], weight)
                local_strict = strict[rows]
                residual = mean[local_strict] - target[rows][local_strict]
                train_sensitivity[value]["squared_error_sum"] += float(np.sum(residual * residual))
                train_sensitivity[value]["component_count"] += int(residual.size)
                local_any = masks["any_sv"][rows]
                residual_any = mean[local_any] - target[rows][local_any]
                train_sensitivity[value]["any_sv_squared_error_sum"] += float(
                    np.sum(residual_any * residual_any)
                )
                train_sensitivity[value]["any_sv_component_count"] += int(residual_any.size)
        print(json.dumps({"split": args.split, "events": stop, "total": events}), flush=True)

    for value in means_out.values():
        value.flush()
    for values in diagnostics_out.values():
        for value in values.values():
            value.flush()
    min_truth_angle.flush()

    report: dict[str, Any] = {
        "contract": {
            "split": args.split,
            "offset_shuffle": "permute SV-minus-visible local angular coordinates using the existing mode/pt/eta/phi matched source indices, then re-anchor on each event's own reco-visible axis",
            "visible_axis": "replace measured SV direction by the event's reco-visible tau direction on the same SV-available sides",
            "test_loaded": False,
        },
        "counts": {
            name: int(mask.sum())
            for name, mask in masks.items()
            if args.split == "validation" or "exact_surface" not in name
        },
        "offset_preservation": {
            "measured_angle_rad": quantile_summary(measured_offset[available]),
            "shuffled_angle_rad": quantile_summary(shuffled_offset[available]),
            "sorted_max_abs_difference_rad": float(np.max(np.abs(
                np.sort(measured_offset[available]) - np.sort(shuffled_offset[available])
            ))),
        },
    }

    if args.split == "train":
        report["train_only_temperature_sensitivity"] = {
            str(value): {
                "inclusive_mse": values["squared_error_sum"] / values["component_count"],
                "any_sv_mse": (
                    values["any_sv_squared_error_sum"] / values["any_sv_component_count"]
                ),
            }
            for value, values in train_sensitivity.items()
        }
    else:
        control_means = {
            "uniform": np.load(args.representations / "mean_uniform.npy", mmap_mode="r"),
            "sv": np.load(args.representations / "mean_sv.npy", mmap_mode="r"),
            **{name: np.load(args.output / f"mean_{name}.npy", mmap_mode="r")
               for name in CONTROL_ARMS},
        }
        report["h_metrics"] = {
            cohort: {
                name: h_metric_block(mean, target, mask)
                for name, mean in control_means.items()
            }
            for cohort, mask in masks.items()
        }
        report["paired_bootstrap_h"] = {}
        base_error, _ = h_event_values(control_means["uniform"], target)
        sv_error, _ = h_event_values(control_means["sv"], target)
        offset_error, _ = h_event_values(control_means["offset_shuffle"], target)
        visible_error, _ = h_event_values(control_means["visible_axis"], target)
        bootstrap_cohorts = (
            "inclusive", "any_sv", "threeprong_x_threeprong",
            "reco_threeprong_x_threeprong", "any_sv_exact_surface_valid",
            "any_sv_exact_surface_invalid",
        )
        for index, cohort in enumerate(bootstrap_cohorts):
            mask = masks[cohort]
            report["paired_bootstrap_h"][cohort] = {
                "sv_minus_uniform": bootstrap_mean_difference(
                    sv_error, base_error, mask, args.bootstrap_draws, 20261000 + 10 * index
                ),
                "sv_minus_offset_shuffle": bootstrap_mean_difference(
                    sv_error, offset_error, mask, args.bootstrap_draws, 20261001 + 10 * index
                ),
                "sv_minus_visible_axis": bootstrap_mean_difference(
                    sv_error, visible_error, mask, args.bootstrap_draws, 20261002 + 10 * index
                ),
            }
        minimum = np.asarray(min_truth_angle)
        report["posterior_truth_direction_support"] = {}
        for cohort in (
            "any_sv", "threeprong_x_threeprong", "reco_threeprong_x_threeprong",
            "any_sv_exact_surface_valid", "any_sv_exact_surface_invalid",
        ):
            side_mask = masks[cohort][:, None] & available
            values = minimum[side_mask]
            report["posterior_truth_direction_support"][cohort] = {
                "available_sides": int(side_mask.sum()),
                "min_angle_rad": quantile_summary(values),
                "fraction_with_draw_within_1sigma": float(np.mean(values <= sigma_rad)),
                "fraction_with_draw_within_2sigma": float(np.mean(values <= 2.0 * sigma_rad)),
            }
        report["posterior_quality"] = {
            cohort: {
                name: {
                    field: quantile_summary(np.asarray(values[field])[mask & available.any(axis=1)])
                    for field in values
                }
                for name, values in diagnostics_out.items()
            }
            for cohort, mask in masks.items()
        }
        draw_validation_figure(
            args.output, measured_offset, shuffled_offset, available, minimum,
            control_means, target, masks,
            {name: {field: np.asarray(value) for field, value in values.items()}
             for name, values in diagnostics_out.items()},
        )

    (args.output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
