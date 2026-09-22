"""Evaluate SV sheet ranking on the exact reconstructed kinematic surface.

Unlike WP-B/C, both the visible tau systems and MET used to construct the
candidate surface are reconstructed.  Truth is used only to define the audit
target: the reco-surface sheet containing the candidate closest to the two
truth tau flight directions.  Events without an exact four-sheet reco surface
remain outside this hard-target evaluation and are counted in the denominator.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wp_a_label_stability as wpa
import wp_b_sv_information as wpb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--wp-b-report", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-events", type=int, default=10_000)
    parser.add_argument("--evaluation-events", type=int, default=20_000)
    parser.add_argument("--proposal-lattice", type=int, default=96)
    parser.add_argument("--keep", type=int, default=256)
    parser.add_argument("--row-chunk", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def parse_sv_response(report: dict) -> dict[int, dict]:
    by_name = report["response_models"]["sv"]
    return {mode: by_name[name] for mode, name in wpb.MODE_NAMES.items()}


def select_bucket(global_ids: np.ndarray, bucket: int, maximum: int, seed: int) -> np.ndarray:
    hashes = wpb.splitmix64(global_ids.astype(np.int64), seed + 17)
    pool = np.flatnonzero((hashes % np.uint64(10)) == np.uint64(bucket))
    order = np.argsort(hashes[pool])
    return pool[order[:min(maximum, len(order))]]


def evaluate_surface(selected: np.ndarray, source: dict[str, np.ndarray],
                     geometry: dict[str, np.ndarray], response: dict[int, dict],
                     ms: object, args: argparse.Namespace, shuffle_seed: int) -> dict[str, np.ndarray]:
    global_ids = source["global_indices"].astype(np.int64)
    ids = global_ids[selected]
    labels = source["labels"][selected].astype(np.int8)
    reco_modes = source["reco_h_mode"][selected].astype(np.int8)
    reco_visible = source["reco_visible_tau_lab4"][selected].astype(np.float64)
    reco_met = source["reco_met_xy"][selected].astype(np.float64)
    truth_tau = (source["truth_visible_tau_lab4"][selected, ..., :3]
                 + source["truth_neutrino_lab4"][selected, ..., :3])
    truth_tau /= np.maximum(np.linalg.norm(truth_tau, axis=-1, keepdims=True), 1.0e-300)

    available = geometry["sv_available"][ids] > 0
    measured = geometry["sv_direction"][ids].astype(np.float64)
    tau_pt = geometry["tau_pt"][ids]
    tau_eta = geometry["tau_eta"][ids]
    tau_phi = geometry["tau_phi"][ids]
    shuffled = wpb.matched_direction_shuffle(
        measured, available, reco_modes, tau_pt, tau_eta, tau_phi,
        np.random.default_rng(shuffle_seed),
    )

    reco_mass = wpa.invariant_mass(reco_visible)
    algebraically_possible = (
        np.isfinite(reco_visible).all(axis=(1, 2))
        & np.isfinite(reco_met).all(axis=1)
        & (reco_visible[..., 3] > 0.0).all(axis=1)
        & (reco_mass < wpa.TAU_MASS_GEV).all(axis=1)
    )
    n = len(selected)
    retained = np.zeros(n, dtype=np.int16)
    min_rms = np.full((n, 4), np.inf)
    evidence_actual = np.full((n, 4), -np.inf)
    evidence_shuffle = np.full((n, 4), -np.inf)
    rng = np.random.default_rng(shuffle_seed + 1)
    eligible = np.flatnonzero(algebraically_possible)
    for chunk_start in range(0, len(eligible), args.row_chunk):
        block = eligible[chunk_start:chunk_start + args.row_chunk]
        vis = reco_visible[block]
        met = reco_met[block]
        unit = ms.lattice(args.proposal_lattice, len(block), rng).transpose(1, 0, 2)
        drawn = ms.sample_region(vis, met, unit)
        picked = ms.compact(drawn["accepted"], args.keep, rng)
        retained[block] = np.sum(picked >= 0, axis=0).astype(np.int16)
        nu_t = np.take_along_axis(drawn["nu_t"], np.maximum(picked, 0)[..., None], axis=0)
        solved = ms.solution_sheets(vis, met, nu_t)
        valid = solved["valid"] & (picked >= 0)[:, None, :]
        tau = vis[None, None, ..., :3] + solved["nu"]
        tau /= np.maximum(np.linalg.norm(tau, axis=-1, keepdims=True), 1.0e-300)
        angle = wpb.opening_angle(tau, truth_tau[block][None, None])
        rms = np.sqrt(np.mean(angle ** 2, axis=-1))
        min_rms[block] = np.min(np.where(valid, rms, np.inf), axis=0).T

        for measurement, target in (
            (measured[block], evidence_actual),
            (shuffled[block], evidence_shuffle),
        ):
            log_like = np.zeros(valid.shape, dtype=np.float64)
            for side in (0, 1):
                cosine = np.sum(tau[..., side, :] * measurement[None, None, :, side, :], axis=-1)
                for mode, model in response.items():
                    rows = (reco_modes[block, side] == mode) & available[block, side]
                    if rows.any():
                        log_like[..., rows] += wpb.mixture_log_likelihood(cosine[..., rows], model)
            for sheet in range(4):
                target[block, sheet] = wpb.logmeanexp(
                    log_like[:, sheet], valid[:, sheet], axis=0
                )

    valid_surface = np.isfinite(min_rms).all(axis=1) & np.isfinite(evidence_actual).all(axis=1)
    target_sheet = np.full(n, -1, dtype=np.int8)
    target_sheet[valid_surface] = np.argmin(min_rms[valid_surface], axis=1).astype(np.int8)
    ordered = np.sort(min_rms, axis=1)
    return {
        "global_indices": ids, "labels": labels, "reco_modes": reco_modes,
        "available": available, "algebraically_possible": algebraically_possible,
        "valid_surface": valid_surface, "target_sheet": target_sheet,
        "best_rms_angle": ordered[:, 0], "margin": ordered[:, 1] - ordered[:, 0],
        "retained": retained, "log_evidence_actual": evidence_actual,
        "log_evidence_shuffle": evidence_shuffle,
    }


def paired_mean_bootstrap(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                          draws: int, seed: int) -> dict[str, float | int]:
    difference = (a - b)[mask]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        samples[draw] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {
        "events": int(len(difference)), "difference": float(difference.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_two_sided_sign": float(2.0 * min((samples <= 0).mean(), (samples >= 0).mean())),
        "draws": draws, "seed": seed,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.module_dir))
    import metspace as ms  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    names = ("labels", "global_indices", "reco_h_mode", "reco_visible_tau_lab4",
             "reco_met_xy", "truth_visible_tau_lab4", "truth_neutrino_lab4")
    with np.load(args.inputs, allow_pickle=False) as data:
        source = {name: np.asarray(data[name]) for name in names}
    with np.load(args.geometry, allow_pickle=False) as data:
        geometry = {name: np.asarray(data[name]) for name in (
            "global_indices", "sv_available", "sv_direction",
            "tau_pt", "tau_eta", "tau_phi",
        )}
    if not np.array_equal(geometry["global_indices"], np.arange(len(geometry["global_indices"]))):
        raise RuntimeError("Geometry audit global indices are not canonical row indices")
    wpb_report = json.loads(args.wp_b_report.read_text())
    response = parse_sv_response(wpb_report)
    global_ids = source["global_indices"].astype(np.int64)
    calibration_rows = select_bucket(global_ids, 6, args.calibration_events, args.seed)
    evaluation_rows = select_bucket(global_ids, 7, args.evaluation_events, args.seed)
    calibration = evaluate_surface(
        calibration_rows, source, geometry, response, ms, args, args.seed + 3001,
    )
    calibration_mask = calibration["valid_surface"] & calibration["available"].any(axis=1)
    temperature_fit = wpb.fit_temperature(
        calibration["log_evidence_actual"], calibration["target_sheet"], calibration_mask,
    )
    temperature = float(temperature_fit["temperature"])

    evaluated = evaluate_surface(
        evaluation_rows, source, geometry, response, ms, args, args.seed + 4001,
    )
    actual = wpb.temperature_posterior(evaluated["log_evidence_actual"], temperature)
    shuffle = wpb.temperature_posterior(evaluated["log_evidence_shuffle"], temperature)
    uniform = np.full_like(actual, 0.25)
    valid = evaluated["valid_surface"]
    any_sv = valid & evaluated["available"].any(axis=1)
    both_sv = valid & evaluated["available"].all(axis=1)
    categories = {"surface_valid": valid, "any_sv": any_sv, "both_sv": both_sv}
    metrics = {
        category: {
            "uniform": wpb.metric_block(uniform, evaluated["target_sheet"], mask),
            "sv_actual": wpb.metric_block(actual, evaluated["target_sheet"], mask),
            "sv_matched_shuffle": wpb.metric_block(shuffle, evaluated["target_sheet"], mask),
        }
        for category, mask in categories.items()
    }
    target = evaluated["target_sheet"]
    row = np.arange(len(target))
    loss_actual = -np.log(np.maximum(actual[row, np.maximum(target, 0)], 1.0e-300))
    loss_uniform = np.full(len(target), math.log(4.0))
    loss_shuffle = -np.log(np.maximum(shuffle[row, np.maximum(target, 0)], 1.0e-300))
    paired_nll = {
        category: {
            "sv_minus_uniform": paired_mean_bootstrap(
                loss_actual, loss_uniform, mask, args.bootstrap, args.seed + offset,
            ),
            "sv_minus_shuffle": paired_mean_bootstrap(
                loss_actual, loss_shuffle, mask, args.bootstrap, args.seed + 100 + offset,
            ),
        }
        for offset, (category, mask) in enumerate(categories.items()) if mask.any()
    }

    original_truth_probability = actual[row, np.maximum(target, 0)]
    permutation_delta = 0.0
    for permutation_tuple in itertools.permutations(range(4)):
        permutation = np.asarray(permutation_tuple)
        inverse = np.argsort(permutation)
        permuted = actual[:, permutation]
        permuted_target = inverse[np.maximum(target, 0)]
        permuted_truth_probability = permuted[row, permuted_target]
        permutation_delta = max(permutation_delta, float(np.max(np.abs(
            original_truth_probability[valid] - permuted_truth_probability[valid]
        ))))

    report = {
        "contract": {
            "split": "train hash bucket 6 for posterior calibration; bucket 7 for evaluation",
            "surface": "reco visible + reco MET exact mass-shell surface",
            "target": "reco-surface sheet containing the candidate nearest to both truth tau directions",
            "truth_use": "evaluation target only",
            "selector_inputs": "reco PV-to-SV direction and reconstructed decay mode",
            "test_loaded": False,
            "seed": args.seed,
        },
        "counts": {
            "evaluation_events": int(len(evaluation_rows)),
            "algebraically_possible": int(evaluated["algebraically_possible"].sum()),
            "surface_valid": int(valid.sum()),
            "surface_valid_fraction": float(valid.mean()),
            "any_sv": int(any_sv.sum()), "both_sv": int(both_sv.sum()),
        },
        "temperature_calibration": temperature_fit,
        "permutation_invariance": {
            "permutations": math.factorial(actual.shape[1]),
            "max_abs_truth_probability": permutation_delta,
            "dtype": str(actual.dtype),
        },
        "best_rms_angle_mrad": wpa.quantiles(evaluated["best_rms_angle"][valid] * 1.0e3),
        "best_second_margin_mrad": wpa.quantiles(evaluated["margin"][valid] * 1.0e3),
        "metrics": metrics,
        "paired_nll_differences": paired_nll,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output / "reco_surface_posteriors.npz", **evaluated,
        posterior_uniform=uniform, posterior_sv_actual=actual,
        posterior_sv_matched_shuffle=shuffle, temperature=np.asarray(temperature),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    values = [valid.mean(), any_sv.mean(), both_sv.mean()]
    axes[0].bar(range(3), values, color=["#999999", "#4C78A8", "#59A14F"])
    axes[0].set_xticks(range(3), ["surface valid", "valid + any SV", "valid + both SV"])
    axes[0].set(ylabel="fraction of fixed reco audit cohort", ylim=(0, 1),
                title="Hard-target coverage")
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.02, f"{value:.3f}", ha="center")
    for name, color in (("sv_actual", "#4C78A8"), ("sv_matched_shuffle", "#E45756")):
        posterior = actual if name == "sv_actual" else shuffle
        p_target = posterior[row[any_sv], target[any_sv]]
        axes[1].hist(p_target, bins=np.linspace(0, 1, 41), density=True,
                     histtype="step", lw=2, label=name, color=color)
    axes[1].axvline(0.25, color="0.3", ls="--", label="uniform")
    axes[1].set(xlabel="posterior probability of nearest-truth reco sheet", ylabel="density",
                title=f"Reco-surface valid + any SV (n={any_sv.sum():,})")
    axes[1].legend(fontsize=8)
    fig.suptitle("TauSpin WP-D: SV ranking on the exact reconstructed surface", fontsize=14)
    fig.savefig(args.output / "reco_surface_sv.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
