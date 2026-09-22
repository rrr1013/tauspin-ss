"""Bootstrap response-relative and truth-nu-relative oracle-gap ratios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from core import h_event_values, jsonable
from train_readouts import auc, auc_with_counts, prepare_auc


def load_npz(
    path: Path, names: tuple[str, ...] | None = None
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        selected = data.files if names is None else names
        return {name: np.asarray(data[name]) for name in selected}


def align_score(path: Path, ids: np.ndarray) -> np.ndarray:
    data = load_npz(path)
    position = {int(value): index for index, value in enumerate(data["global_indices"])}
    result = np.full(len(ids), np.nan, dtype=np.float64)
    rows = np.array([position.get(int(value), -1) for value in ids], dtype=np.int64)
    found = rows >= 0
    result[found] = data["scores"][rows[found]]
    return result


def ratio_summary(values: np.ndarray, point: float, denominator: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    kept = values[finite]
    if not len(kept):
        raise RuntimeError("No finite oracle-gap bootstrap ratios")
    return {
        "fraction": float(point),
        "ci_low": float(np.quantile(kept, 0.025)),
        "ci_high": float(np.quantile(kept, 0.975)),
        "finite_draws": int(len(kept)),
        "denominator_nonpositive_fraction": float(np.mean(denominator <= 0.0)),
    }


def h_ratios(
    error_base: np.ndarray,
    error_sv: np.ndarray,
    error_direction: np.ndarray,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    values = np.stack((error_base[mask], error_sv[mask], error_direction[mask]))
    point_mean = values.mean(axis=1)
    point_numerator = point_mean[0] - point_mean[1]
    point_direction_denominator = point_mean[0] - point_mean[2]
    rng = np.random.default_rng(seed)
    direction_ratio = np.empty(draws, dtype=np.float64)
    truth_nu_ratio = np.empty(draws, dtype=np.float64)
    direction_denominator = np.empty(draws, dtype=np.float64)
    probability = np.full(values.shape[1], 1.0 / values.shape[1])
    for index in range(draws):
        counts = rng.multinomial(values.shape[1], probability)
        means = values @ counts / counts.sum()
        numerator = means[0] - means[1]
        direction_denominator[index] = means[0] - means[2]
        direction_ratio[index] = numerator / direction_denominator[index]
        truth_nu_ratio[index] = numerator / means[0]
    return {
        "events": int(mask.sum()),
        "draws": int(draws),
        "seed": int(seed),
        "same_response_truth_direction_gap_fraction": ratio_summary(
            direction_ratio,
            point_numerator / point_direction_denominator,
            direction_denominator,
        ),
        "relative_mse_reduction_to_exact_technical_target": ratio_summary(
            truth_nu_ratio,
            point_numerator / point_mean[0],
            point_mean[0] * np.ones_like(truth_nu_ratio),
        ),
    }


def auc_ratios(
    score_base: np.ndarray,
    score_sv: np.ndarray,
    score_direction: np.ndarray,
    score_truth_nu: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = mask & np.isfinite(score_base) & np.isfinite(score_sv)
    selected &= np.isfinite(score_direction) & np.isfinite(score_truth_nu)
    y = labels[selected]
    w = weights[selected]
    score_values = (
        score_base[selected], score_sv[selected], score_direction[selected],
        score_truth_nu[selected],
    )
    prepared = tuple(prepare_auc(y, score, w) for score in score_values)
    point_auc = np.array([auc(y, score, w) for score in score_values])
    point_numerator = point_auc[1] - point_auc[0]
    point_direction_denominator = point_auc[2] - point_auc[0]
    point_truth_denominator = point_auc[3] - point_auc[0]
    rng = np.random.default_rng(seed)
    direction_ratio = np.empty(draws, dtype=np.float64)
    truth_nu_ratio = np.empty(draws, dtype=np.float64)
    direction_denominator = np.empty(draws, dtype=np.float64)
    truth_denominator = np.empty(draws, dtype=np.float64)
    probability = np.full(len(y), 1.0 / len(y))
    for index in range(draws):
        counts = rng.multinomial(len(y), probability)
        aucs = np.array([auc_with_counts(item, counts) for item in prepared])
        numerator = aucs[1] - aucs[0]
        direction_denominator[index] = aucs[2] - aucs[0]
        truth_denominator[index] = aucs[3] - aucs[0]
        direction_ratio[index] = numerator / direction_denominator[index]
        truth_nu_ratio[index] = numerator / truth_denominator[index]
    return {
        "events": int(selected.sum()),
        "draws": int(draws),
        "seed": int(seed),
        "same_response_truth_direction_gap_fraction": ratio_summary(
            direction_ratio,
            point_numerator / point_direction_denominator,
            direction_denominator,
        ),
        "truth_nu_functional_gap_fraction": ratio_summary(
            truth_nu_ratio,
            point_numerator / point_truth_denominator,
            truth_denominator,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--validation-posterior", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    source = load_npz(
        args.validation_inputs, ("global_indices", "labels", "weights")
    )
    posterior = load_npz(
        args.validation_posterior, ("global_indices", "h_truth_reco")
    )
    cohorts = load_npz(args.representations / "cohorts.npz")
    ids = source["global_indices"]
    if not np.array_equal(ids, posterior["global_indices"]):
        raise RuntimeError("Input/posterior identity mismatch")
    if not np.array_equal(ids, cohorts["global_indices"]):
        raise RuntimeError("Input/cohort identity mismatch")

    means = {
        "base": np.load(args.representations / "mean_uniform.npy", mmap_mode="r"),
        "sv": np.load(args.representations / "mean_sv.npy", mmap_mode="r"),
        "direction": np.load(
            args.representations / "mean_oracle_direction.npy", mmap_mode="r"
        ),
    }
    target = posterior["h_truth_reco"]
    errors = {name: h_event_values(value, target)[0] for name, value in means.items()}
    scores = {
        name: align_score(args.readouts / f"{name}_validation_scores.npz", ids)
        for name in (
            "baseline_flow_mean", "sv_weighted_mean", "oracle_direction_mean",
            "truth_nu_functional",
        )
    }
    masks = {
        "inclusive": cohorts["mask_inclusive"].astype(bool),
        "threeprong_x_threeprong": cohorts["mask_threeprong_x_threeprong"].astype(bool),
    }
    report = {
        "contract": {
            "direction_arm": "truth tau direction evaluated with the same tempered response; not a detector-independent information oracle",
            "truth_nu_h_denominator": "zero technical-target MSE, so this ratio is the relative MSE reduction",
            "test_loaded": False,
        },
        "h": {},
        "auc": {},
    }
    for index, (cohort, mask) in enumerate(masks.items()):
        report["h"][cohort] = h_ratios(
            errors["base"], errors["sv"], errors["direction"], mask,
            args.draws, 20261200 + 10 * index,
        )
        report["auc"][cohort] = auc_ratios(
            scores["baseline_flow_mean"], scores["sv_weighted_mean"],
            scores["oracle_direction_mean"], scores["truth_nu_functional"],
            source["labels"].astype(np.int8), source["weights"].astype(np.float64),
            mask, args.draws, 20261201 + 10 * index,
        )
    args.output.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
