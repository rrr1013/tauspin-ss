"""Combine the three fixed-arm rotation outputs and bootstrap paired contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("none", "lab", "local")
PAIRS = {
    "pi x pi": (0, 0), "pi x rho": (0, 1), "rho x rho": (1, 1),
    "pi x 3pi": (0, 3), "rho x 3pi": (1, 3), "3pi x 3pi": (3, 3),
}


def wauc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    score, label = score[order], label[order]
    wp = (label == 1).astype(np.float64)
    wn = (label == 0).astype(np.float64)
    _, first = np.unique(score, return_index=True)
    bounds = np.append(first, len(score))
    cum_neg = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(np.sum(gp * (cum_neg[bounds[:-1]] + 0.5 * gn)) / (wp.sum() * wn.sum()))


def auc_preparation(score: np.ndarray, label: np.ndarray):
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    _, first = np.unique(sorted_score, return_index=True)
    return order, first


def weighted_auc_prepared(prepared, label: np.ndarray, count: np.ndarray) -> float:
    order, first = prepared
    y, weight = label[order], count[order].astype(np.float64)
    wp, wn = np.where(y == 1, weight, 0.0), np.where(y == 0, weight, 0.0)
    bounds = np.append(first, len(order))
    cumulative_negative = np.concatenate(([0.0], np.cumsum(wn)))
    gp, gn = np.add.reduceat(wp, first), np.add.reduceat(wn, first)
    return float(
        np.sum(gp * (cumulative_negative[bounds[:-1]] + 0.5 * gn))
        / (wp.sum() * wn.sum())
    )


def safe_cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def pair_mask(modes: np.ndarray, a: int, b: int) -> np.ndarray:
    low, high = np.minimum(modes[:, 0], modes[:, 1]), np.maximum(modes[:, 0], modes[:, 1])
    return (low == a) & (high == b)


def complex_moment(values: np.ndarray, mask: np.ndarray) -> dict:
    phi0 = np.arctan2(values[mask, 0, 1], values[mask, 0, 0])
    phi1 = np.arctan2(values[mask, 1, 1], values[mask, 1, 0])
    moment = np.mean(np.exp(1j * (phi0 - phi1)))
    return {
        "n": int(mask.sum()), "real": float(moment.real), "imag": float(moment.imag),
        "amplitude": float(2 * abs(moment)), "phase": float(np.angle(moment)),
    }


def q(values: np.ndarray, probabilities=(0.5, 0.9, 0.95, 0.99)) -> dict:
    return {str(p): float(np.quantile(values, p)) for p in probabilities}


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--truth-check", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    paths = {arm: getattr(args, arm) for arm in ARMS}
    data = {arm: np.load(path) for arm, path in paths.items()}
    reference = data["none"]
    ids = np.asarray(reference["global_indices"])
    labels = np.asarray(reference["labels"], int)
    modes = np.asarray(reference["truth_modes"], int)
    exact = np.asarray(reference["h_exact"], np.float64)
    for arm in ARMS[1:]:
        if not np.array_equal(ids, data[arm]["global_indices"]):
            raise RuntimeError(f"{arm}: event identity mismatch")
        if not np.array_equal(labels, data[arm]["labels"]):
            raise RuntimeError(f"{arm}: label mismatch")
        if not np.array_equal(modes, data[arm]["truth_modes"]):
            raise RuntimeError(f"{arm}: mode mismatch")
        if np.max(np.abs(exact - data[arm]["h_exact"])) > 1e-7:
            raise RuntimeError(f"{arm}: target mismatch")
    truth_check = json.loads(args.truth_check.read_text())
    if truth_check.get("status") != "complete":
        raise RuntimeError("truth rotation check incomplete")

    summary = {"rows": int(len(ids)), "boot": args.boot, "arms": {}, "contrasts": {}}
    arrays = {"global_indices": ids, "labels": labels, "truth_modes": modes}
    event_metrics = {}
    score_sets = {}
    for arm in ARMS:
        source = data[arm]
        h16 = np.asarray(source["h_c16"], np.float64)
        h17 = np.asarray(source["h_c17"], np.float64)
        score16 = np.asarray(source["score_c16"], np.float64)
        score17 = np.asarray(source["score_c17"], np.float64)
        group_score = np.asarray(source["score_c16_average_h"], np.float64)
        h0, havg = h16[:, 0], h16.mean(1)
        diff16 = h16 - h0[:, None]
        diff17 = h17 - h0[:, None]
        d16 = np.sqrt(np.mean(np.sum(diff16 * diff16, axis=(2, 3)), axis=1))
        d17 = np.sqrt(np.mean(np.sum(diff17 * diff17, axis=(2, 3)), axis=1))
        scale = np.sqrt(np.mean(np.sum(h16 * h16, axis=(2, 3)), axis=1))
        relative = d16 / np.maximum(scale, 1e-6)
        score_rms = np.sqrt(np.mean((score16 - score16[:, :1]) ** 2, axis=1))
        error0 = np.mean((h0 - exact) ** 2, axis=(1, 2))
        error_avg = np.mean((havg - exact) ** 2, axis=(1, 2))
        cosine0 = safe_cosine(h0, exact).mean(1)
        cosine_avg = safe_cosine(havg, exact).mean(1)
        auc16 = [wauc(score16[:, j], labels) for j in range(score16.shape[1])]
        auc17 = [wauc(score17[:, j], labels) for j in range(score17.shape[1])]
        auc0, auc_group = wauc(score16[:, 0], labels), wauc(group_score, labels)
        z_rhorho = (labels == 0) & pair_mask(modes, 1, 1)
        per_pair = {}
        for name, pair in PAIRS.items():
            mask = pair_mask(modes, *pair)
            per_pair[name] = {
                "n": int(mask.sum()),
                "mse_unrotated": float(error0[mask].mean()),
                "mse_c16_average": float(error_avg[mask].mean()),
                "auc_unrotated": wauc(score16[mask, 0], labels[mask]),
                "auc_c16_average": wauc(group_score[mask], labels[mask]),
            }
        report = json.loads(paths[arm].with_suffix(".json").read_text())
        summary["arms"][arm] = {
            "c16_defect_mean": float(d16.mean()), "c16_defect_quantiles": q(d16),
            "c17_defect_mean": float(d17.mean()), "c17_defect_quantiles": q(d17),
            "relative_defect_mean": float(relative.mean()), "relative_defect_quantiles": q(relative),
            "near_zero_scale_events": int((scale < 1e-6).sum()),
            "component_rms": np.sqrt(np.mean(diff16 * diff16, axis=(0, 1))).tolist(),
            "score_rms_mean": float(score_rms.mean()), "score_rms_quantiles": q(score_rms),
            "auc_c16": auc16, "auc_c17": auc17,
            "auc_c16_range": [float(min(auc16)), float(max(auc16))],
            "mse_unrotated": float(error0.mean()), "mse_c16_average": float(error_avg.mean()),
            "cosine_unrotated": float(cosine0.mean()), "cosine_c16_average": float(cosine_avg.mean()),
            "auc_unrotated": auc0, "auc_c16_average": auc_group,
            "z_rhorho_moment_unrotated": complex_moment(h0, z_rhorho),
            "z_rhorho_moment_c16_average": complex_moment(havg, z_rhorho),
            "mode_pairs": per_pair,
            "implementation_controls": report,
        }
        if arm == "lab":
            saturated = np.asarray(source["lab_saturated_c16"], bool)
            unsaturated = ~saturated.any(1)
            summary["arms"][arm]["fully_unsaturated"] = {
                "n": int(unsaturated.sum()), "fraction": float(unsaturated.mean()),
                "defect_mean": float(d16[unsaturated].mean()),
                "defect_quantiles": q(d16[unsaturated]),
            }
            arrays["lab_fully_unsaturated"] = unsaturated
        event_metrics[arm] = {
            "d16": d16, "d17": d17, "relative": relative, "score_rms": score_rms,
            "error0": error0, "error_avg": error_avg,
            "cosine0": cosine0, "cosine_avg": cosine_avg,
        }
        score_sets[(arm, "base")] = score16[:, 0]
        score_sets[(arm, "group")] = group_score
        arrays[f"{arm}_d16"] = d16
        arrays[f"{arm}_d17"] = d17
        arrays[f"{arm}_relative"] = relative
        arrays[f"{arm}_score0"] = score16[:, 0]
        arrays[f"{arm}_score_rms"] = score_rms
        arrays[f"{arm}_error0"] = error0
        arrays[f"{arm}_error_avg"] = error_avg

    rng = np.random.default_rng(20261002)
    label_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    prepared = {key: auc_preparation(score, labels) for key, score in score_sets.items()}
    boot = {
        "local_minus_lab_d16": [], "local_minus_none_d16": [],
        "local_minus_lab_score_rms": [], "local_minus_none_score_rms": [],
    }
    for arm in ARMS:
        boot[f"{arm}_mse_group_minus_base"] = []
        boot[f"{arm}_cosine_group_minus_base"] = []
        boot[f"{arm}_auc_group_minus_base"] = []
    for _ in range(args.boot):
        count = np.zeros(len(labels), np.int64)
        for index in label_indices:
            count[index] = rng.multinomial(len(index), np.full(len(index), 1.0 / len(index)))
        total = float(count.sum())
        mean = lambda values: float(np.dot(count, values) / total)
        boot["local_minus_lab_d16"].append(mean(event_metrics["local"]["d16"] - event_metrics["lab"]["d16"]))
        boot["local_minus_none_d16"].append(mean(event_metrics["local"]["d16"] - event_metrics["none"]["d16"]))
        boot["local_minus_lab_score_rms"].append(mean(event_metrics["local"]["score_rms"] - event_metrics["lab"]["score_rms"]))
        boot["local_minus_none_score_rms"].append(mean(event_metrics["local"]["score_rms"] - event_metrics["none"]["score_rms"]))
        for arm in ARMS:
            boot[f"{arm}_mse_group_minus_base"].append(mean(event_metrics[arm]["error_avg"] - event_metrics[arm]["error0"]))
            boot[f"{arm}_cosine_group_minus_base"].append(mean(event_metrics[arm]["cosine_avg"] - event_metrics[arm]["cosine0"]))
            auc_base = weighted_auc_prepared(prepared[(arm, "base")], labels, count)
            auc_group = weighted_auc_prepared(prepared[(arm, "group")], labels, count)
            boot[f"{arm}_auc_group_minus_base"].append(auc_group - auc_base)
    for name, values in boot.items():
        values = np.asarray(values)
        summary["contrasts"][name] = {
            "estimate": float(values.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    summary["truth_rotation_check"] = truth_check
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "event_metrics.npz", **arrays)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
