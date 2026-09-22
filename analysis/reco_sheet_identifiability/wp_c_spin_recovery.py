"""Propagate a frozen reco-SV sheet posterior to the spin endpoint.

The detector response and posterior temperature are fixed from disjoint
training partitions by WP-B.  This script opens the historically reused
validation cohort once for the final attribution test.  SV information enters
only through four sheet weights; no raw geometry is exposed to the H/Z score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wp_b_sv_information as wpb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--validation-inputs", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--wp-b-report", type=Path, required=True)
    parser.add_argument("--sheet-scores", type=Path, required=True)
    parser.add_argument("--sheet-moments", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=100_000)
    parser.add_argument("--posterior-calibration-events", type=int, default=0)
    parser.add_argument("--proposal-lattice", type=int, default=64)
    parser.add_argument("--keep", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in names}


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    weights = weights[order].astype(np.float64)
    positive = weights * (labels == 1)
    negative = weights * (labels == 0)
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return float("nan")
    group = np.cumsum(np.r_[True, np.diff(scores) != 0]) - 1
    negative_per_group = np.bincount(group, weights=negative)
    below = np.r_[0.0, np.cumsum(negative_per_group)[:-1]]
    rank_weight = below[group] + 0.5 * negative_per_group[group]
    return float((positive * rank_weight).sum() / (total_positive * total_negative))


def aucs(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray,
         mask: np.ndarray) -> dict[str, float | int]:
    return {
        "events": int(mask.sum()),
        "auc_unweighted": weighted_auc(labels[mask], scores[mask], np.ones(mask.sum())),
        "auc_weighted": weighted_auc(labels[mask], scores[mask], weights[mask]),
    }


class RankedScore:
    def __init__(self, labels: np.ndarray, scores: np.ndarray):
        self.order = np.argsort(scores, kind="mergesort")
        ordered = scores[self.order]
        self.labels = labels[self.order]
        self.group = np.cumsum(np.r_[True, np.diff(ordered) != 0]) - 1
        self.groups = int(self.group[-1]) + 1

    def auc(self, weights: np.ndarray) -> float:
        w = weights[self.order].astype(np.float64)
        positive = w * (self.labels == 1)
        negative = w * (self.labels == 0)
        total_positive, total_negative = positive.sum(), negative.sum()
        per_group = np.bincount(self.group, weights=negative, minlength=self.groups)
        below = np.r_[0.0, np.cumsum(per_group)[:-1]]
        return float((positive * (below[self.group] + 0.5 * per_group[self.group])).sum()
                     / (total_positive * total_negative))


def bootstrap_attribution(labels: np.ndarray, scores: dict[str, np.ndarray],
                          mask: np.ndarray, draws: int, seed: int) -> dict[str, object]:
    index = np.flatnonzero(mask)
    ranked = {name: RankedScore(labels[index], value[index]) for name, value in scores.items()}
    unit = np.ones(len(index))
    point = {name: obj.auc(unit) for name, obj in ranked.items()}
    rng = np.random.default_rng(seed)
    differences = {"reco_minus_o0": [], "reco_minus_shuffle": [], "o1_minus_o0": [],
                   "recovery_fraction": []}
    for _ in range(draws):
        weight = np.bincount(rng.integers(0, len(index), len(index)), minlength=len(index))
        value = {name: obj.auc(weight) for name, obj in ranked.items()}
        numerator = value["reco"] - value["o0"]
        denominator = value["o1"] - value["o0"]
        differences["reco_minus_o0"].append(numerator)
        differences["reco_minus_shuffle"].append(value["reco"] - value["shuffle"])
        differences["o1_minus_o0"].append(denominator)
        differences["recovery_fraction"].append(numerator / denominator if denominator != 0 else np.nan)
    out: dict[str, object] = {"draws": draws, "seed": seed, "point_auc": point}
    for name, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        if name == "recovery_fraction":
            point_value = ((point["reco"] - point["o0"]) / (point["o1"] - point["o0"]))
        elif name == "reco_minus_o0":
            point_value = point["reco"] - point["o0"]
        elif name == "reco_minus_shuffle":
            point_value = point["reco"] - point["shuffle"]
        else:
            point_value = point["o1"] - point["o0"]
        out[name] = {
            "value": float(point_value),
            "ci_low": float(np.nanquantile(array, 0.025)),
            "ci_high": float(np.nanquantile(array, 0.975)),
            "p_two_sided_sign": float(2.0 * min(np.mean(array <= 0), np.mean(array >= 0))),
        }
    return out


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.module_dir))
    import metspace as ms  # type: ignore
    import sheetspace as ss  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    names = ("labels", "global_indices", "weights", "modes", "reco_h_mode",
             "truth_visible_tau_lab4", "truth_neutrino_lab4")
    train = load_npz(args.train_inputs, names)
    validation = load_npz(args.validation_inputs, names)
    geometry = load_npz(args.geometry, (
        "global_indices", "sv_available", "sv_direction", "sv_delta_theta",
        "tau_pt", "tau_eta", "tau_phi", "sv_length", "ip_direction",
        "ip_track_count", "truth_tau_direction",
    ))
    if not np.array_equal(geometry["global_indices"], np.arange(len(geometry["global_indices"]))):
        raise RuntimeError("Geometry audit global indices are not canonical row indices")

    hashes = wpb.splitmix64(train["global_indices"].astype(np.int64), args.seed + 17)
    response_rows = (hashes % np.uint64(10)) < np.uint64(4)
    response_ids = train["global_indices"][response_rows].astype(np.int64)
    response_modes = train["reco_h_mode"][response_rows]
    response_sv: dict[int, dict[str, float | int]] = {}
    response_ip: dict[int, dict[str, float | int | None]] = {}
    response_length: dict[int, dict[str, float | int]] = {}
    response_tau_direction = wpb.unit_from_eta_phi(
        geometry["tau_eta"][response_ids], geometry["tau_phi"][response_ids]
    )
    response_ip_direction = geometry["ip_direction"][response_ids]
    response_ip_normal = np.cross(response_tau_direction, response_ip_direction)
    response_ip_norm = np.linalg.norm(response_ip_normal, axis=-1, keepdims=True)
    response_ip_available = ((geometry["ip_track_count"][response_ids] > 0)
                             & (response_ip_norm[..., 0] > 1.0e-12))
    response_ip_normal /= np.maximum(response_ip_norm, 1.0e-300)
    response_truth_tau_p = np.linalg.norm(
        train["truth_visible_tau_lab4"][response_rows, ..., :3]
        + train["truth_neutrino_lab4"][response_rows, ..., :3], axis=-1
    )
    response_truth_sheet = ss.truth_sheet(
        train["truth_visible_tau_lab4"][response_rows].astype(np.float64),
        train["truth_neutrino_lab4"][response_rows, ..., :3].astype(np.float64),
    )["sheet"].astype(np.int8)
    response_sv_available = geometry["sv_available"][response_ids] > 0
    topology_prior = wpb.fit_topology_prior(
        response_modes, response_sv_available, response_truth_sheet,
    )
    for mode in wpb.MODE_NAMES:
        dots = []
        residuals = []
        log_length_over_p = []
        for side in (0, 1):
            available = geometry["sv_available"][response_ids, side] > 0
            subset = available & (response_modes[:, side] == mode)
            angle = geometry["sv_delta_theta"][response_ids[subset], side]
            dots.append(np.cos(angle[np.isfinite(angle)]))
            ip_subset = response_ip_available[:, side] & (response_modes[:, side] == mode)
            residuals.append(np.abs(np.sum(
                response_ip_normal[ip_subset, side]
                * geometry["truth_tau_direction"][response_ids[ip_subset], side], axis=-1
            )))
            length_subset = available & (response_modes[:, side] == mode)
            log_length_over_p.append(
                np.log(np.maximum(geometry["sv_length"][response_ids[length_subset], side], 1.0e-12))
                - np.log(np.maximum(response_truth_tau_p[length_subset, side], 1.0e-12))
            )
        response_sv[mode] = wpb.fit_vmf_uniform(
            np.concatenate(dots) if dots else np.empty(0)
        )
        response_ip[mode] = wpb.fit_halfnormal_uniform(
            np.concatenate(residuals) if residuals else np.empty(0)
        )
        response_length[mode] = wpb.fit_log_length_response(
            np.concatenate(log_length_over_p) if log_length_over_p else np.empty(0)
        )
    response = {"sv": response_sv, "ip": response_ip, "length": response_length}

    wpb_report = json.loads(args.wp_b_report.read_text())
    temperatures = {
        channel: float(value["temperature"])
        for channel, value in wpb_report["temperature_calibration"].items()
    }
    selected = np.arange(min(len(validation["labels"]), args.max_events))
    evaluated = wpb.evaluate_rows(
        selected, validation, geometry, response, ms, ss, args, args.seed + 2001,
    )
    posterior_sv = wpb.temperature_posterior(
        evaluated["log_evidence_sv_actual"], temperatures["sv"]
    )
    posterior_ip = wpb.temperature_posterior(
        evaluated["log_evidence_ip_actual"], temperatures["ip"]
    )
    posterior_length = wpb.temperature_posterior(
        evaluated["log_evidence_length_actual"], temperatures["length"]
    )
    posterior_sv_length = wpb.temperature_posterior(
        evaluated["log_evidence_sv_length_actual"], temperatures["sv_length"]
    )
    posterior_all = wpb.temperature_posterior(
        evaluated["log_evidence_all_geometry_actual"], temperatures["all_geometry"]
    )
    posterior_shuffle = wpb.temperature_posterior(
        evaluated["log_evidence_sv_shuffle"], temperatures["sv"]
    )
    posterior_topology = wpb.apply_topology_prior(
        topology_prior, evaluated["reco_modes"], evaluated["available"],
    )

    with np.load(args.sheet_scores, allow_pickle=False) as data:
        base = {name: np.asarray(data[name]) for name in (
            "mask", "labels", "weights", "modes", "truth_sheet",
            "score_flat_o0_joint", "score_flat_o1_joint",
            "score_flat_o0_marginal", "score_flat_o1_marginal",
            "score_flat_o0_likelihood_ratio", "score_flat_o1_likelihood_ratio",
        )}
    n = len(selected)
    for name in base:
        base[name] = base[name][:n]
    if not np.array_equal(evaluated["labels"], base["labels"]):
        raise RuntimeError("Validation label order differs from the frozen sheet-score bank")

    with np.load(args.sheet_moments, allow_pickle=False) as data:
        first_by_sheet = np.stack(
            [np.asarray(data[f"first_flat_sheet{sheet}"][:n]) for sheet in range(4)], axis=1
        )
        second_by_sheet = np.stack(
            [np.asarray(data[f"second_flat_sheet{sheet}"][:n]) for sheet in range(4)], axis=1
        )
    posteriors = {
        "o0": np.full((n, 4), 0.25),
        "topology": posterior_topology,
        "shuffle": posterior_shuffle,
        "ip": posterior_ip,
        "length": posterior_length,
        "sv_length": posterior_sv_length,
        "all_geometry": posterior_all,
        "sv": posterior_sv,
        "reco": posterior_sv,
        "o1": np.eye(4)[base["truth_sheet"]],
    }
    estimator_bank: dict[str, dict[str, np.ndarray]] = {
        "joint": {}, "marginal": {}, "likelihood_ratio": {},
    }
    for arm, posterior in posteriors.items():
        first = np.einsum("rs,rsaj->raj", posterior, first_by_sheet)
        second = np.einsum("rs,rsij->rij", posterior, second_by_sheet)
        estimated = ms.moment_estimators(first[:, 0], first[:, 1], second)
        for estimator in estimator_bank:
            estimator_bank[estimator][arm] = estimated[estimator]
    available = evaluated["available"]
    primary = base["mask"] & evaluated["valid_surface"]
    categories = {
        "inclusive": primary,
        "any_sv": primary & available.any(axis=1),
        "both_sv": primary & available.all(axis=1),
        "reco_threeprong_threeprong": primary & (evaluated["reco_modes"] == 3).all(axis=1),
    }
    metrics = {
        estimator: {
            category: {name: aucs(base["labels"], value, base["weights"], mask)
                       for name, value in scores.items()}
            for category, mask in categories.items()
        }
        for estimator, scores in estimator_bank.items()
    }
    attribution = {
        estimator: {
            category: bootstrap_attribution(
                base["labels"], estimator_bank[estimator], mask,
                args.bootstrap, args.seed + 100 * est_offset + offset,
            )
            for offset, (category, mask) in enumerate(categories.items())
            if mask.sum() >= 200
        }
        for est_offset, estimator in enumerate(("likelihood_ratio", "joint"))
    }
    closure = {}
    for estimator in estimator_bank:
        closure[estimator] = {}
        for oracle in ("o0", "o1"):
            reference = base[f"score_flat_{oracle}_{estimator}"]
            closure[estimator][oracle] = float(np.max(np.abs(
                estimator_bank[estimator][oracle][primary] - reference[primary]
            )))
    report = {
        "contract": {
            "response_and_temperature": "frozen from disjoint WP-B training partitions",
            "evaluation": "historically reused canonical validation cohort; not a blind test",
            "geometry_entry": "four sheet posterior weights only; no raw SV input to H/Z score",
            "primary_selector": "SV direction alone, frozen because train-partition NLL was better than IP plane, SV length, or their combinations",
            "surface": "truth visible + exact tau-neutrino MET, to isolate sheet selection",
            "test_loaded": False,
            "temperatures": temperatures,
        },
        "counts": {
            "validation_events": n,
            "primary_events": int(primary.sum()),
            "any_sv": int(categories["any_sv"].sum()),
            "both_sv": int(categories["both_sv"].sum()),
        },
        "score_closure_max_abs": closure,
        "metrics": metrics,
        "bootstrap_attribution_unweighted": attribution,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output / "spin_scores.npz", global_indices=evaluated["global_indices"],
        labels=base["labels"], weights=base["weights"], primary=primary,
        sv_available=available, reco_modes=evaluated["reco_modes"],
        posterior_sv=posterior_sv, posterior_ip=posterior_ip,
        posterior_length=posterior_length, posterior_sv_length=posterior_sv_length,
        posterior_all_geometry=posterior_all, posterior_reco=posterior_sv,
        posterior_shuffle=posterior_shuffle, posterior_topology=posterior_topology,
        **{f"score_{estimator}_{name}": value
           for estimator, scores in estimator_bank.items() for name, value in scores.items()},
    )

    order = ["o0", "shuffle", "ip", "length", "reco", "o1"]
    labels_plot = ["uniform (O0)", "SV shuffle", "IP plane", "SV length", "SV direction", "truth sheet (O1)"]
    colors = ["#999999", "#E45756", "#F28E2B", "#B279A2", "#4C78A8", "#59A14F"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for ax, category in zip(axes, ("inclusive", "both_sv")):
        values = [metrics["likelihood_ratio"][category][name]["auc_unweighted"] for name in order]
        ax.bar(np.arange(len(order)), values, color=colors)
        ax.set_xticks(np.arange(len(order)), labels_plot, rotation=18, ha="right")
        ax.set_ylim(min(values) - 0.01, max(values) + 0.01)
        ax.set_ylabel("H/Z likelihood-ratio ROC AUC (unweighted)")
        ax.set_title(
            f"{category.replace('_', ' ')} "
            f"(n={metrics['likelihood_ratio'][category]['o0']['events']:,})"
        )
        for i, value in enumerate(values):
            ax.text(i, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("TauSpin WP-C: spin information recovered by reco IP/SV sheet weights", fontsize=14)
    fig.savefig(args.output / "sv_spin_recovery.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(json.dumps(report["bootstrap_attribution_unweighted"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
