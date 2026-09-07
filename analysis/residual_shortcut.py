#!/usr/bin/env python3
"""Is the direct regression's H/Z advantage spin information, or a non-spin shortcut?

The 2026-09-07 comparison found that frozen readouts on directly regressed truth
spin statistics reach a higher H/Z AUC than readouts on the neutrino-flow
posterior means.  The hypothesis under test here is that the *regression error*

    delta h = h_hat - h

carries H/Z information that the truth statistic h does not, picked up from
non-spin handles in the reco input -- above all the ditau rapidity and sum(E_T)
identified on 2026-09-06 -- and that the downstream classifier exploits it.

Everything runs on one frozen population: the common hybrid-valid validation
events on which every arm of the 2026-09-07 table is defined.  No model is
retrained here, no checkpoint is re-selected, and the test partition is never
read.  Conditional AUC measures shared information, not causation, and every
number is conditional on this simulation, these samples and single-seed readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from nonspin_information import (
    average_ranks,
    build_design,
    cluster_bootstrap,
    combine_strata,
    crossfit_index,
    quantile_strata,
)

# --------------------------------------------------------------------------------------
# Fixed constants.  Set before looking at any result and not changed afterwards.
# --------------------------------------------------------------------------------------

STRATA_BINS = 10
STRATA_BINS_STABILITY = (5, 20)
GRID_BINS = 8
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 1000
BOOTSTRAP_SEED = 20260908
PERMUTATION_SEED = 908202

# Published parent-overlap weighted AUC of the frozen readouts (2026-09-07 run note).
FROZEN_READOUT_AUC = {
    "direct6-truth": 0.613573060633978,
    "direct15-truth": 0.6161464463056125,
    "direct15-hybrid": 0.6040466942444669,
    "flow-full": 0.6105058129311128,
    "flow-full6": 0.6088583741018966,
    "flow-matched": 0.6072977584147914,
    "flow-matched6": 0.6047478290384538,
}

# The two non-spin handles the 2026-09-06 run named, in the same transforms.
NONSPIN_BLOCK = (("abs_vis_rapidity", "linear"), ("met_sumet", "log"))

REFERENCE_ARMS = ("truth15", "truth6", "hybrid15")
ESTIMATE_ARMS = ("direct6-truth", "direct15-truth", "direct15-hybrid",
                 "flow-full", "flow-matched", "raw-full-trained")
# Residual against the arm's own teacher.
RESIDUAL_OWN = {"direct6-truth": "resid-direct6-truth",
                "direct15-truth": "resid-direct15-truth",
                "direct15-hybrid": "resid-direct15-hybrid",
                "flow-full": "resid-flow-full",
                "flow-matched": "resid-flow-matched"}
# Residual against the truth polarimeter statistics, common reference for every arm.
RESIDUAL_TRUTH = {"direct6-truth": "resid-direct6-truth",
                  "direct15-truth": "resid-direct15-truth",
                  "direct15-hybrid": "residT-direct15-hybrid",
                  "flow-full": "residT-flow-full",
                  "flow-matched": "residT-flow-matched"}

PRIMARY_PAIRS = (("direct15-truth", "flow-full"),
                 ("direct6-truth", "flow-full"),
                 ("direct15-hybrid", "flow-matched"))

COMPONENT_LABEL = ([f"h-{a}" for a in "nrk"] + [f"h+{a}" for a in "nrk"]
                   + [f"h+{a} h-{b}" for a in "nrk" for b in "nrk"])


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------


def weighted_auc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray) -> float:
    """Mann-Whitney AUC with per-event weights; ties split evenly.

    Identical formula to the frozen readout summary, so the unweighted call
    reproduces its published numbers exactly.
    """
    if positive.all() or not positive.any():
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    w = weight[order]
    p = positive[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    pos = np.add.reduceat(w * p, starts)
    neg = np.add.reduceat(w * ~p, starts)
    if pos.sum() <= 0 or neg.sum() <= 0:
        return float("nan")
    return float((pos * (np.cumsum(neg) - 0.5 * neg)).sum() / pos.sum() / neg.sum())


def weighted_conditional_auc(score, positive, weight, stratum) -> float:
    """Pair-weight-weighted AUC inside strata: P(s_H > s_Z | same stratum)."""
    total = 0.0
    total_pairs = 0.0
    for value in np.unique(stratum):
        mask = stratum == value
        pos = positive[mask]
        if pos.all() or not pos.any():
            continue
        w = weight[mask]
        pairs = float(w[pos].sum()) * float(w[~pos].sum())
        if pairs <= 0.0:
            continue
        total += pairs * weighted_auc(score[mask], pos, w)
        total_pairs += pairs
    if total_pairs == 0.0:
        return float("nan")
    return total / total_pairs


def unweighted_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC, kept as an independent implementation for the closure check."""
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_everything(args) -> dict:
    with np.load(args.scores, allow_pickle=False) as source:
        scores = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.features, allow_pickle=False) as source:
        nonspin = {key: np.asarray(source[key]) for key in source.files}

    ids = scores["global_indices"]
    if len(np.unique(ids)) != ids.size:
        raise RuntimeError("duplicate global indices in the score table")
    order = np.argsort(nonspin["row_index"])
    position = np.searchsorted(nonspin["row_index"][order], ids)
    if position.max() >= nonspin["row_index"].size:
        raise RuntimeError("score events are outside the non-spin feature table")
    into = order[position]
    if not np.array_equal(nonspin["row_index"][into], ids):
        raise RuntimeError("non-spin feature table does not cover every score event")

    table = {key: value[into] for key, value in nonspin.items() if key != "row_index"}
    table["row_index"] = ids
    table.update({key: value for key, value in scores.items()})
    # H is sample_id 0 in the row map and label 1 in the readout artifacts.
    if not np.array_equal(table["labels"].astype(np.int64), 1 - table["sample_id"].astype(np.int64)):
        raise RuntimeError("label disagreement after the identity join")
    if not (table["split_id"] == 1).all():
        raise RuntimeError("non-validation rows joined into the score table")

    table["is_higgs"] = table["labels"].astype(np.int64) == 1
    table["abs_vis_rapidity"] = np.abs(table["vis_rapidity"])
    table["abs_truth_boson_eta"] = np.abs(table["truth_boson_eta"])

    residuals = {}
    for path in sorted(args.residuals.glob("*/validation_predictions.npz")):
        name = path.parent.name
        with np.load(path, allow_pickle=False) as source:
            block = {key: np.asarray(source[key]) for key in source.files}
        index = {int(g): i for i, g in enumerate(block["global_indices"])}
        ix = np.array([index[int(g)] for g in ids])
        if not np.array_equal(block["labels"][ix].astype(np.int64), table["labels"].astype(np.int64)):
            raise RuntimeError(f"{name}: label mismatch in the residual feature block")
        residuals[name] = block["h_pred"][ix].reshape(len(ids), -1).astype(np.float64)

    closure = {}
    for arm, published in FROZEN_READOUT_AUC.items():
        value = weighted_auc(table["score_" + arm], table["is_higgs"], table["overlap_weights"])
        closure[arm] = value
        if abs(value - published) > 1e-9:
            raise RuntimeError(f"{arm}: frozen readout closure failed ({value!r} vs {published!r})")
    for arm in FROZEN_READOUT_AUC:
        a = weighted_auc(table["score_" + arm], table["is_higgs"], np.ones(len(ids)))
        b = unweighted_auc(table["score_" + arm], table["is_higgs"])
        if abs(a - b) > 1e-9:
            raise RuntimeError(f"{arm}: the two AUC implementations disagree ({a!r} vs {b!r})")
    return {"table": table, "residuals": residuals, "closure": closure}


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def conditioners(table: dict) -> dict:
    """Stratifications held fixed when asking what a score still knows.

    The non-spin conditioner is the pair named on 2026-09-06: the visible ditau
    rapidity and the reconstructed sum(E_T).  The spin conditioner is the frozen
    truth-statistic readout, refined by the single most discriminating truth
    product so that a one-dimensional compression is not the only spin control.
    """
    positive = table["is_higgs"]
    fold = (table["source_file_index"] % 2).astype(np.int64)
    design = build_design(table, NONSPIN_BLOCK, np.ones(len(positive), bool))
    nonspin_index = crossfit_index(design, positive, fold)
    spin_index = table["score_truth15"].astype(np.float64)
    truth_h = table["truth_h"].astype(np.float64)
    longitudinal = truth_h[:, 1, 2] * truth_h[:, 0, 2]

    strata = {
        "|y(tautau)|": quantile_strata(table["abs_vis_rapidity"], STRATA_BINS),
        "sum(E_T)": quantile_strata(table["met_sumet"], STRATA_BINS),
        "|y| x sum(E_T) grid": combine_strata(
            quantile_strata(table["abs_vis_rapidity"], GRID_BINS),
            quantile_strata(table["met_sumet"], GRID_BINS)),
        "non-spin index": quantile_strata(nonspin_index, STRATA_BINS),
        "truth-spin index": quantile_strata(spin_index, STRATA_BINS),
    }
    strata["truth-spin x h+k h-k"] = combine_strata(
        strata["truth-spin index"], quantile_strata(longitudinal, 5))
    strata["truth-spin x non-spin"] = combine_strata(
        strata["truth-spin index"], strata["non-spin index"])
    return {"strata": strata, "nonspin_index": nonspin_index, "spin_index": spin_index,
            "longitudinal": longitudinal}


def score_names(table: dict) -> list[str]:
    return sorted(key[6:] for key in table if key.startswith("score_"))


POPULATIONS = ("all available", "3pi x 3pi")


def analyse(loaded: dict, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    table = loaded["table"]
    masks = {"all available": np.ones(len(table["labels"]), bool),
             "3pi x 3pi": (table["modes"] == 3).all(1)}
    out = {"closure": loaded["closure"],
           "populations": {name: analyse_population(loaded, masks[name], replicates)
                           for name in POPULATIONS}}
    return out


def analyse_population(loaded: dict, mask: np.ndarray, replicates: int) -> dict:
    table = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
             for k, v in loaded["table"].items()}
    residuals = {k: v[mask] for k, v in loaded["residuals"].items()}
    positive = table["is_higgs"]
    weight = table["overlap_weights"].astype(np.float64)
    ones = np.ones(len(positive))
    cluster = table["source_file_index"]
    cond = conditioners(table)
    strata = cond["strata"]
    names = score_names(table)
    scores = {name: table["score_" + name].astype(np.float64) for name in names}

    result: dict = {
        "counts": {"events": int(len(positive)), "H": int(positive.sum()),
                   "Z": int((~positive).sum())},
        "unconditional_auc": {},
        "conditional_auc": {},
        "conditional_auc_unweighted": {},
        "conditional_auc_stability": {},
        "residual_component_auc": {},
        "nonspin_variables": {},
        "permutation_null": {},
    }

    for name in ("abs_vis_rapidity", "met_sumet", "abs_truth_boson_eta", "truth_boson_pt"):
        values = table[name].astype(np.float64)
        result["nonspin_variables"][name] = {
            "weighted_auc": weighted_auc(values, positive, weight),
            "unweighted_auc": weighted_auc(values, positive, ones)}
    result["nonspin_variables"]["non-spin index"] = {
        "weighted_auc": weighted_auc(cond["nonspin_index"], positive, weight),
        "unweighted_auc": weighted_auc(cond["nonspin_index"], positive, ones)}

    for name, score in scores.items():
        result["unconditional_auc"][name] = {
            "weighted": weighted_auc(score, positive, weight),
            "unweighted": weighted_auc(score, positive, ones)}
        result["conditional_auc"][name] = {
            key: weighted_conditional_auc(score, positive, weight, values)
            for key, values in strata.items()}
        result["conditional_auc_unweighted"][name] = {
            key: weighted_conditional_auc(score, positive, ones, values)
            for key, values in strata.items()}

    for bins in STRATA_BINS_STABILITY:
        alt = {
            "non-spin index": quantile_strata(cond["nonspin_index"], bins),
            "truth-spin index": quantile_strata(cond["spin_index"], bins),
        }
        result["conditional_auc_stability"][str(bins)] = {
            name: {key: weighted_conditional_auc(score, positive, weight, values)
                   for key, values in alt.items()}
            for name, score in scores.items()}

    # Component-level view of the residual: which parts of delta h separate H from Z,
    # and how much of that survives holding the two non-spin handles fixed.
    for name, block in residuals.items():
        rows = []
        for j in range(block.shape[1]):
            values = block[:, j]
            rows.append({
                "component": COMPONENT_LABEL[j] if block.shape[1] == 15 else COMPONENT_LABEL[j],
                "auc": weighted_auc(values, positive, weight),
                "auc_given_nonspin": weighted_conditional_auc(
                    values, positive, weight, strata["non-spin index"]),
                "auc_given_truth_spin": weighted_conditional_auc(
                    values, positive, weight, strata["truth-spin index"]),
                "pearson_abs_y": float(np.corrcoef(values, table["abs_vis_rapidity"])[0, 1]),
                "pearson_log_sumet": float(np.corrcoef(values, np.log1p(table["met_sumet"]))[0, 1]),
            })
        result["residual_component_auc"][name] = rows

    # Cross-fitted logistic index on the raw residual vector: a deterministic second
    # estimator of how much H/Z the residual carries, independent of the readout MLP.
    result["residual_index_auc"] = {}
    fold = (cluster % 2).astype(np.int64)
    for name, block in residuals.items():
        design = np.column_stack([block, block ** 2])
        centre = design.mean(0)
        scale = np.where(design.std(0) < 1e-12, 1.0, design.std(0))
        index = crossfit_index((design - centre) / scale, positive, fold)
        result["residual_index_auc"][name] = {
            "weighted_auc": weighted_auc(index, positive, weight),
            "unweighted_auc": weighted_auc(index, positive, ones),
            "given_nonspin": weighted_conditional_auc(index, positive, weight, strata["non-spin index"]),
            "given_truth_spin": weighted_conditional_auc(index, positive, weight, strata["truth-spin index"]),
            "given_both": weighted_conditional_auc(index, positive, weight, strata["truth-spin x non-spin"]),
        }

    # Profiles: how the residual readout score and the residual itself move with the
    # two non-spin handles, separately for H and Z.
    result["profiles"] = {}
    for variable in ("abs_vis_rapidity", "met_sumet"):
        bins = quantile_strata(table[variable].astype(np.float64), STRATA_BINS)
        entry = {"edges": np.quantile(table[variable].astype(np.float64),
                                      np.linspace(0, 1, STRATA_BINS + 1)).tolist(), "bins": []}
        for b in range(STRATA_BINS):
            mask = bins == b
            row = {"bin": b, "n": int(mask.sum()),
                   "centre": float(np.median(table[variable][mask]))}
            for arm in ("truth15", "direct15-truth", "flow-full",
                        "resid-direct15-truth", "residT-flow-full"):
                for cls, sel in (("H", mask & positive), ("Z", mask & ~positive)):
                    row[f"{arm}/{cls}"] = float(np.mean(scores[arm][sel]))
            entry["bins"].append(row)
        result["profiles"][variable] = entry

    # --------------------------------------------------------------------------------
    # Uncertainty.  One cluster bootstrap shared by every statistic, so differences are
    # paired and much better determined than the individual end points.
    # --------------------------------------------------------------------------------
    tracked = []
    for name in names:
        tracked.append(("auc", name, None))
        for key in ("non-spin index", "truth-spin index", "truth-spin x non-spin",
                    "truth-spin x h+k h-k", "|y| x sum(E_T) grid"):
            tracked.append(("auc", name, key))

    def statistics(idx: np.ndarray) -> np.ndarray:
        pos = positive[idx]
        w = weight[idx]
        out = np.empty(len(tracked))
        for i, (_, name, key) in enumerate(tracked):
            score = scores[name][idx]
            out[i] = (weighted_auc(score, pos, w) if key is None
                      else weighted_conditional_auc(score, pos, w, strata[key][idx]))
        return out

    started = time.time()
    samples = bootstrap_matrix(statistics, cluster, replicates, BOOTSTRAP_SEED,
                               len(tracked))
    result["bootstrap_replicates"] = replicates
    result["bootstrap_seconds"] = time.time() - started
    lookup = {(name, key): i for i, (_, name, key) in enumerate(tracked)}
    point = statistics(np.arange(len(positive)))

    result["intervals"] = {}
    for (name, key), i in lookup.items():
        result["intervals"].setdefault(name, {})[key or "unconditional"] = {
            "value": float(point[i]),
            "95pct_interval": np.quantile(samples[:, i], [0.025, 0.975]).tolist()}

    result["shared_with_conditioner"] = {}
    for name in names:
        base = lookup[(name, None)]
        entry = {}
        for key in ("non-spin index", "truth-spin index", "truth-spin x non-spin",
                    "truth-spin x h+k h-k", "|y| x sum(E_T) grid"):
            j = lookup[(name, key)]
            drop = samples[:, base] - samples[:, j]
            entry[key] = {"shared": float(point[base] - point[j]),
                          "paired_95pct_interval": np.quantile(drop, [0.025, 0.975]).tolist()}
        result["shared_with_conditioner"][name] = entry

    result["arm_differences"] = {}
    for a, b in PRIMARY_PAIRS:
        entry = {}
        for key in (None, "non-spin index", "|y| x sum(E_T) grid", "truth-spin index",
                    "truth-spin x h+k h-k"):
            i, j = lookup[(a, key)], lookup[(b, key)]
            diff = samples[:, i] - samples[:, j]
            entry[key or "unconditional"] = {
                "difference": float(point[i] - point[j]),
                "paired_95pct_interval": np.quantile(diff, [0.025, 0.975]).tolist()}
        result["arm_differences"][f"{a} minus {b}"] = entry

    # Does the stratification itself cost AUC?  Permute stratum labels within class.
    rng = np.random.default_rng(PERMUTATION_SEED)
    for key in ("non-spin index", "truth-spin index", "truth-spin x non-spin"):
        values = np.empty(PERMUTATION_REPLICATES)
        base = strata[key].copy()
        for r in range(PERMUTATION_REPLICATES):
            shuffled = base.copy()
            for cls in (True, False):
                sel = np.flatnonzero(positive == cls)
                shuffled[sel] = base[rng.permutation(sel)]
            values[r] = weighted_conditional_auc(
                scores["direct15-truth"], positive, weight, shuffled)
        result["permutation_null"][key] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "unconditional": result["unconditional_auc"]["direct15-truth"]["weighted"]}
    return result


def bootstrap_matrix(statistic, cluster, replicates, seed, width) -> np.ndarray:
    unique, inverse = np.unique(cluster, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(seed)
    out = np.empty((replicates, width))
    for r in range(replicates):
        picked = rng.integers(0, unique.size, unique.size)
        out[r] = statistic(np.concatenate([members[i] for i in picked]))
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_LABEL = {
    "truth15": "truth h, 15 statistics",
    "truth6": "truth h, 6 statistics",
    "hybrid15": "hybrid g, 15 statistics",
    "direct6-truth": "direct 6D, truth teacher",
    "direct15-truth": "direct 15D, truth teacher",
    "direct15-hybrid": "direct 15D, hybrid teacher",
    "flow-full": "flow 15D, full training",
    "flow-matched": "flow 15D, matched training",
    "flow-full6": "flow 6D, full training",
    "flow-matched6": "flow 6D, matched training",
    "raw-full-trained": "raw reco classifier",
    "resid-direct6-truth": r"$\delta h$  direct 6D",
    "resid-direct15-truth": r"$\delta h$  direct 15D",
    "residT-direct15-truth": r"$\delta h$  direct 15D",
    "resid-direct15-hybrid": r"$\delta g$  direct 15D hybrid",
    "residT-direct15-hybrid": r"$\delta h$  direct 15D hybrid",
    "resid-flow-full": r"$\delta g$  flow full",
    "residT-flow-full": r"$\delta h$  flow full",
    "resid-flow-matched": r"$\delta g$  flow matched",
    "residT-flow-matched": r"$\delta h$  flow matched",
    "residG-direct15-truth": r"$\delta g$  direct 15D truth",
}

RESIDUAL_ARMS = ["resid-direct6-truth", "resid-direct15-truth", "residT-direct15-hybrid",
                 "residT-flow-full", "residT-flow-matched"]
ARM_ORDER = ["truth6", "truth15", "hybrid15", "direct6-truth", "direct15-truth",
             "direct15-hybrid", "flow-full", "flow-matched", "raw-full-trained"]
CONDITION_STYLE = {
    "unconditional": ("#333333", "o"),
    "non-spin index": ("#d62728", "s"),
    "|y| x sum(E_T) grid": ("#ff7f0e", "^"),
    "truth-spin index": ("#1f77b4", "v"),
    "truth-spin x h+k h-k": ("#17becf", "P"),
    "truth-spin x non-spin": ("#7f7f7f", "D"),
}
CAVEAT = ("Conditional AUC measures shared information, not causation. Single-seed frozen "
          "readouts, validation only, simulation-conditional; not a spin-only measurement.")


def save(fig, outputs: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(outputs / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def population_title(result: dict, name: str) -> str:
    counts = result["populations"][name]["counts"]
    return f"{name}: {counts['events']:,} events (H {counts['H']:,} / Z {counts['Z']:,})"


def errorbar_forest(ax, entries, colour_of, x_of_interval=True):
    ticks, labels = [], []
    y = 0
    for group, arms in entries:
        for arm in arms:
            value, (lo, hi), colour, label = colour_of(group, arm)
            ax.plot([lo, hi], [y, y], color=colour, lw=2.2, alpha=0.5, solid_capstyle="butt")
            ax.plot([value], [y], "o", color=colour, ms=6, label=label)
            ticks.append(y)
            labels.append(ARM_LABEL[arm])
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)


def figure_overview(result: dict, outputs: Path) -> None:
    groups = [("reference (truth statistics)", ["truth6", "truth15", "hybrid15"]),
              ("regression estimate", ["direct6-truth", "direct15-truth", "direct15-hybrid",
                                       "flow-full", "flow-matched", "raw-full-trained"]),
              (r"residual $\delta h=\hat h-h$", RESIDUAL_ARMS)]
    colours = {"reference (truth statistics)": "#1f77b4",
               "regression estimate": "#d62728",
               r"residual $\delta h=\hat h-h$": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), layout="constrained", sharex=True)
    for ax, population in zip(axes, POPULATIONS):
        block = result["populations"][population]
        seen = set()

        def entry(group, arm, block=block, seen=seen):
            item = block["intervals"][arm]["unconditional"]
            label = None if group in seen else group
            seen.add(group)
            return item["value"], tuple(item["95pct_interval"]), colours[group], label

        errorbar_forest(ax, groups, entry)
        ax.axvline(0.5, color="k", ls=":", lw=1, alpha=0.5)
        ax.axvline(block["unconditional_auc"]["truth15"]["weighted"], color="#1f77b4",
                   ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("parent-overlap weighted H/Z AUC (H positive)")
        ax.set_title(population_title(result, population), fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(r"The residual $\delta h$ separates H from Z almost as well as truth $h$ itself"
                 "\n" + CAVEAT, fontsize=10)
    save(fig, outputs, "residual-auc-overview")


def figure_decomposition(result: dict, outputs: Path) -> None:
    keys = ["unconditional", "non-spin index", "|y| x sum(E_T) grid",
            "truth-spin index", "truth-spin x non-spin"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", sharey=True)
    x = np.arange(len(RESIDUAL_ARMS))
    for ax, population in zip(axes, POPULATIONS):
        block = result["populations"][population]
        for offset, key in zip(np.linspace(-0.3, 0.3, len(keys)), keys):
            colour, marker = CONDITION_STYLE[key]
            values = np.array([block["intervals"][a][key]["value"] for a in RESIDUAL_ARMS])
            bars = np.array([block["intervals"][a][key]["95pct_interval"]
                             for a in RESIDUAL_ARMS])
            ax.errorbar(x + offset, values,
                        yerr=[values - bars[:, 0], bars[:, 1] - values],
                        fmt=marker, color=colour, ls="none", capsize=3, ms=6,
                        label="unconditional" if key == "unconditional" else f"given {key}")
        ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in RESIDUAL_ARMS], rotation=20, ha="right",
                           fontsize=8.5)
        ax.set_title(population_title(result, population), fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("parent-overlap weighted H/Z AUC of the residual readout")
    axes[0].legend(fontsize=8, loc="center right", framealpha=0.95)
    fig.suptitle(r"Holding truth $h$ fixed removes essentially all of $\delta h$'s H/Z power; "
                 "holding the non-spin pair fixed removes none of it\n" + CAVEAT, fontsize=10)
    save(fig, outputs, "residual-decomposition")


def figure_profiles(result: dict, outputs: Path) -> None:
    block = result["populations"]["all available"]
    arms_top = ["truth15", "direct15-truth", "flow-full"]
    arms_bottom = ["resid-direct15-truth", "residT-flow-full"]
    colour = {"truth15": "#1f77b4", "direct15-truth": "#d62728", "flow-full": "#ff7f0e",
              "resid-direct15-truth": "#2ca02c", "residT-flow-full": "#9467bd"}
    variables = [("abs_vis_rapidity", r"$|y_{\tau\tau}|$ (visible ditau)"),
                 ("met_sumet", r"$\Sigma E_T$ [GeV]")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout="constrained")
    for col, (variable, label) in enumerate(variables):
        entry = block["profiles"][variable]
        centres = np.array([row["centre"] for row in entry["bins"]])
        for row_index, arms in enumerate((arms_top, arms_bottom)):
            ax = axes[row_index, col]
            for arm in arms:
                for cls, ls, marker in (("H", "-", "o"), ("Z", "--", "s")):
                    values = np.array([row[f"{arm}/{cls}"] for row in entry["bins"]])
                    ax.plot(centres, values, ls=ls, marker=marker, ms=4, lw=1.4,
                            color=colour[arm], label=f"{ARM_LABEL[arm]}, {cls}")
            ax.set_xlabel(label)
            ax.set_ylabel("mean readout H probability")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
    axes[0, 0].set_title("readouts on the representation", fontsize=10)
    axes[0, 1].set_title("readouts on the representation", fontsize=10)
    axes[1, 0].set_title(r"readouts on the residual $\delta h$", fontsize=10)
    axes[1, 1].set_title(r"readouts on the residual $\delta h$", fontsize=10)
    fig.suptitle("Decile profiles of the two named non-spin handles; solid H, dashed Z; "
                 "all available events (11,065)\nThe H and Z curves are offset from each "
                 "other but flat in the handle: the separation does not come from it",
                 fontsize=10)
    save(fig, outputs, "residual-nonspin-profiles")


def figure_arm_conditional(result: dict, outputs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    block = result["populations"]["all available"]
    x = np.arange(len(ARM_ORDER))
    for key in ("unconditional", "non-spin index", "|y| x sum(E_T) grid", "truth-spin index"):
        colour, marker = CONDITION_STYLE[key]
        values = np.array([block["intervals"][a][key]["value"] for a in ARM_ORDER])
        bars = np.array([block["intervals"][a][key]["95pct_interval"] for a in ARM_ORDER])
        axes[0].errorbar(x, values, yerr=[values - bars[:, 0], bars[:, 1] - values],
                         fmt=marker, color=colour, ls="none", capsize=3, ms=6,
                         label="unconditional" if key == "unconditional" else f"given {key}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=32, ha="right",
                            fontsize=8)
    axes[0].set_ylabel("parent-overlap weighted H/Z AUC")
    axes[0].axhline(0.5, color="k", ls=":", lw=1, alpha=0.4)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("all available events (11,065)", fontsize=10)

    pairs = [f"{a} minus {b}" for a, b in PRIMARY_PAIRS]
    keys = ["unconditional", "non-spin index", "|y| x sum(E_T) grid", "truth-spin index"]
    positions = []
    labels = []
    slot = 0
    for population in POPULATIONS:
        pblock = result["populations"][population]
        for pair in pairs:
            for offset, key in zip(np.linspace(-0.3, 0.3, len(keys)), keys):
                colour, marker = CONDITION_STYLE[key]
                item = pblock["arm_differences"][pair][key]
                lo, hi = item["paired_95pct_interval"]
                axes[1].errorbar([slot + offset], [item["difference"]],
                                 yerr=[[item["difference"] - lo], [hi - item["difference"]]],
                                 fmt=marker, color=colour, ls="none", capsize=3, ms=6,
                                 label=("unconditional" if key == "unconditional"
                                        else f"given {key}") if slot == 0 else None)
            positions.append(slot)
            labels.append(pair.replace(" minus ", "\n$-$ ") + f"\n[{population}]")
            slot += 1
        slot += 0.5
    axes[1].axhline(0.0, color="k", ls=":", lw=1, alpha=0.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel(r"$\Delta$AUC (direct $-$ flow), paired bootstrap")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title("does the direct advantage move when a conditioner is fixed?",
                      fontsize=10)
    fig.suptitle("Same events, same parent-overlap weights, one shared cluster bootstrap "
                 "over source ROOT files\n" + CAVEAT, fontsize=10)
    save(fig, outputs, "residual-arm-conditional")


def figure_components(result: dict, outputs: Path) -> None:
    block = result["populations"]["all available"]
    arms = [("resid-direct15-truth", "#d62728"), ("residT-flow-full", "#9467bd")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), layout="constrained")
    width = 0.38
    for k, (arm, colour) in enumerate(arms):
        rows = block["residual_component_auc"][arm]
        x = np.arange(len(rows))
        axes[0].bar(x + (k - 0.5) * width, [r["auc"] - 0.5 for r in rows], width,
                    color=colour, alpha=0.85, label=ARM_LABEL[arm],
                    hatch="" if k == 0 else "//", edgecolor="white")
        axes[0].plot(x + (k - 0.5) * width, [r["auc_given_truth_spin"] - 0.5 for r in rows],
                     "k_", ms=9, mew=1.6,
                     label="given the truth-spin index" if k == 0 else None)
        axes[1].bar(x + (k - 0.5) * width, [r["pearson_abs_y"] for r in rows], width,
                    color=colour, alpha=0.85,
                    label=ARM_LABEL[arm] + r" vs $|y_{\tau\tau}|$",
                    hatch="" if k == 0 else "//", edgecolor="white")
        axes[1].plot(x + (k - 0.5) * width, [r["pearson_log_sumet"] for r in rows],
                     "k.", ms=7, label=r"same vs $\log(1+\Sigma E_T)$" if k == 0 else None)
    labels = [r["component"] for r in block["residual_component_auc"]["resid-direct15-truth"]]
    for ax, ylabel in ((axes[0], r"AUC $-$ 0.5 of one residual component"),
                       (axes[1], "Pearson correlation with the non-spin handle")):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[1].set_ylim(-0.12, 0.12)
    fig.suptitle(r"Component structure of $\delta h$: the H/Z signal sits in the same "
                 "components as truth $h$, and none of them tracks the non-spin handles\n"
                 "all available events (11,065), parent-overlap weighted", fontsize=10)
    save(fig, outputs, "residual-components")


def figure_summary(result: dict, outputs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), layout="constrained")
    block = result["populations"]["all available"]
    arms = ["truth15", "direct15-truth", "direct6-truth", "flow-full", "flow-matched",
            "raw-full-trained"]
    y = np.arange(len(arms))[::-1]
    for key, colour, offset, label in (
            ("|y| x sum(E_T) grid", "#ff7f0e", 0.19,
             r"shared with $|y_{\tau\tau}|\times\Sigma E_T$"),
            ("truth-spin index", "#1f77b4", -0.19, r"shared with truth $h$")):
        shared = np.array([block["shared_with_conditioner"][a][key]["shared"] for a in arms])
        bars = np.array([block["shared_with_conditioner"][a][key]["paired_95pct_interval"]
                         for a in arms])
        axes[0].barh(y + offset, shared, 0.34, color=colour, alpha=0.85, label=label)
        axes[0].errorbar(shared, y + offset, xerr=[shared - bars[:, 0], bars[:, 1] - shared],
                         fmt="none", ecolor="k", capsize=2.5, lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=9)
    axes[0].set_xlabel("AUC shared with the conditioner (unconditional $-$ conditional)")
    axes[0].set_xlim(left=-0.012)
    axes[0].axvline(0.0, color="k", lw=1)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("what each arm's H/Z power is made of", fontsize=10)

    y = np.arange(len(RESIDUAL_ARMS))[::-1]
    for key, colour, offset, label in (
            ("|y| x sum(E_T) grid", "#ff7f0e", 0.19,
             r"shared with $|y_{\tau\tau}|\times\Sigma E_T$"),
            ("truth-spin index", "#1f77b4", -0.19, r"shared with truth $h$")):
        shared = np.array([block["shared_with_conditioner"][a][key]["shared"]
                           for a in RESIDUAL_ARMS])
        bars = np.array([block["shared_with_conditioner"][a][key]["paired_95pct_interval"]
                         for a in RESIDUAL_ARMS])
        axes[1].barh(y + offset, shared, 0.34, color=colour, alpha=0.85, label=label)
        axes[1].errorbar(shared, y + offset, xerr=[shared - bars[:, 0], bars[:, 1] - shared],
                         fmt="none", ecolor="k", capsize=2.5, lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ARM_LABEL[a] for a in RESIDUAL_ARMS], fontsize=9)
    axes[1].set_xlabel("AUC shared with the conditioner")
    axes[1].axvline(0.0, color="k", lw=1)
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title(r"what the residual $\delta h$'s H/Z power is made of", fontsize=10)
    fig.suptitle("Shared information, not causal contributions, and not additive; "
                 "all available events (11,065)\n" + CAVEAT, fontsize=10)
    save(fig, outputs, "residual-summary")


FIGURES = (figure_overview, figure_decomposition, figure_profiles,
           figure_arm_conditional, figure_components, figure_summary)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "tauspin-residual-data"
    parser.add_argument("--scores", type=Path, default=data / "residual-shortcut-scores.npz")
    parser.add_argument("--residuals", type=Path, default=data / "features")
    parser.add_argument("--features", type=Path,
                        default=root.parent / "tauspin-nonspin-data" / "nonspin-features.npz")
    parser.add_argument("--outputs", type=Path,
                        default=root / "analysis" / "outputs" / "residual-shortcut-v1")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--figures-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    results_path = args.outputs / "results.json"
    if args.figures_only:
        result = json.loads(results_path.read_text())
    else:
        loaded = load_everything(args)
        result = analyse(loaded, args.replicates)
        results_path.write_text(json.dumps(result, indent=2))
    for figure in FIGURES:
        figure(result, args.outputs)
    manifest = {
        "generated_at_unix": time.time(),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in (("scores", args.scores), ("features", args.features))},
        "figures": sorted(p.name for p in args.outputs.glob("*.png")),
        "bootstrap_replicates": args.replicates,
        "test_loaded": False,
        "population": "common hybrid-valid validation events of the 2026-09-07 comparison",
    }
    (args.outputs / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({name: result["populations"][name]["counts"] for name in POPULATIONS},
                     indent=2))


if __name__ == "__main__":
    main()
