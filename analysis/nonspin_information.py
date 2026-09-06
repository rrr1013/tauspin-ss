#!/usr/bin/env python3
"""What does the frozen H/Z classifier use besides the tau-pair spin correlation?

The frozen adaptive-v1 three-seed ensembles are read as-is; nothing is trained,
re-selected or re-calibrated here, and the test partition is never touched.

The estimand is a *conditional* AUC: the discrimination that the frozen score
still shows inside strata of a conditioning block.  A drop relative to the
unconditional AUC measures information shared with that block.  It is not a
causal attribution and the pieces are not additive.

Blocks (fixed before any block AUC was computed):

``P-truth``  generator-level production kinematics; the classifier never saw it.
``P-reco``   the reconstructed proxy of the same production kinematics.
``D-reco``   reconstructed decay-side event summaries.
``S-truth``  the truth spin analysers ``omega-`` / ``omega+`` of 2026-09-04.

Production quantities are fixed before the taus decay, so by construction they
carry no tau-pair spin correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Constants fixed before the results were looked at.
# --------------------------------------------------------------------------------------

STRATA_BINS = 10            # quantile bins for a one-dimensional conditioner
STRATA_BINS_STABILITY = (5, 20)
SPIN_BINS = 6               # per axis for the (omega-, omega+) conditioner
PRODUCTION_GRID_BINS = 8    # per axis for the explicit (boson pT, |eta|) grid
ACTIVITY_BINS = 5           # sum(E_T) quantiles added on top of that grid
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 1000
RIDGE = 1.0                 # L2 penalty on the standardised logit coefficients
BOOTSTRAP_SEED = 20260906
PERMUTATION_SEED = 906202

FROZEN_ENSEMBLE_AUC = {
    "full_reco": 0.6150930579720123,
    "tau_object_only": 0.6008338304557016,
    "event_only": 0.5616469239203451,
}
ARMS = ("full_reco", "tau_object_only", "event_only")

# name -> (column expression, transform).  "log" means log1p before standardising.
P_TRUTH = (
    ("truth_boson_pt", "log"),
    ("abs_truth_boson_eta", "lin"),
)
# Generator-level quantities that are *not* production: MET_Truth[NonInt] is the neutrino
# system, so it is decided by the decay partition and is listed for the inventory only.
D_TRUTH = (
    ("truth_met_et", "log"),
    ("truth_met_sumet", "log"),
)
P_RECO = (
    ("total_pt", "log"),
    ("abs_vis_rapidity", "lin"),
    ("met_sumet", "log"),
    ("averageInteractionsPerCrossing", "lin"),
)
D_RECO = (
    ("vis_mass", "log"),
    ("delta_r_tt", "lin"),
    ("abs_delta_phi_tt", "lin"),
    ("abs_delta_eta_tt", "lin"),
    ("tau_pt_minus", "log"),
    ("tau_pt_plus", "log"),
    ("abs_tau_eta_minus", "lin"),
    ("abs_tau_eta_plus", "lin"),
    ("met_et", "log"),
    ("abs_dphi_met_minus", "lin"),
    ("abs_dphi_met_plus", "lin"),
    ("mt_total", "log"),
    ("tau_nTracks_minus", "lin"),
    ("tau_nTracks_plus", "lin"),
    ("tau_rnnJetScore_minus", "lin"),
    ("tau_rnnJetScore_plus", "lin"),
    ("tau_decayMode_minus", "lin"),
    ("tau_decayMode_plus", "lin"),
)
P_TRUTH_ACTIVITY = P_TRUTH + (("met_sumet", "log"),)
BLOCKS = {
    "P-truth": P_TRUTH,
    "P-truth + activity": P_TRUTH_ACTIVITY,
    "P-reco": P_RECO,
    "D-reco": D_RECO,
}

# Single variables used directly as conditioners, so the block result can be attributed.
SINGLE_CONDITIONERS = (
    "abs_truth_boson_eta",
    "abs_vis_rapidity",
    "met_sumet",
    "met_et",
    "averageInteractionsPerCrossing",
)

DECAY_MODE_LABEL = {0: "1p0n", 1: "1p1n", 2: "1pXn", 3: "3p0n", 4: "3pXn"}


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-corrected average ranks, 1-based."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    is_new = np.ones(n, dtype=bool)
    if n > 1:
        is_new[1:] = sorted_values[1:] != sorted_values[:-1]
    group = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n)
    # mean of the 1-based ranks start+1 .. end
    group_mean = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_mean[group]
    return ranks


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with H as the positive class; nan if a class is empty."""
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def conditional_auc(score: np.ndarray, positive: np.ndarray, stratum: np.ndarray) -> float:
    """Pair-count-weighted AUC inside strata: P(s_H > s_Z | same stratum)."""
    total_pairs = 0.0
    total = 0.0
    for value in np.unique(stratum):
        mask = stratum == value
        pos = positive[mask]
        n_pos, n_neg = int(pos.sum()), int(pos.size - pos.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        cell = auc(score[mask], pos)
        weight = float(n_pos) * float(n_neg)
        total += weight * cell
        total_pairs += weight
    if total_pairs == 0.0:
        return float("nan")
    return total / total_pairs


def quantile_strata(values: np.ndarray, bins: int) -> np.ndarray:
    """Stratum ids from pooled (both classes) quantiles of ``values``."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(values.size, dtype=np.int64)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)


def combine_strata(*strata: np.ndarray) -> np.ndarray:
    """Cross product of stratum labellings, relabelled to a dense range."""
    key = np.zeros(strata[0].size, dtype=np.int64)
    for part in strata:
        key = key * (int(part.max()) + 1) + part.astype(np.int64)
    return np.unique(key, return_inverse=True)[1]


def fit_logistic(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """L2-penalised logistic regression by Newton/IRLS; intercept is not penalised."""
    n, p = design.shape
    matrix = np.column_stack([np.ones(n), design])
    beta = np.zeros(p + 1)
    penalty = np.full(p + 1, ridge)
    penalty[0] = 0.0
    for _ in range(100):
        eta = matrix @ beta
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        weight = np.maximum(prob * (1.0 - prob), 1e-8)
        gradient = matrix.T @ (target - prob) - penalty * beta
        hessian = (matrix * weight[:, None]).T @ matrix + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def crossfit_index(design: np.ndarray, positive: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold logit index; each fold is standardised on the other fold only."""
    index = np.full(design.shape[0], np.nan)
    target = positive.astype(np.float64)
    for held in np.unique(fold):
        train = fold != held
        mean = design[train].mean(axis=0)
        scale = design[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        beta = fit_logistic((design[train] - mean) / scale, target[train], RIDGE)
        test = (design[~train] - mean) / scale
        index[~train] = beta[0] + test @ beta[1:]
    if not np.isfinite(index).all():
        raise RuntimeError("cross-fitted index has non-finite entries")
    return index


def build_design(table: dict[str, np.ndarray], block: tuple, keep: np.ndarray) -> np.ndarray:
    """Standardisable design matrix: transformed variables plus their squares."""
    columns = []
    for name, transform in block:
        values = table[name][keep].astype(np.float64)
        if transform == "log":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    raw = np.column_stack(columns)
    centred = (raw - raw.mean(axis=0)) / np.where(raw.std(axis=0) < 1e-12, 1.0, raw.std(axis=0))
    return np.column_stack([centred, centred ** 2])


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------


def cluster_bootstrap(
    statistic, cluster: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, float]:
    """Percentile interval from resampling whole source ROOT files."""
    unique, inverse = np.unique(cluster, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for r in range(replicates):
        picked = rng.integers(0, unique.size, unique.size)
        idx = np.concatenate([members[i] for i in picked])
        values[r] = statistic(idx)
    finite = values[np.isfinite(values)]
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
        float(finite.std()),
    )


def additive_interaction(
    score: np.ndarray, u_bin: np.ndarray, v_bin: np.ndarray, centres: np.ndarray
) -> float:
    """Projection of <score>(u,v) onto u*v after removing the additive part.

    Same statistic as the 2026-09-04 run: least-squares fit of an additive model
    alpha_i + beta_j to the cell means, then project the residual onto s_i t_j.
    """
    n_u, n_v = centres.size, centres.size
    means = np.full((n_u, n_v), np.nan)
    counts = np.zeros((n_u, n_v))
    for i in range(n_u):
        for j in range(n_v):
            cell = (u_bin == i) & (v_bin == j)
            counts[i, j] = cell.sum()
            if counts[i, j] > 0:
                means[i, j] = score[cell].mean()
    filled = np.isfinite(means)
    if filled.sum() < n_u + n_v:
        return float("nan")
    design = []
    for i in range(n_u):
        for j in range(n_v):
            if not filled[i, j]:
                continue
            row = np.zeros(n_u + n_v)
            row[i] = 1.0
            row[n_u + j] = 1.0
            design.append(row)
    design = np.array(design)
    observed = means[filled]
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residual = np.full((n_u, n_v), np.nan)
    residual[filled] = observed - design @ coefficients
    product = np.outer(centres, centres)
    mask = filled
    return float(np.sum(residual[mask] * product[mask]) / np.sum(product[mask] ** 2))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_everything(args: argparse.Namespace) -> dict:
    with np.load(args.row_map, allow_pickle=False) as source:
        row = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.ensemble, allow_pickle=False) as source:
        ens = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.features, allow_pickle=False) as source:
        table = {key: np.asarray(source[key]) for key in source.files}

    key_ens = (
        ens["identity_0"].astype(np.int64) * (1 << 40)
        + ens["identity_1"].astype(np.int64) * (1 << 24)
        + ens["identity_2"].astype(np.int64)
    )
    global_row = table["row_index"]
    key_tab = (
        row["sample_id"][global_row].astype(np.int64) * (1 << 40)
        + row["ntuple_file_index"][global_row].astype(np.int64) * (1 << 24)
        + row["ntuple_entry"][global_row].astype(np.int64)
    )
    if len(np.unique(key_ens)) != key_ens.size or len(np.unique(key_tab)) != key_tab.size:
        raise RuntimeError("identity triples are not unique; refusing to join")
    order = np.argsort(key_ens)
    position = np.searchsorted(key_ens[order], key_tab)
    if position.max() >= key_ens.size or not np.array_equal(key_ens[order][position], key_tab):
        raise RuntimeError("feature table and ensemble identities do not match")
    into = order[position]

    # H is sample_id 0 in the row map and label 1 in the ensemble artifact.
    if not np.array_equal(ens["labels"][into].astype(np.int64), 1 - table["sample_id"].astype(np.int64)):
        raise RuntimeError("label disagreement after the identity join")
    if not np.array_equal(
        ens["source_file_index"][into].astype(np.int64), table["source_file_index"].astype(np.int64)
    ):
        raise RuntimeError("source-file disagreement after the identity join")

    for arm in ARMS:
        table[arm] = ens[f"{arm}_scores"][into]
    table["report_mask"] = ens["report_mask"][into]
    table["is_higgs"] = table["sample_id"] == 0

    for arm in ARMS:
        recomputed = auc(table[arm][table["report_mask"]], table["is_higgs"][table["report_mask"]])
        if abs(recomputed - FROZEN_ENSEMBLE_AUC[arm]) > 1e-9:
            raise RuntimeError(f"{arm}: frozen closure failed ({recomputed!r})")

    spin = None
    if args.spin_table is not None:
        with np.load(args.spin_table, allow_pickle=False) as source:
            spin = {key: np.asarray(source[key]) for key in source.files}
    return {"row": row, "table": table, "spin": spin, "closure": dict(FROZEN_ENSEMBLE_AUC)}


def add_derived(table: dict[str, np.ndarray]) -> None:
    table["abs_truth_boson_eta"] = np.abs(table["truth_boson_eta"])
    table["abs_vis_rapidity"] = np.abs(table["vis_rapidity"])
    table["abs_delta_phi_tt"] = np.abs(table["delta_phi_tt"])
    table["abs_delta_eta_tt"] = np.abs(table["delta_eta_tt"])
    table["abs_tau_eta_minus"] = np.abs(table["tau_eta_minus"])
    table["abs_tau_eta_plus"] = np.abs(table["tau_eta_plus"])
    table["abs_dphi_met_minus"] = np.abs(table["dphi_met_minus"])
    table["abs_dphi_met_plus"] = np.abs(table["dphi_met_plus"])


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def block_index(table: dict[str, np.ndarray], block: tuple, keep: np.ndarray) -> np.ndarray:
    design = build_design(table, block, keep)
    fold = (table["source_file_index"][keep] % 2).astype(np.int64)
    return crossfit_index(design, table["is_higgs"][keep], fold)


def single_variable_auc(table: dict[str, np.ndarray], keep: np.ndarray) -> list[dict]:
    positive = table["is_higgs"][keep]
    rows = []
    for name, block in (
        *[(n, "P-truth") for n, _ in P_TRUTH],
        *[(n, "D-truth") for n, _ in D_TRUTH],
        *[(n, "P-reco") for n, _ in P_RECO],
        *[(n, "D-reco") for n, _ in D_RECO],
    ):
        value = auc(table[name][keep].astype(np.float64), positive)
        rows.append({"variable": name, "block": block, "auc": value,
                     "unsigned": 0.5 + abs(value - 0.5)})
    rows.sort(key=lambda r: -r["unsigned"])
    return rows


def analyse(table: dict[str, np.ndarray], spin: dict | None, reuse: dict | None = None) -> dict:
    keep = table["report_mask"]
    positive = table["is_higgs"][keep]
    cluster = table["source_file_index"][keep]
    scores = {arm: table[arm][keep].astype(np.float64) for arm in ARMS}

    indices = {name: block_index(table, block, keep) for name, block in BLOCKS.items()}
    strata = {name: quantile_strata(values, STRATA_BINS) for name, values in indices.items()}
    strata["P-reco + D-reco"] = combine_strata(strata["P-reco"], strata["D-reco"])
    single = {name: quantile_strata(table[name][keep].astype(np.float64), STRATA_BINS)
              for name in SINGLE_CONDITIONERS}
    # A one-dimensional logit index dilutes a strong variable with a useless one, so the
    # generated production kinematics is also held fixed on an explicit two-dimensional grid.
    production_grid = combine_strata(
        quantile_strata(table["truth_boson_pt"][keep].astype(np.float64), PRODUCTION_GRID_BINS),
        quantile_strata(table["abs_truth_boson_eta"][keep].astype(np.float64), PRODUCTION_GRID_BINS),
    )
    strata["P-truth grid"] = production_grid
    strata["P-truth grid + sumET"] = combine_strata(
        production_grid,
        quantile_strata(table["met_sumet"][keep].astype(np.float64), ACTIVITY_BINS),
    )

    result: dict = {
        "counts": {"report": int(keep.sum()), "H": int(positive.sum()),
                   "Z": int((~positive).sum())},
        "single_variable_auc": single_variable_auc(table, keep),
        "block_index_auc": {name: auc(values, positive) for name, values in indices.items()},
        "unconditional_auc": {arm: auc(scores[arm], positive) for arm in ARMS},
        "conditional_auc": {},
        "conditional_auc_stability": {},
        "permutation_null": {},
    }

    for arm in ARMS:
        result["conditional_auc"][arm] = {
            name: conditional_auc(scores[arm], positive, values)
            for name, values in strata.items()
        }
        result.setdefault("conditional_auc_single", {})[arm] = {
            name: conditional_auc(scores[arm], positive, values)
            for name, values in single.items()
        }

    # Is the hadronic-activity difference still there once the generated production
    # kinematics is held fixed?  This is the mechanism question for sum(E_T).
    production_2d = production_grid
    result["activity_at_fixed_production"] = {
        "sumet_auc": auc(table["met_sumet"][keep].astype(np.float64), positive),
        "sumet_auc_given_P-truth_index": conditional_auc(
            table["met_sumet"][keep].astype(np.float64), positive, strata["P-truth"]),
        "sumet_auc_given_boson_pt_x_abs_eta": conditional_auc(
            table["met_sumet"][keep].astype(np.float64), positive, production_2d),
        "boson_pt_auc": auc(table["truth_boson_pt"][keep].astype(np.float64), positive),
        "abs_boson_eta_auc": auc(table["abs_truth_boson_eta"][keep].astype(np.float64), positive),
    }

    for bins in STRATA_BINS_STABILITY:
        alt = {name: quantile_strata(indices[name], bins) for name in BLOCKS}
        result["conditional_auc_stability"][str(bins)] = {
            arm: {name: conditional_auc(scores[arm], positive, values)
                  for name, values in alt.items()}
            for arm in ARMS
        }

    if reuse is not None and "permutation_null" in reuse:
        result["permutation_null"] = reuse["permutation_null"]
    rng = np.random.default_rng(PERMUTATION_SEED)
    for name in ([] if reuse is not None and "permutation_null" in reuse
                 else ("P-truth", "P-reco", "D-reco", "P-truth grid",
                       "P-truth grid + sumET", "P-reco + D-reco")):
        values = np.empty(PERMUTATION_REPLICATES)
        base = strata[name].copy()
        for r in range(PERMUTATION_REPLICATES):
            shuffled = base.copy()
            for cls in (True, False):
                mask = positive == cls
                shuffled[mask] = rng.permutation(base[mask])
            values[r] = conditional_auc(scores["full_reco"], positive, shuffled)
        result["permutation_null"][name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }

    # ---- one shared cluster bootstrap for every headline number -----------------------
    conditioners = dict(strata)
    conditioners.update({f"var:{name}": values for name, values in single.items()})
    names = []
    for arm in ARMS:
        names.append((arm, "unconditional"))
        for name in conditioners:
            names.append((arm, name))
    for name in BLOCKS:
        names.append(("__index__", name))

    def statistics(idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(names))
        pos = positive[idx]
        for k, (arm, name) in enumerate(names):
            if arm == "__index__":
                out[k] = auc(indices[name][idx], pos)
            elif name == "unconditional":
                out[k] = auc(scores[arm][idx], pos)
            else:
                out[k] = conditional_auc(scores[arm][idx], pos, conditioners[name][idx])
        return out

    unique, inverse = np.unique(cluster, return_inverse=True)
    if reuse is not None and "bootstrap" in reuse:
        result["bootstrap"] = reuse["bootstrap"]
        result["bootstrap_drop"] = reuse["bootstrap_drop"]
        result["bootstrap_clusters"] = reuse["bootstrap_clusters"]
        return result, {"indices": indices, "strata": strata, "scores": scores,
                        "positive": positive, "keep": keep}
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_REPLICATES, len(names)))
    for r in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(0, unique.size, unique.size)
        idx = np.concatenate([members[i] for i in picked])
        draws[r] = statistics(idx)
    result["bootstrap"] = {
        f"{arm}|{name}": {
            "q025": float(np.nanquantile(draws[:, k], 0.025)),
            "q975": float(np.nanquantile(draws[:, k], 0.975)),
            "std": float(np.nanstd(draws[:, k])),
        }
        for k, (arm, name) in enumerate(names)
    }
    # Paired drops: the conditioned and unconditioned values come from the same resample,
    # so the difference is far better determined than either endpoint.
    base_column = {arm: k for k, (arm, name) in enumerate(names) if name == "unconditional"}
    result["bootstrap_drop"] = {}
    for k, (arm, name) in enumerate(names):
        if arm == "__index__" or name == "unconditional":
            continue
        diff = draws[:, base_column[arm]] - draws[:, k]
        result["bootstrap_drop"][f"{arm}|{name}"] = {
            "point": float(result["unconditional_auc"][arm] - (
                result["conditional_auc_single"][arm][name[4:]] if name.startswith("var:")
                else result["conditional_auc"][arm][name])),
            "q025": float(np.nanquantile(diff, 0.025)),
            "q975": float(np.nanquantile(diff, 0.975)),
            "std": float(np.nanstd(diff)),
        }
    result["bootstrap_clusters"] = int(unique.size)

    figure_data = {
        "indices": indices,
        "strata": strata,
        "scores": scores,
        "positive": positive,
        "keep": keep,
    }
    return result, figure_data


def analyse_spin(table: dict, spin: dict, indices: dict, result: dict,
                 reuse: dict | None = None) -> dict:
    """Two-by-two conditioning on the truth spin analysers and on production."""
    keep = table["report_mask"]
    rows_keep = table["row_index"][keep]
    lookup = {int(r): i for i, r in enumerate(spin["row_index"])}
    matched = np.array([lookup.get(int(r), -1) for r in rows_keep])
    cohort = matched >= 0
    take = matched[cohort]

    positive = table["is_higgs"][keep][cohort]
    cluster = table["source_file_index"][keep][cohort]
    score = table["full_reco"][keep][cohort].astype(np.float64)
    omega_minus = spin["cos_star_minus"][take]
    omega_plus = spin["cos_star_plus"][take]
    modes = (spin["mode_minus"][take], spin["mode_plus"][take])

    spin_strata = combine_strata(
        quantile_strata(omega_minus, SPIN_BINS), quantile_strata(omega_plus, SPIN_BINS)
    )
    prod_strata = quantile_strata(indices["P-reco"][cohort], STRATA_BINS)
    both = combine_strata(prod_strata, spin_strata)

    out = {
        "counts": {"cohort": int(cohort.sum()), "H": int(positive.sum()),
                   "Z": int((~positive).sum())},
        "mode_pairs": {
            f"{DECAY_MODE_LABEL[a]} x {DECAY_MODE_LABEL[b]}":
                int(((modes[0] == a) & (modes[1] == b)).sum())
            for a in (0, 1) for b in (0, 1)
        },
        "auc": {
            "unconditional": auc(score, positive),
            "given_production": conditional_auc(score, positive, prod_strata),
            "given_spin": conditional_auc(score, positive, spin_strata),
            "given_both": conditional_auc(score, positive, both),
        },
    }

    # Interaction statistic of 2026-09-04, class by class, with and without production strata.
    edges = np.linspace(-1.0, 1.0, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    u_bin = np.clip(np.searchsorted(edges, omega_minus, side="right") - 1, 0, 9)
    v_bin = np.clip(np.searchsorted(edges, omega_plus, side="right") - 1, 0, 9)
    out["interaction"] = {}
    for label, mask in (("H", positive), ("Z", ~positive)):
        overall = additive_interaction(score[mask], u_bin[mask], v_bin[mask], centres)
        pieces, weights = [], []
        for value in np.unique(prod_strata[mask]):
            cell = prod_strata[mask] == value
            piece = additive_interaction(
                score[mask][cell], u_bin[mask][cell], v_bin[mask][cell], centres
            )
            if np.isfinite(piece):
                pieces.append(piece)
                weights.append(cell.sum())
        conditioned = float(np.average(pieces, weights=weights)) if pieces else float("nan")
        out["interaction"][label] = {"overall": overall, "within_production_strata": conditioned}

    def statistics(idx: np.ndarray) -> np.ndarray:
        pos = positive[idx]
        return np.array([
            auc(score[idx], pos),
            conditional_auc(score[idx], pos, prod_strata[idx]),
            conditional_auc(score[idx], pos, spin_strata[idx]),
            conditional_auc(score[idx], pos, both[idx]),
        ])

    if reuse is not None and "spin_cohort" in reuse:
        out["bootstrap"] = reuse["spin_cohort"]["bootstrap"]
        out["bootstrap_drop"] = reuse["spin_cohort"]["bootstrap_drop"]
        result["spin_cohort"] = out
        return {"cohort": cohort, "score": score, "positive": positive,
                "omega_minus": omega_minus, "omega_plus": omega_plus,
                "prod_strata": prod_strata, "spin_strata": spin_strata}
    unique, inverse = np.unique(cluster, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    draws = np.empty((BOOTSTRAP_REPLICATES, 4))
    for r in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(0, unique.size, unique.size)
        idx = np.concatenate([members[i] for i in picked])
        draws[r] = statistics(idx)
    keys = ("unconditional", "given_production", "given_spin", "given_both")
    for k, key in enumerate(keys):
        out.setdefault("bootstrap", {})[key] = {
            "q025": float(np.nanquantile(draws[:, k], 0.025)),
            "q975": float(np.nanquantile(draws[:, k], 0.975)),
            "std": float(np.nanstd(draws[:, k])),
        }
    for k, key in enumerate(keys[1:], start=1):
        diff = draws[:, 0] - draws[:, k]
        out.setdefault("bootstrap_drop", {})[key] = {
            "point": float(out["auc"]["unconditional"] - out["auc"][key]),
            "q025": float(np.nanquantile(diff, 0.025)),
            "q975": float(np.nanquantile(diff, 0.975)),
            "std": float(np.nanstd(diff)),
        }
    result["spin_cohort"] = out
    return {"cohort": cohort, "score": score, "positive": positive,
            "omega_minus": omega_minus, "omega_plus": omega_plus,
            "prod_strata": prod_strata, "spin_strata": spin_strata}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

BLOCK_COLOUR = {"P-truth": "#1f77b4", "D-truth": "#9467bd",
                "P-truth + activity": "#ff7f0e",
                "P-reco": "#d62728", "D-reco": "#2ca02c"}
ARM_STYLE = {
    "full_reco": ("#111111", "-", "o"),
    "tau_object_only": ("#d62728", "--", "s"),
    "event_only": ("#1f77b4", ":", "^"),
}
ARM_LABEL = {
    "full_reco": "full reco",
    "tau_object_only": "tau objects only",
    "event_only": "event only",
}


def save(fig, outputs: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(outputs / f"{name}.{suffix}", dpi=200 if suffix == "png" else None)
    import matplotlib.pyplot as plt

    plt.close(fig)


def figure_variable_auc(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["single_variable_auc"]
    fig, ax = plt.subplots(figsize=(8.4, 0.28 * len(rows) + 2.2), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([0.5, r["auc"]], [yi, yi], color=BLOCK_COLOUR[r["block"]], lw=1.2, alpha=0.6)
        ax.plot(r["auc"], yi, marker="o", ms=5, color=BLOCK_COLOUR[r["block"]])
    ax.axvline(0.5, color="grey", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r["variable"] for r in rows], fontsize=8)
    ax.set_xlabel("single-variable AUC (H positive; < 0.5 favours Z)")
    ax.set_title(
        "What discriminates H from Z, variable by variable\n"
        f"held-out report subset, {result['counts']['report']:,} events, unit weight",
        fontsize=10,
    )
    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", label=b)
               for b, c in BLOCK_COLOUR.items()]
    for name, value in result["block_index_auc"].items():
        ax.axvline(value, color=BLOCK_COLOUR[name], ls="--", lw=1.0, alpha=0.8)
        handles.append(plt.Line2D([], [], color=BLOCK_COLOUR[name], ls="--",
                                  label=f"{name} index  {value:.4f}"))
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    save(fig, outputs, "nonspin-variable-auc-forest")


def figure_production_distributions(table, figure_data, result, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    keep = figure_data["keep"]
    positive = figure_data["positive"]
    panels = [
        ("truth_boson_pt", "generated boson $p_{T}$ [GeV]", "log"),
        ("abs_truth_boson_eta", r"generated boson $|\eta|$", "lin"),
        ("met_sumet", r"reconstructed $\sum E_{T}$ [GeV]", "lin"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), constrained_layout=True)
    for ax, (name, label, _) in zip(axes, panels):
        values = table[name][keep].astype(np.float64)
        lo, hi = np.quantile(values, (0.001, 0.999))
        bins = np.linspace(lo, hi, 61)
        for cls, colour, style, tag in ((positive, "#d62728", "-", "H"),
                                        (~positive, "#1f77b4", "--", "Z")):
            ax.hist(values[cls], bins=bins, density=True, histtype="step",
                    color=colour, ls=style, lw=1.6, label=tag)
        ax.set_xlabel(label)
        ax.set_ylabel("events / bin (area normalised)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    ax = axes[3]
    index = figure_data["indices"]["P-reco"]
    lo, hi = np.quantile(index, (0.001, 0.999))
    bins = np.linspace(lo, hi, 61)
    for cls, colour, style, tag in ((positive, "#d62728", "-", "H"),
                                    (~positive, "#1f77b4", "--", "Z")):
        ax.hist(index[cls], bins=bins, density=True, histtype="step",
                color=colour, ls=style, lw=1.6, label=tag)
    ax.set_xlabel("P-reco cross-fitted logit index")
    ax.set_ylabel("events / bin (area normalised)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(
        "Production-side inputs: what is actually different between the two samples "
        f"(report subset, {result['counts']['report']:,} events)", fontsize=11)
    save(fig, outputs, "nonspin-production-distributions")


def figure_conditional_auc(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    blocks = ["unconditional", "P-truth grid", "P-truth grid + sumET", "P-reco",
              "D-reco", "P-reco + D-reco"]
    block_labels = ["no\nconditioning",
                    r"generated $(p_{T},|\eta|)$" "\ngrid",
                    r"that grid" "\n" r"$\times\ \sum E_{T}$",
                    "reconstructed\nproduction\n(P-reco)",
                    "reconstructed\ndecay summary\n(D-reco)", "P-reco\n+ D-reco"]
    singles = list(result["conditional_auc_single"]["full_reco"].keys())
    single_labels = {
        "abs_truth_boson_eta": r"generated $|\eta_{\mathrm{boson}}|$",
        "abs_vis_rapidity": r"visible $|y_{\tau\tau}|$",
        "met_sumet": r"$\sum E_{T}$",
        "met_et": r"$E_{T}^{\mathrm{miss}}$",
        "averageInteractionsPerCrossing": r"$\langle\mu\rangle$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.18, 1.0]})
    for ax, keys, labels, title, prefix in (
        (axes[0], blocks, block_labels, "conditioning on a variable block", ""),
        (axes[1], singles, [single_labels[k] for k in singles],
         "conditioning on one variable at a time", "var:"),
    ):
        width = 0.24
        for k, arm in enumerate(ARMS):
            colour, _, marker = ARM_STYLE[arm]
            values, lo, hi = [], [], []
            for key in keys:
                if key == "unconditional":
                    value = result["unconditional_auc"][arm]
                    band = result["bootstrap"][f"{arm}|unconditional"]
                elif prefix:
                    value = result["conditional_auc_single"][arm][key]
                    band = result["bootstrap"][f"{arm}|var:{key}"]
                else:
                    value = result["conditional_auc"][arm][key]
                    band = result["bootstrap"][f"{arm}|{key}"]
                values.append(value)
                lo.append(value - band["q025"])
                hi.append(band["q975"] - value)
            x = np.arange(len(keys)) + (k - 1) * width
            ax.errorbar(x, values, yerr=[lo, hi], fmt=marker, color=colour, ls="none",
                        capsize=3, ms=6, label=ARM_LABEL[arm])
            for xi, value in zip(x, values):
                ax.annotate(f"{value:.4f}", (xi, value), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=6.8, color=colour)
            ax.plot(x, values, color=colour, lw=0.7, alpha=0.35)
        ax.axhline(0.5, color="grey", lw=1.0)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("AUC of the frozen score inside strata (H positive)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower left")
    fig.suptitle(
        "How much of each frozen arm survives when a block or a variable is held fixed  |  "
        f"report subset {result['counts']['report']:,} events, {STRATA_BINS} quantile strata, "
        f"95% cluster bootstrap over {result['bootstrap_clusters']} source ROOT files",
        fontsize=10.5)
    save(fig, outputs, "nonspin-conditional-auc")


def figure_strata_profile(figure_data, result, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    positive = figure_data["positive"]
    strata = figure_data["strata"]["P-reco"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), constrained_layout=True)
    ax = axes[0]
    bins = np.unique(strata)
    for arm in ARMS:
        colour, style, marker = ARM_STYLE[arm]
        values = []
        for b in bins:
            mask = strata == b
            values.append(auc(figure_data["scores"][arm][mask], positive[mask]))
        ax.plot(bins + 1, values, color=colour, ls=style, marker=marker, ms=5,
                label=ARM_LABEL[arm])
    ax.axhline(0.5, color="grey", lw=1.0)
    ax.set_xlabel("P-reco index decile (1 = most Z-like production)")
    ax.set_ylabel("within-stratum AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    counts_h = np.array([((strata == b) & positive).sum() for b in bins])
    counts_z = np.array([((strata == b) & ~positive).sum() for b in bins])
    ax.step(bins + 1, counts_h, where="mid", color="#d62728", lw=1.6, label="H")
    ax.step(bins + 1, counts_z, where="mid", color="#1f77b4", lw=1.6, ls="--", label="Z")
    ax.set_xlabel("P-reco index decile")
    ax.set_ylabel("events in stratum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(
        "Where the production block sits: class balance shifts across deciles, "
        "the within-stratum AUC does not collapse", fontsize=10.5)
    save(fig, outputs, "nonspin-strata-profile")


def figure_spin_decomposition(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    spin = result["spin_cohort"]
    keys = ["unconditional", "given_production", "given_spin", "given_both"]
    labels = ["no conditioning", "given production\n(P-reco)",
              r"given truth spin" "\n" r"($\omega_-,\omega_+$)", "given both"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)
    ax = axes[0]
    values = [spin["auc"][k] for k in keys]
    lo = [values[i] - spin["bootstrap"][k]["q025"] for i, k in enumerate(keys)]
    hi = [spin["bootstrap"][k]["q975"] - values[i] for i, k in enumerate(keys)]
    x = np.arange(len(keys))
    ax.errorbar(x, values, yerr=[lo, hi], fmt="o", color="#111111", capsize=4, ms=7)
    for xi, value in zip(x, values):
        ax.annotate(f"{value:.4f}", (xi, value), textcoords="offset points",
                    xytext=(11, 3), ha="left", fontsize=8.5)
    for i, key in enumerate(keys[1:], start=1):
        drop = spin["bootstrap_drop"][key]
        ax.annotate(f"paired drop\n{drop['point']:.4f}\n[{drop['q025']:.4f}, {drop['q975']:.4f}]",
                    (x[i], values[i]), textcoords="offset points", xytext=(0, -46),
                    ha="center", fontsize=7.4, color="#444444")
    ax.axhline(0.5, color="grey", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.5, len(keys) - 0.3)
    ax.set_ylabel("AUC of the frozen full-reco score")
    ax.set_title(
        r"$\pi/\rho$ truth cohort, " f"{spin['counts']['cohort']:,} events "
        f"(H {spin['counts']['H']:,} / Z {spin['counts']['Z']:,})\n"
        "conditional AUCs are not an additive decomposition", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    width = 0.32
    for k, (label, colour) in enumerate((("H", "#d62728"), ("Z", "#1f77b4"))):
        entry = spin["interaction"][label]
        ax.bar(k - width / 2, entry["overall"], width=width, color=colour,
               label="all events" if k == 0 else None)
        ax.bar(k + width / 2, entry["within_production_strata"], width=width,
               facecolor="white", edgecolor=colour, hatch="////", linewidth=1.4,
               label="averaged inside P-reco deciles" if k == 0 else None)
        for xi, value in ((k - width / 2, entry["overall"]),
                          (k + width / 2, entry["within_production_strata"])):
            ax.annotate(f"{value:.4f}", (xi, value), textcoords="offset points",
                        xytext=(0, -13), ha="center", fontsize=8.5)
    ax.axhline(0.0, color="grey", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H", "Z"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(r"$\omega_-\omega_+$ projection $T$ of the score")
    ax.legend(fontsize=8.5, loc="center left")
    ax.set_title("The spin interaction of 2026-09-04 is not removed by\n"
                 "holding the reconstructed production block fixed\n"
                 "(point estimates; no interval is drawn for $T$)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    save(fig, outputs, "nonspin-spin-decomposition")


def figure_summary(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    ax = axes[0]
    arm = "full_reco"
    base = result["unconditional_auc"][arm]
    entries = [("P-truth grid", r"generated production $(p_{T},|\eta|)$"),
               ("P-truth grid + sumET", r"that grid $\times\ \sum E_{T}$"),
               ("P-reco", "reconstructed production"),
               ("D-reco", "reconstructed decay summaries"),
               ("P-reco + D-reco", "both event-level blocks")]
    y = np.arange(len(entries))[::-1]
    for yi, (key, label) in zip(y, entries):
        value = result["conditional_auc"][arm][key]
        ax.barh(yi, base - value, left=value, color="#888888", alpha=0.55, height=0.5)
        ax.plot(value, yi, "o", color="#111111", ms=6)
        drop = result["bootstrap_drop"][f"{arm}|{key}"]
        ax.annotate(f"{value:.4f}   shared {drop['point']:.4f} "
                    f"[{drop['q025']:.4f}, {drop['q975']:.4f}]", (value, yi),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    va="bottom", fontsize=7.8)
    ax.axvline(base, color="#111111", lw=1.4)
    ax.annotate(f"unconditional\n{base:.4f}", (base, len(entries) - 1.0),
                textcoords="offset points", xytext=(7, 0), va="center", fontsize=8.5)
    lowest = min(result["conditional_auc"][arm][key] for key, _ in entries)
    ax.set_xlim(lowest - 0.012, base + 0.006)
    ax.set_ylim(-0.7, len(entries) - 0.15)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in entries], fontsize=9)
    ax.set_xlabel("AUC of the frozen full-reco score inside strata")
    ax.set_title("Grey bar = discrimination shared with the block\n"
                 "(shared information, not a causal attribution)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    rows = [r for r in result["single_variable_auc"]][:8]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["unsigned"] - 0.5 for r in rows],
            color=[BLOCK_COLOUR[r["block"]] for r in rows], height=0.55, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([r["variable"] for r in rows], fontsize=8.5)
    ax.set_xlabel("|AUC - 0.5| of the single variable")
    ax.set_title("The eight strongest single variables", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Where the frozen classifier's H/Z discrimination is shared", fontsize=11)
    save(fig, outputs, "nonspin-summary")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--spin-table", type=Path, default=None)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--reuse-uncertainty", type=Path, default=None,
                        help="manifest.json of an identical earlier run; reuses its "
                             "bootstrap and permutation draws instead of redoing them")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    loaded = load_everything(args)
    table = loaded["table"]
    add_derived(table)
    args.outputs.mkdir(parents=True, exist_ok=True)

    reuse = None
    if args.reuse_uncertainty is not None:
        reuse = json.loads(args.reuse_uncertainty.read_text())["results"]
    result, figure_data = analyse(table, loaded["spin"], reuse)
    if loaded["spin"] is not None:
        analyse_spin(table, loaded["spin"], figure_data["indices"], result, reuse)
    write_manifest(args, loaded, result)

    figure_variable_auc(result, args.outputs)
    figure_production_distributions(table, figure_data, result, args.outputs)
    figure_conditional_auc(result, args.outputs)
    figure_strata_profile(figure_data, result, args.outputs)
    if "spin_cohort" in result:
        figure_spin_decomposition(result, args.outputs)
    figure_summary(result, args.outputs)

    digests = {p.name: sha256_file(p) for p in sorted(args.outputs.glob("*"))
               if p.name != "artifact-sha256.json"}
    (args.outputs / "artifact-sha256.json").write_text(json.dumps(digests, indent=2, sort_keys=True))
    print(json.dumps(
        {k: result[k] for k in ("counts", "block_index_auc", "unconditional_auc",
                                "conditional_auc")},
        indent=2, sort_keys=True))


def write_manifest(args, loaded, result) -> None:
    manifest = {
        "format_version": 1,
        "estimand": (
            "pair-count-weighted AUC of the frozen score inside quantile strata of a "
            "conditioning block; a drop measures information shared with the block and is "
            "not a causal attribution or an additive decomposition"
        ),
        "blocks": {name: [v for v, _ in block] for name, block in BLOCKS.items()},
        "constants": {
            "strata_bins": STRATA_BINS,
            "strata_bins_stability": list(STRATA_BINS_STABILITY),
            "spin_bins": SPIN_BINS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "ridge": RIDGE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_seed": PERMUTATION_SEED,
        },
        "frozen_closure_auc": loaded["closure"],
        "input_sha256": {
            "nonspin-features.npz": sha256_file(args.features),
            "analysis-surface-row-map.npz": sha256_file(args.row_map),
            "terminal-ensembles.npz": sha256_file(args.ensemble),
            **({"truth-spin-table.npz": sha256_file(args.spin_table)}
               if args.spin_table is not None else {}),
        },
        "results": result,
    }
    (args.outputs / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
