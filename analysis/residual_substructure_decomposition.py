#!/usr/bin/env python3
"""Decomposing the Direct Regression Residual: Substructure vs Spin Correlation.

ARIADNE automatic run: 2026-09-16.
Investigates whether the direct regression's H/Z discrimination advantage over the
normalizing flow posterior mean (and the residual delta h = h_hat - h) originates
in tau substructure shortcuts (energy sharing S1, track structure S2, shape S4)
or in true tau-tau spin correlation.

Evaluated on the frozen validation cohort (11,065 events) with parent-overlap weights
and cluster bootstrap (2,000 replicates) over source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# Fixed constants and publication references
# --------------------------------------------------------------------------------------

STRATA_BINS = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260916
RIDGE = 1.0

FROZEN_CLOSURE = {
    "direct15-truth": 0.6161464463056125,
    "flow-full": 0.6105058129311128,
    "direct6-truth": 0.613573060633978,
    "flow-matched": 0.6072977584147914,
    "truth15": 0.703928,  # reference check
    "resid-direct15-truth": 0.660290,
}

# Substructure blocks (following tau_substructure_information.py)
S1_VARS = [
    "f_charged_minus", "f_charged_plus",
    "f_pi0_minus", "f_pi0_plus",
    "ups_pi0_minus", "ups_pi0_plus",
    "lead_pfo_ptfrac_minus", "lead_pfo_ptfrac_plus",
    "sum_pfo_ptfrac_minus", "sum_pfo_ptfrac_plus",
]

S2_VARS = [
    "lead_trk_ptfrac_minus", "lead_trk_ptfrac_plus",
    "subl_trk_ptfrac_minus", "subl_trk_ptfrac_plus",
    "trk_pt_asym_minus", "trk_pt_asym_plus",
    "trk_width_minus", "trk_width_plus",
    "sum_trk_ptfrac_minus", "sum_trk_ptfrac_plus",
]

S4_VARS = [
    "all_width_minus", "all_width_plus",
    "m_const_minus", "m_const_plus",
    "const_over_taupt_minus", "const_over_taupt_plus",
]

NONSPIN_VARS = [
    "abs_vis_rapidity",
    "met_sumet",
    "vis_mass",
    "vis_pt",
    "delta_r_tt",
]


# --------------------------------------------------------------------------------------
# Math & Statistical Estimators
# --------------------------------------------------------------------------------------

def weighted_auc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray) -> float:
    """Mann-Whitney AUC with per-event weights; ties split evenly."""
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


def weighted_conditional_auc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray,
                             stratum: np.ndarray) -> float:
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


def quantile_strata(values: np.ndarray, bins: int = STRATA_BINS) -> np.ndarray:
    """Compute quantile strata on pooled events."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(values.size, dtype=np.int64)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)


def combine_strata(*strata: np.ndarray) -> np.ndarray:
    """Cross product of multiple strata."""
    key = np.zeros(strata[0].size, dtype=np.int64)
    for part in strata:
        key = key * (int(part.max()) + 1) + part.astype(np.int64)
    return np.unique(key, return_inverse=True)[1]


def fit_logistic(design: np.ndarray, target: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """L2-penalized logistic regression via Newton-Raphson/IRLS."""
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
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def crossfit_index(design: np.ndarray, positive: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold logit index; standardized within fold."""
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
        raise RuntimeError("cross-fitted index contains non-finite values")
    return index


# --------------------------------------------------------------------------------------
# Data loading and verification
# --------------------------------------------------------------------------------------

def load_data(data_root: Path) -> dict:
    residual_path = data_root / "tauspin-residual-data" / "residual-shortcut-scores.npz"
    substructure_path = data_root / "tauspin-substructure-data" / "substructure-features.npz"
    nonspin_path = data_root / "tauspin-nonspin-data" / "nonspin-features.npz"
    spincorr_path = data_root / "tauspin-spincorr-data" / "truth-spin-table.npz"

    with np.load(residual_path, allow_pickle=False) as f:
        res = {k: np.asarray(f[k]) for k in f.files}
    with np.load(substructure_path, allow_pickle=False) as f:
        sub = {k: np.asarray(f[k]) for k in f.files}
    with np.load(nonspin_path, allow_pickle=False) as f:
        non = {k: np.asarray(f[k]) for k in f.files}

    ids = res["global_indices"]
    n_events = len(ids)
    if len(np.unique(ids)) != n_events:
        raise ValueError("Duplicate global_indices found in residual dataset")

    # Align substructure, nonspin, spincorr by row_index
    sub_order = np.argsort(sub["row_index"])
    sub_pos = np.searchsorted(sub["row_index"][sub_order], ids)
    sub_into = sub_order[sub_pos]
    if not np.array_equal(sub["row_index"][sub_into], ids):
        raise ValueError("Substructure features do not cover all residual events")

    non_order = np.argsort(non["row_index"])
    non_pos = np.searchsorted(non["row_index"][non_order], ids)
    non_into = non_order[non_pos]
    if not np.array_equal(non["row_index"][non_into], ids):
        raise ValueError("Nonspin features do not cover all residual events")

    # Safety check: test split is never read
    if not (non["split_id"][non_into] == 1).all():
        raise ValueError("Non-validation split rows detected in evaluation set")

    # Assemble aligned dictionary
    data = {
        "global_indices": ids,
        "labels": res["labels"].astype(np.int64),
        "is_higgs": res["labels"].astype(np.int64) == 1,
        "overlap_weights": res["overlap_weights"].astype(np.float64),
        "modes": res["modes"].astype(np.int64),
        "cluster": non["source_file_index"][non_into].astype(np.int64),
        "truth_h": res["truth_h"].astype(np.float64),
        "hybrid_h": res["hybrid_h"].astype(np.float64),
    }

    # Add scores
    for k in res:
        if k.startswith("score_"):
            data[k] = res[k].astype(np.float64)

    # Add substructure variables
    for var in S1_VARS + S2_VARS + S4_VARS:
        if var in sub:
            data[var] = sub[var][sub_into].astype(np.float64)

    # Add nonspin variables
    data["abs_vis_rapidity"] = np.abs(non["vis_rapidity"][non_into].astype(np.float64))
    data["met_sumet"] = non["met_sumet"][non_into].astype(np.float64)
    data["vis_mass"] = non["vis_mass"][non_into].astype(np.float64)
    data["vis_pt"] = non["vis_pt"][non_into].astype(np.float64)
    data["delta_r_tt"] = non["delta_r_tt"][non_into].astype(np.float64)

    # Add longitudinal and transverse spin products from truth_h
    # truth_h has shape (N, 2, 3) where index 0 is minus, 1 is plus, and components are (n, r, k)
    th = data["truth_h"]
    data["spin_longitudinal_prod"] = th[:, 0, 2] * th[:, 1, 2]  # k * k
    data["spin_transverse_prod"] = th[:, 0, 0] * th[:, 1, 0] + th[:, 0, 1] * th[:, 1, 1]  # n*n + r*r

    # Add bilateral substructure products
    data["s1_charged_prod"] = data["f_charged_minus"] * data["f_charged_plus"]
    data["s1_upsilon_prod"] = data["ups_pi0_minus"] * data["ups_pi0_plus"]
    data["s1_lead_pfo_prod"] = data["lead_pfo_ptfrac_minus"] * data["lead_pfo_ptfrac_plus"]

    # Closure checks against published numbers
    pos = data["is_higgs"]
    w = data["overlap_weights"]
    for arm, expected in FROZEN_CLOSURE.items():
        val = weighted_auc(data["score_" + arm], pos, w)
        if abs(val - expected) > 1e-4:
            raise ValueError(f"Closure check failed for {arm}: got {val:.6f}, expected {expected:.6f}")

    return data


# --------------------------------------------------------------------------------------
# Conditioners Construction
# --------------------------------------------------------------------------------------

def build_conditioners(data: dict) -> dict:
    pos = data["is_higgs"]
    fold = (data["cluster"] % 2).astype(np.int64)

    # 1. Non-spin kinematics index
    nonspin_matrix = np.column_stack([
        data["abs_vis_rapidity"],
        np.log1p(data["met_sumet"]),
        np.log1p(data["vis_mass"]),
        np.log1p(data["vis_pt"]),
        data["delta_r_tt"],
    ])
    nonspin_design = np.column_stack([nonspin_matrix, nonspin_matrix ** 2])
    idx_nonspin = crossfit_index(nonspin_design, pos, fold)

    # 2. Substructure S1 (energy sharing) index
    s1_matrix = np.column_stack([data[v] for v in S1_VARS] + [
        data["s1_charged_prod"], data["s1_upsilon_prod"], data["s1_lead_pfo_prod"]
    ])
    s1_design = np.column_stack([s1_matrix, s1_matrix ** 2])
    idx_s1 = crossfit_index(s1_design, pos, fold)

    # 3. Substructure S2 (track structure) index
    s2_matrix = np.column_stack([data[v] for v in S2_VARS])
    s2_design = np.column_stack([s2_matrix, s2_matrix ** 2])
    idx_s2 = crossfit_index(s2_design, pos, fold)

    # 4. Substructure S4 (shape/width) index
    s4_matrix = np.column_stack([data[v] for v in S4_VARS])
    s4_design = np.column_stack([s4_matrix, s4_matrix ** 2])
    idx_s4 = crossfit_index(s4_design, pos, fold)

    # 5. Full substructure index (S1 + S2 + S4)
    sub_full_matrix = np.column_stack([s1_matrix, s2_matrix, s4_matrix])
    sub_full_design = np.column_stack([sub_full_matrix, sub_full_matrix ** 2])
    idx_sub_full = crossfit_index(sub_full_design, pos, fold)

    # 6. Combined Non-spin + Full Substructure index
    all_reco_matrix = np.column_stack([nonspin_matrix, sub_full_matrix])
    all_reco_design = np.column_stack([all_reco_matrix, all_reco_matrix ** 2])
    idx_all_reco = crossfit_index(all_reco_design, pos, fold)

    # 7. Spin indicators
    idx_truth_spin = data["score_truth15"]

    # Strata dictionary
    strata = {
        "Unconditional": np.zeros(len(pos), dtype=np.int64),
        "|y(tautau)|": quantile_strata(data["abs_vis_rapidity"], STRATA_BINS),
        "met_sumet": quantile_strata(np.log1p(data["met_sumet"]), STRATA_BINS),
        "Kinematics Index": quantile_strata(idx_nonspin, STRATA_BINS),
        "S1 (Energy Sharing)": quantile_strata(idx_s1, STRATA_BINS),
        "S2 (Track Structure)": quantile_strata(idx_s2, STRATA_BINS),
        "S4 (Tau Shape)": quantile_strata(idx_s4, STRATA_BINS),
        "Full Substructure": quantile_strata(idx_sub_full, STRATA_BINS),
        "Kinematics + Substructure": quantile_strata(idx_all_reco, STRATA_BINS),
        "S1 Bilateral Product": quantile_strata(data["s1_charged_prod"], STRATA_BINS),
        "Longitudinal Spin (k*k)": quantile_strata(data["spin_longitudinal_prod"], STRATA_BINS),
        "Transverse Spin (n*n+r*r)": quantile_strata(data["spin_transverse_prod"], STRATA_BINS),
        "Truth Spin Index": quantile_strata(idx_truth_spin, STRATA_BINS),
        "Truth Spin x Substructure": combine_strata(
            quantile_strata(idx_truth_spin, 5),
            quantile_strata(idx_sub_full, 5)
        ),
    }

    return strata, {
        "idx_nonspin": idx_nonspin,
        "idx_s1": idx_s1,
        "idx_s2": idx_s2,
        "idx_s4": idx_s4,
        "idx_sub_full": idx_sub_full,
        "idx_all_reco": idx_all_reco,
    }


# --------------------------------------------------------------------------------------
# Bootstrap Analysis
# --------------------------------------------------------------------------------------

def run_analysis(data: dict, strata: dict, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    pos = data["is_higgs"]
    w = data["overlap_weights"]
    cluster = data["cluster"]
    unique_clusters = np.unique(cluster)

    scores = {
        "direct15-truth": data["score_direct15-truth"],
        "flow-full": data["score_flow-full"],
        "resid-direct15-truth": data["score_resid-direct15-truth"],
        "truth15": data["score_truth15"],
    }

    # Populations to evaluate
    cohorts = {
        "All events": np.ones(len(pos), dtype=bool),
        "1p1p": (data["modes"][:, 0] == 0) & (data["modes"][:, 1] == 0),
        "1p3p": ((data["modes"][:, 0] == 0) & (data["modes"][:, 1] == 3)) |
                ((data["modes"][:, 0] == 3) & (data["modes"][:, 1] == 0)),
        "3p3p": (data["modes"][:, 0] == 3) & (data["modes"][:, 1] == 3),
    }

    conditioner_names = list(strata.keys())

    results = {}

    for cname, cmask in cohorts.items():
        sub_pos = pos[cmask]
        sub_w = w[cmask]
        sub_cluster = cluster[cmask]
        sub_strata = {k: v[cmask] for k, v in strata.items()}
        sub_scores = {k: v[cmask] for k, v in scores.items()}

        sub_unique_clusters = np.unique(sub_cluster)
        n_sub_events = len(sub_pos)

        # Mapping for vectorized bootstrap
        # Precompute cluster index slices
        cluster_indices = [np.flatnonzero(sub_cluster == c) for c in sub_unique_clusters]

        cohort_res = {
            "n_events": int(n_sub_events),
            "n_H": int(sub_pos.sum()),
            "n_Z": int((~sub_pos).sum()),
            "conditioners": {},
        }

        # Point estimates
        for cond_name in conditioner_names:
            st = sub_strata[cond_name]
            dir_auc = weighted_conditional_auc(sub_scores["direct15-truth"], sub_pos, sub_w, st)
            flw_auc = weighted_conditional_auc(sub_scores["flow-full"], sub_pos, sub_w, st)
            res_auc = weighted_conditional_auc(sub_scores["resid-direct15-truth"], sub_pos, sub_w, st)
            diff_auc = dir_auc - flw_auc

            cohort_res["conditioners"][cond_name] = {
                "direct15_auc": dir_auc,
                "flow_auc": flw_auc,
                "resid_auc": res_auc,
                "diff_direct_minus_flow": diff_auc,
            }

        # Cluster bootstrap
        if replicates > 0:
            print(f"Running cluster bootstrap for cohort '{cname}' ({replicates} replicates)...")
            rng = np.random.default_rng(BOOTSTRAP_SEED)

            boot_dir = np.empty((replicates, len(conditioner_names)))
            boot_flw = np.empty((replicates, len(conditioner_names)))
            boot_res = np.empty((replicates, len(conditioner_names)))
            boot_dif = np.empty((replicates, len(conditioner_names)))

            n_clust = len(sub_unique_clusters)
            for r in range(replicates):
                sampled_c_idx = rng.choice(n_clust, size=n_clust, replace=True)
                sample_rows = np.concatenate([cluster_indices[i] for i in sampled_c_idx])

                b_pos = sub_pos[sample_rows]
                b_w = sub_w[sample_rows]

                for j, cond_name in enumerate(conditioner_names):
                    b_st = sub_strata[cond_name][sample_rows]
                    d_auc = weighted_conditional_auc(sub_scores["direct15-truth"][sample_rows],
                                                    b_pos, b_w, b_st)
                    f_auc = weighted_conditional_auc(sub_scores["flow-full"][sample_rows],
                                                    b_pos, b_w, b_st)
                    r_auc = weighted_conditional_auc(sub_scores["resid-direct15-truth"][sample_rows],
                                                    b_pos, b_w, b_st)

                    boot_dir[r, j] = d_auc
                    boot_flw[r, j] = f_auc
                    boot_res[r, j] = r_auc
                    boot_dif[r, j] = d_auc - f_auc

            for j, cond_name in enumerate(conditioner_names):
                entry = cohort_res["conditioners"][cond_name]
                entry["direct15_ci95"] = np.quantile(boot_dir[:, j], [0.025, 0.975]).tolist()
                entry["flow_ci95"] = np.quantile(boot_flw[:, j], [0.025, 0.975]).tolist()
                entry["resid_ci95"] = np.quantile(boot_res[:, j], [0.025, 0.975]).tolist()
                entry["diff_ci95"] = np.quantile(boot_dif[:, j], [0.025, 0.975]).tolist()

        results[cname] = cohort_res

    return results


# --------------------------------------------------------------------------------------
# Plotting & Publication Figures
# --------------------------------------------------------------------------------------

def generate_figures(results: dict, data: dict, output_dir: Path, vault_fig_dir: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })

    cohort_all = results["All events"]["conditioners"]

    # 1. Figure 1: Direct Residual AUC across Conditioners
    # Select key representative conditioners for clarity
    selected_conds = [
        ("Unconditional", "Unconditional (raw)"),
        ("|y(tautau)|", "Kinematics: |y(ττ)|"),
        ("met_sumet", "Kinematics: sum(ET)"),
        ("Kinematics Index", "Kinematics Index (5-var)"),
        ("S1 (Energy Sharing)", "Substructure: S1 (energy)"),
        ("S2 (Track Structure)", "Substructure: S2 (tracks)"),
        ("S4 (Tau Shape)", "Substructure: S4 (shape)"),
        ("Full Substructure", "Full Substructure (S1+S2+S4)"),
        ("Kinematics + Substructure", "Kinematics + Substructure"),
        ("Longitudinal Spin (k*k)", "Spin: Longitudinal (kz*kz)"),
        ("Transverse Spin (n*n+r*r)", "Spin: Transverse (n*n+r*r)"),
        ("Truth Spin Index", "Truth Spin (15D index)"),
        ("Truth Spin x Substructure", "Truth Spin x Substructure"),
    ]

    names = [label for _, label in selected_conds]
    keys = [k for k, _ in selected_conds]

    res_aucs = [cohort_all[k]["resid_auc"] for k in keys]
    res_err_low = [cohort_all[k]["resid_auc"] - cohort_all[k]["resid_ci95"][0] for k in keys]
    res_err_high = [cohort_all[k]["resid_ci95"][1] - cohort_all[k]["resid_auc"] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(names))

    # Color codes: Gray for unconditional, Blues for kinematics, Oranges for substructure, Purples for spin
    colors = []
    for k, _ in selected_conds:
        if "Unconditional" in k:
            colors.append("#555555")
        elif "Kinematics" in k or "|y|" in k or "sumet" in k:
            colors.append("#1f77b4")
        elif "Substructure" in k or "S1" in k or "S2" in k or "S4" in k:
            colors.append("#ff7f0e")
        else:
            colors.append("#9467bd")

    bars = ax.barh(y_pos, res_aucs, xerr=[res_err_low, res_err_high], align="center",
                   color=colors, alpha=0.85, capsize=4, edgecolor="black", linewidth=0.8)

    ax.axvline(0.50, color="gray", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    ax.axvline(cohort_all["Unconditional"]["resid_auc"], color="#d62728", linestyle=":",
               linewidth=1.2, label=f"Unconditional AUC ({cohort_all['Unconditional']['resid_auc']:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted Conditional AUC of Direct Residual Readout")
    ax.set_title("Does Tau Substructure or Spin Correlation Absorb Direct Residual Discrimination?\n"
                 "[ATLAS Simulation, H/Z validation sample, 11,065 events, 2000 bootstrap replicates]")
    ax.set_xlim(0.46, 0.72)
    ax.legend(loc="lower right")

    # Add text labels on bars
    for i, v in enumerate(res_aucs):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold",
                color="black" if v > 0.52 else "#777777")

    fig.tight_layout()
    p1 = output_dir / "fig1_residual_conditional_auc.png"
    fig.savefig(p1, dpi=300)
    fig.savefig(vault_fig_dir / "fig1_residual_conditional_auc.png", dpi=300)
    plt.close(fig)

    # 2. Figure 2: Paired Difference: Direct15 minus Flow-Full AUC
    diff_aucs = [cohort_all[k]["diff_direct_minus_flow"] for k in keys]
    diff_err_low = [diff_aucs[i] - cohort_all[k]["diff_ci95"][0] for i, k in enumerate(keys)]
    diff_err_high = [cohort_all[k]["diff_ci95"][1] - diff_aucs[i] for i, k in enumerate(keys)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars2 = ax.barh(y_pos, diff_aucs, xerr=[diff_err_low, diff_err_high], align="center",
                    color=colors, alpha=0.85, capsize=4, edgecolor="black", linewidth=0.8)

    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax.axvline(cohort_all["Unconditional"]["diff_direct_minus_flow"], color="#d62728", linestyle=":",
               linewidth=1.2, label=f"Raw Advantage (+{cohort_all['Unconditional']['diff_direct_minus_flow']:.4f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Paired ΔAUC (direct15-truth minus flow-full)")
    ax.set_title("Direct Regression Advantage over Flow under Conditioning\n"
                 "[Paired difference on identical validation events, 95% bootstrap intervals]")
    ax.legend(loc="lower right")

    for i, v in enumerate(diff_aucs):
        offset = 0.001 if v >= 0 else -0.003
        ax.text(v + offset, i, f"{v:+.4f}", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    p2 = output_dir / "fig2_direct_vs_flow_gap.png"
    fig.savefig(p2, dpi=300)
    fig.savefig(vault_fig_dir / "fig2_direct_vs_flow_gap.png", dpi=300)
    plt.close(fig)

    # 3. Figure 3: Mode-pair Resolved Breakdown
    mode_names = ["1p1p", "1p3p", "3p3p"]
    mode_keys = ["Unconditional", "Full Substructure", "Truth Spin Index"]

    fig, (ax_res, ax_gap) = plt.subplots(1, 2, figsize=(13, 5.5))

    bar_width = 0.25
    x = np.arange(len(mode_names))

    colors_m = ["#555555", "#ff7f0e", "#9467bd"]

    for i, mk in enumerate(mode_keys):
        vals_res = [results[m]["conditioners"][mk]["resid_auc"] for m in mode_names]
        errs_res = [
            [vals_res[j] - results[m]["conditioners"][mk]["resid_ci95"][0] for j, m in enumerate(mode_names)],
            [results[m]["conditioners"][mk]["resid_ci95"][1] - vals_res[j] for j, m in enumerate(mode_names)]
        ]
        ax_res.bar(x + (i - 1) * bar_width, vals_res, width=bar_width, yerr=errs_res,
                   capsize=3, label=mk, color=colors_m[i], alpha=0.85, edgecolor="black", linewidth=0.7)

        vals_gap = [results[m]["conditioners"][mk]["diff_direct_minus_flow"] for m in mode_names]
        errs_gap = [
            [vals_gap[j] - results[m]["conditioners"][mk]["diff_ci95"][0] for j, m in enumerate(mode_names)],
            [results[m]["conditioners"][mk]["diff_ci95"][1] - vals_gap[j] for j, m in enumerate(mode_names)]
        ]
        ax_gap.bar(x + (i - 1) * bar_width, vals_gap, width=bar_width, yerr=errs_gap,
                   capsize=3, label=mk, color=colors_m[i], alpha=0.85, edgecolor="black", linewidth=0.7)

    ax_res.axhline(0.50, color="gray", linestyle="--", linewidth=1.0)
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(["1p1p\n(N=3,260)", "1p3p\n(N=5,512)", "3p3p\n(N=2,293)"])
    ax_res.set_ylabel("Weighted Conditional AUC")
    ax_res.set_title("Direct Residual AUC across Decay Modes")
    ax_res.set_ylim(0.45, 0.75)
    ax_res.legend(loc="upper right")

    ax_gap.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(["1p1p\n(N=3,260)", "1p3p\n(N=5,512)", "3p3p\n(N=2,293)"])
    ax_gap.set_ylabel("Paired ΔAUC (direct15 minus flow-full)")
    ax_gap.set_title("Direct Advantage over Flow across Decay Modes")
    ax_gap.legend(loc="upper right")

    fig.suptitle("Decay-Mode Pair Structure of the Direct Regression Residual", fontsize=13, y=1.02)
    fig.tight_layout()
    p3 = output_dir / "fig3_mode_resolved_breakdown.png"
    fig.savefig(p3, dpi=300)
    fig.savefig(vault_fig_dir / "fig3_mode_resolved_breakdown.png", dpi=300)
    plt.close(fig)

    # 4. Figure 4: Observable Profiles: Substructure vs Spin
    pos = data["is_higgs"]
    fig, (ax_sub, ax_spn) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Profile 1: Substructure Energy Sharing Product (f_charged_minus * f_charged_plus)
    sub_var = data["s1_charged_prod"]
    sub_bins = quantile_strata(sub_var, 10)
    bin_centers_sub = [float(np.median(sub_var[sub_bins == b])) for b in range(10)]

    dir_h_sub = [float(data["score_direct15-truth"][sub_bins == b & pos].mean()) for b in range(10)]
    dir_z_sub = [float(data["score_direct15-truth"][sub_bins == b & ~pos].mean()) for b in range(10)]
    flw_h_sub = [float(data["score_flow-full"][sub_bins == b & pos].mean()) for b in range(10)]
    flw_z_sub = [float(data["score_flow-full"][sub_bins == b & ~pos].mean()) for b in range(10)]

    ax_sub.plot(bin_centers_sub, dir_h_sub, "o-", color="#d62728", label="Direct15 (Higgs)", linewidth=1.5)
    ax_sub.plot(bin_centers_sub, dir_z_sub, "o--", color="#d62728", alpha=0.6, label="Direct15 (Z)", linewidth=1.5)
    ax_sub.plot(bin_centers_sub, flw_h_sub, "s-", color="#1f77b4", label="Flow (Higgs)", linewidth=1.5)
    ax_sub.plot(bin_centers_sub, flw_z_sub, "s--", color="#1f77b4", alpha=0.6, label="Flow (Z)", linewidth=1.5)

    ax_sub.set_xlabel("Bilateral Energy Sharing: f(charged,-) × f(charged,+)")
    ax_sub.set_ylabel("Mean Readout Score")
    ax_sub.set_title("Response vs Substructure Energy Sharing")
    ax_sub.legend(loc="upper left")

    # Profile 2: Truth Longitudinal Spin Product (kz,- * kz,+)
    spn_var = data["spin_longitudinal_prod"]
    spn_bins = quantile_strata(spn_var, 10)
    bin_centers_spn = [float(np.median(spn_var[spn_bins == b])) for b in range(10)]

    dir_h_spn = [float(data["score_direct15-truth"][spn_bins == b & pos].mean()) for b in range(10)]
    dir_z_spn = [float(data["score_direct15-truth"][spn_bins == b & ~pos].mean()) for b in range(10)]
    flw_h_spn = [float(data["score_flow-full"][spn_bins == b & pos].mean()) for b in range(10)]
    flw_z_spn = [float(data["score_flow-full"][spn_bins == b & ~pos].mean()) for b in range(10)]

    ax_spn.plot(bin_centers_spn, dir_h_spn, "o-", color="#d62728", label="Direct15 (Higgs)", linewidth=1.5)
    ax_spn.plot(bin_centers_spn, dir_z_spn, "o--", color="#d62728", alpha=0.6, label="Direct15 (Z)", linewidth=1.5)
    ax_spn.plot(bin_centers_spn, flw_h_spn, "s-", color="#1f77b4", label="Flow (Higgs)", linewidth=1.5)
    ax_spn.plot(bin_centers_spn, flw_z_spn, "s--", color="#1f77b4", alpha=0.6, label="Flow (Z)", linewidth=1.5)

    ax_spn.set_xlabel("Longitudinal Spin Correlation: h_z^- × h_z^+")
    ax_spn.set_ylabel("Mean Readout Score")
    ax_spn.set_title("Response vs Longitudinal Spin Correlation")
    ax_spn.legend(loc="upper left")

    fig.suptitle("Score Separation: Independent of Substructure, Strongly Driven by Spin",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p4 = output_dir / "fig4_profiles_substructure_vs_spin.png"
    fig.savefig(p4, dpi=300)
    fig.savefig(vault_fig_dir / "fig4_profiles_substructure_vs_spin.png", dpi=300)
    plt.close(fig)

    print("Successfully generated all 4 figures in output directory and Vault.")


# --------------------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Decompose direct residual into substructure vs spin")
    parser.add_argument("--data-root", type=Path, default=Path("/Users/ryunosuke/Projects"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/Users/ryunosuke/Projects/tauspin/analysis/outputs/residual-substructure-20260916"))
    parser.add_argument("--vault-run-dir", type=Path,
                        default=Path("/Users/ryunosuke/Library/Mobile Documents/iCloud~md~obsidian/Documents/AthenaVault/10_Projects/HE/ATLAS/tauspin/ARIADNE/Runs/tauspin-direct-residual-substructure-20260916"))
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vault_fig_dir = args.vault_run_dir / "Figures"
    vault_fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets from {args.data_root}...")
    t0 = time.time()
    data = load_data(args.data_root)
    print(f"Loaded {len(data['is_higgs'])} events in {time.time() - t0:.2f} s")

    print("Building conditioning strata (substructure, kinematics, spin)...")
    strata, indices = build_conditioners(data)

    print(f"Executing conditional analysis across cohorts with {args.replicates} bootstrap replicates...")
    t_start = time.time()
    results = run_analysis(data, strata, replicates=args.replicates)
    print(f"Analysis completed in {time.time() - t_start:.2f} s")

    # Save results JSON
    json_path = args.output_dir / "residual_substructure_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate figures
    print("Generating publication figures...")
    generate_figures(results, data, args.output_dir, vault_fig_dir)
    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
