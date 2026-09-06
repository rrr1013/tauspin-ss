#!/usr/bin/env python3
"""Is the frozen H/Z classifier's discrimination inside the taus?

The 2026-09-06 non-spin run held every event-level and per-tau *scalar* summary
fixed and still left 72% of the frozen score's excess AUC unexplained.  The
transformer, however, reads individual core tracks and neutral PFOs.  This run
gives those constituents named summaries and asks how much of the gap they close.

Estimand and estimator are the ones of ``nonspin_information.py``: a conditional
AUC inside quantile strata of a cross-fitted logit index, with a cluster
bootstrap over source ROOT files.  Nothing is trained, re-selected or
re-calibrated, and the test partition is never touched.

Blocks (fixed before any block AUC was computed; see the run note):

``S1`` charged / neutral energy sharing   -- the physical carrier of tau analysing power
``S2`` core-track structure               -- prong sharing, spread, impact parameters
``S3`` neutral (PFO) structure            -- multiplicity and spread
``S4`` overall tau shape                  -- width, constituent mass, momentum closure
``S5`` detector-only track quality        -- hits, selector, isolation tracks; cannot carry spin
``E-reco`` the non-spin run's ``P-reco + D-reco`` variables, as the baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nonspin_information import (  # noqa: E402
    ARMS,
    FROZEN_ENSEMBLE_AUC,
    P_RECO,
    D_RECO,
    auc,
    combine_strata,
    conditional_auc,
    fit_logistic,
    quantile_strata,
)

# --------------------------------------------------------------------------------------
# Constants fixed before the results were looked at.
# --------------------------------------------------------------------------------------

STRATA_BINS = 10
STRATA_BINS_STABILITY = (5, 20)
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 1000
RIDGE = 1.0
WINSOR = (0.001, 0.999)      # clip bounds, taken on the training fold only
BOOTSTRAP_SEED = 20260906
PERMUTATION_SEED = 906203

# Per-side substructure variables.  "log" means log1p before standardising.
S1 = (
    ("f_charged", "lin"),
    ("f_pi0", "lin"),
    ("ups_pi0", "lin"),
    ("lead_pfo_ptfrac", "lin"),
    ("sum_pfo_ptfrac", "lin"),
)
S2 = (
    ("lead_trk_ptfrac", "lin"),
    ("subl_trk_ptfrac", "lin"),
    ("trk_pt_asym", "lin"),
    ("trk_width", "lin"),
    ("trk_dr_max", "lin"),
    ("sum_trk_ptfrac", "lin"),
    ("max_abs_d0", "log"),
    ("max_abs_z0sin", "log"),
)
S3 = (
    ("n_pfo", "lin"),
    ("n_pi0", "lin"),
    ("pfo_width", "lin"),
)
S4 = (
    ("all_width", "lin"),
    ("m_const", "log"),
    ("const_over_taupt", "lin"),
)
S5 = (
    ("frac_passsel", "lin"),
    ("mean_pix", "lin"),
    ("mean_sct", "lin"),
    ("mean_trt", "lin"),
    ("n_iso", "lin"),
)
SIDED_BLOCKS = {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "S5": S5}
SUBSTRUCTURE_ORDER = ("S1", "S2", "S3", "S4", "S5")
BLOCK_TITLE = {
    "S1": "S1 energy sharing",
    "S2": "S2 track structure",
    "S3": "S3 neutral structure",
    "S4": "S4 tau shape",
    "S5": "S5 detector only",
}
E_RECO = P_RECO + D_RECO


def sum_per_side() -> int:
    """Number of substructure variables defined for one tau."""
    return sum(len(SIDED_BLOCKS[name]) for name in SUBSTRUCTURE_ORDER)


def sided(block: tuple, sides=("minus", "plus")) -> tuple:
    """Expand per-side variable names into concrete table columns."""
    return tuple((f"{name}_{side}", transform) for name, transform in block for side in sides)


# --------------------------------------------------------------------------------------
# Estimator: same shape as the non-spin run, plus a training-fold winsorisation.
# --------------------------------------------------------------------------------------


def transform_columns(table: dict[str, np.ndarray], block: tuple, keep: np.ndarray) -> np.ndarray:
    columns = []
    for name, transform in block:
        values = table[name][keep].astype(np.float64)
        if transform == "log":
            values = np.log1p(np.maximum(values, 0.0))
        columns.append(values)
    return np.column_stack(columns)


def crossfit_index(raw: np.ndarray, positive: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold logit index of the winsorised, standardised variables and their squares.

    Clip bounds, mean and scale come from the training fold only, so the held-out
    index never sees its own fold.  Winsorisation is needed because one known
    upstream event carries ``track_pt = 7.7e10 GeV``.
    """
    index = np.full(raw.shape[0], np.nan)
    target = positive.astype(np.float64)
    for held in np.unique(fold):
        train = fold != held
        low = np.quantile(raw[train], WINSOR[0], axis=0)
        high = np.quantile(raw[train], WINSOR[1], axis=0)
        clipped_train = np.clip(raw[train], low, high)
        mean = clipped_train.mean(axis=0)
        scale = clipped_train.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design_train = (clipped_train - mean) / scale
        beta = fit_logistic(np.column_stack([design_train, design_train ** 2]), target[train], RIDGE)
        design_test = (np.clip(raw[~train], low, high) - mean) / scale
        design_test = np.column_stack([design_test, design_test ** 2])
        index[~train] = beta[0] + design_test @ beta[1:]
    if not np.isfinite(index).all():
        raise RuntimeError("cross-fitted index has non-finite entries")
    return index


def fold_split_auc(raw, positive, fold, index) -> list[float]:
    """Index AUC evaluated separately in each held-out fold (the stability check)."""
    return [auc(index[fold == held], positive[fold == held]) for held in np.unique(fold)]


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
    with np.load(args.event_features, allow_pickle=False) as source:
        table = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.substructure, allow_pickle=False) as source:
        sub = {key: np.asarray(source[key]) for key in source.files}

    # The two extractions walked the same ntuple with the same identity rule.
    for key in ("row_index", "sample_id", "source_file_index"):
        if not np.array_equal(table[key], sub[key]):
            raise RuntimeError(f"substructure and event tables disagree on {key}")
    for name, values in sub.items():
        if name not in table:
            table[name] = values

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

    if not np.array_equal(
        ens["labels"][into].astype(np.int64), 1 - table["sample_id"].astype(np.int64)
    ):
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

    table["abs_vis_rapidity"] = np.abs(table["vis_rapidity"])
    table["abs_delta_phi_tt"] = np.abs(table["delta_phi_tt"])
    table["abs_delta_eta_tt"] = np.abs(table["delta_eta_tt"])
    table["abs_tau_eta_minus"] = np.abs(table["tau_eta_minus"])
    table["abs_tau_eta_plus"] = np.abs(table["tau_eta_plus"])
    table["abs_dphi_met_minus"] = np.abs(table["dphi_met_minus"])
    table["abs_dphi_met_plus"] = np.abs(table["dphi_met_plus"])
    return {"row": row, "table": table, "closure": dict(FROZEN_ENSEMBLE_AUC)}


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def build_blocks() -> dict[str, tuple]:
    blocks: dict[str, tuple] = {"E-reco": E_RECO}
    for name in SUBSTRUCTURE_ORDER:
        blocks[name] = sided(SIDED_BLOCKS[name])
    every = tuple(item for name in SUBSTRUCTURE_ORDER for item in SIDED_BLOCKS[name])
    blocks["S-all"] = sided(every)
    blocks["S-all (tau- only)"] = sided(every, sides=("minus",))
    blocks["S-all (tau+ only)"] = sided(every, sides=("plus",))
    blocks["E-reco + S-all"] = E_RECO + sided(every)
    return blocks


def single_variable_auc(table: dict[str, np.ndarray], keep: np.ndarray) -> list[dict]:
    positive = table["is_higgs"][keep]
    rows = []
    for block in SUBSTRUCTURE_ORDER:
        for name, _ in SIDED_BLOCKS[block]:
            for side in ("minus", "plus"):
                column = f"{name}_{side}"
                value = auc(table[column][keep].astype(np.float64), positive)
                rows.append({"variable": column, "block": block, "auc": value,
                             "unsigned": 0.5 + abs(value - 0.5)})
    rows.sort(key=lambda r: -r["unsigned"])
    return rows


def analyse(table: dict[str, np.ndarray]) -> tuple[dict, dict]:
    keep = table["report_mask"]
    positive = table["is_higgs"][keep]
    cluster = table["source_file_index"][keep]
    fold = (cluster % 2).astype(np.int64)
    scores = {arm: table[arm][keep].astype(np.float64) for arm in ARMS}

    blocks = build_blocks()
    raw = {name: transform_columns(table, block, keep) for name, block in blocks.items()}
    indices = {name: crossfit_index(values, positive, fold) for name, values in raw.items()}
    strata = {name: quantile_strata(values, STRATA_BINS) for name, values in indices.items()}
    # Holding both conditioners fixed at once, rather than merging them into one logit.
    strata["E-reco x S-all"] = combine_strata(strata["E-reco"], strata["S-all"])
    strata["tau- x tau+"] = combine_strata(
        strata["S-all (tau- only)"], strata["S-all (tau+ only)"]
    )

    result: dict = {
        "counts": {"report": int(keep.sum()), "H": int(positive.sum()),
                   "Z": int((~positive).sum()),
                   "variables_per_side": sum(len(SIDED_BLOCKS[n]) for n in SUBSTRUCTURE_ORDER)},
        "single_variable_auc": single_variable_auc(table, keep),
        "block_index_auc": {name: auc(values, positive) for name, values in indices.items()},
        "block_index_auc_by_fold": {
            name: fold_split_auc(raw[name], positive, fold, values)
            for name, values in indices.items()
        },
        "block_sizes": {name: len(block) for name, block in blocks.items()},
        "n_strata": {name: int(values.max()) + 1 for name, values in strata.items()},
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

    for bins in STRATA_BINS_STABILITY:
        alt = {name: quantile_strata(indices[name], bins)
               for name in ("E-reco", "S-all", "S1", "S5")}
        result["conditional_auc_stability"][str(bins)] = {
            arm: {name: conditional_auc(scores[arm], positive, values)
                  for name, values in alt.items()}
            for arm in ARMS
        }

    rng = np.random.default_rng(PERMUTATION_SEED)
    for name in ("E-reco", "S-all", "E-reco x S-all"):
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

    names: list[tuple[str, str]] = []
    for arm in ARMS:
        names.append((arm, "unconditional"))
        for name in strata:
            names.append((arm, name))
    for name in indices:
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
                out[k] = conditional_auc(scores[arm][idx], pos, strata[name][idx])
        return out

    unique, inverse = np.unique(cluster, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_REPLICATES, len(names)))
    for r in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(0, unique.size, unique.size)
        idx = np.concatenate([members[i] for i in picked])
        draws[r] = statistics(idx)
    result["bootstrap_clusters"] = int(unique.size)
    result["bootstrap"] = {
        f"{arm}|{name}": {
            "q025": float(np.nanquantile(draws[:, k], 0.025)),
            "q975": float(np.nanquantile(draws[:, k], 0.975)),
            "std": float(np.nanstd(draws[:, k])),
        }
        for k, (arm, name) in enumerate(names)
    }
    base_column = {arm: k for k, (arm, name) in enumerate(names) if name == "unconditional"}
    result["bootstrap_drop"] = {}
    for k, (arm, name) in enumerate(names):
        if arm == "__index__" or name == "unconditional":
            continue
        diff = draws[:, base_column[arm]] - draws[:, k]
        result["bootstrap_drop"][f"{arm}|{name}"] = {
            "point": float(result["unconditional_auc"][arm]
                           - result["conditional_auc"][arm][name]),
            "q025": float(np.nanquantile(diff, 0.025)),
            "q975": float(np.nanquantile(diff, 0.975)),
            "std": float(np.nanstd(diff)),
        }
    return result, {"indices": indices, "strata": strata, "scores": scores,
                    "positive": positive, "keep": keep}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

BLOCK_COLOUR = {
    "S1": "#1f77b4", "S2": "#d62728", "S3": "#2ca02c", "S4": "#9467bd", "S5": "#7f7f7f",
    "E-reco": "#ff7f0e", "S-all": "#000000",
}


def save(fig, outputs: Path, name: str) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(outputs / f"{name}.pdf", bbox_inches="tight")
    print(f"wrote {outputs / (name + '.png')}")


def figure_variable_auc(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["single_variable_auc"]
    fig, ax = plt.subplots(figsize=(7.2, 10.5))
    y = np.arange(len(rows))[::-1]
    for pos, entry in zip(y, rows):
        ax.plot([0.5, entry["auc"]], [pos, pos], color=BLOCK_COLOUR[entry["block"]],
                linewidth=1.2, alpha=0.55)
        ax.plot([entry["auc"]], [pos], marker="o", markersize=4.5,
                color=BLOCK_COLOUR[entry["block"]])
    for name in SUBSTRUCTURE_ORDER + ("E-reco", "S-all"):
        value = result["block_index_auc"][name]
        ax.axvline(value, color=BLOCK_COLOUR[name], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(0.5, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([r["variable"] for r in rows], fontsize=6.5)
    ax.set_xlabel("single-variable AUC (H positive; below 0.5 means the variable favours Z)")
    ax.set_title("Substructure variables one at a time, and the cross-fitted block indices\n"
                 f"held-out report subset, {result['counts']['report']:,} events",
                 fontsize=10)
    handles = [plt.Line2D([], [], color=BLOCK_COLOUR[n], marker="o", linestyle="-",
                          label=BLOCK_TITLE[n]) for n in SUBSTRUCTURE_ORDER]
    handles += [plt.Line2D([], [], color=BLOCK_COLOUR[n], linestyle="--", label=f"{n} index")
                for n in ("E-reco", "S-all")]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    save(fig, outputs, "substructure-variable-auc")
    plt.close(fig)


def figure_conditional_auc(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["E-reco", "S1", "S2", "S3", "S4", "S5", "S-all",
             "E-reco x S-all", "E-reco + S-all"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), width_ratios=[1.0, 1.05])
    colours = {"full_reco": "#c0392b", "tau_object_only": "#2471a3", "event_only": "#7d8b8f"}
    markers = {"full_reco": "o", "tau_object_only": "s", "event_only": "^"}

    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for arm in ARMS:
        base = result["unconditional_auc"][arm]
        ax.axvline(base, color=colours[arm], linestyle=":", linewidth=1.1, alpha=0.85)
        for pos, name in zip(y, order):
            value = result["conditional_auc"][arm][name]
            interval = result["bootstrap"][f"{arm}|{name}"]
            ax.plot([interval["q025"], interval["q975"]], [pos, pos], color=colours[arm],
                    linewidth=1.4, alpha=0.75)
            ax.plot([value], [pos], marker=markers[arm], color=colours[arm], markersize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("AUC of the frozen score inside strata (H positive)")
    ax.set_title("What survives when a block is held fixed", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([], [], color=colours[a], marker=markers[a], linestyle="-",
                          label=a.replace("_", " ")) for a in ARMS]
    handles += [plt.Line2D([], [], color="black", linestyle=":", label="unconditional")]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.95)

    ax = axes[1]
    for arm in ARMS:
        for pos, name in zip(y, order):
            drop = result["bootstrap_drop"][f"{arm}|{name}"]
            ax.plot([drop["q025"], drop["q975"]], [pos, pos], color=colours[arm],
                    linewidth=1.4, alpha=0.75)
            ax.plot([drop["point"]], [pos], marker=markers[arm], color=colours[arm],
                    markersize=5.5)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("shared with the block: unconditional AUC minus conditional AUC")
    ax.set_title("Paired drop, same bootstrap resample", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    excess = result["unconditional_auc"]["full_reco"] - 0.5
    ax.annotate(f"full-reco excess AUC over 0.5 = {excess:.4f}", xy=(0.98, 0.97),
                xycoords="axes fraction", ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
    fig.suptitle(
        "Frozen adaptive-v1 ensembles, held-out report subset "
        f"({result['counts']['report']:,} events), {STRATA_BINS} quantile strata, "
        f"cluster bootstrap over {result['bootstrap_clusters']} source ROOT files",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outputs, "substructure-conditional-auc")
    plt.close(fig)


def figure_distributions(table, figure_data, result, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    keep = figure_data["keep"]
    positive = figure_data["positive"]
    chosen = [r["variable"] for r in result["single_variable_auc"][:3]]
    panels = chosen + ["S-all index"]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6))
    for ax, name in zip(axes, panels):
        if name == "S-all index":
            values = figure_data["indices"]["S-all"]
            label = "cross-fitted S-all logit index"
        else:
            values = table[name][keep].astype(np.float64)
            label = name
        low, high = np.quantile(values, [0.001, 0.999])
        edges = np.linspace(low, high, 46)
        for mask, colour, style, tag in (
            (positive, "#c0392b", "-", "H"), (~positive, "#2471a3", "--", "Z")
        ):
            counts, _ = np.histogram(values[mask], bins=edges)
            width = np.diff(edges)
            density = counts / max(mask.sum(), 1) / width
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, density, where="mid", color=colour, linestyle=style,
                    linewidth=1.5, label=tag)
        single = result["block_index_auc"]["S-all"] if name == "S-all index" else next(
            r["auc"] for r in result["single_variable_auc"] if r["variable"] == name)
        ax.set_title(f"{label}\nAUC = {single:.5f}", fontsize=8.5)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("normalised density", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "The three most H/Z-separating substructure variables, and all of them combined  |  "
        f"report subset, H {result['counts']['H']:,} (solid red) vs Z {result['counts']['Z']:,} "
        "(dashed blue), area normalised inside the 0.1-99.9 percentile range",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, outputs, "substructure-distributions")
    plt.close(fig)


def figure_sidedness(result: dict, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    order = ["S-all (tau- only)", "S-all (tau+ only)", "tau- x tau+", "S-all"]
    labels = [r"$\tau^-$ substructure only", r"$\tau^+$ substructure only",
              r"$\tau^-$ strata $\times$ $\tau^+$ strata", "both sides in one index"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colours = {"full_reco": "#c0392b", "tau_object_only": "#2471a3", "event_only": "#7d8b8f"}
    markers = {"full_reco": "o", "tau_object_only": "s", "event_only": "^"}
    y = np.arange(len(order))[::-1]
    for arm in ARMS:
        for pos, name in zip(y, order):
            drop = result["bootstrap_drop"][f"{arm}|{name}"]
            ax.plot([drop["q025"], drop["q975"]], [pos, pos], color=colours[arm],
                    linewidth=1.5, alpha=0.75)
            ax.plot([drop["point"]], [pos], marker=markers[arm], color=colours[arm],
                    markersize=6)
    single_sum = sum(result["bootstrap_drop"][f"full_reco|{n}"]["point"]
                     for n in ("S-all (tau- only)", "S-all (tau+ only)"))
    ax.axvline(single_sum, color="#c0392b", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.annotate("sum of the two one-sided drops (full reco)", xy=(single_sum, len(order) - 0.45),
                fontsize=7.5, color="#c0392b", ha="right", va="top", rotation=0,
                xytext=(single_sum - 0.0002, len(order) - 0.45))
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("shared with the conditioner: unconditional AUC minus conditional AUC")
    ax.set_title("One tau or two: how much the frozen score shares with one-sided\n"
                 "and two-sided substructure summaries", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([], [], color=colours[a], marker=markers[a], linestyle="-",
                          label=a.replace("_", " ")) for a in ARMS]
    ax.legend(handles=handles, fontsize=8, loc="center right", framealpha=0.95)
    save(fig, outputs, "substructure-sidedness")
    plt.close(fig)


def figure_strata_profile(figure_data, result, outputs: Path) -> None:
    import matplotlib.pyplot as plt

    positive = figure_data["positive"]
    strata = figure_data["strata"]["S-all"]
    scores = figure_data["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    colours = {"full_reco": "#c0392b", "tau_object_only": "#2471a3", "event_only": "#7d8b8f"}
    markers = {"full_reco": "o", "tau_object_only": "s", "event_only": "^"}
    deciles = np.unique(strata)
    ax = axes[0]
    for arm in ARMS:
        values = [auc(scores[arm][strata == d], positive[strata == d]) for d in deciles]
        ax.plot(deciles + 1, values, marker=markers[arm], color=colours[arm],
                linewidth=1.4, label=arm.replace("_", " "))
        ax.axhline(result["unconditional_auc"][arm], color=colours[arm], linestyle=":",
                   linewidth=1.0, alpha=0.8)
    ax.axhline(0.5, color="black", linewidth=0.9)
    ax.set_xlabel("decile of the cross-fitted S-all substructure index")
    ax.set_ylabel("AUC inside the decile")
    ax.set_title("Discrimination that survives inside a substructure decile", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    n_h = np.array([int((positive & (strata == d)).sum()) for d in deciles])
    n_z = np.array([int((~positive & (strata == d)).sum()) for d in deciles])
    ax.bar(deciles + 1 - 0.2, n_h, width=0.4, color="#c0392b", label="H")
    ax.bar(deciles + 1 + 0.2, n_z, width=0.4, color="#2471a3", label="Z")
    ax.set_xlabel("decile of the cross-fitted S-all substructure index")
    ax.set_ylabel("events in the report subset")
    ax.set_title("Why the index carries information: the H/Z ratio moves across deciles",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, outputs, "substructure-strata-profile")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--row-map", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--event-features", type=Path, required=True)
    p.add_argument("--substructure", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--figures-only", type=Path, default=None,
                   help="redraw from an existing results.json instead of recomputing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_everything(args)
    table = loaded["table"]
    if args.figures_only is not None:
        result = json.loads(args.figures_only.read_text())
        keep = table["report_mask"]
        positive = table["is_higgs"][keep]
        fold = (table["source_file_index"][keep] % 2).astype(np.int64)
        blocks = build_blocks()
        indices = {name: crossfit_index(transform_columns(table, block, keep), positive, fold)
                   for name, block in blocks.items()}
        strata = {name: quantile_strata(values, STRATA_BINS) for name, values in indices.items()}
        figure_data = {"indices": indices, "strata": strata,
                       "scores": {arm: table[arm][keep].astype(np.float64) for arm in ARMS},
                       "positive": positive, "keep": keep}
    else:
        result, figure_data = analyse(table)
        args.outputs.mkdir(parents=True, exist_ok=True)
        (args.outputs / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (args.outputs / "manifest.json").write_text(json.dumps({
            "run": "tauspin-tau-substructure-20260906",
            "estimand": "AUC of the frozen score inside quantile strata of a cross-fitted "
                        "index; the drop is information shared with the block, a lower "
                        "bound and not a causal attribution",
            "frozen_closure_auc": loaded["closure"],
            "strata_bins": STRATA_BINS,
            "winsor": list(WINSOR),
            "ridge": RIDGE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_replicates": PERMUTATION_REPLICATES,
            "test_partition_read": False,
            "inputs": {
                "analysis-surface-row-map.npz": sha256_file(args.row_map),
                "terminal-ensembles.npz": sha256_file(args.ensemble),
                "nonspin-features.npz": sha256_file(args.event_features),
                "substructure-features.npz": sha256_file(args.substructure),
            },
        }, indent=2, sort_keys=True))

    figure_variable_auc(result, args.outputs)
    figure_conditional_auc(result, args.outputs)
    figure_distributions(table, figure_data, result, args.outputs)
    figure_sidedness(result, args.outputs)
    figure_strata_profile(figure_data, result, args.outputs)


if __name__ == "__main__":
    main()
