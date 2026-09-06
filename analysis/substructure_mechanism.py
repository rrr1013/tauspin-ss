#!/usr/bin/env python3
"""Follow-up to the substructure run: where does the little substructure signal come from?

Two questions that the main result forces, both answered with the same estimator
and the same frozen inputs, nothing retrained and the test partition never read.

1. Does substructure add anything the event-level block did not already have?
   Paired bootstrap difference between the ``E-reco`` drop and the
   ``E-reco x S-all`` drop, on the same resamples.

2. Is the separation that the substructure index does have just the tau's
   direction and the event's hadronic activity in disguise?  The substructure
   indices are conditioned on the reconstructed production block, on the visible
   ditau rapidity alone and on ``sum(E_T)`` alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import tau_substructure_information as ts
from nonspin_information import P_RECO, auc, combine_strata, conditional_auc, quantile_strata

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260907


def figure_mechanism(result: dict, outputs: Path) -> None:
    """Each substructure probe, before and after the event-level conditioners."""
    import matplotlib.pyplot as plt

    probes = [
        ("index[S-all]", "all 48 substructure variables", "#000000"),
        ("index[S5]", "S5 detector-only index", "#7f7f7f"),
    ]
    conditioners = [
        ("unconditional", "no conditioner"),
        ("given_sumET", r"$\Sigma E_T$ held fixed"),
        ("given_|y_tautau|", r"$|y_{\tau\tau}|$ held fixed"),
        ("given_|y| x sumET", r"$|y_{\tau\tau}| \times \Sigma E_T$ held fixed"),
        ("given_P-reco", "P-reco block held fixed"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    offsets = {0: +0.16, 1: -0.16}
    y = np.arange(len(conditioners))[::-1]
    for row, (probe, label, colour) in enumerate(probes):
        for pos, (suffix, _) in zip(y, conditioners):
            key = f"{probe}_{suffix}" if suffix != "unconditional" else f"{probe}_unconditional"
            value = result["point"][key]
            interval = result["interval"][key]
            ax.plot([interval["q025"], interval["q975"]], [pos + offsets[row]] * 2,
                    color=colour, linewidth=1.6, alpha=0.8)
            ax.plot([value], [pos + offsets[row]], marker="o" if row == 0 else "s",
                    color=colour, markersize=6, label=label if pos == y[0] else None)
    z = result["point"]["z0sin_minus_unconditional"]
    zc = result["point"]["z0sin_minus_given_tau_eta"]
    ax.axvline(0.5, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in conditioners], fontsize=9)
    ax.set_xlabel("AUC of the substructure index itself (H positive)")
    ax.set_title("The substructure index does not have its own H/Z information\n"
                 "hold the event-level production block fixed and it goes to 0.5", fontsize=10)
    ax.annotate(
        "single strongest substructure variable, "
        rf"$\max|z_0\sin\theta|(\tau^-)$: {z:.4f} $\to$ {zc:.4f}"
        "\nwhen the tau's own $|\\eta|$ is held fixed",
        xy=(0.98, 0.06), xycoords="axes fraction", ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7"))
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    outputs.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs / "substructure-mechanism.png", dpi=160, bbox_inches="tight")
    fig.savefig(outputs / "substructure-mechanism.pdf", bbox_inches="tight")
    print(f"wrote {outputs / 'substructure-mechanism.png'}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--row-map", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--event-features", type=Path, required=True)
    p.add_argument("--substructure", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = ts.load_everything(args)["table"]
    keep = table["report_mask"]
    positive = table["is_higgs"][keep]
    cluster = table["source_file_index"][keep]
    fold = (cluster % 2).astype(np.int64)
    score = table["full_reco"][keep].astype(np.float64)

    blocks = ts.build_blocks()
    wanted = {"E-reco": blocks["E-reco"], "S-all": blocks["S-all"],
              "S5": blocks["S5"], "P-reco": P_RECO}
    indices = {name: ts.crossfit_index(ts.transform_columns(table, block, keep), positive, fold)
               for name, block in wanted.items()}
    strata = {name: quantile_strata(values, ts.STRATA_BINS) for name, values in indices.items()}
    strata["E-reco x S-all"] = combine_strata(strata["E-reco"], strata["S-all"])
    rapidity = table["abs_vis_rapidity"][keep].astype(np.float64)
    sumet = table["met_sumet"][keep].astype(np.float64)
    strata["|y_tautau|"] = quantile_strata(rapidity, ts.STRATA_BINS)
    strata["sumET"] = quantile_strata(sumet, ts.STRATA_BINS)
    strata["|y| x sumET"] = combine_strata(strata["|y_tautau|"], strata["sumET"])
    strata["P-reco"] = strata["P-reco"]

    probes = ["S-all", "S5"]
    conditioners = ["P-reco", "|y_tautau|", "sumET", "|y| x sumET"]

    point = {
        "score_unconditional": auc(score, positive),
        "score_given_E-reco": conditional_auc(score, positive, strata["E-reco"]),
        "score_given_E-reco_x_S-all": conditional_auc(score, positive, strata["E-reco x S-all"]),
    }
    for probe in probes:
        point[f"index[{probe}]_unconditional"] = auc(indices[probe], positive)
        for name in conditioners:
            point[f"index[{probe}]_given_{name}"] = conditional_auc(
                indices[probe], positive, strata[name])
    point["z0sin_minus_unconditional"] = auc(table["max_abs_z0sin_minus"][keep], positive)
    point["z0sin_minus_given_tau_eta"] = conditional_auc(
        table["max_abs_z0sin_minus"][keep], positive,
        quantile_strata(table["abs_tau_eta_minus"][keep].astype(np.float64), ts.STRATA_BINS))

    names = list(point.keys())

    def statistics(idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(names))
        pos = positive[idx]
        for k, name in enumerate(names):
            if name == "score_unconditional":
                out[k] = auc(score[idx], pos)
            elif name == "score_given_E-reco":
                out[k] = conditional_auc(score[idx], pos, strata["E-reco"][idx])
            elif name == "score_given_E-reco_x_S-all":
                out[k] = conditional_auc(score[idx], pos, strata["E-reco x S-all"][idx])
            elif name == "z0sin_minus_unconditional":
                out[k] = auc(table["max_abs_z0sin_minus"][keep][idx], pos)
            elif name == "z0sin_minus_given_tau_eta":
                out[k] = conditional_auc(
                    table["max_abs_z0sin_minus"][keep][idx], pos,
                    quantile_strata(table["abs_tau_eta_minus"][keep].astype(np.float64),
                                    ts.STRATA_BINS)[idx])
            else:
                probe = name[name.index("[") + 1:name.index("]")]
                if name.endswith("_unconditional"):
                    out[k] = auc(indices[probe][idx], pos)
                else:
                    conditioner = name.split("_given_", 1)[1]
                    out[k] = conditional_auc(indices[probe][idx], pos, strata[conditioner][idx])
        return out

    unique, inverse = np.unique(cluster, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_REPLICATES, len(names)))
    for r in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(0, unique.size, unique.size)
        idx = np.concatenate([members[i] for i in picked])
        draws[r] = statistics(idx)

    column = {name: k for k, name in enumerate(names)}
    result = {
        "point": point,
        "interval": {
            name: {"q025": float(np.nanquantile(draws[:, k], 0.025)),
                   "q975": float(np.nanquantile(draws[:, k], 0.975))}
            for name, k in column.items()
        },
        "paired": {},
    }
    pairs = [
        ("substructure increment over E-reco",
         "score_given_E-reco", "score_given_E-reco_x_S-all"),
        ("S-all index lost to P-reco",
         "index[S-all]_unconditional", "index[S-all]_given_P-reco"),
        ("S5 index lost to P-reco",
         "index[S5]_unconditional", "index[S5]_given_P-reco"),
        ("S-all index lost to |y| x sumET",
         "index[S-all]_unconditional", "index[S-all]_given_|y| x sumET"),
        ("z0sin lost to the tau direction",
         "z0sin_minus_unconditional", "z0sin_minus_given_tau_eta"),
    ]
    for label, first, second in pairs:
        diff = draws[:, column[first]] - draws[:, column[second]]
        result["paired"][label] = {
            "point": float(point[first] - point[second]),
            "q025": float(np.nanquantile(diff, 0.025)),
            "q975": float(np.nanquantile(diff, 0.975)),
        }

    args.outputs.mkdir(parents=True, exist_ok=True)
    (args.outputs / "mechanism.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    figure_mechanism(result, args.outputs)
    for name in names:
        interval = result["interval"][name]
        print(f"{name:38s} {point[name]:.6f}  [{interval['q025']:.6f}, {interval['q975']:.6f}]")
    print()
    for label, entry in result["paired"].items():
        print(f"{label:38s} {entry['point']:+.6f}  [{entry['q025']:+.6f}, {entry['q975']:+.6f}]")


if __name__ == "__main__":
    main()
