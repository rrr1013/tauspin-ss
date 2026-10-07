"""Revised summary after the review probes (probes_lephad.py): the ATLAS-composition-anchored
factorised estimate with realistic resolution, IP-width nuisance, 'other' and non-prompt-fake
variants is the main result; the direct method is shown with its MC-statistics spread."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_figures_lephad import project

R = Path(__file__).resolve().parent / "results" / "lephad"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def main():
    p = json.load(open(R / "probes.json"))
    dct, fac = p["direct"], p["factorised"]
    r_fac = np.array([v["R"] for cat in fac.values() for v in cat.values()])
    rows = [("null control (IP shuffled)", dct["P1 null K'+IP"]["mean"] / dct["P0 K'"]["mean"]),
            ("direct, nominal resolution", dct["P0 K'+IP"]["mean"] / dct["P0 K'"]["mean"]),
            ("direct, realistic resolution", dct["P2 real K'+IP"]["mean"] / dct["P0 K'"]["mean"]),
            ("direct, realistic + 30 % non-prompt fakes", dct["P3 nonprompt f0.3 mean100um K'+IP"]["mean"] / dct["P0 K'"]["mean"]),
            ("direct, realistic + 50 % non-prompt fakes", dct["P3 nonprompt f0.5 mean100um K'+IP"]["mean"] / dct["P0 K'"]["mean"]),
            ("direct, with standard d0 cut (realistic)", dct["P5 ttva real K'+IP"]["mean"] / dct["P5 ttva real K'"]["mean"])]
    ess = {me: dct[f"P4 min_ess{me} K'+IP"]["mean"] / dct[f"P4 min_ess{me} K'"]["mean"] for me in ("50", "200", "400")}
    boot = [dct[f"P4 bootstrap{b} K'+IP"]["mean"] / dct[f"P4 bootstrap{b} K'"]["mean"] for b in range(3)]
    lo, hi = float(r_fac.min()), float(r_fac.max())
    proj = {"R_lh_range": [lo, hi], "R_hh_channel_range": [1.0, 1.027]}
    for tag, rl, rh in (("low", lo, 1.0), ("high", hi, 1.027)):
        bb, comb = project(rl, rh)
        proj[tag] = {"bbtautau": bb, "ATLAS_combined": comb}
    summ = {"direct_ratios": dict(rows), "direct_min_ess": ess, "direct_bootstrap": boot,
            "factorised_R": {c: {k: v["R"] for k, v in d.items()} for c, d in fac.items()}, "projection": proj}
    json.dump(summ, open(R / "summary_v2.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    labels = [r[0] for r in rows] + [f"direct, merge threshold {k}" for k in ess] + [f"direct, MC bootstrap {i}" for i in range(3)]
    vals = [r[1] for r in rows] + list(ess.values()) + boot
    yy = np.arange(len(labels))[::-1]
    ax.scatter(vals, yy, color=["#999999"] + ["#2a78d6"] * 5 + ["#7fb2ea"] * 6, zorder=3)
    ax.axvspan(lo, hi, color="#eb6834", alpha=0.25, label=f"ATLAS-anchored, all variants: x{lo:.2f}-{hi:.2f}")
    ax.axvline(1, color="#999", lw=0.8)
    ax.set_yticks(yy, labels, fontsize=8)
    ax.set_xlabel("ratio of tau_lep tau_had HH significance: + lepton IP / kinematics only")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(a) lepton lifetime information in tau_lep tau_had (3 ab$^{-1}$)", fontsize=10)
    ax = axes[1]
    labs = ["ATLAS\nprojection", f"+ lepton IP\nx{lo:.2f}", f"+ lepton IP x{hi:.2f}\n+ tau_had tau_had spin"]
    bb = [3.5, project(lo, 1.0)[0], project(hi, 1.027)[0]]
    cc = [4.3, project(lo, 1.0)[1], project(hi, 1.027)[1]]
    x = np.arange(3)
    ax.bar(x - 0.18, bb, 0.36, color="#2a78d6", label="bbtautau")
    ax.bar(x + 0.18, cc, 0.36, color="#eb6834", label="ATLAS HH combination")
    for xi, a, b in zip(x, bb, cc):
        ax.text(xi - 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labs, fontsize=8.5)
    ax.set_ylim(0, 5.3)
    ax.set_ylabel("expected significance at 3 ab$^{-1}$ [sigma]")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("(b) illustrative transfer (ratios applied to ATL-PHYS-PUB-2024-016)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"lephad_gain_v2.{ext}", dpi=180)


if __name__ == "__main__":
    main()
