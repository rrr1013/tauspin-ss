"""Final projection table and summary figure.

Scenario ratios R (tau_had tau_had most signal-like bin, profiled normalisation
systematics) come from
  phase 1   ATLAS full-sim cohort (pi / rho / 3pi on both sides), phase1/*.json
  phase 2   HH-matched hold-out with the realistic decay-mode mix, v1/matched_rw.json
tau_lep tau_had uses the phase-1 single-side polarisation ratios (SLT 0.75, LTT 0.25),
which are upper estimates since they ignore the decay-mode dilution.
The transfer to ATLAS / ATLAS+CMS follows projection.combine.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from projection import REF, combine

HERE = Path(__file__).resolve().parent
P1 = HERE / "outputs" / "phase1"
V1 = HERE / "outputs" / "v1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def main():
    t1 = json.load(open(P1 / "significance_top_bin.json"))["arms"]
    l1 = json.load(open(P1 / "significance_lephad.json"))["arms"]
    m2 = json.load(open(V1 / "matched_rw.json"))["regions"]["sig_eff_20"]
    sc = json.load(open(V1 / "scan_composition.json"))
    rl = {a: 0.75 * l1[a]["SLT"]["R_syst"] + 0.25 * l1[a]["LTT"]["R_syst"] for a in l1}
    scen = [
        ("tauspin reco h, realistic modes (phase 2)", m2["tauspin"]["R_syst_mean"], rl["reco_IPSV"]),
        ("tauspin reco h, pi/rho/3pi cohort (phase 1)", t1["reco_IPSV"]["R_syst"], rl["reco_IPSV"]),
        ("ideal IP direction (phase 1)", t1["reco_idealIP"]["R_syst"], rl["reco_idealIP"]),
        ("exact h, realistic modes (phase 2)", m2["exact"]["R_syst"], rl["exact"]),
        ("exact h, pi/rho/3pi cohort (phase 1)", t1["exact"]["R_syst"], rl["exact"]),
    ]
    rows = [{"scenario": "published baseline (S2, 3 ab-1)", "R_hadhad": 1.0, "R_lephad": 1.0, **combine(1.0, 1.0)}]
    for name, rh, rlh in scen:
        rows.append({"scenario": name, "R_hadhad": rh, "R_lephad": rlh, **combine(rh, rlh)})
    for r in rows:
        print(f"{r['scenario']:46s} R_hh {r['R_hadhad']:.3f} R_lh {r['R_lephad']:.3f}  bbtt {r['bbtt']:.2f}  "
              f"ATLAS {r['atlas']:.2f}  ATLAS+CMS {r['atlas_cms']:.2f}  kl {r['kl_lo'] * 100:+.0f}/{r['kl_hi'] * 100:+.0f}%")
    json.dump(rows, open(V1 / "final_projection.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    # (a) gain by background type
    ax = axes[0]
    groups = ["Z+HF", "top", "fakes", "singleH"]
    lab = {"Z+HF": "only\nZ+HF", "top": "only\ntop", "fakes": "only\nfakes", "singleH": "only\nsingle H"}
    x = np.arange(len(groups) + 1)
    ex = [sc["pure"]["exact"][g] for g in groups] + [sc["table5"]["exact"]]
    ts = [sc["pure"]["tauspin"][g] for g in groups] + [sc["table5"]["tauspin"]]
    ax.bar(x - 0.18, (np.array(ex) - 1) * 100, 0.34, color="#eb6834", label="exact $h$ (ceiling)")
    ax.bar(x + 0.18, (np.array(ts) - 1) * 100, 0.34, color="#2a78d6", label="tauspin regressed $h$")
    for xi, v in zip(x, ts):
        ax.annotate(f"{(v - 1) * 100:+.1f}", (xi + 0.18, (v - 1) * 100), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=8)
    for xi, v in zip(x, ex):
        ax.annotate(f"{(v - 1) * 100:+.1f}", (xi - 0.18, (v - 1) * 100), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(x, [lab[g] for g in groups] + ["ATLAS\ntop-bin mix"], fontsize=8.5)
    ax.set_ylabel("significance gain of the bin, $Z_{\\rm spin}/Z-1$ [%]")
    ax.set_title("(a) spin gain vs background type\n(HH-matched kinematics, stat. only)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    # (b) R for scenarios
    ax = axes[1]
    names = [s[0] for s in scen]
    y = np.arange(len(scen))[::-1]
    ax.plot([(s[1] - 1) * 100 for s in scen], y, "o", color="#2a78d6", ms=8, label=r"$\tau_{\rm had}\tau_{\rm had}$")
    ax.plot([(s[2] - 1) * 100 for s in scen], y - 0.15, "s", color="#1baf7a", ms=6,
            label=r"$\tau_{\rm lep}\tau_{\rm had}$ (phase 1, upper)")
    for yy, s in zip(y, scen):
        ax.annotate(f"{(s[1] - 1) * 100:+.1f}%", ((s[1] - 1) * 100, yy), xytext=(6, 3), textcoords="offset points",
                    fontsize=8)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("$R-1$ [%] (with normalisation systematics)")
    ax.set_xlim(-0.5, 16)
    ax.axvline(0, color="#999", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(b) gain of the most signal-like bin", fontsize=10)
    # (c) projection
    ax = axes[2]
    keys = [("bbtt", "ATLAS\nbb$\\tau\\tau$"), ("atlas", "ATLAS\ncombined"), ("atlas_cms", "ATLAS+CMS\ncombined")]
    sel = [rows[0], rows[1], rows[2], rows[4], rows[5]]
    short = ["published", "tauspin reco\n(phase 2)", "tauspin reco\n(phase 1)", "exact $h$\n(phase 2)",
             "exact $h$\n(phase 1)"]
    cols = ["#52514e", "#2a78d6", "#4a3aa7", "#eb6834", "#e34948"]
    xx = np.arange(len(keys))
    for i, (r, nm, c) in enumerate(zip(sel, short, cols)):
        ax.bar(xx + (i - 2) * 0.16, [r[k] for k, _ in keys], 0.15, color=c, label=nm.replace("\n", " "))
    for j, (k, _) in enumerate(keys):
        ax.annotate(f"{rows[0][k]:.1f}σ → {rows[1][k]:.2f}σ\n(ceiling {rows[4][k]:.2f}σ)", (j, rows[5][k]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(xx, [n for _, n in keys])
    ax.set_ylabel("expected significance of SM HH, 3 ab$^{-1}$ [σ]")
    ax.set_ylim(0, 11.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("(c) transfer to the HL-LHC projection (S2)", fontsize=10)
    fig.tight_layout()
    out = HERE / "outputs" / "figures"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"final_spin_gain_projection.{ext}", dpi=180)


if __name__ == "__main__":
    main()
