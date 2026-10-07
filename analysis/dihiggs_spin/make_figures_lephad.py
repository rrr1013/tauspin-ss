"""Figure and projection for the tau_lep tau_had extension (lepton lifetime / tauspin inputs).

Transfer to the ATLAS HL-LHC projection (ATL-PHYS-PUB-2024-016): bbtautau 3.5 sigma
(tau_had tau_had 3.1, tau_lep tau_had 1.8), ATLAS HH combination 4.3 sigma; the other channels
are sqrt(4.3^2 - 3.5^2).  tau_lep tau_had is scaled by the measured ratio R_lh; tau_had tau_had by
the earlier spin result (channel +1.1 to +2.7 %, run tauspin-dihiggs-spin-projection-20261006).
bbtautau is rescaled by the ratio of the quadrature sums (3.1 (+) 1.8 = 3.58 vs the quoted 3.5).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results" / "lephad"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def project(r_lh, r_hh_ch):
    lh, hh = 1.8 * r_lh, 3.1 * r_hh_ch
    bbtt = 3.5 * np.hypot(lh, hh) / np.hypot(1.8, 3.1)
    other = np.sqrt(4.3 ** 2 - 3.5 ** 2)
    return bbtt, np.hypot(bbtt, other)


def main():
    c = json.load(open(R / "cls.json"))["scenarios"]
    h = json.load(open(R / "cls_hspin.json"))["scenarios"]["no_ttva/nominal"]
    f = json.load(open(R / "factorized.json"))["categories"]
    k0 = h["K"]["mean"]
    ratios = {
        "lepton |d0|/sigma": h["K+d0"]["mean"] / k0,
        "lepton IP (local frame)": h["K+IP"]["mean"] / k0,
        "had. polarisation only": h["K+Hspin"]["mean"] / k0,
        "lepton IP, resolution x1.5": c["no_ttva/ip_res_x1.5"]["K+IP"]["mean"] / c["no_ttva/ip_res_x1.5"]["K"]["mean"],
        "lepton IP, resolution x0.8": c["no_ttva/ip_res_x0.8"]["K+IP"]["mean"] / c["no_ttva/ip_res_x0.8"]["K"]["mean"],
        "ATLAS-anchored (SLT / LTT)": (f["SLT"]["as_published"]["R"] + f["LTT"]["as_published"]["R"]) / 2,
    }
    proj = {}
    for name, r in (("nominal low (IP local)", ratios["lepton IP (local frame)"]), ("nominal high (|d0|/sigma)", ratios["lepton |d0|/sigma"]),
                    ("resolution x1.5", ratios["lepton IP, resolution x1.5"])):
        for hh_name, rh in (("no tau_had-tau_had gain", 1.0), ("tau_had-tau_had spin +2.7 %", 1.027)):
            bbtt, comb = project(r, rh)
            proj[f"{name} | {hh_name}"] = {"R_lh": r, "bbtautau": bbtt, "ATLAS_combined": comb}
    json.dump({"ratios": ratios, "projection": proj}, open(R / "summary.json", "w"), indent=1)
    for k, v in ratios.items():
        print(f"{k:32s} R = {v:.3f}")
    for k, v in proj.items():
        print(f"{k:60s} bbtautau {v['bbtautau']:.2f}  ATLAS {v['ATLAS_combined']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rows = [("kinematics only (K)", h["K"]["mean"], h["K"]["sd"], "#999999"),
            ("K + had. polarisation (tauspin inputs, no IP/SV)", h["K+Hspin"]["mean"], h["K+Hspin"]["sd"], "#4a3aa7"),
            ("K + lepton |d0|/sigma", h["K+d0"]["mean"], h["K+d0"]["sd"], "#2a78d6"),
            ("K + lepton IP (local frame)", h["K+IP"]["mean"], h["K+IP"]["sd"], "#2a78d6"),
            ("K + lepton IP, resolution x1.5", c["no_ttva/ip_res_x1.5"]["K+IP"]["mean"], c["no_ttva/ip_res_x1.5"]["K+IP"]["sd"], "#7fb2ea"),
            ("K + lepton IP, resolution x0.8", c["no_ttva/ip_res_x0.8"]["K+IP"]["mean"], c["no_ttva/ip_res_x0.8"]["K+IP"]["sd"], "#1c4f8f")]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, sd, col) in zip(y, rows):
        ax.barh(yi, v, xerr=sd, color=col, height=0.6)
        ax.text(v + 0.05, yi, f"{v:.2f}  (x{v / h['K']['mean']:.2f})", va="center", fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("expected tau_lep tau_had HH significance, our simulation [sigma]")
    ax.set_xlim(0, 2.9)
    ax.set_title("(a) tau_lep tau_had, 3 ab$^{-1}$, SLT (+) LTT, no d0 requirement", fontsize=10)
    ax = axes[1]
    labels = ["ATLAS\nprojection", "+ lepton IP\n(res. x1.5)", "+ lepton IP\n(nominal)", "+ lepton |d0|/sigma\n+ tau_had tau_had spin"]
    vals_bb = [3.5, project(ratios["lepton IP, resolution x1.5"], 1.0)[0],
               project(ratios["lepton IP (local frame)"], 1.0)[0], project(ratios["lepton |d0|/sigma"], 1.027)[0]]
    vals_c = [4.3, project(ratios["lepton IP, resolution x1.5"], 1.0)[1],
              project(ratios["lepton IP (local frame)"], 1.0)[1], project(ratios["lepton |d0|/sigma"], 1.027)[1]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, vals_bb, 0.36, color="#2a78d6", label="bbtautau")
    ax.bar(x + 0.18, vals_c, 0.36, color="#eb6834", label="ATLAS HH combination")
    for xi, a, b in zip(x, vals_bb, vals_c):
        ax.text(xi - 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("expected significance at 3 ab$^{-1}$ [sigma]")
    ax.set_ylim(0, 5.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("(b) transfer to the ATLAS HL-LHC projection (ratios applied)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(R / f"lephad_gain.{ext}", dpi=180)


if __name__ == "__main__":
    main()
