"""Figures for the H -> tau tau entanglement feasibility."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE / "outputs" / "ent"
FIG = HERE / "outputs" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
LEV = {"exact": ("exact $h$ (reference)", "#e34948", "D", "-"),
       "tauspin": ("tauspin $\\hat h$ (IP + SV, local frame)", "#2a78d6", "o", "-"),
       "tauspin_noIPSV": ("tauspin $\\hat h$ without IP/SV", "#4a3aa7", "s", "--"),
       "textbook": ("textbook observables ($\\Upsilon$, $x$, hybrid $h$)", "#1baf7a", "^", "-.")}


def fig_templates():
    z = np.load(E / "closure_g.npz")
    G, Gh, isH = z["G"], z["Gh"], z["isH"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, j, lab in ((axes[0], 0, "$g_s=E[h^-_n h^+_n + h^-_r h^+_r - h^-_k h^+_k \\,|\\, x]$"),
                       (axes[1], 1, "$g_Z=E[\\rho_Z - 1 \\,|\\, x]$")):
        e = np.linspace(*np.percentile(G[:, j], [0.5, 99.5]), 41)
        c = 0.5 * (e[1:] + e[:-1])
        for key, col, ls, nm in (("wH", "#2a78d6", "-", "H: Bell state ($p=1$)"),
                                 ("wW13", "#eda100", ":", "separable boundary ($p=1/3$)"),
                                 ("wZ", "#eb6834", "--", "Z$\\to\\tau\\tau$ spin state")):
            hh, _ = np.histogram(G[:, j], e, weights=z[key])
            ax.step(e, np.r_[hh, hh[-1]] / z[key].sum(), where="post", color=col, ls=ls, lw=1.6,
                    label="spin-flat H(125)+j reweighted: " + nm)
        hr, _ = np.histogram(Gh[isH, j], e)
        ax.errorbar(c, hr / isH.sum(), yerr=np.sqrt(hr) / isH.sum(), fmt="o", ms=3, color="#0b0b0b",
                    label="real H(125)+j (independent sample)")
        ax.set_xlabel(lab + "  (tauspin $\\hat h$)")
        ax.set_ylabel("fraction of events / bin")
        ax.set_yscale("log")
        ax.set_ylim(top=1.0)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_title("(a) projection on the entanglement witness", fontsize=10)
    axes[1].set_title("(b) projection on the Z spin state", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"ent_templates_closure.{ext}", dpi=180)


def fig_needs_and_lumi():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.1, 1.15]})
    # (a) events needed for 5 sigma vs S/B
    nom = json.load(open(E / "ent_nominal.json"))["levels"]
    noi = json.load(open(E / "ent_noIPSV.json"))["levels"]
    ax = axes[0]
    sbs = [("inf", np.inf), ("SB1.0_dB0.02", 1.0), ("SB0.3_dB0.02", 0.3), ("SB0.1_dB0.02", 0.1)]
    xs = [3, 1.0, 0.3, 0.1]
    for name, src in (("exact", nom), ("tauspin", nom), ("tauspin_noIPSV", {"tauspin_noIPSV": noi["tauspin"]}),
                      ("textbook", nom)):
        lv = src[name] if name != "tauspin_noIPSV" else src["tauspin_noIPSV"]
        ys = [lv["scan"][k]["S_for_5sigma"] for k, _ in sbs]
        lab, col, mk, ls = LEV[name]
        ax.plot(xs, ys, marker=mk, color=col, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs, ["no bkg", "1", "0.3", "0.1"])
    ax.set_xlabel("signal-to-background ratio $S/B$")
    ax.set_ylabel("selected H$\\to\\tau_h\\tau_h$ events for 5$\\sigma$")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(a) events needed (background norm. $\\pm$2 %)", fontsize=10)
    # (b) significance vs luminosity
    lum = json.load(open(E / "ent_lumi.json"))
    ax = axes[1]
    for name in ("exact", "tauspin", "tauspin_noIPSV", "textbook"):
        rows = np.array(lum[name])
        lab, col, mk, ls = LEV[name]
        ax.plot(rows[:, 0], rows[:, 1], marker=mk, color=col, ls=ls, label=lab)
    for y in (3, 5):
        ax.axhline(y, color="#999", lw=0.8, ls=":")
    ax.axvline(450, color="#bbb", lw=0.8)
    ax.axvline(3000, color="#bbb", lw=0.8)
    ax.axvline(6000, color="#bbb", lw=0.8)
    ax.text(450, 12.5, " Run 2+3", fontsize=8)
    ax.text(3000, 12.5, " HL-LHC", fontsize=8)
    ax.text(6000, 12.5, " ×2 exp.", fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([300, 450, 1000, 2000, 3000, 6000], ["300", "450", "1000", "2000", "3000", "6000"])
    ax.minorticks_off()
    ax.set_xlabel("integrated luminosity [fb$^{-1}$] (one experiment; 6000 ≈ ATLAS+CMS)")
    ax.set_ylabel("expected entanglement significance [$\\sigma$]")
    ax.set_ylim(0, 13.5)
    ax.set_title("(b) H$\\to\\tau_h\\tau_h$, ATLAS-like categories, response unc. 10 %", fontsize=10)
    # (c) per-category at 3 ab^-1
    fin = json.load(open(E / "ent_final.json"))["levels"]
    ax = axes[2]
    cats = list(fin["tauspin"]["resp0.1"]["per_category"])
    y = np.arange(len(cats))
    for k, name in enumerate(("tauspin", "textbook")):
        v = [fin[name]["resp0.1"]["per_category"][c] for c in cats]
        lab, col, mk, ls = LEV[name]
        ax.barh(y + (k - 0.5) * 0.38, v, 0.36, color=col, label=f"{lab.split(' (')[0]}: combined "
                f"{fin[name]['resp0.1']['combined']:.1f}$\\sigma$")
    ax.set_yticks(y, [c.replace("_", " ") for c in cats], fontsize=8)
    ax.set_xlabel("significance per category at 3 ab$^{-1}$ [$\\sigma$]")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("(c) where the sensitivity comes from", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"ent_needs_lumi_categories.{ext}", dpi=180)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_templates()
    fig_needs_and_lumi()
