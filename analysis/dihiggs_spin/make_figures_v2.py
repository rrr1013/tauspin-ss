"""Phase-2 (spin-flat HH) figures: templates and closure, region dependence,
composition dependence and the transfer to the HL-LHC projections."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE / "outputs" / "v2"
FIG = HERE / "outputs" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"H": "#2a78d6", "Z": "#eb6834", "W": "#1baf7a", "U": "#eda100"}
LAB = {"H": r"H$\to\tau\tau$ (signal, single H)", "Z": r"Z$\to\tau\tau$ (Z+HF)",
       "W": r"W$\to\tau\nu$ pair (true-$\tau$ top)", "U": "no spin information (fakes)"}
LS = {"H": "-", "Z": "--", "W": "-.", "U": ":"}


def whist(ax, x, w, bins, **kw):
    h, _ = np.histogram(x, bins, weights=w)
    e2, _ = np.histogram(x, bins, weights=w ** 2)
    n = w.sum()
    c = 0.5 * (bins[1:] + bins[:-1])
    ax.step(bins, np.r_[h, h[-1]] / n, where="post", **kw)
    return c, h / n, np.sqrt(e2) / n


def fig_templates():
    z = np.load(V2 / "closure_U_D.npz")
    g = json.load(open(V2 / "spin_gain_U.json"))
    cl = json.load(open(V2 / "closure_U.json"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    ax = axes[0]
    Du, Dr = z["Du"], z["Dr"]
    bins = np.linspace(*np.percentile(Du, [0.5, 99.5]), 31)
    for X in ("H", "Z", "W", "U"):
        whist(ax, Du, z[f"w{X}"], bins, color=COL[X], ls=LS[X], lw=1.6, label="spin-flat HH reweighted to " + LAB[X])
    c, hr, er = np.histogram(Dr, bins)[0], None, None
    h, _ = np.histogram(Dr, bins)
    ax.errorbar(0.5 * (bins[1:] + bins[:-1]), h / len(Dr), yerr=np.sqrt(h) / len(Dr), fmt="o", ms=3.5,
                color="#0b0b0b", label=f"real (spin-correlated) HH, closure $\\chi^2$/ndf = {cl['chi2_D'][0]:.1f}/{cl['chi2_D'][1]}")
    ax.set_xlabel(r"spin discriminant $\ln D$ from $\hat h$ (tauspin regressor)")
    ax.set_ylabel("fraction of events / bin")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("(a) spin templates, all selected HH-like events", fontsize=10)
    ax = axes[1]
    hu = z["hpu"][:, :, 2].ravel()
    bins = np.linspace(-1.2, 1.2, 31)
    for X in ("H", "Z", "W"):
        whist(ax, hu, np.repeat(z[f"w{X}"], 2), bins, color=COL[X], ls=LS[X], lw=1.6, label=LAB[X])
    h, _ = np.histogram(z["hpr"][:, :, 2].ravel(), bins)
    ax.errorbar(0.5 * (bins[1:] + bins[:-1]), h / h.sum(), yerr=np.sqrt(h) / h.sum(), fmt="o", ms=3.5,
                color="#0b0b0b", label="real HH")
    ax.set_xlabel(r"regressed longitudinal component $\hat h_k$ (both taus)")
    ax.set_ylabel("fraction of taus / bin")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("(b) what carries the top rejection: $\\hat h_k$", fontsize=10)
    ax = axes[2]
    v = g["variants"]["Z_long"]
    effs = [100, 50, 20, 5]
    x = np.arange(len(effs))
    for key, lab, col, mk in (("exact", "exact $h$ (implemented modes)", "#e34948", "D"),
                              ("spin", "tauspin $\\hat h$", "#2a78d6", "o"),
                              ("kin+spin", "kinematics + $\\hat h$ over kinematics", "#4a3aa7", "s"),
                              ("obs", "textbook observables", "#1baf7a", "^"),
                              ("kin", "kinematics only (spin-induced)", "#52514e", "v")):
        y = []
        for e in effs:
            r = v[f"sig_eff_{e}"]
            y.append((r["increment_kin_to_kinspin"] if key == "kin+spin" else r[key]["R"]) - 1)
        ax.plot(x, np.array(y) * 100, marker=mk, color=col, label=lab, ms=6)
    ax.set_xticks(x, [f"{e} %" for e in effs])
    ax.set_xlabel("kinematic-BDT region (HH signal efficiency)")
    ax.set_ylabel("$R-1$ of the most signal-like bin [%]")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.1, 25)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("(c) gain vs kinematic region (ATLAS top-bin mix)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"v2_spin_templates_closure_region.{ext}", dpi=180)


def fig_final():
    f = json.load(open(V2 / "final_v2.json"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    sc = f["scan"]
    t = [r["top_fraction"] for r in sc]
    ax.plot(t, [(r["exact"] - 1) * 100 for r in sc], "-", color="#e34948", marker="D", ms=4, label="exact $h$ (implemented modes)")
    ax.plot(t, [(r["spin"] - 1) * 100 for r in sc], "-", color="#2a78d6", marker="o", ms=4, label="tauspin $\\hat h$")
    ax.axvline(t[0], color="#999", lw=0.8, ls="--")
    ax.annotate("ATLAS Run-2\ntop-bin mix", (t[0], 5.5), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel("top (true $\\tau$) fraction of the background, Z+HF share moved to top")
    ax.set_ylabel("$R-1$ (stat. only) [%]")
    ax.set_ylim(0, 16.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(a) spin gain grows with the top share", fontsize=10)
    ax = axes[1]
    rows = [r for r in f["rows"] if r["f"] in (0.5, 0.62, 1.0)]
    groups = [("bbtt_rel", "ATLAS bb$\\tau\\tau$"), ("atlas_rel", "ATLAS combined"), ("atlas_cms_rel", "ATLAS+CMS")]
    for i, (scen, col, lab) in enumerate((("tauspin spin-only (Z_long)", "#2a78d6", "tauspin $\\hat h$"),
                                          ("exact h (implemented modes)", "#e34948", "exact $h$ (reference)"))):
        for j, (k, _) in enumerate(groups):
            vals = [r[k] * 100 for r in rows if r["scenario"] == scen]
            cen = [r[k] * 100 for r in rows if r["scenario"] == scen and r["f"] == 0.62]
            xx = j + (i - 0.5) * 0.3
            ax.plot([xx, xx], [min(vals), max(vals)], color=col, lw=6, alpha=0.35, solid_capstyle="butt")
            ax.plot([xx] * 2, cen, "o", color=col, ms=6, label=lab if j == 0 else None)
            ax.annotate(f"{min(cen):+.1f}…{max(cen):+.1f}%", (xx, max(vals)), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(range(3), [g for _, g in groups])
    ax.set_ylabel("relative gain in expected SM HH significance [%]")
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(b) transfer: bars f = 0.5–1 and rest = 1 … top-rich; dots f = 0.62", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"v2_final_gain_projection.{ext}", dpi=180)


if __name__ == "__main__":
    fig_templates()
    fig_final()
