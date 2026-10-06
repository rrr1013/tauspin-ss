"""Phase-1 figures: spin-only templates on the ATLAS validation cohort and the
significance ratio R versus the level of spin reconstruction."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase1_significance import TABLE5_HADHAD, yields

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "phase1"
FIG = HERE / "outputs" / "figures"
HYPS = ("H", "Z", "W", "U", "WU")
COL = {"H": "#2a78d6", "Z": "#eb6834", "W": "#1baf7a", "U": "#eda100", "WU": "#4a3aa7"}
LAB = {"H": r"H$\to\tau\tau$ (HH signal, single H)", "Z": r"Z$\to\tau\tau$ (Z+HF)",
       "W": r"W$\to\tau\nu$ pair (true-$\tau$ top)", "U": "no spin correlation (multijet-fake proxy)",
       "WU": r"one W $\tau$ + unpolarised (t$\bar{\rm t}$-fake proxy)"}
LS = {"H": "-", "Z": "--", "W": "-.", "U": ":", "WU": (0, (5, 1, 1, 1))}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def discriminant(probs, frac):
    p = probs / probs.sum(1, keepdims=True)
    den = sum(f * p[:, HYPS.index(h)] for h, f in frac.items())
    return np.log(p[:, 0] + 1e-12) - np.log(den + 1e-12)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    st = np.load(OUT / "probs_pool.npz")
    w = {h: st[f"w_{h}"] for h in HYPS}
    s, bkg = yields(TABLE5_HADHAD)
    by = {}
    for _, (y, hyp, _) in bkg.items():
        by[hyp] = by.get(hyp, 0) + y
    frac = {h: v / sum(by.values()) for h, v in by.items()}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, arm, title in ((axes[0], "exact", "exact $h$ (truth)"),
                           (axes[1], "reco_IPSV", r"reco $h_{\rm pred}$ (3 IP + SV, ATLAS full sim)")):
        d = discriminant(st[f"probs_{arm}"], frac)
        lo, hi = np.percentile(d, [0.5, 99.5])
        bins = np.linspace(lo, hi, 41)
        for h in HYPS:
            ax.hist(d, bins=bins, weights=w[h] / w[h].sum(), histtype="step", color=COL[h], ls=LS[h], lw=1.6,
                    label=LAB[h])
        ax.set_xlabel(r"spin discriminant $\ln D$,  $D=p_H/\sum_b f_b\,p_b$")
        ax.set_ylabel("fraction of events / bin")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    # R vs reconstruction level
    t = json.load(open(OUT / "significance_top_bin.json"))["arms"]
    lh = json.load(open(OUT / "significance_lephad.json"))["arms"]
    arms = ["reco_noIPSV", "reco_IPSV", "reco_idealIP", "exact"]
    names = ["reco, no IP/SV", "reco, 3 IP + SV", "reco, ideal IP", "exact $h$"]
    ax = axes[2]
    y = np.arange(len(arms))
    ax.plot([(t[a]["R_syst"] - 1) * 100 for a in arms], y, "o", color="#2a78d6", ms=8,
            label=r"$\tau_{\rm had}\tau_{\rm had}$ (pair: correlation + polarisation)")
    ax.plot([(lh[a]["LTT"]["R_syst"] - 1) * 100 for a in arms], y - 0.12, "s", color="#eb6834", ms=7,
            label=r"$\tau_{\rm lep}\tau_{\rm had}$ LTT (one-side polarisation)")
    ax.plot([(lh[a]["SLT"]["R_syst"] - 1) * 100 for a in arms], y + 0.12, "^", color="#1baf7a", ms=7,
            label=r"$\tau_{\rm lep}\tau_{\rm had}$ SLT (one-side polarisation)")
    for a, yy in zip(arms, y):
        ax.annotate(f"{(t[a]['R_syst'] - 1) * 100:+.1f}%", ((t[a]["R_syst"] - 1) * 100, yy),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, names)
    ax.set_xlabel("gain in expected significance of the most\nsignal-like bin, $Z_{\\rm spin}/Z - 1$ [%]")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlim(-0.5, 15)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.25))
    ax.set_title("spin binning added to the ATLAS Run-2 top bin (3 ab$^{-1}$)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"phase1_spin_templates_and_gain.{ext}", dpi=180)
    print("saved")


if __name__ == "__main__":
    main()
