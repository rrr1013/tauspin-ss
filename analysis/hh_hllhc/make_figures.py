"""Figures of the HL-LHC bbtautau TauSpin study (reads outputs/, writes outputs/figures)."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hepdata_inputs import load_latest, load_run2
from projection import DEFAULT, SpinTemplates, build_spec, proc_mix, DI_HYPS, SI_HYPS

OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs"))
FIG = OUT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED = "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 160, "legend.frameon": False})
PROC_STYLE = {  # fixed colour per process (never by rank)
    "top": (C[0], "Top (true τ, W→τ)"), "top_lepfake": (C[4], "tt̄ lepton→τ fake"),
    "zhf": (C[1], "Z/γ*→ττ + HF"), "single_h": (C[2], "single H"),
    "fake_mj": (C[3], "jet→τ fake (τhτh MJ; τℓτh all)"), "fake_tt": (C[5], "jet→τ fake (tt̄, W)"),
    "fake": (C[3], "jet→τ fake (τhτh MJ; τℓτh all)"), "other": (C[6], "other")}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_composition():
    run2 = load_run2()
    lat = load_latest("run3")
    panels = [("Run-2 legacy (2209.10910) τhτh", run2["hadhad"]),
              ("Run-2 legacy τℓτh SLT", run2["slt"]),
              ("Latest Run 3 (2607.26879) τhτh SR Hi", lat["hadhad_Hi"]),
              ("Latest Run 3 τℓτh SR Hi", lat["lephad_Hi"])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.2), gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for j, (title, rec) in enumerate(panels):
        procs = [p for p in PROC_STYLE if p in rec]
        B = sum(rec[p] for p in procs)
        x = np.arange(1, len(B) + 1)
        bottom = np.zeros_like(B)
        ax = axes[0, j]
        for p in procs:
            f = rec[p] / B
            ax.bar(x, f, bottom=bottom, width=0.86, color=PROC_STYLE[p][0], label=PROC_STYLE[p][1],
                   edgecolor="white", linewidth=1.0)
            bottom += f
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(0, 1)
        ax.set_ylabel("background fraction" if j == 0 else "")
        ax2 = axes[1, j]
        ax2.semilogy(x, rec["signal"] / B, "o-", color=INK, ms=4, lw=1.2)
        ax2.set_xlabel("final-discriminant bin (signal-like →)")
        ax2.set_ylabel("S/B (SM HH)" if j == 0 else "")
        ax2.grid(axis="y", color="#e6e5df", lw=0.6)
    h, l = [], []
    for ax in axes[0]:
        for a, b in zip(*ax.get_legend_handles_labels()):
            if b not in l:
                h.append(a); l.append(b)
    fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    save(fig, "fig1_composition")


def fig_spin_templates():
    rec = load_run2()["hadhad"]
    b = len(rec["signal"]) - 1
    bkg = [p for p in rec if p not in ("signal", "data", "postfit_unc", "_closure")]
    mix = proc_mix({p: None for p in ["signal"] + bkg}, True, {})
    tot = sum(rec[p][b] for p in bkg)
    frac = sum(mix[p] * rec[p][b] / tot for p in bkg)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, arm, title in zip(axes, ("textbook", "tauspin", "exact"),
                              ("textbook observables (Υ, x, hybrid h)", "TauSpin ĥ (IP+SV, ATLAS-calibrated)",
                               "exact polarimeter h (oracle)")):
        st = SpinTemplates(arm, "s0_ct0.5", 6, True)
        sig = rec["signal"]
        cum = np.r_[0, np.cumsum(sig[::-1])][::-1] / sig.sum()
        T, E = st.ditau(frac, (cum[b + 1], cum[b]))
        x = np.arange(1, T.shape[1] + 1)
        names = {"H": "H (signal, single H)", "Z": "Z/γ*→ττ", "W": "W-τ pair (top)",
                 "WU": "one W-τ + fake", "U": "unpolarised (fakes)"}
        styles = {"H": (C[2], "-", "o"), "Z": (C[1], "--", "s"), "W": (C[0], "-.", "^"),
                  "WU": (C[5], ":", "v"), "U": (C[3], (0, (5, 1, 1, 1)), "D")}
        for i, h in enumerate(DI_HYPS):
            c, ls, m = styles[h]
            ax.errorbar(x, T[i], yerr=E[i], color=c, ls=ls, marker=m, ms=5, lw=1.6, capsize=0,
                        label=names[h])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("spin sub-bin of D (signal-quantile, signal-like →)")
        ax.set_ylabel("fraction per hypothesis")
        ax.set_ylim(0, max(0.25, T.max() * 1.15))
    axes[0].legend(fontsize=8)
    fig.suptitle("Spin discriminant templates in the most signal-like τhτh bin (Run-2 legacy bin 14; "
                 "spin-flat HH reweighted, out-of-fold)", fontsize=10)
    save(fig, "fig2_spin_templates")


def fig_lifetime():
    d = json.load(open(OUT / "lifetime" / "lifetime_templates.json"))
    edges = ["0–1", "1–2", "2–3", "3–5", ">5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    v = d["variants"]["nominal"]
    tau = np.array(v["channels"]["slt"][13]["templates"]["nocut"]["tau_signal"])
    pr = np.array(v["pooled"]["nocut"]["prompt"])
    npr = np.array(v["pooled"]["nocut"]["nonprompt"])
    vr = d["variants"]["real"]
    pr_r = np.array(vr["pooled"]["nocut"]["prompt"])
    x = np.arange(5)
    w = 0.2
    for k, (arr, lab, c, hatch) in enumerate([(tau, "τ→ℓ (HH signal, SLT bin 14)", C[2], None),
                                               (pr, "prompt W→ℓ (nominal resolution)", C[0], None),
                                               (pr_r, "prompt W→ℓ (degraded resolution)", C[0], "//"),
                                               (npr, "non-prompt stress (HF-like)", C[3], None)]):
        ax.bar(x + (k - 1.5) * w, arr, width=w * 0.92, color=c if hatch is None else "white",
               edgecolor=c, hatch=hatch, label=lab, linewidth=1.2)
    ax.set_xticks(x, edges)
    ax.set_xlabel("lepton |d₀|/σ(d₀)")
    ax.set_ylabel("fraction")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.legend(fontsize=8)
    ax.set_title("Lepton transverse impact-parameter significance (τℓτh, 14 TeV MC)", fontsize=10)
    ax = axes[1]
    for g, c, m in (("top", C[0], "o"), ("single_h", C[2], "s"), ("fake", C[3], "^")):
        fr = [r["tau_fraction"][g] if (r["tau_fraction"][g] is not None and r["n"][g] >= 50) else np.nan
              for r in v["channels"]["slt"]]
        ax.plot(np.arange(1, 16), fr, marker=m, color=c, lw=1.6, ms=5, label=f"{g} (bins with ≥50 MC events)")
    ax.axhline(0.08, color=C[0], ls=":", lw=1.2)
    ax.text(15.2, 0.08, "nominal top 0.08", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.axhline(0.25, color=C[0], ls="--", lw=1.0)
    ax.text(15.2, 0.25, "variant 0.25", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("SLT final-discriminant bin (mapped by signal fraction)")
    ax.set_ylabel("fraction of leptons from τ decay")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("τ→ℓ share of each background group", fontsize=10)
    save(fig, "fig3_lifetime")


def load_study(name):
    p = OUT / "study" / f"study_{name}.json"
    return json.load(open(p))[name]


def fig_main():
    rows = [("ATLAS HL-LHC baseline (PUB-2025-006)", None, None),
            ("textbook spin observables", "nominal", "textbook"),
            ("TauSpin ĥ (spin only)", "nominal", "tauspin"),
            ("μ lifetime only (|d₀|/σ, muons)", "lt_muonly", "none"),
            ("TauSpin ĥ + μ lifetime", "lt_muonly", "tauspin"),
            ("TauSpin ĥ + e,μ lifetime (optimistic)", "lt_nominal", "tauspin"),
            ("exact-h oracle + μ lifetime", "lt_muonly", "exact")]
    latest = {"nominal": "latest_run3_spinonly", "lt_muonly": "latest_run3_muonly", "lt_nominal": "latest_run3"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, base, title in ((axes[0], "bbtt", 3.54, "bbττ: surrogate gain × ATLAS 3.54σ"),
                                 (axes[1], "comb", 4.26, "HH combination: transferred from ATLAS 4.26σ")):
        for yi, (lab, var, arm) in zip(y, rows):
            if var is None:
                ax.plot([base], [yi], "D", color=INK, ms=7)
                ax.text(base, yi + 0.22, f"{base:.2f}σ", ha="center", fontsize=8, color=INK)
                continue
            v1 = load_study(var)["arms"][arm]["atlas_baseline"][key]
            v2 = load_study(latest[var])["arms"][arm]["atlas_baseline"][key]
            ax.plot([v1], [yi], "o", color=C[0], ms=8, label="gain from Run-2 legacy composition" if yi == y[1] else None)
            ax.plot([v2], [yi], "s", color=C[1], ms=7, mfc="white", mew=1.8,
                    label="gain from latest Run-3 composition" if yi == y[1] else None)
            ax.text(max(v1, v2) + 0.015, yi, f"{v1:.2f} / {v2:.2f}", va="center", fontsize=8, color=INK)
        ax.axvline(base, color=MUTED, lw=0.8, ls=":")
        if key == "comb":
            for val, lab in ((4.34, "ATLAS: τ-ID +5%"), (4.44, "b-tag +5%"), (4.52, "both")):
                ax.axvline(val, color="#b9b8b0", lw=1.0, ls="--")
                ax.text(val + 0.004, y[-1] - 0.45, lab, rotation=90, fontsize=7, color=MUTED, ha="left", va="bottom")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("expected SM HH significance [σ]")
        ax.grid(axis="x", color="#eeede8", lw=0.6)
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].set_xlim(3.45, 4.1)
    axes[1].set_xlim(4.2, 4.7)
    axes[0].set_ylim(y[-1] - 0.6, y[0] + 0.6)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.text(0.5, -0.04, "Gain ratios R from the HEPData/figure-based likelihood (baseline syst. calibrated to PUB-2024-016) "
             "applied to ATLAS bbττ 3.54σ; combination via Z² + c(ΔZ_bbττ²), c = 0.88. Dashed: ATLAS's own "
             "algorithmic-improvement scenarios (PUB-2025-006 Table 9).", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig4_main_result")


def fig_robustness():
    spin_vars = [("nominal", "nominal"), ("stat_only", "statistics only"), ("spin_shape25", "25% spin-response contrast unc."),
                 ("nsub4", "4 spin sub-bins"), ("nsub10", "10 spin sub-bins"), ("no_condK", "no K conditioning"),
                 ("Z_cT0", "Z transverse corr. = 0"), ("singleH_30pctZ", "30% of single H with Z spin"),
                 ("fakes_Hlike", "fakes H-like spin"), ("fakes_Wlike", "fakes W-like spin"),
                 ("latest_run3_spinonly", "latest Run-3 composition"), ("latest_run2_spinonly", "latest Run-2 composition")]
    lt_vars = [("lt_muonly", "μ only (nominal)"), ("lt_muonly_real", "μ: degraded d₀ resolution"),
               ("lt_muonly_top25", "μ: top & tt̄-fake τ→ℓ share 25%"), ("lt_muonly_np30", "μ: 30% non-prompt fakes"),
               ("lt_muonly_noLHshape", "μ: no lep-had shape syst."), ("lt_muonly_LHshape08", "μ: stronger lep-had shape syst."),
               ("lt_muonly_binsys5", "μ: +5% per-bin bkg syst. (sub-bins as in-situ control)"), ("lt_muonly_worst", "μ: worst combination + 25% spin unc."),
               ("latest_run3_muonly", "μ: latest Run-3 composition"), ("latest_run2_muonly", "μ: latest Run-2 composition"),
               ("lt_nominal", "e+μ, no cut (optimistic)"), ("lt_ecut5", "e+μ, electrons |d₀/σ|<5"),
               ("lt_ttva", "e+μ, TTVA-shape truncation stress"), ("lt_stat", "e+μ, statistics only")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"wspace": 0.75})
    for ax, vars_, arms, title in ((axes[0], spin_vars, (("textbook", C[3], "^"), ("tauspin", C[0], "o"), ("exact", C[2], "s")), "spin only (τhτh and τh side of τℓτh)"),
                                   (axes[1], lt_vars, (("none", C[5], "v"), ("tauspin", C[0], "o"), ("exact", C[2], "s")), "with lepton lifetime in τℓτh")):
        y = np.arange(len(vars_))[::-1]
        for yi, (v, lab) in zip(y, vars_):
            if not (OUT / "study" / f"study_{v}.json").exists():
                ax.text(0, yi, "  (not run)", fontsize=7, color=MUTED, va="center")
                continue
            d = load_study(v)
            for arm, c, m in arms:
                if arm in d["arms"]:
                    r = d["arms"][arm]["R_mean"]["combined"] - 1
                    ax.plot(100 * r, yi, m, color=c, ms=7, mfc=c if arm != "exact" else "white", mew=1.6,
                            label={"none": "lifetime only", "textbook": "textbook spin", "tauspin": "TauSpin ĥ",
                                   "exact": "exact h"}[arm] if yi == y[0] else None)
        ax.set_yticks(y, [l for _, l in vars_])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("bbττ significance gain R − 1 [%]")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color="#eeede8", lw=0.6)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, "fig5_robustness")


def fig_where():
    """Statistics-only Asimov Z^2 contribution of every baseline bin, with and without the
    extra information (no profiling; illustrative of where the gain arises)."""
    cfg = {**DEFAULT, "lifetime": {"cut": "muonly"}}
    spin = SpinTemplates("tauspin", "s0_ct0.5", 6, True)
    base_spec, _ = build_spec({**DEFAULT}, None)
    new_spec, _ = build_spec(cfg, spin)

    def z2(spec, ch):
        c = [x for x in spec["channels"] if x["name"] == ch][0]
        s = np.array([x for x in c["samples"] if x["name"] == "signal"][0]["data"])
        b = sum(np.array(x["data"]) for x in c["samples"] if x["name"] != "signal")
        t = 2 * ((s + b) * np.log1p(s / np.maximum(b, 1e-12)) - s)
        nb = len(load_run2()[ch]["signal"])
        return t.reshape(nb, -1).sum(1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ch, title in ((axes[0], "hadhad", "τhτh (Run-2 legacy bins)"), (axes[1], "slt", "τℓτh SLT (Run-2 legacy bins)")):
        a, b = z2(base_spec, ch), z2(new_spec, ch)
        x = np.arange(1, len(a) + 1)
        ax.bar(x - 0.2, a, 0.38, color="#b9b8b0", label="baseline")
        ax.bar(x + 0.2, b, 0.38, color=C[0], label="+ TauSpin ĥ (+ μ |d₀|/σ in τℓτh)")
        ax.set_xlabel("final-discriminant bin (signal-like →)")
        ax.set_ylabel("Z² contribution (stat. only, 3 ab⁻¹)")
        ax.set_title(f"{title}: ΣZ² {a.sum():.2f} → {b.sum():.2f} (known bkg, no profiling)", fontsize=10)
        ax.legend(fontsize=8)
    save(fig, "fig6_where_gain")


def fig_validation():
    lat = json.load(open(OUT / "latest" / "latest_bins.json"))
    tab = {  # tabaux01 / tabaux02 (process totals per SR); ttbar includes lepton fakes
        ("run2", "hadhad", "Lo"): {"fake_mj": 970, "ttbar+tW+other": 722 + 497 + 21 + 78, "fake_tt": 1352, "zhf": 515, "single_h": 23.2},
        ("run3", "hadhad", "Lo"): {"fake_mj": 5670, "ttbar+tW+other": 666 + 564 + 20 + 88, "fake_tt": 2581, "zhf": 379, "single_h": 21.9},
        ("run2", "hadhad", "Hi"): {"fake_mj": 1550, "ttbar+tW+other": 1343 + 713 + 96 + 150, "fake_tt": 1338, "zhf": 753, "single_h": 57},
        ("run3", "hadhad", "Hi"): {"fake_mj": 1153, "ttbar+tW+other": 563 + 329 + 34 + 76, "fake_tt": 855, "zhf": 346, "single_h": 26.7},
        ("run2", "hadhad", "VBF"): {"fake_mj": 382, "ttbar+tW+other": 206 + 93.9 + 6.8 + 24, "fake_tt": 194, "zhf": 52.4, "single_h": 3.8},
        ("run3", "hadhad", "VBF"): {"fake_mj": 973, "ttbar+tW+other": 114 + 69 + 2.6 + 18.6, "fake_tt": 269, "zhf": 32.3, "single_h": 2.58},
        ("run2", "lephad", "Hi"): {"fake": 16990, "ttbar+tW+other": 21859 + 507 + 1360 + 267, "zhf": 739, "single_h": 91},
        ("run3", "lephad", "Hi"): {"fake": 9270, "ttbar+tW+other": 8847 + 204.4 + 460 + 125, "zhf": 310, "single_h": 40.5},
        ("run2", "lephad", "Lo"): {"fake": 32900, "ttbar+tW+other": 29485 + 649 + 1210 + 329, "zhf": 1400, "single_h": 81.8},
        ("run3", "lephad", "Lo"): {"fake": 17310, "ttbar+tW+other": 11640 + 261 + 390 + 160, "zhf": 606, "single_h": 34.9},
        ("run2", "lephad", "VBF"): {"fake": 6770, "ttbar+tW+other": 9804 + 218 + 460 + 81.7, "zhf": 193, "single_h": 10.9},
        ("run3", "lephad", "VBF"): {"fake": 3940, "ttbar+tW+other": 4066 + 85.7 + 160 + 37.1, "zhf": 82, "single_h": 5.14},
    }
    sig_tab = {("run2", "Hi", "hadhad"): 16.3, ("run3", "Hi", "hadhad"): 7.4, ("run2", "Hi", "lephad"): 14.8,
               ("run3", "Hi", "lephad"): 6.1}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    markers = {"fake_mj": ("o", C[3]), "fake": ("o", C[3]), "fake_tt": ("^", C[5]), "zhf": ("s", C[1]),
               "single_h": ("D", C[2]), "ttbar+tW+other": ("v", C[0])}
    labels_done = set()
    for i, ((run, ch, sr), t) in enumerate(tab.items()):
        y = {k: sum(v) for k, v in lat[f"{run}_{ch}_{sr}"]["yields"].items()}
        y["ttbar+tW+other"] = y["ttbar"] + y["tW"] + y["other"]
        for k, ref in t.items():
            m, c = markers[k]
            lab = {"fake_mj": "jet→τ fake (MJ / lep-had)", "fake": "jet→τ fake (MJ / lep-had)",
                   "zhf": "Z+HF", "single_h": "single H", "fake_tt": "jet→τ fake (tt̄)",
                   "ttbar+tW+other": "tt̄ + tW + other"}.get(k, k)
            ax.plot(i, y[k] / ref, m, color=c, ms=6, label=lab if lab not in labels_done else None)
            labels_done.add(lab)
    for (run, sr, ch), ref in sig_tab.items():
        ggf = sum(lat[f"{run}_{ch}_{sr}"]["yields"]["ggF_x20"]) / 20 * 2.59
        i = list(tab).index((run, ch, sr))
        ax.plot(i, ggf / ref, "*", color=INK, ms=10, label="ggF HH (×20 line / 20 × 2.59)" if "sig" not in labels_done else None)
        labels_done.add("sig")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(tab)), [f"{r} {c} {s}" for r, c, s in tab], rotation=60, fontsize=7)
    ax.set_ylabel("figure extraction / auxiliary table")
    ax.set_ylim(0.97, 1.03)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Latest ATLAS (2607.26879): recovered bin yields — region-total closure only", fontsize=10)
    ax = axes[1]
    cal = json.load(open(OUT / "calib" / "calibration_v2.json"))
    rows = [r for r in cal["scan"] if r["lephad_shape"] == 0.6 and np.isfinite(r["ratio_combined"])]
    s = [r["single_h"] for r in rows]
    for key, c, m, lab, tgt in (("ratio_combined", C[0], "o", "bbττ", 3.5 / 4.6), ("ratio_hadhad", C[1], "s", "τhτh", 3.1 / 4.0),
                                ("ratio_lephad", C[2], "^", "τℓτh", 1.8 / 2.3)):
        r = [x[key] for x in rows]
        ax.plot(s, r, marker=m, color=c, lw=1.6, label=f"{lab} (ATLAS {tgt:.3f})")
        ax.axhline(tgt, color=c, ls=":", lw=1.0)
    ax.axvline(0.18, color=MUTED, ls="--", lw=1.0)
    ax.text(0.181, 0.84, "chosen 18%", fontsize=8, color=MUTED)
    ax.set_xlabel("single-Higgs norm. unc. (correlated); lep-had shape tilt 0.6")
    ax.set_ylabel("Z(baseline syst.) / Z(stat. only)")
    ax.legend(fontsize=8)
    ax.set_title("Systematic model calibrated to ATL-PHYS-PUB-2024-016", fontsize=10)
    save(fig, "fig7_validation")


if __name__ == "__main__":
    for f in (fig_composition, fig_spin_templates, fig_lifetime, fig_main, fig_robustness, fig_where, fig_validation):
        f()
        print("done", f.__name__, flush=True)
