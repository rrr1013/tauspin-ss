"""Figures of the synthesis run (reads results/*.json; a few literature numbers of
earlier runs are typed in with their source run in the comment)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 160})
PAIRS = ["pi x pi", "pi x rho", "rho x rho", "pi x 3pi", "rho x 3pi", "3pi x 3pi"]


def load(name):
    return json.loads((RES / name).read_text())


def fig_ladder():
    """Where the AUC goes: learned readouts (weighted, validation 59,390) and solution-set
    ceilings (unweighted, truth-visible + exact MET unless stated)."""
    x1 = load("x1_auc.json")["auc"]
    fin = load("x10_final_auc.json")["auc"] if (RES / "x10_final_auc.json").exists() else {}
    tr = load("x4_truth_ipg_unw.json")["auc"]
    rr = load("x4_reco_ipg_w.json")["auc"]
    learned = [
        ("direct H/Z classifier, raw reco (09-06)", 0.5971),                    # reco-h-supervision
        ("point h, current reco, no PV-IP (09-20/23)", x1["base_hard"]),
        ("point h + PV-IP lead track (09-23)", 0.6250),                         # beyond-ceiling P9
        ("point h + all IP + SV, 2 seeds (09-23)", x1["f22ens_hard"]),
    ]
    if "best" in fin:
        learned.append(("point h + all IP + SV, 4-seed ensemble (this run)", fin["best"]))
    learned += [
        ("same, IP direction replaced by truth (ideal IP)", x1["ideal_hard"]),
        ("same, truth tau direction in local frame (09-23)", 0.6864),          # beyond-ceiling oracle
        ("exact truth h, analytic LR", x1["exact_h_LR"]),
    ]
    ceilings = [
        ("reco vis + reco MET, flat (this run)", rr["flat"]),
        ("reco vis + reco MET, full IP + SV + m (this run)", rr["ipg_sv_mass"]),
        ("truth vis + exact MET, flat (09-19)", tr["flat"]),
        ("  + IP azimuth, 3 tracks (09-23)", tr["ip3"]),
        ("  + IP azimuth + SV + m_parent (09-23)", tr["ip3_sv_mass"]),
        ("  + full IP likelihood (this run)", tr["ipg"]),
        ("  + full IP + SV + m_parent (this run)", tr["ipg_sv_mass"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    rows = [(n, v, C1, "o") for n, v in learned] + [(None, None, None, None)] + [(n, v, C2, "s") for n, v in ceilings]
    y = 0
    labels, ys = [], []
    for n, v, c, m in rows[::-1]:
        if n is None:
            y += 0.6
            continue
        ax.plot([0.5, v], [y, y], color=c, lw=1.2, alpha=0.35)
        ax.plot(v, y, m, color=c, ms=7, mec="white", mew=1.2)
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8, color="#333")
        labels.append(n); ys.append(y)
        y += 1
    ax.set_yticks(ys, labels)
    ax.set_xlim(0.55, 0.74)
    ax.set_xlabel("H/Z AUC")
    ax.plot([], [], "o", color=C1, label="learned point-h + fixed readout (weighted, reco inputs)")
    ax.plot([], [], "s", color=C2, label="likelihood ratio on the kinematic solution set (unweighted)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.1), frameon=False, fontsize=8, ncol=1)
    ax.set_title("Where the H/Z spin AUC is lost (validation; test never opened)", fontsize=9, loc="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_a_auc_ladder.png")
    plt.close(fig)


def fig_stages():
    d = load("x2_all_common_w.json")
    stages = [("truth", "truth_exact"), ("truth, 2 GeV MET tol.", "truth_exact2"), ("pi sides reco", "recopi_exact2"), ("3pi sides reco", "reco3pi_exact2"),
              ("rho sides reco", "recorho_exact2"), ("all visible reco", "reco_exact2"),
              ("MET reco only", "truth_recomet"), ("visible + MET reco", "reco_recomet")]
    panels = [("all pairs", None)] + [(p, p) for p in PAIRS]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey=False)
    for ax, (title, pair) in zip(axes.ravel(), panels):
        src = d["auc"] if pair is None else d["by_pair"][pair]
        x = np.arange(len(stages))
        for meas, c, m, lab in (("flat", C1, "o", "flat measure (no vertex info)"),
                                ("ip3_sv", C2, "s", "IP azimuth + SV likelihood")):
            vals = [src[f"{k}:{meas}"] for _, k in stages]
            ax.axhline(vals[0], color=c, lw=0.8, ls=":", alpha=0.8)
            ax.plot(x, vals, m, color=c, ms=6, mec="white", mew=1, ls="none", label=lab)
        ax.axhline(src["f22ens_hard"], color=GRAY, ls="--", lw=1, label="learned point h + IP/SV (same rows)")
        ax.set_title(f"{title}" + ("" if pair is None else f"  (n={d['by_pair'][pair]['events']:,})"), fontsize=9, loc="left")
        ax.set_xticks(x, [s for s, _ in stages], rotation=55, ha="right", fontsize=7.5)
        ax.set_ylabel("H/Z AUC (weighted)")
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center", frameon=False, fontsize=8.5)
    fig.suptitle(f"Truth -> reco, one stage at a time (each point swaps ONE ingredient from truth to reco; dotted = all truth). "
                 f"Solution-set likelihood ratio, common cohort n={d['events']:,}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b_truth_to_reco_stages.png")
    plt.close(fig)


def fig_ip():
    s6 = load("s6_ip_resolution.json")
    tr = load("x4_truth_ipg_unw.json")
    rr = load("x4_reco_ipg_w.json")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    ax = axes[0]
    pe = np.array(s6["bins_pt"]); centres = np.sqrt(np.maximum(pe[:-1], 2.5) * np.minimum(pe[1:], 160))
    for k, (c, ls) in enumerate(((C1, "-"), (C2, "--"), (C3, ":"))):
        eb = s6["bins_eta"]
        ax.plot(centres, [r[k] for r in s6["sigma_T_um"]], ls, color=c, marker="o", ms=4, label=f"transverse, |eta| {eb[k]}-{eb[k+1]}")
        ax.plot(centres, [r[k] for r in s6["sigma_L_um"]], ls, color=c, marker="^", ms=4, mfc="white", label=f"longitudinal, |eta| {eb[k]}-{eb[k+1]}")
    ax.axhline(67, color=GRAY, lw=1, ls="-.")
    ax.text(8, 69, "median expected true |IP| (67 um)", fontsize=7.5, color="#555")
    ax.set_xscale("log"); ax.set_xlabel("track pT [GeV]"); ax.set_ylabel("IP error [um] (TRAIN, track+PV)")
    ax.set_title("(a) IP error is anisotropic", fontsize=9, loc="left")
    ax.legend(fontsize=6.3, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    meas = [("flat", "flat"), ("SV", "sv"), ("IP az.", "ip3"), ("IP az.\n+SV", "ip3_sv"),
            ("full IP", "ipg"), ("full IP\n+SV+m", "ipg_sv_mass")]
    for ax, src, title in ((axes[1], tr, "(b) truth visible + exact MET"), (axes[2], rr, "(c) reco visible + reco MET")):
        auc = src["auc"]
        x = np.arange(len(meas))
        vals = [auc.get(k, np.nan) for _, k in meas]
        ax.bar(x, np.array(vals) - 0.5, bottom=0.5, color=[GRAY, C3, C1, C1, C2, C2], width=0.62,
               edgecolor="white", linewidth=1)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.001, f"{v:.3f}", ha="center", fontsize=7.5)
        sh = auc.get("ipg_shuffle")
        if sh:
            ax.axhline(sh, color=GRAY, ls=":", lw=1)
            ax.text(len(meas) - 0.5, sh + 0.001, "full IP, shuffled", ha="right", fontsize=7, color="#555")
        ax.set_xticks(x, [m for m, _ in meas], fontsize=7.5)
        lo = min(v for v in vals if np.isfinite(v))
        ax.set_ylim(lo - 0.02, max(vals) + 0.012)
        ax.set_ylabel("H/Z AUC")
        ax.set_title(title + f"  (n={src['events']:,})", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_ip_information.png")
    plt.close(fig)


def fig_scales():
    """Angular scales per decay mode (validation; sources in the note)."""
    modes = ["pi (1p0n)", "rho (1p1n)", "3pi (3p0n)"]
    rows = [
        ("tau-visible cone angle (median)", [7.49, 5.02, 3.06], C1, "o"),
        ("separation of the two mass-shell roots (median)", [8.33, 5.66, 3.48], C1, "D"),
        ("reco visible-axis error (median)", [0.21, 2.52, 0.24], C2, "s"),
        ("reco visible-axis error (90%)", [0.92, 7.64, 0.81], C2, "v"),
        ("tau direction from PV-IP (median, 1-prong lead track)", [1.97, 1.97, np.nan], C3, "^"),
        ("tau direction from SV (core width)", [np.nan, np.nan, 4.55], C3, "P"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, (lab, vals, c, m) in enumerate(rows):
        off = (i - 2.5) * 0.11
        ax.plot(np.array(vals), np.arange(3) + off, m, color=c, ms=7, mec="white", mew=1, ls="none", label=lab)
    ax.set_yticks(range(3), modes)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("angle [mrad]")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("The quantity to solve (blue) vs reco errors (orange) vs vertex handles (green)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_angular_scales.png")
    plt.close(fig)


def fig_requirement():
    d = load("x9_ipsim_unw.json")
    scales = [1.0, 0.7, 0.5, 0.25]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    auc = d["auc"]
    ax.plot(scales, [auc[f"sim{f}_ipg"] for f in scales], "-o", color=C2, ms=6, mec="white", label="full IP likelihood, simulated IP")
    ax.plot(scales, [auc[f"sim{f}_ipgsvm"] for f in scales], "--s", color=C1, ms=6, mec="white", label="+ SV + parent mass")
    ax.plot([1.0], [auc["real_ipg"]], "D", color="#333", ms=7, label="full IP, measured IP (closure)")
    ax.axhline(auc["real_flat"], color=GRAY, ls=":", lw=1)
    ax.text(0.26, auc["real_flat"] + 0.002, "no vertex info", fontsize=7.5, color="#555")
    ax.axhline(auc["exact_h_LR"], color=GRAY, ls="-.", lw=1)
    ax.text(0.26, auc["exact_h_LR"] - 0.006, "exact h", fontsize=7.5, color="#555")
    ax.invert_xaxis()
    ax.set_xlabel("IP error scale (x current track+PV error)")
    ax.set_ylabel("H/Z AUC (unweighted)")
    ax.set_title("(a) all pairs, truth visible + exact MET", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax = axes[1]
    for pair, c, m in (("pi x pi", C1, "o"), ("rho x rho", C2, "s"), ("3pi x 3pi", C3, "^"), ("pi x 3pi", "#4a3aa7", "D")):
        v = d["by_pair"][pair]
        ax.plot(scales, [v[f"sim{f}_ipg"] for f in scales], "-", marker=m, color=c, ms=5.5, mec="white", label=f"{pair} (n={v['events']:,})")
    ax.invert_xaxis()
    ax.set_xlabel("IP error scale")
    ax.set_title("(b) by decay-mode pair (full IP, simulated)", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_e_ip_requirement.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_requirement()
    fig_ladder()
    fig_stages()
    fig_ip()
    fig_scales()
    print(sorted(p.name for p in FIG.glob("*.png")))
