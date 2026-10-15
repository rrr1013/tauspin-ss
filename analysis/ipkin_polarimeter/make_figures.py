"""Figures of the classical-posterior vs learned polarimeter comparison.

  fig1_sensitivity.png   entanglement significance vs background-shape knowledge, and the
                         expected 68 % interval of the CP angle, for each reconstruction level
  fig2_modepairs.png     corr(g_T, exact T) per reco decay-mode pair
  fig3_complement.png    learned vs classical g_T event by event, and corr(g_T, T) vs the
                         smallest 1-prong transverse impact parameter
  fig5_ipsv_gain.png     what IP/SV add per reco mode pair: corr(g_T, T) without -> with IP/SV for the
                         learned regressor (10/07 no-IP/SV retraining) and the classical posterior
  fig4_singleside.png    per-side closure of h (corr with exact) per reco class, split by
                         whether the reco decay class is the true one: learned, classical
                         posterior, true-neutrino ceiling
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"tauspin": "#2a78d6", "ipkin": "#eb6834", "ipkin_m125": "#eb6834", "tauspin+ipkin": "#1baf7a",
     "tauspin+ipkin_m125": "#1baf7a", "phicp": "#52514e", "phicp_zmf": "#52514e", "ipkin_noip": "#e87ba4", "textbook_pair": "#9a9993",
     "exact": "#0b0b0b", "oracle": "#0b0b0b"}
LAB = {"exact": "exact h (reference)", "tauspin": "learned ĥ (tauspin, IP+SV)",
       "ipkin": "classical posterior (IP+SV+MET)", "ipkin_m125": "classical posterior + m_ττ=125 prior",
       "tauspin+ipkin": "learned + classical", "tauspin+ipkin_m125": "learned + classical (m_ττ prior)",
       "ipkin_noip": "classical posterior, no IP/SV", "phicp": "LHC CP directions (lab frame)",
       "phicp_zmf": "LHC CP observables (φ*_CP ZMF, 10/07 baseline)",
       "textbook_pair": "textbook observables"}
MK = {"phicp_zmf": "X", "tauspin": "o", "ipkin": "s", "ipkin_m125": "D", "tauspin+ipkin": "^", "tauspin+ipkin_m125": "v",
      "phicp": "X", "ipkin_noip": "P", "textbook_pair": "h", "exact": "*"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})


def fig1(aux, cp, out):
    levels = ["exact", "tauspin+ipkin_m125", "tauspin+ipkin", "tauspin", "ipkin_m125", "ipkin", "phicp_zmf", "phicp",
              "ipkin_noip", "textbook_pair"]
    levels = [l for l in levels if l in aux["levels"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
    taus = [("tau0", "sidebands only"), ("tau1", "aux. sample τ=1"), ("tau10", "τ=10"), ("tau100", "shape known")]
    y = np.arange(len(levels))[::-1]
    for j, (k, nm) in enumerate(taus):
        z = [aux["levels"][l].get(k, {}).get("Z") for l in levels]
        z = [np.nan if v is None else v for v in z]
        ax[0].scatter(z, y + (j - 1.5) * 0.17, s=26, marker="osD^"[j], color=["#0b0b0b", "#52514e", "#9a9993", "#c3c2b7"][j],
                      label=nm, zorder=3)
    ax[0].set_yticks(y, [LAB[l] for l in levels])
    for t, l in zip(ax[0].get_yticklabels(), levels):
        t.set_color(C.get(l, "#0b0b0b") if l in ("tauspin", "ipkin", "ipkin_m125", "tauspin+ipkin", "tauspin+ipkin_m125") else "#0b0b0b")
    ax[0].axvline(3, color="#9a9993", lw=0.8, ls="--")
    ax[0].set_xlabel("expected entanglement significance [σ]\n(null: all separable Bell-diagonal states)")
    ax[0].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="lower right")
    ax[0].set_title("Entanglement test, 3 ab⁻¹, τhτh", fontsize=10, loc="left")
    for j, (k, nm) in enumerate((("nominal", "sidebands only"), ("known_bkg_shape", "shape known"))):
        v = [cp["levels"][l][k]["phi68_deg"] for l in levels]
        for yy, vv, l in zip(y, v, levels):
            if vv is None:
                q = cp["levels"][l][k]["q"]
                txt = "< 5°" if all(np.isfinite(list(q.values()))) else "fit failed"
                ax[1].text(1.0 if txt == "< 5°" else 31, yy + (j - 0.5) * 0.25, f"{nm}: {txt}", fontsize=7, color="#52514e",
                           va="center", ha="left" if txt == "< 5°" else "right")
        ax[1].scatter([np.nan if vv is None else vv for vv in v], y + (j - 0.5) * 0.25, s=30, marker="oD"[j],
                      color=["#0b0b0b", "#9a9993"][j], label=nm, zorder=3)
    ax[1].set_yticks(y, [""] * len(levels))
    ax[1].set_xlim(0, 32)
    ax[1].set_xlabel("expected 68% half-width of φ_τ [deg]\n(smaller is better)")
    ax[1].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="center right")
    ax[1].set_title("τ Yukawa CP angle", fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Spin-flat H(125)+jet, parametric HL-LHC response, Asimov; same cross-fitted (g_T, g_L)/(g_T, g_A) "
             "regression and fit for every row (10/07 framework). Not an ATLAS result.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160)


def fig2(bd, out):
    pairs = [k for k in bd["pairs"]]
    levels = ["tauspin", "ipkin", "tauspin+ipkin", "phicp_zmf"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(pairs))[::-1]
    for j, l in enumerate(levels):
        v = [bd["pairs"][p][l]["corr"] for p in pairs]
        ax.scatter(v, y + (j - 1.5) * 0.16, s=30, marker=MK[l], color=C[l], label=LAB[l], zorder=3)
    ax.set_yticks(y, [f"{p}  ({100 * bd['pairs'][p]['wfrac']:.0f}%)" for p in pairs])
    ax.set_xlabel("corr( g_T , exact h⁻_n h⁺_n + h⁻_r h⁺_r )   per event, spin-flat sample")
    ax.set_title("Transverse spin statistic by reconstructed decay-mode pair (share of events)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def fig3(cache, bd, out):
    z = np.load(cache, allow_pickle=True)
    a, b, w = z["G_tauspin"][:, 0], z["G_ipkin"][:, 0], z["w"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    lim = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
    hb = ax[0].hist2d(a, b, bins=70, range=[lim, lim], weights=w, cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax[0].plot(lim, lim, color="#52514e", lw=0.8, ls="--")
    ax[0].set_xlabel("g_T from learned ĥ")
    ax[0].set_ylabel("g_T from classical posterior")
    ax[0].set_title(f"event by event (weighted corr {bd['all']['corr_tauspin_ipkin']:.2f})", fontsize=10, loc="left")
    fig.colorbar(hb[3], ax=ax[0], label="events (weighted, log)")
    keys = list(bd["ipsig"])
    x = np.arange(len(keys))
    for l in ("tauspin", "ipkin", "tauspin+ipkin", "phicp_zmf"):
        ax[1].plot(x, [bd["ipsig"][k][l] for k in keys], marker=MK[l], color=C[l], lw=2, ms=7, label=LAB[l])
    ax[1].set_xticks(x, [k.replace("um", "").replace("80-1e+09", "≥80") for k in keys])
    ax[1].set_xlabel("smallest 1-prong |b| projected ⊥ visible τ [μm]")
    ax[1].set_ylabel("corr( g_T , exact T )")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("dependence on the IP lever arm (≥1 one-prong side)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def fig4(mig, out):
    classes = ["pi", "rho", "a1(1p)", "a1(3p)"]
    rows = [(c, j) for c in classes for j in range(3)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    series = [("tauspin", "o"), ("ipkin_m125", "D"), ("ipkin", "s"), ("oracle", "_")]
    lab = dict(LAB, oracle="true neutrinos + reco visibles, reco-mode polarimeter")
    x = np.arange(len(rows))
    for k, (nm, mk) in enumerate(series):
        v = [mig[c][nm]["correct"][j] for c, j in rows]
        ax[0].scatter(x + (k - 1.5) * 0.17, v, s=30 if mk != "_" else 90, marker=mk, color=C[nm], label=lab[nm],
                      zorder=3, linewidths=2)
    ax[0].set_xticks(x, [f"{'nrk'[j]}\n{c}" if j == 1 else "nrk"[j] for c, j in rows])
    ax[0].set_ylabel("corr( h component , exact )  per τ side")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("sides whose reco decay class is correct", fontsize=10, loc="left")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    rows2 = [(c, j) for c in ("pi", "rho") for j in range(3)]
    x2 = np.arange(len(rows2))
    for k, (nm, mk) in enumerate(series):
        v = [mig[c][nm]["migrated"][j] for c, j in rows2]
        ax[1].scatter(x2 + (k - 1.5) * 0.17, v, s=30 if mk != "_" else 90, marker=mk, color=C[nm], zorder=3, linewidths=2)
    ax[1].set_xticks(x2, [f"{'nrk'[j]}\n{c} ({100 * mig[c]['frac_migrated']:.0f}%)" if j == 1 else "nrk"[j] for c, j in rows2])
    ax[1].set_ylim(-0.15, 1.02)
    ax[1].axhline(0, color="#9a9993", lw=0.8)
    ax[1].set_title("sides with a migrated decay class (share)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def fig5(rc, extra, out):
    pairs = ["pi-pi", "pi-rho", "pi-a1(3p)", "rho-rho", "rho-a1(1p)", "rho-a1(3p)", "a1(1p)-a1(3p)", "a1(3p)-a1(3p)"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(pairs))[::-1]
    rows = [("tauspin_noIPSV", "tauspin", C["tauspin"], "o", "learned: without → with IP/SV", +0.13),
            ("ipkin_noip", "ipkin", C["ipkin"], "s", "classical: without → with IP/SV", -0.13)]
    for lo, hi, col, mk, nm, dy in rows:
        a = [rc["noipsv"][p][lo] for p in pairs]
        b = [rc["noipsv"][p][hi] for p in pairs]
        for yy, x0, x1 in zip(y + dy, a, b):
            ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.scatter(a, y + dy, s=22, marker=mk, facecolors="white", edgecolors=col, zorder=3)
        ax.scatter(b, y + dy, s=34, marker=mk, color=col, zorder=3, label=nm)
    if extra and "ipkin_pi0" in extra:
        ax.scatter([extra["ipkin_pi0"]["pairs"][p] for p in pairs], y - 0.13, s=40, marker="|", color="#0b0b0b",
                   zorder=4, label="classical with π⁰ response marginalised")
    ax.set_yticks(y, pairs)
    ax.axvline(0, color="#9a9993", lw=0.8)
    ax.set_xlabel("corr( g_T , exact T )  per event, spin-flat sample")
    ax.set_title("What the impact parameters / secondary vertex add, by reco decay-mode pair", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = a.dir
    j = lambda n: json.load(open(f"{D}/{n}"))
    bd = j("breakdown.json")
    aux, cp = j("ent_robust_aux.json"), j("ent_cp.json")
    for name, key in (("ent_robust_aux_zmf.json", "levels"), ("ent_cp_zmf.json", "levels")):
        try:
            extra = j(name)[key]
        except FileNotFoundError:
            continue
        (aux if "robust" in name else cp)["levels"].update(extra)
    fig1(aux, cp, f"{D}/fig1_sensitivity.png")
    fig2(bd, f"{D}/fig2_modepairs.png")
    fig3(f"{D}/ent_robust_G.npz", bd, f"{D}/fig3_complement.png")
    fig4(j("migration.json"), f"{D}/fig4_singleside.png")
    try:
        extra = j("extra_levels.json")
    except FileNotFoundError:
        extra = None
    fig5(j("review_checks.json"), extra, f"{D}/fig5_ipsv_gain.png")


if __name__ == "__main__":
    main()
