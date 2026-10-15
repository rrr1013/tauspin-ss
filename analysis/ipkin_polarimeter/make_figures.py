"""Figures of the classical-posterior vs learned polarimeter comparison.

  fig1_sensitivity.png   entanglement significance vs background-shape knowledge, and the
                         expected 68 % interval of the CP angle, for each reconstruction level
  fig2_modepairs.png     corr(g_T, exact T) per reco decay-mode pair
  fig3_complement.png    learned vs classical g_T event by event, and corr(g_T, T) vs the
                         smallest 1-prong transverse impact parameter
  fig4_singleside.png    per-side closure of h (corr with exact) per reco class: learned,
                         classical posterior, true-neutrino ceiling
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"tauspin": "#2a78d6", "ipkin": "#eb6834", "ipkin_m125": "#eb6834", "tauspin+ipkin": "#1baf7a",
     "tauspin+ipkin_m125": "#1baf7a", "phicp": "#52514e", "ipkin_noip": "#e87ba4", "textbook_pair": "#9a9993",
     "exact": "#0b0b0b", "oracle": "#0b0b0b"}
LAB = {"exact": "exact h (reference)", "tauspin": "learned ĥ (tauspin, IP+SV)",
       "ipkin": "classical posterior (IP+SV+MET)", "ipkin_m125": "classical posterior + m_ττ=125 prior",
       "tauspin+ipkin": "learned + classical", "tauspin+ipkin_m125": "learned + classical (m_ττ prior)",
       "ipkin_noip": "classical posterior, no IP/SV", "phicp": "LHC CP observables (φ*_CP set)",
       "textbook_pair": "textbook observables"}
MK = {"tauspin": "o", "ipkin": "s", "ipkin_m125": "D", "tauspin+ipkin": "^", "tauspin+ipkin_m125": "v",
      "phicp": "X", "ipkin_noip": "P", "textbook_pair": "h", "exact": "*"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})


def fig1(rob, aux, cp, out):
    levels = ["exact", "tauspin+ipkin_m125", "tauspin+ipkin", "tauspin", "ipkin_m125", "ipkin", "phicp",
              "ipkin_noip", "textbook_pair"]
    levels = [l for l in levels if l in aux["levels"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    taus = [("tau0", "sidebands only"), ("tau1", "aux. sample τ=1"), ("tau10", "τ=10"), ("tau100", "shape known")]
    y = np.arange(len(levels))[::-1]
    for j, (k, nm) in enumerate(taus):
        z = [aux["levels"][l][k]["Z"] for l in levels]
        ax[0].scatter(z, y + (j - 1.5) * 0.17, s=26, marker="osD^"[j], color=["#0b0b0b", "#52514e", "#9a9993", "#c3c2b7"][j],
                      label=nm, zorder=3)
    ax[0].set_yticks(y, [LAB[l] for l in levels])
    ax[0].axvline(3, color="#9a9993", lw=0.8, ls="--")
    ax[0].set_xlabel("expected entanglement significance [σ]  (separable-state null)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title("Entanglement test, 3 ab⁻¹, τhτh", fontsize=10, loc="left")
    for j, (k, nm) in enumerate((("nominal", "sidebands only"), ("known_bkg_shape", "shape known"))):
        v = [cp["levels"][l][k]["phi68_deg"] for l in levels]
        ax[1].scatter(v, y + (j - 0.5) * 0.25, s=30, marker="oD"[j], color=["#0b0b0b", "#9a9993"][j], label=nm, zorder=3)
    ax[1].set_yticks(y, [""] * len(levels))
    ax[1].set_xlabel("expected 68% half-width of φ_τ [deg]  (smaller is better)")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].set_title("τ Yukawa CP angle", fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Spin-flat H(125)+jet, parametric HL-LHC response, Asimov; same cross-fitted (g_T, g_L)/(g_T, g_A) "
             "regression and fit for every row (10/07 framework). Not an ATLAS result.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160)


def fig2(bd, out):
    pairs = [k for k in bd["pairs"]]
    levels = ["tauspin", "ipkin", "tauspin+ipkin", "phicp"]
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
    for l in ("tauspin", "ipkin", "tauspin+ipkin", "phicp"):
        ax[1].plot(x, [bd["ipsig"][k][l] for k in keys], marker=MK[l], color=C[l], lw=2, ms=7, label=LAB[l])
    ax[1].set_xticks(x, [k.replace("um", "") for k in keys])
    ax[1].set_xlabel("smallest 1-prong transverse impact parameter |b_T| [μm]")
    ax[1].set_ylabel("corr( g_T , exact T )")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("dependence on the IP lever arm (≥1 one-prong side)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def fig4(single, oracle, out):
    classes = ["pi", "rho", "a1(1p)", "a1(3p)"]
    rows = [(c, k) for c in classes for k in "nrk"]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(rows))
    series = [("tauspin", single["tauspin"]), ("ipkin_m125", single["ipkin_m125"]), ("ipkin", single["ipkin"]),
              ("oracle", oracle)]
    for j, (nm, src) in enumerate(series):
        v = [src[c][k] for c, k in rows]
        ax.scatter(x + (j - 1.5) * 0.17, v, s=28, marker={"oracle": "_"}.get(nm, MK.get(nm, "o")),
                   color=C[nm], label={"oracle": "true neutrinos, reco visibles (ceiling)"}.get(nm, LAB.get(nm)),
                   zorder=3, linewidths=2)
    ax.set_xticks(x, [f"{c}\n{k}" if k == "r" else k for c, k in rows])
    ax.set_ylabel("corr( h component , exact )  per τ side")
    ax.set_title("Single-τ polarimeter closure by reconstructed decay class", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = a.dir
    j = lambda n: json.load(open(f"{D}/{n}"))
    bd = j("breakdown.json")
    fig1(j("ent_robust.json"), j("ent_robust_aux.json"), j("ent_cp.json"), f"{D}/fig1_sensitivity.png")
    fig2(bd, f"{D}/fig2_modepairs.png")
    fig3(f"{D}/ent_robust_G.npz", bd, f"{D}/fig3_complement.png")
    fig4(j("corr_full.json")["single"], j("oracle_single.json"), f"{D}/fig4_singleside.png")


if __name__ == "__main__":
    main()
