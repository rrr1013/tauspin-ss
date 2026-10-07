"""Summary table and figures of the revised projection (ent_robust.py, ent_cp.py)."""
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
       "phicp_zmf": ("LHC CP observables ($\\varphi^*_{CP}$ IP/$\\rho$, ZMF) + textbook", "#eb6834", "v", "-."),
       "textbook": ("textbook ($\\Upsilon$, $x$, hybrid $h$)", "#1baf7a", "^", ":")}
ORDER = ["exact", "tauspin", "tauspin_noIPSV", "phicp_zmf", "textbook"]
TAUS = [0.3, 1, 3, 10, 100]


def merge(*names):
    out = {}
    for n in names:
        p = E / n
        if p.exists():
            for lev, v in json.load(open(p))["levels"].items():
                out.setdefault(lev, {}).update(v)
    return out


def sigma_phi(q):
    """Expected 68 % half-width from the small-angle quadratic of q(phi)."""
    g = np.array([float(k) for k in q if 0 < float(k) <= 20])
    v = np.array([q[k] for k in q if 0 < float(k) <= 20])
    m = np.isfinite(v)
    if m.sum() == 0:
        return None
    a = (v[m] * g[m] ** 2).sum() / (g[m] ** 4).sum()
    return float(1 / np.sqrt(a)) if a > 0 else None


def main():
    ent = merge("ent_robust.json", "ent_robust_zmf.json")
    aux = merge("ent_robust_aux.json", "ent_robust_aux_zmf.json")
    cp = merge("ent_cp.json", "ent_cp_zmf.json")
    summ = {"entanglement": {}, "cp": {}}
    for lev in ORDER:
        e, a = ent.get(lev, {}), aux.get(lev, {})
        summ["entanglement"][lev] = {"sidebands_only": e.get("nominal", {}).get("Z"),
                                     **{f"tau{t:g}": a.get(f"tau{t:g}", {}).get("Z") for t in TAUS},
                                     **{k: v["Z"] for k, v in e.items() if isinstance(v, dict) and "Z" in v and k != "nominal"},
                                     **{k: v["Z"] for k, v in a.items() if "lumi" in k}}
        c = cp.get(lev, {})
        summ["cp"][lev] = {k: {"sigma_phi_deg": sigma_phi(v["q"]), "Z_CPodd": v.get("Z_CPodd")} for k, v in c.items()}
    json.dump(summ, open(E / "robust_summary.json", "w"), indent=1)
    for lev in ORDER:
        s = summ["entanglement"][lev]
        print(f"{lev:15s} ent: free {s['sidebands_only']}  " + "  ".join(f"tau{t:g} {s.get(f'tau{t:g}')}" for t in TAUS))
        print(f"{'':15s} cp : " + "  ".join(f"{k} {None if v['sigma_phi_deg'] is None else round(v['sigma_phi_deg'], 1)}"
                                         for k, v in summ["cp"][lev].items()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    xs = [0.1] + TAUS
    for lev in ORDER:
        s = summ["entanglement"][lev]
        ys = [s["sidebands_only"]] + [s.get(f"tau{t:g}") for t in TAUS]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        lab, col, mk, ls = LEV[lev]
        ax.plot(*zip(*pts), marker=mk, color=col, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 8], ["1", "2", "3", "5", "8"])
    ax.set_xscale("log")
    ax.set_xticks(xs, ["side-\nbands", "0.3", "1", "3", "10", "100"])
    ax.minorticks_off()
    for y in (3, 5):
        ax.axhline(y, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("background spin-shape knowledge: control sample / signal-region yield")
    ax.set_ylabel("expected significance vs all separable states [$\\sigma$]")
    ax.set_title("(a) spin entanglement in H$\\to\\tau_h\\tau_h$, 3 ab$^{-1}$, one experiment", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    phis = np.linspace(0, 45, 200)
    for lev in ORDER:
        for key, ls in (("nominal", "-"), ("known_bkg_shape", "--")):
            v = summ["cp"].get(lev, {}).get(key)
            if not v or v["sigma_phi_deg"] is None:
                continue
            lab, col, mk, _ = LEV[lev]
            ax.plot(phis, (phis / v["sigma_phi_deg"]) ** 2, color=col, ls=ls,
                    label=f"{lab.split(' (')[0]}: $\\pm${v['sigma_phi_deg']:.1f}$^\\circ$" + (" (shape known)" if ls == "--" else ""))
    ax.axhline(1, color="#999", lw=0.8, ls=":")
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 45)
    ax.set_xlabel("CP-mixing angle $\\varphi_\\tau$ [deg]")
    ax.set_ylabel("$-2\\Delta\\ln L$ (Asimov, $\\varphi_\\tau = 0$)")
    ax.set_title("(b) tau Yukawa CP angle, $\\tau_h\\tau_h$ only, 3 ab$^{-1}$, one experiment", fontsize=10)
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"ent_cp_robust.{ext}", dpi=180)


if __name__ == "__main__":
    main()
