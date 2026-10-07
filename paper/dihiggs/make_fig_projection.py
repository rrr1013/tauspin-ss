"""Projection figure of the paper: ATLAS HL-LHC HH significance with the tau information added.
Transfer: R_ch = sqrt(f R_bin^2 + (1 - f) R_rest^2) (f = 0.5 ... 1), channels in quadrature,
bbtautau rescaled from 3.1 (+) 1.8 to the quoted 3.5, other ATLAS channels sqrt(4.3^2 - 3.5^2),
ATLAS+CMS^2 = 7.2^2 + 2 (bbtautau^2 - 3.5^2)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def proj(rhh, rlh):
    bb = 3.5 * math.hypot(3.1 * rhh, 1.8 * rlh) / math.hypot(3.1, 1.8)
    return bb, math.hypot(bb, math.sqrt(4.3 ** 2 - 3.5 ** 2)), math.sqrt(7.2 ** 2 + 2 * (bb ** 2 - 3.5 ** 2))

rch = lambda r, f, rest=1.0: math.sqrt(f * r * r + (1 - f) * rest * rest)
cases = [("$\\tau_h\\tau_h$ spin", (1.0119, 1.0), (1.0391, 1.0)),
         ("$\\tau_\\ell\\tau_h$ lepton lifetime", (1.0, rch(1.132, 0.5)), (1.0, 1.210)),
         ("both", (1.0119, rch(1.132, 0.5)), (1.0391, 1.210))]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (k, base, name) in zip(axes, ((0, 3.5, "ATLAS $b\\bar b\\tau\\tau$"), (1, 4.3, "ATLAS HH combination"), (2, 7.2, "ATLAS+CMS HH"))):
    for i, (lab, lo, hi) in enumerate(cases):
        a, b = proj(*lo)[k], proj(*hi)[k]
        ax.plot([a, b], [i, i], lw=8, color="#2a78d6", solid_capstyle="butt")
        ax.text(b + 0.01 * base, i, f"{a:.2f}-{b:.2f}", va="center", fontsize=9)
    ax.axvline(base, color="#555", ls="--", lw=1)
    ax.set_yticks(range(len(cases)), [c[0] for c in cases] if k == 0 else [""] * len(cases))
    ax.set_xlim(base * 0.99, base * 1.10)
    ax.set_xlabel("expected significance at 3 ab$^{-1}$ [$\\sigma$]")
    ax.set_title(f"{name} (published {base})", fontsize=10)
    ax.invert_yaxis()
fig.tight_layout()
fig.savefig("figures/fig_projection.pdf")
fig.savefig("figures/fig_projection.png", dpi=180)
