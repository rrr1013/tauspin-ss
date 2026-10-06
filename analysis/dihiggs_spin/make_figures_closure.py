"""Closure figure: regressed vs exact h on the held-out equal-mass H/Z(125)+jet
events, per component, with the tauspin ATLAS full-simulation references."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
ATLAS_CORR = {"n": 0.62, "r": 0.58, "k": 0.765}


def main(run="v1"):
    d = np.load(HERE / "outputs" / run / "hpred_partial_holdout.npz", allow_pickle=True)
    c = json.load(open(HERE / "outputs" / run / "closure_hz.json"))
    h, hp = d["h_exact"], d["h_pred"]
    ok = np.isfinite(h).all((1, 2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for j, comp in enumerate("nrk"):
        ax = axes[j]
        x = h[ok, :, j].ravel()
        y = hp[ok, :, j].ravel()
        hist, xe, ye = np.histogram2d(x, y, bins=60, range=[[-1, 1], [-1.2, 1.2]])
        hist = hist / hist.sum(1, keepdims=True).clip(1)      # column-normalised: P(pred | exact)
        ax.pcolormesh(xe, ye, hist.T, cmap="Blues", shading="flat")
        prof = [np.median(y[(x >= a) & (x < b)]) for a, b in zip(xe[:-1], xe[1:])]
        ax.plot(0.5 * (xe[:-1] + xe[1:]), prof, color="#eb6834", lw=1.8, label="median")
        ax.plot([-1, 1], [-1, 1], color="#52514e", lw=0.8, ls="--", label="ideal")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"$h_{comp}$: corr {r:.2f} (tauspin ATLAS full sim {ATLAS_CORR[comp]:.2f})", fontsize=9.5)
        ax.set_xlabel(f"exact $h_{comp}$")
        ax.set_ylabel(f"regressed $\\hat h_{comp}$")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"held-out H(125)+j / Z(125)+j, both sides, {ok.sum():,} events;  "
                 f"H/Z AUC fixed readout {c['auc_readout_hpred_valid_h']:.3f} (ATLAS 0.626), "
                 f"exact-h LR {c['auc_exact_LR']:.3f} (ATLAS 0.719-0.729)", fontsize=9.5)
    fig.tight_layout()
    out = HERE / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"phase2_closure_h_regression.{ext}", dpi=180)


if __name__ == "__main__":
    main()
