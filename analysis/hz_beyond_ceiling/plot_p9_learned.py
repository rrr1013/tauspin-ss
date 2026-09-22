"""Figure: learned point-h with local-frame geometry (parity input).

(a) weighted H/Z AUC of the fixed readout with paired-bootstrap CI vs baseline,
(b) per-component (n, r, k) correlation of predicted vs exact h,
(c) weighted AUC by truth decay-mode pair.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])  # ~/hz-beyond-ceiling-20260923/artifacts
out = Path(sys.argv[2])
auc = json.loads((root / "p9_readout" / "auc.json").read_text())
arms = [("baseline_current", "full reco (current)", "0.45", "o"),
        ("ip_legacy", "+ legacy IP", "0.65", "s"),
        ("ip_shuffle", "+ PV-IP shuffled", "#E45756", "v"),
        ("ip_pv", "+ PV-referenced IP", "#4C78A8", "o"),
        ("oracle", "+ truth tau direction (oracle)", "#59A14F", "*")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
base = auc["weighted_auc"]["readout:baseline_current"]
for i, (a, lab, col, mk) in enumerate(arms):
    v = auc["weighted_auc"][f"readout:{a}"]
    if a != "baseline_current":
        lo, hi = (base + x for x in auc["paired_delta"][f"readout:{a}"]["ci95"])
        ax[0].errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, ms=9, capsize=3)
    else:
        ax[0].plot(v, i, mk, color=col, ms=9)
    ax[0].text(v, i + 0.22, f"{v:.4f}", fontsize=8, color=col, ha="center")
ax[0].axvline(base, color="0.45", ls=":")
ax[0].set_yticks(range(len(arms)), [x[1] for x in arms], fontsize=9)
ax[0].set_xlabel("weighted H/Z AUC, fixed h readout")
ax[0].set_title(f"validation N={auc['events']:,}; bars = paired 95% CI vs baseline", fontsize=9)
ax[0].set_ylim(-0.6, len(arms) - 0.3)

comp = {}
for a, *_ in arms:
    d = np.load(root / "p9" / a / "validation_predictions.npz")
    hp, h = d["h_pred"].reshape(-1, 2, 3), d["h"].reshape(-1, 2, 3)
    comp[a] = [np.corrcoef(hp[..., c].ravel(), h[..., c].ravel())[0, 1] for c in range(3)]
x = np.arange(3)
for j, (a, lab, col, _) in enumerate(arms):
    ax[1].bar(x + (j - 2) * 0.16, comp[a], 0.15, color=col, label=lab)
ax[1].set_xticks(x, [r"$h_n$ (transverse)", r"$h_r$ (transverse)", r"$h_k$ (longitudinal)"])
ax[1].set_ylim(0.4, 1.0)
ax[1].set_ylabel("Pearson correlation, predicted vs exact h (both taus)")
ax[1].set_title("which h components the geometry recovers", fontsize=9)
ax[1].legend(fontsize=7, loc="upper left")

pairs = list(auc["by_pair"])
for j, (a, lab, col, _) in enumerate(arms):
    vals = [auc["by_pair"][p][f"readout:{a}"] for p in pairs]
    ax[2].plot(range(len(pairs)), vals, marker="o" if j != 4 else "*", color=col, label=lab,
               ls="-" if a in ("baseline_current", "ip_pv", "oracle") else "--")
ax[2].plot(range(len(pairs)), [auc["by_pair"][p]["ceiling:exact_h_LR"] for p in pairs], "k:",
           marker="x", label="exact truth h (analytic LR)")
ax[2].set_xticks(range(len(pairs)), [f"{p}\nN={auc['by_pair'][p]['events']:,}" for p in pairs],
                 fontsize=8)
ax[2].set_ylabel("weighted AUC")
ax[2].set_title("by truth decay-mode pair", fontsize=9)
ax[2].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(json.dumps(comp))
