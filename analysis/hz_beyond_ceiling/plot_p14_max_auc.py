"""Figure: how far reco H/Z AUC goes with IP and SV (learned) vs the geometry-aware ceiling.

usage: plot_p14_max_auc.py LEARNED_AUC_JSON NULLSPACE_AUC_JSON OUT.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = json.loads(Path(sys.argv[1]).read_text())
N = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
W, D = L["weighted_auc"], L["paired_delta"]
learned = [
    ("readout:base_s42", "full reco (current), seed 42", "0.35", "o"),
    ("readout:base_s43", "full reco (current), seed 43", "0.35", "o"),
    ("readout:sv_s42", "+ SV only (local frame)", "#F28E2B", "D"),
    ("readout:ip1v1_s42", "+ lead-track IP (P9)", "#9DB9D8", "s"),
    ("readout:full_s42", "+ 3-track IP + SV, no track offset (v1)", "#6F9BC8", "s"),
    ("readout:ip22_s42", "+ 3-track IP + track offset, no SV (v2)", "#4C78A8", "s"),
    ("readout:full22_shuffle_s42", "+ v2 geometry shuffled between events (null)", "#E45756", "v"),
    ("readout:full22_s42", "+ 3-track IP + SV + track offset (v2), seed 42", "#1F4E79", "o"),
    ("readout:full22_s43", "+ 3-track IP + SV + track offset (v2), seed 43", "#1F4E79", "o"),
    ("readout:full22_e100_s42", "v2, 100 epochs", "#1F4E79", "^"),
    ("readout:ens_full22", "v2, 2-seed h ensemble (best measured)", "#0B2545", "P"),
    ("readout:idealip22_s42", "v2 with ideal IP direction (truth azimuth)", "#8E6AC8", "*"),
    ("readout:oracle_s42", "+ truth tau direction (oracle, local frame)", "#59A14F", "*"),
]
learned = [r for r in learned if r[0] in W]
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.35, 1.1, 1.2]})
base = W["readout:base_s42"]
for i, (k, lab, col, mk) in enumerate(learned):
    v = W[k]
    if k in D:
        lo, hi = (base + x for x in D[k]["ci95"])
        ax[0].errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, ms=8, capsize=3)
    else:
        ax[0].plot(v, i, mk, color=col, ms=9)
    ax[0].text(v + 0.002, i + 0.2, f"{v:.4f}", fontsize=7.5, color=col)
ax[0].axvline(base, color="0.5", ls=":")
ax[0].axvline(W["ceiling:exact_h_LR"], color="#59A14F", ls="--", lw=1)
ax[0].text(W["ceiling:exact_h_LR"] - 0.001, -0.7, "exact truth h", fontsize=7.5, color="#59A14F", ha="right")
ax[0].set_yticks(range(len(learned)), [r[1] for r in learned], fontsize=8.5)
ax[0].set_xlabel("weighted H/Z AUC (fixed h readout), full reco")
ax[0].set_title(f"(a) learned point-h, validation N={L['events']:,}\nbars: paired 95% CI vs seed-42 baseline", fontsize=9)

U = N["unweighted"]
rows = [("flat_LR", "visible + MET + mass shells", "0.35"), ("ip1_LR", "+ lead-track IP", "#9DB9D8"),
        ("ip3_LR", "+ all core-track IPs", "#6F9BC8"), ("sv_LR", "+ SV only", "#F28E2B"),
        ("ip3_shuffle_LR", "+ IPs shuffled (null)", "#E45756"), ("ip3_sv_LR", "+ all IPs + SV", "#1F4E79"),
        ("ip3_sv_mass_LR", "+ all IPs + SV + parent mass", "#0B2545"), ("truth_h_LR", "exact truth h", "#59A14F")]
for i, (k, lab, col) in enumerate(rows):
    v = U[k]
    ax[1].barh(i, v - 0.5, left=0.5, color=col)
    ax[1].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
ax[1].set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
ax[1].set_xlim(0.5, 0.76)
ax[1].set_xlabel("H/Z AUC of null-space likelihood ratio (unit weight)")
ax[1].set_title(f"(b) information ceiling: truth visible + exact MET, N={N['events']:,}\n"
                "geometry enters as train-fitted likelihoods", fontsize=9)

pairs = list(L["by_pair"])
series = [("readout:base_s42", "full reco (current)", "0.35", "o", "-"),
          ("readout:full22_s42", "+ IP + SV + offset (v2)", "#1F4E79", "o", "-"),
          ("readout:idealip22_s42", "v2 with ideal IP direction", "#8E6AC8", "*", "--"),
          ("readout:oracle_s42", "+ truth tau direction", "#59A14F", "*", "-"),
          ("ceiling:exact_h_LR", "exact truth h", "k", "x", ":")]
for k, lab, col, mk, ls in series:
    if k in W:
        ax[2].plot(range(len(pairs)), [L["by_pair"][p][k] for p in pairs], marker=mk, color=col, ls=ls, label=lab)
ax[2].set_xticks(range(len(pairs)), [f"{p}\nN={L['by_pair'][p]['events']:,}" for p in pairs], fontsize=7.5)
ax[2].set_ylabel("weighted AUC")
ax[2].set_title("(c) learned point-h by truth decay-mode pair", fontsize=9)
ax[2].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(out, dpi=150)
