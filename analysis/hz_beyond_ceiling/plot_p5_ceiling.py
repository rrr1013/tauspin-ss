"""Figure: spin-only H/Z likelihood-ratio AUC on the MET null space, with and without IP."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
rows = [("flat_LR", "visible + MET + mass shells (09-19 ceiling)", "0.45", "o"),
        ("ip_legacy_LR", "+ legacy IP (z0 w.r.t. beam spot)", "0.6", "s"),
        ("ip_shuffle_LR", "+ PV-IP, shuffled between events (null)", "#E45756", "v"),
        ("mass_LR", "+ parent mass 91.19 GeV (H/Z common)", "#F28E2B", "D"),
        ("ip_LR", "+ PV-referenced IP azimuth", "#4C78A8", "o"),
        ("ip_mass_LR", "+ PV-IP + parent mass", "#1F4E79", "P"),
        ("truth_h_LR", "exact truth h (all neutrinos known)", "#59A14F", "*")]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
delta = d["paired_delta_vs_flat_LR"]
base = d["unweighted"]["flat_LR"]
for i, (k, lab, col, mk) in enumerate(rows):
    v = d["unweighted"][k]
    if k in delta:
        lo, hi = (base + x for x in delta[k]["ci95"])
        ax[0].errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt=mk, color=col, ms=8, capsize=3)
    else:
        ax[0].plot(v, i, mk, color=col, ms=10 if mk == "*" else 8)
    ax[0].text(v + 0.003, i + 0.18, f"{v:.4f}", fontsize=8, color=col)
ax[0].axvline(base, color="0.45", ls=":", lw=1)
ax[0].set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=9)
ax[0].set_xlabel("H/Z AUC of null-space likelihood ratio (unit weight)")
ax[0].set_title(f"truth visible + exact tau-neutrino MET, validation N={d['events']:,}\n"
                "bars: paired event-bootstrap 95% CI of the difference to the 09-19 ceiling", fontsize=9)
ax[0].set_xlim(0.59, 0.73)
pairs = list(d["by_pair"])
keys = [("flat_LR", "ceiling", "0.45"), ("ip_LR", "+ PV-IP", "#4C78A8"),
        ("ip_mass_LR", "+ PV-IP + mass", "#1F4E79"), ("truth_h_LR", "exact h", "#59A14F")]
for j, (k, lab, col) in enumerate(keys):
    xs = [p + (j - 1.5) * 0.18 for p in range(len(pairs))]
    ax[1].bar(xs, [d["by_pair"][p][k] - 0.5 for p in pairs], 0.17, bottom=0.5, color=col, label=lab)
ax[1].set_xticks(range(len(pairs)), [f"{p}\nN={d['by_pair'][p]['events']:,}" for p in pairs], fontsize=9)
ax[1].set_ylim(0.5, 0.75)
ax[1].set_ylabel("AUC")
ax[1].set_title("by truth decay-mode pair (IP from the leading core track only)", fontsize=9)
ax[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(out, dpi=150)
