"""Phase 1 significance gain from adding the spin dimension to the most
signal-like bin of the ATLAS Run-2 HH -> bb tau_had tau_had analysis.

Composition: arXiv:2209.10910 Table 5 (tau_had tau_had, most signal-like MVA bin,
139 fb^-1, post-fit), scaled to 3000 fb^-1 at 14 TeV with the cross-section
factors of ATL-PHYS-PUB-2024-016 Table 1.  Each background is mapped to a spin
hypothesis of phase1_reweight.py.  The spin discriminant
D = p_H / sum_c f_c p_c (f_c = background fractions) is binned in quantiles of the
total background, and the significance of mu_HH = 1 is computed

  * stat-only, with the Asimov formula summed over bins, and
  * with Gaussian-constrained normalisation nuisance parameters per process,
    from the profiled Fisher information (sigma_mu at the Asimov point).

The ratio R = Z(spin-binned) / Z(single bin) is the quantity transferred to the
ATLAS projection.  Template statistical uncertainty: event bootstrap (100).
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
HYPS = ("H", "Z", "W", "U", "WU")
LUMI_SCALE = 3000.0 / 139.0

# (yield at 139 fb^-1, spin hypothesis, 13->14 TeV factor, relative norm. unc.)
TABLE5_HADHAD = {
    "ggF HH":   (1.58,   "H",  1.18, None),
    "VBF HH":   (0.0227, "H",  1.19, None),
    "ttbar":    (0.5,    "W",  1.18, 0.10),
    "single top": (0.47, "W",  1.18, 0.10),
    "Z+HF":     (2.3,    "Z",  1.18, 0.10),
    "MJ fakes": (0.47,   "U",  1.18, 0.20),
    "ttbar fakes": (0.29, "WU", 1.18, 0.20),
    "single H": (1.7,    "H",  1.15, 0.15),
    "other":    (0.4,    "Z",  1.18, 0.20),
}
SIGNALS = ("ggF HH", "VBF HH")


def yields(table):
    s = sum(v[0] * v[2] for k, v in table.items() if k in SIGNALS) * LUMI_SCALE
    bkg = {k: (v[0] * v[2] * LUMI_SCALE, v[1], v[3]) for k, v in table.items() if k not in SIGNALS}
    return s, bkg


def templates(probs, w, s_hyp_frac, nbins):
    """Quantile-binned templates of D for each hypothesis (unit-normalised)."""
    p = probs / probs.sum(1, keepdims=True)
    denom = sum(f * p[:, HYPS.index(h)] for h, f in s_hyp_frac.items())
    d = np.log(p[:, 0] + 1e-12) - np.log(denom + 1e-12)
    wb = sum(f * w[h] for h, f in s_hyp_frac.items())
    o = np.argsort(d)
    cb = np.cumsum(wb[o]) / wb.sum()
    edges_idx = np.searchsorted(cb, np.linspace(0, 1, nbins + 1)[1:-1])
    bin_of = np.zeros(len(d), int)
    bin_of[o] = np.searchsorted(edges_idx, np.arange(len(d)), side="right")
    T = {h: np.bincount(bin_of, weights=w[h], minlength=nbins) / w[h].sum() for h in HYPS}
    return T


def asimov_z(s, b):
    s, b = np.asarray(s, float), np.asarray(b, float)
    return float(np.sqrt(np.sum(2 * ((s + b) * np.log1p(s / b) - s))))


def fisher_z(s_bins, b_bins_by_proc, uncs):
    """1/sigma_mu with profiled Gaussian normalisation nuisances (Asimov, mu=1)."""
    procs = list(b_bins_by_proc)
    nu = s_bins + sum(b_bins_by_proc.values())
    derivs = [s_bins] + [b_bins_by_proc[p] * uncs[p] for p in procs]
    n = len(derivs)
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            F[i, j] = np.sum(derivs[i] * derivs[j] / nu)
    F[1:, 1:] += np.eye(n - 1)  # unit-Gaussian constraints
    cov = np.linalg.inv(F)
    return float(1.0 / np.sqrt(cov[0, 0]))


def evaluate(probs, w, table, nbins):
    s, bkg = yields(table)
    by_hyp = {}
    for name, (y, hyp, _) in bkg.items():
        by_hyp[hyp] = by_hyp.get(hyp, 0.0) + y
    btot = sum(by_hyp.values())
    frac = {h: by_hyp.get(h, 0.0) / btot for h in HYPS}
    T = templates(probs, w, {h: f for h, f in frac.items() if f > 0}, nbins)
    s_bins = s * T["H"]
    b_proc = {name: y * T[hyp] for name, (y, hyp, _) in bkg.items()}
    uncs = {name: u for name, (_, _, u) in bkg.items()}
    b_bins = sum(b_proc.values())
    one = {name: np.array([y]) for name, (y, _, _) in bkg.items()}
    out = {
        "S": s, "B": btot,
        "Z_stat_1bin": asimov_z([s], [btot]),
        "Z_stat_spin": asimov_z(s_bins, b_bins),
        "Z_syst_1bin": fisher_z(np.array([s]), one, uncs),
        "Z_syst_spin": fisher_z(s_bins, b_proc, uncs),
    }
    out["R_stat"] = out["Z_stat_spin"] / out["Z_stat_1bin"]
    out["R_syst"] = out["Z_syst_spin"] / out["Z_syst_1bin"]
    return out


def main(nbins=20, nboot=100, seed=1):
    st = np.load(OUT / "probs_pool.npz")
    w = {h: st[f"w_{h}"] for h in HYPS}
    arms = [k[len("probs_"):] for k in st.files if k.startswith("probs_")]
    rng = np.random.default_rng(seed)
    res = {"composition": "arXiv:2209.10910 Table 5 tau_had tau_had most signal-like bin, x3000/139, 14 TeV factors",
           "nbins": nbins, "arms": {}}
    for arm in arms:
        probs = st[f"probs_{arm}"]
        central = evaluate(probs, w, TABLE5_HADHAD, nbins)
        boots = []
        for _ in range(nboot):
            i = rng.integers(0, len(probs), len(probs))
            boots.append(evaluate(probs[i], {h: w[h][i] for h in HYPS}, TABLE5_HADHAD, nbins))
        for key in ("R_stat", "R_syst"):
            arr = np.array([b[key] for b in boots])
            central[key + "_boot_sd"] = float(arr.std())
        res["arms"][arm] = central
        print(arm, {k: round(v, 4) if isinstance(v, float) else v for k, v in central.items()}, flush=True)
    (OUT / "significance_top_bin.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
