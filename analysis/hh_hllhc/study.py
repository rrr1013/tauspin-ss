"""Main study: HL-LHC bbtautau significance with and without TauSpin information, and its
transfer to the ATLAS HL-LHC HH projection.

Pre-registered nominal (fixed before the systematics-calibrated results were looked at):
  n_sub = 6 spin sub-bins per baseline bin, hadhad templates conditioned on K,
  Z transverse correlation c_T = 0.5, single H = H spin, fakes = unpolarised,
  systematics = 'baseline-calibrated' (single-H norm 18% correlated + fixed Z+HF 10%,
  fakes 10%, other 15%, top 5%, free top/Z+HF norms), no extra spin-shape uncertainty;
  the spin-shape uncertainty (25% contrast per hypothesis) is the main robustness variant.
Gains are ratios R = Z(with spin)/Z(baseline) of the same likelihood; ATLAS numbers are
scaled by R.  Seeds 0-2 differ only in classifier initialisation.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from projection import DEFAULT, SpinTemplates, build_spec, significance

OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs")) / "study"

SYST_FIXED = {"zhf": 0.10, "fake": 0.10, "fake_mj": 0.10, "fake_tt": 0.10, "other": 0.15, "top": 0.05}
SCEN = {
    "stat": {},
    "baseline": {"norm_sys": {**SYST_FIXED, "single_h": 0.18}},
}
# ATL-PHYS-PUB-2025-006 Table 5 (3000 fb^-1, baseline) and PUB-2024-016 Table 4
ATLAS = {"bbgg": 2.43, "bbtt": 3.54, "bbbb": 0.99, "ML": 0.99, "bbll": 0.48, "comb": 4.26,
         "bbtt_hadhad": 3.1, "bbtt_lephad": 1.8}
ATLAS_NOSYST = {"bbgg": 2.73, "bbtt": 4.60, "bbbb": 1.76, "ML": 1.20, "bbll": 1.60, "comb": 5.98}


def zs(cfg, spin):
    spec, _ = build_spec(cfg, spin)
    out = {}
    for ch in [c["name"] for c in spec["channels"]]:
        out[ch] = significance({**spec, "channels": [c for c in spec["channels"] if c["name"] == ch]})
    out["hadhad"] = significance({**spec, "channels": [c for c in spec["channels"] if c["name"].startswith("hadhad")]})
    out["lephad"] = significance({**spec, "channels": [c for c in spec["channels"] if not c["name"].startswith("hadhad")]})
    out["combined"], fit = significance(spec, return_fit=True)
    out["fit"] = fit
    return out


def transfer(R, table):
    """Scale the ATLAS bbtautau significance by R and propagate to the combination with the
    effective correlation factor c = Z_comb^2 / sum Z_i^2 of the published table."""
    others = ("bbgg", "bbbb", "ML", "bbll")
    s2 = sum(table[k] ** 2 for k in others) + table["bbtt"] ** 2
    c = table["comb"] ** 2 / s2
    new = table["bbtt"] * R
    comb = np.sqrt(table["comb"] ** 2 + c * (new ** 2 - table["bbtt"] ** 2))
    return {"bbtt": float(new), "comb": float(comb), "c": float(c)}


def main(variants):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, v in variants.items():
        cfg = {**DEFAULT, **SCEN[v.get("syst", "baseline")]}
        cfg["spin_mix"] = v.get("spin_mix", {})
        cfg["source"] = v.get("source", "run2legacy")
        if cfg["source"] == "latest_run3":
            from projection import XS_RUN3
            cfg["xs_ratio"] = XS_RUN3
        cfg["spin_shape_unc"] = v.get("shape", 0.0)
        base = zs(cfg, None)
        cfg["lifetime"] = v.get("lifetime")
        row = {"config": v, "baseline": base, "arms": {}}
        for arm in v.get("arms", ("textbook", "tauspin", "exact")):
            seeds = (0,) if arm in ("exact", "none") else v.get("seeds", (0, 1, 2))
            per = []
            for s in seeds:
                spin = None if arm == "none" else SpinTemplates(
                    arm, f"s{s}_ct{v.get('c_t', 0.5):g}", v.get("n_sub", 6), v.get("cond_k", True))
                z = zs(cfg, spin)
                per.append(z)
            R = {k: [p[k] / base[k] for p in per] for k in base if k != "fit"}
            Rm = {k: float(np.mean(x)) for k, x in R.items()}
            row["arms"][arm] = {"z": per, "R": R, "R_mean": Rm,
                                "R_seed_sd": {k: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0 for k, x in R.items()},
                                "atlas_baseline": transfer(Rm["combined"], ATLAS),
                                "atlas_nosyst": transfer(Rm["combined"], ATLAS_NOSYST)}
            print(name, arm, {k: round(x, 4) for k, x in Rm.items()},
                  row["arms"][arm]["atlas_baseline"], flush=True)
        res[name] = row
        json.dump(res, open(OUT / f"study_{sys.argv[1] if len(sys.argv) > 1 else 'main'}.json", "w"), indent=1)


VARIANTS = {
    "nominal": {},
    "stat_only": {"syst": "stat"},
    "spin_shape25": {"shape": 0.25},
    "nsub4": {"n_sub": 4}, "nsub10": {"n_sub": 10},
    "no_condK": {"cond_k": False},
    "Z_cT0": {"c_t": 0.0, "seeds": (0,)},
    "singleH_30pctZ": {"spin_mix": {"single_h": {"H": 0.7, "Z": 0.3}}},
    "fakes_Hlike": {"spin_mix": {"fake_mj": {"H": 1.0}, "fake_tt": {"H": 1.0}, "other": {"H": 1.0}}},
    "fakes_Wlike": {"spin_mix": {"fake_mj": {"WU": 1.0}, "fake_tt": {"W": 1.0}}},
    # lepton lifetime in lep-had (TauSpin PV-referenced IP on the light lepton), alone
    # ('none') and together with the spin discriminant
    "lt_nominal": {"lifetime": {}, "arms": ("none", "tauspin", "exact")},
    "lt_stat": {"syst": "stat", "lifetime": {}, "arms": ("none", "tauspin", "exact")},
    "lt_real_res": {"lifetime": {"variant": "real"}, "arms": ("none", "tauspin")},
    "lt_ttva": {"lifetime": {"ttva": True}, "arms": ("none", "tauspin")},
    "lt_top_tau25": {"lifetime": {"top_tau_fraction": 0.25}, "arms": ("none", "tauspin")},
    "lt_fake_np30": {"lifetime": {"fake_nonprompt": 0.30}, "arms": ("none", "tauspin")},
    "latest_run3": {"source": "latest_run3", "lifetime": {}, "arms": ("none", "textbook", "tauspin", "exact")},
    "latest_run3_spinonly": {"source": "latest_run3", "arms": ("textbook", "tauspin", "exact")},
    "latest_run2": {"source": "latest_run2", "lifetime": {}, "arms": ("none", "tauspin", "exact")},
    "latest_run2_spinonly": {"source": "latest_run2", "arms": ("tauspin", "exact")},
    "latest_run3_stat": {"source": "latest_run3", "syst": "stat", "lifetime": {}, "arms": ("none", "tauspin", "exact")},
    "lt_worst": {"lifetime": {"variant": "real", "ttva": True, "top_tau_fraction": 0.25,
                              "fake_nonprompt": 0.30}, "shape": 0.25, "arms": ("none", "tauspin")},
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    sel = VARIANTS if which == "all" else {which: VARIANTS[which]}
    main(sel)
