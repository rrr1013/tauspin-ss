"""Transfer the measured significance ratios to the published HL-LHC projections.

Inputs: classify_s*.json (phase 2, tau_had tau_had), phase-1 json files
(tau_lep tau_had single-side polarisation).
Reference values, 3 ab^-1, S2 (ATL-PHYS-PUB-2025-018 Table 2; ATL-PHYS-PUB-2024-016 Table 4):
  ATLAS bb tautau: tau_had tau_had 3.1, tau_lep tau_had 1.8, combined 3.5
  ATLAS combination 4.3 (other channels: sqrt(4.3^2 - 3.5^2) = 2.50)
  ATLAS+CMS 7.2, where the ATLAS bb tautau projection is used for both experiments
  ATLAS+CMS kappa_lambda 68 % CI: -27 % / +31 %
Approximations (stated in the output): channels and experiments add in
quadrature; the kappa_lambda interval scales with the inverse combined
significance.  Both ignore correlated systematics and the m_HH-shape
information specific to kappa_lambda.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REF = {"hadhad": 3.1, "lephad": 1.8, "bbtt": 3.5, "atlas": 4.3, "atlas_cms": 7.2, "kl": (-0.27, 0.31)}


def combine(r_hh, r_lh):
    bbtt = np.hypot(REF["hadhad"] * r_hh, REF["lephad"] * r_lh)
    bbtt_ref = np.hypot(REF["hadhad"], REF["lephad"])
    scale = REF["bbtt"] / bbtt_ref             # quadrature of 3.1 and 1.8 gives 3.58, ATLAS quotes 3.5
    bbtt *= scale
    other_atlas = np.sqrt(REF["atlas"] ** 2 - REF["bbtt"] ** 2)
    atlas = np.hypot(bbtt, other_atlas)
    other_comb = np.sqrt(REF["atlas_cms"] ** 2 - 2 * REF["bbtt"] ** 2)
    comb = np.sqrt(other_comb ** 2 + 2 * bbtt ** 2)
    kl = tuple(x * REF["atlas_cms"] / comb for x in REF["kl"])
    return dict(bbtt=bbtt, atlas=atlas, atlas_cms=comb, kl_lo=kl[0], kl_hi=kl[1])


def main(rundir):
    files = sorted(glob.glob(f"{rundir}/classify_s*.json"))
    runs = [json.load(open(f)) for f in files]
    here = Path(__file__).resolve().parent / "outputs" / "phase1"
    lh = json.load(open(here / "significance_lephad.json"))["arms"]
    # lephad: average of SLT and LTT weighted by their tau_lep tau_had significance share (SLT dominates)
    r_lh = {arm: 0.75 * lh[arm]["SLT"]["R_syst"] + 0.25 * lh[arm]["LTT"]["R_syst"] for arm in lh}
    lh_map = {"K": None, "K+obs": "reco_noIPSV", "K+low": "reco_IPSV", "K+h": "reco_IPSV", "K+exact": "exact"}
    out = {}
    for name in runs[0]["sets"]:
        for tag in ("atlas_calibrated", "raw"):
            rs = np.array([r["sets"][name][tag]["R_syst"] for r in runs])
            rst = np.array([r["sets"][name][tag]["R_stat"] for r in runs])
            key = f"{name} [{tag}]"
            rl = 1.0 if lh_map.get(name) is None else r_lh[lh_map[name]]
            proj = combine(rs.mean(), rl)
            out[key] = {"R_syst_mean": float(rs.mean()), "R_syst_sd": float(rs.std(ddof=1)) if len(rs) > 1 else 0.0,
                        "R_stat_mean": float(rst.mean()), "n_seeds": len(rs), "R_lephad_used": rl,
                        "hadhad": REF["hadhad"] * rs.mean(), **{k: float(v) for k, v in proj.items()}}
    json.dump(out, open(f"{rundir}/projection.json", "w"), indent=2)
    print(f"{'set':34s} {'R_syst':>13s} {'tauhtauh':>8s} {'bbtt':>6s} {'ATLAS':>6s} {'ATL+CMS':>7s} {'kl 68%':>14s}")
    for k, v in out.items():
        if "raw" in k:
            continue
        print(f"{k:34s} {v['R_syst_mean']:.4f}±{v['R_syst_sd']:.4f} {v['hadhad']:8.2f} {v['bbtt']:6.2f} "
              f"{v['atlas']:6.2f} {v['atlas_cms']:7.2f}  {v['kl_lo'] * 100:+.0f}%/{v['kl_hi'] * 100:+.0f}%")


if __name__ == "__main__":
    main(sys.argv[1])
