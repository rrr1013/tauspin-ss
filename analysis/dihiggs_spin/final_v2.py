"""Final numbers from the spin-flat HH analysis (spin_gain_U.json, spin_D_eff20.npz).

Bin gain R (most signal-like tau_had tau_had bin, K region at 20 % signal
efficiency, Z_long) is transferred to the channel with an information fraction f
of that bin: R_ch = sqrt(f R_top^2 + (1 - f) R_rest^2), where R_rest = 1
(lower edge) or the gain of a top-rich bin from the composition scan (upper
edge).  f = 0.62 is the stat-only share of the Run-2 top bin (Z 3.15 of 4.0 at
3 ab^-1); the envelope uses f in [0.5, 1].  Combination: channels and
experiments in quadrature (ATLAS+CMS uses the ATLAS bb tautau projection for
both experiments).  tau_lep tau_had is not included (no realistic-mode
evaluation); its phase-1 single-side estimate is reported separately.
"""
import json

import numpy as np

from classify import asimov_z
from spin_gain_U import HYPS, TOPBIN, UNC, disc, edges_for, significance, templates

REF = dict(hadhad=3.1, lephad=1.8, bbtt=3.5, atlas=4.3, atlas_cms=7.2)


def channel(r_top, r_rest, f):
    return np.sqrt(f * r_top ** 2 + (1 - f) * r_rest ** 2)


def combine(r_ch):
    scale = REF["bbtt"] / np.hypot(REF["hadhad"], REF["lephad"])
    bbtt = scale * np.hypot(REF["hadhad"] * r_ch, REF["lephad"])
    other = np.sqrt(REF["atlas"] ** 2 - REF["bbtt"] ** 2)
    atlas = np.hypot(bbtt, other)
    other_c = np.sqrt(REF["atlas_cms"] ** 2 - 2 * REF["bbtt"] ** 2)
    comb = np.sqrt(other_c ** 2 + 2 * bbtt ** 2)
    return {"bbtt_rel": bbtt / REF["bbtt"] - 1, "atlas_rel": atlas / REF["atlas"] - 1,
            "atlas_cms_rel": comb / REF["atlas_cms"] - 1, "bbtt": bbtt, "atlas": atlas, "atlas_cms": comb}


def main(R="outputs/v2"):
    g = json.load(open(f"{R}/spin_gain_U.json"))
    z = np.load(f"{R}/spin_D_eff20.npz")
    W = {Y: z[f"w_{Y}"] for Y in HYPS}
    region = z["region"]
    fmap = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}
    base = {k: v for k, v in TOPBIN.items() if k != "signal"}
    # composition scan: move the Z+HF share into top
    scan = []
    for x in np.linspace(0, 1, 11):
        b = dict(base)
        moved = x * base["Z+HF"]
        b["Z+HF"] -= moved
        b["top"] += moved
        tot = sum(b.values())
        frac = {G: b[G] / tot for G in UNC}
        row = {"top_fraction": frac["top"]}
        for name, P in (("spin", z["probs"]), ("exact", z["dens"])):
            D = disc(P, frac, fmap)
            e = edges_for(D, W, region, frac, fmap)
            T = templates(D, W, region, e)
            s = TOPBIN["signal"] * 3000 / 139 * T["H"]
            bg = sum(b[G] * 3000 / 139 * T[fmap[G]] for G in UNC)
            row[name] = asimov_z(s, bg) / asimov_z(np.array([s.sum()]), np.array([bg.sum()]))
        scan.append(row)
    v = g["variants"]["Z_long"]["sig_eff_20"]
    vf = g["variants"]["Z_full"]["sig_eff_20"]
    r_rest_up = {"spin": max(r["spin"] for r in scan if r["top_fraction"] <= 0.61),
                 "exact": max(r["exact"] for r in scan if r["top_fraction"] <= 0.61)}
    scen = {
        "tauspin spin-only (Z_long)": v["spin"]["R"],
        "tauspin spin-only, with spin shape nuisance": v["spin"]["R_shape"],
        "tauspin increment over kinematics": v["increment_kin_to_kinspin"],
        "tauspin spin-only (Z with transverse correlations)": vf["spin"]["R"],
        "exact h (implemented modes)": v["exact"]["R"],
    }
    rows = []
    for name, r in scen.items():
        key = "exact" if name.startswith("exact") else "spin"
        for f in (0.5, 0.62, 1.0):
            # top-rich edge: scale the excess of this scenario by the top-rich / nominal excess of its family
            nominal = scen["tauspin spin-only (Z_long)"] if key == "spin" else scen["exact h (implemented modes)"]
            rr_up = 1 + (r - 1) * (r_rest_up[key] - 1) / max(nominal - 1, 1e-9)
            for edge, rr in (("rest=1", 1.0), ("rest=top-rich", rr_up)):
                rc = channel(r, rr, f)
                rows.append({"scenario": name, "R_bin": r, "f": f, "rest": edge, "R_channel": rc, **combine(rc)})
    out = {"scan": scan, "rows": rows, "r_rest_upper": r_rest_up}
    json.dump(out, open(f"{R}/final_v2.json", "w"), indent=1)
    for row in rows:
        if row["f"] == 0.62:
            print(f"{row['scenario']:52s} {row['rest']:14s} R_bin {row['R_bin']:.4f} R_ch {row['R_channel']:.4f}  "
                  f"bbtt {row['bbtt_rel'] * 100:+.2f}%  ATLAS {row['atlas_rel'] * 100:+.2f}%  ATLAS+CMS {row['atlas_cms_rel'] * 100:+.2f}%")
    for r in scan:
        print(f"top fraction {r['top_fraction']:.2f}: spin {r['spin']:.4f} exact {r['exact']:.4f}")


if __name__ == "__main__":
    main()
