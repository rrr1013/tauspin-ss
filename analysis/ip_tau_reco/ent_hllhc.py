"""HL-LHC projection of the H -> tau_had tau_had entanglement significance.

Yields: ATLAS H -> tau tau cross-section measurement, 139 fb^-1 (arXiv:2201.08269,
Tables 8 and 11), tau_had tau_had signal regions.  Scaled to 3 ab^-1 at 14 TeV
(x 3000/139, signal x 1.13, backgrounds x 1.15) and to an MMC mass window
110 < m_tautau < 145 GeV with acceptances measured on our simulation (H 0.61,
Z 0.24, fakes 0.21, top 0.16, other 0.25).  The ATLAS categories are inclusive
in mass; the window replaces their mass fit (a simplification).
Spin templates: spin-flat H(125)+jet with the tauspin regressor (2D statistic of
entanglement_feasibility.py).  The background is given the Z -> tau tau spin state
for all components (fakes would carry less spin structure; see the variants).
Per-category Z_ent (Asimov, p = 1 vs p = 1/3, background normalisation constrained
to 2 % or 5 %), combined in quadrature.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from entanglement_feasibility import HYPS, spin_s, templates, weights, z_ent, z_target

LUMI = 3000 / 139
CATS = {  # name: (S, Z->tautau, fakes, top, other) at 139 fb^-1
    "VBF_0": (113, 2051, 1027, 0, 57.1), "VBF_1": (43.6, 115, 39.6, 0, 1.7),
    "VH_0": (109, 4636, 1627, 0, 209), "VH_1": (23.0, 539, 112, 0, 43.1),
    "ttH_0": (12.0, 265, 182, 239, 15.7), "ttH_1": (6.6, 20, 6.5, 15.2, 5.0),
    "boost_100_120_1j": (69.5, 5635, 3388, 0, 61), "boost_100_120_2j": (32.0, 2640, 1729, 0, 74.2),
    "boost_120_200_1j": (147, 11863, 2312, 0, 116), "boost_120_200_2j": (148, 10076, 2072, 0, 251),
    "boost_200_300": (130, 7252, 293, 0, 157), "boost_300_inf": (41.7, 973, 54, 0, 53.6),
}
ACC = {"S": 0.61, "Z": 0.24, "F": 0.21, "T": 0.16, "O": 0.25}


def hl_yields():
    out = {}
    for k, (s, z, f, t, o) in CATS.items():
        S = s * LUMI * 1.13 * ACC["S"]
        B = (z * ACC["Z"] + f * ACC["F"] + t * ACC["T"] + o * ACC["O"]) * LUMI * 1.15
        fake_frac = f * ACC["F"] * LUMI * 1.15 / B
        out[k] = (S, B, fake_frac)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hpred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lumi-scale", type=float, default=1.0, help="multiply the 3 ab^-1 yields (e.g. 0.12 for ~360 fb^-1)")
    ap.add_argument("--fakes-unpolarised", action="store_true", help="fake background without spin structure")
    ap.add_argument("--z-transverse", action="store_true")
    ap.add_argument("--mass-binned", default="", help="json with per-component m_tautau bin acceptances (replaces the window)")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    h = d["h_exact"]
    W, s, ok = weights(h, args.z_transverse)
    rmode = np.stack([d[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)
    feats = {"tauspin": np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode], 1),
             "textbook": np.concatenate([np.stack([d[f"spin_{v}_{q}"] for q in (0, 1)
                                                   for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")], 1), rmode], 1)}
    fold = np.arange(len(h)) % 2
    zt = z_target(h, args.z_transverse)
    Gs = {"exact": np.stack([np.where(ok, s, 0.0), zt], 1)}
    for name, X in feats.items():
        G = np.zeros((len(h), 2))
        for j, target in enumerate((np.where(ok, s, 0.0), zt)):
            for f in (0, 1):
                reg = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                                    l2_regularization=1.0, early_stopping=True, random_state=0)
                reg.fit(X[fold != f], target[fold != f])
                G[fold == f, j] = reg.predict(X[fold == f])
        Gs[name] = G
    Y = hl_yields()
    if args.mass_binned:
        acc = json.load(open(args.mass_binned))
        Yb = {}
        for k, (s_, z, f, t, o) in CATS.items():
            for i in range(len(acc["S"])):
                S = s_ * LUMI * 1.13 * acc["S"][i]
                comp = {"Z": z * acc["Z"][i], "F": f * acc["F"][i], "T": t * acc["T"][i], "O": o * acc["O"][i]}
                B = sum(comp.values()) * LUMI * 1.15
                Yb[f"{k}_m{i}"] = (S, B, comp["F"] * LUMI * 1.15 / B)
        Y = Yb
    res = {"yields_hl": {k: {"S": v[0] * args.lumi_scale, "B": v[1] * args.lumi_scale, "fake_frac": v[2]} for k, v in Y.items()},
           "lumi_scale": args.lumi_scale, "fakes_unpolarised": args.fakes_unpolarised,
           "z_transverse": args.z_transverse, "levels": {}}
    for name, G in Gs.items():
        T = templates(G, W)
        row = {}
        for bunc in (0.02, 0.05):
            zc = {}
            for k, (S, B, ff) in Y.items():
                Tk = dict(T)
                if args.fakes_unpolarised:
                    Tk["Z"] = (1 - ff) * T["Z"] + ff * T["U"]
                zc[k] = z_ent(Tk, S * args.lumi_scale, B * args.lumi_scale, bunc)
            row[f"dB{bunc}"] = {"per_category": zc, "combined": float(np.sqrt(sum(z ** 2 for z in zc.values())))}
        res["levels"][name] = row
        print(name, {k: round(v["combined"], 2) for k, v in row.items()},
              "| best cats:", sorted(((round(z, 2), c) for c, z in row["dB0.02"]["per_category"].items()), reverse=True)[:4],
              flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
