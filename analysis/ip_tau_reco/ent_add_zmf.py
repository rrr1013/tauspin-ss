"""Add the ZMF phi*_CP baseline level ('phicp_zmf') to the cached statistics of
ent_robust.py (g_T, g_L) and ent_cp.py (g_T, g_A).

Features: the phicp set of ent_robust.py (textbook + pair products + lab-frame transverse
directions) plus the standard acoplanarity angle phi*_CP of the LHC CP analyses
(cos, sin, the rho-method sign and the |lambda_perp| of both sides, phicp_zmf.py).
"""
import argparse

import numpy as np

from ent_cp import targets as cp_targets
from ent_robust import phicp, regress, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--zmf", required=True)
    ap.add_argument("--robust-cache", required=True)
    ap.add_argument("--cp-cache", required=True)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    z = np.load(args.zmf)
    idx = {u: i for i, u in enumerate(z["uid"])}
    j = np.array([idx[u] for u in d["uid"]])
    phi, ok = z["phi"][j], np.isfinite(z["phi"][j])
    X = np.concatenate([phicp(d), np.stack([np.where(ok, np.cos(phi), 0), np.where(ok, np.sin(phi), 0), ok * 1.0,
                                            z["sgn"][j], np.log1p(z["lmag"][j][:, 0]), np.log1p(z["lmag"][j][:, 1])], 1)], 1)
    h, w = d["h_exact"], d["w"]
    fold = np.arange(len(h)) % 2
    # sanity: modulation of phi* under the H and CP-odd reweightings of the spin-flat sample
    hz = np.where(np.isfinite(h).all((1, 2))[:, None, None], h, 0.0)
    tT = hz[:, 0, 0] * hz[:, 1, 0] + hz[:, 0, 1] * hz[:, 1, 1]
    tL = hz[:, 0, 2] * hz[:, 1, 2]
    for nm, wt in (("flat", np.ones(len(h))), ("H CP-even", 1 + tT - tL), ("CP-odd", 1 - tT - tL)):
        m = ok
        print(nm, "<cos phi*> =", round(float(np.average(np.cos(phi[m]), weights=(w * wt)[m])), 4), flush=True)
    for cache, T in ((args.robust_cache, targets(h)[1]), (args.cp_cache, cp_targets(h)[:, :2])):
        c = dict(np.load(cache, allow_pickle=True))
        c["G_phicp_zmf"] = regress(X, T, fold, None, w)[0]
        print(cache, "corr", [round(float(np.corrcoef(c["G_phicp_zmf"][:, k], T[:, k])[0, 1]), 4) for k in range(2)],
              "vs phicp", [round(float(np.corrcoef(c["G_phicp"][:, k], T[:, k])[0, 1]), 4) for k in range(2)], flush=True)
        np.savez_compressed(cache, **c)


if __name__ == "__main__":
    main()
