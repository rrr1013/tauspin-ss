"""Supplementary levels after the review: pi0-marginalised classical posterior and the SV-response
stress (evaluation sample with twice the SV resolution).

Each level is regressed with the same cross-fitted HistGradientBoosting as ent_robust and fitted
with the nominal entanglement test (sidebands only); per reco mode pair the weighted corr(g_T, T)
is written as well.  Levels:
  ipkin_pi0            classical posterior with R = 16 pi0 candidates per event
  tauspin+ipkin_pi0
  tauspin_sv2          the fixed 10/07 regressor applied to the degraded-SV features
  ipkin_sv2mis         classical posterior on the degraded SV, likelihood still assuming 10 um
  ipkin_sv2            same, likelihood updated to 20 um
  tauspin+ipkin_sv2    learned (degraded input) + classical (updated likelihood)
"""
import argparse
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "../ip_tau_reco")
import ent_robust as er  # noqa: E402
from add_levels import merge, post_feats  # noqa: E402
from breakdown import CLS, wcorr  # noqa: E402

E = "/home/rbaba/dihiggs-spin-20261006/runs/ent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="ipkin_pi0,tauspin+ipkin_pi0,tauspin_sv2,ipkin_sv2mis,ipkin_sv2,tauspin+ipkin_sv2")
    a = ap.parse_args()
    O = a.out
    d = np.load(f"{E}/dataset_HU.npz", allow_pickle=True)
    rm = er.rm(d)
    hp = np.load(f"{E}/hpred_HU.npz", allow_pickle=True)
    tsp = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9)], 1)
    post = lambda v: post_feats(merge(f"{O}/chunks", v, d["uid"]))
    learned_sv2 = None
    if glob.glob(f"{O}/hpred_HU_sv2.npz"):
        h2 = np.load(f"{O}/hpred_HU_sv2.npz", allow_pickle=True)
        assert np.array_equal(h2["uid"], d["uid"])
        learned_sv2 = np.concatenate([h2["h_pred"].reshape(-1, 6), h2["hh_pred"].reshape(-1, 9)], 1)
    build = {
        "ipkin_pi0": lambda: np.concatenate([post("ip_pi0"), rm], 1),
        "tauspin+ipkin_pi0": lambda: np.concatenate([tsp, post("ip_pi0"), rm], 1),
        "tauspin_sv2": lambda: np.concatenate([learned_sv2, rm], 1),
        "ipkin_sv2mis": lambda: np.concatenate([post("ip_sv2mis"), rm], 1),
        "ipkin_sv2": lambda: np.concatenate([post("ip_sv2"), rm], 1),
        "tauspin+ipkin_sv2": lambda: np.concatenate([learned_sv2, post("ip_sv2"), rm], 1),
    }
    c = dict(np.load(f"{O}/ent_robust_G.npz", allow_pickle=True))
    fold = np.arange(len(c["T"])) % 2
    levels = a.levels.split(",")
    for lev in levels:
        c[f"G_{lev}"] = er.regress(build[lev](), c["T"], fold, None, c["w"])[0]
        print("regressed", lev, flush=True)
    np.savez_compressed(f"{O}/ent_robust_G_extra.npz", **c)
    er.load_prep(f"{O}/ent_robust_G_extra.npz")
    T, w = c["T"][:, 0], c["w"]
    srm = np.stack([np.stack([d[f"spin_rmode{m}_{q}"] for m in range(5)], 1).argmax(1) for q in (0, 1)], 1)
    lo, hi = np.sort(srm, 1).T
    res = {}
    for lev in levels + ["tauspin", "ipkin"]:
        _, r = er.job((lev, "nominal", {}))
        G = c[f"G_{lev}"][:, 0]
        res[lev] = {"Z": r["Z"], "corr_gT": wcorr(G, T, w), "pairs": {}}
        for i in range(5):
            for j in range(i, 5):
                m = (lo == i) & (hi == j)
                if m.sum() >= 500:
                    res[lev]["pairs"][f"{CLS[i]}-{CLS[j]}"] = wcorr(G[m], T[m], w[m])
    json.dump(res, open(f"{O}/extra_levels.json", "w"), indent=1)
    for lev, r in res.items():
        p = r["pairs"]
        print(f"{lev:20s} Z={r['Z']:.3f} corr={r['corr_gT']:.3f} pi-pi={p['pi-pi']:.3f} rho-rho={p['rho-rho']:.3f} "
              f"rho-a1(3p)={p['rho-a1(3p)']:.3f} a1(3p)-a1(3p)={p['a1(3p)-a1(3p)']:.3f}")


if __name__ == "__main__":
    main()
