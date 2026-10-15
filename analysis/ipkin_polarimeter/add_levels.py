"""Merge the posterior-polarimeter chunks and add them as levels to copies of the cached
statistics of ent_robust.py (g_T, g_L) and ent_cp.py (g_T, g_A).

Every level goes through the same cross-fitted regression as the existing levels
(ent_robust.regress: HistGradientBoosting, folds event % 2, event weights), with the same
feature layout as the tauspin level: E[h-], E[h+] (6), E[h-_i h+_j] (9), reco decay modes.
Undefined posterior means (3pXn sides) enter as 0, like the missing polarimeters elsewhere.

Levels
  ipkin              IP/SV + MET + unpolarised decay prior
  ipkin_m125         same + m_tautau prior (125, 3 GeV)
  ipkin_noip         MET + prior only
  ipkin_noip_m125    MET + prior + m_tautau prior
  tauspin+ipkin      learned regressor outputs and ipkin posterior means together
  tauspin+ipkin_m125
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ip_tau_reco"))
from ent_cp import targets as cp_targets  # noqa: E402
from ent_robust import regress, rm, targets  # noqa: E402

VARIANTS = ("ip", "ip_m125", "noip", "noip_m125")


def merge(chunkdir, v, uid):
    files = sorted(glob.glob(f"{chunkdir}/{v}_[0-9]*.npz"))
    parts = [dict(np.load(f)) for f in files if not f.endswith(".tmp.npz")]
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if parts[0][k].ndim >= 1}
    assert np.array_equal(z["uid"], uid), f"{v}: chunks do not cover the dataset in order"
    return z


def post_feats(z):
    return np.nan_to_num(np.concatenate([z["Eh"].reshape(-1, 6), z["Ehh"].reshape(-1, 9)], 1), nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    R = "/home/rbaba/ipkin-polarimeter-20261016"
    E = "/home/rbaba/dihiggs-spin-20261006/runs/ent"
    ap.add_argument("--data", default=f"{E}/dataset_HU.npz")
    ap.add_argument("--hpred", default=f"{E}/hpred_HU.npz")
    ap.add_argument("--chunks", default=f"{R}/out/chunks")
    ap.add_argument("--robust-src", default=f"{E}/ent_robust_G.npz")
    ap.add_argument("--cp-src", default=f"{E}/ent_cp_G.npz")
    ap.add_argument("--outdir", default=f"{R}/out")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    h, w = d["h_exact"], d["w"]
    fold = np.arange(len(h)) % 2
    post = {v: merge(args.chunks, v, d["uid"]) for v in VARIANTS}
    for v, z in post.items():
        np.savez_compressed(f"{args.outdir}/post_{v}.npz", **z)
    tsp = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9)], 1)
    name = {"ip": "ipkin", "ip_m125": "ipkin_m125", "noip": "ipkin_noip", "noip_m125": "ipkin_noip_m125"}
    feats = {name[v]: np.concatenate([post_feats(post[v]), rm(d)], 1) for v in VARIANTS}
    feats["tauspin+ipkin"] = np.concatenate([tsp, post_feats(post["ip"]), rm(d)], 1)
    feats["tauspin+ipkin_m125"] = np.concatenate([tsp, post_feats(post["ip_m125"]), rm(d)], 1)
    for src, T, fname in ((args.robust_src, targets(h)[1], "ent_robust_G.npz"),
                         (args.cp_src, cp_targets(h)[:, :2], "ent_cp_G.npz")):
        dst = f"{args.outdir}/{fname}"
        shutil.copy(src, dst)
        c = dict(np.load(dst, allow_pickle=True))
        for lev, X in feats.items():
            c[f"G_{lev}"] = regress(X, T, fold, None, w)[0]
            print(fname, lev, "corr", [round(float(np.corrcoef(c[f"G_{lev}"][:, k], T[:, k])[0, 1]), 4) for k in range(2)],
                  flush=True)
        for lev in ("tauspin", "phicp", "textbook_pair"):
            print(fname, lev, "corr", [round(float(np.corrcoef(c[f"G_{lev}"][:, k], T[:, k])[0, 1]), 4) for k in range(2)])
        np.savez_compressed(dst, **c)


if __name__ == "__main__":
    main()
