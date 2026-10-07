"""Add a tauspin-type level (regressor predictions + decay modes, as in ent_robust.py) to the
cached statistics of ent_robust.py (g_T, g_L) and ent_cp.py (g_T, g_A).

Used for the information ablation 'tauspin_lhcinfo': the tauspin network retrained
without SV, the longitudinal IP component and MET (train_h.py --drop sv_,ip_k,met_),
i.e. with the transverse IP and the visible decay products only.
"""
import argparse

import numpy as np

from ent_cp import targets as cp_targets
from ent_robust import regress, rm, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hpred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--robust-cache", required=True)
    ap.add_argument("--cp-cache", required=True)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    X = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm(d)], 1)
    h, w = d["h_exact"], d["w"]
    fold = np.arange(len(h)) % 2
    for cache, T in ((args.robust_cache, targets(h)[1]), (args.cp_cache, cp_targets(h)[:, :2])):
        c = dict(np.load(cache, allow_pickle=True))
        c[f"G_{args.name}"] = regress(X, T, fold, None, w)[0]
        print(cache, args.name, "corr", [round(float(np.corrcoef(c[f"G_{args.name}"][:, k], T[:, k])[0, 1]), 4) for k in range(2)], flush=True)
        np.savez_compressed(cache, **c)


if __name__ == "__main__":
    main()
