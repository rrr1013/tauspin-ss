"""Regression-seed spread of the levels that carry the claims: re-run the cross-fitted
HistGradientBoosting of ent_robust with other random_state values (early-stopping split) and
refit the nominal entanglement test (sidebands only)."""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "../ip_tau_reco")
import ent_robust as er  # noqa: E402
from add_levels import post_feats  # noqa: E402

O = sys.argv[1]
d = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz", allow_pickle=True)
hp = np.load("/home/rbaba/dihiggs-spin-20261006/runs/ent/hpred_HU.npz", allow_pickle=True)
ip = np.load(f"{O}/post_ip.npz")
tsp = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9)], 1)
F = {"tauspin": np.concatenate([tsp, er.rm(d)], 1), "ipkin": np.concatenate([post_feats(ip), er.rm(d)], 1),
     "tauspin+ipkin": np.concatenate([tsp, post_feats(ip), er.rm(d)], 1)}
c = dict(np.load(f"{O}/ent_robust_G.npz", allow_pickle=True))
T, w = c["T"], c["w"]
fold = np.arange(len(T)) % 2
res = {}
for seed in (1, 2):
    for lev, X in F.items():
        G = np.zeros_like(T)
        for j in range(2):
            for f in (0, 1):
                r = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                                  l2_regularization=1.0, early_stopping=True, random_state=seed)
                r.fit(X[fold != f], T[fold != f, j], sample_weight=w[fold != f])
                G[fold == f, j] = r.predict(X[fold == f])
        c[f"G_{lev}_s{seed}"] = G
np.savez_compressed(f"{O}/ent_robust_G_seeds.npz", **c)
er.load_prep(f"{O}/ent_robust_G_seeds.npz")
for lev in F:
    for seed in (0, 1, 2):
        name = lev if seed == 0 else f"{lev}_s{seed}"
        _, r = er.job((name, "nominal", {}))
        res[name] = {"Z": r["Z"], "corr_gT": float(np.corrcoef(er.PREP["P"]["levels"][name]["G"][:, 0], T[:, 0])[0, 1])}
json.dump(res, open(f"{O}/seed_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
