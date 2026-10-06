"""Variation: H transverse correlations scaled by 0.92 (closure: real HH C_nn/C_rr
0.94/0.98 vs 1.05/1.05 for the spin-flat sample reweighted with the ideal rho_H)."""
import json
import numpy as np
import spin_gain_U as sg

orig = sg.densities


def scaled(h, zt):
    rho = orig(h, zt)
    hm, hp = h[:, 0], h[:, 1]
    rho["H"] = rho["H"] - 0.08 * (hm[:, 0] * hp[:, 0] + hm[:, 1] * hp[:, 1])
    return rho


sg.densities = scaled
d = np.load("outputs/v2/dataset_hhU.npz", allow_pickle=True)
hp = np.load("outputs/v2/hpred_hhU.npz", allow_pickle=True)
K = np.load("outputs/v2/dataset_hhU_K.npz", allow_pickle=True)["K"]
h = d["h_exact"]; ok = np.isfinite(h).all((1, 2))
rho = scaled(np.where(ok[:, None, None], h, 0.0), False)
W = {X: np.where(ok, rho[X], 1.0) for X in sg.HYPS}
o = np.argsort(K); cdf = np.cumsum(W["H"][o]) / W["H"].sum()
region = K >= np.interp(0.8, cdf, K[o])
tot = sum(v for k, v in sg.TOPBIN.items() if k != "signal")
frac = {G: sg.TOPBIN[G] / tot for G in sg.UNC}
fmap = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}
rm = np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)
X = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm], 1)
p = sg.fit_probs(X, W, region, 0)
D = sg.disc(p, frac, fmap); e = sg.edges_for(D, W, region, frac, fmap)
dens = np.stack([np.where(ok, rho[Y], 1.0) for Y in sg.HYPS], 1)
De = sg.disc(dens, frac, fmap); ee = sg.edges_for(De, W, region, frac, fmap)
out = {"spin_R": sg.significance(sg.templates(D, W, region, e), fmap)["R"],
       "exact_R": sg.significance(sg.templates(De, W, region, ee), fmap)["R"]}
print(out)
json.dump(out, open("outputs/v2/var_htrans.json", "w"))
