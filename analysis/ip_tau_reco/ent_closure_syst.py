"""Closure and response systematics for the entanglement feasibility.

Closure: the (g_s, g_z) regressors are trained on the spin-flat H(125)+jet sample and
applied to the held-out real H(125)+jet and Z(125)+jet events of the regressor
training sample (same reconstruction, never used to train the h regressor).  The
spin-flat sample reweighted to H (p = 1) must reproduce the real H; reweighted to
Z (longitudinal, with and without the measured transverse terms) it is compared
with the real Z(125)+jet, whose production adds transverse correlations.

Response systematic: a nuisance theta scales the spin-dependent part of every
template, T_X(theta) = T_flat + (1 + sigma theta) (T_X - T_flat), with a unit
Gaussian prior; sigma = 0.1, 0.2, 0.3 (an in-situ Z -> tau tau calibration would
fix sigma).  Z_ent is recomputed for the mass-binned HL-LHC categories.
"""
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from entanglement_feasibility import spin_s, templates, weights, z_target
from ent_hllhc import CATS, LUMI

FLAT = "outputs/ent/dataset_HU.npz"
FLAT_HP = "outputs/ent/hpred_HU.npz"
HO = "/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/outputs/v1/train_holdout_slim.npz"
HO_HP = "/Users/ryunosuke/Projects/tauspin/analysis/dihiggs_spin/outputs/v1/hpred_partial_holdout.npz"


def chi2(a, wa, b, wb, edges):
    ha, _ = np.histogram(a, edges, weights=wa)
    va, _ = np.histogram(a, edges, weights=wa ** 2)
    hb, _ = np.histogram(b, edges, weights=wb)
    vb, _ = np.histogram(b, edges, weights=wb ** 2)
    na, nb = wa.sum(), wb.sum()
    m = (va + vb) > 0
    return float(((ha / na - hb / nb) ** 2 / (va / na ** 2 + vb / nb ** 2))[m].sum()), int(m.sum())


def feats(hpred, hh, rmode):
    return np.concatenate([hpred.reshape(-1, 6), hh.reshape(-1, 9), rmode], 1)


def z_ent_shape(T, S, B, bunc, sig_resp):
    """Asimov p=1 vs p=1/3 with S free, B constrained, response-scale nuisance."""
    n = S * T["H"] + B * T["Z"]
    flat = T["U"]

    def nll(par, Tsig_key):
        Sv, Bv = np.exp(par[:2])
        th = par[2]
        k = 1 + sig_resp * th
        Ts = flat + k * (T[Tsig_key] - flat)
        Tb = flat + k * (T["Z"] - flat)
        lam = np.maximum(Sv * Ts + Bv * Tb, 1e-12)
        return float((lam - n * np.log(lam)).sum() + 0.5 * ((Bv - B) / (bunc * B)) ** 2 + 0.5 * th ** 2)
    f1 = minimize(nll, [np.log(S), np.log(B), 0.0], args=("H",), method="Nelder-Mead",
                  options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10)).fun
    best = np.inf
    for th0 in (0.0, 3.0, 8.0):
        r = minimize(nll, [np.log(S), np.log(B), th0], args=("W13",), method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
        best = min(best, r.fun)
    return float(np.sqrt(max(2 * (best - f1), 0)))


def main():
    d = np.load(FLAT, allow_pickle=True)
    hp = np.load(FLAT_HP, allow_pickle=True)
    h = d["h_exact"]
    W, s, ok = weights(h, False)
    rm = np.stack([d[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)
    X = feats(hp["h_pred"], hp["hh_pred"], rm)
    zt = z_target(h, False)
    regs = []
    for target in (np.where(ok, s, 0.0), zt):
        r = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, random_state=0)
        regs.append(r.fit(X, target))
    G = np.stack([r.predict(X) for r in regs], 1)
    # hold-out real H / Z
    ho = np.load(HO, allow_pickle=True)
    hh = np.load(HO_HP, allow_pickle=True)
    rmh = np.stack([ho[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)
    Xh = feats(hh["h_pred"], hh["hh_pred"], rmh)
    Gh = np.stack([r.predict(Xh) for r in regs], 1)
    isH = ho["proc"] == "trainH"
    out = {"closure": {}}
    hz = np.where(np.isfinite(ho["h_exact"]).all((1, 2))[:, None, None], ho["h_exact"], 0.0)
    for j, nm in enumerate(("g_s", "g_z")):
        e = np.quantile(G[:, j], np.linspace(0, 1, 21))
        e[0], e[-1] = -np.inf, np.inf
        out["closure"][f"{nm}_flat->H_vs_realH"] = chi2(G[:, j], W["H"], Gh[isH, j], np.ones(isH.sum()), e)
        out["closure"][f"{nm}_flat->Zlong_vs_realZ"] = chi2(G[:, j], W["Z"], Gh[~isH, j], np.ones((~isH).sum()), e)
        Wt, _, _ = weights(h, True)
        out["closure"][f"{nm}_flat->Zfull_vs_realZ"] = chi2(G[:, j], Wt["Z"], Gh[~isH, j], np.ones((~isH).sum()), e)
        out["closure"][f"{nm}_means"] = {"flat->H": float(np.average(G[:, j], weights=W["H"])),
                                          "realH": float(Gh[isH, j].mean()),
                                          "flat->Z": float(np.average(G[:, j], weights=W["Z"])),
                                          "flat->Zfull": float(np.average(G[:, j], weights=Wt["Z"])),
                                          "realZ": float(Gh[~isH, j].mean()),
                                          "flat->W13": float(np.average(G[:, j], weights=W["W13"]))}
    print(json.dumps(out["closure"], indent=1))
    np.savez_compressed("outputs/ent/closure_g.npz", G=G, Gh=Gh, isH=isH, wH=W["H"], wZ=W["Z"], wW13=W["W13"],
                        wU=W["U"], wZfull=weights(h, True)[0]["Z"])
    # response systematic on the mass-binned HL-LHC categories
    T = templates(G, W)
    acc = json.load(open("outputs/ent/mass_acc.json"))
    out["response_syst"] = {}
    for sig in (0.0, 0.1, 0.2, 0.3):
        z2 = 0.0
        for k, (s_, z, f, t, o) in CATS.items():
            for i in range(len(acc["S"])):
                S = s_ * LUMI * 1.13 * acc["S"][i]
                B = (z * acc["Z"][i] + f * acc["F"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15
                z2 += z_ent_shape(T, S, B, 0.02, sig) ** 2
        out["response_syst"][str(sig)] = float(np.sqrt(z2))
        print("response sigma", sig, "Z_ent", round(np.sqrt(z2), 3), flush=True)
    json.dump(out, open("outputs/ent/closure_syst.json", "w"), indent=1)


if __name__ == "__main__":
    main()
