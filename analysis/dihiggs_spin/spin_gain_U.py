"""Primary phase-2 estimate: spin gain from a spin-flat HH sample.

The same MadGraph HH events are showered with unpolarised, uncorrelated tau
decays (Pythia TauDecays:mode = 3), so the generator spin density is 1 and the
weight to any spin hypothesis is rho_X(h_exact) itself (bounded, no density
model, no kinematic matching).  Events whose decay modes have no implemented
polarimeter keep weight 1 for every hypothesis (no spin information credited).
Hypotheses: H (signal, single H), Z_long (C_kk = +1, P_tau = -0.147),
Z_full (Z_long + C_nn = +0.49, C_rr = -0.46, measured on Pythia qqbar -> Z),
W (tau-_L tau+_R product: true-tau top), U (no spin: fakes).
Region: K-classifier score above the cut giving 100 / 50 / 20 / 5 % H-weighted
signal efficiency.  Within the region a multiclass classifier (2-fold) on
  spin      h_pred, h-h+ products, reco modes
  kin       all kinematic inputs of the K classifier
  kin+spin  both
  obs       Upsilon, x, hybrid h, reco modes
gives the composition-optimal discriminant D = p_H / sum f_b p_b;
the exact analytic likelihood ratio is the reference.
Statistics: Asimov discovery q0 with profiled Gaussian normalisation nuisances
per background group (Z+HF 10 %, top 10 %, fakes 20 %, single H 15 %) and,
optionally, one spin-template shape nuisance that moves every hypothesis
template towards the composition-averaged template by 25 % per unit (prior
N(0,1)), i.e. a 25 % uncertainty on the spin contrast.
Closure: the spin classifier applied to the real (spin-correlated) HH events
must reproduce the template of the spin-flat sample reweighted to H.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

LUMI = 3000 / 139
TOPBIN = {"signal": (1.58 + 0.0227) * 1.18, "Z+HF": (2.3 + 0.4) * 1.18, "top": (0.5 + 0.47) * 1.18,
          "fakes": (0.47 + 0.29) * 1.18, "singleH": 1.7 * 1.15}
UNC = {"Z+HF": 0.10, "top": 0.10, "fakes": 0.20, "singleH": 0.15}
HYPS = ("H", "Z", "W", "U")


def densities(h, z_transverse):
    hm, hp = h[:, 0], h[:, 1]
    nn, rr, kk = hm[:, 0] * hp[:, 0], hm[:, 1] * hp[:, 1], hm[:, 2] * hp[:, 2]
    km, kp = hm[:, 2], hp[:, 2]
    z = 1 - 0.147 * (km + kp) + kk + (0.49 * nn - 0.46 * rr if z_transverse else 0)
    return {"H": 1 + nn + rr - kk, "Z": z, "W": (1 - km) * (1 - kp), "U": np.ones_like(kk)}


def q0_asimov(s, bg, shape=None, shape_sigma=0.0):
    """Discovery significance on Asimov data n = s + b with profiled nuisances.
    bg: dict group -> template yields; shape: dict group -> (alt - nominal) shift for the signal-free part,
    plus key 'signal'."""
    groups = list(bg)
    n = s + sum(bg.values())
    keep = n > 0
    s, n = s[keep], n[keep]
    B = np.stack([bg[g][keep] for g in groups])
    sig = np.array([UNC[g] for g in groups])
    if shape is not None:
        dS = shape["signal"][keep]
        dB = np.stack([shape[g][keep] for g in groups])

    def nll(theta, mu):
        th = theta[:len(groups)]
        lam_b = (B * (1 + sig[:, None] * th[:, None])).sum(0)
        lam_s = mu * s
        pen = 0.5 * (th ** 2).sum()
        if shape is not None:
            a = theta[-1]
            lam_s = lam_s + mu * a * shape_sigma * dS
            lam_b = lam_b + a * shape_sigma * (dB * (1 + sig[:, None] * th[:, None])).sum(0)
            pen += 0.5 * a * a
        lam = np.maximum(lam_s + lam_b, 1e-9)
        return float((lam - n * np.log(lam)).sum() + pen)
    npar = len(groups) + (1 if shape is not None else 0)
    best1 = nll(np.zeros(npar), 1.0)
    r0 = minimize(nll, np.zeros(npar), args=(0.0,), method="L-BFGS-B")
    return float(np.sqrt(max(2 * (r0.fun - best1), 0.0)))


def templates(D, W, sel, edges):
    b = np.searchsorted(edges, D)
    nb = len(edges) + 1
    return {X: np.bincount(b[sel], weights=W[X][sel], minlength=nb) / W[X][sel].sum() for X in HYPS}


def significance(T, frac_map, shape_sigma=0.0):
    s = TOPBIN["signal"] * LUMI * T["H"]
    bg = {G: TOPBIN[G] * LUMI * T[frac_map[G]] for G in UNC}
    s1 = np.array([TOPBIN["signal"] * LUMI])
    one = {G: np.array([TOPBIN[G] * LUMI]) for G in UNC}
    out = {"Z1": q0_asimov(s1, one), "Z": q0_asimov(s, bg)}
    if shape_sigma > 0:
        # shape nuisance: every template moves towards the composition-weighted average template
        tot = sum(bg.values()) + s
        avg = tot / tot.sum()
        shape = {"signal": TOPBIN["signal"] * LUMI * (avg - T["H"])}
        for G in UNC:
            shape[G] = TOPBIN[G] * LUMI * (avg - T[frac_map[G]])
        out["Z_shape"] = q0_asimov(s, bg, shape, shape_sigma)
        out["R_shape"] = out["Z_shape"] / out["Z1"]
    out["R"] = out["Z"] / out["Z1"]
    return out


def fit_probs(X, W, region, seed):
    probs = np.full((len(X), len(HYPS)), np.nan)
    idx = np.flatnonzero(region)
    fold = np.zeros(len(X), int)
    fold[idx] = np.arange(len(idx)) % 2
    for f in (0, 1):
        tr = region & (fold != f)
        te = region & (fold == f)
        Xt = np.concatenate([X[tr]] * len(HYPS))
        yt = np.repeat(np.arange(len(HYPS)), tr.sum())
        wt = np.concatenate([W[Y][tr] / W[Y][tr].sum() for Y in HYPS]) * tr.sum()
        c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, min_samples_leaf=100,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                           n_iter_no_change=30, random_state=seed)
        c.fit(Xt, yt, sample_weight=wt)
        probs[te] = c.predict_proba(X[te])
    return probs


def disc(p, frac, fmap):
    p = p / p.sum(1, keepdims=True)
    den = sum(frac[G] * p[:, HYPS.index(fmap[G])] for G in UNC)
    return np.log(p[:, 0] + 1e-12) - np.log(den + 1e-12)


def edges_for(D, W, sel, frac, fmap, nbins=20):
    wb = sum(frac[G] * W[fmap[G]] for G in UNC)
    o = np.argsort(D[sel])
    c = np.cumsum(wb[sel][o]) / wb[sel].sum()
    return np.interp(np.linspace(0, 1, nbins + 1)[1:-1], c, D[sel][o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default="outputs/v2/dataset_hhU.npz")
    ap.add_argument("--uhp", default="outputs/v2/hpred_hhU.npz")
    ap.add_argument("--uk", default="outputs/v2/dataset_hhU_K.npz")
    ap.add_argument("--out", default="outputs/v2/spin_gain_U.json")
    ap.add_argument("--nboot", type=int, default=100)
    args = ap.parse_args()
    d = np.load(args.u, allow_pickle=True)
    hp = np.load(args.uhp, allow_pickle=True)
    K = np.load(args.uk, allow_pickle=True)["K"]
    assert (hp["uid"] == d["uid"]).all()
    h = d["h_exact"]
    ok = np.isfinite(h).all((1, 2))
    tot = sum(v for k, v in TOPBIN.items() if k != "signal")
    frac = {G: TOPBIN[G] / tot for G in UNC}
    fmap = {"Z+HF": "Z", "top": "W", "fakes": "U", "singleH": "H"}
    rmode = np.stack([d[f"spin_rmode{m}_{s}"] for s in (0, 1) for m in range(5)], 1)
    kin_cols = sorted(k for k in d.files if k.startswith("kin_"))
    Xs = np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rmode], 1)
    Xk = np.stack([d[c] for c in kin_cols], 1)
    obs_cols = [f"spin_{v}_{s}" for s in (0, 1) for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")]
    Xo = np.concatenate([np.stack([d[c] for c in obs_cols], 1), rmode], 1)
    sets = {"spin": Xs, "kin": Xk, "kin+spin": np.concatenate([Xk, Xs], 1), "obs": Xo}
    out = {"n_events": int(len(h)), "frac_h_defined": float(ok.mean()), "variants": {}}
    rng = np.random.default_rng(7)
    for ztag in ("Z_long", "Z_full"):
        rho = densities(np.where(ok[:, None, None], h, 0.0), ztag == "Z_full")
        W = {X: np.where(ok, rho[X], 1.0) for X in HYPS}
        o = np.argsort(K)
        cdf = np.cumsum(W["H"][o]) / W["H"].sum()
        var = {}
        for eff in (1.0, 0.5, 0.2, 0.05):
            kcut = -np.inf if eff == 1.0 else float(np.interp(1 - eff, cdf, K[o]))
            region = K >= kcut
            ess = {X: float(W[X][region].sum() ** 2 / (W[X][region] ** 2).sum()) for X in HYPS}
            res = {"k_cut": kcut, "n": int(region.sum()), "ess": ess}
            dens = np.stack([np.where(ok, rho[X], 1.0) for X in HYPS], 1)
            De = disc(dens, frac, fmap)
            e = edges_for(De, W, region, frac, fmap)
            res["exact"] = significance(templates(De, W, region, e), fmap, 0.25)
            for name, X in sets.items():
                if ztag == "Z_full" and name not in ("spin",):
                    continue
                p = fit_probs(X, W, region, seed=0)
                D = disc(p, frac, fmap)
                e = edges_for(D, W, region, frac, fmap)
                z = significance(templates(D, W, region, e), fmap, 0.25)
                if name == "spin" and args.nboot:
                    bs = []
                    idx = np.flatnonzero(region)
                    for _ in range(args.nboot):
                        cnt = np.bincount(rng.choice(idx, len(idx)), minlength=len(D)).astype(float)
                        Wb = {Y: W[Y] * cnt for Y in HYPS}
                        bs.append(significance(templates(D, Wb, cnt > 0, e), fmap)["R"])
                    z["R_boot_sd"] = float(np.std(bs))
                res[name] = z
                if name == "spin" and ztag == "Z_long" and eff == 0.2:
                    np.savez_compressed(Path(args.out).with_name("spin_D_eff20.npz"), D=D, region=region,
                                        probs=p, dens=dens, **{f"w_{Y}": W[Y] for Y in HYPS}, edges=e)
            if "kin" in res:
                res["increment_kin_to_kinspin"] = res["kin+spin"]["Z"] / res["kin"]["Z"]
            var[f"sig_eff_{int(eff * 100)}"] = res
            print(ztag, eff, "n", res["n"], "ESS", {k: round(v) for k, v in ess.items()},
                  "| exact R", round(res["exact"]["R"], 4), "| spin R", round(res["spin"]["R"], 4),
                  "(shape", round(res["spin"].get("R_shape", float("nan")), 4), ")",
                  *(["| kin", round(res["kin"]["R"], 4), "| kin+spin", round(res["kin+spin"]["R"], 4),
                     "| inc", round(res["increment_kin_to_kinspin"], 4), "| obs", round(res["obs"]["R"], 4)]
                    if "kin" in res else []), flush=True)
        out["variants"][ztag] = var
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
