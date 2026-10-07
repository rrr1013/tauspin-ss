"""Probes requested by the skeptical review of the tau_lep tau_had lepton-IP gain.

Baseline K' = kinematics + the two non-IP inputs that were inside the IP set (E_lep/E_tau from
the MMC, lepton flavour), so that the ratio isolates the impact parameter.
  P0  nominal resolution: K', K'+d0 (|d0|/sigma, sigma_d0), K'+IP (+ local-frame vector, z0)
  P1  null control: true impact vectors permuted within (pT, flavour) bins -> R must be ~1
  P2  realistic resolution: (+) 10 um beam spot, x (1 + 0.2 |eta|), electron tails 7 % at 4x
  P3  non-prompt injection (with P2 resolution): a fraction f of the prompt-lepton fake events
      (tt -> l + jets, tW -> l + jets) gets a b-decay-like impact vector (|IP| exponential with
      mean 100 um, or 50 um), f = 0.3, 0.5
  P4  MC statistics: merging threshold 50 / 200 / 400; Poisson bootstrap of the background MC
  P5  baseline with the standard track-to-vertex requirement (P2 resolution)
Factorised ATLAS-anchored estimate (with P2 resolution): "other" as prompt or 50/50, an IP
template width nuisance (+-15 % resolution, morphing towards the x1.5 templates, profiled), and
the non-prompt injection into the fakes template.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from build_lephad import lepton_ip_features
from classify_lephad import crossfit, evaluate
from factorized_lephad import CLASS, LUMI, TABLES, q0
from reco import eta, pt

IPCOLS = ("d0sig", "absd0", "z0sig", "n", "r", "k", "sd0")
FAKE_PROMPT = ("ttljp", "tWlj")


def ip_feats(D, ip_true=None, **kw):
    d = {"lep": D["lep_p4"], "lep_id": D["lep_id"], "lep_ip_true": D["lep_ip_true"]}
    mm = {"nu4": np.stack([D["mmc_nu_lep"], D["mmc_nu_lep"]], 1), "e_tau": D["mmc_e_tau"]}
    f = lepton_ip_features(d, mm, kw.pop("ipscale", 1.0), ip_true=ip_true, **kw)
    return {k.replace("lip_", ""): v for k, v in f.items()}


def nonprompt_ip(lep, n_sel, mean_um, rng):
    """b-decay-like impact vector: |IP| exponential, direction uniform perpendicular to the lepton."""
    k = lep[:, :3] / np.linalg.norm(lep[:, :3], axis=-1, keepdims=True)
    a = np.cross(k, np.array([0.0, 0.0, 1.0]))
    a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = np.cross(k, a)
    ph = rng.uniform(0, 2 * np.pi, len(lep))
    r = rng.exponential(mean_um, len(lep))
    return (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b) * r[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    D = dict(np.load(args.data, allow_pickle=True))
    proc, w, uid = D["proc"], D["w"], D["uid"]
    base = (D["kin_m_bb"] < 150) & (D["kin_m_tautau"] > 60)
    y = (proc == "hhC").astype(int)
    pidx = np.searchsorted(np.unique(proc), proc)
    kin = sorted(k for k in D if k.startswith("kin_") and k != "kin_mmc_valid")
    Kp = np.stack([D[c] for c in kin] + [D["lip_x"], D["lip_is_e"]], 1)
    lep = D["lep_p4"]
    rng = np.random.default_rng(7)
    res = {"direct": {}, "factorised": {}}

    def run(name, X, sel=base, wts=w, min_ess=100.0, seeds=args.seeds):
        zs = []
        for seed in range(seeds):
            fold = ((uid // 3 + seed) * 7 + pidx * 13 + uid) % 3
            sc = crossfit(X[sel], y[sel], wts[sel], fold[sel], seed)
            zs.append(evaluate(sc, y[sel], wts[sel], proc[sel], D["trigger"][sel], min_ess)["combined"])
        res["direct"][name] = {"mean": float(np.mean(zs)), "sd": float(np.std(zs, ddof=1)) if len(zs) > 1 else 0.0, "runs": zs}
        print(name, round(np.mean(zs), 3), "+-", round(res["direct"][name]["sd"], 3), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

    def with_ip(F, cols):
        return np.concatenate([Kp, np.stack([F[c] for c in cols], 1)], 1)

    nom = ip_feats(D)
    real = ip_feats(D, beamspot=10.0, eta_slope=0.2, e_tail=0.07)
    # P0
    run("P0 K'", Kp)
    run("P0 K'+d0", with_ip(nom, ("d0sig", "sd0")))
    run("P0 K'+IP", with_ip(nom, IPCOLS))
    # P1 null: permute true IP within (pT, flavour) bins
    ipt = D["lep_ip_true"].copy()
    ptb = np.digitize(pt(lep), [20, 27, 35, 45, 60, 80, 120])
    key = ptb * 2 + (np.abs(D["lep_id"]) == 11)
    perm = np.arange(len(w))
    for kk in np.unique(key):
        idx = np.flatnonzero(key == kk)
        perm[idx] = rng.permutation(idx)
    null = ip_feats(D, ip_true=ipt[perm])
    run("P1 null K'+IP", with_ip(null, IPCOLS))
    # P2 realistic resolution
    run("P2 real K'+d0", with_ip(real, ("d0sig", "sd0")))
    run("P2 real K'+IP", with_ip(real, IPCOLS))
    # P5 TTVA baseline with the realistic resolution
    tv = base & (real["ttva"] > 0)
    run("P5 ttva real K'", Kp, sel=tv)
    run("P5 ttva real K'+IP", with_ip(real, IPCOLS), sel=tv)
    # P3 non-prompt injection
    cand = np.isin(proc, FAKE_PROMPT) & ~D["lep_from_tau"]
    for frac in (0.3, 0.5):
        for mean in (100.0, 50.0):
            ip_inj = D["lep_ip_true"].copy()
            pick = cand & (rng.random(len(w)) < frac)
            ip_inj[pick] = nonprompt_ip(lep[pick], pick.sum(), mean, rng)
            F = ip_feats(D, ip_true=ip_inj, beamspot=10.0, eta_slope=0.2, e_tail=0.07)
            run(f"P3 nonprompt f{frac} mean{mean:g}um K'+IP", with_ip(F, IPCOLS))
    # P4 MC statistics
    for me in (50.0, 200.0, 400.0):
        run(f"P4 min_ess{me:g} K'", Kp, min_ess=me)
        run(f"P4 min_ess{me:g} K'+IP", with_ip(nom, IPCOLS), min_ess=me)
    for b in range(3):
        wb = np.where(y == 1, w, w * rng.poisson(1.0, len(w)))
        run(f"P4 bootstrap{b} K'", Kp, wts=wb, seeds=1)
        run(f"P4 bootstrap{b} K'+IP", with_ip(nom, IPCOLS), wts=wb, seeds=1)

    # ---------------- factorised (ATLAS composition), realistic resolution, with nuisances
    real15 = ip_feats(D, ipscale=1.15, beamspot=10.0, eta_slope=0.2, e_tail=0.07)
    tau_origin = D["lep_from_tau"].astype(int)
    cols = ("d0sig", "absd0", "z0sig", "n", "r", "k", "sd0", "is_e")
    Xr = np.stack([real[c] for c in cols] + [pt(lep)], 1)
    Xr15 = np.stack([real15[c] for c in cols] + [pt(lep)], 1)
    fold = uid % 3
    s_r, s_r15 = np.zeros(len(w)), np.zeros(len(w))
    for f in range(3):
        tr = base & (fold != f)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0).fit(Xr[tr], tau_origin[tr])
        s_r[fold == f] = clf.predict_proba(Xr[fold == f])[:, 1]
        s_r15[fold == f] = clf.predict_proba(Xr15[fold == f])[:, 1]
    inj = D["lep_ip_true"].copy()
    pick = cand & (rng.random(len(w)) < 0.5)
    inj[pick] = nonprompt_ip(lep[pick], pick.sum(), 100.0, rng)
    Fi = ip_feats(D, ip_true=inj, beamspot=10.0, eta_slope=0.2, e_tail=0.07)
    Xi = np.stack([Fi[c] for c in cols] + [pt(lep)], 1)
    s_i = np.zeros(len(w))
    for f in range(3):
        tr = base & (fold != f)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0).fit(Xr[tr], tau_origin[tr])
        s_i[fold == f] = clf.predict_proba(Xi[fold == f])[:, 1]
    for cat, tv_ in (("SLT", 0), ("LTT", 1)):
        m0 = base & (D["trigger"] == tv_)
        edges = np.quantile(s_r[m0 & (y == 1)], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        keep = real["ttva"] > 0     # templates of TTVA-passing leptons (published-yield reading)
        def tmpl(score, procs=None, prompt=None):
            m = m0 & keep
            if procs is not None:
                m &= np.isin(proc, procs)
            if prompt is not None:
                m &= (tau_origin == (0 if prompt else 1))
            h, _ = np.histogram(score[m], edges, weights=np.abs(w[m]))
            return h / max(h.sum(), 1e-30)
        out = {}
        for other_mode in ("prompt", "50/50"):
            for fake_mode in ("as simulated", "50% non-prompt"):
                names = list(TABLES[cat])
                Y = {c: TABLES[cat][c][0] * LUMI * TABLES[cat][c][1] for c in names}
                T, T15 = {}, {}
                for c in names:
                    if c == "other":
                        tp, tt = tmpl(s_r, prompt=True), tmpl(s_r, prompt=False)
                        tp15, tt15 = tmpl(s_r15, prompt=True), tmpl(s_r15, prompt=False)
                        T[c] = tp if other_mode == "prompt" else 0.5 * (tp + tt)
                        T15[c] = tp15 if other_mode == "prompt" else 0.5 * (tp15 + tt15)
                    else:
                        sc = s_i if (c == "fakes" and fake_mode != "as simulated") else s_r
                        T[c] = tmpl(sc, CLASS[c])
                        T15[c] = tmpl(s_r15, CLASS[c])
                prior = np.array([TABLES[cat][c][2] for c in names if c != "HH"])
                bk = [c for c in names if c != "HH"]
                z_un = np.sqrt(q0(np.array([Y["HH"]]), np.array([[Y[c]] for c in bk]), prior))
                # width nuisance: morph all templates towards the +15 % resolution templates
                from scipy.optimize import minimize
                n = Y["HH"] * T["HH"] + sum(Y[c] * T[c] for c in bk)
                def nll(par, with_sig):
                    th = par[-1]
                    lam = sum(Y[c] * (1 + TABLES[cat][c][2] * par[i]) * np.clip(T[c] + th * (T15[c] - T[c]), 1e-12, None)
                              for i, c in enumerate(bk))
                    if with_sig:
                        lam = lam + Y["HH"] * np.clip(T["HH"] + th * (T15["HH"] - T["HH"]), 1e-12, None)
                    return float((lam - n * np.log(np.maximum(lam, 1e-12))).sum() + 0.5 * (par ** 2).sum())
                l0 = minimize(nll, np.zeros(len(bk) + 1), args=(False,), method="L-BFGS-B").fun
                l1 = minimize(nll, np.zeros(len(bk) + 1), args=(True,), method="L-BFGS-B").fun
                z_sp = np.sqrt(max(2 * (l0 - l1), 0))
                out[f"other={other_mode}, fakes={fake_mode}"] = {"Z_bin": float(z_un), "Z_split": float(z_sp), "R": float(z_sp / z_un)}
                print("factorised", cat, other_mode, fake_mode, "R", round(z_sp / z_un, 3), flush=True)
        res["factorised"][cat] = out
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
