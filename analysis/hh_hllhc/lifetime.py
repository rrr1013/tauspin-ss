"""Lepton-lifetime templates for the lep-had channels (TauSpin's PV-referenced IP applied to
the light lepton): |d0|/sigma(d0) of tau -> l leptons versus prompt W -> l leptons.

Input: the 14 TeV parametric lep-had cohort of the 2026-10-08 run (prepared.npz; latest
SLT-like selection, 584k raw / 247,608 selected events, corrected transverse perigee d0,
scalar Gaussian smearing sigma_d0 = 10 (+) 150/pT um, x1.2 for electrons, 2% 4-sigma tails;
the 'real' variant adds an eta-dependent degradation and electron tails).  These events
only provide (a) the |d0|/sigma shapes per lepton origin and (b) the tau->l fraction of each
background group, both conditioned on a kinematic score K_lh in the same signal-fraction
range as each ATLAS SLT bin.  Yields come from HEPData, not from this cohort.

Origins: 'tau' (lepton from a tau), 'prompt' (W -> l), 'nonprompt' (heavy-flavour-like
displaced fake lepton; synthetic: isotropic perpendicular displacement, exponential with
mean 100 um, the named stress of the 2026-10-08 run).
d0-significance bins are fixed physics bins [0,1,2,3,5,inf) (not optimised); the TTVA
scenario applies |d0/sigma| < 3 (mu) / < 5 (e) before filling.
"""
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb

PREP = Path(os.environ.get("HH_LEPHAD", Path.home() / "dihiggs-latest-20261008/prepared_latest_slt_hilo/prepared.npz"))
OUT = Path(os.environ.get("HH_OUT", Path(__file__).resolve().parent / "outputs")) / "lifetime"
EDGES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, np.inf])
GROUPS = {"signal": ("hhC",), "zhf": ("zbb",), "single_h": ("tth", "zh"),
          "top": ("ttlt", "ttll", "tWll"), "fake": ("ttljp", "ttlj", "tWlj")}


def kin_score(D, sel, fold):
    X = np.c_[D["tokens"].reshape(len(sel), -1), D["globals"]].astype(np.float32)
    y = (D["proc"] == "hhC").astype(int)
    w = D["w"].astype(float)
    tr, va = sel & (fold == 0), sel & (fold == 1)
    ws = w.copy()
    for c in (0, 1):  # balance classes
        m = y == c
        ws[m] *= 1.0 / ws[m & tr].sum()
    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=ws[tr] * tr.sum())
    dva = xgb.DMatrix(X[va], label=y[va], weight=ws[va] * va.sum())
    bst = xgb.train(dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         max_depth=5, eta=0.05, min_child_weight=50, subsample=0.8,
                         colsample_bytree=0.8, nthread=8, seed=0),
                    dtr, 4000, evals=[(dva, "val")], early_stopping_rounds=80, verbose_eval=False)
    if bst.best_iteration >= 3919:
        raise RuntimeError("training ceiling")
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1)), int(bst.best_iteration)


def hist(x, w):
    h = np.histogram(x, EDGES, weights=w)[0]
    return h / max(h.sum(), 1e-300)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = dict(np.load(PREP, allow_pickle=True))
    fold = D["uid"] % 3
    from hepdata_inputs import load_latest, load_run2
    chans = {ch: (r["signal"], None) for ch, r in load_run2().items() if ch in ("slt", "ltt")}
    for run in ("run2", "run3"):
        for ch, r in load_latest(run).items():
            if ch.startswith("lephad"):
                mc = {"Hi": 1, "Lo": 0}.get(ch.split("_")[1])
                chans[f"latest_{run}:{ch}"] = (r["signal"], mc)
    res = {"edges": EDGES.tolist(), "groups": GROUPS, "variants": {}}
    rng = np.random.default_rng(12345)
    for variant in ("nominal", "real"):
        sel = D[f"sel_{variant}"].astype(bool)
        K, it = kin_score(D, sel, fold)
        d0sig, sd0 = D[f"ip_{variant}"][:, 0].astype(float), D[f"ip_{variant}"][:, 1].astype(float)
        # synthetic heavy-flavour displacement for prompt leptons (stress template)
        r = rng.exponential(100.0, len(sd0))
        d0_np = r * np.cos(rng.uniform(0, 2 * np.pi, len(sd0))) + sd0 * rng.standard_normal(len(sd0))
        np_sig = np.abs(d0_np) / sd0
        tau = D["lep_from_tau"].astype(bool)
        is_e = D["lep_is_e"].astype(bool)
        w = D["w"].astype(float)
        ev_all = sel & (fold == 2)
        vrow = {"kin_iter": it, "channels": {}}
        for ch, (Y, mc) in chans.items():
            ev = ev_all if mc is None else ev_all & (D["mass_category"] == mc)
            sig = ev & (D["proc"] == "hhC")
            o = np.argsort(-K[sig])
            cw = np.cumsum(w[sig][o])
            cw /= cw[-1]
            cum = np.r_[0, np.cumsum(Y[::-1])][::-1] / Y.sum()
            bins = []
            for b in range(len(Y)):
                lo_f, hi_f = cum[b + 1], cum[b]
                k_hi = np.interp(lo_f, cw, K[sig][o]) if lo_f > 0 else np.inf
                k_lo = np.interp(hi_f, cw, K[sig][o]) if hi_f < 1 else -np.inf
                rr = ev & (K <= k_hi) & (K > k_lo)
                row = {"k_range": [float(k_lo), float(k_hi)], "templates": {}, "tau_fraction": {}, "n": {}}
                for ttva in (False, True):
                    cut = np.where(is_e, 5.0, 3.0) if ttva else np.inf
                    ok_t, ok_p = rr & tau & (d0sig < cut), rr & ~tau & (d0sig < cut)
                    ok_n = rr & ~tau & (np_sig < cut)
                    ok_s = ok_t & (D["proc"] == "hhC")
                    row["templates"]["ttva" if ttva else "nocut"] = {
                        "tau": hist(d0sig[ok_t], w[ok_t]).tolist(),
                        "tau_signal": hist(d0sig[ok_s], w[ok_s]).tolist(),
                        "n_tau_signal": int(ok_s.sum()),
                        "prompt": hist(d0sig[ok_p], w[ok_p]).tolist(),
                        "nonprompt": hist(np_sig[ok_n], w[ok_n]).tolist()}
                for g, procs in GROUPS.items():
                    m = rr & np.isin(D["proc"], procs)
                    row["tau_fraction"][g] = float(w[m & tau].sum() / max(w[m].sum(), 1e-300)) if m.any() else None
                    row["n"][g] = int(m.sum())
                bins.append(row)
            vrow["channels"][ch] = bins
        ev = ev_all
        sig = ev & (D["proc"] == "hhC")
        o = np.argsort(-K[sig])
        cw = np.cumsum(w[sig][o])
        cw /= cw[-1]
        # pooled prompt / nonprompt shapes (|d0|/sigma of a prompt lepton is a resolution
        # pull and depends little on kinematics) and pooled background-group tau fractions
        vrow["pooled"] = {}
        for ttva in (False, True):
            cut = np.where(is_e, 5.0, 3.0) if ttva else np.inf
            ok_p, ok_n = ev & ~tau & (d0sig < cut), ev & ~tau & (np_sig < cut)
            vrow["pooled"]["ttva" if ttva else "nocut"] = {
                "prompt": hist(d0sig[ok_p], w[ok_p]).tolist(),
                "nonprompt": hist(np_sig[ok_n], w[ok_n]).tolist(),
                "n_prompt": int(ok_p.sum())}
        hi = ev & (K > np.interp(0.5, cw, K[sig][o]))  # most signal-like half of the signal
        vrow["pooled_tau_fraction_highK"] = {
            g: float(w[hi & np.isin(D["proc"], p) & tau].sum() / max(w[hi & np.isin(D["proc"], p)].sum(), 1e-300))
            for g, p in GROUPS.items()}
        vrow["pooled_n_highK"] = {g: int((hi & np.isin(D["proc"], p)).sum()) for g, p in GROUPS.items()}
        res["variants"][variant] = vrow
        print(variant, "kin iter", it, flush=True)
    json.dump(res, open(OUT / "lifetime_templates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
