"""Robust HL-LHC projection of the H -> tau_had tau_had entanglement test.

Answers the skeptical review of the first projection (ent_final.py):
  M1/M4/m1  the background spin shape is no longer taken from simulation.  In each
            category the genuine-tau background (Z, top, other) and the jet fakes get
            their own free shape in the spin statistic, shared by the nine m_tautau
            bins; the bins differ in signal and background composition, so the
            sidebands measure the background shapes in situ (no MC template, no MC
            template statistics).  Stress tests let the true background shape drift
            with m_tautau, which this model cannot absorb.
  M2        the null hypothesis is the whole set of separable Bell-diagonal spin
            states, |c_n| + |c_r| + |c_k| <= 1 (the octahedron), profiled; the
            statistic is the generalised likelihood ratio against all physical
            Bell-diagonal states (the tetrahedron).  The Werner point p = 1/3 is kept
            as a variant.
  M3        separate transverse and longitudinal analysing-power responses k_T, k_L,
            global (correlated over all categories and mass bins), priors 0 / 10 / 20 %
            or k_T free.
  M5        stronger baselines: textbook observables with pair products, and the
            observables of the LHC CP analyses (impact-parameter vector for pi, rho
            decay plane with the y sign for pi pi0, secondary-vertex direction for three
            prongs; pair products of their transverse directions) plus the textbook set.
  M6        luminosity is scaled inside one fit with global nuisances.

Spin statistic: (g_T, g_L) = E_flat[(h-_n h+_n + h-_r h+_r, h-_k h+_k) | x] by
cross-fitted regression on the spin-flat H(125)+jet sample, 6 x 6 quantile bins.  The
signal template of a Bell-diagonal state c is exact: A0 + k_T (c_n D_n + c_r D_r) +
k_L c_k D_k with D_i the binned sums of h-_i h+_i of the flat sample.
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from ent_final import PT_RANGE
from ent_hllhc import CATS, LUMI
from entanglement_feasibility import weights

torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
OCT = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.]])
TET = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, -1.]])
TRUE_C = np.array([1, 1, -1.])
NQ = 6


# ----------------------------------------------------------------------------- features
def rm(d):
    return np.stack([d[f"spin_rmode{m}_{q}"] for q in (0, 1) for m in range(5)], 1)


def textbook(d):
    return np.concatenate([np.stack([d[f"spin_{v}_{q}"] for q in (0, 1)
                                     for v in ("upsilon", "x", "hhyb_n", "hhyb_r", "hhyb_k")], 1), rm(d)], 1)


def textbook_pair(d):
    hy = np.stack([np.stack([d[f"spin_hhyb_{c}_{q}"] for c in "nrk"], 1) for q in (0, 1)], 1)
    return np.concatenate([textbook(d), (hy[:, 0, :, None] * hy[:, 1, None, :]).reshape(-1, 9)], 1)


def cp_vector(d, q):
    """Transverse direction used by the LHC CP analyses: IP (pi), rho plane (pi pi0), SV (3 prongs)."""
    mode = np.stack([d[f"spin_rmode{m}_{q}"] for m in range(5)], 1)
    ip = np.stack([d[f"spin_ip_n_{q}"], d[f"spin_ip_r_{q}"]], 1)
    pi0 = np.stack([d[f"spin_pi00_dn_{q}"] + d[f"spin_pi01_dn_{q}"], d[f"spin_pi00_dr_{q}"] + d[f"spin_pi01_dr_{q}"]], 1)
    ch = np.stack([d[f"spin_ch0_dn_{q}"], d[f"spin_ch0_dr_{q}"]], 1)
    rho = d[f"spin_upsilon_{q}"][:, None] * (ch - pi0)
    sv = np.stack([d[f"spin_sv_n_{q}"], d[f"spin_sv_r_{q}"]], 1)
    u = np.where(mode[:, :1] > 0, ip, np.where((mode[:, 1:2] + mode[:, 2:3]) > 0, rho, sv))
    mag = np.linalg.norm(u, axis=1)
    return u / np.maximum(mag, 1e-9)[:, None], np.log1p(mag)


def phicp(d):
    (u0, m0), (u1, m1) = cp_vector(d, 0), cp_vector(d, 1)
    prod = (u0[:, :, None] * u1[:, None, :]).reshape(-1, 4)
    return np.concatenate([textbook_pair(d), u0, u1, m0[:, None], m1[:, None], prod], 1)


# ----------------------------------------------------------------------------- statistic
def targets(h):
    ok = np.isfinite(h).all((1, 2))
    hz = np.where(ok[:, None, None], h, 0.0)
    p = hz[:, 0] * hz[:, 1]  # (N, 3): h-_i h+_i
    return p, np.stack([p[:, 0] + p[:, 1], p[:, 2]], 1)


def regress(X, T, fold, Xapply=None, w=None):
    G = np.zeros((len(X), T.shape[1]))
    Ga = None if Xapply is None else np.zeros((len(Xapply), T.shape[1]))
    for j in range(T.shape[1]):
        for f in (0, 1):
            r = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                              l2_regularization=1.0, early_stopping=True, random_state=0)
            r.fit(X[fold != f], T[fold != f, j], sample_weight=None if w is None else w[fold != f])
            G[fold == f, j] = r.predict(X[fold == f])
            if Xapply is not None:
                Ga[:, j] += 0.5 * r.predict(Xapply)
    return G, Ga


class Binner:
    def __init__(self, G, nq=NQ):
        self.nq = nq
        self.e1 = np.quantile(G[:, 0], np.linspace(0, 1, nq + 1)[1:-1])
        b1 = np.searchsorted(self.e1, G[:, 0])
        self.e2 = [np.quantile(G[b1 == i, 1], np.linspace(0, 1, nq + 1)[1:-1]) for i in range(nq)]
        self.nb = nq * nq

    def __call__(self, G):
        a = np.searchsorted(self.e1, G[:, 0])
        out = a * self.nq
        for i in range(self.nq):
            m = a == i
            out[m] += np.searchsorted(self.e2[i], G[m, 1])
        return out


def category_templates(b, nb, p, wz, ptt, w):
    A0, D, Z = [], [], []
    for c in CATS:
        lo, hi = PT_RANGE[c]
        m = (ptt >= lo) & (ptt < hi)
        n = w[m].sum()
        A0.append(np.bincount(b[m], weights=w[m], minlength=nb) / n)
        D.append(np.stack([np.bincount(b[m], weights=w[m] * p[m, i], minlength=nb) / n for i in range(3)]))
        z = np.bincount(b[m], weights=w[m] * wz[m], minlength=nb)
        Z.append(z / z.sum())
    return np.array(A0), np.array(D), np.array(Z)


def yields(acc, scale):
    S, Bg, Bf = [], [], []
    for s_, zz, f, t, o in CATS.values():
        S.append([s_ * LUMI * 1.13 * acc["S"][i] * scale for i in range(len(acc["S"]))])
        Bg.append([(zz * acc["Z"][i] + t * acc["T"][i] + o * acc["O"][i]) * LUMI * 1.15 * scale for i in range(len(acc["S"]))])
        Bf.append([f * acc["F"][i] * LUMI * 1.15 * scale for i in range(len(acc["S"]))])
    return np.array(S), np.array(Bg), np.array(Bf)


# ----------------------------------------------------------------------------- model / fit
class Model:
    def __init__(self, A0, D, S, Bg, Bf, sig_T, sig_L, cats=None, tau_g=0.0, tau_f=0.0):
        sel = np.arange(len(A0)) if cats is None else np.asarray(cats)
        t = lambda a: torch.tensor(np.asarray(a)[sel])
        self.A0, self.D, self.S, self.Bg, self.Bf = t(A0), t(D), t(S), t(Bg), t(Bf)
        self.sig_T, self.sig_L = sig_T, sig_L
        self.C, self.I, self.nb = self.S.shape[0], self.S.shape[1], self.A0.shape[1]
        # auxiliary measurements of the background shapes (embedding / fake-factor control
        # samples) with tau times the signal-region yield; 0 = shapes free (sidebands only)
        self.tau_g, self.tau_f = tau_g, tau_f
        self.aux = None

    def sig_shape(self, c, kT, kL):
        s = self.A0 + kT * (c[0] * self.D[:, 0] + c[1] * self.D[:, 1]) + kL * c[2] * self.D[:, 2]
        s = torch.clamp(s, min=1e-12)
        return s / s.sum(1, keepdim=True)

    def lam(self, c, mu, kT, kL, ag, af, shg, shf):
        sig = mu * self.S[:, :, None] * self.sig_shape(c, kT, kL)[:, None, :]
        g = (self.Bg * torch.exp(ag))[:, :, None] * shg[:, None, :]
        f = (self.Bf * torch.exp(af))[:, :, None] * shf[:, None, :]
        return sig + g + f

    def asimov(self, c, shg_true, shf_true, drift=None):
        """drift: (target shape (C, nb), t_i over mass bins) replaces shg_true by a mass-dependent mix."""
        with torch.no_grad():
            one = torch.tensor(1.0)
            shg = torch.tensor(shg_true)
            lam = self.lam(torch.tensor(c), one, one, one, torch.zeros(self.C, self.I), torch.zeros(self.C, self.I),
                           shg, torch.tensor(shf_true))
            if drift is not None:
                X, tt = torch.tensor(drift[0]), torch.tensor(drift[1])
                mix = (1 - tt)[None, :, None] * shg[:, None, :] + tt[None, :, None] * X[:, None, :]
                lam = lam - (self.Bg[:, :, None] * shg[:, None, :]) + self.Bg[:, :, None] * mix
            return lam

    def set_aux(self, shg_aux, shf_aux):
        self.aux = (self.tau_g * self.Bg.sum(1)[:, None] * torch.tensor(shg_aux),
                    self.tau_f * self.Bf.sum(1)[:, None] * torch.tensor(shf_aux))

    def fit(self, n, family, starts, init_sh):
        """Profile all nuisances; family: 'oct', 'tet' or a fixed 3-vector."""
        best = (np.inf, None)
        for st in starts:
            P = {"lmu": torch.zeros(1), "ag": torch.zeros(self.C, self.I), "af": torch.zeros(self.C, self.I),
                 "tg": torch.log(torch.tensor(init_sh[0]) + 1e-9).clone(), "tf": torch.log(torch.tensor(init_sh[1]) + 1e-9).clone(),
                 "thT": torch.zeros(1), "thL": torch.zeros(1)}
            if isinstance(family, str):
                P["lc"] = torch.tensor(st, dtype=torch.float64)
            for v in P.values():
                v.requires_grad_(True)
            V = torch.tensor(OCT if family == "oct" else TET) if isinstance(family, str) else None

            def nll():
                c = torch.softmax(P["lc"], 0) @ V if V is not None else torch.tensor(family)
                kT = 1 + (self.sig_T if self.sig_T is not None else 1.0) * P["thT"][0]
                kL = 1 + self.sig_L * P["thL"][0]
                lam = self.lam(c, torch.exp(P["lmu"][0]), kT, kL, P["ag"], P["af"],
                               torch.softmax(P["tg"], 1), torch.softmax(P["tf"], 1))
                v = (lam - n - torch.where(n > 0, n * torch.log(lam / torch.clamp(n, min=1e-300)), torch.zeros_like(n))).sum()
                if self.aux is not None:
                    for m, sh, tau, B in ((self.aux[0], torch.softmax(P["tg"], 1), self.tau_g, self.Bg),
                                          (self.aux[1], torch.softmax(P["tf"], 1), self.tau_f, self.Bf)):
                        if tau > 0:
                            mu_ = tau * B.sum(1)[:, None] * sh
                            v = v + (mu_ - m - torch.where(m > 0, m * torch.log(mu_ / torch.clamp(m, min=1e-300)), torch.zeros_like(m))).sum()
                v = v + 0.5 * (P["ag"] / 0.05).pow(2).sum() + 0.5 * (P["af"] / 0.10).pow(2).sum()
                v = v + 0.5 * P["thL"].pow(2).sum() + (0.5 * P["thT"].pow(2).sum() if self.sig_T is not None else 0.0)
                return v, c
            opt = torch.optim.LBFGS(list(P.values()), lr=1, max_iter=400, history_size=50,
                                    tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn="strong_wolfe")
            prev = np.inf
            for _ in range(30):
                def closure():
                    opt.zero_grad()
                    v, _ = nll()
                    v.backward()
                    return v
                opt.step(closure)
                v, c = nll()
                v = float(v)
                if not np.isfinite(v):
                    break
                if abs(prev - v) < 1e-7:
                    break
                prev = v
            if np.isfinite(v) and v < best[0]:
                best = (v, c.detach().numpy().copy(), float(1 + (1.0 if self.sig_T is None else self.sig_T) * P["thT"].item()),
                        float(1 + self.sig_L * P["thL"].item()))
        if best[1] is None and starts and starts[0] is not None and len(starts) < 6:
            k = len(starts[0])
            return self.fit(n, family, list(starts) + [np.zeros(k), np.log(np.random.default_rng(1).dirichlet(np.ones(k)))],
                            init_sh)
        return best


OCT_STARTS = [np.log(np.array(w) + 1e-3) for w in ([1, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1])]
TET_STARTS = [np.log(np.array(w) + 1e-3) for w in ([1, 0, 0, 0], [1, 1, 1, 1])]


def test(model, n, init_sh, null="oct", tet_starts=TET_STARTS[:1]):
    alt = model.fit(n, "tet", tet_starts, init_sh)
    nul = model.fit(n, "oct", OCT_STARTS, init_sh) if null == "oct" else model.fit(n, TRUE_C / 3, [None], init_sh)
    q = 2 * (nul[0] - alt[0])
    return {"Z": float(np.sqrt(max(q, 0))), "q": float(q), "c_null": nul[1].tolist(), "c_alt": alt[1].tolist(),
            "kT_null": nul[2], "kL_null": nul[3]}


# ----------------------------------------------------------------------------- main
# ----------------------------------------------------------------------------- driver
PREP = {}


def prepare(args):
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    hn = np.load(args.hpred_noipsv, allow_pickle=True)
    h = d["h_exact"]
    w = d["w"]
    p, T = targets(h)
    W, _, _ = weights(h, False)
    ptt = d["kin_pt_tautau"]
    fold = np.arange(len(h)) % 2
    feats = {"tauspin": np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm(d)], 1),
             "tauspin_noIPSV": np.concatenate([hn["h_pred"].reshape(-1, 6), hn["hh_pred"].reshape(-1, 9), rm(d)], 1),
             "textbook": textbook(d), "textbook_pair": textbook_pair(d), "phicp": phicp(d)}
    ho_feats = {}
    if args.holdout:
        ho = np.load(args.holdout, allow_pickle=True)
        hh = np.load(args.holdout_hpred, allow_pickle=True)
        isZ = ho["proc"] == "trainZ"
        ho_feats = {"tauspin": np.concatenate([hh["h_pred"].reshape(-1, 6), hh["hh_pred"].reshape(-1, 9), rm(ho)], 1)[isZ],
                    "textbook": textbook(ho)[isZ]}
    out = {}
    for lev in args.levels.split(","):
        if lev == "exact":
            G, Gz = T.copy(), None
        else:
            G, Gz = regress(feats[lev], T, fold, ho_feats.get(lev), w)
        out[lev] = {"G": G, "Gz": Gz}
        print("regressed", lev, flush=True)
    return dict(levels=out, p=p, T=T, wz=W["Z"], ptt=ptt, w=w, acc=json.load(open(args.acc)))


def load_prep(path):
    z = np.load(path, allow_pickle=True)
    levs = [k[2:] for k in z.files if k.startswith("G_")]
    PREP["P"] = dict(levels={l: {"G": z["G_" + l], "Gz": z["Gz_" + l] if ("Gz_" + l) in z.files else None} for l in levs},
                     p=z["p"], T=z["T"], wz=z["wz"], ptt=z["ptt"], w=z["w"], acc=json.loads(str(z["acc"])))
    torch.set_num_threads(3)


def templates_for(lev, nq):
    P = PREP["P"]
    G, Gz = P["levels"][lev]["G"], P["levels"][lev]["Gz"]
    bn = Binner(G, nq)
    A0, D, Zsh = category_templates(bn(G), bn.nb, P["p"], P["wz"], P["ptt"], P["w"])
    Ush = A0 / A0.sum(1, keepdims=True)
    Xr = None
    if Gz is not None:
        zr = np.bincount(bn(Gz), minlength=bn.nb).astype(float)
        Xr = np.tile(zr / zr.sum(), (len(CATS), 1))
    return A0, D, Zsh, Ush, Xr


def job(spec):
    lev, name, kw = spec
    torch.set_num_threads(3)
    P = PREP["P"]
    A0, D, Zsh, Ush, Xr = templates_for(lev, kw.get("nq", NQ))
    S, Bg, Bf = yields(P["acc"], kw.get("scale", 1.0))
    cats = kw.get("cats")
    M = Model(A0, D, S, Bg, Bf, kw.get("sT", 0.1), kw.get("sL", 0.1), cats, kw.get("tau", 0.0), kw.get("tau", 0.0))
    sel = slice(None) if cats is None else np.asarray(cats)
    if kw.get("tau", 0.0) > 0:
        M.set_aux(Zsh[sel], Ush[sel])
    ti = np.linspace(0, 1, len(P["acc"]["S"]))
    drift = {"spin": (Ush, ti), "kin": (Xr, ti)}.get(kw.get("drift"))
    if kw.get("drift") == "kin" and Xr is None:
        return spec, None
    dr = None if drift is None else (drift[0][sel], drift[1])
    c_true = np.array(kw.get("c_true", TRUE_C))
    n = M.asimov(c_true, Zsh[sel], Ush[sel], dr)
    t = time.time()
    r = test(M, n, (Zsh[sel], Ush[sel]), kw.get("null", "oct"),
             TET_STARTS if (drift is not None or "c_true" in kw) else TET_STARTS[:1])
    r["sec"] = time.time() - t
    print(lev, name, round(r["Z"], 3), np.round(r["c_null"], 3), f"{r['sec']:.0f}s", flush=True)
    return spec, r


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/ent/dataset_HU.npz")
    ap.add_argument("--hpred", default="outputs/ent/hpred_HU.npz")
    ap.add_argument("--hpred-noipsv", default="outputs/ent/hpred_noipsv.npz")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--holdout-hpred", default="")
    ap.add_argument("--acc", default="outputs/ent/mass_acc.json")
    ap.add_argument("--out", default="outputs/ent/ent_robust.json")
    ap.add_argument("--levels", default="exact,tauspin,tauspin_noIPSV,textbook,textbook_pair,phicp")
    ap.add_argument("--full", default="tauspin,phicp,textbook", help="levels that get all variations")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--aux-scan", action="store_true", help="only the background-shape-knowledge scan")
    ap.add_argument("--cache", default="outputs/ent/ent_robust_G.npz")
    args = ap.parse_args()
    import os
    if not os.path.exists(args.cache):
        P = prepare(args)
        arrs = {f"G_{l}": v["G"] for l, v in P["levels"].items()}
        arrs.update({f"Gz_{l}": v["Gz"] for l, v in P["levels"].items() if v["Gz"] is not None})
        np.savez_compressed(args.cache, p=P["p"], T=P["T"], wz=P["wz"], ptt=P["ptt"], w=P["w"],
                            acc=json.dumps(P["acc"]), **arrs)
        print("cached", args.cache, flush=True)
    load_prep(args.cache)
    ctx = mp.get_context("spawn")
    sb = np.array([v[0] / (v[1] + v[2] + v[3] + v[4]) for v in CATS.values()])
    if args.aux_scan:
        specs = [(lev, f"tau{t:g}", {"tau": t}) for lev in args.levels.split(",") for t in (0, 0.3, 1, 3, 10, 100)]
        specs += [(lev, f"tau{t:g}_lumi6000", {"tau": t, "scale": 2.0}) for lev in args.full.split(",") for t in (1, 10)]
        with ctx.Pool(args.workers, initializer=load_prep, initargs=(args.cache,)) as pool:
            res = pool.map(job, specs, chunksize=1)
        out = {"config": vars(args), "levels": {}}
        for (lev, name, kw), r in res:
            out["levels"].setdefault(lev, {})[name] = r
        json.dump(out, open(args.out, "w"), indent=1)
        return
    stage1 = [(lev, "nominal", {}) for lev in args.levels.split(",")]
    with ctx.Pool(args.workers, initializer=load_prep, initargs=(args.cache,)) as pool:
        res1 = dict(((s[0], s[1]), r) for s, r in pool.map(job, stage1, chunksize=1))
    specs = []
    for lev in args.full.split(","):
        cn = res1[(lev, "nominal")]["c_null"]
        specs += [(lev, "resp0", {"sT": 0.0, "sL": 0.0}), (lev, "resp0.2", {"sT": 0.2, "sL": 0.2}),
                  (lev, "kT_free", {"sT": None}), (lev, "werner_null", {"null": "werner"}),
                  (lev, "bins4", {"nq": 4}), (lev, "bins8", {"nq": 8}),
                  (lev, "high_SB_only", {"cats": list(np.where(sb > 0.05)[0])}),
                  (lev, "drift_spin", {"drift": "spin"}), (lev, "drift_spin_null_injected", {"drift": "spin", "c_true": cn}),
                  (lev, "null_injected", {"c_true": cn}),
                  (lev, "drift_kin_realZ", {"drift": "kin"}), (lev, "drift_kin_realZ_null_injected", {"drift": "kin", "c_true": cn})]
        specs += [(lev, f"lumi{int(round(3000 * s))}", {"scale": s}) for s in (0.15, 1 / 3, 2 / 3, 2.0)]
        specs += [(lev, f"cat_{c}", {"cats": [k]}) for k, c in enumerate(CATS)]
    with ctx.Pool(args.workers, initializer=load_prep, initargs=(args.cache,)) as pool:
        res2 = pool.map(job, specs, chunksize=1)
    out = {"config": vars(args), "high_SB_cats": [c for c, x in zip(CATS, sb) if x > 0.05], "levels": {}}
    P = PREP["P"]
    for lev in args.levels.split(","):
        G, T = P["levels"][lev]["G"], P["T"]
        out["levels"][lev] = {"corr_gT": float(np.corrcoef(G[:, 0], T[:, 0])[0, 1]),
                              "corr_gL": float(np.corrcoef(G[:, 1], T[:, 1])[0, 1]),
                              "nominal": res1[(lev, "nominal")]}
    for (lev, name, kw), r in res2:
        if r is not None:
            out["levels"][lev][name] = r
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
