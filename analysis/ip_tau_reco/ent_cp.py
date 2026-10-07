"""HL-LHC projection of the CP-mixing angle of the tau Yukawa coupling in H -> tau_had tau_had.

The Higgs-tau coupling cos(phi) tau-bar tau + sin(phi) tau-bar i gamma5 tau gives, in the
canonical (n, r, k) basis, the ditau spin density
    rho_phi = 1 - h-_k h+_k + cos(2 phi) (h-_n h+_n + h-_r h+_r) + sin(2 phi) (h-_n h+_r - h-_r h+_n)
(maximally entangled for every phi).  A measurement of phi is a phase, insensitive to the
overall transverse analysing power that limits the entanglement test (ent_robust.py:
k_T free).  Same machinery as ent_robust.py: spin-flat H(125)+jet, cross-fitted
statistics (g_T, g_A) = E_flat[(h-_n h+_n + h-_r h+_r, h-_n h+_r - h-_r h+_n) | x],
6 x 6 bins, ATLAS-like categories with nine m_tautau bins, genuine-tau and fake
background shapes free per category (sidebands) or known, global k_T, k_L.
Asimov at phi = 0; the profile-likelihood scan gives the expected 68 % / 95 % interval
and the expected exclusion of the pure CP-odd hypothesis (phi = 90 deg).
A 139 fb^-1 point is run for comparison with the Run-2 measurements.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ent_final import PT_RANGE
from ent_hllhc import CATS
from ent_robust import Binner, phicp, regress, rm, textbook, textbook_pair, yields
from entanglement_feasibility import weights

torch.set_default_dtype(torch.float64)
PREP = {}


def targets(h):
    ok = np.isfinite(h).all((1, 2))
    hz = np.where(ok[:, None, None], h, 0.0)
    m, p = hz[:, 0], hz[:, 1]
    tT = m[:, 0] * p[:, 0] + m[:, 1] * p[:, 1]
    tA = m[:, 0] * p[:, 1] - m[:, 1] * p[:, 0]
    tL = m[:, 2] * p[:, 2]
    return np.stack([tT, tA, tL], 1)


def prepare(args):
    d = np.load(args.data, allow_pickle=True)
    hp = np.load(args.hpred, allow_pickle=True)
    hn = np.load(args.hpred_noipsv, allow_pickle=True)
    h, w = d["h_exact"], d["w"]
    T = targets(h)
    W, _, _ = weights(h, False)
    fold = np.arange(len(h)) % 2
    feats = {"tauspin": np.concatenate([hp["h_pred"].reshape(-1, 6), hp["hh_pred"].reshape(-1, 9), rm(d)], 1),
             "tauspin_noIPSV": np.concatenate([hn["h_pred"].reshape(-1, 6), hn["hh_pred"].reshape(-1, 9), rm(d)], 1),
             "textbook": textbook(d), "textbook_pair": textbook_pair(d), "phicp": phicp(d)}
    arrs = {}
    for lev in args.levels.split(","):
        arrs[f"G_{lev}"] = T[:, :2].copy() if lev == "exact" else regress(feats[lev], T[:, :2], fold, None, w)[0]
        print("regressed", lev, flush=True)
    np.savez_compressed(args.cache, T=T, wz=W["Z"], ptt=d["kin_pt_tautau"], w=w, **arrs)


def load(cache, acc):
    z = np.load(cache)
    PREP.update({k: z[k] for k in z.files})
    PREP["acc"] = json.load(open(acc))
    torch.set_num_threads(3)


def templates(lev, nq=6):
    G, T, w, ptt, wz = PREP[f"G_{lev}"], PREP["T"], PREP["w"], PREP["ptt"], PREP["wz"]
    bn = Binner(G, nq)
    b = bn(G)
    A0, D, Z = [], [], []
    for c in CATS:
        lo, hi = PT_RANGE[c]
        m = (ptt >= lo) & (ptt < hi)
        n = w[m].sum()
        A0.append(np.bincount(b[m], weights=w[m], minlength=bn.nb) / n)
        D.append(np.stack([np.bincount(b[m], weights=w[m] * T[m, i], minlength=bn.nb) / n for i in range(3)]))
        z = np.bincount(b[m], weights=w[m] * wz[m], minlength=bn.nb)
        Z.append(z / z.sum())
    A0 = np.array(A0)
    return A0, np.array(D), np.array(Z), A0 / A0.sum(1, keepdims=True)


class CPModel:
    def __init__(self, A0, D, S, Bg, Bf, sT, sL, tau):
        t = torch.tensor
        self.A0, self.D, self.S, self.Bg, self.Bf = t(A0), t(D), t(S), t(Bg), t(Bf)
        self.sT, self.sL, self.tau = sT, sL, tau
        self.C, self.I = self.S.shape

    def shape(self, phi, kT, kL):
        s = self.A0 + kT * (torch.cos(2 * phi) * self.D[:, 0] + torch.sin(2 * phi) * self.D[:, 1]) - kL * self.D[:, 2]
        s = torch.clamp(s, min=1e-12)
        return s / s.sum(1, keepdim=True)

    def lam(self, phi, mu, kT, kL, ag, af, shg, shf):
        return (mu * self.S[:, :, None] * self.shape(phi, kT, kL)[:, None, :]
                + (self.Bg * torch.exp(ag))[:, :, None] * shg[:, None, :]
                + (self.Bf * torch.exp(af))[:, :, None] * shf[:, None, :])

    def nll_fit(self, n, phi, shg0, shf0, aux=None):
        P = {"lmu": torch.zeros(1), "ag": torch.zeros(self.C, self.I), "af": torch.zeros(self.C, self.I),
             "tg": torch.log(torch.tensor(shg0) + 1e-9), "tf": torch.log(torch.tensor(shf0) + 1e-9),
             "thT": torch.zeros(1), "thL": torch.zeros(1)}
        for v in P.values():
            v.requires_grad_(True)
        ph = torch.tensor(float(phi))

        def f():
            kT = 1 + (1.0 if self.sT is None else self.sT) * P["thT"][0]
            kL = 1 + self.sL * P["thL"][0]
            shg, shf = torch.softmax(P["tg"], 1), torch.softmax(P["tf"], 1)
            lam = self.lam(ph, torch.exp(P["lmu"][0]), kT, kL, P["ag"], P["af"], shg, shf)
            v = (lam - n - torch.where(n > 0, n * torch.log(lam / torch.clamp(n, min=1e-300)), torch.zeros_like(n))).sum()
            if aux is not None:
                for m, sh, B in ((aux[0], shg, self.Bg), (aux[1], shf, self.Bf)):
                    mu_ = self.tau * B.sum(1)[:, None] * sh
                    v = v + (mu_ - m - torch.where(m > 0, m * torch.log(mu_ / torch.clamp(m, min=1e-300)), torch.zeros_like(m))).sum()
            v = v + 0.5 * (P["ag"] / 0.05).pow(2).sum() + 0.5 * (P["af"] / 0.10).pow(2).sum() + 0.5 * P["thL"].pow(2).sum()
            if self.sT is not None:
                v = v + 0.5 * P["thT"].pow(2).sum()
            return v
        opt = torch.optim.LBFGS(list(P.values()), lr=1, max_iter=400, history_size=50, tolerance_grad=1e-10,
                                tolerance_change=1e-13, line_search_fn="strong_wolfe")
        prev = np.inf
        for _ in range(30):
            def closure():
                opt.zero_grad()
                v = f()
                v.backward()
                return v
            opt.step(closure)
            v = float(f())
            if abs(prev - v) < 1e-7:
                break
            prev = v
        return v


def job(spec):
    lev, name, kw = spec
    A0, D, Zsh, Ush = templates(lev, kw.get("nq", 6))
    S, Bg, Bf = yields(PREP["acc"], kw.get("scale", 1.0))
    M = CPModel(A0, D, S, Bg, Bf, kw.get("sT", 0.1), kw.get("sL", 0.1), kw.get("tau", 0.0))
    with torch.no_grad():
        one = torch.tensor(1.0)
        n = M.lam(torch.tensor(0.0), one, one, one, torch.zeros(M.C, M.I), torch.zeros(M.C, M.I),
                  torch.tensor(Zsh), torch.tensor(Ush))
    aux = None
    if M.tau > 0:
        aux = (M.tau * M.Bg.sum(1)[:, None] * torch.tensor(Zsh), M.tau * M.Bf.sum(1)[:, None] * torch.tensor(Ush))
    t = time.time()
    l0 = M.nll_fit(n, 0.0, Zsh, Ush, aux)
    grid = kw.get("grid", [5, 10, 15, 20, 30, 45, 90])
    q = {g: 2 * (M.nll_fit(n, np.radians(g), Zsh, Ush, aux) - l0) for g in grid}
    gs = np.array([0] + grid, float)
    qs = np.array([0.0] + [q[g] for g in grid])
    sig1 = float(np.interp(1.0, qs, gs)) if qs.max() > 1 else None
    sig2 = float(np.interp(4.0, qs, gs)) if qs.max() > 4 else None
    r = {"q": {str(g): float(v) for g, v in q.items()}, "phi68_deg": sig1, "phi95_deg": sig2,
         "Z_CPodd": float(np.sqrt(max(q[90], 0))) if 90 in q else None, "sec": time.time() - t}
    print(lev, name, "68%", None if sig1 is None else round(sig1, 2), "95%", None if sig2 is None else round(sig2, 2),
          "CP-odd", None if r["Z_CPodd"] is None else round(r["Z_CPodd"], 2), f"{r['sec']:.0f}s", flush=True)
    return spec, r


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/ent/dataset_HU.npz")
    ap.add_argument("--hpred", default="outputs/ent/hpred_HU.npz")
    ap.add_argument("--hpred-noipsv", default="outputs/ent/hpred_noipsv.npz")
    ap.add_argument("--acc", default="outputs/ent/mass_acc.json")
    ap.add_argument("--cache", default="outputs/ent/ent_cp_G.npz")
    ap.add_argument("--out", default="outputs/ent/ent_cp.json")
    ap.add_argument("--levels", default="exact,tauspin,tauspin_noIPSV,textbook,textbook_pair,phicp")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not os.path.exists(args.cache):
        prepare(args)
    specs = []
    for lev in args.levels.split(","):
        specs += [(lev, "nominal", {}), (lev, "known_bkg_shape", {"tau": 100.0}),
                  (lev, "run2_139fb", {"scale": 139 / 3000 / 1.15, "grid": [10, 20, 30, 45, 60, 90]}),
                  (lev, "lumi6000", {"scale": 2.0})]
        if lev in ("tauspin", "phicp"):
            specs += [(lev, "kT_free", {"sT": None}), (lev, "bins4", {"nq": 4}), (lev, "bins8", {"nq": 8}),
                      (lev, "lumi1000", {"scale": 1 / 3})]
    with mp.get_context("spawn").Pool(args.workers, initializer=load, initargs=(args.cache, args.acc)) as pool:
        res = pool.map(job, specs, chunksize=1)
    out = {"config": vars(args), "levels": {}}
    for (lev, name, kw), r in res:
        out["levels"].setdefault(lev, {})[name] = r
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
