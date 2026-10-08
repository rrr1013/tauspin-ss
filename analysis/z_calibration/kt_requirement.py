"""How well must the transverse analysing-power response k_T be known for the
H -> tau_had tau_had entanglement test?  Reuses the robust HL-LHC model of
analysis/ip_tau_reco/ent_robust.py unchanged and only varies the k_T (and k_L) prior:

  Gaussian priors of width sigma_T in {0, .1, .3, .5, 1, 2}, k_T free, and a
  calibration-free physical bound: the true response cannot exceed what the decay
  physics allows.  For any binning b, Cauchy-Schwarz gives
      sum_b D_i,data(b)^2 / P(b) <= E_flat[(h-_i h+_i)^2],
  and with D_data = k D_MC this is k <= k_max = sqrt(E[(h_i h_i)^2] / sum_b D_MC(b)^2/P(b)),
  evaluated per category and component; the tightest category bound is used.
  E[(h_i h_i)^2] depends only on the decay-level h distribution and the acceptance.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ip_tau_reco'))
import ent_robust as er  # noqa: E402


class BoundModel(er.Model):
    """k = k_max * sigmoid(theta) (flat-ish inside the physical range) instead of a Gaussian."""

    def __init__(self, *a, kT_max=None, kL_max=None, **k):
        super().__init__(*a, **k)
        self.kT_max, self.kL_max = kT_max, kL_max

    def fit(self, n, family, starts, init_sh):
        best = (np.inf, None)
        for st in starts:
            P = {"lmu": torch.zeros(1), "ag": torch.zeros(self.C, self.I), "af": torch.zeros(self.C, self.I),
                 "tg": torch.log(torch.tensor(init_sh[0]) + 1e-9).clone(), "tf": torch.log(torch.tensor(init_sh[1]) + 1e-9).clone(),
                 "thT": torch.tensor([float(np.log(1 / (self.kT_max - 1)))]) if self.kT_max else torch.zeros(1),
                 "thL": torch.tensor([float(np.log(1 / (self.kL_max - 1)))]) if self.kL_max else torch.zeros(1)}
            if isinstance(family, str):
                P["lc"] = torch.tensor(st, dtype=torch.float64)
            for v in P.values():
                v.requires_grad_(True)
            V = torch.tensor(er.OCT if family == "oct" else er.TET) if isinstance(family, str) else None

            def ks():
                kT = self.kT_max * torch.sigmoid(P["thT"][0]) if self.kT_max else 1 + (1.0 if self.sig_T is None else self.sig_T) * P["thT"][0]
                kL = self.kL_max * torch.sigmoid(P["thL"][0]) if self.kL_max else 1 + self.sig_L * P["thL"][0]
                return kT, kL

            def nll():
                c = torch.softmax(P["lc"], 0) @ V if V is not None else torch.tensor(family)
                kT, kL = ks()
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
                if not self.kL_max:
                    v = v + 0.5 * P["thL"].pow(2).sum()
                if not self.kT_max and self.sig_T is not None:
                    v = v + 0.5 * P["thT"].pow(2).sum()
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
                if not np.isfinite(v) or abs(prev - v) < 1e-7:
                    break
                prev = v
            if np.isfinite(v) and v < best[0]:
                kT, kL = ks()
                best = (v, c.detach().numpy().copy(), float(kT), float(kL))
        return best


def physical_bounds(lev, nq=er.NQ):
    """k_max for transverse (n and r) and longitudinal components, tightest over categories."""
    P = er.PREP["P"]
    G = P["levels"][lev]["G"]
    bn = er.Binner(G, nq)
    b = bn(G)
    p, w, ptt = P["p"], P["w"], P["ptt"]
    kmax = {"T": [], "L": []}
    for c in er.CATS:
        lo, hi = er.PT_RANGE[c]
        m = (ptt >= lo) & (ptt < hi)
        ww = w[m]
        Pb = np.bincount(b[m], weights=ww, minlength=bn.nb) / ww.sum()
        per = []
        for i in range(3):
            D = np.bincount(b[m], weights=ww * p[m, i], minlength=bn.nb) / ww.sum()
            e2 = np.sum(ww * p[m, i] ** 2) / ww.sum()
            per.append(np.sqrt(e2 / np.sum(D ** 2 / np.maximum(Pb, 1e-12))))
        kmax["T"].append(min(per[0], per[1]))
        kmax["L"].append(per[2])
    return {k: float(np.min(v)) for k, v in kmax.items()}, {k: [float(x) for x in v] for k, v in kmax.items()}


def job(spec):
    lev, name, kw = spec
    torch.set_num_threads(2)
    P = er.PREP["P"]
    A0, D, Zsh, Ush, Xr = er.templates_for(lev, er.NQ)
    S, Bg, Bf = er.yields(P["acc"], 1.0)
    M = BoundModel(A0, D, S, Bg, Bf, kw.get("sT", 0.1), kw.get("sL", 0.1), None, kw.get("tau", 0.0), kw.get("tau", 0.0),
                   kT_max=kw.get("kT_max"), kL_max=kw.get("kL_max"))
    if kw.get("tau", 0.0) > 0:
        M.set_aux(Zsh, Ush)
    n = M.asimov(er.TRUE_C, Zsh, Ush)
    t = time.time()
    alt = M.fit(n, "tet", er.TET_STARTS[:1], (Zsh, Ush))
    nul = M.fit(n, "oct", er.OCT_STARTS, (Zsh, Ush))
    q = 2 * (nul[0] - alt[0])
    r = {"Z": float(np.sqrt(max(q, 0))), "q": float(q), "c_null": nul[1].tolist(), "kT_null": nul[2], "kL_null": nul[3],
         "sec": time.time() - t}
    print(lev, name, round(r["Z"], 3), np.round(r["c_null"], 3), round(r["kT_null"], 2), round(r["kL_null"], 2), flush=True)
    return spec, r


def main():
    import multiprocessing as mp
    import os
    ap = argparse.ArgumentParser()
    ent = Path.home() / 'Projects/tauspin-wt-dihiggs/analysis/ip_tau_reco/outputs/ent'
    ap.add_argument("--data", default=str(ent / "dataset_HU.npz"))
    ap.add_argument("--hpred", default=str(ent / "hpred_HU.npz"))
    ap.add_argument("--hpred-noipsv", default=str(ent / "hpred_noipsv.npz"))
    ap.add_argument("--acc", default=str(ent / "mass_acc.json"))
    ap.add_argument("--cache", default=str(ent / "ent_zcal_G.npz"))
    ap.add_argument("--levels", default="tauspin,phicp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "results" / "kt_requirement.json"))
    args = ap.parse_args()
    args.holdout = ""
    args.holdout_hpred = ""
    if not os.path.exists(args.cache):
        Pp = er.prepare(args)
        arrs = {f"G_{l}": v["G"] for l, v in Pp["levels"].items()}
        np.savez_compressed(args.cache, p=Pp["p"], T=Pp["T"], wz=Pp["wz"], ptt=Pp["ptt"], w=Pp["w"],
                            acc=json.dumps(Pp["acc"]), **arrs)
    er.load_prep(args.cache)
    out = {"bounds": {}, "fits": {}}
    specs = []
    for lev in args.levels.split(","):
        kb, per = physical_bounds(lev)
        out["bounds"][lev] = {"tightest": kb, "per_category": per}
        print(lev, "physical bounds", kb, flush=True)
        for tau in (0.0, 1.0):
            sfx = "" if tau == 0 else "_tau1"
            base = {"tau": tau}
            specs += [(lev, f"sT{s:g}{sfx}", dict(base, sT=s)) for s in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)]
            specs += [(lev, f"kT_free{sfx}", dict(base, sT=None)),
                      (lev, f"bound_T{sfx}", dict(base, kT_max=kb["T"])),
                      (lev, f"bound_TL{sfx}", dict(base, kT_max=kb["T"], kL_max=kb["L"]))]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=er.load_prep, initargs=(args.cache,)) as pool:
        res = pool.map(job, specs, chunksize=1)
    for (lev, name, kw), r in res:
        out["fits"].setdefault(lev, {})[name] = r
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
