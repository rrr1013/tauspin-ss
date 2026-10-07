"""Go/no-go probe: m_tautau resolution of MMC vs IP-MMC on the v1 tau_had tau_had samples.

Inputs: v1 reco files (reco.py output: smeared tracks, pi0, IP of each track with
respect to the true PV, smeared SV) and the v1 dataset (MMC dR table, MET sigma,
the standard MMC mass).  The primary vertex is smeared here (sigma_T, sigma_Z),
which shifts both the impact vectors and the SV flight vector.
"""
import argparse
import glob
import json

import numpy as np
import torch

from ipmmc import PT_EDGES, run

SIG_D0 = lambda pt: 1e-3 * np.hypot(10.0, 150.0 / np.maximum(pt, 0.5))   # mm, as in reco.py
SIG_Z0 = lambda pt: 1e-3 * np.hypot(20.0, 300.0 / np.maximum(pt, 0.5))
SIG_SV_PERP = 0.010


def load(pattern, nmax):
    files = sorted(glob.glob(pattern))
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    keys = ["vis", "ch", "ip", "sv", "nch", "met", "tau_p4", "is_true", "q", "weight"]
    d = {k: np.concatenate([p[k] for p in parts])[:nmax] for k in keys}
    return d


def build_sides(d, pv_sig, rng):
    n = len(d["met"])
    dpv = rng.standard_normal((n, 3)) * np.array([pv_sig[0], pv_sig[0], pv_sig[1]])
    sides, diag = [], {}
    for s in (0, 1):
        trk_p = d["ch"][:, s, 0, :3]
        trk = trk_p / np.linalg.norm(trk_p, axis=-1, keepdims=True)
        pt = np.hypot(trk_p[:, 0], trk_p[:, 1])
        b = d["ip"][:, s, 0] - dpv
        b = b - (b * trk).sum(-1, keepdims=True) * trk
        e1 = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        e2 = np.cross(trk, e1)
        var = np.stack([SIG_D0(pt) ** 2 / 2 + pv_sig[0] ** 2, SIG_D0(pt) ** 2 / 2 + pv_sig[0] ** 2,
                        SIG_Z0(pt) ** 2 + pv_sig[1] ** 2], -1)
        sig_bperp = np.sqrt((e2 ** 2 * var).sum(-1))
        svv = d["sv"][:, s] - dpv
        svl = np.linalg.norm(svv, axis=-1)
        sv_dir = svv / np.maximum(svl, 1e-12)[:, None]
        sig_sv = np.hypot(SIG_SV_PERP, pv_sig[0]) / np.maximum(svl, 1e-3)
        vis = d["vis"][:, s]
        ptv = np.hypot(vis[:, 0], vis[:, 1])
        ptbin = np.clip(np.searchsorted(PT_EDGES, ptv, side="right") - 1, 0, len(PT_EDGES) - 2)
        three = d["nch"][:, s] >= 3
        sides.append(dict(vis=vis, trk=trk, b=b, sig_bperp=sig_bperp, three=three, sv_dir=sv_dir,
                          sv_len=svl, sig_sv=sig_sv, ptbin=ptbin, prong3=three.astype(int)))
        # IP-azimuth quality, as measured in tauspin on ATLAS full simulation
        t = d["tau_p4"][:, s, :3]
        t = t / np.linalg.norm(t, axis=-1, keepdims=True)
        tperp = t - (t * trk).sum(-1, keepdims=True) * trk
        cosd = (tperp * e1).sum(-1) / np.maximum(np.linalg.norm(tperp, axis=-1), 1e-12)
        dpsi = np.arccos(np.clip(cosd, -1, 1))
        alpha = np.arccos(np.clip((t * trk).sum(-1), -1, 1))
        one = ~three
        diag[f"side{s}_dpsi_median"] = float(np.median(dpsi[one]))
        diag[f"side{s}_transverse_err_mrad_median"] = float(np.median(alpha[one] * np.sin(dpsi[one])) * 1e3)
    return sides, diag


def to_torch(sides, dev):
    out = []
    for s in sides:
        o = {}
        for k, v in s.items():
            t = torch.as_tensor(v, device=dev)
            o[k] = t.to(torch.float64) if t.dtype in (torch.float32, torch.float64) else t
        out.append(o)
    return out


def summary(m, mtrue):
    ok = np.isfinite(m)
    r = m[ok] / mtrue[ok]
    q = np.percentile(r, [16, 50, 84])
    return {"valid": float(ok.mean()), "r16": q[0], "r50": q[1], "r84": q[2], "half68": (q[2] - q[0]) / 2,
            "within10": float((np.abs(r - 1) < 0.1).mean()), "tail_gt30": float((np.abs(r - 1) > 0.3).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="/home/rbaba/dihiggs-spin-20261006/runs/v1")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--S", type=int, default=2000)
    ap.add_argument("--pv", default="0.010,0.030")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(f"{args.v1}/dataset.npz", allow_pickle=True)
    table, met_sigma = ds["dr_table"], 1.4 * float(ds["met_sigma"])
    pv_sig = [float(x) for x in args.pv.split(",")]
    rng = np.random.default_rng(1)
    res = {"pv_sigma_mm": pv_sig, "met_sigma": met_sigma, "S": args.S}
    store = {}
    for proc in ("hh", "zbb", "ttll"):
        d = load(f"{args.v1}/reco/{proc}_run_*.npz", args.n)
        mtrue = np.sqrt(np.maximum(d["tau_p4"].sum(1)[:, 3] ** 2 - (d["tau_p4"].sum(1)[:, :3] ** 2).sum(-1), 0))
        sides, diag = build_sides(d, pv_sig, rng)
        st = to_torch(sides, dev)
        met = torch.as_tensor(d["met"], device=dev, dtype=torch.float64)
        r = {"diag": diag, "n": int(len(mtrue))}
        for tag, use_ip in (("mmc_dir", False), ("ipmmc", True)):
            o = run(st, met, met_sigma, table, use_ip=use_ip, S=args.S, dev=dev)
            r[tag] = summary(o["m_maxw"], mtrue)
            store[f"{proc}_{tag}"] = o["m_maxw"]
        store[f"{proc}_mtrue"] = mtrue
        res[proc] = r
        print(proc, json.dumps(r, default=lambda x: round(x, 4)), flush=True)
    np.savez_compressed(args.out.replace(".json", ".npz"), **store)
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
