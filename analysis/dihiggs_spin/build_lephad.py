"""Dataset for the tau_lep tau_had extension: kinematics (ATLAS-like BDT inputs), lepton
impact-parameter features (tauspin-style local frame) and hadronic-side tauspin inputs.

Normalisation: cross section x 3 ab^-1 / generated events x event weight.
  tt (NNLO+NNLL, 14 TeV) 984.5 pb; tW (both charges, approx. NNLO) 84 pb; W BRs e 0.1071,
  mu 0.1063, tau 0.1125, hadrons 0.6741; Z+bb from the generator x 1.3; HH, ZH, ttH as
  build_dataset.py.
Lepton IP features (the extension of tauspin's PV-referenced impact parameter):
  |d0|/sigma, d0 (signed, um), z0 sin(theta)/sigma, the impact vector in the local basis
  (n, r, k) of the lepton (as tauspin's tau tracks), its component along the
  MMC-estimated tau flight direction transverse to the lepton ("lifetime sign": a lepton
  from a tau decay is displaced towards the side of its tau, a prompt lepton has no
  preferred sign), lepton pT and flavour.
Hadronic side (tauspin inputs of one tau): Upsilon, x (MMC), reco mode, leading-track IP
  and SV in the local basis, MET projections.
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from build_dataset import local_basis
from mmc_lep import build_dr_table, run_mmc_lh
from reco import delta_r, dphi, eta, mass, phi, pt
from reco_lephad import lepton_resolution, signed_d0

LUMI_FB = 3000.0
BR_BBTT = 2 * 0.5824 * 0.06272
TT, TW = 984.5e3, 84.0e3
BW = {"e": 0.1071, "mu": 0.1063, "tau": 0.1125, "had": 0.6741}
BL = BW["e"] + BW["mu"]
XS_FB = {
    "hhC": 36.69 * BR_BBTT / 0.5,
    "ttll": TT * BW["tau"] ** 2,
    "ttlj": TT * 2 * BW["tau"] * BW["had"],
    "ttlt": TT * 2 * BL * BW["tau"],
    "ttljp": TT * 2 * BL * BW["had"],
    "tWll": TW * (BL + BW["tau"]) ** 2,
    "tWlj": TW * 2 * (BL + BW["tau"]) * BW["had"],
    "zh": 0.8839e3 * 0.1512 * 0.06272,
    "tth": 0.6113e3 * 0.06272,
}
K_ZBB = 1.3
PROCS = ("hhC", "zbb", "ttll", "ttlj", "ttlt", "ttljp", "tWll", "tWlj", "zh", "tth")


def load(proc, files):
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    n_gen = sum(int(p["n_gen"]) for p in parts)
    sig = np.mean([float(p["sigma_pb"]) for p in parts])
    xs = XS_FB.get(proc, sig * 1e3 * K_ZBB)
    keys = [k for k in parts[0] if parts[0][k].ndim >= 1 and k != "n_gen"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["uid"] = np.concatenate([i * 1_000_000 + p["event"] for i, p in enumerate(parts)])
    d["w"] = xs * LUMI_FB / n_gen * d["weight"]
    d["proc"] = np.full(len(d["w"]), proc)
    return d


def lepton_ip_features(d, mm, ipscale):
    lep = d["lep"]
    # lepton impact parameter (um) in the local basis of the lepton; re-smeared from the
    # true impact vector with the resolution scaled by `ipscale` (1 = reco_lephad.py)
    rng = np.random.default_rng(12345)
    s_d0, s_z0 = lepton_resolution(lep, d["lep_id"])
    s_d0, s_z0 = s_d0 * ipscale, s_z0 * ipscale
    tail = np.where(rng.random(len(lep)) < 0.02, 4.0, 1.0)
    ip = d["lep_ip_true"].copy()
    ip[:, :2] += (s_d0 * tail)[:, None] * rng.standard_normal((len(lep), 2)) / np.sqrt(2)
    ip[:, 2] += s_z0 * tail * rng.standard_normal(len(lep))
    d0 = signed_d0(ip, lep)
    sth = pt(lep) / np.maximum(np.linalg.norm(lep[:, :3], axis=-1), 1e-9)
    z0s = ip[:, 2] * sth
    n_, r_, k_ = local_basis(lep)
    ipt = ip - (ip * k_).sum(-1, keepdims=True) * k_
    nu_l = mm["nu4"][:, 0]
    taul = lep + nu_l
    tdir = taul[:, :3] / np.maximum(np.linalg.norm(taul[:, :3], axis=-1, keepdims=True), 1e-9)
    tperp = tdir - (tdir * k_).sum(-1, keepdims=True) * k_
    tperp /= np.maximum(np.linalg.norm(tperp, axis=-1, keepdims=True), 1e-12)
    ipf = {"lip_d0sig": np.abs(d0) / s_d0, "lip_absd0": np.abs(d0),
           "lip_z0sig": z0s / s_z0,
           "lip_ttva": ((np.abs(d0) / s_d0 < np.where(np.abs(d["lep_id"]) == 11, 5.0, 3.0)) & (np.abs(z0s) < 500)).astype(float),
           "lip_n": (ip * n_).sum(-1), "lip_r": (ip * r_).sum(-1), "lip_k": (ip * k_).sum(-1),
           "lip_life": (ipt * tperp).sum(-1) / s_d0, "lip_sd0": s_d0,
           "lip_is_e": (np.abs(d["lep_id"]) == 11).astype(float), "lip_x": lep[:, 3] / np.maximum(mm["e_tau"][:, 0], lep[:, 3])}
    return ipf


def features(d, table, met_sigma, ipscales=(1.0,)):
    lep, tau, met = d["lep"], d["tau_vis"], d["met"]
    vis = np.stack([lep, tau], 1)
    vis_m = vis.copy()
    vis_m[:, 1, 3] = np.sqrt((tau[:, :3] ** 2).sum(-1) + np.minimum(mass(tau), 1.6) ** 2)
    kind = np.stack([np.zeros(len(lep), int), np.where(d["tau_nch"] >= 2, 3, 1)], 1)
    mm = run_mmc_lh(vis_m, met, kind, table, 1.4 * met_sigma)
    retry = np.flatnonzero(~mm["valid"])
    if len(retry):
        r2 = run_mmc_lh(vis_m[retry], met[retry], kind[retry], table, 4.0 * met_sigma, n_samples=4000, seed=1)
        for k in mm:
            mm[k][retry] = r2[k]
    fail = ~mm["valid"]
    met4 = np.concatenate([met, np.zeros((len(met), 1)), np.hypot(met[:, 0], met[:, 1])[:, None]], 1)
    mm["m_maxw"][fail] = mass(vis.sum(1) + met4)[fail]
    mm["ditau4"][fail] = (vis.sum(1) + met4)[fail]
    mm["e_tau"][fail] = vis[fail, :, 3]
    b1, b2 = d["b1"], d["b2"]
    bb, tt = b1 + b2, mm["ditau4"]
    metv = np.hypot(met[:, 0], met[:, 1])
    metphi = np.arctan2(met[:, 1], met[:, 0])
    f = {"m_tautau": mm["m_maxw"], "mmc_valid": mm["valid"].astype(float), "m_bb": mass(bb),
         "m_hh": mass(bb + tt), "dr_lt": delta_r(lep, tau), "dr_bb": delta_r(b1, b2),
         "pt_lep": pt(lep), "pt_tau": pt(tau), "pt_b1": pt(b1), "pt_b2": pt(b2), "met": metv,
         "mt_lep": np.sqrt(np.maximum(2 * pt(lep) * metv * (1 - np.cos(dphi(phi(lep), metphi))), 0)),
         "mt_tau": np.sqrt(np.maximum(2 * pt(tau) * metv * (1 - np.cos(dphi(phi(tau), metphi))), 0)),
         "pt_bb": pt(bb), "pt_tautau": pt(tt), "dphi_bb_tt": np.abs(dphi(phi(bb), phi(lep + tau))),
         "m_vis": mass(lep + tau), "dpt_lt": pt(lep) - pt(tau), "ht": d["ht"], "njet": d["njet"].astype(float),
         "eta_lep": eta(lep), "eta_tau": eta(tau), "trigger": d["trigger"].astype(float)}
    f["m_hh_star"] = f["m_hh"] - f["m_bb"] - f["m_tautau"] + 250.0
    # hadronic-side tauspin inputs
    n2, r2, k2 = local_basis(tau)
    tip = d["tau_ip"][:, 0] * 1e3
    sv = d["tau_sv"]
    L = np.linalg.norm(sv, axis=-1)
    three = d["tau_nch"] >= 3
    e_ch = d["tau_ch"].sum(1)[:, 3]
    e_pi0 = d["tau_pi0"].sum(1)[:, 3]
    met3 = np.concatenate([met, np.zeros((len(met), 1))], 1)
    sp = {"upsilon": (e_ch - e_pi0) / np.maximum(e_ch + e_pi0, 1e-6), "x_tau": tau[:, 3] / np.maximum(mm["e_tau"][:, 1], tau[:, 3]),
          "tip_n": (tip * n2).sum(-1), "tip_r": (tip * r2).sum(-1), "tip_k": (tip * k2).sum(-1),
          "tsv_n": np.where(three, (sv * n2).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0), "tsv_r": np.where(three, (sv * r2).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0),
          "tsv_L": np.where(three, L, 0), "tmet_n": (met3 * n2).sum(-1), "tmet_r": (met3 * r2).sum(-1)}
    for m in range(5):
        sp[f"rmode{m}"] = (d["tau_rmode"] == m).astype(float)
    ipfs = {sc: lepton_ip_features(d, mm, sc) for sc in ipscales}
    return f, ipfs, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recodir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--met-sigma", type=float, default=0.0, help="0: measure from the HH sample")
    ap.add_argument("--ipscales", default="1.0,1.5,0.8", help="lepton IP resolution scales (first = nominal)")
    args = ap.parse_args()
    data = {}
    for p in PROCS:
        files = sorted(glob.glob(f"{args.recodir}/{p}_run_*.npz"))
        if files:
            data[p] = load(p, files)
            print(p, len(data[p]["w"]), "events, yield", round(float(data[p]["w"].sum()), 2), flush=True)
    # MMC dR table from true taus of the non-fake samples (lepton side: lepton + its tau's neutrinos)
    vis_l, nu_l, kind_l, dmet = [], [], [], []
    for p in ("hhC", "zbb", "ttll", "ttlt"):
        if p not in data:
            continue
        d = data[p]
        tr = d["uid"] % 3 == 0
        li = d["lep_tau_idx"]
        ok = tr & (li >= 0)
        vis_l.append(d["lep"][ok]); nu_l.append(np.take_along_axis(d["tau_nu_p4"][ok], li[ok][:, None, None], 1)[:, 0]); kind_l.append(np.zeros(ok.sum(), int))
        ti = d["tau_had_idx"]
        ok = tr & (ti >= 0)
        vis_l.append(d["tau_vis"][ok]); nu_l.append(np.take_along_axis(d["tau_nu_p4"][ok], ti[ok][:, None, None], 1)[:, 0])
        kind_l.append(np.where(d["tau_nch"][ok] >= 2, 3, 1))
        dmet.append(d["met"][tr] - d["met_true"][tr])
    table = build_dr_table(np.concatenate(vis_l), np.concatenate(nu_l), np.concatenate(kind_l))
    met_sigma = args.met_sigma or float(np.sqrt(0.5 * np.mean(np.concatenate(dmet) ** 2)))
    print("MET sigma per component", met_sigma, flush=True)
    out = {}
    for p, d in data.items():
        scales = [float(x) for x in args.ipscales.split(",")]
        f, ipfs, sp = features(d, table, met_sigma, scales)
        for pre, dd in (("kin_", f), ("had_", sp)):
            for k, v in dd.items():
                out.setdefault(pre + k, []).append(v)
        for sc, ipf in ipfs.items():
            tag = "lip_" if sc == scales[0] else f"lip{sc:g}_"
            for k, v in ipf.items():
                out.setdefault(tag + k.replace("lip_", ""), []).append(v)
        for k in ("w", "proc", "uid", "lep_from_tau", "lep_mother", "lep_id", "pass_ttva", "tau_is_true", "h_had", "trigger"):
            out.setdefault(k, []).append(d[k])
        print("features", p, flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["dr_table"] = table
    out["met_sigma"] = np.array(met_sigma)
    np.savez_compressed(args.out, **out)


if __name__ == "__main__":
    main()
