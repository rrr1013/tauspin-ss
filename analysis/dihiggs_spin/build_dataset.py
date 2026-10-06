"""Merge the reconstructed samples, normalise them to 3 ab^-1 at 14 TeV, run the
MMC and build the classifier features.

Splits by event index: split = event % 3, 0 -> h-regressor training (and the
MMC dR table), 1 -> event-classifier training, 2 -> evaluation.

Cross sections (14 TeV):
  ggF HH (kappa_lambda = 1)  36.69 fb (NNLO FTapprox, LHC HH WG), BR(bb tautau) = 2 x 0.5824 x 0.06272
  ttbar                      984.5 pb (NNLO+NNLL); BR(W -> tau nu) = 0.1125, BR(W -> qq) = 0.6741
  ZH (Z -> bb, H -> tautau)  0.8839 pb x 0.1512 x 0.06272
  ttH (H -> tautau)          0.6113 pb x 0.06272
  Z(-> tautau) bb            MadGraph LO x 1.3
The absolute normalisation is later calibrated to the ATLAS Run-2 composition;
only shapes and relative selection efficiencies come from here.
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from mmc import build_dr_table, run_mmc
from polarimeter_pythia import polarimeter_nrk
from reco import delta_r, dphi, eta, mass, phi, pt

LUMI_FB = 3000.0
BR_BBTT = 2 * 0.5824 * 0.06272
XS_FB = {
    "hh": 36.69 * BR_BBTT / 0.5,           # sample has BR(bb)=BR(tautau)=0.5 -> bbtautau fraction 0.5
    "ttll": 984.5e3 * 0.1125 ** 2,
    "ttlj": 984.5e3 * 2 * 0.1125 * 0.6741,
    "zh": 0.8839e3 * 0.1512 * 0.06272,
    "tth": 0.6113e3 * 0.06272,
}
K_ZBB = 1.3
RMODE_TO_CLS = np.array([0, 1, 2, 3, -1])


def load(proc, files):
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    n_gen = sum(int(p["n_gen"]) for p in parts)
    sig = np.mean([float(p["sigma_pb"]) for p in parts])
    xs = XS_FB[proc] if proc in XS_FB else sig * 1e3 * K_ZBB
    keys = [k for k in parts[0] if parts[0][k].ndim >= 1 and k not in ("n_gen",)]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    # unique event id across chunks: chunk index * 1e6 + local index
    d["uid"] = np.concatenate([i * 1_000_000 + p["event"] for i, p in enumerate(parts)])
    # weight = cross section x L / generated events x (tau-ID or fake-rate factor, LHE weight)
    d["w"] = xs * LUMI_FB / n_gen * d["weight"]
    d["proc"] = np.full(len(d["w"]), proc)
    return d


def local_basis(v):
    k = v[..., :3] / np.linalg.norm(v[..., :3], axis=-1, keepdims=True)
    z = np.zeros_like(k)
    z[..., 2] = 1.0
    n = np.cross(k, z)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    r = np.cross(n, k)
    return n, r, k


def features(d, table, met_sigma):
    vis, met = d["vis"], d["met"]
    prong = np.where(d["nch"] >= 2, 3, 1)
    # the mass shell has no solution for m_vis > m_tau (11 % of smeared visible taus,
    # 7 % in ATLAS full simulation): cap the visible mass at 1.6 GeV for the MMC only
    vis_m = vis.copy()
    vis_m[..., 3] = np.sqrt((vis[..., :3] ** 2).sum(-1) + np.minimum(mass(vis), 1.6) ** 2)
    mm = run_mmc(vis_m, met, prong, table, 1.4 * met_sigma)
    retry = np.flatnonzero(~mm["valid"])
    if len(retry):
        r2 = run_mmc(vis_m[retry], met[retry], prong[retry], table, 4.0 * met_sigma, n_samples=4000, seed=1)
        for k in mm:
            mm[k][retry] = r2[k]
    mm["status"] = np.where(mm["valid"], 1.0, 0.0)
    mm["status"][retry] = np.where(mm["valid"][retry], 0.5, 0.0)
    # fallback: effective mass of the visible taus and the transverse MET
    fail = ~mm["valid"]
    met4 = np.concatenate([met, np.zeros((len(met), 1)), np.hypot(met[:, 0], met[:, 1])[:, None]], 1)
    mm["m_maxw"][fail] = mass(vis.sum(1) + met4)[fail]
    mm["ditau4"][fail] = (vis.sum(1) + met4)[fail]
    mm["e_tau"][fail] = vis[fail, :, 3]
    # side order for spin quantities: side 0 = tau- (reco charge)
    swap = d["q"][:, 0] > 0
    S = lambda a: np.where(swap.reshape((-1,) + (1,) * (a.ndim - 1)), a[:, ::-1], a)
    vis_s, ch_s, chq_s, pi0_s = S(vis), S(d["ch"]), S(d["ch_q"]), S(d["pi0"])
    ip_s, sv_s, rmode_s, nu_s = S(d["ip"]), S(d["sv"]), S(d["rmode"]), S(mm["nu4"])
    nch_s = S(d["nch"])
    # hybrid polarimeter: reco visible components + MMC neutrinos
    cls = RMODE_TO_CLS[rmode_s]
    tau4 = vis_s + nu_s
    h_hyb = polarimeter_nrk(ch_s, chq_s, pi0_s, nu_s, tau4, cls)
    h_hyb = np.nan_to_num(h_hyb)
    f = {}
    b1, b2 = d["b1"], d["b2"]
    bb = b1 + b2
    ttv = vis.sum(1)
    tt = mm["ditau4"]
    f["m_tautau"] = mm["m_maxw"]
    f["m_bb"] = mass(bb)
    f["m_hh"] = mass(bb + tt)
    f["m_hh_star"] = f["m_hh"] - f["m_bb"] - f["m_tautau"] + 250.0
    f["dr_tautau"] = delta_r(vis[:, 0], vis[:, 1])
    f["dr_bb"] = delta_r(b1, b2)
    lead = np.where(pt(vis[:, 0]) >= pt(vis[:, 1]), 0, 1)
    t1 = np.take_along_axis(vis, lead[:, None, None], 1)[:, 0]
    t2 = np.take_along_axis(vis, (1 - lead)[:, None, None], 1)[:, 0]
    for nm, p in (("tau1", t1), ("tau2", t2), ("b1", b1), ("b2", b2)):
        f[f"pt_{nm}"] = pt(p)
        f[f"eta_{nm}"] = eta(p)
    metv = np.hypot(met[:, 0], met[:, 1])
    metphi = np.arctan2(met[:, 1], met[:, 0])
    f["met"] = metv
    for nm, p in (("tau1", t1), ("tau2", t2)):
        f[f"mt_{nm}"] = np.sqrt(np.maximum(2 * pt(p) * metv * (1 - np.cos(dphi(phi(p), metphi))), 0))
    # MET phi centrality (ATLAS definition)
    a, b = phi(t1), phi(t2)
    sab = np.sin(b - a)
    sab = np.where(np.abs(sab) < 1e-6, 1e-6, sab)
    A = np.sin(metphi - a) / sab
    B = np.sin(b - metphi) / sab
    f["met_centrality"] = (A + B) / np.sqrt(A ** 2 + B ** 2)
    f["dphi_bb_tautau"] = np.abs(dphi(phi(bb), phi(ttv)))
    f["pt_bb"] = pt(bb)
    f["pt_tautau"] = pt(tt)
    f["pt_hh"] = pt(bb + tt)
    f["ht"] = d["ht"]
    f["njet"] = d["njet"].astype(float)
    f["dr_b1_tau1"] = delta_r(b1, t1)
    f["dr_b1_tau2"] = delta_r(b1, t2)
    f["m_vis"] = mass(ttv)
    f["mmc_status"] = mm["status"]
    # spin inputs (side 0 = tau-, side 1 = tau+)
    spin = {}
    e_tau = np.where(swap[:, None], mm["e_tau"][:, ::-1], mm["e_tau"])
    for s in (0, 1):
        ev = vis_s[:, s, 3]
        e_ch = ch_s[:, s].sum(1)[:, 3]
        e_pi0 = pi0_s[:, s].sum(1)[:, 3]
        spin[f"upsilon_{s}"] = (e_ch - e_pi0) / np.maximum(e_ch + e_pi0, 1e-6)
        spin[f"x_{s}"] = ev / np.maximum(e_tau[:, s], ev)
        spin[f"mvis_{s}"] = mass(vis_s[:, s])
        for m in range(5):
            spin[f"rmode{m}_{s}"] = (rmode_s[:, s] == m).astype(float)
        n_, r_, k_ = local_basis(vis_s[:, s])
        # leading-track impact vector in the local basis [um]
        ip = ip_s[:, s, 0] * 1e3
        spin[f"ip_n_{s}"], spin[f"ip_r_{s}"], spin[f"ip_k_{s}"] = (ip * n_).sum(-1), (ip * r_).sum(-1), (ip * k_).sum(-1)
        sv = sv_s[:, s]
        L = np.linalg.norm(sv, axis=-1)
        three = nch_s[:, s] >= 3
        spin[f"sv_n_{s}"] = np.where(three, (sv * n_).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0.0)
        spin[f"sv_r_{s}"] = np.where(three, (sv * r_).sum(-1) / np.maximum(L, 1e-9) * 1e3, 0.0)
        spin[f"sv_L_{s}"] = np.where(three, L, 0.0)
        spin[f"met_n_{s}"] = (np.concatenate([met, np.zeros((len(met), 1))], 1) * n_).sum(-1)
        spin[f"met_r_{s}"] = (np.concatenate([met, np.zeros((len(met), 1))], 1) * r_).sum(-1)
        # visible components in the visible-tau rest frame, local basis
        for c in range(3):
            comp = ch_s[:, s, c]
            spin[f"ch{c}_frac_{s}"] = comp[:, 3] / np.maximum(ev, 1e-6)
            spin[f"ch{c}_dn_{s}"] = (comp[:, :3] * n_).sum(-1) / np.maximum(ev, 1e-6) * 1e3
            spin[f"ch{c}_dr_{s}"] = (comp[:, :3] * r_).sum(-1) / np.maximum(ev, 1e-6) * 1e3
        for c in range(2):
            comp = pi0_s[:, s, c]
            spin[f"pi0{c}_frac_{s}"] = comp[:, 3] / np.maximum(ev, 1e-6)
            spin[f"pi0{c}_dn_{s}"] = (comp[:, :3] * n_).sum(-1) / np.maximum(ev, 1e-6) * 1e3
            spin[f"pi0{c}_dr_{s}"] = (comp[:, :3] * r_).sum(-1) / np.maximum(ev, 1e-6) * 1e3
        for j, c in enumerate("nrk"):
            spin[f"hhyb_{c}_{s}"] = h_hyb[:, s, j]
    return f, spin, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recodir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    procs = ["hh", "zbb", "ttll", "ttlj", "zh", "tth"]
    data = {}
    for p in procs:
        files = sorted(glob.glob(f"{args.recodir}/{p}_run_*.npz"))
        if files:
            data[p] = load(p, files)
            print(p, len(data[p]["w"]), "events, yield", data[p]["w"].sum())
    # MMC dR table and MET resolution from split-0 true taus
    vis_l, nu_l, pr_l, dmet = [], [], [], []
    for p, d in data.items():
        tr = (d["uid"] % 3 == 0)
        for s in (0, 1):
            ok = tr & d["is_true"][:, s]
            vis_l.append(d["vis"][ok, s])
            nu_l.append(d["tau_nu_p4"][ok, s] if p != "ttlj" else d["tau_nu_p4"][ok, 0])
            pr_l.append(np.where(d["nch"][ok, s] >= 2, 3, 1))
        if p in ("hh", "zbb", "zh"):
            dmet.append(d["met"][tr] - d["met_true"][tr])
    table = build_dr_table(np.concatenate(vis_l), np.concatenate(nu_l), np.concatenate(pr_l))
    met_sigma = float(np.sqrt(0.5 * np.mean(np.concatenate(dmet) ** 2)))
    print("MET sigma per component", met_sigma)
    out = {}
    for p, d in data.items():
        f, spin, mm = features(d, table, met_sigma)
        for k, v in f.items():
            out.setdefault("kin_" + k, []).append(v)
        for k, v in spin.items():
            out.setdefault("spin_" + k, []).append(v)
        swap = d["q"][:, 0] > 0
        h = d.get("h_exact", np.full((len(d["w"]), 2, 3), np.nan))
        hcls = d.get("h_cls", np.full((len(d["w"]), 2), -1))
        # h_exact was stored with side 0 = tau- by true charge; reco charge equals true for true taus
        out.setdefault("h_exact", []).append(h)
        out.setdefault("h_cls", []).append(hcls)
        out.setdefault("w", []).append(d["w"])
        out.setdefault("proc", []).append(d["proc"])
        out.setdefault("uid", []).append(d["uid"])
        out.setdefault("is_true", []).append(d["is_true"])
        out.setdefault("tau_mother", []).append(d["tau_mother"])
        out.setdefault("m_tautau_true", []).append(mass(d["tau_p4"].sum(1)))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["dr_table"] = table
    out["met_sigma"] = np.array(met_sigma)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, len(out["w"]))


if __name__ == "__main__":
    main()
