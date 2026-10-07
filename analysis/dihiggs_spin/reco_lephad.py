"""tau_lep tau_had bb selection with lepton impact parameters (extension of reco.py).

Lepton candidates are the e/mu from leptonic tau decays (taus from W/Z/H; impact
vector from the tau decay vertex along the lepton direction) and the prompt e/mu
from W/Z decays (impact vector zero).  Smearing (Run-2 ATLAS-like, as the tau
tracks of reco.py, which are calibrated on the tauspin ATLAS full-simulation cohort):
  d0  sigma = 10 (+) 150/pT um (muons), x1.2 for electrons (bremsstrahlung),
  z0  sigma = 20 (+) 300/pT um,
  a 2 % non-Gaussian tail with 4x the core width for every lepton (prompt included);
  primary vertex at the origin (beam-line constrained).
Momentum: muon 2 % (+) 0.03 % pT, electron 1.5 % (+) 10 %/sqrt(E).
Identification and isolation: 0.90 (e), 0.95 (mu) per lepton, as an event weight.
The standard ATLAS track-to-vertex association |d0/sigma| < 5 (e) / 3 (mu),
|z0 sin theta| < 0.5 mm is stored as a flag (`pass_ttva`), not applied.

tau_had: a true hadronic tau (reco.py response, tau ID 0.85 / 0.75) or, for the
processes with a light jet and no hadronic tau (ttlj, ttljp, tWlj), the fake candidate
of reco.py with its fake factor as event weight.
Selection (ATLAS arXiv:2209.10910): exactly one lepton (e pT > 18, mu pT > 15, |eta| < 2.5)
and one tau_had-vis (|eta| < 2.3) of opposite charge; SLT: lepton pT > 27, tau pT > 20;
LTT: lepton below the SLT threshold and tau pT > 30; exactly two b-tagged jets (pT > 45,
20 GeV).  m_bb < 150 GeV and m_tautau(MMC) > 60 GeV are applied downstream.  ATLAS does not
state a d0-significance requirement for the signal-region leptons; the standard
track-to-vertex association is kept as a flag for a scenario.
"""
import argparse
from pathlib import Path

import numpy as np

from polarimeter_pythia import exact_h
from reco import (delta_r, eta, impact_vector, phi, pt, reco_fake_taus, reco_true_taus,
                  smear_jet)

FAKE_PROCS = ("ttlj", "ttljp", "tWlj")
LEP_EFF = {11: 0.90, 13: 0.95}


def lepton_resolution(p, lid):
    ptv = np.maximum(pt(p), 0.5)
    s_d0 = np.hypot(10.0, 150.0 / ptv) * np.where(np.abs(lid) == 11, 1.2, 1.0)
    s_z0 = np.hypot(20.0, 300.0 / ptv)
    return s_d0, s_z0            # um


def smear_lepton(p, lid, rng):
    e = p[..., 3]
    rel = np.where(np.abs(lid) == 13, np.hypot(0.02, 3e-4 * pt(p)), np.hypot(0.015, 0.10 / np.sqrt(np.maximum(e, 1e-3))))
    return p * (1 + rel * rng.standard_normal(e.shape))[..., None]


def smear_lepton_ip(ip_um, p, lid, rng):
    """ip_um: true 3D impact vector [um]; returns smeared vector and (s_d0, s_z0)."""
    s_d0, s_z0 = lepton_resolution(p, lid)
    tail = np.where(rng.random(s_d0.shape) < 0.02, 4.0, 1.0)
    out = ip_um.copy()
    out[..., :2] += (s_d0 * tail)[..., None] * rng.standard_normal(ip_um[..., :2].shape) / np.sqrt(2)
    out[..., 2] += s_z0 * tail * rng.standard_normal(ip_um[..., 2].shape)
    return out, s_d0, s_z0


def signed_d0(ip, p):
    """Transverse impact parameter with the ATLAS sign convention (sign of (ip x p)_z)."""
    return np.sign(ip[..., 0] * p[..., 1] - ip[..., 1] * p[..., 0]) * np.hypot(ip[..., 0], ip[..., 1])


def process(path, proc, seed):
    d = dict(np.load(path))
    rng = np.random.default_rng(seed)
    n = len(d["weight"])
    lhe = np.sign(d["weight"])
    t = reco_true_taus(d, rng)
    # ---------------- leptons: up to 2 from tau decays + up to 2 prompt
    good_mother = np.isin(np.abs(d["tau_mother"]), (23, 24, 25))
    tl_ok = d["tau_ok"] & (d["tau_nlep"] == 1) & good_mother
    tl_p4 = d["tau_lep_p4"]
    tl_id = d.get("tau_lep_id", np.where(rng.random((n, 2)) < 0.5, 11, 13) * -d["tau_q"])
    tl_ip = impact_vector(d["tau_vdec"], tl_p4[..., :3], np.zeros((n, 1, 3))) * 1e3
    pl_ok = d["lep_id"] != 0
    cand_p4 = np.concatenate([tl_p4, d["lep_p4"]], 1)            # (n, 4, 4)
    cand_id = np.concatenate([tl_id, d["lep_id"]], 1)
    cand_ok = np.concatenate([tl_ok, pl_ok], 1)
    cand_from_tau = np.array([True, True, False, False])[None, :].repeat(n, 0)
    cand_ip_true = np.concatenate([tl_ip, np.zeros((n, 2, 3))], 1)
    cand_p4r = smear_lepton(cand_p4, cand_id, rng)
    ptl = pt(cand_p4r)
    thr = np.where(np.abs(cand_id) == 11, 18.0, 15.0)
    sel_l = cand_ok & (ptl > thr) & (np.abs(eta(cand_p4r)) < 2.5) & np.isin(np.abs(cand_id), (11, 13))
    nlep = sel_l.sum(1)
    li = np.argmax(np.where(sel_l, ptl, -1), 1)
    g = lambda a: np.take_along_axis(a, li.reshape((n, 1) + (1,) * (a.ndim - 2)), 1)[:, 0]
    lep, lid, from_tau, ip_true = g(cand_p4r), g(cand_id), g(cand_from_tau), g(cand_ip_true)
    ip_reco, s_d0, s_z0 = smear_lepton_ip(ip_true, lep, lid, rng)
    lq = -np.sign(lid)                         # e-/mu- have positive PDG id
    lep_tau_side = np.where(from_tau, li, -1)  # which stored tau produced the lepton
    # ---------------- tau_had
    had = t["ok"] & (t["rmode"] >= 0)
    had &= ~(np.arange(2)[None, :] == lep_tau_side[:, None])
    vpt = np.where(had, pt(t["vis"]), -1)
    ti = np.argmax(vpt, 1)
    has_true = had.any(1)
    gt = lambda a: np.take_along_axis(a, ti.reshape((n, 1) + (1,) * (a.ndim - 2)), 1)[:, 0]
    tau = {k: gt(t[k]) for k in ("ch", "ch_q", "pi0", "vis", "vis_true", "rmode", "ip", "sv", "nch", "q")}
    tau_id_eff = np.where(tau["nch"] == 1, 0.85, np.where(tau["nch"] == 3, 0.75, 0.0))
    evw = lhe * np.where(np.abs(lid) == 11, LEP_EFF[11], LEP_EFF[13])
    is_true = has_true.copy()
    exclude_jet = np.full(n, -1)
    if proc in FAKE_PROCS:
        f = reco_fake_taus(d, rng)
        use_fake = ~has_true & f["sel"]
        for k in ("ch", "ch_q", "pi0", "vis", "vis_true", "rmode", "ip", "sv", "nch", "q"):
            tau[k] = np.where(use_fake.reshape((n,) + (1,) * (tau[k].ndim - 1)), f[k], tau[k])
        tau_w = np.where(use_fake, f["rate"], tau_id_eff)
        exclude_jet = np.where(use_fake, f["idx"], -1)
        has_tau = has_true | use_fake
        is_true = has_true
    else:
        tau_w = tau_id_eff
        has_tau = has_true
    evw = evw * tau_w
    base = (nlep == 1) & has_tau & (tau["q"] * lq < 0)
    if proc in ("hh", "hhC"):
        base &= (d["n_h_bb"] == 1) & (d["n_h_tt"] == 1)
    ptt, ptlep = pt(tau["vis"]), pt(lep)
    slt = ptlep > 27
    ltt = ~slt & (ptt > 30)
    base &= np.where(slt, ptt > 20, ptt > 30) & (np.abs(eta(tau["vis"])) < 2.3) & (slt | ltt) \
        & (delta_r(lep, tau["vis"]) > 0.2)
    # ---------------- jets / b-tagging (reco.py)
    jets = smear_jet(d["jet_p4"], rng)
    jt = d["jet_p4"]
    nj = jt.shape[1]
    keep = (jt[..., 3] > 0) & ~(np.arange(nj)[None, :] == exclude_jet[:, None])
    keep &= ~(delta_r(jt, tau["vis"][:, None, :]) < 0.4)
    keep &= ~(delta_r(jt, lep[:, None, :]) < 0.4)
    flav = d["jet_flav"]
    eff = np.where(flav == 5, 0.77, np.where(flav == 4, 0.20, 0.01))
    tagged = keep & (rng.random(flav.shape) < eff) & (np.abs(eta(jets)) < 2.5) & (pt(jets) > 20)
    nb = tagged.sum(1)
    order = np.argsort(-np.where(tagged, pt(jets), -1), axis=1)
    b1 = np.take_along_axis(jets, order[:, :1, None], 1)[:, 0]
    b2 = np.take_along_axis(jets, order[:, 1:2, None], 1)[:, 0]
    sel = base & (nb == 2) & (pt(b1) > 45) & (pt(b2) > 20)
    other = keep & ~tagged & (pt(jets) > 20) & (np.abs(eta(jets)) < 4.5)
    # MET: truth minus object mismeasurement plus soft term
    dvis = (tau["vis"] - tau["vis_true"])[:, :2] + (lep - g(cand_p4))[:, :2]
    djet = ((jets - jt)[..., :2] * keep[..., None]).sum(1)
    met = d["met_true"] - dvis - djet + 18.0 * rng.standard_normal((n, 2))
    # spin truth of the hadronic side (tau-, tau+ order of exact_h)
    if proc in FAKE_PROCS:
        h_had = np.full((n, 3), np.nan)
        lep_mother = np.zeros(n, np.int32)
    else:
        h, hcls, horder, hboth = exact_h(d)
        # exact_h orders tau-, tau+ by true charge; the stored tau index ti has charge tau_q
        side = np.where(np.take_along_axis(d["tau_q"], ti[:, None], 1)[:, 0] < 0, 0, 1)
        h_had = np.take_along_axis(h, side[:, None, None], 1)[:, 0]
    lep_mother = np.where(from_tau, np.take_along_axis(d["tau_mother"], np.clip(li, 0, 1)[:, None], 1)[:, 0], 0)
    d0 = signed_d0(ip_reco, lep)
    sth = pt(lep) / np.maximum(np.linalg.norm(lep[:, :3], axis=-1), 1e-9)
    z0s = ip_reco[:, 2] * sth
    pass_ttva = (np.abs(d0) / s_d0 < np.where(np.abs(lid) == 11, 5.0, 3.0)) & (np.abs(z0s) < 500.0)
    s = np.flatnonzero(sel)
    return {
        "proc": np.array(proc), "weight": evw[s], "event": s, "trigger": np.where(slt, 0, 1)[s],
        "lep": lep[s], "lep_id": lid[s], "lep_from_tau": from_tau[s], "lep_mother": lep_mother[s],
        "lep_ip": ip_reco[s], "lep_ip_true": ip_true[s], "lep_sd0": s_d0[s], "lep_sz0": s_z0[s],
        "lep_d0": d0[s], "lep_z0sth": z0s[s], "pass_ttva": pass_ttva[s],
        "tau_vis": tau["vis"][s], "tau_vis_true": tau["vis_true"][s], "tau_rmode": tau["rmode"][s],
        "tau_ch": tau["ch"][s], "tau_ch_q": tau["ch_q"][s], "tau_pi0": tau["pi0"][s], "tau_ip": tau["ip"][s],
        "tau_sv": tau["sv"][s], "tau_nch": tau["nch"][s], "tau_is_true": is_true[s], "h_had": h_had[s],
        "b1": b1[s], "b2": b2[s], "met": met[s], "met_true": d["met_true"][s],
        "ht": (pt(jets) * (other | tagged)).sum(1)[s], "njet": (other | tagged).sum(1)[s],
        "lep_tau_idx": lep_tau_side[s], "tau_had_idx": np.where(is_true, ti, -1)[s],
        "tau_p4": d["tau_p4"][s], "tau_nu_p4": d["tau_nu_p4"][s], "tau_mother": d["tau_mother"][s],
        "sigma_pb": d["sigma_pb"], "n_gen": np.array(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    for path in args.inputs:
        name = Path(path).name
        proc = name.split("_run")[0]
        seed = int.from_bytes(name.encode()[-8:], "little") % (2 ** 31)
        out = process(path, proc, seed)
        np.savez_compressed(Path(args.outdir) / name, **out)
        print(name, "selected", len(out["weight"]), "sum w", float(out["weight"].sum()))


if __name__ == "__main__":
    main()
