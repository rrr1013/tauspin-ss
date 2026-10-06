"""Parametric HL-LHC detector response, tau_had tau_had bb selection and
per-event features for the di-Higgs spin study.

Response parameters are calibrated on the ATLAS Run-2 full-simulation tauspin
validation cohort (analysis/cp_mixing/data, 59,390 events):
  pi0 cluster energy   sigma/E = 0.95/sqrt(E) (+) 0.06 (+) 3.2/E   (measured
                       robust sd 0.62 / 0.36 / 0.22 / 0.15 / 0.087 for
                       E < 10 / 10-20 / 20-40 / 40-80 / > 80 GeV)
  pi0 cluster angle    sigma = 0.06/sqrt(E) (+) 0.003 rad, eta x0.7, phi x1.3
                       (measured dR68 0.038 / 0.023 / 0.015 / 0.011 / 0.007)
  charged track        sigma(pT)/pT = 0.02 (+) 4e-4 pT  (measured 0.040 at
                       the median visible pT of 74 GeV)
  rho with lost pi0    10 % (measured 6,714 / 66,868); pi with an extra
                       pi0 cluster 8.7 %; 3p -> 3pXn 3 %
  impact parameters    d0: 10 (+) 150/pT um, z0: 20 (+) 300/pT um (ATLAS tau
                       tracks 13-22 um transverse, 20-55 um longitudinal)
  3-prong SV           10 um transverse, 250 um along the flight direction
  MET                  truth MET minus the object mismeasurement, plus a
                       Gaussian soft term of 18 GeV per component
Jets: sigma/E = 0.8/sqrt(E) (+) 0.05, b-tag 77 % / c 20 % / light 1 %.
tau ID: 1-prong 0.85, 3-prong 0.75 (true taus); fakes enter with a per-jet
fake factor (1-prong 1e-2, 3-prong 3e-3) as an event weight.
Selection (tau_had tau_had DTT-like): two opposite-sign tau_had-vis with
pT > 40, 30 GeV, |eta| < 2.5; no prompt e/mu; exactly two b-tagged jets with
pT > 45, 20 GeV, |eta| < 2.5; m_tautau(MMC) > 60 GeV is applied downstream.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from polarimeter_pythia import classify, exact_h

TAU_MASS = 1.77686
FK_RATE = {1: 1e-2, 3: 3e-3}


# ---------------------------------------------------------------- kinematics
def pt(p):
    return np.hypot(p[..., 0], p[..., 1])


def eta(p):
    return np.arcsinh(p[..., 2] / np.maximum(pt(p), 1e-9))


def phi(p):
    return np.arctan2(p[..., 1], p[..., 0])


def mass(p):
    return np.sqrt(np.maximum(p[..., 3] ** 2 - (p[..., :3] ** 2).sum(-1), 0))


def from_ptetaphim(ptv, et, ph, m):
    px, py, pz = ptv * np.cos(ph), ptv * np.sin(ph), ptv * np.sinh(et)
    return np.stack([px, py, pz, np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)], -1)


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def delta_r(p, q):
    return np.hypot(eta(p) - eta(q), dphi(phi(p), phi(q)))


# ---------------------------------------------------------------- response
def smear_track(p, rng):
    ptv = pt(p)
    s = np.hypot(0.02, 4e-4 * ptv)
    f = 1 + s * rng.standard_normal(ptv.shape)
    out = p.copy()
    out[..., :3] *= f[..., None]
    out[..., 3] = np.sqrt((out[..., :3] ** 2).sum(-1) + 0.13957 ** 2)
    return np.where((p[..., 3] > 0)[..., None], out, 0.0)


def smear_pi0(p, rng):
    e = np.maximum(p[..., 3], 1e-6)
    s_e = np.sqrt((0.95 / np.sqrt(e)) ** 2 + 0.06 ** 2 + (3.2 / e) ** 2)
    e_new = np.maximum(e * (1 + s_e * rng.standard_normal(e.shape)), 0.3)
    s_a = np.hypot(0.06 / np.sqrt(e), 0.003)
    et = eta(p) + 0.7 * s_a * rng.standard_normal(e.shape)
    ph = phi(p) + 1.3 * s_a * rng.standard_normal(e.shape)
    pabs = np.sqrt(np.maximum(e_new ** 2 - 0.135 ** 2, 0))
    out = from_ptetaphim(pabs / np.cosh(et), et, ph, 0.135)
    return np.where((p[..., 3] > 0)[..., None], out, 0.0)


def smear_jet(p, rng):
    e = np.maximum(p[..., 3], 1e-6)
    s = np.hypot(0.8 / np.sqrt(e), 0.05)
    f = np.maximum(1 + s * rng.standard_normal(e.shape), 0.05)
    return np.where((p[..., 3] > 0)[..., None], p * f[..., None], 0.0)


def impact_vector(v0, pdir, pv):
    """3D impact vector of the straight line (v0, pdir) w.r.t. pv [mm]."""
    d = v0 - pv
    t = pdir / np.maximum(np.linalg.norm(pdir, axis=-1, keepdims=True), 1e-12)
    return d - (d * t).sum(-1, keepdims=True) * t


def smear_ip(ip, p, rng):
    """Smear an impact vector: transverse (d0) and longitudinal (z0) parts."""
    ptv = np.maximum(pt(p), 0.5)
    s_t = 1e-3 * np.hypot(10.0, 150.0 / ptv)
    s_z = 1e-3 * np.hypot(20.0, 300.0 / ptv)
    out = ip.copy()
    out[..., :2] += s_t[..., None] * rng.standard_normal(ip[..., :2].shape) / np.sqrt(2)
    out[..., 2] += s_z * rng.standard_normal(ip[..., 2].shape)
    return out


def smear_sv(v, flight_dir, rng):
    u = flight_dir / np.maximum(np.linalg.norm(flight_dir, axis=-1, keepdims=True), 1e-12)
    g = rng.standard_normal(v.shape) * 0.010
    g = g - (g * u).sum(-1, keepdims=True) * u
    return v + g + u * (0.250 * rng.standard_normal(v.shape[:-1]))[..., None]


# ---------------------------------------------------------------- per-tau reco
def reco_true_taus(d, rng):
    """Reconstructed visible components of the stored (true) taus."""
    n = len(d["weight"])
    cls = classify(d["tau_nch"], d["tau_npi0"], d["tau_nlep"], d["tau_nother"])
    had = d["tau_ok"] & (d["tau_nlep"] == 0)
    ch = smear_track(d["tau_ch_p4"], rng)
    pi0 = smear_pi0(d["tau_pi0_p4"], rng)
    oth = smear_jet(d["tau_oth_p4"], rng)
    # reco mode (0 1p0n, 1 1p1n, 2 1pXn, 3 3p0n, 4 3pXn) with ATLAS-like migration
    nch, npi0 = d["tau_nch"], d["tau_npi0"]
    u = rng.random((n, 2))
    rmode = np.full((n, 2), -1, np.int8)
    one = nch == 1
    three = nch == 3
    rmode[one & (npi0 == 0)] = np.where(u[one & (npi0 == 0)] < 0.087, 1, 0)
    lost = one & (npi0 == 1) & (u < 0.10)
    rmode[one & (npi0 == 1)] = 1
    rmode[lost] = 0
    rmode[one & (npi0 >= 2)] = np.where(u[one & (npi0 >= 2)] < 0.45, 1, 2)
    rmode[three & (npi0 == 0)] = np.where(u[three & (npi0 == 0)] < 0.03, 4, 3)
    rmode[three & (npi0 >= 1)] = np.where(u[three & (npi0 >= 1)] < 0.25, 3, 4)
    # pi0 clusters seen by the substructure algorithm
    pi0_seen = pi0.copy()
    pi0_seen[lost] = 0.0
    extra = one & (npi0 == 0) & (rmode == 1)
    if extra.any():          # fake pi0 cluster from pile-up / converted photon near the track
        trk = ch[extra][:, 0]
        e_fake = rng.uniform(1.0, 5.0, len(trk))
        etf = eta(trk) + 0.02 * rng.standard_normal(len(trk))
        phf = phi(trk) + 0.02 * rng.standard_normal(len(trk))
        pi0_seen[extra, 0] = from_ptetaphim(e_fake / np.cosh(etf), etf, phf, 0.135)
    vis = ch.sum(2) + pi0.sum(2) + oth
    vis_true = d["tau_ch_p4"].sum(2) + d["tau_pi0_p4"].sum(2) + d["tau_oth_p4"]
    extra_vis = np.where(extra[..., None], pi0_seen[:, :, 0], 0.0)
    vis = vis + extra_vis
    # geometry
    pv = np.zeros((n, 3))
    ip = impact_vector(d["tau_vdec"][:, :, None, :], d["tau_ch_p4"][..., :3], pv[:, None, None, :])
    ip = smear_ip(ip, d["tau_ch_p4"], rng)
    flight = d["tau_vdec"] - d["tau_vprod"]
    sv = smear_sv(d["tau_vdec"], flight, rng)
    return dict(ok=had, cls=cls, rmode=rmode, ch=ch, ch_q=d["tau_ch_q"], pi0=pi0_seen, vis=vis,
                vis_true=vis_true, ip=ip, sv=sv, nch=nch, q=d["tau_q"].astype(int))


def reco_fake_taus(d, rng):
    """Fake tau_had-vis candidate: highest-pT eligible light jet core."""
    n, nj = d["fk_nch"].shape
    nch, npi0 = d["fk_nch"], d["fk_npi0"]
    jp = d["jet_p4"]
    qsum = d["fk_ch_q"].sum(-1)
    elig = ((nch == 1) | (nch == 3)) & (np.abs(qsum) == 1) & (d["jet_flav"] == 0) \
        & (d["fk_iso_n"] == 0) & (pt(jp) > 20) & (np.abs(eta(jp)) < 2.5)
    first = np.where(elig.any(1), elig.argmax(1), -1)
    sel = first >= 0
    idx = np.clip(first, 0, nj - 1)
    g = lambda a: np.take_along_axis(a, idx.reshape((n, 1) + (1,) * (a.ndim - 2)), 1)[:, 0]
    ch = smear_track(g(d["fk_ch_p4"])[:, :3], rng)
    pi0 = smear_pi0(g(d["fk_pi0_p4"])[:, :2], rng)
    jet = g(jp)
    neu = g(d["fk_neu_e"])
    vis = ch.sum(1) + pi0.sum(1)
    # remaining neutral hadronic energy enters the visible tau along its axis
    dirv = vis[:, :3] / np.maximum(np.linalg.norm(vis[:, :3], axis=-1, keepdims=True), 1e-9)
    vis = vis + np.concatenate([dirv * neu[:, None], neu[:, None]], -1)
    nc, npz = g(nch), g(npi0)
    rmode = np.where(nc == 1, np.where(npz == 0, 0, np.where(npz == 1, 1, 2)), np.where(npz == 0, 3, 4))
    v = g(d["fk_ch_v"])[:, :3]
    ip = impact_vector(v, g(d["fk_ch_p4"])[:, :3, :3], np.zeros((n, 1, 3)))
    ip = smear_ip(ip, g(d["fk_ch_p4"])[:, :3], rng)
    sv = smear_sv(v[:, 0], jet[:, :3], rng)
    rate = np.where(nc == 1, FK_RATE[1], FK_RATE[3])
    return dict(sel=sel, idx=idx, ch=ch, ch_q=g(d["fk_ch_q"])[:, :3], pi0=pi0, vis=vis, vis_true=jet,
                rmode=rmode.astype(np.int8), ip=ip, sv=sv, nch=nc, q=g(d["fk_ch_q"]).sum(-1),
                rate=rate)


# ---------------------------------------------------------------- event selection
def process(path, proc, seed, no_btag=False):
    d = dict(np.load(path))
    rng = np.random.default_rng(seed)
    n = len(d["weight"])
    lhe = np.sign(d["weight"])          # LHE weights are +-sigma; normalisation is applied downstream
    t = reco_true_taus(d, rng)
    tau_id_eff = np.where(t["nch"] == 1, 0.85, np.where(t["nch"] == 3, 0.75, 0.0))
    good_tau = t["ok"] & (t["rmode"] >= 0)
    if proc == "ttlj":
        f = reco_fake_taus(d, rng)
        # side 0: the true tau_had, side 1: the fake candidate
        true_side = good_tau[:, 0]
        sides = {k: np.stack([t[k][:, 0], f[k]], 1) for k in ("ch", "ch_q", "pi0", "vis", "vis_true", "rmode", "ip", "sv", "nch")}
        sides["q"] = np.stack([t["q"][:, 0], f["q"]], 1)
        evw = lhe * tau_id_eff[:, 0] * f["rate"]
        base = true_side & f["sel"]
        is_true = np.stack([np.ones(n, bool), np.zeros(n, bool)], 1)
        exclude_jet = f["idx"]
    else:
        sides = {k: t[k] for k in ("ch", "ch_q", "pi0", "vis", "vis_true", "rmode", "ip", "sv", "nch", "q")}
        evw = lhe * tau_id_eff[:, 0] * tau_id_eff[:, 1]
        base = good_tau.all(1)
        is_true = np.ones((n, 2), bool)
        exclude_jet = np.full(n, -1)
    if proc == "hh":
        base &= (d["n_h_bb"] == 1) & (d["n_h_tt"] == 1)
    vis = sides["vis"]
    # order sides by visible pT for the pT thresholds
    lead = np.where(pt(vis[:, 0]) >= pt(vis[:, 1]), 0, 1)
    pt1 = pt(np.take_along_axis(vis, lead[:, None, None], 1)[:, 0])
    pt2 = pt(np.take_along_axis(vis, (1 - lead)[:, None, None], 1)[:, 0])
    sel = base & (pt1 > 40) & (pt2 > 30) & (np.abs(eta(vis)) < 2.5).all(1) \
        & (sides["q"].sum(1) == 0) & (d["nlep_prompt"] == 0)
    # jets: drop the fake-tau jet and jets overlapping a visible tau
    jets = smear_jet(d["jet_p4"], rng)
    jt = d["jet_p4"]
    njet = jt.shape[1]
    keep = (jt[..., 3] > 0)
    keep &= ~(np.arange(njet)[None, :] == exclude_jet[:, None])
    for s in (0, 1):
        keep &= ~(delta_r(jt, vis[:, s][:, None, :]) < 0.4)
    flav = d["jet_flav"]
    eff = np.where(flav == 5, 0.77, np.where(flav == 4, 0.20, 0.01))
    tagged = keep & (rng.random(flav.shape) < eff) & (np.abs(eta(jets)) < 2.5) & (pt(jets) > 20)
    nb = tagged.sum(1)
    order = np.argsort(-np.where(tagged, pt(jets), -1), axis=1)
    b1 = np.take_along_axis(jets, order[:, :1, None], 1)[:, 0]
    b2 = np.take_along_axis(jets, order[:, 1:2, None], 1)[:, 0]
    if not no_btag:
        sel &= (nb == 2) & (pt(b1) > 45) & (pt(b2) > 20)
    other = keep & ~tagged & (pt(jets) > 20) & (np.abs(eta(jets)) < 4.5)
    # MET: truth minus object mismeasurement plus soft term
    dvis = (vis - sides["vis_true"])[..., :2].sum(1)
    djet = ((jets - jt)[..., :2] * keep[..., None]).sum(1)
    met = d["met_true"] - dvis - djet + 18.0 * rng.standard_normal((n, 2))
    if proc == "ttlj":
        # the fake-tau jet is a jet, not missing momentum: its reco - truth shift is in dvis already
        pass
    s = np.flatnonzero(sel)
    h, hcls, horder, hboth = exact_h(d) if proc != "ttlj" else (np.full((n, 2, 3), np.nan),) * 1 + (None, None, None)
    out = {
        "proc": np.array(proc), "weight": evw[s], "event": s,
        "vis": vis[s], "vis_true": sides["vis_true"][s], "q": sides["q"][s], "rmode": sides["rmode"][s],
        "ch": sides["ch"][s], "ch_q": sides["ch_q"][s], "pi0": sides["pi0"][s], "ip": sides["ip"][s],
        "sv": sides["sv"][s], "nch": sides["nch"][s], "is_true": is_true[s],
        "b1": b1[s], "b2": b2[s], "met": met[s], "met_true": d["met_true"][s],
        "ht": (pt(jets) * (other | tagged)).sum(1)[s], "njet": (other | tagged).sum(1)[s],
        "tau_mother": d["tau_mother"][s], "tau_p4": d["tau_p4"][s], "tau_nu_p4": d["tau_nu_p4"][s],
        "sigma_pb": d["sigma_pb"], "n_gen": np.array(n),
    }
    if proc != "ttlj":
        # exact_h orders the sides tau-, tau+ (true charge); the features use the same order
        out["h_exact"] = h[s]
        out["h_cls"] = hcls[s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--no-btag", action="store_true", help="h-regressor training samples: no b-jet requirement")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        name = Path(path).stem
        proc = name.split("_run_")[0]
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        out = process(path, proc, seed, args.no_btag)
        out = {k: v for k, v in out.items() if v is not None}
        np.savez_compressed(Path(args.outdir) / f"{name}.npz", **out)
        print(name, "selected", len(out["weight"]), "of", int(out["n_gen"]))


if __name__ == "__main__":
    main()
