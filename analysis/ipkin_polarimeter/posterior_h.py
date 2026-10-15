"""Classical (non-ML) posterior polarimeter for the spin-flat H(125)+jet sample.

For each event the tau pair is reconstructed with the IP/SV-assisted likelihood of
analysis/ip_tau_reco/ipmmc2.py, without any learned component:

  prior       unpolarised tau decay: x = E_vis / E_tau uniform on [m_vis^2/m_tau^2, 1],
              azimuth phi of the tau around the visible axis uniform (mass shell exact);
  1-prong     tau direction in the (track, impact-vector) plane (Gaussian in the
              azimuth around the track) x exponential decay-length prior at L = |b|/sin a;
  3-prong     tau direction along PV -> SV (angular Gaussian) x decay-length prior;
  MET         Gaussian in both transverse components (couples the two sides).

Sampling.  Each side is sampled from its prior (S_side draws) and weighted by its own
geometry likelihood; M draws per side are kept by systematic resampling.  The M x M
pairs are weighted by the MET likelihood and K pairs are resampled.  On every kept
pair the exact polarimeter of the reconstructed visible system (polarimeter_nrk, the
generator currents, shared (n, r, k) pair basis) is evaluated, so the outputs are
posterior means under the spin-flat prior:
    Eh  (2, 3)   E[h-], E[h+]
    Ehh (3, 3)   E[h-_i h+_j]
use_ip = False drops the geometry likelihood (MET + prior only): the same estimator
without impact parameters or secondary vertices.  mass_prior = (m, sigma) adds a
Gaussian m_tautau term to the pair weight (the signal-mass hypothesis; the learned
regressor was trained on equal-mass 125 GeV H/Z and carries this prior implicitly).

Side order and inputs follow build_dataset.features: side 0 = tau- by reco charge,
visible mass capped at 1.6 GeV for the mass shell only, leading-track impact vector,
reco decay class RMODE_TO_CLS[rmode] (3pXn has no polarimeter).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dihiggs_spin"))
sys.path.insert(0, str(HERE.parent / "ip_tau_reco"))
from polarimeter_pythia import polarimeter_nrk  # noqa: E402
from ipmmc2 import sample_tau  # noqa: E402

RMODE_TO_CLS = np.array([0, 1, 2, 3, -1])
SIG_D0 = lambda pt: 1e-3 * np.hypot(10.0, 150.0 / np.maximum(pt, 0.5))   # mm, reco.py smear_ip
SIG_Z0 = lambda pt: 1e-3 * np.hypot(20.0, 300.0 / np.maximum(pt, 0.5))
SIG_SV_PERP = 0.010                                                       # mm, reco.py smear_sv


def load_reco(recodir):
    """Concatenate the reco chunks in the order of build_dataset.load (sorted glob)."""
    files = sorted(glob.glob(f"{recodir}/trainHU_run_*.npz"))
    keys = ["vis", "ch", "ch_q", "pi0", "ip", "sv", "nch", "rmode", "q", "met", "met_true",
            "tau_p4", "tau_nu_p4", "h_exact"]
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["uid"] = np.concatenate([i * 1_000_000 + p["event"] for i, p in enumerate(parts)])
    return d


def ordered(d):
    """Side 0 = tau- by reco charge, as in build_dataset.features."""
    swap = d["q"][:, 0] > 0
    S = lambda a: np.where(swap.reshape((-1,) + (1,) * (a.ndim - 1)), a[:, ::-1], a)
    keys = ["vis", "ch", "ch_q", "pi0", "ip", "sv", "nch", "rmode", "tau_p4", "tau_nu_p4"]
    o = {k: S(d[k]) for k in keys}
    o["met"] = d["met"]
    return o


def build_sides(o):
    sides = []
    for s in (0, 1):
        trk_p = o["ch"][:, s, 0, :3]
        trk = trk_p / np.linalg.norm(trk_p, axis=-1, keepdims=True)
        pt = np.hypot(trk_p[:, 0], trk_p[:, 1])
        b = o["ip"][:, s, 0]
        b = b - (b * trk).sum(-1, keepdims=True) * trk
        bmag = np.linalg.norm(b, axis=-1)
        e1 = b / np.maximum(bmag, 1e-12)[:, None]
        e2 = np.cross(trk, e1)
        var = np.stack([SIG_D0(pt) ** 2 / 2, SIG_D0(pt) ** 2 / 2, SIG_Z0(pt) ** 2], -1)
        sig_bperp = np.sqrt((e2 ** 2 * var).sum(-1))
        svv = o["sv"][:, s]
        svl = np.linalg.norm(svv, axis=-1)
        vis = o["vis"][:, s].copy()
        mv = np.sqrt(np.maximum(vis[:, 3] ** 2 - (vis[:, :3] ** 2).sum(-1), 0))
        vis[:, 3] = np.sqrt((vis[:, :3] ** 2).sum(-1) + np.minimum(mv, 1.6) ** 2)
        sides.append(dict(vis=vis, trk=trk, e1=e1, e2=e2, bmag=bmag,
                          sig_psi=sig_bperp / np.maximum(bmag, 1e-9),
                          three=o["nch"][:, s] >= 3, sv_dir=svv / np.maximum(svl, 1e-12)[:, None],
                          sv_len=svl, sig_sv=SIG_SV_PERP / np.maximum(svl, 1e-3)))
    return sides


def systematic(w, M, gen):
    """Systematic resampling indices (B, M) from normalised weights w (B, S)."""
    B = w.shape[0]
    c = torch.cumsum(w, 1)
    c[:, -1] = 1.0
    u = (torch.arange(M, dtype=w.dtype)[None, :] + torch.rand(B, 1, generator=gen, dtype=w.dtype)) / M
    return torch.clamp(torch.searchsorted(c, u), max=w.shape[1] - 1)


def ess(w):
    return (w.sum(1) ** 2 / torch.clamp((w ** 2).sum(1), min=1e-300))


def pi0_candidates(pi0, R, rng):
    """R draws of the true pi0 clusters given the measured ones, from the reco.py response
    (sigma_E/E = 0.95/sqrt(E) + 0.06 + 3.2/E, angle 0.06/sqrt(E) + 3 mrad, eta x0.7, phi x1.3),
    evaluated at the measured energy.  pi0 (B, 2, 2, 4) -> (B, R, 2, 2, 4)."""
    e = np.maximum(pi0[..., 3], 1e-6)
    s_e = np.sqrt((0.95 / np.sqrt(e)) ** 2 + 0.06 ** 2 + (3.2 / e) ** 2)
    s_a = np.hypot(0.06 / np.sqrt(e), 0.003)
    sh = (pi0.shape[0], R) + pi0.shape[1:-1]
    E = np.maximum(e[:, None] * (1 + s_e[:, None] * rng.standard_normal(sh)), 0.3)
    pt = np.hypot(pi0[..., 0], pi0[..., 1])
    et = np.arcsinh(pi0[..., 2] / np.maximum(pt, 1e-9))[:, None] + 0.7 * s_a[:, None] * rng.standard_normal(sh)
    ph = np.arctan2(pi0[..., 1], pi0[..., 0])[:, None] + 1.3 * s_a[:, None] * rng.standard_normal(sh)
    p = np.sqrt(np.maximum(E ** 2 - 0.135 ** 2, 0))
    c = np.stack([p / np.cosh(et) * np.cos(ph), p / np.cosh(et) * np.sin(ph), p * np.tanh(et), E], -1)
    c[0 * R:] = c
    out = np.where((pi0[..., 3] > 0)[:, None, ..., None], c, 0.0)
    out[:, 0] = pi0       # replica 0 keeps the measured clusters
    return out


def cap_mass(vis):
    mv = np.sqrt(np.maximum(vis[..., 3] ** 2 - (vis[..., :3] ** 2).sum(-1), 0))
    out = vis.copy()
    out[..., 3] = np.sqrt((vis[..., :3] ** 2).sum(-1) + np.minimum(mv, 1.6) ** 2)
    return out


def run_batch(o, sides, idx, met_sigma, use_ip, S_side, M, K, gen, mass_prior=None, R=1, rng=None):
    """R > 1: marginalise the pi0 measurement (pi0_candidates) with R replicas of S_side / R draws."""
    B = len(idx)
    st = [{k: torch.as_tensor(v[idx]) for k, v in s.items()} for s in sides]
    for s in st:
        for k, v in s.items():
            if v.dtype == torch.float32:
                s[k] = v.double()
    pi0m = o["pi0"][idx]
    pi0c = pi0_candidates(pi0m, R, rng) if R > 1 else pi0m[:, None]          # (B, R, 2, 2, 4)
    visc = o["vis"][idx][:, None] - pi0m.sum(2)[:, None] + pi0c.sum(3)      # (B, R, 2, 4) actual candidate visibles
    Sr = S_side // R
    keep, side_ess, rep_idx = [], [], []
    for q, s in enumerate(st):
        sr = {k: v.repeat_interleave(R, 0) for k, v in s.items()}
        sr["vis"] = torch.as_tensor(cap_mass(visc[:, :, q]).reshape(B * R, 4))
        tau4, ok, lw = sample_tau(sr, Sr, gen, "cpu", use_ip)
        tau4, ok, lw = tau4.reshape(B, R * Sr, 4), ok.reshape(B, R * Sr), lw.reshape(B, R * Sr)
        lw = torch.where(ok & torch.isfinite(lw), lw, torch.full_like(lw, -1e30))
        w = torch.exp(lw - lw.max(1, keepdim=True).values) * ok
        tot = w.sum(1, keepdim=True)
        valid = tot[:, 0] > 0
        wn = torch.where(valid[:, None], w / torch.clamp(tot, min=1e-300), torch.full_like(w, 1.0 / (R * Sr)))
        side_ess.append(torch.where(valid, ess(w), torch.zeros(B, dtype=w.dtype)))
        j = systematic(wn, M, gen)
        keep.append((torch.gather(tau4, 1, j[..., None].expand(B, M, 4)), valid))
        rep_idx.append((j // Sr).numpy())
    (ta, va), (tb, vb) = keep
    vis0, vis1 = st[0]["vis"], st[1]["vis"]
    # MET compares with tau_T - measured visible_T (a pi0 mismeasurement enters MET with opposite sign)
    nu_t = (ta - vis0[:, None])[:, :, None, :2] + (tb - vis1[:, None])[:, None, :, :2]   # (B, M, M, 2)
    met = torch.as_tensor(o["met"][idx])
    lmet = -0.5 * ((met[:, None, None, :] - nu_t) ** 2).sum(-1) / met_sigma ** 2
    if mass_prior is not None:
        d4 = ta[:, :, None, :] + tb[:, None, :, :]
        mpair = torch.sqrt(torch.clamp(d4[..., 3] ** 2 - (d4[..., :3] ** 2).sum(-1), min=0.0))
        lmet = lmet - 0.5 * ((mpair - mass_prior[0]) / mass_prior[1]) ** 2
    lmet = lmet.reshape(B, M * M)
    wp = torch.exp(lmet - lmet.max(1, keepdim=True).values)
    pair_ess = ess(wp)
    jp = systematic(wp / wp.sum(1, keepdim=True), K, gen)
    ia, ib = (jp // M).numpy(), (jp % M).numpy()
    tA = np.take_along_axis(ta.numpy(), ia[..., None], 1)
    tB = np.take_along_axis(tb.numpy(), ib[..., None], 1)
    ra = np.take_along_axis(rep_idx[0], ia, 1)                    # replica of each kept pair, (B, K)
    rb = np.take_along_axis(rep_idx[1], ib, 1)
    bi = np.arange(B)[:, None]
    vA, vB = visc[bi, ra, 0], visc[bi, rb, 1]                     # (B, K, 4) candidate visibles
    # neutrinos from the sampled taus (capped visible mass), the polarimeter on the candidate visible system
    nu = np.stack([tA - cap_mass(vA), tB - cap_mass(vB)], 2)      # (B, K, 2, 4)
    tau4 = np.stack([vA, vB], 2) + nu
    pi0k = np.stack([pi0c[bi, ra, 0], pi0c[bi, rb, 1]], 2)        # (B, K, 2, 2, 4)
    rep = lambda a: np.repeat(a[:, None], K, 1).reshape((B * K,) + a.shape[1:])
    cls = RMODE_TO_CLS[o["rmode"][idx]]
    h = polarimeter_nrk(rep(o["ch"][idx]), rep(o["ch_q"][idx]), pi0k.reshape(B * K, 2, 2, 4),
                        nu.reshape(B * K, 2, 4), tau4.reshape(B * K, 2, 4), rep(cls))
    h = h.reshape(B, K, 2, 3)
    fin = np.isfinite(h).all(-1)                                   # (B, K, 2)
    hz = np.where(fin[..., None], h, 0.0)
    nfin = fin.sum(1)                                              # (B, 2)
    Eh = hz.sum(1) / np.maximum(nfin, 1)[..., None]
    both = fin.all(-1)
    Ehh = np.einsum("bki,bkj->bij", hz[:, :, 0] * both[..., None], hz[:, :, 1] * both[..., None]) \
        / np.maximum(both.sum(1), 1)[:, None, None]
    has = cls >= 0
    Eh = np.where(has[..., None] & (nfin[..., None] > 0), Eh, np.nan)
    Ehh = np.where((has.all(1) & (both.sum(1) > 0))[:, None, None], Ehh, np.nan)
    d4 = tA + tB
    return dict(Eh=Eh, Ehh=Ehh, Etau=np.stack([tA[..., 3].mean(1), tB[..., 3].mean(1)], 1),
                Etau_sd=np.stack([tA[..., 3].std(1), tB[..., 3].std(1)], 1),
                mpair=np.sqrt(np.maximum(d4[..., 3] ** 2 - (d4[..., :3] ** 2).sum(-1), 0)).mean(1), side_ess=torch.stack(side_ess, 1).numpy(), pair_ess=pair_ess.numpy(),
                valid=(va & vb).numpy(), frac_fin=nfin / K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recodir", default="/home/rbaba/dihiggs-spin-20261006/runs/ent/reco")
    ap.add_argument("--dataset", default="/home/rbaba/dihiggs-spin-20261006/runs/ent/dataset_HU.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1)
    ap.add_argument("--no-ip", action="store_true")
    ap.add_argument("--S-side", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--met-sigma", type=float, default=0.0, help="0: dataset met_sigma")
    ap.add_argument("--mass-prior", default="", help="m,sigma [GeV]: Gaussian prior on m_tautau (signal hypothesis)")
    ap.add_argument("--pi0-replicas", type=int, default=1, help=">1: marginalise the pi0 measurement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = load_reco(args.recodir)
    ds = np.load(args.dataset, allow_pickle=True)
    assert np.array_equal(ds["uid"], d["uid"]), "reco chunks and dataset_HU are not aligned"
    met_sigma = args.met_sigma or float(ds["met_sigma"])
    o = ordered(d)
    sides = build_sides(o)
    stop = len(d["uid"]) if args.stop < 0 else min(args.stop, len(d["uid"]))
    gen = torch.Generator().manual_seed(args.seed * 1_000_003 + args.start)
    rng = np.random.default_rng(args.seed * 1_000_003 + args.start)
    outs = []
    for a in range(args.start, stop, args.batch):
        idx = np.arange(a, min(a + args.batch, stop))
        mp = tuple(float(x) for x in args.mass_prior.split(",")) if args.mass_prior else None
        outs.append(run_batch(o, sides, idx, met_sigma, not args.no_ip, args.S_side, args.M, args.K, gen, mp,
                              args.pi0_replicas, rng))
    res = {k: np.concatenate([r[k] for r in outs]) for k in outs[0]}
    res["uid"] = d["uid"][args.start:stop]
    res["met_sigma"] = np.array(met_sigma)
    np.savez_compressed(args.out, **res)
    print("wrote", args.out, stop - args.start, flush=True)


if __name__ == "__main__":
    main()
