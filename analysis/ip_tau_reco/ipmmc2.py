"""IP/SV-assisted tau-pair likelihood with the physical decay prior.

Each tau is parametrised by its visible energy fraction x = E_vis / E_tau and the
azimuth phi of the tau direction around the visible axis.  For an unpolarised
tau the visible system is isotropic in the tau rest frame, so x is uniform on
[m_vis^2 / m_tau^2, 1] and phi is uniform: sampling (x, phi) from this prior
satisfies the tau mass shell by construction (cos theta_tau,vis follows from x).
Likelihood factors:
  MET       Gaussian in the two transverse components (always on);
  IP        1-prong: the tau direction must lie in the (track, b) plane:
            Gaussian in the azimuth difference around the track (sigma_b,perp/|b|)
            times the exponential decay-length prior at L = |b| / sin(alpha);
  SV        3-prong: 2D Gaussian in the angle between the tau and PV->SV
            directions times the decay-length prior at the measured length.
use_ip=False gives the same estimator without geometry (MET + prior only).
Estimator: weighted mode of m_tautau and weighted mean tau four-momenta.
"""
import numpy as np
import torch

M_TAU = 1.77686
CTAU_MM = 0.08711


def _unit(v):
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-12)


def _basis(a):
    z = torch.zeros_like(a)
    z[..., 2] = 1.0
    u1 = _unit(torch.cross(a, z, dim=-1))
    u2 = torch.cross(a, u1, dim=-1)
    return u1, u2


def sample_tau(side, S, gen, dev, use_ip):
    B = side["vis"].shape[0]
    vis = side["vis"]
    ev = vis[:, 3]
    pv = vis[:, :3]
    pvm = torch.linalg.norm(pv, dim=-1)
    mv2 = torch.clamp(ev ** 2 - pvm ** 2, min=0.0)
    xmin = torch.clamp(mv2 / M_TAU ** 2, max=0.999)
    x = xmin[:, None] + (1 - xmin[:, None]) * torch.rand(B, S, generator=gen, device=dev, dtype=torch.float64)
    x = torch.clamp(x, min=1e-4)
    E = ev[:, None] / x
    P = torch.sqrt(torch.clamp(E ** 2 - M_TAU ** 2, min=1e-12))
    cth = (2 * E * ev[:, None] - M_TAU ** 2 - mv2[:, None]) / (2 * P * pvm[:, None])
    ok = (cth <= 1.0) & (cth >= -1.0)
    cth = torch.clamp(cth, -1.0, 1.0)
    sth = torch.sqrt(1 - cth ** 2)
    phi = (torch.rand(B, S, generator=gen, device=dev, dtype=torch.float64) * 2 - 1) * np.pi
    vhat = _unit(pv)
    u1, u2 = _basis(vhat)
    t = cth[..., None] * vhat[:, None, :] + sth[..., None] * (torch.cos(phi)[..., None] * u1[:, None, :]
                                                              + torch.sin(phi)[..., None] * u2[:, None, :])
    tau4 = torch.cat([P[..., None] * t, E[..., None]], -1)
    lw = torch.zeros(B, S, device=dev, dtype=torch.float64)
    if use_ip:
        lam = P / M_TAU * CTAU_MM
        # 1-prong: impact-parameter plane
        p = side["trk"][:, None, :]
        cosa = torch.clamp((t * p).sum(-1), -1.0, 1.0)
        sina = torch.sqrt(torch.clamp(1 - cosa ** 2, min=1e-18))
        tperp = _unit(t - cosa[..., None] * p)
        e1 = side["e1"][:, None, :]
        e2 = side["e2"][:, None, :]
        dpsi = torch.atan2((tperp * e2).sum(-1), (tperp * e1).sum(-1))
        sig = side["sig_psi"][:, None]
        L1 = side["bmag"][:, None] * torch.clamp(torch.cos(dpsi), min=1e-6) / sina
        lw1 = -0.5 * (dpsi / sig) ** 2 - L1 / lam - torch.log(lam) - torch.log(sina)
        lw1 = torch.where(sig > 1.5, torch.zeros_like(lw1), lw1)     # |b| below resolution: no information
        # 3-prong: SV direction
        d = side["sv_dir"][:, None, :]
        ang = torch.arccos(torch.clamp((t * d).sum(-1), -1.0, 1.0))
        s3 = side["sig_sv"][:, None]
        lw3 = -0.5 * (ang / s3) ** 2 - side["sv_len"][:, None] / lam - torch.log(lam)
        lw = torch.where(side["three"][:, None], lw3, lw1)
    return tau4, ok, lw


def run(sides, met, met_sigma, use_ip=True, S=4000, batch=256, seed=0, dev="cpu",
        n_hist=1200, lo=0.0, hi=1200.0):
    N = met.shape[0]
    out = dict(m_maxw=np.full(N, np.nan), m_mean=np.full(N, np.nan), valid=np.zeros(N, bool),
               ditau4=np.zeros((N, 4)), tau4=np.zeros((N, 2, 4)), ess=np.zeros(N))
    gen = torch.Generator(device=dev).manual_seed(seed)
    for a in range(0, N, batch):
        sl = slice(a, min(a + batch, N))
        sd = [{k: v[sl] for k, v in s.items()} for s in sides]
        B = sd[0]["vis"].shape[0]
        ta, oka, lwa = sample_tau(sd[0], S, gen, dev, use_ip)
        tb, okb, lwb = sample_tau(sd[1], S, gen, dev, use_ip)
        nu = (ta - sd[0]["vis"][:, None, :])[..., :2] + (tb - sd[1]["vis"][:, None, :])[..., :2]
        lmet = -0.5 * ((met[sl][:, None, :] - nu) ** 2).sum(-1) / met_sigma ** 2
        lw = lwa + lwb + lmet
        ok = oka & okb & torch.isfinite(lw)
        d4 = ta + tb
        m = torch.sqrt(torch.clamp(d4[..., 3] ** 2 - (d4[..., :3] ** 2).sum(-1), min=0.0))
        lw = torch.where(ok, lw, torch.full_like(lw, -1e30))
        w = torch.where(ok, torch.exp(lw - lw.max(1, keepdim=True).values), torch.zeros_like(lw))
        tot = w.sum(1)
        valid = tot > 0
        wn = w / torch.clamp(tot, min=1e-30)[:, None]
        bins = torch.clamp(((m - lo) / (hi - lo) * n_hist).long(), 0, n_hist - 1)
        h = torch.zeros(B, n_hist, device=dev, dtype=torch.float64).scatter_add_(1, bins, wn)
        k = torch.tensor([0.25, 0.5, 0.25], device=dev, dtype=torch.float64).view(1, 1, 3)
        h = torch.nn.functional.conv1d(h[:, None, :], k, padding=1)[:, 0]
        centre = lo + (torch.arange(n_hist, device=dev, dtype=torch.float64) + 0.5) * (hi - lo) / n_hist
        mode = centre[h.argmax(1)]
        out["m_maxw"][sl] = torch.where(valid, mode, torch.full_like(mode, float("nan"))).cpu().numpy()
        out["m_mean"][sl] = (wn * m).sum(1).cpu().numpy()
        out["valid"][sl] = valid.cpu().numpy()
        out["ess"][sl] = (tot ** 2 / torch.clamp((w ** 2).sum(1), min=1e-30)).cpu().numpy()
        out["ditau4"][sl] = (wn[..., None] * d4).sum(1).cpu().numpy()
        out["tau4"][sl] = (wn[..., None, None] * torch.stack([ta, tb], -2)).sum(1).cpu().numpy()
    return out
