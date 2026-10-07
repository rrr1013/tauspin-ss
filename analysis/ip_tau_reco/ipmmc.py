"""IP/SV-assisted tau-pair reconstruction (IP-MMC).

Extends the Missing Mass Calculator of the tauspin mass-reconstruction run by
parametrising each tau by its flight direction instead of the neutrino azimuth:

  1-prong: t = cos(a) p + sin(a) (cos(psi) e1 + sin(psi) e2), with p the
           leading-track direction, e1 the direction of the measured impact
           vector b (component perpendicular to p) and e2 = p x e1.  The decay
           vertex lies on the track, so t lies in the (p, b) plane: psi ~ 0
           within sigma_psi = sigma_b,perp / |b|.  The decay length implied by
           |b| = L sin(a) is weighted by its exponential prior,
           w = exp(-L / lambda) / (lambda sin a), lambda = (|p_tau| / m_tau) c tau.
  3-prong: t is sampled around the PV -> SV direction with the SV angular
           resolution, weighted by the decay-length prior.
Given t, the tau mass shell fixes |p_tau| (two roots); nu = tau - visible.
Every sample is weighted by the MET likelihood (Gaussian), the dR(nu, vis)
prior of the MMC and the solid-angle Jacobian sin(a) of the (a, psi) proposal.
Setting use_ip=False replaces psi by a uniform angle and drops the decay-
length weight, which recovers a direction-parametrised MMC with the same
priors: the difference between the two is the information carried by the
impact parameters.
Estimator: weighted mode (MAXW) of m_tautau and the weighted mean tau
four-momenta (for m_HH).
"""
import numpy as np
import torch

M_TAU = 1.77686
CTAU_MM = 0.08711
PT_EDGES = np.array([0., 40., 55., 70., 85., 100., 120., 150., 200., 1e9])
DR_MAX, DR_BINS = 1.0, 100


def _unit(v):
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-12)


def _dr_logpdf(nu, vis, ptbin, prong3, table):
    def eta(p):
        return torch.asinh(p[..., 2] / torch.clamp(torch.hypot(p[..., 0], p[..., 1]), min=1e-9))
    dphi = torch.remainder(torch.atan2(nu[..., 1], nu[..., 0]) - torch.atan2(vis[..., 1], vis[..., 0]) + np.pi,
                           2 * np.pi) - np.pi
    dr = torch.hypot(dphi, eta(nu) - eta(vis))
    idx = torch.clamp((dr / DR_MAX * DR_BINS).long(), 0, DR_BINS - 1)
    row = prong3 * (len(PT_EDGES) - 1) + ptbin
    flat = table.reshape(-1, DR_BINS)
    return torch.log(flat[row, idx])


def mass_shell_momenta(d, vis):
    """|p_tau| roots for tau direction d (..., 3) and visible four-vector (..., 4)."""
    ev = vis[..., 3]
    pv = vis[..., :3]
    mv2 = torch.clamp(ev ** 2 - (pv ** 2).sum(-1), min=0.0)
    A = 0.5 * (M_TAU ** 2 + mv2)
    c = (d * pv).sum(-1)
    a2 = ev ** 2 - c ** 2
    disc = (A * c) ** 2 - a2 * (ev ** 2 * M_TAU ** 2 - A ** 2)
    ok = disc >= 0
    r = torch.sqrt(torch.clamp(disc, min=0.0))
    a2s = torch.where(a2.abs() > 1e-12, a2, torch.full_like(a2, 1e-12))
    P0, P1 = (A * c + r) / a2s, (A * c - r) / a2s
    ok0 = ok & (P0 > 0) & ((A + P0 * c) > 0)
    ok1 = ok & (P1 > 0) & ((A + P1 * c) > 0)
    return (P0, ok0), (P1, ok1)


def sample_directions(side, S, gen, use_ip, dev):
    """Tau-direction samples (B, S, 3) and log-weights (B, S) for one tau."""
    B = side["vis"].shape[0]
    vis = side["vis"]
    pabs = torch.linalg.norm(vis[:, :3], dim=-1)
    amax = torch.clamp(8 * M_TAU / pabs, 0.02, 0.35)[:, None]
    u = torch.rand(B, S, generator=gen, device=dev, dtype=torch.float64)
    alpha = u * amax
    three = side["three"][:, None]
    # 1-prong frame
    p = side["trk"]
    b = side["b"]
    bperp = b - (b * p).sum(-1, keepdim=True) * p
    bmag = torch.linalg.norm(bperp, dim=-1)
    e1 = _unit(bperp)
    e2 = torch.cross(p, e1, dim=-1)
    sig_psi = torch.clamp(side["sig_bperp"] / torch.clamp(bmag, min=1e-9), max=10.0)
    if use_ip:
        psi = torch.randn(B, S, generator=gen, device=dev, dtype=torch.float64) * sig_psi[:, None]
        psi = torch.where(sig_psi[:, None] > 1.5, (torch.rand(B, S, generator=gen, device=dev,
                                                               dtype=torch.float64) * 2 - 1) * np.pi, psi)
    else:
        psi = (torch.rand(B, S, generator=gen, device=dev, dtype=torch.float64) * 2 - 1) * np.pi
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    t1 = ca[..., None] * p[:, None, :] + sa[..., None] * (torch.cos(psi)[..., None] * e1[:, None, :]
                                                          + torch.sin(psi)[..., None] * e2[:, None, :])
    # 3-prong: around the SV direction (or around the visible axis without geometry)
    dsv = side["sv_dir"]
    q1 = _unit(torch.cross(dsv, torch.tensor([0., 0., 1.], device=dev, dtype=torch.float64).expand_as(dsv), dim=-1))
    q2 = torch.cross(dsv, q1, dim=-1)
    if use_ip:
        sth = side["sig_sv"][:, None]
        g1 = torch.randn(B, S, generator=gen, device=dev, dtype=torch.float64) * sth
        g2 = torch.randn(B, S, generator=gen, device=dev, dtype=torch.float64) * sth
        t3 = _unit(dsv[:, None, :] + g1[..., None] * q1[:, None, :] + g2[..., None] * q2[:, None, :])
        logw3 = torch.zeros(B, S, device=dev, dtype=torch.float64)
    else:
        vdir = _unit(vis[:, :3])
        r1 = _unit(torch.cross(vdir, torch.tensor([0., 0., 1.], device=dev, dtype=torch.float64).expand_as(vdir), dim=-1))
        r2 = torch.cross(vdir, r1, dim=-1)
        phi = (torch.rand(B, S, generator=gen, device=dev, dtype=torch.float64) * 2 - 1) * np.pi
        t3 = _unit(ca[..., None] * vdir[:, None, :] + sa[..., None] * (torch.cos(phi)[..., None] * r1[:, None, :]
                                                                      + torch.sin(phi)[..., None] * r2[:, None, :]))
        logw3 = torch.log(torch.clamp(sa, min=1e-12))
    t = torch.where(three[..., None], t3, t1)
    logw1 = torch.log(torch.clamp(sa, min=1e-12))            # solid-angle Jacobian of the (a, psi) proposal
    logw = torch.where(three, logw3, logw1)
    aux = dict(alpha=alpha, psi=psi, bmag=bmag, sv_len=side["sv_len"])
    return t, logw, aux


def run(sides, met, met_sigma, table, use_ip=True, S=2000, batch=512, seed=0, dev="cpu",
        n_hist=1200, lo=0.0, hi=1200.0):
    """sides: list of two dicts of torch tensors (B, ...): vis (4), trk (3, unit), b (3, mm),
    sig_bperp (mm), three (bool), sv_dir (3, unit), sv_len (mm), sig_sv (rad), ptbin, prong3.
    met: (B, 2).  Returns dict with m_maxw, valid, ditau4, tau4 (B, 2, 4)."""
    table_t = torch.as_tensor(table, device=dev, dtype=torch.float64)
    N = met.shape[0]
    out = dict(m_maxw=np.full(N, np.nan), valid=np.zeros(N, bool), ditau4=np.zeros((N, 4)),
               tau4=np.zeros((N, 2, 4)))
    gen = torch.Generator(device=dev).manual_seed(seed)
    for a in range(0, N, batch):
        sl = slice(a, min(a + batch, N))
        sd = [{k: v[sl] for k, v in s.items()} for s in sides]
        B = sd[0]["vis"].shape[0]
        roots, lws = [], []
        for s in sd:
            t, logw, aux = sample_directions(s, S, gen, use_ip, dev)
            vis = s["vis"][:, None, :].expand(B, S, 4)
            (P0, ok0), (P1, ok1) = mass_shell_momenta(t, vis)
            per = []
            for P, ok in ((P0, ok0), (P1, ok1)):
                ptau = P[..., None] * t
                etau = torch.sqrt(P ** 2 + M_TAU ** 2)
                tau4 = torch.cat([ptau, etau[..., None]], -1)
                nu = tau4 - vis
                lw = logw + _dr_logpdf(nu, vis, s["ptbin"][:, None].expand(B, S), s["prong3"][:, None].expand(B, S), table_t)
                if use_ip:
                    lam = P / M_TAU * CTAU_MM
                    one = ~s["three"][:, None].expand(B, S)
                    L1 = aux["bmag"][:, None] * torch.clamp(torch.cos(aux["psi"]), min=1e-6) / torch.clamp(torch.sin(aux["alpha"]), min=1e-9)
                    lwL1 = -L1 / lam - torch.log(lam) - torch.log(torch.clamp(torch.sin(aux["alpha"]), min=1e-9))
                    lwL3 = -aux["sv_len"][:, None] / lam - torch.log(lam)
                    lw = lw + torch.where(one, lwL1, lwL3)
                per.append((tau4, ok & torch.isfinite(lw), lw))
            roots.append(per)
        masses, lws_all, oks, d4s, t4s = [], [], [], [], []
        for r0 in (0, 1):
            for r1 in (0, 1):
                ta, oka, lwa = roots[0][r0]
                tb, okb, lwb = roots[1][r1]
                nu_t = (ta - sd[0]["vis"][:, None, :])[..., :2] + (tb - sd[1]["vis"][:, None, :])[..., :2]
                dm = met[sl][:, None, :] - nu_t
                lmet = -0.5 * (dm ** 2).sum(-1) / met_sigma ** 2
                d4 = ta + tb
                m = torch.sqrt(torch.clamp(d4[..., 3] ** 2 - (d4[..., :3] ** 2).sum(-1), min=0.0))
                ok = oka & okb & torch.isfinite(m)
                masses.append(m); lws_all.append(lwa + lwb + lmet); oks.append(ok); d4s.append(d4)
                t4s.append(torch.stack([ta, tb], -2))
        m = torch.cat(masses, 1)
        lw = torch.cat(lws_all, 1)
        ok = torch.cat(oks, 1)
        d4 = torch.cat(d4s, 1)
        t4 = torch.cat(t4s, 1)
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
        out["valid"][sl] = valid.cpu().numpy()
        out["ditau4"][sl] = (wn[..., None] * torch.nan_to_num(d4)).sum(1).cpu().numpy()
        out["tau4"][sl] = (wn[..., None, None] * torch.nan_to_num(t4)).sum(1).cpu().numpy()
    return out
