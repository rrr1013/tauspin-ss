"""Missing Mass Calculator for tau_lep tau_had (extension of mmc.py).

The leptonic side has two neutrinos: their invariant mass m_nunu is sampled
uniformly in [0, m_tau - m_lep] as in the original MMC (Elagin et al., NIM A 654
(2011) 481), and the mass shell becomes
    m_tau^2 = m_vis^2 + m_nunu^2 + 2 (E_vis E_nu - p_vis . p_nu),
i.e. the quadratic of mmc_kinematics.solve_longitudinal with pT_nu^2 -> pT_nu^2 + m_nunu^2
and an extra -m_nunu^2/2 in the linear term.  The dR(nu, vis) likelihood has a third
row for leptonic decays (kind 0).  Everything else (azimuth window, MET sampling,
four root pairs, weighted mode) follows mmc.run_mmc.
"""
import numpy as np
import torch

from mmc import PT_EDGES, DR_MAX, DR_BINS, pt_bin_of
from mmc_kinematics import delta_r, four_mass_t, phi_of, solve_transverse

TAU_MASS = 1.77686
KINDS = (0, 1, 3)            # leptonic, 1-prong, 3-prong


def build_dr_table(vis, nu, kind):
    ptv = np.hypot(vis[:, 0], vis[:, 1])
    dph = np.angle(np.exp(1j * (np.arctan2(nu[:, 1], nu[:, 0]) - np.arctan2(vis[:, 1], vis[:, 0]))))
    eta_n = np.arcsinh(nu[:, 2] / np.maximum(np.hypot(nu[:, 0], nu[:, 1]), 1e-9))
    eta_v = np.arcsinh(vis[:, 2] / np.maximum(ptv, 1e-9))
    dr = np.hypot(dph, eta_n - eta_v)
    edges = np.linspace(0, DR_MAX, DR_BINS + 1)
    table = np.zeros((len(KINDS), len(PT_EDGES) - 1, DR_BINS))
    for ki, k in enumerate(KINDS):
        for bi in range(len(PT_EDGES) - 1):
            sel = (kind == k) & (ptv >= PT_EDGES[bi]) & (ptv < PT_EDGES[bi + 1])
            if sel.sum() < 200:
                sel = kind == k
            h, _ = np.histogram(dr[sel], bins=edges)
            h = np.convolve(h.astype(float), np.ones(3) / 3, mode="same")
            h = np.maximum(h, h.sum() * 1e-6)
            table[ki, bi] = h / (h.sum() * (DR_MAX / DR_BINS))
    return table


def _logpdf(dr, pt_bin, kind_idx, table_t):
    idx = torch.clamp((dr / DR_MAX * DR_BINS).long(), 0, DR_BINS - 1)
    flat = table_t.reshape(-1, DR_BINS)
    row = kind_idx * (len(PT_EDGES) - 1) + pt_bin
    return torch.log(torch.gather(flat[row.reshape(-1)], 1, idx.reshape(-1, 1)).reshape(idx.shape))


def _longitudinal(vis, pt_nu, phi_nu, m_nn):
    px_v, py_v, pz_v, e_v = vis[..., 0], vis[..., 1], vis[..., 2], vis[..., 3]
    m_vis2 = torch.clamp(e_v ** 2 - (px_v ** 2 + py_v ** 2 + pz_v ** 2), min=0.0)
    nx, ny = torch.cos(phi_nu) * pt_nu, torch.sin(phi_nu) * pt_nu
    b = 0.5 * (TAU_MASS ** 2 - m_vis2 - m_nn ** 2) + px_v * nx + py_v * ny
    a2 = e_v ** 2 - pz_v ** 2
    disc = (b * pz_v) ** 2 - a2 * (e_v ** 2 * (pt_nu ** 2 + m_nn ** 2) - b ** 2)
    real = disc >= 0
    root = torch.sqrt(torch.clamp(disc, min=0.0))
    a2s = torch.where(a2.abs() > 1e-9, a2, torch.ones_like(a2))
    pz0, pz1 = (b * pz_v + root) / a2s, (b * pz_v - root) / a2s
    return pz0, pz1, real & ((b + pz_v * pz0) > 0), real & ((b + pz_v * pz1) > 0)


def run_mmc_lh(vis, met, kind, table, met_sigma, n_samples=1000, batch=1024, seed=0, n_hist=1600, lo=0.0, hi=1000.0):
    """vis (N,2,4), met (N,2), kind (N,2) in {0 lep, 1, 3}.  Returns m_maxw, valid, ditau4, e_tau."""
    gen = torch.Generator().manual_seed(seed)
    table_t = torch.as_tensor(table, dtype=torch.float64)
    N = len(vis)
    res = dict(m_maxw=np.full(N, np.nan), valid=np.zeros(N, bool), ditau4=np.zeros((N, 4)), e_tau=np.zeros((N, 2)),
               nu4=np.zeros((N, 2, 4)))
    pb_all = pt_bin_of(vis)
    ki_all = np.searchsorted(np.array(KINDS), np.asarray(kind))
    for a in range(0, N, batch):
        v = torch.as_tensor(vis[a:a + batch], dtype=torch.float64)
        m = torch.as_tensor(met[a:a + batch], dtype=torch.float64)
        pb = torch.as_tensor(pb_all[a:a + batch])
        ki = torch.as_tensor(ki_all[a:a + batch])
        B, S = len(v), n_samples
        ptv = torch.hypot(v[..., 0], v[..., 1])
        hw = torch.clamp(6 * TAU_MASS / ptv, 0.04, 0.35).unsqueeze(1)
        dph = (torch.rand(B, S, 2, generator=gen, dtype=torch.float64) * 2 - 1) * hw
        phi_nu = phi_of(v).unsqueeze(1) + dph
        met_s = m.unsqueeze(1) - torch.randn(B, S, 2, generator=gen, dtype=torch.float64) * met_sigma
        pt1, pt2, ok_t = solve_transverse(phi_nu[..., 0], phi_nu[..., 1], met_s[..., 0], met_s[..., 1])
        pts = torch.stack((pt1, pt2), -1)
        lep = (ki == 0).unsqueeze(1).expand(B, S, 2)
        m_nn = torch.where(lep, torch.rand(B, S, 2, generator=gen, dtype=torch.float64) * TAU_MASS,
                           torch.zeros(B, S, 2, dtype=torch.float64))
        vv = v.unsqueeze(1).expand(B, S, 2, 4)
        pz0, pz1, ok0, ok1 = _longitudinal(vv, pts, phi_nu, m_nn)
        masses, logws, oks, d4s, ets, nus = [], [], [], [], [], []
        for r0 in (0, 1):
            for r1 in (0, 1):
                pz = torch.stack((pz0[..., 0] if r0 == 0 else pz1[..., 0], pz0[..., 1] if r1 == 0 else pz1[..., 1]), -1)
                okr = torch.stack((ok0[..., 0] if r0 == 0 else ok1[..., 0], ok0[..., 1] if r1 == 0 else ok1[..., 1]), -1)
                px, py = pts * torch.cos(phi_nu), pts * torch.sin(phi_nu)
                e = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + m_nn ** 2)
                nu4 = torch.stack((px, py, pz, e), -1)
                tau4 = vv + nu4
                d4 = tau4.sum(2)
                mtt = four_mass_t(d4)
                lw = _logpdf(delta_r(nu4, vv), pb.unsqueeze(1).expand(B, S, 2), ki.unsqueeze(1).expand(B, S, 2), table_t).sum(-1)
                good = ok_t & okr.all(-1) & torch.isfinite(mtt) & (mtt > 0) & (mtt < 4000)
                masses.append(mtt); logws.append(lw); oks.append(good); d4s.append(d4); ets.append(tau4[..., 3]); nus.append(nu4)
        mass, logw, ok = torch.cat(masses, 1), torch.cat(logws, 1), torch.cat(oks, 1)
        d4, et, nus = torch.cat(d4s, 1), torch.cat(ets, 1), torch.cat(nus, 1)
        logw = torch.where(ok, logw, torch.full_like(logw, -1e30))
        w = torch.where(ok, torch.exp(logw - logw.max(1, keepdim=True).values), torch.zeros_like(logw))
        tot = w.sum(1)
        valid = tot > 0
        wn = w / torch.clamp(tot, min=1e-30).unsqueeze(1)
        bidx = torch.clamp(((torch.nan_to_num(mass) - lo) / (hi - lo) * n_hist).long(), 0, n_hist - 1)
        hist = torch.zeros(B, n_hist, dtype=wn.dtype).scatter_add_(1, bidx, wn)
        kern = torch.tensor([0.25, 0.5, 0.25], dtype=hist.dtype)
        hist = torch.nn.functional.conv1d(hist.unsqueeze(1), kern.view(1, 1, 3), padding=1).squeeze(1)
        centre = lo + (torch.arange(n_hist, dtype=hist.dtype) + 0.5) * (hi - lo) / n_hist
        mode = centre[hist.argmax(1)]
        sl = slice(a, a + B)
        res["m_maxw"][sl] = torch.where(valid, mode, torch.full_like(mode, float("nan"))).numpy()
        res["valid"][sl] = valid.numpy()
        res["ditau4"][sl] = (wn.unsqueeze(-1) * torch.nan_to_num(d4)).sum(1).numpy()
        res["e_tau"][sl] = (wn.unsqueeze(-1) * torch.nan_to_num(et)).sum(1).numpy()
        res["nu4"][sl] = (wn[..., None, None] * torch.nan_to_num(nus)).sum(1).numpy()
    return res
