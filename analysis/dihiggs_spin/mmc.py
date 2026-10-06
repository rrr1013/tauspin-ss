"""Missing Mass Calculator as implemented in the tauspin mass-reconstruction run
(NN/mass_reconstruction_20260920/mass_estimators.py, run_mmc), extended to
return the likelihood-weighted ditau four-momentum and per-tau neutrino energy.

Neutrino azimuths are drawn uniformly within +-w of each visible tau, with
w = clip(6 m_tau / pT_vis, 0.04, 0.35) rad (the mass shell confines the
neutrino to a cone of opening ~m_tau/E; a fixed 0.35 rad wastes most samples
on the softer taus of HH events),
the true neutrino transverse sum is drawn from a Gaussian around the measured
MET, the two transverse momenta follow from the 2x2 system and the
longitudinal momenta from the tau mass shell (all four root pairs).  Each
solution is weighted by P(dR(nu, vis) | pT_vis, prong) from simulation.  The
estimator is the weighted mode (MAXW) of m_tautau.
"""
import numpy as np
import torch

from mmc_kinematics import (delta_r, four_mass_t, neutrino_four, phi_of,
                            solve_longitudinal, solve_transverse)

PT_EDGES = np.array([0., 40., 55., 70., 85., 100., 120., 150., 200., 1e9])
DR_MAX, DR_BINS = 1.0, 100


def build_dr_table(vis, nu, prong):
    """P(dR | pT bin, prong) from simulation; vis, nu (N, 4), prong in {1, 3}."""
    ptv = np.hypot(vis[:, 0], vis[:, 1])
    dph = np.angle(np.exp(1j * (np.arctan2(nu[:, 1], nu[:, 0]) - np.arctan2(vis[:, 1], vis[:, 0]))))
    eta_n = np.arcsinh(nu[:, 2] / np.maximum(np.hypot(nu[:, 0], nu[:, 1]), 1e-9))
    eta_v = np.arcsinh(vis[:, 2] / np.maximum(ptv, 1e-9))
    dr = np.hypot(dph, eta_n - eta_v)
    edges = np.linspace(0, DR_MAX, DR_BINS + 1)
    table = np.zeros((2, len(PT_EDGES) - 1, DR_BINS))
    for pi, p in enumerate((1, 3)):
        for bi in range(len(PT_EDGES) - 1):
            sel = (prong == p) & (ptv >= PT_EDGES[bi]) & (ptv < PT_EDGES[bi + 1])
            if sel.sum() < 200:
                sel = prong == p
            h, _ = np.histogram(dr[sel], bins=edges)
            h = np.convolve(h.astype(float), np.ones(3) / 3, mode="same")
            h = np.maximum(h, h.sum() * 1e-6)
            table[pi, bi] = h / (h.sum() * (DR_MAX / DR_BINS))
    return table


def pt_bin_of(vis):
    ptv = np.hypot(vis[..., 0], vis[..., 1])
    return np.clip(np.searchsorted(PT_EDGES, ptv, side="right") - 1, 0, len(PT_EDGES) - 2)


def _dr_logpdf(dr, pt_bin, prong_idx, table_t):
    idx = torch.clamp((dr / DR_MAX * DR_BINS).long(), 0, DR_BINS - 1)
    flat = table_t.reshape(-1, DR_BINS)
    row = prong_idx * (len(PT_EDGES) - 1) + pt_bin
    picked = torch.gather(flat[row.reshape(-1)], 1, idx.reshape(-1, 1))
    return torch.log(picked.reshape(idx.shape))


def run_mmc(vis, met, prong, table, met_sigma, n_samples=1000, batch=1024, seed=0,
            dphi_halfwidth=None, n_hist=1600, lo=0.0, hi=1000.0):
    """vis (N,2,4) numpy, met (N,2), prong (N,2) in {1,3}.  Returns dict of numpy arrays:
    m_maxw, valid, ditau4 (weighted mean, N x 4), e_tau (weighted mean tau energies, N x 2)."""
    gen = torch.Generator().manual_seed(seed)
    table_t = torch.as_tensor(table, dtype=torch.float64)
    N = len(vis)
    res = dict(m_maxw=np.full(N, np.nan), valid=np.zeros(N, bool), ditau4=np.zeros((N, 4)),
               e_tau=np.zeros((N, 2)), nu4=np.zeros((N, 2, 4)))
    pb_all = pt_bin_of(vis)
    pi_all = (np.asarray(prong) == 3).astype(int)
    for a in range(0, N, batch):
        v = torch.as_tensor(vis[a:a + batch], dtype=torch.float64)
        m = torch.as_tensor(met[a:a + batch], dtype=torch.float64)
        pb = torch.as_tensor(pb_all[a:a + batch])
        pr = torch.as_tensor(pi_all[a:a + batch])
        B, S = len(v), n_samples
        if dphi_halfwidth is None:
            ptv = torch.hypot(v[..., 0], v[..., 1])
            hw = torch.clamp(6 * 1.77686 / ptv, 0.04, 0.35).unsqueeze(1)
        else:
            hw = dphi_halfwidth
        dph = (torch.rand(B, S, 2, generator=gen, dtype=torch.float64) * 2 - 1) * hw
        phi_nu = phi_of(v).unsqueeze(1) + dph
        met_s = m.unsqueeze(1) - torch.randn(B, S, 2, generator=gen, dtype=torch.float64) * met_sigma
        pt1, pt2, ok_t = solve_transverse(phi_nu[..., 0], phi_nu[..., 1], met_s[..., 0], met_s[..., 1])
        pts = torch.stack((pt1, pt2), -1)
        vv = v.unsqueeze(1).expand(B, S, 2, 4)
        pz0, pz1, ok0, ok1 = solve_longitudinal(vv, pts, phi_nu)
        masses, logws, oks, d4s, ets, nus = [], [], [], [], [], []
        for r0 in (0, 1):
            for r1 in (0, 1):
                pz = torch.stack((pz0[..., 0] if r0 == 0 else pz1[..., 0],
                                  pz0[..., 1] if r1 == 0 else pz1[..., 1]), -1)
                okr = torch.stack((ok0[..., 0] if r0 == 0 else ok1[..., 0],
                                   ok0[..., 1] if r1 == 0 else ok1[..., 1]), -1)
                nu4 = neutrino_four(pts, phi_nu, pz)
                tau4 = vv + nu4
                d4 = tau4.sum(2)
                mtt = four_mass_t(d4)
                lw = _dr_logpdf(delta_r(nu4, vv), pb.unsqueeze(1).expand(B, S, 2),
                                pr.unsqueeze(1).expand(B, S, 2), table_t).sum(-1)
                good = ok_t & okr.all(-1) & torch.isfinite(mtt) & (mtt > 0) & (mtt < 4000)
                masses.append(mtt); logws.append(lw); oks.append(good); d4s.append(d4); ets.append(tau4[..., 3]); nus.append(nu4)
        mass = torch.cat(masses, 1)
        logw = torch.cat(logws, 1)
        ok = torch.cat(oks, 1)
        d4 = torch.cat(d4s, 1)
        et = torch.cat(ets, 1)
        nus = torch.cat(nus, 1)
        logw = torch.where(ok, logw, torch.full_like(logw, -1e30))
        w = torch.where(ok, torch.exp(logw - logw.max(1, keepdim=True).values), torch.zeros_like(logw))
        tot = w.sum(1)
        valid = tot > 0
        wn = w / torch.clamp(tot, min=1e-30).unsqueeze(1)
        b = torch.clamp(((torch.nan_to_num(mass) - lo) / (hi - lo) * n_hist).long(), 0, n_hist - 1)
        hist = torch.zeros(B, n_hist, dtype=wn.dtype).scatter_add_(1, b, wn)
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
