# Copied unchanged from tauspin NN/mass_reconstruction_20260920/kinematics.py (commit 4dac16a).
"""Shared kinematics for the tau/ditau mass reconstruction comparison.

Conventions
-----------
All four-vectors are (px, py, pz, E) in GeV, lab frame.
The visible tau system is the *massive* reconstructed visible four-vector
(`reco_visible_tau_lab4`).  The companion array `reco_tau_lab4` in the same
file is the massless projection of the same direction and must not be used for
mass construction; see `audit_visible_definitions`.
"""
from __future__ import annotations

import numpy as np
import torch

TAU_MASS = 1.77686


def four_mass_np(f):
    return np.sqrt(np.maximum(f[..., 3] ** 2 - np.sum(f[..., :3] ** 2, axis=-1), 0.0))


def four_mass_t(f):
    return torch.sqrt(torch.clamp(f[..., 3] ** 2 - (f[..., :3] ** 2).sum(-1), min=0.0))


def phi_of(p):
    return torch.atan2(p[..., 1], p[..., 0])


def eta_of(p):
    pt = torch.hypot(p[..., 0], p[..., 1])
    return torch.asinh(p[..., 2] / torch.clamp(pt, min=1e-9))


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def solve_transverse(phi1, phi2, met_x, met_y):
    """Split a transverse missing momentum between two neutrinos of given azimuths.

    Returns (pt1, pt2, ok).  `ok` is False where the 2x2 system is singular or a
    solution would require a negative neutrino transverse momentum.
    """
    det = torch.cos(phi1) * torch.sin(phi2) - torch.sin(phi1) * torch.cos(phi2)
    safe = det.abs() > 1e-6
    det_safe = torch.where(safe, det, torch.ones_like(det))
    pt1 = (met_x * torch.sin(phi2) - met_y * torch.cos(phi2)) / det_safe
    pt2 = (met_y * torch.cos(phi1) - met_x * torch.sin(phi1)) / det_safe
    ok = safe & (pt1 > 0) & (pt2 > 0)
    return pt1, pt2, ok


def solve_longitudinal(vis, pt_nu, phi_nu, tau_mass=TAU_MASS):
    """Both mass-shell roots for the neutrino longitudinal momentum.

    m_tau^2 = m_vis^2 + 2 (E_vis E_nu - p_vis . p_nu) with a massless neutrino
    becomes a quadratic in pz_nu.  Returns (pz_root0, pz_root1, ok) where `ok`
    marks a real root that also satisfies the un-squared equation.
    """
    px_v, py_v, pz_v, e_v = vis[..., 0], vis[..., 1], vis[..., 2], vis[..., 3]
    m_vis2 = torch.clamp(e_v ** 2 - (px_v ** 2 + py_v ** 2 + pz_v ** 2), min=0.0)
    nx, ny = torch.cos(phi_nu) * pt_nu, torch.sin(phi_nu) * pt_nu
    b = 0.5 * (tau_mass ** 2 - m_vis2) + px_v * nx + py_v * ny
    a2 = e_v ** 2 - pz_v ** 2
    disc = (b * pz_v) ** 2 - a2 * (e_v ** 2 * pt_nu ** 2 - b ** 2)
    real = disc >= 0
    root = torch.sqrt(torch.clamp(disc, min=0.0))
    a2s = torch.where(a2.abs() > 1e-9, a2, torch.ones_like(a2))
    pz0 = (b * pz_v + root) / a2s
    pz1 = (b * pz_v - root) / a2s
    # un-squared condition: E_vis * E_nu = b + pz_vis * pz_nu  must be positive
    ok0 = real & ((b + pz_v * pz0) > 0)
    ok1 = real & ((b + pz_v * pz1) > 0)
    return pz0, pz1, ok0, ok1


def neutrino_four(pt_nu, phi_nu, pz_nu):
    px = pt_nu * torch.cos(phi_nu)
    py = pt_nu * torch.sin(phi_nu)
    e = torch.sqrt(px * px + py * py + pz_nu * pz_nu)
    return torch.stack((px, py, pz_nu, e), dim=-1)


def delta_r(nu4, vis4):
    dphi = wrap_pi(phi_of(nu4) - phi_of(vis4))
    deta = eta_of(nu4) - eta_of(vis4)
    return torch.hypot(dphi, deta)


def momentum_from_mass_shell(vis, direction, tau_mass=TAU_MASS):
    """Neutrino momentum magnitude fixed by the tau mass shell, given its direction.

    With a massless neutrino of unit direction d,
        m_tau^2 = m_vis^2 + 2 |p_nu| (E_vis - p_vis . d)
    which is linear in |p_nu|: a single positive solution exists whenever
    E_vis > p_vis . d, i.e. for every direction as long as the visible system is
    massive.  Parameterising by direction therefore removes the quadratic
    two-root ambiguity of the (phi, pT) parameterisation: the two roots there
    correspond to two different directions here.
    """
    e_v = vis[..., 3]
    m_vis2 = torch.clamp(e_v ** 2 - (vis[..., :3] ** 2).sum(-1), min=0.0)
    denom = e_v - (vis[..., :3] * direction).sum(-1)
    ok = denom > 1e-6
    p = 0.5 * (tau_mass ** 2 - m_vis2) / torch.where(ok, denom, torch.ones_like(denom))
    return p, ok & (p > 0)


def direction_from_offsets(vis, d_eta, d_phi):
    """Unit direction at (eta_vis + d_eta, phi_vis + d_phi)."""
    pt = torch.hypot(vis[..., 0], vis[..., 1])
    eta = torch.asinh(vis[..., 2] / torch.clamp(pt, min=1e-9)) + d_eta
    phi = torch.atan2(vis[..., 1], vis[..., 0]) + d_phi
    theta = 2.0 * torch.atan(torch.exp(-eta))
    st = torch.sin(theta)
    return torch.stack((st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)), dim=-1)
