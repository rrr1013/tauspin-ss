"""Visible-only and visible+MET polarisation observables, per tau side.

Ntuple four-vectors are lab (px, py, pz, E) -- same convention as
mode_pair_auc_origin/polarimeter.frames.
"""
import numpy as np

E, PX, PY, PZ = 3, 0, 1, 2


def pt(v):
    return np.hypot(v[..., PX], v[..., PY])


def p3(v):
    return v[..., :3]


def mass(v):
    m2 = v[..., E] ** 2 - np.sum(v[..., :3] ** 2, -1)
    return np.sqrt(np.clip(m2, 0, None))


def boost_to_rest(v, ref):
    """Boost v into the rest frame of ref (both (..., 4) with (px,py,pz,E))."""
    m = np.sqrt(np.clip(ref[..., E] ** 2 - np.sum(ref[..., :3] ** 2, -1), 1e-12, None))
    beta = ref[..., :3] / ref[..., E, None]
    b2 = np.sum(beta ** 2, -1)
    gamma = ref[..., E] / m
    bp = np.sum(beta * v[..., :3], -1)
    coef = np.where(b2 > 1e-12, (gamma - 1.0) / np.maximum(b2, 1e-12), 0.0)
    p_new = v[..., :3] + (coef * bp - gamma * v[..., E])[..., None] * beta
    e_new = gamma * (v[..., E] - bp)
    return np.concatenate([p_new, e_new[..., None]], -1)


def upsilon(pions, counts, neutral):
    """(pT_charged - pT_neutral) / (pT_charged + pT_neutral), per side."""
    idx = np.arange(pions.shape[2])[None, None, :]
    use = idx < counts[:, :, None]
    ch = np.sum(np.where(use[..., None], pions, 0.0), axis=2)
    ptc, ptn = pt(ch), pt(neutral)
    return (ptc - ptn) / np.maximum(ptc + ptn, 1e-9), ch


def cos_theta_star(lead, visible):
    """cos of the leading charged pion in the visible rest frame, w.r.t. the
    visible system's lab direction.  No knowledge of E_tau enters."""
    rest = boost_to_rest(lead, visible)
    a = p3(rest)
    b = p3(visible)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return np.sum(a * b, -1) / np.maximum(na * nb, 1e-12)


def leading_charged(pions, counts):
    idx = np.arange(pions.shape[2])[None, None, :]
    use = idx < counts[:, :, None]
    p = np.where(use, pt(pions), -np.inf)
    j = np.argmax(p, axis=2)
    return np.take_along_axis(pions, j[:, :, None, None], axis=2)[:, :, 0]


def collinear_fractions(visible, met):
    """x_i = E_vis,i / E_tau,i with neutrinos parallel to each visible system and
    magnitudes fixed by the transverse MET.  Returns (x, valid)."""
    vx, vy = visible[:, :, PX], visible[:, :, PY]
    det = vx[:, 0] * vy[:, 1] - vx[:, 1] * vy[:, 0]
    ok = np.abs(det) > 1e-6
    a = np.full((len(det), 2), np.nan)
    a[ok, 0] = (met[ok, 0] * vy[ok, 1] - met[ok, 1] * vx[ok, 1]) / det[ok]
    a[ok, 1] = (met[ok, 1] * vx[ok, 0] - met[ok, 0] * vy[ok, 0]) / det[ok]
    valid = ok & np.all(a > 0, axis=1)
    x = np.where(a > 0, 1.0 / (1.0 + a), np.nan)
    return x, valid
