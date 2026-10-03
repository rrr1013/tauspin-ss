"""Event weights that turn the generator-current cohort into a CLEO-current world.

w = prod_{3pi sides} u_side * f(h_CLEO) / f(h_gen),
u = omega_CLEO / omega_gen * n(m3pi),   n from the flat toy (generator level),
f = 1 + P_tau (h-_k + h+_k) + h-^T C h+  in the canonical h convention,
H: P_tau = 0, C = diag(1,1,-1);  Z (v2): P_tau = -0.147037, C = diag(0,0,1).
pi and rho sides are model independent and use the same h in both factors.
"""
from pathlib import Path

import numpy as np

from currents import dalitz

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
COHORT = HERE.parents[0] / 'cp_mixing' / 'data'
P_TAU_Z = -0.147037
C_H, C_Z = np.array([1., 1., -1.]), np.array([0., 0., 1.])
M_EDGES = np.linspace(0.40, 1.777, 56)


def spin_factor(h, label):
    """f for each event; label 1 = H, 0 = Z."""
    p = h[:, 0] * h[:, 1]
    fH = 1 + p @ C_H
    fZ = 1 + P_TAU_Z * (h[:, 0, 2] + h[:, 1, 2]) + p @ C_Z
    return np.where(label == 1, fH, fZ)


def toy_normalisation(edges=M_EDGES):
    t = np.load(DATA / 'toy.npz')
    te = np.load(DATA / 'toy_eval.npz')
    m = t['m3pi']
    wc, wg = te['out_cleo'][:, 4], te['out_gen'][:, 4]
    b = np.clip(np.digitize(m, edges) - 1, 0, len(edges) - 2)
    num = np.bincount(b, wg, len(edges) - 1)
    den = np.bincount(b, wc, len(edges) - 1)
    cnt = np.bincount(b, minlength=len(edges) - 1)
    n = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, n, cnt


def load(spin_reweight=True, lineshape='generator'):
    """Return dict with cohort arrays and the CLEO-world weight (relative to generator world)."""
    s = np.load(COHORT / 'truth_surface_cleo.npz')
    e = np.load(DATA / 'eval.npz')
    assert np.array_equal(s['global_indices'], e['global_indices'])
    modes, y, w0 = s['modes'], s['labels'].astype(int), s['weights'].astype(float)
    h_gen, h_cleo = e['h_gen'].astype(float), e['h_cleo'].astype(float)
    m3 = np.full(modes.shape, np.nan)
    s13 = np.full(modes.shape, np.nan)
    s23 = np.full(modes.shape, np.nan)
    for side in (0, 1):
        sel = np.flatnonzero(modes[:, side] == 3)
        q2, a, b = dalitz(e[f'ordered_side{side}'])
        assert len(sel) == len(q2)
        m3[sel, side], s13[sel, side], s23[sel, side] = np.sqrt(q2), a, b
    centres, n, _ = toy_normalisation()
    is3 = modes == 3
    u = np.ones(modes.shape)
    ratio = e['omega_cleo'] / e['omega_gen']
    if lineshape == 'generator':
        ok = np.isfinite(n)
        nn = np.interp(m3, centres[ok], n[ok])
        u[is3] = (ratio * nn)[is3]
    elif lineshape == 'cleo':
        # CLEO rate including its a1 lineshape; generator lineshape removed by the toy
        # only up to F_a1 of the generator, which is not available: not implemented.
        raise NotImplementedError
    w = np.prod(u, axis=1)
    if spin_reweight:
        w = w * spin_factor(h_cleo, y) / spin_factor(h_gen, y)
    return dict(modes=modes, y=y, w0=w0, h_gen=h_gen, h_cleo=h_cleo, m3=m3, s13=s13, s23=s23,
                u=u, w=w, ids=s['global_indices'], is3=is3)
