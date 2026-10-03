"""Generator (TauDecay UFO, Kuehn-Santamaria) 3pi current with its rate, as a
drop-in for HybridPolarimeter.current.

HybridPolarimeter calls current(ordered, masses, charges) with the three pions in
the tau rest frame (px,py,pz,E), same-sign pions first, and reads columns 0-2 as
the canonical h and column 4 as omega.  The CLEO C++ returns its spin-averaged
rate omega = P.(Pi - g_VA Pi5) there.  Here column 4 is the generator's
spin-averaged |M|^2 = Tr(R)/2 without the a1 form factor F_a1(Q^2), which is a
common factor at fixed m(3pi) (this analysis normalises rates per m(3pi) bin).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
from polarimeter import (G0, SIGMA, V_MINUS_A, current_taudecay, mdot,  # noqa: E402
                         slash)


def rate_and_h(J, N, m_tau=1.777):
    """Return (r0, h_physical) for tau- at rest; same algebra as polarimeter()."""
    A = np.einsum('...ab,bc->...ac', slash(J), V_MINUS_A)
    Abar = np.einsum('ab,...cb,cd->...ad', G0, np.conj(A), G0)
    K = Abar @ slash(N.astype(complex)) @ A
    U = np.zeros((4, 2), complex)
    U[0, 0] = U[1, 1] = np.sqrt(2 * m_tau)
    R = np.einsum('ai,ab,...bc,cj->...ij', np.conj(U), G0, K, U)
    r0 = np.real(np.einsum('...ii->...', R)) / 2
    r = np.stack([np.real(np.einsum('...ij,ji->...', R, s)) / 2 for s in SIGMA], -1)
    return r0, r / r0[..., None]


def generator_current_with_rate(pions, masses, charges, chunk=100000):
    pions = np.asarray(pions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    charges = np.asarray(charges)
    out = np.empty((len(pions), 5))
    for a in range(0, len(pions), chunk):
        p, ch, m = pions[a:a + chunk], charges[a:a + chunk], masses[a:a + chunk]
        sgn = np.where(ch < 0, 1.0, -1.0)[:, None, None]   # CP mirror for tau+
        e = np.concatenate([p[..., 3:4], sgn * p[..., :3]], -1)
        tau = np.zeros((len(p), 4))
        tau[:, 0] = m
        nu = tau - e.sum(1)
        J = current_taudecay(e[:, 0], e[:, 1], e[:, 2])
        r0, h = rate_and_h(J, nu)
        out[a:a + chunk, :3] = -h
        out[a:a + chunk, 3] = 1.0
        out[a:a + chunk, 4] = r0
    return out


def dalitz(ordered):
    """m(3pi)^2 and the two same-sign/opposite-sign pair masses^2 (E,px,py,pz any order of xyz)."""
    p = np.asarray(ordered, dtype=np.float64)
    d = lambda a, b: a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], -1)
    Q = p.sum(1)
    s13 = d(p[:, 0] + p[:, 2], p[:, 0] + p[:, 2])
    s23 = d(p[:, 1] + p[:, 2], p[:, 1] + p[:, 2])
    return d(Q, Q), s13, s23


# ---- generator a1 form factor (sm__taudecay_UFO Fortran/functions.f) ----------
# FFCT3(S) = Fa1(S, mode=1) is used for every 3pi channel, including pi-pi-pi+.
PI0 = 0.1349766
A1M, A1G, FPI = 1.23, 0.42, 0.13041


def _gfun(s):
    pi3, pi1 = 2 * PI0 + 0.13957018, PI0
    x = s - pi3**2
    low = 4.1 / s * x**3 * (1 - 3.3 * x + 5.8 * x**2)
    high = 1.623 + 10.38 / s - 9.32 / s**2 + 0.65 / s**3
    return np.where(s < (0.77549 + pi1)**2, low, high)


def fa1_generator(s):
    s = np.asarray(s, dtype=np.float64)
    w = np.sqrt(s)
    pi3 = 2 * PI0 + 0.13957018
    gs = np.where(s > pi3**2, A1G * (w / A1M) * _gfun(s) / _gfun(A1M**2), 0.0)
    return 4 / 3 / FPI * (-A1M**2 / ((s - A1M**2) + 1j * w * gs))
