"""Exact tau polarimeter for Pythia 8 tau decays, in the project's canonical
sign and shared (n, r, k) ditau basis.

The generic V-A polarimeter, the frames and the sign convention are those of
analysis/mode_pair_auc_origin/polarimeter.py (canonical h = -1 x physical h on
both sides, side 0 = tau-, side 1 = tau+, k = tau+ direction in the pair frame).
The hadronic currents are those Pythia 8 uses when it decays the taus:
  pi nu      : J = p_pi
  rho nu     : J = (p_pi - p_pi0)_T   (common Breit-Wigner factor drops out of h)
  3pi nu     : CLEO fit as implemented in Pythia 8 HMETau2ThreePions
               (src/HelicityMatrixElements.cc), modes pi- pi- pi+ and pi0 pi0 pi-.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mode_pair_auc_origin"))
from polarimeter import frames, mdot, polarimeter  # noqa: E402

MPI, MPI0 = 0.13957, 0.1349768

# CLEO / Herwig++ constants of HMETau2ThreePions::initResonances
RHO_M = np.array([.7743, 1.370, 1.720])
RHO_G = np.array([.1491, .386, .250])
RHO_WP = np.array([1, 0.12, 0]) * np.exp(1j * np.array([0, 3.11018, 0]))
RHO_WD = np.array([0.37, 0.87, 0]) * np.exp(1j * np.array([-0.471239, 1.66504, 0]))
F0M, F2M, SIGM = 1.186, 1.275, 0.860
F0G, F2G, SIGG = 0.350, 0.185, 0.880
F0W = 0.77 * np.exp(1j * -1.69646)
F2W = 0.71 * np.exp(1j * 1.75929)
SIGW = 2.1 * np.exp(1j * 0.722566)


def _sqrtpos(x):
    return np.sqrt(np.maximum(x, 0.0))


def _bw(m0, m1, s, M, G, power):
    gs = _sqrtpos((s - (m0 + m1) ** 2) * (s - (m0 - m1) ** 2)) / (2 * _sqrtpos(s))
    gM = _sqrtpos((M * M - (m0 + m1) ** 2) * (M * M - (m0 - m1) ** 2)) / (2 * M)
    return M * M / (M * M - s - 1j * G * M * M / _sqrtpos(s) * (gs / gM) ** power)


def sBW(m0, m1, s, M, G):
    return _bw(m0, m1, s, M, G, 1)


def pBW(m0, m1, s, M, G):
    return _bw(m0, m1, s, M, G, 3)


def dBW(m0, m1, s, M, G):
    return _bw(m0, m1, s, M, G, 5)


def _a1_width(s):
    picM, pinM, kM, ksM = 0.1753, 0.1676, 0.496, 0.894
    piW, kW = 0.2384 ** 2 / 1.0252088, 4.7621 ** 2
    picG = np.where(s < picM, 0.0, np.where(
        s < 0.823, 5.80900 * (s - picM) ** 3 * (1 - 3.00980 * (s - picM) + 4.5792 * (s - picM) ** 2),
        -13.91400 + 27.67900 * s - 13.39300 * s ** 2 + 3.19240 * s ** 3 - 0.10487 * s ** 4))
    pinG = np.where(s < pinM, 0.0, np.where(
        s < 0.823, 6.28450 * (s - pinM) ** 3 * (1 - 2.95950 * (s - pinM) + 4.33550 * (s - pinM) ** 2),
        -15.41100 + 32.08800 * s - 17.66600 * s ** 2 + 4.93550 * s ** 3 - 0.37498 * s ** 4))
    kG = np.where(s > (ksM + kM) ** 2,
                  0.5 * _sqrtpos((s - (ksM + kM) ** 2) * (s - (ksM - kM) ** 2)) / s, 0.0)
    return piW * (picG + pinG + kW * kG)


def _a1_bw(s):
    a1M = 1.331
    return a1M * a1M / (a1M * a1M - s - 1j * _a1_width(s))


def current_cleo(q2, q3, q4, neutral):
    """Pythia 8 three-pion current.  q2, q3 the same-type pions (pi- pi- or
    pi0 pi0), q4 the odd one (pi+ or pi-); four-vectors (E, x, y, z)."""
    m2, m3, m4 = (MPI0, MPI0, MPI) if neutral else (MPI, MPI, MPI)
    q = q2 + q3 + q4
    s1, s2, s3, s4 = mdot(q, q), mdot(q3 + q4, q3 + q4), mdot(q2 + q4, q2 + q4), mdot(q2 + q3, q2 + q3)
    a1 = _a1_bw(s1)
    if not neutral:
        F1 = sum(-RHO_WP[i] * pBW(m3, m4, s2, RHO_M[i], RHO_G[i])
                 - RHO_WD[i] / 3 * pBW(m2, m4, s3, RHO_M[i], RHO_G[i]) * (s2 - s4) for i in range(3))
        F1 = F1 - 2 / 3 * (SIGW * sBW(m2, m4, s3, SIGM, SIGG) + F0W * sBW(m2, m4, s3, F0M, F0G))
        F1 = F1 + F2W * (0.5 * (s4 - s3) * dBW(m3, m4, s2, F2M, F2G)
                         - 1 / (18 * s3) * (4 * m2 ** 2 - s3) * (s1 + s3 - m2 ** 2) * dBW(m2, m4, s3, F2M, F2G))
        F2 = sum(-RHO_WP[i] * pBW(m2, m4, s3, RHO_M[i], RHO_G[i])
                 - RHO_WD[i] / 3 * pBW(m3, m4, s2, RHO_M[i], RHO_G[i]) * (s3 - s4) for i in range(3))
        F2 = F2 - 2 / 3 * (SIGW * sBW(m3, m4, s2, SIGM, SIGG) + F0W * sBW(m3, m4, s2, F0M, F0G))
        F2 = F2 + F2W * (0.5 * (s4 - s2) * dBW(m2, m4, s3, F2M, F2G)
                         - 1 / (18 * s2) * (4 * m2 ** 2 - s2) * (s1 + s2 - m2 ** 2) * dBW(m3, m4, s2, F2M, F2G))
        F3 = sum(-RHO_WD[i] * (1 / 3 * (s3 - s4) * pBW(m3, m4, s2, RHO_M[i], RHO_G[i])
                               - 1 / 3 * (s2 - s4) * pBW(m2, m4, s3, RHO_M[i], RHO_G[i])) for i in range(3))
        F3 = F3 - 2 / 3 * (SIGW * sBW(m3, m4, s2, SIGM, SIGG) + F0W * sBW(m3, m4, s2, F0M, F0G))
        F3 = F3 + 2 / 3 * (SIGW * sBW(m2, m4, s3, SIGM, SIGG) + F0W * sBW(m2, m4, s3, F0M, F0G))
        F3 = F3 + F2W * (-1 / (18 * s2) * (4 * m2 ** 2 - s2) * (s1 + s2 - m2 ** 2) * dBW(m3, m4, s2, F2M, F2G)
                         + 1 / (18 * s3) * (4 * m2 ** 2 - s3) * (s1 + s3 - m2 ** 2) * dBW(m2, m4, s3, F2M, F2G))
        F2 = -F2
    else:
        F1 = sum(RHO_WP[i] * pBW(m3, m4, s2, RHO_M[i], RHO_G[i])
                 - RHO_WD[i] / 3 * pBW(m2, m4, s3, RHO_M[i], RHO_G[i]) * (s4 - s2 - m4 ** 2 + m2 ** 2)
                 for i in range(3))
        F1 = F1 + 2 / 3 * (SIGW * sBW(m2, m3, s4, SIGM, SIGG) + F0W * sBW(m2, m3, s4, F0M, F0G))
        F1 = F1 + F2W / (18 * s4) * (s1 - m4 ** 2 + s4) * (4 * m2 ** 2 - s4) * dBW(m2, m3, s4, F2M, F2G)
        F2 = sum(-RHO_WP[i] / 3 * pBW(m2, m4, s3, RHO_M[i], RHO_G[i])
                 - RHO_WD[i] * pBW(m3, m4, s2, RHO_M[i], RHO_G[i]) * (s4 - s3 - m4 ** 2 + m3 ** 2)
                 for i in range(3))
        F2 = F2 + 2 / 3 * (SIGW * sBW(m2, m3, s4, SIGM, SIGG) + F0W * sBW(m2, m3, s4, F0M, F0G))
        F2 = F2 + F2W / (18 * s4) * (s1 - m4 ** 2 + s4) * (4 * m2 ** 2 - s4) * dBW(m2, m3, s4, F2M, F2G)
        F2 = -F2
        F3 = sum(RHO_WD[i] * (-1 / 3 * (s4 - s3 - m4 ** 2 + m3 ** 2) * pBW(m3, m4, s2, RHO_M[i], RHO_G[i])
                              + 1 / 3 * (s4 - s2 - m4 ** 2 + m2 ** 2) * pBW(m2, m4, s3, RHO_M[i], RHO_G[i]))
                 for i in range(3))
        F3 = F3 - F2W / 2 * (s2 - s3) * dBW(m2, m3, s4, F2M, F2G)
    F1, F2, F3 = a1 * F1, a1 * F2, a1 * F3
    u = (F3 - F2)[..., None] * q2 + (F1 - F3)[..., None] * q3 + (F2 - F1)[..., None] * q4
    u = u - (mdot(u, q) / s1)[..., None] * q
    return u


def classify(nch, npi0, nlep, nother):
    """Decay class: 0 pi, 1 rho, 2 pi 2pi0, 3 three charged pions, -1 other/leptonic."""
    had = (nlep == 0) & (nother == 0)
    cls = np.full(nch.shape, -1, np.int8)
    cls[had & (nch == 1) & (npi0 == 0)] = 0
    cls[had & (nch == 1) & (npi0 == 1)] = 1
    cls[had & (nch == 1) & (npi0 == 2)] = 2
    cls[had & (nch == 3) & (npi0 == 0)] = 3
    return cls


def polarimeter_nrk(ch, chq, pi0, nu, tau4, cls):
    """Canonical h in the shared (n, r, k) basis of the given tau pair.

    All arrays are ordered side 0 = tau-, side 1 = tau+: ch (n,2,3,4), chq (n,2,3),
    pi0 (n,2,2,4), nu (n,2,4), tau4 (n,2,4) lab (px,py,pz,E); cls (n,2) decay class.
    Returns h (n,2,3), NaN where cls < 0.
    """
    from polarimeter import boost
    n = len(ch)
    h = np.full((n, 2, 3), np.nan)
    if n == 0:
        return h
    fr = frames(ch, tau4 - ch.sum(2) - nu, nu)
    rest_ch = fr["rest"]["pions"]
    bt, beta = fr["bt"], fr["beta"]
    rest_pi0 = boost(boost(pi0, beta[:, None, None, :]), bt[:, :, None, :])
    for side in (0, 1):
        sgn = 1.0 if side == 0 else -1.0
        e = lambda v: np.concatenate([v[..., 3:4], sgn * v[..., :3]], -1)
        P, Z, N = e(rest_ch[:, side]), e(rest_pi0[:, side]), e(fr["rest"]["nu"][:, side])
        c = cls[:, side]
        J = np.zeros((n, 4), complex)
        m0 = c == 0
        J[m0] = P[m0, 0]
        m1 = c == 1
        q = P[m1, 0] - Z[m1, 0]
        Q = P[m1, 0] + Z[m1, 0]
        J[m1] = q - (mdot(q, Q) / mdot(Q, Q))[:, None] * Q
        m3 = c == 3
        if m3.any():
            tq = -1 if side == 0 else 1
            # pions with the tau's charge first (False sorts before True)
            o = np.argsort(chq[m3, side] != tq, axis=-1, kind="stable")
            Po = np.take_along_axis(P[m3], o[..., None], axis=1)
            J[m3] = current_cleo(Po[:, 0], Po[:, 1], Po[:, 2], neutral=False)
        m2 = c == 2
        if m2.any():
            J[m2] = current_cleo(Z[m2, 0], Z[m2, 1], P[m2, 0], neutral=True)
        ok = (c >= 0) & np.isfinite(N).all(-1) & (np.abs(J).sum(-1) > 0)
        with np.errstate(all="ignore"):
            hp = -np.einsum("nij,nj->ni", fr["basis"][ok], polarimeter(J[ok], N[ok]))
        h[ok, side] = hp
    return h


def exact_h(d):
    """Exact canonical h (n, 2, 3) for a shower file, plus decay class (tau-, tau+ order),
    the side order used and the mask of events with an opposite-charge tau pair."""
    n = len(d["tau_p4"])
    cls = classify(d["tau_nch"], d["tau_npi0"], d["tau_nlep"], d["tau_nother"])
    order = np.where(d["tau_q"][:, :1] < 0, [[0, 1]], [[1, 0]])
    take = lambda a: np.take_along_axis(a, order.reshape((n, 2) + (1,) * (a.ndim - 2)), axis=1)
    cls = take(cls)
    both = d["tau_ok"].all(1) & (d["tau_q"].sum(1) == 0)
    h = np.full((n, 2, 3), np.nan)
    sel = np.flatnonzero(both)
    h[sel] = polarimeter_nrk(take(d["tau_ch_p4"])[sel], take(d["tau_ch_q"])[sel], take(d["tau_pi0_p4"])[sel],
                             take(d["tau_nu_p4"])[sel], take(d["tau_p4"])[sel], cls[sel])
    return h, cls, order, both
