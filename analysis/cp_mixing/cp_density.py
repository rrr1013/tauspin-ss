"""Production spin density matrix for Phi -> tau- tau+ with a CP-mixed Yukawa.

    L = -(y/sqrt2) Phi  tau_bar (cos(phi) + i gamma5 sin(phi)) tau

The polarised squared amplitude is evaluated with explicit spin projectors,

    T(s-, s+) = Tr[ (p-_slash + m)(1 + g5 s-_slash)/2 . Gamma
                    . (p+_slash - m)(1 + g5 s+_slash)/2 . Gamma_bar ],

which equals  N (1 + B- . s- + B+ . s+ + s-^T C s+).  B and C are extracted by
finite differences over the six axis directions, so no spinor phase convention
enters.  Everything is exact in beta = sqrt(1 - 4 m_tau^2 / m_Phi^2).

Basis: identical to analysis/mode_pair_auc_origin/polarimeter.frames --
k_hat along the tau+ momentum in the pair rest frame, n = beam x k,
r = beam_perp, right handed (n x r = k).  The project's canonical h is -1 times
the physical polarimeter on both sides, so h-^T C h+ is unchanged by that flip
and C below applies directly to the stored canonical h.

Four-vectors here are (E, x, y, z).
"""
import numpy as np

from polarimeter import GAMMA, G5, METRIC, slash  # noqa: F401

M_TAU = 1.777
M_PHI = 91.187


def spin_four_vector(shat, p3, m):
    """Rest-frame unit spin direction -> spin 4-vector for a particle with 3-momentum p3."""
    shat = np.asarray(shat, float)
    p3 = np.asarray(p3, float)
    E = np.sqrt(m * m + p3 @ p3)
    s0 = (p3 @ shat) / m
    sv = shat + (p3 @ shat) / (m * (E + m)) * p3
    return np.concatenate(([s0], sv))


def polarised_T(shat_minus, shat_plus, phi, m_tau=M_TAU, m_phi=M_PHI):
    """T(s-, s+) in the Phi rest frame with k_hat = +z = tau+ direction."""
    E = m_phi / 2.0
    pmag = np.sqrt(max(E * E - m_tau * m_tau, 0.0))
    p_plus3 = np.array([0.0, 0.0, pmag])       # tau+ along +k
    p_minus3 = -p_plus3
    p_minus = np.concatenate(([E], p_minus3))
    p_plus = np.concatenate(([E], p_plus3))
    s_minus = spin_four_vector(shat_minus, p_minus3, m_tau)
    s_plus = spin_four_vector(shat_plus, p_plus3, m_tau)

    G = np.cos(phi) * np.eye(4, dtype=complex) + 1j * np.sin(phi) * G5
    Gbar = G                                    # g0 Gdag g0 = G for (cos + i sin g5)
    Pm = (slash(p_minus) + m_tau * np.eye(4)) @ (np.eye(4) + G5 @ slash(s_minus)) / 2
    Pp = (slash(p_plus) - m_tau * np.eye(4)) @ (np.eye(4) + G5 @ slash(s_plus)) / 2
    return np.real(np.trace(Pm @ G @ Pp @ Gbar))


def density(phi, m_tau=M_TAU, m_phi=M_PHI):
    """(N, B_minus, B_plus, C) with rows/cols ordered (n, r, k)."""
    ax = np.eye(3)
    T = np.empty((3, 2, 3, 2))
    for i in range(3):
        for a, sa in enumerate((+1, -1)):
            for j in range(3):
                for b, sb in enumerate((+1, -1)):
                    T[i, a, j, b] = polarised_T(sa * ax[i], sb * ax[j], phi, m_tau, m_phi)
    N = T.mean()
    Bm = np.array([(T[i, 0].mean() - T[i, 1].mean()) / (2 * N) for i in range(3)])
    Bp = np.array([(T[:, :, j, 0].mean() - T[:, :, j, 1].mean()) / (2 * N) for j in range(3)])
    C = np.array([[(T[i, 0, j, 0] - T[i, 0, j, 1] - T[i, 1, j, 0] + T[i, 1, j, 1]) / (4 * N)
                   for j in range(3)] for i in range(3)])
    return N, Bm, Bp, C


def c_matrix(phi, m_tau=M_TAU, m_phi=M_PHI):
    return density(phi, m_tau, m_phi)[3]


def c_analytic(phi):
    """Ultra-relativistic closed form, for reference."""
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    return np.array([[c, s, 0.], [-s, c, 0.], [0., 0., -1.]])
