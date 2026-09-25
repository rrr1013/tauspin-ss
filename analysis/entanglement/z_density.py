"""Spin density of tau+ tau- from an unpolarised Z, by the polarised Dirac trace.

The v2 Z sample was generated as pp -> Z+j and Z -> tau tau separately, so the
ditau spin correlation is that of an *unpolarised* Z (see the 2026-09-21 v3 run
note).  Averaging over the three Z polarisations leaves k_hat (the tau+
direction in the pair rest frame) as the only preferred direction, so

    B- = -B+ = P_tau k_hat ,   C = diag(a, a, c)

independent of the beam axis, which is why no basis convention enters.

Vertex: gamma^mu (v - a gamma5) with the Standard Model tau couplings
    v = -1/2 + 2 sin^2(theta_W),   a = -1/2     (conventional normalisation),
polarisation sum  sum_lambda eps^mu eps*^nu = -g^{mu nu} + q^mu q^nu / M^2.
"""
import numpy as np

from polarimeter import G5, GAMMA, METRIC, slash  # noqa: F401

M_TAU = 1.777
M_Z = 91.187
SIN2W = 0.23152


def couplings(sin2w=SIN2W):
    return -0.5 + 2 * sin2w, -0.5          # v, a


def polarised_T(shat_minus, shat_plus, m_tau=M_TAU, m_z=M_Z, sin2w=SIN2W):
    """sum over Z polarisations of |M|^2 for Z -> tau-(s-) tau+(s+)."""
    from cp_density import spin_four_vector
    v, a = couplings(sin2w)
    E = m_z / 2.0
    pmag = np.sqrt(max(E * E - m_tau * m_tau, 0.0))
    p_plus3 = np.array([0.0, 0.0, pmag])       # tau+ along +k
    p_minus3 = -p_plus3
    p_minus = np.concatenate(([E], p_minus3))
    p_plus = np.concatenate(([E], p_plus3))
    q = p_minus + p_plus
    s_minus = spin_four_vector(shat_minus, p_minus3, m_tau)
    s_plus = spin_four_vector(shat_plus, p_plus3, m_tau)

    Pm = (slash(p_minus) + m_tau * np.eye(4)) @ (np.eye(4) + G5 @ slash(s_minus)) / 2
    Pp = (slash(p_plus) - m_tau * np.eye(4)) @ (np.eye(4) + G5 @ slash(s_plus)) / 2
    V = [GAMMA[mu] @ (v * np.eye(4) - a * G5) for mu in range(4)]
    Vb = [(v * np.eye(4) + a * G5) @ GAMMA[mu] for mu in range(4)]   # g0 Vdag g0

    tot = 0.0
    for mu in range(4):
        for nu in range(4):
            gmn = METRIC[mu] if mu == nu else 0.0
            psum = -gmn + q[mu] * q[nu] * METRIC[mu] * METRIC[nu] / (m_z * m_z)
            if psum == 0.0:
                continue
            # raise/lower: vertices carry upper indices, contract with psum_{mu nu}
            tr = np.real(np.trace(Pm @ V[mu] @ Pp @ Vb[nu]))
            tot += psum * METRIC[mu] * METRIC[nu] * tr
    return tot


def density(m_tau=M_TAU, m_z=M_Z, sin2w=SIN2W):
    ax = np.eye(3)
    T = np.empty((3, 2, 3, 2))
    for i in range(3):
        for ia, sa in enumerate((+1, -1)):
            for j in range(3):
                for jb, sb in enumerate((+1, -1)):
                    T[i, ia, j, jb] = polarised_T(sa * ax[i], sb * ax[j], m_tau, m_z, sin2w)
    N = T.mean()
    Bm = np.array([(T[i, 0].mean() - T[i, 1].mean()) / (2 * N) for i in range(3)])
    Bp = np.array([(T[:, :, j, 0].mean() - T[:, :, j, 1].mean()) / (2 * N) for j in range(3)])
    C = np.array([[(T[i, 0, j, 0] - T[i, 0, j, 1] - T[i, 1, j, 0] + T[i, 1, j, 1]) / (4 * N)
                   for j in range(3)] for i in range(3)])
    return N, Bm, Bp, C


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cp_mixing'))
    N, Bm, Bp, C = density()
    v, a = couplings()
    print('v, a =', v, a, '  P_tau = -2va/(v^2+a^2) =', -2 * v * a / (v * v + a * a))
    print('B- =', np.round(Bm, 5), '  B+ =', np.round(Bp, 5))
    print('C  =\n', np.array2string(C, precision=5, suppress_small=True))
    print('beta =', np.sqrt(1 - 4 * M_TAU**2 / M_Z**2))
