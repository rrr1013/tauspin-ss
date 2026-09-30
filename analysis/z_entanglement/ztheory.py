"""tau- tau+ spin state from a polarised Z (or any spin-1 state) at rest.

For Z -> tau-(p-) tau+(p+) with vertex gamma^mu (v - a gamma5) and a Z spin
density written as a Cartesian 3x3 Hermitian matrix R_ij = <eps_i eps_j*>
(Z rest frame, spatial components, tr R = 1),

    T(s-, s+) = sum_ij R_ij Tr[(p-_slash + m)(1 + g5 s-_slash)/2 G^i
                               (p+_slash - m)(1 + g5 s+_slash)/2 Gbar^j]
              = N (1 + B-.s- + B+.s+ + s-^T C s+).

The trace is evaluated once in the frame where the tau+ moves along +z
(x = n, y = r, z = k); an event's R is rotated into its own (n, r, k) frame.
Everything is exact in the tau mass.  R = 1/3 reproduces the unpolarised Z of
analysis/entanglement/z_density.py.  B, C are for the *physical* spin; the
project's canonical h is -1 times the physical polarimeter on both sides, so C
applies to canonical h directly while B flips sign.

Four-vectors are (E, x, y, z) as in polarimeter.py.
"""
import numpy as np

from polarimeter import G5, GAMMA, slash

M_TAU = 1.77686
M_Z = 91.1876
SIN2W = 0.23152          # overridden by the value read from the LHE banner


def couplings(sin2w=SIN2W):
    return -0.5 + 2 * sin2w, -0.5          # v, a for the tau


def spin_four_vector(shat, p3, m):
    shat = np.asarray(shat, float)
    p3 = np.asarray(p3, float)
    E = np.sqrt(m * m + p3 @ p3)
    return np.concatenate(([(p3 @ shat) / m], shat + (p3 @ shat) / (m * (E + m)) * p3))


def trace_table(mass=M_Z, sin2w=SIN2W, m_tau=M_TAU):
    """t[a, sa, b, sb, i, j] = Tr[...G^i ... Gbar^j] for s- = sa e_a, s+ = sb e_b."""
    v, a = couplings(sin2w)
    E = mass / 2
    pz = np.sqrt(E * E - m_tau * m_tau)
    pm3, pp3 = np.array([0, 0, -pz]), np.array([0, 0, pz])
    pm, pp = np.r_[E, pm3], np.r_[E, pp3]
    I4 = np.eye(4)
    G = [GAMMA[i] @ (v * I4 - a * G5) for i in (1, 2, 3)]
    Gb = [(v * I4 + a * G5) @ GAMMA[i] for i in (1, 2, 3)]    # g0 G^dag g0
    ax = np.eye(3)
    t = np.zeros((3, 2, 3, 2, 3, 3), complex)
    for ia in range(3):
        for sa_i, sa in enumerate((1, -1)):
            Pm = (slash(pm) + m_tau * I4) @ (I4 + G5 @ slash(spin_four_vector(sa * ax[ia], pm3, m_tau))) / 2
            for ib in range(3):
                for sb_i, sb in enumerate((1, -1)):
                    Pp = (slash(pp) - m_tau * I4) @ (I4 + G5 @ slash(spin_four_vector(sb * ax[ib], pp3, m_tau))) / 2
                    for i in range(3):
                        for j in range(3):
                            t[ia, sa_i, ib, sb_i, i, j] = np.trace(Pm @ G[i] @ Pp @ Gb[j])
    return t


class ZDecay:
    """Map from a Z spin density (in the event's n, r, k frame) to (N, B-, B+, C)."""

    def __init__(self, mass=M_Z, sin2w=SIN2W, m_tau=M_TAU):
        t = trace_table(mass, sin2w, m_tau)
        # linear coefficients, each contracted with R_ij later (complex, ij last)
        self.N = t.mean(axis=(0, 1, 2, 3))
        self.Bm = (t[:, 0].mean(axis=(1, 2)) - t[:, 1].mean(axis=(1, 2))) / 2
        self.Bp = (t[:, :, :, 0].mean(axis=(0, 1)) - t[:, :, :, 1].mean(axis=(0, 1))) / 2
        self.C = (t[:, 0, :, 0] - t[:, 0, :, 1] - t[:, 1, :, 0] + t[:, 1, :, 1]) / 4

    def state(self, R):
        """R (..., 3, 3) in (n, r, k) coordinates -> N, B- (...,3), B+ (...,3), C (...,3,3)."""
        N = np.real(np.einsum('...ij,ij->...', R, self.N))
        Bm = np.real(np.einsum('...ij,aij->...a', R, self.Bm)) / N[..., None]
        Bp = np.real(np.einsum('...ij,aij->...a', R, self.Bp)) / N[..., None]
        C = np.real(np.einsum('...ij,abij->...ab', R, self.C)) / N[..., None, None]
        return N, Bm, Bp, C


def rotate(R, O):
    """R given in frame F, O (..., 3, 3) rows = new axes expressed in F -> R in the new frame."""
    return np.einsum('...ai,...ij,...bj->...ab', O, R, O)


def helicity_basis(k, ref):
    """(n, r, k) rows with n = ref x k, r = ref_perp (normalised); matches frames() for ref = beam."""
    n = np.cross(ref, k)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    r = ref - np.sum(ref * k, -1, keepdims=True) * k
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return np.stack((n, r, k), -2)


# ---- Z polarisation from the tau direction distribution --------------------------
def direction_moment_map(zdec, n_grid=200):
    """Linear map from the 9 real parameters of R to (<k>, <k k^T>) of the tau+ direction.

    W(k) = N(R rotated into the k frame); the map is built numerically so that
    the tau mass and couplings enter exactly.  Returns (A, basis) with
    moments = A @ params, where R = sum_p params_p basis_p.
    """
    basis = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3), complex); E[i, j] = E[j, i] = 1; basis.append(E)
    for i in range(3):
        for j in range(i + 1, 3):
            E = np.zeros((3, 3), complex); E[i, j] = 1j; E[j, i] = -1j; basis.append(E)
    basis = np.array(basis)                          # 6 symmetric + 3 antisymmetric
    # Gauss-Legendre in cos(theta), uniform in phi
    x, w = np.polynomial.legendre.leggauss(n_grid)
    ph = (np.arange(2 * n_grid) + 0.5) * np.pi / n_grid
    ct, PH = np.meshgrid(x, ph, indexing='ij')
    st = np.sqrt(1 - ct**2)
    k = np.stack((st * np.cos(PH), st * np.sin(PH), ct), -1).reshape(-1, 3)
    wt = np.repeat(w, 2 * n_grid) * (np.pi / n_grid)
    ref = np.array([0.3, -0.5, 0.81]); ref /= np.linalg.norm(ref)   # generic, irrelevant for N
    O = helicity_basis(k, np.broadcast_to(ref, k.shape))
    feats = np.concatenate((k, np.einsum('ni,nj->nij', k, k).reshape(-1, 9)), 1)
    A = np.zeros((1 + 12, 9))
    for p, E in enumerate(basis):
        N = np.real(np.einsum('nij,ij->n', rotate(E, O), zdec.N))
        A[0, p] = np.sum(wt * N)
        A[1:, p] = np.sum((wt * N)[:, None] * feats, 0)
    return A, basis


def fit_R(mom_k, mom_kk, A, basis):
    """Solve <k>, <kk> (normalised moments) for R with tr R = 1."""
    # unknown params q; constraints: A[0] q = Z (norm), A[1:] q = Z * moments; tr R = 1
    tr = np.array([np.real(np.trace(E)) for E in basis])
    m = np.concatenate((mom_k, np.asarray(mom_kk).reshape(9)))
    # A[1:] q - m (A[0] q) = 0 ; tr q = 1
    M = np.vstack((A[1:] - np.outer(m, A[0]), tr))
    rhs = np.r_[np.zeros(12), 1.0]
    q, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return np.einsum('p,pij->ij', q, basis)


# ---- two-qubit quantities ------------------------------------------------------
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def rho4(Bm, Bp, C):
    I2 = np.eye(2)
    r = np.kron(I2, I2).astype(complex)
    for a in range(3):
        r = r + Bm[a] * np.kron(PAULI[a], I2) + Bp[a] * np.kron(I2, PAULI[a])
        for b in range(3):
            r = r + C[a, b] * np.kron(PAULI[a], PAULI[b])
    return r / 4


def concurrence(r):
    sy = np.kron(PAULI[1], PAULI[1])
    rt = sy @ r.conj() @ sy
    ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(r @ rt)))[::-1], 0, None))
    return max(0.0, ev[0] - ev[1] - ev[2] - ev[3])


def ppt_min_eig(r):
    rt = r.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)   # partial transpose on B
    return float(np.min(np.linalg.eigvalsh(rt)))


def horodecki_m12(C):
    ev = np.sort(np.linalg.eigvalsh(C.T @ C))[::-1]
    return float(ev[0] + ev[1])


def quantum_summary(Bm, Bp, C):
    r = rho4(Bm, Bp, C)
    return dict(concurrence=concurrence(r), ppt_min=ppt_min_eig(r), m12=horodecki_m12(C),
                rho_min=float(np.min(np.linalg.eigvalsh(r))))
