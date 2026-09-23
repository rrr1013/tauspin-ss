"""Generic tau polarimeter from the explicit V-A spinor amplitude, and the
TauDecay-UFO three-pion current that generated the H/Z samples.

For tau- at rest with spin states |b>, M_b = ubar(N) J.gamma (1-g5) u(P,b).
R_ab = sum_nu M_a^* M_b, and |M(chi)|^2 = chi^dag R chi = r0 (1 + h.s), so
h = Tr(R sigma)/Tr(R).  Physical convention: tau- -> pi- nu gives h = +pihat.
tau+ follows from CP: h_tau+({p}) = h_tau-({-p}) with conjugated charges.

The canonical project h (HybridPolarimeter) is -1 times this physical h on both
sides; products h- h+ are identical.  Four-vectors here are (E, x, y, z).
"""
import numpy as np

_I2 = np.eye(2)
_Z2 = np.zeros((2, 2))
SIGMA = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]),
         np.array([[1, 0], [0, -1]], complex)]
G0 = np.block([[_I2, _Z2], [_Z2, -_I2]]).astype(complex)
GAMMA = np.stack([G0] + [np.block([[_Z2, s], [-s, _Z2]]) for s in SIGMA])
G5 = np.block([[_Z2, _I2], [_I2, _Z2]]).astype(complex)
METRIC = np.array([1., -1., -1., -1.])
V_MINUS_A = np.eye(4) - G5


def mdot(a, b):
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def slash(p):
    return np.einsum('...m,mab->...ab', p * METRIC, GAMMA)


def polarimeter(J, N, m_tau=1.777):
    """J (...,4) complex hadronic current, N (...,4) neutrino, both in the tau- rest frame."""
    A = np.einsum('...ab,bc->...ac', slash(J), V_MINUS_A)
    Abar = np.einsum('ab,...cb,cd->...ad', G0, np.conj(A), G0)
    K = Abar @ slash(N.astype(complex)) @ A
    U = np.zeros((4, 2), complex)
    U[0, 0] = U[1, 1] = np.sqrt(2 * m_tau)
    R = np.einsum('ai,ab,...bc,cj->...ij', np.conj(U), G0, K, U)
    r0 = np.real(np.einsum('...ii->...', R)) / 2
    r = np.stack([np.real(np.einsum('...ij,ji->...', R, s)) / 2 for s in SIGMA], -1)
    return r / r0[..., None]


# ---- TauDecay UFO (sm__taudecay_UFO, Fortran/functions.f) -------------------
# J^mu = F_a1(Q^2) T^mu_nu [B(s13)(p1-p3) + B(s23)(p2-p3)]^nu, p1,p2 same-sign.
# F_a1 is a common complex factor and drops out of h.
PIM = 0.13957018
ROM, ROG, ROM1, ROG1, BETA1 = 0.77549, 0.1491, 1.465, 0.40, -0.145


def _klambda(a, b):
    return 1 + a * a + b * b - 2 * (a * b + a + b)


def _bw_rho(s, M, G):
    qs = np.sqrt(np.maximum(_klambda(PIM**2 / s, PIM**2 / s), 0))
    qm = np.sqrt(_klambda(PIM**2 / M**2, PIM**2 / M**2))
    w = np.sqrt(s)
    gs = np.where(s > (2 * PIM)**2, G * (w / M) * (qs / qm)**3, 0.)
    return -M**2 / ((s - M**2) + 1j * w * gs)


def _b_rho0(s):
    return (_bw_rho(s, ROM, ROG) + BETA1 * _bw_rho(s, ROM1, ROG1)) / (1 + BETA1)


def current_taudecay(p1, p2, p3):
    Q = p1 + p2 + p3
    Q2 = mdot(Q, Q)
    out = 0
    for pa in (p1, p2):
        d = pa - p3
        out = out + _b_rho0(mdot(pa + p3, pa + p3))[..., None] * (d - (mdot(Q, d) / Q2)[..., None] * Q)
    return out


# ---- frames, identical to HybridPolarimeter ---------------------------------
def boost(v, beta):
    """Passive boost of (...,px,py,pz,E) into the frame moving with beta."""
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1 / np.sqrt(1 - b2)
    bv = np.sum(v[..., :3] * beta, axis=-1)
    factor = gamma * gamma / (gamma + 1) * bv - gamma * v[..., 3]
    return np.concatenate((v[..., :3] + factor[..., None] * beta,
                           (gamma * (v[..., 3] - bv))[..., None]), axis=-1)


def frames(pions, pi0, nu4):
    """lab (px,py,pz,E) -> pair CM -> each tau rest frame; shared (n,r,k) basis."""
    tau = pions.sum(2) + pi0 + nu4
    pair = tau.sum(1)
    beta = pair[:, :3] / pair[:, 3, None]
    tc = boost(tau, beta[:, None, :])
    k = tc[:, 1, :3] / np.linalg.norm(tc[:, 1, :3], axis=-1)[:, None]
    beam = np.array([0., 0., -1.])
    n = np.cross(beam, k)
    n /= np.linalg.norm(n, axis=-1)[:, None]
    r = beam - np.sum(beam * k, -1)[:, None] * k
    r /= np.linalg.norm(r, axis=-1)[:, None]
    bt = tc[..., :3] / tc[..., 3, None]
    rest = dict(
        pions=boost(boost(pions, beta[:, None, None, :]), bt[:, :, None, :]),
        pi0=boost(boost(pi0, beta[:, None, :]), bt),
        nu=boost(boost(nu4, beta[:, None, :]), bt))
    return dict(beta=beta, bt=bt, basis=np.stack((n, r, k), -2), rest=rest)


def canonical_h(fr, modes, charges, side, mode, current=current_taudecay):
    """h in the project's canonical sign and (n,r,k) basis for one side/mode."""
    sel = np.flatnonzero(modes[:, side] == mode)
    sgn = 1.0 if side == 0 else -1.0          # CP mirror for tau+
    e = lambda v: np.concatenate([v[..., 3:4], sgn * v[..., :3]], -1)
    P = e(fr['rest']['pions'][sel, side])
    Z0 = e(fr['rest']['pi0'][sel, side])
    N = e(fr['rest']['nu'][sel, side])
    if mode == 0:
        J = P[:, 0].astype(complex)
    elif mode == 1:
        J = (P[:, 0] - Z0).astype(complex)
    else:
        ch = -1 if side == 0 else 1
        order = np.argsort(charges[sel, side] != ch, axis=-1, kind='stable')
        Po = np.take_along_axis(P, order[..., None], axis=1)
        J = current(Po[:, 0], Po[:, 1], Po[:, 2])
    h = -np.einsum('nij,nj->ni', fr['basis'][sel], polarimeter(J, N))
    return sel, h
