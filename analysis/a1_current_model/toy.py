"""Flat tau -> 3pi nu phase space at the sample's m(3pi) values (tau- at rest).

For each 3pi side of the cohort, K configurations with the same m(3pi), flat in
the Dalitz plot and isotropic in orientation, and the 3pi system recoiling
against the neutrino in a random direction.  Pions are (px,py,pz,E), ordered
pi-, pi-, pi+ as HybridPolarimeter passes them for tau-.
"""
import argparse

import numpy as np

MPI, MTAU = 0.13957018, 1.777


def random_rotation(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def three_body(rng, M):
    """Flat Dalitz (s13, s23) at mass M (array); returns (n,3,4) in the 3pi rest frame."""
    n = len(M)
    out = np.empty((n, 3, 4))
    todo = np.arange(n)
    while len(todo):
        Mi = M[todo]
        lo, hi = (2 * MPI) ** 2, (Mi - MPI) ** 2
        s13 = lo + (hi - lo) * rng.random(len(todo))
        s23 = lo + (hi - lo) * rng.random(len(todo))
        E1 = (Mi**2 + MPI**2 - s23) / (2 * Mi)
        E2 = (Mi**2 + MPI**2 - s13) / (2 * Mi)
        E3 = Mi - E1 - E2
        ok = (E1 > MPI) & (E2 > MPI) & (E3 > MPI)
        p1 = np.sqrt(np.maximum(E1**2 - MPI**2, 0))
        p2 = np.sqrt(np.maximum(E2**2 - MPI**2, 0))
        p3 = np.sqrt(np.maximum(E3**2 - MPI**2, 0))
        c12 = (p3**2 - p1**2 - p2**2) / (2 * p1 * p2 + 1e-300)
        ok &= np.abs(c12) <= 1
        idx, c12, p1, p2, E1, E2, E3 = todo[ok], c12[ok], p1[ok], p2[ok], E1[ok], E2[ok], E3[ok]
        s12 = np.sqrt(1 - c12**2)
        v1 = np.stack([np.zeros_like(p1), np.zeros_like(p1), p1], -1)
        v2 = np.stack([p2 * s12, np.zeros_like(p2), p2 * c12], -1)
        v3 = -(v1 + v2)
        R = random_rotation(rng, len(idx))
        for j, (v, E) in enumerate(((v1, E1), (v2, E2), (v3, E3))):
            out[idx, j, :3] = np.einsum('nij,nj->ni', R, v)
            out[idx, j, 3] = E
        todo = todo[~ok]
    return out


def boost_from_rest(p, beta):
    """Active boost of (...,px,py,pz,E) at rest into a frame where it moves with beta."""
    b2 = np.sum(beta**2, -1)
    g = 1 / np.sqrt(1 - b2)
    bp = np.sum(p[..., :3] * beta, -1)
    f = (g - 1) / np.where(b2 > 0, b2, 1) * bp + g * p[..., 3]
    return np.concatenate([p[..., :3] + f[..., None] * beta, (g * (p[..., 3] + bp))[..., None]], -1)


def generate(rng, M):
    pions = three_body(rng, M)
    n = len(M)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    EQ = (MTAU**2 + M**2) / (2 * MTAU)
    q = np.sqrt(np.maximum(EQ**2 - M**2, 0))
    beta = (q / EQ)[:, None] * u
    return boost_from_rest(pions, beta[:, None, :])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    e = np.load(a.eval)
    from currents import dalitz
    Q2 = np.concatenate([dalitz(e[f'ordered_side{s}'])[0] for s in (0, 1)])
    M = np.repeat(np.sqrt(Q2), a.k)
    rng = np.random.default_rng(20261004)
    p = generate(rng, M)
    tau = p.sum(1)
    tau[:, 3] += 0  # 3pi only; neutrino is P - sum
    np.savez_compressed(a.output, pions=p, m3pi=M, parent=np.repeat(np.arange(len(Q2)), a.k))
    print(len(M), 'toy decays')
