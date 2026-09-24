"""Shared CP-sensitivity estimators.

Nominal sample: CP-even H -> tau tau (phi_tau = 0), spin density
p_0(h) proportional to f = 1 + h-^T C(0) h+ times the isotropic measure.

Reweighting to phi:      w(phi) = (1 + h-^T C(phi) h+) / f
Score at phi = 0:        S = d/dphi log(1 + h-^T C(phi) h+) |_0 = 2 * Tp / f
                         with Tp = (h- x h+) . k_hat = h-_n h+_r - h-_r h+_n

For any observable T the linear response of its mean is exactly
    d<T>/dphi |_0 = Cov_0(T, S)
so the one-observable sensitivity for N events is
    sigma_N(phi_tau) = sqrt(Var_0(T) / N) / |Cov_0(T, S)|
and the Cramer-Rao bound over all observables is 1 / sqrt(N Var_0(S)).
"""
import numpy as np

from cp_density import c_matrix


def bilinear(hm, hp, C):
    return np.einsum('ni,ij,nj->n', hm, C, hp)


def triple(hm, hp):
    """(h- x h+) . k_hat with k_hat = e_3 of the (n, r, k) basis."""
    return hm[:, 0] * hp[:, 1] - hm[:, 1] * hp[:, 0]


def score(h_exact, C0=None):
    """CP score at phi = 0 from the exact polarimeters."""
    C0 = c_matrix(0.0) if C0 is None else C0
    hm, hp = h_exact[:, 0].astype(float), h_exact[:, 1].astype(float)
    f = 1.0 + bilinear(hm, hp, C0)
    return 2.0 * triple(hm, hp) / f, f


def weights(h_exact, phi, C0=None):
    C0 = c_matrix(0.0) if C0 is None else C0
    hm, hp = h_exact[:, 0].astype(float), h_exact[:, 1].astype(float)
    f = 1.0 + bilinear(hm, hp, C0)
    return (1.0 + bilinear(hm, hp, c_matrix(phi))) / f


def sensitivity(T, S, n_events=None, boot=0, seed=0):
    """sigma(phi_tau) in radians for n_events, from observable T and score S."""
    T = np.asarray(T, float)
    n = len(T) if n_events is None else n_events
    cov = np.cov(np.stack([T, S]))
    resp = cov[0, 1]
    sig = np.sqrt(cov[0, 0] / n) / abs(resp) if resp != 0 else np.inf
    out = {'n': int(len(T)), 'n_events': int(n), 'response': float(resp),
           'sd_T': float(np.sqrt(cov[0, 0])), 'sigma_rad': float(sig),
           'sigma_deg': float(np.rad2deg(sig))}
    if boot:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(boot):
            k = rng.integers(0, len(T), len(T))
            c = np.cov(np.stack([T[k], S[k]]))
            if c[0, 1] != 0:
                vals.append(np.sqrt(c[0, 0] / n) / abs(c[0, 1]))
        out['sigma_deg_se'] = float(np.rad2deg(np.std(vals)))
    return out


def cramer_rao(S, n_events=None):
    n = len(S) if n_events is None else n_events
    v = np.var(S)
    return {'fisher_per_event': float(v), 'sigma_deg': float(np.rad2deg(1 / np.sqrt(n * v)))}


def acoplanarity(h):
    """Delta = angle(h-_perp) - angle(h+_perp) in [0, 2pi); peaks at -2 phi_tau."""
    am = np.arctan2(h[:, 0, 1], h[:, 0, 0])
    ap = np.arctan2(h[:, 1, 1], h[:, 1, 0])
    return np.mod(am - ap, 2 * np.pi)


def asimov_binned(x, S, phi, bins=32, rng=(0, 2 * np.pi), n_events=None):
    """Asimov Gaussian significance of phi from the binned shape of x.

    Nominal histogram from unit weights, alternative from w(phi) supplied as
    an exact-h reweighting; both normalised to n_events.  Returns Delta chi^2.
    """
    n = len(x) if n_events is None else n_events
    w = 1.0 + phi * S          # first order; callers pass exact w for large phi
    h0, edges = np.histogram(x, bins=bins, range=rng)
    h1, _ = np.histogram(x, bins=bins, range=rng, weights=w)
    scale = n / h0.sum()
    h0 = h0 * scale
    h1 = h1 * (n / h1.sum())
    k = h0 > 0
    return float(np.sum((h1[k] - h0[k])**2 / h0[k])), edges
