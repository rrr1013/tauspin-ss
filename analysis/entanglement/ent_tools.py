"""Ditau spin density matrix, entanglement observables, and a calibrated estimator.

Physics conventions
-------------------
The spin density of the tau pair is parameterised in the (n, r, k) basis of
`analysis/mode_pair_auc_origin/polarimeter.frames` (k along the tau+ momentum in
the pair rest frame, n = beam x k, r = beam_perp, right handed):

    rho = 1/4 ( 1x1 + B- . sigma x 1 + 1 x B+ . sigma + C_ij sigma_i x sigma_j )

The joint distribution of the two physical polarimeter vectors is

    p(h-, h+) = p0(h-) p0(h+) ( 1 + B- . h- + B+ . h+ + h-^T C h+ )

where p0 is the *unpolarised* measure of the polarimeter direction under the
analysis selection.  It is not isotropic here: the visible-pT selection tilts it
(<h_k> != 0), which is exactly why the textbook plug-in C = 9 <h- h+^T> fails on
this sample.

This project's canonical `h` is -1 times the physical polarimeter on both sides
(see analysis/cp_mixing/cp_density.py).  A fit against the stored `h` therefore
returns (b-, b+, C) = (-B-, -B+, C): the correlation block is unchanged and the
single-spin blocks flip sign.  `theta_to_density` applies that flip.

Estimator
---------
theta = (B-, B+, C) as a 15-vector; phi(h) = (h-, h+, h- outer h+) the matching
15 features built from the *exact* polarimeter; g the same 15 features built
from whatever vector is actually available (exact h, or a regression output
h_pred).  Then, exactly,

    E_theta[g] = ( A + R theta ) / Z(theta) ,
    A = E0[g] ,  R = E0[g phi^T] ,  Z(theta) = 1 + e . theta ,  e = E0[phi]

with E0 the unpolarised expectation.  Z is the normalisation of the spin factor
over the *selected* phase space; it is 1 only if E0[phi] vanishes, which it does
not here (the selection leaves E0[h_k] != 0).  Dropping it costs about 3%.  A and R are properties of the selection
and of the reconstruction, not of the spin state, so they are calibrated once
from simulation -- E0 is reached from a generated sample by the weight 1/f,
f = 1 + theta_gen . phi -- and the measurement is then still a linear solve, because Z is linear in theta:

    ( R - <g> e^T ) theta_hat = <g>_sample - A .

For g = phi, R is a symmetric positive definite Gram matrix and this is ordinary
moment unfolding.  For g built from h_pred the same R absorbs the regression
attenuation, so no separate dilution correction is needed; the price is purely
statistical, through R^-1, and that price is what this run measures.

Calibrating A and R needs the generated theta, so a reco-level measurement of
this kind is simulation-calibrated by construction.  That is a claim boundary,
not a bug: what is tested is closure against *injected* states the calibration
was never told about.  Note also that the 15 moment equations alone do not
identify theta: theta = 0 reproduces the sample mean trivially if E0 is allowed
to be the observed distribution.  The unpolarised measure has to come from
outside the moments.
"""
import numpy as np

SIG = np.array([[[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ----------------------------------------------------------------- parameters
def pack(Bm, Bp, C):
    return np.concatenate([np.asarray(Bm, float), np.asarray(Bp, float),
                           np.asarray(C, float).reshape(9)])


def unpack(theta):
    t = np.asarray(theta, float)
    return t[:3].copy(), t[3:6].copy(), t[6:].reshape(3, 3).copy()


def features(h):
    """phi = (h-, h+, h- outer h+) for h of shape (N, 2, 3)."""
    hm = np.asarray(h[:, 0], float)
    hp = np.asarray(h[:, 1], float)
    return np.concatenate([hm, hp, np.einsum('ni,nj->nij', hm, hp).reshape(-1, 9)], axis=1)


def f_nominal(phi_exact, theta):
    """1 + theta . phi, the spin factor of the generated sample."""
    return 1.0 + np.asarray(phi_exact, float) @ np.asarray(theta, float)


# ------------------------------------------------------------------ estimator
def calibrate(phi_exact, g, theta_gen, weights=None):
    """A = E0[g], R = E0[g phi^T] from a sample generated with theta_gen."""
    f = f_nominal(phi_exact, theta_gen)
    if f.min() <= 0:
        raise ValueError('non-positive spin factor in the calibration sample')
    w = 1.0 / f if weights is None else np.asarray(weights, float) / f
    s = w.sum()
    A = (w[:, None] * g).sum(0) / s
    R = np.einsum('n,ni,nj->ij', w, g, phi_exact) / s
    e = (w[:, None] * phi_exact).sum(0) / s
    info = {'ess_frac': float(s ** 2 / (len(w) * (w ** 2).sum())),
            'f_min': float(f.min()), 'cond_R': float(np.linalg.cond(R))}
    return {'A': A, 'R': R, 'e': e, 'info': info}


def unfold(cal, g, weights=None):
    """Solve (R - <g> e^T) theta = <g> - A."""
    w = np.ones(len(g)) if weights is None else np.asarray(weights, float)
    obs = (w[:, None] * g).sum(0) / w.sum()
    return np.linalg.solve(cal['R'] - np.outer(obs, cal['e']), obs - cal['A'])


def naive_theta(g, weights=None):
    """Textbook plug-in: B = 3<h>, C = 9<h- h+^T>, i.e. isotropic p0."""
    w = np.ones(len(g)) if weights is None else np.asarray(weights, float)
    m = (w[:, None] * g).sum(0) / w.sum()
    return np.concatenate([3 * m[:3], 3 * m[3:6], 9 * m[6:]])


def moment_unfold_C(h, weights=None):
    """Model-light alternative: C = M-^-1 <h- h+^T> M+^-1 from *observed*
    per-side second moments.  Needs no knowledge of the generated state, but
    only applies when the available vector is itself a polarimeter direction."""
    h = np.asarray(h, float)
    w = np.ones(len(h)) if weights is None else np.asarray(weights, float)
    s = w.sum()
    Mm = np.einsum('n,ni,nj->ij', w, h[:, 0], h[:, 0]) / s
    Mp = np.einsum('n,ni,nj->ij', w, h[:, 1], h[:, 1]) / s
    raw = np.einsum('n,ni,nj->ij', w, h[:, 0], h[:, 1]) / s
    return np.linalg.solve(Mm, np.linalg.solve(Mp, raw.T).T)


# ------------------------------------------------------- entanglement content
def theta_to_density(theta, flip_single_spin=True):
    """rho from a theta fitted against the canonical (sign-flipped) h."""
    bm, bp, C = unpack(theta)
    if flip_single_spin:
        bm, bp = -bm, -bp
    rho = np.kron(I2, I2).astype(complex)
    for i in range(3):
        rho = rho + bm[i] * np.kron(SIG[i], I2) + bp[i] * np.kron(I2, SIG[i])
    for i in range(3):
        for j in range(3):
            rho = rho + C[i, j] * np.kron(SIG[i], SIG[j])
    return rho / 4.0


def partial_transpose(rho):
    r = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1)
    return r.reshape(4, 4)


def concurrence(rho):
    Y = np.kron(SIG[1], SIG[1])
    R = rho @ Y @ rho.conj() @ Y
    ev = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(R)), 0, None)))[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def observables(theta):
    """Entanglement / Bell content of a theta."""
    _, _, C = unpack(theta)
    rho = theta_to_density(theta)
    ev_rho = np.sort(np.real(np.linalg.eigvals(rho)))
    ev_pt = np.sort(np.real(np.linalg.eigvals(partial_transpose(rho))))
    s = np.linalg.svd(C, compute_uv=False)
    return {
        'neg_eig_pt': float(-ev_pt[0]),           # > 0  <=>  entangled (PPT)
        'negativity': float(sum(-e for e in ev_pt if e < 0)),
        'min_eig_rho': float(ev_rho[0]),          # < 0  <=>  unphysical estimate
        'purity': float(np.real(np.trace(rho @ rho))),
        'concurrence': concurrence(rho),
        'm12': float(s[0] ** 2 + s[1] ** 2),      # > 1  <=>  CHSH violable
        'chsh_max': float(2 * np.sqrt(s[0] ** 2 + s[1] ** 2)),
        'trace_C': float(np.trace(C)),
        'singular_values': s.tolist(),
    }


def witness_coefficients(theta_ref):
    """Optimal entanglement witness of the reference state, as a linear
    functional of theta: <W> = const + coef . theta, negative iff entangled.

    W = |v><v|^{T2} with v the eigenvector of rho_ref^{T2} of lowest eigenvalue;
    tr(rho W) = <v| rho^{T2} |v> is then the PPT-violating eigenvalue itself.
    """
    rho = theta_to_density(theta_ref)
    _, vec = np.linalg.eigh(partial_transpose(rho))
    v = vec[:, 0]
    P = np.outer(v, v.conj())
    W = partial_transpose(P)                    # tr(rho W) = <v|rho^{T2}|v>
    coef = np.zeros(15)
    for i in range(3):
        coef[i] = -np.real(np.trace(W @ np.kron(SIG[i], I2))) / 4
        coef[3 + i] = -np.real(np.trace(W @ np.kron(I2, SIG[i]))) / 4
    for i in range(3):
        for j in range(3):
            coef[6 + 3 * i + j] = np.real(np.trace(W @ np.kron(SIG[i], SIG[j]))) / 4
    const = float(np.real(np.trace(W)) / 4)
    return coef, const


# ------------------------------------------------------------------ inference
def theta_covariance(g, cal, weights=None):
    """Cov(theta_hat) by the delta method, calibration held fixed."""
    w = np.ones(len(g)) if weights is None else np.asarray(weights, float)
    mu = (w[:, None] * g).sum(0) / w.sum()
    d = g - mu
    Cg = np.einsum('n,ni,nj->ij', w ** 2, d, d) / (w.sum() ** 2)
    M = cal['R'] - np.outer(mu, cal['e'])
    theta = np.linalg.solve(M, mu - cal['A'])
    # d theta / d <g> = M^-1 ( I + theta e^T )
    J = np.linalg.solve(M, np.eye(15) + np.outer(theta, cal['e']))
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    return J @ Cg @ J.T, n_eff


def bootstrap(g, cal, n_boot=2000, seed=0, weights=None, stat=None):
    """Bootstrap the linear estimator with the calibration held fixed."""
    stat = observables if stat is None else stat
    rng = np.random.default_rng(seed)
    n = len(g)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    out, thetas = {}, []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        obs = (w[k, None] * g[k]).sum(0) / w[k].sum()
        th = np.linalg.solve(cal['R'] - np.outer(obs, cal['e']), obs - cal['A'])
        thetas.append(th)
        for key, val in stat(th).items():
            if isinstance(val, list):
                continue
            out.setdefault(key, []).append(val)
    return ({k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()},
            np.array(thetas))


def project_physical(theta, n_iter=200):
    """Nearest physical state: project rho onto the positive semidefinite,
    unit-trace cone (the standard fix in quantum state tomography) and read the
    corresponding theta back.  Applied after the linear estimate, it keeps every
    reported density matrix a density matrix."""
    rho = theta_to_density(theta)
    ev, vec = np.linalg.eigh((rho + rho.conj().T) / 2)
    # simplex projection of the eigenvalues, trace preserved at 1
    lo, hi = ev.min() - 1.0, ev.max()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if np.clip(ev - mid, 0, None).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    lam = np.clip(ev - 0.5 * (lo + hi), 0, None)
    rho_p = vec @ np.diag(lam) @ vec.conj().T
    bm = np.array([np.real(np.trace(rho_p @ np.kron(SIG[i], I2))) for i in range(3)])
    bp = np.array([np.real(np.trace(rho_p @ np.kron(I2, SIG[i]))) for i in range(3)])
    C = np.array([[np.real(np.trace(rho_p @ np.kron(SIG[i], SIG[j]))) for j in range(3)]
                  for i in range(3)])
    return pack(-bm, -bp, C)          # back into canonical-h sign convention
