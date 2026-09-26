"""Estimators for the single-tau polarisation P_tau from polarimeter vectors.

Conventions (see pol_density.py).  For the stored *canonical* h the ditau
density of the Z sample is

    f_Z(P) = 1 + P (h-_k + h+_k) + h-^T C_Z h+ ,   C_Z = diag(0, 0, 1)

so B_can = -P k_hat and the nominal sample has P = P_0 = -0.147037.  For the H
sample

    f_H = 1 + h-^T C_H h+ ,   C_H = diag(1, 1, -1) ,   B = 0 exactly.

Two independent things are computed here.

1.  Linear-response sensitivity.  For any observable T,
        d<T>/dP = Cov_0(T, S) ,   S = dlog f/dP = (h-_k + h+_k) / f ,
    so sigma_N(P) = sqrt(Var T / N) / |Cov(T, S)|.  S needs the *exact* h, T may
    be anything (reco h_pred, a visible-only variable, ...).  This prices every
    method on identical events and never refits a network.

2.  Moment inversion with the H sample as the control region.  Writing the
    unpolarised (spin-averaged, post-selection) measure moments as
    m^s_i = E_0[h^s_i], M^s_ij = E_0[h^s_i h^s_j] and assuming the two sides are
    independent under p_0,

        E[h^-_k] = ( m^-_k + P (M^-_kk + m^-_k m^+_k) + (C M^-)_k . m^+ ) / D
        D        = 1 + P (m^-_k + m^+_k) + m^-^T C m^+

    which is linear in P once (m, M) are known.  p_0 is taken from the H rows
    (where B = 0 is exact), so the Z measurement is not circular.
"""
import numpy as np

P0_SAMPLE = -0.147037
C_Z = np.diag([0.0, 0.0, 1.0])
C_H = np.diag([1.0, 1.0, -1.0])


def bilinear(hm, hp, C):
    return np.einsum('ni,ij,nj->n', hm, C, hp)


def f_density(h, P, C):
    hm, hp = h[:, 0], h[:, 1]
    return 1.0 + P * (hm[:, 2] + hp[:, 2]) + bilinear(hm, hp, C)


def score(h, P=P0_SAMPLE, C=C_Z):
    """dlog f/dP at P, from the exact canonical h."""
    f = f_density(h, P, C)
    return (h[:, 0, 2] + h[:, 1, 2]) / f, f


def reweight(h, P_new, P_old=P0_SAMPLE, C=C_Z):
    return f_density(h, P_new, C) / f_density(h, P_old, C)


def wcov(x, y, w=None):
    if w is None:
        return np.cov(np.stack([np.asarray(x, float), np.asarray(y, float)]))
    w = np.asarray(w, float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    dx, dy = x - mx, y - my
    W = w.sum()
    return np.array([[np.sum(w * dx * dx), np.sum(w * dx * dy)],
                     [np.sum(w * dx * dy), np.sum(w * dy * dy)]]) / W


def sensitivity(T, S, n_events=None, w=None, boot=0, seed=0):
    """sigma(P_tau) for n_events from observable T and score S."""
    T = np.asarray(T, float)
    S = np.asarray(S, float)
    ok = np.isfinite(T) & np.isfinite(S)
    T, S = T[ok], S[ok]
    ww = None if w is None else np.asarray(w, float)[ok]
    n = len(T) if n_events is None else n_events
    c = wcov(T, S, ww)
    resp = c[0, 1]
    sig = np.sqrt(c[0, 0] / n) / abs(resp) if resp != 0 else np.inf
    out = {'n': int(len(T)), 'n_events': int(n), 'response': float(resp),
           'sd_T': float(np.sqrt(c[0, 0])), 'sigma': float(sig)}
    if boot:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(boot):
            k = rng.integers(0, len(T), len(T))
            cc = wcov(T[k], S[k], None if ww is None else ww[k])
            if cc[0, 1] != 0:
                vals.append(np.sqrt(cc[0, 0] / n) / abs(cc[0, 1]))
        out['sigma_se'] = float(np.std(vals))
    return out


def cramer_rao(S, n_events=None, w=None):
    S = np.asarray(S, float)
    n = len(S) if n_events is None else n_events
    v = wcov(S, S, w)[0, 0]
    return {'fisher_per_event': float(v), 'sigma': float(1 / np.sqrt(n * v))}


def p0_moments(h, P, C, w=None):
    """Unpolarised-measure moments m^s, M^s by 1/f unfolding of a known state."""
    f = f_density(h, P, C)
    u = 1.0 / f if w is None else np.asarray(w, float) / f
    U = u.sum()
    m = np.array([[np.sum(u * h[:, s, i]) / U for i in range(3)] for s in (0, 1)])
    M = np.array([[[np.sum(u * h[:, s, i] * h[:, s, j]) / U for j in range(3)]
                   for i in range(3)] for s in (0, 1)])
    return m, M


def predict_first_moment(P, C, m, M):
    """E[h^-_k], E[h^+_k] under the product-p0 model."""
    D = 1.0 + P * (m[0, 2] + m[1, 2]) + m[0] @ C @ m[1]
    num_m = m[0, 2] + P * (M[0][2, 2] + m[0, 2] * m[1, 2]) \
        + np.einsum('i,ij,j->', M[0][2], C, m[1])
    num_p = m[1, 2] + P * (M[1][2, 2] + m[0, 2] * m[1, 2]) \
        + np.einsum('i,ij,j->', m[0], C, M[1][:, 2])
    return np.array([num_m / D, num_p / D])


def invert_first_moment(obs, C, m, M, bracket=(-2.0, 2.0)):
    """Solve sum_s E[h^s_k](P) = sum_s obs_s for P (monotone in P; bisection)."""
    tgt = float(np.sum(obs))

    def g(P):
        return float(np.sum(predict_first_moment(P, C, m, M))) - tgt

    lo, hi = bracket
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Control-region estimator: joint p_0 moments from a control sample of known
# state, then the mean of any observable T in a target sample of state
# (P, C_target) is exactly
#     E[T](P) = (E_0[T] + P E_0[T u] + E_0[T v]) / (1 + P E_0[u] + E_0[v])
# with u = h^-_k + h^+_k and v = h^- C_target h^+.  No side-independence is
# assumed, so this works for network outputs as well as for the exact h.
# ---------------------------------------------------------------------------

def control_moments(h_ctrl, T_ctrl, C_ctrl, C_target, P_ctrl=0.0, w=None):
    f = f_density(h_ctrl, P_ctrl, C_ctrl)
    u0 = 1.0 / f if w is None else np.asarray(w, float) / f
    U = u0.sum()
    u = h_ctrl[:, 0, 2] + h_ctrl[:, 1, 2]
    v = bilinear(h_ctrl[:, 0], h_ctrl[:, 1], C_target)
    m = {k: float(np.sum(u0 * val) / U) for k, val in
         (('T', T_ctrl), ('u', u), ('v', v), ('Tu', T_ctrl * u), ('Tv', T_ctrl * v))}
    m['ess_fraction'] = float(u0.sum() ** 2 / np.sum(u0 ** 2) / len(u0))
    m['max_weight_fraction'] = float(u0.max() / U)
    return m


def predict_mean(m, P):
    return (m['T'] + P * m['Tu'] + m['Tv']) / (1.0 + P * m['u'] + m['v'])


def invert_mean(m, obs):
    """Solve predict_mean(m, P) = obs.  Linear-fractional, closed form."""
    a, b = m['T'] + m['Tv'], m['Tu']
    c, d = 1.0 + m['v'], m['u']
    den = b - obs * d
    return (obs * c - a) / den if abs(den) > 1e-12 else np.nan
