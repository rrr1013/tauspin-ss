"""Pipeline closure on a toy with a known Z spin density (no LHE needed).

Events: each has a Z density R_e drawn from a two-component mixture (helicity
+-1 along z with random weights, plus a longitudinal admixture), tau+ direction
k ~ W_{R_e}(k), and physical polarimeters h-, h+ (|h| = 1) drawn from
1 + B-.h- + B+.h+ + h- C h+.  The analysis sees only k, canonical h = -h and
must recover <R> from k alone and predict the measured C in bins of cos theta.
"""
import json
import sys
import numpy as np
import ztheory as zt
import zent

rng = np.random.default_rng(7)
Z = zt.ZDecay()
A, basis = zt.direction_moment_map(Z, 120)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000

def unit_sphere(m):
    v = rng.normal(size=(m, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)

# event Z densities: f+ |+><+| + f- |-><-| + f0 |0><0| along z (+ small coherence 0/+)
ep = np.array([-1, -1j, 0]) / np.sqrt(2); em = np.array([1, -1j, 0]) / np.sqrt(2); e0 = np.array([0, 0, 1.])
fp = rng.uniform(0.3, 0.7, n); f0 = rng.uniform(0, 0.3, n); fm = 1 - fp - f0
fm = np.clip(fm, 0, None); s = fp + fm + f0; fp, fm, f0 = fp / s, fm / s, f0 / s
R = (fp[:, None, None] * np.outer(ep, ep.conj()) + fm[:, None, None] * np.outer(em, em.conj())
     + f0[:, None, None] * np.outer(e0, e0))
R = R + 0.1 * np.sqrt(fp * f0)[:, None, None] * (np.outer(ep, e0) + np.outer(e0, ep.conj()))
ref = np.array([0, 0, 1.])
# sample k by accept-reject on N(k)
k = np.empty((n, 3)); todo = np.arange(n)
Nmax = np.real(np.einsum('ij,ij->', np.eye(3), Z.N)) * 2.5
while todo.size:
    kk = unit_sphere(todo.size)
    O = zt.helicity_basis(kk, np.broadcast_to(ref, kk.shape))
    N = Z.state(zt.rotate(R[todo], O))[0]
    acc = rng.uniform(0, Nmax, todo.size) < N
    assert N.max() < Nmax
    k[todo[acc]] = kk[acc]; todo = todo[~acc]
O = zt.helicity_basis(k, np.broadcast_to(ref, k.shape))
_, Bm, Bp, C = Z.state(zt.rotate(R, O))
# sample physical h in the (n,r,k) basis
hm = np.empty((n, 3)); hp = np.empty((n, 3)); todo = np.arange(n)
while todo.size:
    a, b = unit_sphere(todo.size), unit_sphere(todo.size)
    f = 1 + np.einsum('na,na->n', Bm[todo], a) + np.einsum('na,na->n', Bp[todo], b) + np.einsum('na,nab,nb->n', a, C[todo], b)
    acc = rng.uniform(0, 4, todo.size) < f
    hm[todo[acc]] = a[acc]; hp[todo[acc]] = b[acc]; todo = todo[~acc]
hm_can, hp_can = -hm, -hp

Rtrue = R.mean(0)                      # every event has equal weight here
Rfit = zent.fit_R_from_directions(k, A, basis)
out = {'n': n, 'max_abs_dR': float(np.abs(Rfit - Rtrue).max())}
_, Bm_p, Bp_p, C_p = Z.state(zt.rotate(Rfit, O))
ct = k[:, 2]
edges = np.linspace(-1, 1, 6)
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ct >= lo) & (ct < hi)
    bm_, bp_, c_ = zent.measured_state(hm_can[m], hp_can[m])
    e = zent.measured_state_err(hm_can[m], hp_can[m])
    pred = C_p[m].mean(0)
    pulls = ((c_ - pred) / e[2]).ravel()
    rows.append(dict(bin=[lo, hi], n=int(m.sum()), Cmeas_diag=np.diag(c_).round(3).tolist(),
                     Cpred_diag=np.diag(pred).round(3).tolist(), Bk_meas=round(bm_[2], 3),
                     Bk_pred=round(Bm_p[m, 2].mean(), 3), max_pull=float(np.abs(pulls).max()),
                     chi2_9=float((pulls**2).sum())))
out['bins'] = rows
print(json.dumps(out, indent=1))
