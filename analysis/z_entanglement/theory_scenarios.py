"""Expected ditau entanglement for simple Z polarisation scenarios (no data).

R in Collins-Soper axes from the textbook angular coefficients with the
Lam-Tung relation A2 = A0 (LO q qbar -> Z g) and A1 = 0, plus a vector part
<S_z> along z_CS.  Output: concurrence of the predicted state at each tau
direction, in the CS-referenced helicity basis and in the adaptive basis
(rotated about k to diagonalise the transverse block), versus cos theta_CS.
"""
import json
import numpy as np
import ztheory as zt

Z = zt.ZDecay()
ep = np.array([-1, -1j, 0]) / np.sqrt(2); em = np.array([1, -1j, 0]) / np.sqrt(2); e0 = np.array([0, 0, 1.])
P = lambda e: np.outer(e, e.conj())


def R_scenario(A0, A2, sz):
    """A0 = 2 rho_00 (longitudinal fraction); A2 = linear x-y (cos 2phi); sz = <S_z> vector part."""
    f0 = A0 / 2
    ft = 1 - f0
    R = ft * (0.5 * (1 + sz / ft) * P(ep) + 0.5 * (1 - sz / ft) * P(em)) + f0 * P(e0)
    # A2 term: coherence between helicity +1 and -1 along z (linear polarisation along x)
    R = R + (A2 / 4) * (np.outer(ep, em.conj()) + np.outer(em, ep.conj())) * -1
    return R


def curve(R, nct=41, nphi=72):
    cts = np.linspace(-0.975, 0.975, nct)
    out = []
    for c in cts:
        s = np.sqrt(1 - c * c)
        ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
        k = np.stack((s * np.cos(ph), s * np.sin(ph), np.full(nphi, c)), -1)
        O = zt.helicity_basis(k, np.broadcast_to(e0, k.shape))
        N, Bm, Bp, C = Z.state(zt.rotate(R, O))
        w = N / N.sum()
        conc_ev = [zt.concurrence(zt.rho4(Bm[i], Bp[i], C[i])) for i in range(nphi)]
        Cav = np.einsum('n,nab->ab', w, C)
        Bmav, Bpav = w @ Bm, w @ Bp
        out.append(dict(ct=float(c), rate=float(N.mean()), conc_event_mean=float(w @ np.array(conc_ev)),
                        conc_cs_basis=zt.concurrence(zt.rho4(Bmav, Bpav, Cav))))
    return out


res = {}
for name, (A0, A2, sz) in {'DY_LO (A0=0)': (0, 0, 0.2), 'A0=A2=0.5': (0.5, 0.5, 0.2),
                           'A0=A2=0.85': (0.85, 0.85, 0.2), 'unpolarised': (2 / 3, 0, 0)}.items():
    R = R_scenario(A0, A2, sz)
    # check the angular coefficients by the moments of W
    cv = curve(R)
    rate = np.array([r['rate'] for r in cv])
    ce = np.array([r['conc_event_mean'] for r in cv])
    cc = np.array([r['conc_cs_basis'] for r in cv])
    res[name] = dict(R_trace=float(np.real(np.trace(R))), eig_min=float(np.linalg.eigvalsh(R).min()),
                     conc_at=dict(zip(['ct=0', 'ct=0.5', 'ct=0.9'],
                                      [float(np.interp(x, [r['ct'] for r in cv], ce)) for x in (0, 0.5, 0.9)])),
                     integrated_event_mean=float((rate * ce).sum() / rate.sum()),
                     integrated_cs_basis_bins=float((rate * cc).sum() / rate.sum()),
                     curve=cv)
    print(name, 'minEig(R)=%.3f' % res[name]['eig_min'], {k: round(v, 3) for k, v in res[name]['conc_at'].items()},
          'rate-weighted <conc> event %.3f  CS-basis-bins %.3f' % (res[name]['integrated_event_mean'], res[name]['integrated_cs_basis_bins']))
json.dump(res, open('results/theory_scenarios.json', 'w'), indent=1)
