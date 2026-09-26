"""Production spin density of the two samples, for the FIRST moment B.

Reuses the polarised Dirac traces of the 2026-09-25 CP run (`cp_density`) and
the 2026-09-26 entanglement run (`z_density`).  Nothing new is derived about C;
what this module adds is that the *sign* of B matters here, and the project's
canonical `h` is -1 times the physical polarimeter on both sides
(see cp_density's docstring).  So, written for the stored canonical h,

    p(h-, h+) proportional to  1 - B- . h-_can - B+ . h+_can + h-_can^T C h+_can

with B, C from the traces below.  Basis (n, r, k), k_hat along the tau+
momentum in the pair rest frame.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE.parents[0] / 'cp_mixing'))
sys.path.insert(0, str(HERE.parents[0] / 'entanglement'))

from cp_density import density as h_density          # noqa: E402
import z_density as ZD                               # noqa: E402

SIN2W_SAMPLE = ZD.SIN2W


def higgs_state(phi_tau=0.0):
    """B-, B+, C of the CP-even scalar.  B is exactly zero."""
    _, Bm, Bp, C = h_density(phi_tau)
    return Bm, Bp, C


def z_state(sin2w=SIN2W_SAMPLE):
    """B-, B+, C of an unpolarised Z with SM tau couplings."""
    _, Bm, Bp, C = ZD.density(sin2w=sin2w)
    return Bm, Bp, C


def p_tau_of_sin2w(sin2w):
    """P_tau = -A_tau = -2 v a / (v^2 + a^2) with v = -1/2 + 2 s^2, a = -1/2."""
    v, a = ZD.couplings(sin2w)
    return -2 * v * a / (v * v + a * a)


def dp_dsin2w(sin2w=SIN2W_SAMPLE, eps=1e-6):
    return (p_tau_of_sin2w(sin2w + eps) - p_tau_of_sin2w(sin2w - eps)) / (2 * eps)


if __name__ == '__main__':
    BmH, BpH, CH = higgs_state()
    BmZ, BpZ, CZ = z_state()
    print('H  B- =', np.round(BmH, 8), ' B+ =', np.round(BpH, 8))
    print('H  C  =\n', np.array2string(CH, precision=5, suppress_small=True))
    print('Z  B- =', np.round(BmZ, 5), ' B+ =', np.round(BpZ, 5))
    print('Z  C  =\n', np.array2string(CZ, precision=5, suppress_small=True))
    print('P_tau(sin2w=%.5f) = %.6f' % (SIN2W_SAMPLE, p_tau_of_sin2w(SIN2W_SAMPLE)))
    print('dP_tau/dsin2w      = %.4f' % dp_dsin2w())
