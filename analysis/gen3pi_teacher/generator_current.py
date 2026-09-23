"""Drop-in replacement for the CLEO 3pi current inside HybridPolarimeter.

HybridPolarimeter calls ``self.current(ordered, masses, charges)`` with the three
pions in the tau rest frame (px,py,pz,E), same-sign pions first, and uses the
first three output columns as the polarimeter direction without any further
charge sign.  Columns 3 and 4 (raw norm, omega) are only checked for being
positive / non-zero.

This returns the polarimeter of the generator (TauDecay UFO) current, in the
project's canonical sign (-1 x physical for both tau charges), with the
neutrino fixed by four-momentum conservation, N = (0,0,0,M) - sum(pions).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
from polarimeter import current_taudecay, polarimeter  # noqa: E402


def generator_current(pions, masses, charges, chunk=200000):
    pions = np.asarray(pions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    charges = np.asarray(charges)
    out = np.empty((len(pions), 5))
    for a in range(0, len(pions), chunk):
        p = pions[a:a + chunk]
        ch = charges[a:a + chunk]
        m = masses[a:a + chunk]
        sgn = np.where(ch < 0, 1.0, -1.0)[:, None, None]   # CP mirror for tau+
        e = np.concatenate([p[..., 3:4], sgn * p[..., :3]], -1)
        tau = np.zeros((len(p), 4))
        tau[:, 0] = m
        nu = tau - e.sum(1)
        J = current_taudecay(e[:, 0], e[:, 1], e[:, 2])
        out[a:a + chunk, :3] = -polarimeter(J, nu, m_tau=1.777)
        out[a:a + chunk, 3] = 1.0
        out[a:a + chunk, 4] = 1.0
    return out


def install(hybrid):
    """Replace the 3pi current of a HybridPolarimeter instance in place."""
    hybrid.current = generator_current
    hybrid.current_origin = 'TauDecay UFO (sm__taudecay_UFO) Kuehn-Santamaria current'
    return hybrid
