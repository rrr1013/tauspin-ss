"""Derive C(phi) from the spinor trace and check it against the H sample."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cp_density import M_PHI, M_TAU, c_analytic, density  # noqa: E402

np.set_printoptions(precision=5, suppress=True)
beta = np.sqrt(1 - 4 * M_TAU**2 / M_PHI**2)
print(f'beta = {beta:.6f}  gamma = {1/np.sqrt(1-beta**2):.3f}')
for deg in (0, 15, 30, 45, 60, 90):
    phi = np.deg2rad(deg)
    N, Bm, Bp, C = density(phi)
    print(f'--- phi_tau = {deg:3d} deg')
    print('  B- =', Bm, ' B+ =', Bp)
    print('  C  =\n', C)
    print('  C(UR analytic) - C =\n', c_analytic(phi) - C)
