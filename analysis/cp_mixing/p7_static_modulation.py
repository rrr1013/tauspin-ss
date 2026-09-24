"""Why the reco-h acoplanarity peak moves less than 2 phi_tau.

The predicted h of the two sides are correlated by the network itself (both are
built from the same reconstructed event), so the Delta phi_h distribution has a
spin-independent component that does not move with phi_tau.  Z events carry no
transverse spin correlation at all, so the modulation left there measures that
component directly.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE))
from cp_tools import acoplanarity  # noqa: E402


def modulation(x, w=None, bins=16):
    n, e = np.histogram(x, bins=bins, range=(0, 2 * np.pi), weights=w)
    n2, _ = np.histogram(x, bins=bins, range=(0, 2 * np.pi),
                         weights=None if w is None else w**2)
    d = n / n.sum() / np.diff(e)
    err = np.sqrt(n2 if w is not None else n) / n.sum() / np.diff(e)
    c = 0.5 * (e[1:] + e[:-1])
    M = np.stack([np.ones_like(c), np.cos(c), np.sin(c)], 1)
    W = np.diag(1 / np.maximum(err, 1e-12)**2)
    b = np.linalg.solve(M.T @ W @ M, M.T @ W @ d)
    return float(np.hypot(b[1], b[2]) / b[0]), float(np.rad2deg(np.arctan2(-b[2], b[1])))


D = np.load(HERE / 'data' / 'truth_surface_cleo.npz')
P = np.load(HERE / 'data' / 'gen3pi_full22_s42.npz')
y = D['labels'].astype(bool)
modes = D['modes']
rr = (modes[:, 0] == 1) & (modes[:, 1] == 1)
out = {}
for tag, h in (('exact_h', P['h'].astype(float)), ('reco_h_full22', P['h_pred'].astype(float))):
    for lab, k in (('H_rhorho', rr & y), ('Z_rhorho', rr & ~y), ('H_all', y), ('Z_all', ~y)):
        a, ph = modulation(acoplanarity(h[k]))
        out[f'{tag}:{lab}'] = {'n': int(k.sum()), 'amplitude': a, 'phase_deg': ph}
(HERE / 'results' / 'static_modulation.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f'{k:28s} n={v["n"]:6d}  A={v["amplitude"]:.3f}  phase={v["phase_deg"]:+7.1f} deg')
