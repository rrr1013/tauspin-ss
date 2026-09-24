"""Alignment and convention audit for the CP-mixing run.

Checks that the cached truth surface (CLEO h_ref, truth pions/pi0/nu) and the
gen3pi point-h prediction files describe the same 59,390 validation events in
the same row order, that the stored generator-current h agrees with a local
rebuild from polarimeter.py, and pins down which side is tau- and what the
(n,r,k) handedness is.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mode_pair_auc_origin'))
from polarimeter import canonical_h, frames  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'
ARMS = ['base_s43', 'full22_s42', 'full22_s43', 'idealip22_s42']

D = np.load(DATA / 'truth_surface_cleo.npz')
print('truth surface rows', len(D['labels']), 'H', int(D['labels'].sum()))

ref = None
for a in ARMS:
    P = np.load(DATA / f'gen3pi_{a}.npz')
    assert np.array_equal(P['global_indices'], D['global_indices']), a
    assert np.array_equal(P['event_numbers'], D['event_numbers']), a
    assert np.array_equal(P['truth_modes'], D['modes']), a
    assert np.allclose(P['labels'], D['labels']), a
    if ref is None:
        ref = P['h']
    else:
        assert np.allclose(ref, P['h'], atol=1e-6), a
    print(f'{a:16s} |h_pred| mean {np.linalg.norm(P["h_pred"], axis=-1).mean():.4f}'
          f'  cos(h,hpred) {np.mean(np.sum(P["h"] * P["h_pred"], -1) / np.linalg.norm(P["h_pred"], axis=-1)):.4f}')
print('stored target h identical across arms; |h| dev', np.abs(np.linalg.norm(ref, axis=-1) - 1).max())

# local rebuild of the generator-current h
fr = frames(D['pions'], D['pi0'], D['nu4'])
h_gen = np.array(D['h_ref'], dtype=float)
for md in (0, 1, 3):
    for s in (0, 1):
        sel, h = canonical_h(fr, D['modes'], D['pion_charges'], s, md)
        h_gen[sel, s] = h
print('rebuild vs stored target: max |diff|', np.abs(h_gen - ref).max(),
      ' mean dot', np.mean(np.sum(h_gen * ref, -1)))
print('CLEO h_ref vs generator h: mean dot per side',
      [float(np.mean(np.sum(D['h_ref'][:, s] * h_gen[:, s], -1))) for s in (0, 1)])

# which side is tau-?
q = D['pion_charges']
cnt = D['pion_counts']
for s in (0, 1):
    tot = np.array([q[i, s, :cnt[i, s]].sum() for i in range(2000)])
    print(f'side {s}: mean total pion charge over first 2000 events = {tot.mean():+.3f}')

# handedness of (n, r, k)
B = fr['basis']
n, r, k = B[:, 0], B[:, 1], B[:, 2]
print('mean (n x r) . k =', float(np.mean(np.sum(np.cross(n, r) * k, -1))))

# measured spin correlation matrices with the generator h
y = D['labels']
for lab, nm in ((1, 'H'), (0, 'Z')):
    m = y == lab
    C = 9 * np.einsum('ni,nj->ij', h_gen[m, 0], h_gen[m, 1]) / m.sum()
    print(f'C_{nm} (generator h, n/r/k):\n', np.array2string(C, precision=4, suppress_small=False))
    print(f'  <h> side0 {3*h_gen[m,0].mean(0).round(4)}  side1 {3*h_gen[m,1].mean(0).round(4)}')
