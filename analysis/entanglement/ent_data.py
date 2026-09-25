"""Shared loading of the canonical 59,390-event validation surface.

Same artifacts as the 2026-09-25 CP run (analysis/cp_mixing/data), nothing new
is produced upstream.  `h_gen` is the generator-current canonical polarimeter
(the 2026-09-24 teacher); `h_pred` is the point-h regression output of a given
arm.  Row order is asserted identical across every file.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[0] / 'cp_mixing' / 'data'
sys.path.insert(0, str(HERE.parents[0] / 'mode_pair_auc_origin'))
sys.path.insert(0, str(HERE.parents[0] / 'cp_mixing'))

from polarimeter import canonical_h, frames  # noqa: E402

MODES = {0: 'pi', 1: 'rho', 3: '3pi'}
PAIRS = [(0, 0, 'pi x pi'), (1, 1, 'rho x rho'), (3, 3, '3pi x 3pi'),
         (0, 1, 'pi x rho'), (0, 3, 'pi x 3pi'), (1, 3, 'rho x 3pi')]
ARMS = [('base_s43', 'reco, no geometry'),
        ('full22_s42', 'reco + 3 IP + SV (seed 42)'),
        ('full22_s43', 'reco + 3 IP + SV (seed 43)'),
        ('idealip22_s42', 'reco + ideal-IP oracle')]


def pair_mask(modes, a, b):
    return (((modes[:, 0] == a) & (modes[:, 1] == b))
            | ((modes[:, 0] == b) & (modes[:, 1] == a)))


def load_surface(data=DATA):
    D = np.load(Path(data) / 'truth_surface_cleo.npz')
    fr = frames(D['pions'], D['pi0'], D['nu4'])
    h_gen = np.array(D['h_ref'], float)
    for md in (0, 1, 3):
        for s in (0, 1):
            sel, h = canonical_h(fr, D['modes'], D['pion_charges'], s, md)
            h_gen[sel, s] = h
    return {'h_gen': h_gen, 'h_cleo': np.array(D['h_ref'], float),
            'labels': D['labels'].astype(bool), 'modes': D['modes'],
            'weights': np.array(D['weights'], float),
            'global_indices': D['global_indices']}


def load_arm(arm, surface, data=DATA, prefix='gen3pi', split=''):
    suffix = f'_{split}' if split else ''
    P = np.load(Path(data) / f'{prefix}_{arm}{suffix}.npz')
    if not split:
        assert np.array_equal(P['global_indices'], surface['global_indices']), arm
        assert np.array_equal(P['truth_modes'], surface['modes']), arm
        assert np.abs(np.asarray(P['h'], float) - surface['h_gen']).max() < 1e-5, arm
    return {'h_pred': np.array(P['h_pred'], float), 'h': np.array(P['h'], float),
            'labels': np.asarray(P['labels']).astype(bool),
            'modes': np.asarray(P['truth_modes'])}
