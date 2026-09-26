"""Shared loading for the polarisation run.  Same artifacts as the CP and
entanglement runs (analysis/cp_mixing/data); nothing new is produced upstream."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'entanglement'))
import ent_data as ED  # noqa: E402

DATA = ED.DATA
MODES = ED.MODES
PAIRS = ED.PAIRS
ARMS = ED.ARMS + [('full22_shuffle_s42', 'shuffle control (other event h_pred)')]
RESULTS = HERE / 'results'
FIGS = HERE / 'figures'


def load_surface():
    return ED.load_surface()


def load_arm(arm, surface, prefix='gen3pi', split=''):
    if arm.startswith('full22_shuffle'):
        P = np.load(DATA / f'cleo_{arm}.npz')          # only a CLEO-teacher copy exists
        return {'h_pred': np.array(P['h_pred'], float), 'h': np.array(P['h'], float),
                'labels': np.asarray(P['labels']).astype(bool),
                'modes': np.asarray(P['truth_modes'])}
    return ED.load_arm(arm, surface, prefix=prefix, split=split)


def load_ladder():
    return np.load(DATA / 'ladder_validation.npz')


def side_mode_mask(modes, mode):
    """(n, 2) boolean mask of tau sides in a given truth decay mode."""
    return modes == mode
