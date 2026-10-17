"""Run gen3pi_teacher/train_point_h.py unchanged, for the training-composition control.

Two shims, both inert for the existing mixed arms:
  * the dataset's oracle truth-input file is taken from TRUTH_INPUTS_DIR (the row-aligned subset written by
    s5_make_subset_targets.py), because the dataset checks it row by row against the kept rows;
  * when the TRAIN split holds a single class (H only / Z only), the balanced H/Z stream gets two pseudo-classes
    (alternating positions of that class).  The label is used only by this sampler -- it is not a network input
    and not in the loss -- so the stream semantics (paired draws, steps per epoch) stay those of the main recipe.
Same env as train_point_h.py.
"""
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation')
sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN')
import reco_data_original_geometry as R  # noqa: E402
import reco_truth_common_v2 as c  # noqa: E402

R.TRUTH_INPUTS_DIR = Path(os.environ['TRUTH_INPUTS_DIR'])
_Stream = c.BalancedCommonStreamDataset


class SingleClassStream(_Stream):
    def __init__(self, packed, *, shuffle, balanced, seed=c.SEED):
        labels = packed.labels
        if bool((labels == 1).any()) and bool((labels == 0).any()):
            super().__init__(packed, shuffle=shuffle, balanced=balanced, seed=seed)
            return
        IterableDataset.__init__(self)
        self.packed, self.shuffle, self.balanced, self.seed, self.epoch = packed, shuffle, balanced, seed, 0
        allidx = torch.arange(len(labels))
        self.sample_indices = {1: allidx[0::2], 0: allidx[1::2]}
        print(f'single-class TRAIN ({int(labels[0])}): pseudo-classes {len(allidx[0::2])} / {len(allidx[1::2])}', flush=True)


c.BalancedCommonStreamDataset = SingleClassStream
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher' / 'train_point_h.py'), run_name='__main__')
