"""Spin-flat weighted point-h training: the unchanged gen3pi train_point_h.py recipe, full TRAIN, with the per-event MSE
weighted by w = clip(1 / (1 + h-^T C h+), 0, W_MAX) (exact h, C = C_H = diag(1,1,-1) for H, C_Z = diag(0,0,1) for Z),
normalised to mean 1 in each batch.  Under these weights the training population has no tau-tau spin correlation, so the
network cannot learn a cross-side spin prior; everything else (inputs, arm, architecture, seed, sampler, schedule,
checkpoint window) is the existing recipe.  The validation MSE used by the checkpoint window is weighted the same way.

Implementation: c.move_batch stashes the batch labels and target; torch.nn.functional.mse_loss is wrapped so that, for
that exact target tensor, the weighted per-event loss is returned (all other calls fall through unchanged).
env: as s5_train.py (H_TARGETS_DIR, TRUTH_INPUTS_DIR, P11_DIR, GEO_*, SEED) plus W_MAX (default 10).
"""
import os
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation')
sys.path.insert(0, '/home/rbaba/tauspin-ip-sv-common-v2/NN')
import reco_data_original_geometry as R  # noqa: E402
import reco_truth_common_v2 as c  # noqa: E402

R.TRUTH_INPUTS_DIR = Path(os.environ['TRUTH_INPUTS_DIR'])
W_MAX = float(os.environ.get('W_MAX', '10'))
STASH = {}
_move, _mse = c.move_batch, torch.nn.functional.mse_loss
CH = torch.tensor([1.0, 1.0, -1.0])
CZ = torch.tensor([0.0, 0.0, 1.0])


def move_batch(batch, device):
    b = _move(batch, device)
    if 'labels' in b and 'h' in b:
        STASH['target'], STASH['labels'] = b['h'], b['labels']
    return b


def mse_loss(pred, target, *args, **kwargs):
    if target is not STASH.get('target') or target.dim() != 3 or args or kwargs:
        return _mse(pred, target, *args, **kwargs)
    lab = STASH['labels'].to(target.dtype)[:, None]
    C = lab * CH.to(target) + (1 - lab) * CZ.to(target)
    s = (target[:, 0] * C * target[:, 1]).sum(-1)
    w = torch.clamp(1.0 / torch.clamp(1.0 + s, min=1e-6), max=W_MAX)
    w = w / w.mean()
    per = ((pred - target) ** 2).mean(dim=(1, 2))
    return (w * per).mean()


c.move_batch = move_batch
torch.nn.functional.mse_loss = mse_loss
print(f'spin-flat weighted MSE, W_MAX={W_MAX}', flush=True)
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher' / 'train_point_h.py'), run_name='__main__')
