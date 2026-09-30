"""Point-h training for the simple-vs-ATLAS IP/SV comparison, one arm per call.

Runs the unchanged hz_beyond_ceiling p11 loop (train_original_geometry.py: TauSpinTransformer
point-h regression, seed, 50 epochs, cosine schedule, rolling-3 validation h-MSE checkpoint
selection) with
  * the common reduced input: tau channels 6-9 and track channels 5-14 are zeroed, which is
    exactly the schema of the simple-smearing pack (those channels are zero there; for ATLAS they
    hold the ntuple d0/z0 and other detector variables),
  * the 22-channel geometry block GEO_KEY from GEO_FEATURES appended to each tau token,
  * h targets with the generator-current 3pi (ATLAS: gen3pi_teacher targets; simple: h_gen of
    build_simple_inputs.py).

env: SAMPLE (atlas|simple), GEO_FEATURES, GEO_KEY (none|legacy|lab|local|local_shuffle), SEED
usage: python train_ipsv.py --arm geo16 --output-dir OUT
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('GEO_DIM', '22')
SAMPLE = os.environ['SAMPLE']
P11_DIR = Path('/home/rbaba/hz-beyond-ceiling-20260923/scripts')
sys.path.insert(0, str(P11_DIR))
import p11_train_full_geometry as P  # noqa: E402

T = P.T
Geo = P.Geo16Dataset
TAU_REDUCED = slice(6, 10)
TRACK_REDUCED = slice(5, 15)
ATLAS_TARGETS = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/targets')


def reduce_event(event):
    tau = event['tau_features'].clone()
    tau[:, TAU_REDUCED] = 0
    track = event['track_features'].clone()
    track[:, TRACK_REDUCED] = 0
    event['tau_features'] = tau
    event['track_features'] = track
    return event


class AtlasReduced(Geo):
    def __init__(self, split, *args, **kwargs):
        kwargs['targets_dir'] = ATLAS_TARGETS
        super().__init__(split, *args, **kwargs)

    def __getitem__(self, index):
        return reduce_event(super().__getitem__(index))


class SimpleReduced(torch.utils.data.Dataset):
    """Simple-smearing pack rows with h_gen targets; same item keys as RecoHDataset."""
    DATA = Path('/home/rbaba/simple-smearing-20260910/data/pack')

    def __init__(self, split, *args, **kwargs):
        inp = np.load(os.environ['SIMPLE_INPUTS'])
        ids = inp[f'{split}_global_indices']
        pack = torch.load(self.DATA / f'{split}.pt', map_location='cpu', weights_only=True)
        pos = {int(v): i for i, v in enumerate(pack['global_indices'].numpy())}
        self._rows = np.asarray([pos[int(v)] for v in ids], dtype=np.int64)
        self._p = pack
        self.labels = torch.as_tensor(inp[f'{split}_labels'], dtype=torch.int64)
        if not torch.equal(self.labels, pack['labels'][self._rows]):
            raise RuntimeError('label identity mismatch')
        self.h = torch.as_tensor(inp[f'{split}_h'], dtype=torch.float32)
        if not torch.isfinite(self.h).all():
            raise RuntimeError('nonfinite h target')
        self.truth_modes = torch.as_tensor(inp[f'{split}_modes'], dtype=torch.int64)
        self.overlap_weight = torch.ones(len(ids))
        self.global_index = torch.as_tensor(ids, dtype=torch.int64)
        self.file_index = pack['file_indices'][self._rows].long()
        self.entry_index = pack['entry_indices'][self._rows].long()
        self.truth_tau_direction = torch.as_tensor(inp[f'{split}_truth_tau_direction'])
        geo = np.load(os.environ['GEO_FEATURES'])
        if not np.array_equal(geo[f'{split}_global_indices'], ids):
            raise RuntimeError('geometry identity mismatch')
        self.geometry_tau = torch.as_tensor(geo[f'{split}_{os.environ["GEO_KEY"]}'], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        p, r = self._p, int(self._rows[index])
        ts, te = int(p['reco_track_offsets'][r]), int(p['reco_track_offsets'][r + 1])
        ps, pe = int(p['reco_pfo_offsets'][r]), int(p['reco_pfo_offsets'][r + 1])
        return reduce_event({
            'event_features': p['reco_event_features'][r],
            'tau_features': torch.cat((p['reco_tau_features'][r], self.geometry_tau[index]), dim=-1),
            'tau_decay_mode': p['reco_tau_decay_mode'][r],
            'track_features': p['reco_track_features'][ts:te],
            'track_sides': p['reco_track_sides'][ts:te],
            'pfo_features': p['reco_pfo_features'][ps:pe],
            'pfo_sides': p['reco_pfo_sides'][ps:pe],
            'label': self.labels[index],
            'event_number': p['event_numbers'][r],
            'h': self.h[index],
            'truth_modes': self.truth_modes[index],
            'overlap_weight': self.overlap_weight[index],
            'global_index': self.global_index[index],
            'file_index': self.file_index[index],
            'entry_index': self.entry_index[index],
            'truth_tau_direction': self.truth_tau_direction[index],
        })


T.RecoHDataset = AtlasReduced if SAMPLE == 'atlas' else SimpleReduced

if __name__ == '__main__':
    T.main()
