"""Parity-preserving point-h ablation with local-frame geometry.

Runs the unchanged 2026-09-20 trainer ``train_original_geometry.py`` (full
current reco input: all 10 tau and 19 track channels, plus seven appended tau
slots).  The model arm is ``ip_sv`` so that all seven appended slots are
visible; their content is replaced by

    [f0, f1, f2, f3, 0, 0, 0]

where f is the 4-vector of ``p3_geometry_features.py`` for the key given in
the environment variable GEO_KEY (ip_pv, ip_legacy, ip_shuffle, oracle).
The baseline is run with the original trainer and arm ``baseline_current``.

usage: GEO_KEY=ip_pv GEO_FEATURES=.../geo_features.npz python p9_train_parity.py \
           --arm ip_sv --output-dir OUT
"""
import os
import sys
from pathlib import Path

CODE = Path("/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

import numpy as np
import torch

import train_original_geometry as T

KEY = os.environ["GEO_KEY"]
FEATURES = np.load(os.environ["GEO_FEATURES"])
Base = T.RecoHDataset


class LocalGeometryDataset(Base):
    def __init__(self, split, *args, **kwargs):
        super().__init__(split, *args, **kwargs)
        ids = self.global_index.numpy()
        src = FEATURES[f"{split}_global_indices"]
        pos = {int(v): i for i, v in enumerate(src)}
        rows = np.asarray([pos[int(v)] for v in ids])
        geo = np.asarray(FEATURES[f"{split}_{KEY}"])[rows]
        if self.geometry_tau.shape[:-1] != geo.shape[:-1] or self.geometry_tau.shape[-1] != 7:
            raise RuntimeError(f"unexpected geometry shape {tuple(self.geometry_tau.shape)}")
        full = np.zeros(tuple(self.geometry_tau.shape), np.float32)
        full[..., :4] = geo
        self.geometry_tau = torch.as_tensor(full)


T.RecoHDataset = LocalGeometryDataset

if __name__ == "__main__":
    T.main()
