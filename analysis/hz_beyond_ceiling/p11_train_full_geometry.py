"""Parity-preserving point-h training with GEO_DIM (16 or 22) appended local-frame geometry channels.

Runs the unchanged ``train_original_geometry.py`` loop (full current reco
input), but widens the appended tau block from 7 to 16 channels and fills it
with ``p10_geometry_full.py`` features selected by GEO_KEY.  The new tau
projector columns are initialised by the parent class exactly as before; the
original ten columns are copied from the unwidened network.

env: GEO_KEY (full, ip3, sv, full_shuffle), GEO_FEATURES (npz),
     SEED (optional, default 42)
usage: python p11_train_full_geometry.py --arm {geo16|base16} --output-dir OUT [--epochs N]
"""
import os
import sys
from pathlib import Path

CODE = Path("/home/rbaba/tauspin-ip-sv-common-v2/NN/ip_sv_ablation")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent))

import numpy as np
import torch

import reco_truth_common_v2 as c
import models_original_geometry as M

G = int(os.environ.get('GEO_DIM', '16'))
M.GEOMETRY_DIM = G
M.TAU_DIM = M.OLD_TAU_DIM + G
M.ARM_MASKS["geo16"] = (1,) * M.TAU_DIM
M.ARM_MASKS["base16"] = (1,) * M.OLD_TAU_DIM + (0,) * G  # current full reco, geometry hidden
if "SEED" in os.environ:
    c.SEED = int(os.environ["SEED"])

import train_original_geometry as T  # noqa: E402  (imports the patched module objects)

T.ARMS = tuple(T.ARMS) + ("geo16", "base16")
KEY = os.environ["GEO_KEY"]
FEATURES = np.load(os.environ["GEO_FEATURES"])
Base = T.RecoHDataset


class Geo16Dataset(Base):
    def __init__(self, split, *args, **kwargs):
        super().__init__(split, *args, **kwargs)
        ids = self.global_index.numpy()
        src = FEATURES[f"{split}_global_indices"]
        pos = {int(v): i for i, v in enumerate(src)}
        rows = np.asarray([pos[int(v)] for v in ids])
        geo = np.asarray(FEATURES[f"{split}_{KEY}"], dtype=np.float32)[rows]
        if geo.shape[-1] != G or geo.shape[:-1] != tuple(self.geometry_tau.shape[:-1]):
            raise RuntimeError(f"geometry shape {geo.shape} vs {tuple(self.geometry_tau.shape)}")
        self.geometry_tau = torch.as_tensor(geo)


T.RecoHDataset = Geo16Dataset

if __name__ == "__main__":
    T.main()
