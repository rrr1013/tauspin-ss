"""p11 point-h training (hz_beyond_ceiling) with the h targets read from another directory.

Everything else - inputs, geometry channels, architecture, seed, 50-epoch recipe,
checkpoint selection - is the unchanged p11_train_full_geometry.py path.

env: H_TARGETS_DIR (directory with train.npz / validation.npz), P11_DIR (where
     p11_train_full_geometry.py lives), plus the p11 variables GEO_DIM,
     GEO_FEATURES, GEO_KEY, SEED.
usage: python train_point_h.py --arm {geo16|base16} --output-dir OUT
"""
import json
import os
import sys
from pathlib import Path

TARGETS = Path(os.environ['H_TARGETS_DIR'])
sys.path.insert(0, os.environ['P11_DIR'])

import p11_train_full_geometry as P  # noqa: E402

Base = P.T.RecoHDataset


class TargetsOverride(Base):
    def __init__(self, split, *args, **kwargs):
        kwargs['targets_dir'] = TARGETS
        super().__init__(split, *args, **kwargs)


P.T.RecoHDataset = TargetsOverride

if __name__ == '__main__':
    P.T.main()
    out = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    (out / 'targets.json').write_text(json.dumps({'h_targets_dir': str(TARGETS)}, indent=2))
