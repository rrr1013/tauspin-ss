"""Training-composition control: copies of the gen3pi targets whose TRAIN h_valid is restricted.

  Honly    train rows with label H only
  Zonly    train rows with label Z only
  mixhalf  a random subset of all train rows (both labels), of size (n_H + n_Z) / 2, seed 20261018

VALIDATION is copied unchanged, so every arm predicts the same 59,390 validation events.  The point-h
training code (gen3pi_teacher/train_point_h.py -> hz_beyond_ceiling p11) keeps only h_valid rows, so the
architecture, inputs, recipe and seed are unchanged and only the training composition differs.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/targets')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    with np.load(SRC / 'train.npz') as d:
        tr = {k: d[k] for k in d.files}
    valid, lab = tr['h_valid'].astype(bool), tr['labels']
    nH, nZ = int((valid & (lab == 1)).sum()), int((valid & (lab == 0)).sum())
    rng = np.random.default_rng(20261018)
    idx = np.flatnonzero(valid)
    half = np.zeros_like(valid)
    half[rng.choice(idx, (nH + nZ) // 2, replace=False)] = True
    subsets = {'Honly': valid & (lab == 1), 'Zonly': valid & (lab == 0), 'mixhalf': half}
    summary = {'n_H': nH, 'n_Z': nZ}
    for name, keep in subsets.items():
        out = a.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'train.npz', **{**tr, 'h_valid': keep})
        shutil.copy2(SRC / 'validation.npz', out / 'validation.npz')
        summary[name] = {'rows': int(keep.sum()), 'H': int((keep & (lab == 1)).sum()), 'Z': int((keep & (lab == 0)).sum())}
    (a.output_dir / 'subsets.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
