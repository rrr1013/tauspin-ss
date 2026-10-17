"""Training-composition control: copies of the gen3pi targets whose TRAIN h_valid is restricted.

  Honly    train rows with label H only
  Zonly    train rows with label Z only
  mixhalf  a random subset of all train rows (both labels), of size (n_H + n_Z) / 2, seed 20261018

The oracle truth-input file that the dataset checks row by row against the kept rows
(reco_data_original_geometry.TRUTH_INPUTS_DIR, ladder inputs v1) is written per subset as <name>/truth_inputs/;
s5_train.py points the dataset there.  VALIDATION is copied unchanged, so every arm predicts the same 59,390 validation events.  The point-h
training code (gen3pi_teacher/train_point_h.py -> hz_beyond_ceiling p11) keeps only h_valid rows, so the
architecture, inputs, recipe and seed are unchanged and only the training composition differs.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/targets')
LADDER = Path('/home/rbaba/tauspin-full-reco-information-ladder-20260917-inputs-v1')
TRUTH_KEYS = ('global_indices', 'truth_visible_tau_lab4', 'truth_neutrino_lab4')


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
    with np.load(LADDER / 'train.npz') as d:
        lt = {k: d[k] for k in TRUTH_KEYS}
    with np.load(LADDER / 'validation.npz') as d:
        lv = {k: d[k] for k in TRUTH_KEYS}
    pos = {int(g): i for i, g in enumerate(lt['global_indices'])}
    for name, keep in subsets.items():
        out = a.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'train.npz', **{**tr, 'h_valid': keep})
        shutil.copy2(SRC / 'validation.npz', out / 'validation.npz')
        (out / 'truth_inputs').mkdir(exist_ok=True)
        rows = np.array([pos[int(g)] for g in tr['global_indices'][keep]])
        np.savez(out / 'truth_inputs' / 'train.npz', **{k: v[rows] for k, v in lt.items()})
        np.savez(out / 'truth_inputs' / 'validation.npz', **lv)
        summary[name] = {'rows': int(keep.sum()), 'H': int((keep & (lab == 1)).sum()), 'Z': int((keep & (lab == 0)).sum())}
    (a.output_dir / 'subsets.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
