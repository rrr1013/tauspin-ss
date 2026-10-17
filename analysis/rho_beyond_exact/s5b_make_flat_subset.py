"""Spin-flat training subset for the composition control (adds <out>/flat next to s5's subsets).

Each TRAIN row (both labels) is accepted with probability a = min(1, 0.5 / (1 + s)), s = h-^T C h+ with the exact h and
C = C_H = diag(1,1,-1) for H, C_Z = diag(0,0,1) for Z.  The exact h are unit vectors, so s is in [-1, 1]; the spin-
correlated density is p0 (1 + s), hence the accepted rows follow p0 (spin-uncorrelated) up to the clip at s < -0.5, and the
expected acceptance is 1/2 -- the same size as the other s5 arms.  Rows whose h is invalid on a side keep a = 0.5.
Seed 20261018.  The oracle truth-input file is written row-aligned as for s5.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_make_subset_targets import LADDER, SRC, TRUTH_KEYS  # noqa: E402

CH, CZ = np.array([1.0, 1.0, -1.0]), np.array([0.0, 0.0, 1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    with np.load(SRC / 'train.npz') as d:
        tr = {k: d[k] for k in d.files}
    valid, lab, h = tr['h_valid'].astype(bool), tr['labels'], tr['h'].astype(float)
    C = np.where(lab[:, None] == 1, CH, CZ)
    s = np.sum(h[:, 0] * C * h[:, 1], -1)
    s = np.where(np.isfinite(s), s, 0.0)
    acc = np.minimum(1.0, 0.5 / np.maximum(1 + s, 1e-9))
    keep = valid & (np.random.default_rng(20261018).random(len(acc)) < acc)
    out = a.output_dir / 'flat'
    (out / 'truth_inputs').mkdir(parents=True, exist_ok=True)
    np.savez(out / 'train.npz', **{**tr, 'h_valid': keep})
    shutil.copy2(SRC / 'validation.npz', out / 'validation.npz')
    with np.load(LADDER / 'train.npz') as d:
        lt = {k: d[k] for k in TRUTH_KEYS}
    with np.load(LADDER / 'validation.npz') as d:
        lv = {k: d[k] for k in TRUTH_KEYS}
    pos = {int(g): i for i, g in enumerate(lt['global_indices'])}
    rows = np.array([pos[int(g)] for g in tr['global_indices'][keep]])
    np.savez(out / 'truth_inputs' / 'train.npz', **{k: v[rows] for k, v in lt.items()})
    np.savez(out / 'truth_inputs' / 'validation.npz', **lv)
    kept = keep[valid]
    chk = {}
    for l, name, Cm in ((1, 'H', CH), (0, 'Z', CZ)):
        m = valid & (lab == l) & np.isfinite(h).all((1, 2))
        for tag, sel in (('before', m), ('after', m & keep)):
            chk[f'{name} {tag} 9<h-h+>'] = [float(9 * np.mean(h[sel, 0, i] * h[sel, 1, i])) for i in range(3)]
    summary = {'rows': int(keep.sum()), 'H': int((keep & (lab == 1)).sum()), 'Z': int((keep & (lab == 0)).sum()),
               'clipped_fraction': float(np.mean((0.5 / np.maximum(1 + s, 1e-9) > 1)[valid])),
               'acceptance': float(kept.mean()), 'spin_correlation_check': chk}
    (out / 'subset.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
