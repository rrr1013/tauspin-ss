"""Merge row-chunked s3 outputs and export per-measure score files for s2_auc.

usage: python s4_merge.py --inputs chunk_*.npz --output merged.npz --scores-dir DIR --tag NAME
Writes merged.npz (all arrays concatenated in row order) and DIR/<tag>_<measure>.npz with
``global_indices`` and ``scores`` = log likelihood ratio (NaN where no hypothesis was valid).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scores-dir', type=Path, required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    def start_of(path):
        m = re.search(r'_r(\d+)\.npz$', path.name)
        return int(m.group(1)) if m else 0
    files = sorted(a.inputs, key=start_of)
    parts = [dict(np.load(f)) for f in files]
    merged = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    ids = merged['global_indices']
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError('duplicate rows across chunks')
    np.savez_compressed(a.output, **merged)
    a.scores_dir.mkdir(parents=True, exist_ok=True)
    info = {'rows': int(len(ids))}
    for key in merged:
        if key.startswith('log_lr_'):
            measure = key[len('log_lr_'):]
            np.savez_compressed(a.scores_dir / f'{a.tag}_{measure}.npz', global_indices=ids,
                                scores=merged[key])
            info[measure] = {'nan': int(np.isnan(merged[key]).sum()),
                             'ess_median': float(np.median(merged[f'ess_{measure}']))}
    print(json.dumps(info))


if __name__ == '__main__':
    main()
