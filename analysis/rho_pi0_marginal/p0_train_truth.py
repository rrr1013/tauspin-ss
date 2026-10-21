"""Truth pi+- / pi0 / nu of every TRAIN event with a truth-rho side, re-read from the ROOT files.

Used only to fit the reco pi0 response (likelihood of the reco pi0 given the true pi0) and the rho line shape check.
Rows: gen3pi targets TRAIN, h_valid & any side with truth mode 1.  Output keyed by global_indices.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gen3pi_teacher'))
from build_targets import read_truth  # noqa: E402

GEN = Path('/home/rbaba/gen3pi-teacher-20260924/artifacts/targets')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    t = np.load(GEN / 'train.npz')
    v = t['h_valid'].astype(bool)
    rows = np.flatnonzero(v & (t['modes'] == 1).any(1))
    tr = read_truth(t['labels'][rows], t['file_indices'][rows], t['entry_indices'][rows], t['event_numbers'][rows],
                    t['modes'][rows])
    np.savez_compressed(a.output, global_indices=t['global_indices'][rows], modes=t['modes'][rows],
                        labels=t['labels'][rows], pi_lead=tr['pions'][:, :, 0], pi0=tr['pi0'], nu4=tr['nu4'])
    print(json.dumps({'rows': int(len(rows)), 'elapsed_s': time.time() - t0}), flush=True)


if __name__ == '__main__':
    main()
