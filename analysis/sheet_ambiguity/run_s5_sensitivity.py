"""Sensitivity checks on the oracle ladder, from the saved per-event scores.

Three things a reader should be allowed to doubt:

  * the sheet label is meaningless where the two mass-shell roots nearly merge,
  * the strict cohort throws away events where one root was already impossible,
  * the two sides are supposed to be interchangeable, so O2a and O2b should agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'met_nullspace'))

import run_m2_report as m2

MODE_NAMES = m2.MODE_NAMES


def auc(labels, score, mask) -> float:
    return m2.weighted_auc(labels[mask], score[mask], np.ones(int(mask.sum())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    with np.load(args.output / 'scores.npz') as handle:
        data = {key: handle[key] for key in handle.files}

    labels = data['labels']
    mask = data['mask']
    separation = data['root_separation']
    o0 = data['score_flat_o0_likelihood_ratio']
    o1 = data['score_flat_o1_likelihood_ratio']
    o3 = data['score_o3_likelihood_ratio']
    o4 = data['score_o4_likelihood_ratio']

    report: dict[str, object] = {}

    # 1. Drop events where the two roots nearly merge on either side.
    cuts = {}
    for threshold in (0.0, 0.1, 0.5, 1.0):
        keep = mask & (separation >= threshold).all(axis=1)
        base, oracle = auc(labels, o0, keep), auc(labels, o1, keep)
        cuts[f'{threshold:g}GeV'] = {
            'events': int(keep.sum()),
            'o0': base, 'o1': oracle, 'o3': auc(labels, o3, keep), 'o4': auc(labels, o4, keep),
            'gain': oracle - base,
            'f_discrete': (oracle - base) / (auc(labels, o4, keep) - base),
        }
    report['root_separation_cut'] = cuts

    # 2. The two sides should be interchangeable.
    side = m2.bootstrap_auc_difference(
        labels, data['score_flat_o2a_likelihood_ratio'],
        data['score_flat_o2b_likelihood_ratio'], mask,
        draws=args.bootstrap, seed=args.seed)
    report['o2a_minus_o2b'] = side

    # 3. Does the gain come from the events the geometry says it should?
    angle = 1e3 * data['root_angle'].min(axis=1)
    bins = np.quantile(angle[mask], np.linspace(0, 1, 6))
    profile = []
    for start, stop in zip(bins[:-1], bins[1:]):
        inside = mask & (angle >= start) & (angle < stop)
        if inside.sum() < 500:
            continue
        base, oracle = auc(labels, o0, inside), auc(labels, o1, inside)
        profile.append({'angle_mrad_low': float(start), 'angle_mrad_high': float(stop),
                        'events': int(inside.sum()), 'o0': base, 'o1': oracle,
                        'gain': oracle - base})
    report['gain_vs_root_angle'] = profile

    (args.output / 'sensitivity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
