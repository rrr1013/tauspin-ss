"""What tau direction resolution a decay-vertex measurement can physically give.

The resolution curve says how well the tau flight direction has to be measured.
This turns that requirement into a length: a tau that flies `L` before decaying,
whose decay vertex is measured with position resolution `sigma_x` per transverse
coordinate, gives a direction resolution of about `sigma_x / L`.

This is a geometric scaling argument on the truth kinematics of this cohort, not
a detector simulation and not an ATLAS performance figure.  The primary vertex
is taken as exact, the vertex fit is taken as unbiased and isotropic, and the
decay length is drawn from the proper-time exponential of each event's own boost
because the truth surface stores momenta, not vertices.  Only the three-prong
case is treated: there the three tracks define a vertex directly.  The one-prong
impact-parameter geometry is a different construction and is not covered here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nullspace as ns

TAU_CTAU_UM = 87.03          # PDG mean decay length c*tau in micrometres
MODE_NAMES = {0: '1p0n', 1: '1p1n', 3: '3p0n'}
VERTEX_SIGMA_UM = (10.0, 20.0, 30.0, 50.0, 100.0)
ANGLE_TARGETS_MRAD = (1.0, 2.0, 3.0, 5.0, 10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    surface = ns.load_surface(args.surface)
    tau = surface['tau']                                     # (R,2,4)
    momentum = np.linalg.norm(tau[..., :3], axis=-1)
    mass = np.sqrt(np.maximum(ns.minkowski_dot(tau, tau), 1e-12))
    boost = momentum / mass                                  # beta gamma
    mean_length_um = boost * TAU_CTAU_UM
    rng = np.random.default_rng(args.seed)
    length_um = rng.exponential(mean_length_um)

    modes = surface['modes']
    report: dict[str, object] = {
        'ctau_um': TAU_CTAU_UM,
        'seed': args.seed,
        'sides': int(modes.size),
        'note': ('Decay lengths are drawn from each side own proper-time exponential; '
                 'the truth surface stores momenta, not vertices.  Primary vertex exact, '
                 'vertex fit unbiased and isotropic.  Three-prong only.'),
        'boost_quantiles': np.quantile(boost, [0.05, 0.5, 0.95]).tolist(),
        'tau_momentum_GeV_quantiles': np.quantile(momentum, [0.05, 0.5, 0.95]).tolist(),
        'mean_decay_length_um_quantiles': np.quantile(
            mean_length_um, [0.05, 0.5, 0.95]).tolist(),
        'sampled_decay_length_um_quantiles': np.quantile(
            length_um, [0.05, 0.5, 0.95]).tolist(),
        'by_mode': {},
    }
    for code, name in MODE_NAMES.items():
        side = modes == code
        if side.sum() < 200:
            continue
        entry: dict[str, object] = {
            'sides': int(side.sum()),
            'boost_median': float(np.median(boost[side])),
            'mean_decay_length_um_quantiles': np.quantile(
                mean_length_um[side], [0.05, 0.5, 0.95]).tolist(),
            'resolution': {},
        }
        for sigma_x in VERTEX_SIGMA_UM:
            angle_mrad = 1e3 * sigma_x / np.maximum(length_um[side], 1e-6)
            entry['resolution'][f'{sigma_x:g}um'] = {
                'sigma_alpha_mrad_quantiles': np.quantile(
                    angle_mrad, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                'fraction_below': {f'{target:g}mrad': float((angle_mrad < target).mean())
                                   for target in ANGLE_TARGETS_MRAD},
            }
        report['by_mode'][name] = entry

    (args.output / 'vertex_geometry.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
