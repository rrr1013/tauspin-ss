"""Implementation checks for the null-space construction.

These are closure tests against the existing canonical artifacts, not physics
results.  They must all pass before any study script is trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import nullspace as ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=4000)
    parser.add_argument('--nn-root', type=Path, default=ns.DEFAULT_NN_ROOT)
    args = parser.parse_args()

    surface = ns.load_surface(args.surface)
    polarimeter, origin = ns.load_polarimeter(args.nn_root)
    rows = slice(0, args.rows)
    truth_nu = surface['nu4'][rows][..., :3]
    report: dict[str, object] = {'hybrid_polarimeter': str(origin), 'rows': args.rows}

    # 1. Truth neutrinos must reproduce the stored canonical exact h.
    result = ns.evaluate_h(polarimeter, surface, truth_nu[None], rows)
    h = result['h'][0]
    reference = surface['h_ref'][rows].astype(np.float64)
    valid = result['valid'][0]
    report['valid_fraction'] = float(valid.mean())
    report['h_parity_max_abs'] = float(np.max(abs(h[valid] - reference[valid])))
    report['h_truth_basis_identity_max_abs'] = float(
        np.max(abs(result['h_truth_basis'][0][valid] - h[valid])))
    report['failure_bits'] = {str(int(bit)): int(count) for bit, count
                              in zip(*np.unique(result['failures'][0], return_counts=True))}

    # 2. The azimuthal family preserves the tau mass shell and the massless neutrino.
    visible = surface['visible'][rows]
    delta = np.linspace(0.0, 2.0 * np.pi, 9)[:, None, None]
    family = ns.azimuth_family(visible[None], truth_nu[None], delta)
    family4 = np.concatenate((family, np.linalg.norm(family, axis=-1)[..., None]), axis=-1)
    tau = visible[None] + family4
    report['family_tau_mass_max_deviation_GeV'] = float(
        np.max(abs(ns.invariant_mass(tau) - ns.invariant_mass(surface['tau'][rows])[None])))
    report['family_nu_mass_squared_max_abs'] = float(np.max(abs(ns.minkowski_dot(family4, family4))))
    report['family_delta0_max_abs'] = float(np.max(abs(family[0] - truth_nu)))
    report['family_delta2pi_max_abs'] = float(np.max(abs(family[-1] - truth_nu)))

    # 3. The mass-shell solution for nu_z must contain the truth solution.
    tau_mass = ns.invariant_mass(surface['tau'][rows])
    solved = ns.solve_nu_z(visible, truth_nu[..., :2], tau_mass)
    closest = np.min(abs(solved['nz'] - truth_nu[..., 2:3]), axis=-1)
    report['nu_z_recovery_max_abs_GeV'] = float(np.max(closest))
    report['nu_z_both_roots_valid_fraction'] = float(solved['valid'].all(axis=-1).mean())
    report['nu_z_no_root_valid_fraction'] = float((~solved['valid']).all(axis=-1).mean())

    # 4. Knowing the tau direction exactly must close the side.
    direction = surface['tau'][rows][..., :3]
    direction = direction / np.linalg.norm(direction, axis=-1)[..., None]
    closed = ns.close_side_from_direction(visible, direction, tau_mass)
    gap = np.linalg.norm(closed['nu'] - truth_nu[..., None, :], axis=-1)
    gap = np.where(closed['valid'], gap, np.inf)
    report['direction_closure_max_abs_GeV'] = float(np.max(np.min(gap, axis=-1)))
    report['direction_closure_valid_roots_mean'] = float(closed['valid'].sum(axis=-1).mean())

    # 5. h must actually move along the family (otherwise the study is vacuous).
    many = ns.evaluate_h(polarimeter, surface, ns.azimuth_family(
        visible[None], truth_nu[None], np.linspace(0.0, 2.0 * np.pi, 17)[:, None, None]), rows)
    good = many['valid'].all(axis=0)
    swing = many['h'][:, good] - many['h'][:1, good]
    report['family_h_max_component_swing'] = np.max(abs(swing), axis=(0, 1)).tolist()

    print(json.dumps(report, indent=2))
    failures = [
        report['h_parity_max_abs'] > 3e-6,
        report['family_tau_mass_max_deviation_GeV'] > 1e-9,
        report['family_nu_mass_squared_max_abs'] > 1e-7,
        report['family_delta0_max_abs'] > 1e-12,
        report['nu_z_recovery_max_abs_GeV'] > 1e-6,
        report['direction_closure_max_abs_GeV'] > 1e-3,
        report['h_truth_basis_identity_max_abs'] > 3e-8,
    ]
    if any(failures):
        raise SystemExit('closure checks failed: ' + repr(failures))
    print('all closure checks passed')


if __name__ == '__main__':
    main()
