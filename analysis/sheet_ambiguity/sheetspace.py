"""The discrete part of the tau-pair null space: which mass-shell root.

The MET-constrained solution set is two dimensional but not connected in the
way a two-dimensional set usually is.  Above every allowed transverse momentum
`nu_t` each side's mass shell is a quadratic with two roots, so the set is four
sheets glued along the curve where the discriminants vanish.  The previous run
found that about 40% of the variance of the pair statistic `T` on that set sits
*between* the sheets rather than within them, which makes the discrete choice a
separate object worth resolving on its own.

This module adds only what that question needs on top of `metspace`:

  * a globally consistent sheet label, and the truth sheet,
  * the tau flight direction of a hypothesis,
  * the angular separation between the two roots of one side,
  * the likelihood of an idealised tau-direction measurement.

Root convention
---------------
`nullspace.solve_nu_z` returns `nz = (A vz +- E_vis sqrt(disc)) / a` with the
`+` branch first.  Since `a = E_vis^2 - vz^2 > 0`, root index 0 is always the
larger `nz`, everywhere on the allowed region.  The label is therefore global,
not a local branch choice, and the sheet index

    sheet = 2 * root(side 0) + root(side 1)

matches the ordering used by `metspace.solution_sheets` and
`run_m1_moments.build_sheets`, which is `((0,0),(0,1),(1,0),(1,1))`.

Conventions follow the rest of the project: four-vectors are (px,py,pz,E) in
GeV, side 0 is tau- and side 1 is tau+.
"""
from __future__ import annotations

import numpy as np

import nullspace as ns

UPSTREAM_TAU_MASS = ns.UPSTREAM_TAU_MASS

SHEETS = ((0, 0), (0, 1), (1, 0), (1, 1))


def sheet_index(root0: np.ndarray, root1: np.ndarray) -> np.ndarray:
    """Root indices per side -> the sheet index used by the moment accumulator."""
    return 2 * np.asarray(root0, dtype=np.int64) + np.asarray(root1, dtype=np.int64)


def truth_sheet(visible: np.ndarray, truth_nu: np.ndarray,
                tau_mass: float = UPSTREAM_TAU_MASS) -> dict[str, np.ndarray]:
    """Which root each side's truth neutrino sits on, and how well defined that is.

    `visible` (R,2,4), `truth_nu` (R,2,3).  The truth transverse momentum is fed
    back into the mass shell and the root closest to the truth `nu_z` is taken.
    The two residuals and the root separation are returned as well: where the
    roots nearly merge the label is not meaningful, and that fraction has to be
    reported rather than hidden.
    """
    nu_t = truth_nu[..., :2]
    solved = ns.solve_nu_z(visible, nu_t, np.asarray(tau_mass, dtype=np.float64))
    nz = solved['nz']                                        # (R,2,2roots)
    residual = np.abs(nz - truth_nu[..., 2, None])
    root = np.argmin(residual, axis=-1)                      # (R,2)
    chosen = np.take_along_axis(residual, root[..., None], axis=-1)[..., 0]
    other = np.take_along_axis(residual, 1 - root[..., None], axis=-1)[..., 0]
    return {
        'root': root,
        'sheet': sheet_index(root[:, 0], root[:, 1]),
        'residual_chosen': chosen,
        'residual_other': other,
        'root_separation': np.abs(nz[..., 0] - nz[..., 1]),
        'both_roots_valid': solved['valid'].all(axis=-1),
        'nz_roots': nz,
        'discriminant': solved['discriminant'],
    }


def tau_direction(visible: np.ndarray, nu_xyz: np.ndarray) -> np.ndarray:
    """Unit tau flight direction of a hypothesis.  (...,2,4),(...,2,3) -> (...,2,3)."""
    momentum = visible[..., :3] + nu_xyz
    return momentum / np.maximum(np.linalg.norm(momentum, axis=-1)[..., None], 1e-300)


def opening_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between two unit vectors, accurate at small angles."""
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def tangent_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane orthogonal to `direction`."""
    seed = np.zeros(direction.shape)
    axis = np.argmin(np.abs(direction), axis=-1)
    np.put_along_axis(seed, axis[..., None], 1.0, axis=-1)
    first = np.cross(direction, seed)
    first = first / np.maximum(np.linalg.norm(first, axis=-1)[..., None], 1e-300)
    second = np.cross(direction, first)
    return first, second


def measured_direction(truth_direction: np.ndarray, noise: np.ndarray,
                       sigma: float) -> np.ndarray:
    """Idealised direction measurement: truth tilted by `sigma * noise`.

    `noise` (...,2) are standard normal draws in the tangent plane, fixed once
    per event and reused for every `sigma`, so the resolution curve is a common
    random numbers scan rather than an independent measurement per point.  For
    small `sigma` the angular displacement is `sigma * |noise|`, i.e. an
    isotropic Gaussian of width `sigma` per transverse component.
    """
    first, second = tangent_frame(truth_direction)
    tilted = truth_direction + sigma * (noise[..., 0, None] * first
                                        + noise[..., 1, None] * second)
    return tilted / np.maximum(np.linalg.norm(tilted, axis=-1)[..., None], 1e-300)


def direction_log_likelihood(hypothesis_direction: np.ndarray,
                             measurement: np.ndarray, sigma: float) -> np.ndarray:
    """log p(measurement | hypothesis) for both sides, summed.

    A tau direction measured with isotropic resolution `sigma` per transverse
    component gives, in the small-angle plane, a two-dimensional Gaussian whose
    log density is `-angle^2 / (2 sigma^2)` up to a constant that cancels in the
    normalised posterior.  `hypothesis_direction` (...,2,3), `measurement`
    broadcastable to it.
    """
    angle = opening_angle(hypothesis_direction, measurement)
    return -0.5 * np.sum(angle * angle, axis=-1) / (sigma * sigma)
