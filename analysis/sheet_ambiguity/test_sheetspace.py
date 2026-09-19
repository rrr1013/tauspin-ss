"""Kinematic checks for the sheet labelling.  Needs no ROOT and no cohort."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'azimuth_nullspace'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nullspace as ns
import sheetspace as ss

TAU_MASS = ss.UPSTREAM_TAU_MASS


def synthetic_events(rows: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Exactly on-shell tau decays: a massive visible system plus a massless nu.

    Built in the tau rest frame and boosted, so the tau mass shell and the
    neutrino masslessness hold to machine precision by construction.
    """
    rng = np.random.default_rng(seed)
    vis_mass = rng.uniform(0.2, 1.5, size=(rows, 2))
    energy_star = (TAU_MASS ** 2 + vis_mass ** 2) / (2.0 * TAU_MASS)
    momentum_star = (TAU_MASS ** 2 - vis_mass ** 2) / (2.0 * TAU_MASS)
    cos_star = rng.uniform(-1.0, 1.0, size=(rows, 2))
    phi_star = rng.uniform(0.0, 2.0 * np.pi, size=(rows, 2))
    sin_star = np.sqrt(1.0 - cos_star ** 2)
    unit = np.stack((sin_star * np.cos(phi_star), sin_star * np.sin(phi_star), cos_star), axis=-1)
    vis_star = np.concatenate((momentum_star[..., None] * unit, energy_star[..., None]), axis=-1)
    nu_star = np.concatenate((-momentum_star[..., None] * unit, momentum_star[..., None]), axis=-1)

    tau_p = rng.normal(0.0, 40.0, size=(rows, 2, 3))
    tau_p[..., 2] += rng.normal(0.0, 150.0, size=(rows, 2))
    tau_e = np.sqrt(np.sum(tau_p ** 2, axis=-1) + TAU_MASS ** 2)
    beta = -tau_p / tau_e[..., None]
    return ns.boost(vis_star, beta), ns.boost(nu_star, beta)[..., :3]


def test_truth_sheet_recovers_the_truth_root() -> None:
    visible, truth_nu = synthetic_events(4000)
    label = ss.truth_sheet(visible, truth_nu, TAU_MASS)
    assert label['residual_chosen'].max() < 1e-5, label['residual_chosen'].max()
    assert label['both_roots_valid'].all()
    assert label['sheet'].min() >= 0 and label['sheet'].max() <= 3
    # The label agrees with the sign convention: root 0 is the larger nz.
    larger = (truth_nu[..., 2] >= label['nz_roots'].mean(axis=-1))
    assert np.array_equal(label['root'] == 0, larger)


def test_sheet_index_matches_the_accumulator_ordering() -> None:
    for index, (root0, root1) in enumerate(ss.SHEETS):
        assert ss.sheet_index(root0, root1) == index
        assert index // 2 == root0 and index % 2 == root1


def test_tangent_frame_is_orthonormal() -> None:
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(5000, 3))
    direction /= np.linalg.norm(direction, axis=-1)[..., None]
    first, second = ss.tangent_frame(direction)
    for vector in (first, second):
        assert np.abs(np.linalg.norm(vector, axis=-1) - 1.0).max() < 1e-12
        assert np.abs(np.sum(vector * direction, axis=-1)).max() < 1e-12
    assert np.abs(np.sum(first * second, axis=-1)).max() < 1e-12


def test_measured_direction_has_the_requested_width() -> None:
    rng = np.random.default_rng(11)
    direction = rng.normal(size=(200000, 3))
    direction /= np.linalg.norm(direction, axis=-1)[..., None]
    noise = rng.normal(size=(200000, 2))
    for sigma in (1e-5, 1e-4, 1e-3):
        measured = ss.measured_direction(direction, noise, sigma)
        angle = ss.opening_angle(direction, measured)
        # Rayleigh with scale sigma: mean = sigma sqrt(pi/2), rms = sigma sqrt(2).
        assert abs(angle.mean() / (sigma * np.sqrt(np.pi / 2)) - 1.0) < 0.01
        assert abs(np.sqrt((angle ** 2).mean()) / (sigma * np.sqrt(2.0)) - 1.0) < 0.01


def test_direction_likelihood_peaks_at_the_truth() -> None:
    visible, truth_nu = synthetic_events(2000, seed=19)
    truth_direction = ss.tau_direction(visible, truth_nu)
    zero = ss.direction_log_likelihood(truth_direction, truth_direction, 1e-3)
    assert np.abs(zero).max() < 1e-20
    rng = np.random.default_rng(5)
    other = ss.tau_direction(visible, truth_nu + rng.normal(0.0, 1.0, truth_nu.shape))
    assert (ss.direction_log_likelihood(other, truth_direction, 1e-3) < zero).all()


def test_truth_neutrino_sits_on_a_sheet_of_the_met_constrained_set() -> None:
    """The truth solution is one of the four sheets above its own transverse point."""
    visible, truth_nu = synthetic_events(3000, seed=23)
    met = truth_nu[:, 0, :2] + truth_nu[:, 1, :2]
    nu_t_pair = np.stack((truth_nu[:, :, :2][:, 0], met - truth_nu[:, :, :2][:, 0]), axis=-2)
    assert np.abs(nu_t_pair - truth_nu[..., :2]).max() < 1e-9
    label = ss.truth_sheet(visible, truth_nu, TAU_MASS)
    nz = np.take_along_axis(label['nz_roots'], label['root'][..., None], axis=-1)[..., 0]
    assert np.abs(nz - truth_nu[..., 2]).max() < 1e-5


if __name__ == '__main__':
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        try:
            function()
            print(f'PASS {name}')
        except AssertionError as error:
            failures += 1
            print(f'FAIL {name}: {error}')
    raise SystemExit(failures)
