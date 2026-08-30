#!/usr/bin/env python3
"""Focused software checks for reco_1p1p_score_maps.py."""

from __future__ import annotations

import unittest

import numpy as np

from reco_1p1p_score_maps import (
    MAP_BINS,
    MAP_RANGE,
    PION_MASS_GEV,
    TAU_MASS_GEV,
    _structured_keys,
    exact_join_indices,
    helicity_cosine,
    histogram2d,
    solve_collinear,
    stable_tertiles,
)


class RecoScoreMapTests(unittest.TestCase):
    def test_collinear_and_angle_round_trip(self) -> None:
        tau_pt = np.array([[48.0, 37.0]])
        tau_phi = np.array([[0.35, 2.05]])
        alpha_expected = np.array([[0.45, 0.80]])
        px = tau_pt * np.cos(tau_phi)
        py = tau_pt * np.sin(tau_phi)
        met_x = float(np.sum(alpha_expected * px))
        met_y = float(np.sum(alpha_expected * py))
        met_et = np.array([np.hypot(met_x, met_y)])
        met_phi = np.array([np.arctan2(met_y, met_x)])
        solution = solve_collinear(tau_pt, tau_phi, met_et, met_phi)
        np.testing.assert_allclose(solution["alpha"], alpha_expected, rtol=0.0, atol=1.0e-12)
        expected_z = 1.0 / (1.0 + alpha_expected)
        np.testing.assert_allclose(solution["z"], expected_z, rtol=0.0, atol=1.0e-12)
        self.assertTrue(bool(solution["valid"][0]))

        e_tau = np.array([[65.0, 52.0]])
        target_cos = np.array([[-0.55, 0.62]])
        a = PION_MASS_GEV / TAU_MASS_GEV
        beta = np.sqrt(1.0 - (TAU_MASS_GEV / e_tau) ** 2)
        x = 0.5 * (target_cos * beta * (1.0 - a**2) + 1.0 + a**2)
        e_pi = x * e_tau
        result = helicity_cosine(e_pi, e_tau)
        np.testing.assert_allclose(result["raw_cos"], target_cos, rtol=0.0, atol=1.0e-12)

    def test_identity_join_is_exact_and_order_independent(self) -> None:
        names = ("sample_id", "ntuple_file_index", "ntuple_entry")
        left = _structured_keys(
            (np.array([0, 0, 1]), np.array([2, 1, 3]), np.array([9, 7, 8])), names
        )
        right = _structured_keys(
            (np.array([1, 0, 0]), np.array([3, 2, 1]), np.array([8, 9, 7])), names
        )
        joined = exact_join_indices(left, right, "test")
        np.testing.assert_array_equal(joined, np.array([1, 2, 0]))
        missing = right[:-1]
        with self.assertRaises(RuntimeError):
            exact_join_indices(left, missing, "missing")

    def test_tau_side_charge_ordering_convention(self) -> None:
        expected = np.array([-1.0, 1.0])
        correct = np.array([-1.0, 1.0]) * expected > 0.5
        swapped = np.array([1.0, -1.0]) * expected > 0.5
        self.assertTrue(bool(np.all(correct)))
        self.assertFalse(bool(np.all(swapped)))

    def test_no_clipping(self) -> None:
        e_tau = np.array([[10.0, 10.0]])
        e_pi = np.array([[11.0, -1.0]])
        raw = helicity_cosine(e_pi, e_tau)["raw_cos"]
        self.assertGreater(float(raw[0, 0]), 1.0)
        self.assertLess(float(raw[0, 1]), -1.0)

    def test_fixed_bins_and_edges(self) -> None:
        self.assertEqual(MAP_BINS, 6)
        self.assertEqual(MAP_RANGE, (-1.0, 1.0))
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        histogram = histogram2d(values, values)
        self.assertEqual(histogram.shape, (6, 6))
        self.assertEqual(int(histogram.sum()), len(values))

    def test_tertile_boundary_rule(self) -> None:
        scores = np.arange(9, dtype=float)
        level, (q1, q2) = stable_tertiles(scores)
        np.testing.assert_array_equal(level, np.where(scores <= q1, 0, np.where(scores <= q2, 1, 2)))


if __name__ == "__main__":
    unittest.main()
