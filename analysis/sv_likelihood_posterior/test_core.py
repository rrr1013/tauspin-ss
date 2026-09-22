import unittest

import numpy as np

from core import (
    gaussian_log_weight,
    mixture_log_likelihood,
    normalized_weights,
    offset_preserving_shuffle,
    opening_angle,
    sv_log_weight,
    weighted_representations,
)


class CoreTests(unittest.TestCase):
    def test_unavailable_side_is_uniform(self) -> None:
        tau = np.zeros((3, 5, 2, 3), dtype=float)
        tau[..., 0] = 1.0
        measured = np.zeros((3, 2, 3), dtype=float)
        measured[..., 0] = 1.0
        available = np.zeros((3, 2), dtype=bool)
        modes = np.full((3, 2), 3, dtype=np.int8)
        model = {3: {"pi_core": 0.9, "kappa": 100.0}}
        log_weight = sv_log_weight(tau, measured, available, modes, model, 4.0)
        weight, diagnostics = normalized_weights(log_weight)
        self.assertTrue(np.allclose(weight, 0.2))
        self.assertTrue(np.allclose(diagnostics["ess"], 5.0))

    def test_vmf_uniform_tail_is_finite_at_opposite_direction(self) -> None:
        cosine = np.array([1.0, 0.0, -1.0])
        result = mixture_log_likelihood(cosine, {"pi_core": 0.9, "kappa": 50_000.0})
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[0], result[1])
        self.assertTrue(np.isclose(result[1], result[2], atol=1.0e-8))

    def test_gaussian_prefers_aligned_candidate(self) -> None:
        tau = np.zeros((1, 2, 2, 3), dtype=float)
        tau[0, 0, :, 0] = 1.0
        tau[0, 1, :, 1] = 1.0
        measured = np.zeros((1, 2, 3), dtype=float)
        measured[..., 0] = 1.0
        log_weight = gaussian_log_weight(tau, measured, np.ones((1, 2), bool), 0.01)
        self.assertGreater(log_weight[0, 0], log_weight[0, 1])

    def test_weighted_representation_matches_selected_draw(self) -> None:
        h = np.arange(1 * 3 * 2 * 3, dtype=float).reshape(1, 3, 2, 3)
        weight = np.array([[0.0, 1.0, 0.0]])
        mean, moments = weighted_representations(h, weight)
        self.assertTrue(np.array_equal(mean[0], h[0, 1]))
        self.assertEqual(moments.shape, (1, 15))

    def test_offset_shuffle_preserves_angle_to_visible_axis(self) -> None:
        axis = np.asarray([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ])
        direction = np.asarray([
            [[0.995, 0.1, 0.0], [0.0, 0.98, 0.2]],
            [[0.1, 0.0, 0.995], [0.97, 0.0, 0.24]],
        ])
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        source = np.asarray([[1, 1], [0, 0]])
        available = np.ones((2, 2), dtype=bool)
        shuffled = offset_preserving_shuffle(direction, axis, source, available)
        original_angle = opening_angle(direction, axis)
        shuffled_angle = opening_angle(shuffled, axis)
        np.testing.assert_allclose(shuffled_angle[0], original_angle[1], atol=1.0e-12)
        np.testing.assert_allclose(shuffled_angle[1], original_angle[0], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
