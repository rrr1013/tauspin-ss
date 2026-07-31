from __future__ import annotations

import unittest

from select_v3_hpo import canonical_parameters, rank_trials


class V3SelectionTest(unittest.TestCase):
    def test_auc_gap_then_loss_then_parameters_then_trial(self) -> None:
        rows = [
            self.row(0, 0.9000, 0.30, 300, {"model_profile": "wide"}),
            self.row(1, 0.8995, 0.20, 300, {"model_profile": "wide"}),
            self.row(2, 0.8980, 0.10, 100, {"model_profile": "small"}),
        ]
        ranked, selected, best_auc = rank_trials(rows)
        self.assertEqual(best_auc, 0.9000)
        self.assertEqual(selected["trial_number"], 1)
        self.assertEqual([row["trial_number"] for row in ranked], [1, 0, 2])
        self.assertFalse(ranked[2]["within_auc_gap"])

    def test_parameter_count_precedes_trial_number(self) -> None:
        rows = [
            self.row(1, 0.9, 0.2, 200, {"model_profile": "small"}),
            self.row(0, 0.9, 0.2, 300, {"model_profile": "small"}),
            self.row(2, 0.9, 0.2, 200, {"model_profile": "wide"}),
        ]
        ranked, selected, _ = rank_trials(rows)
        self.assertEqual(selected["trial_number"], 1)
        self.assertEqual([row["trial_number"] for row in ranked], [1, 2, 0])

    @staticmethod
    def row(
        trial_number: int,
        objective_auc: float,
        minimum_validation_loss: float,
        trainable_parameter_count: int,
        parameters: dict,
    ) -> dict:
        return {
            "trial_number": trial_number,
            "objective_auc": objective_auc,
            "minimum_validation_loss": minimum_validation_loss,
            "trainable_parameter_count": trainable_parameter_count,
            "parameters": parameters,
            "parameters_canonical": canonical_parameters(parameters),
        }


if __name__ == "__main__":
    unittest.main()
