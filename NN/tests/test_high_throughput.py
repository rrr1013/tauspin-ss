from __future__ import annotations

import unittest

import torch

from dataset import (
    CLS_TYPE,
    EVENT_TYPE,
    PFO_TYPE,
    TAU_TYPE,
    TRACK_TYPE,
    event_partition_count,
    event_partition_indices,
)
from hpo_utils import portable_model_state_dict
from model import TauSpinTransformer


class EventPartitionTest(unittest.TestCase):
    def test_complete_disjoint_balanced_coverage_per_shard(self) -> None:
        shard_sizes = (1, 11, 12, 13, 50_003)
        counts = [0] * 12
        event_offset = 0
        for n_events in shard_sizes:
            partitions = [
                set(
                    event_partition_indices(
                        n_events,
                        worker,
                        12,
                        event_offset,
                    )
                )
                for worker in range(12)
            ]
            self.assertEqual(set().union(*partitions), set(range(n_events)))
            for left in range(12):
                for right in range(left + 1, 12):
                    self.assertTrue(
                        partitions[left].isdisjoint(partitions[right])
                    )
            for worker in range(12):
                counts[worker] += event_partition_count(
                    n_events,
                    worker,
                    12,
                    event_offset,
                )
            event_offset += n_events
        self.assertLessEqual(max(counts) - min(counts), 1)


class DenseProjectionTest(unittest.TestCase):
    def test_dense_projection_matches_boolean_projection(self) -> None:
        torch.manual_seed(7)
        dimensions = {"event": 4, "tau": 5, "track": 6, "pfo": 3}
        model = TauSpinTransformer(
            dimensions,
            8,
            d_model=16,
            n_head=4,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ).eval()
        object_type = torch.tensor(
            [
                [EVENT_TYPE, TAU_TYPE, TRACK_TYPE, PFO_TYPE, 0],
                [EVENT_TYPE, TAU_TYPE, TAU_TYPE, TRACK_TYPE, PFO_TYPE],
            ]
        )
        batch = {
            "event_features": torch.randn(2, 5, dimensions["event"]),
            "tau_features": torch.randn(2, 5, dimensions["tau"]),
            "track_features": torch.randn(2, 5, dimensions["track"]),
            "pfo_features": torch.randn(2, 5, dimensions["pfo"]),
            "object_type": object_type,
            "tau_side": torch.tensor([[0, 1, 1, 2, 0], [0, 1, 2, 2, 2]]),
            "decay_mode": torch.tensor([[0, 1, 0, 0, 0], [0, 2, 3, 0, 0]]),
            "padding_mask": object_type == 0,
        }
        with torch.no_grad():
            actual = model(batch)
            expected = self._legacy_forward(model, batch)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @staticmethod
    def _legacy_forward(
        model: TauSpinTransformer,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        object_type = batch["object_type"]
        tokens = torch.zeros(
            *object_type.shape,
            model.cls_token.shape[-1],
            dtype=batch["event_features"].dtype,
        )
        projectors = (
            (EVENT_TYPE, "event_features", model.event_projector),
            (TAU_TYPE, "tau_features", model.tau_projector),
            (TRACK_TYPE, "track_features", model.track_projector),
            (PFO_TYPE, "pfo_features", model.pfo_projector),
        )
        for type_id, feature_name, projector in projectors:
            mask = object_type == type_id
            tokens[mask] = projector(batch[feature_name][mask])
        tokens = (
            tokens
            + model.object_type_embedding(object_type)
            + model.tau_side_embedding(batch["tau_side"])
            + model.decay_mode_embedding(batch["decay_mode"])
        )
        batch_size = tokens.shape[0]
        cls_type = torch.full((batch_size, 1), CLS_TYPE, dtype=torch.int64)
        cls = (
            model.cls_token.expand(batch_size, -1, -1)
            + model.object_type_embedding(cls_type)
        )
        tokens = torch.cat((cls, tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(batch_size, 1, dtype=torch.bool),
                batch["padding_mask"],
            ),
            dim=1,
        )
        encoded = model.encoder(tokens, src_key_padding_mask=padding_mask)
        return model.classifier(encoded[:, 0]).squeeze(-1)

    def test_compiled_wrapper_prefix_is_not_saved(self) -> None:
        eager = torch.nn.Linear(3, 2)

        class Wrapper(torch.nn.Module):
            def __init__(self, original: torch.nn.Module) -> None:
                super().__init__()
                self._orig_mod = original

        keys = portable_model_state_dict(Wrapper(eager)).keys()
        self.assertEqual(set(keys), {"weight", "bias"})
        self.assertFalse(any(key.startswith("_orig_mod.") for key in keys))


if __name__ == "__main__":
    unittest.main()
