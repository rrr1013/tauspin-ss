#!/usr/bin/env python3
"""Self-contained checks for the decay-mode-pair discrimination run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decaymode_pair_power as mod


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_average_ranks_handles_ties() -> None:
    values = np.array([10.0, 20.0, 20.0, 5.0])
    ranks = mod.average_ranks(values)
    np.testing.assert_allclose(ranks, [2.0, 3.5, 3.5, 1.0])
    # A constant vector must give the midpoint rank everywhere.
    np.testing.assert_allclose(mod.average_ranks(np.zeros(6)), np.full(6, 3.5))


def test_auc_matches_brute_force() -> None:
    generator = np.random.default_rng(7)
    for _ in range(20):
        n = int(generator.integers(4, 40))
        scores = np.round(generator.normal(size=n), 1)  # rounding forces ties
        positive = generator.random(n) < 0.5
        if positive.all() or not positive.any():
            continue
        pos = scores[positive]
        neg = scores[~positive]
        brute = float(
            np.mean((pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :]))
        )
        np.testing.assert_allclose(mod.auc_from_scores(scores, positive), brute, atol=1e-12)


def test_auc_is_degenerate_without_both_classes() -> None:
    check(np.isnan(mod.auc_from_scores(np.arange(5.0), np.ones(5, dtype=bool))), "single-class AUC must be nan")


def test_cell_indices_are_unordered_and_complete() -> None:
    modes = np.array([[0, 1], [1, 0], [3, 3], [-1, 2], [4, 0]])
    cells, keys = mod.cell_indices(modes)
    check(cells[0] == cells[1], "the tau side order must not change the cell")
    check(cells[3] == -1, "an unknown mode must fall outside every cell")
    check(len(keys) == 15, "five modes give fifteen unordered pairs")
    check(keys[cells[2]] == (3, 3), "3p0n x 3p0n must map onto its own key")
    check(keys[cells[4]] == (0, 4), "cell keys must be sorted low to high")


def test_raw_decay_modes_round_trip() -> None:
    metadata = {
        "tau_decay_mode_to_id": {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5},
        "tau_decay_unknown_id": 0,
        "tau_decay_num_embeddings": 6,
    }
    ids = np.array([[1, 5], [0, 3], [2, 2]])
    raw = mod.raw_decay_modes(ids, metadata)
    np.testing.assert_array_equal(raw, [[0, 4], [-1, 2], [1, 1]])


def test_cluster_resampler_preserves_cluster_blocks() -> None:
    clusters = np.array([0, 0, 1, 1, 1, 2, 0, 0, 3, 3])
    positive = np.array([True] * 5 + [False] * 5)
    resampler = mod.ClusterResampler(clusters, positive)
    generator = np.random.default_rng(3)
    for _ in range(50):
        picked = resampler.draw(generator)
        check(len(picked) > 0, "a draw must not be empty")
        # Every drawn row keeps its own class, and each class keeps its cluster count.
        for is_pos in (True, False):
            rows = picked[positive[picked] == is_pos]
            drawn_clusters = clusters[rows]
            expected_clusters = len(np.unique(clusters[positive == is_pos]))
            # Cluster blocks are drawn whole, so counts per drawn cluster are
            # integer multiples of that cluster's true size.
            for cluster in np.unique(drawn_clusters):
                true_size = int(np.count_nonzero((clusters == cluster) & (positive == is_pos)))
                drawn_size = int(np.count_nonzero(drawn_clusters == cluster))
                check(drawn_size % true_size == 0, "a cluster must be drawn as a whole block")
            check(len(np.unique(drawn_clusters)) <= expected_clusters, "no cluster may be invented")


def test_per_cell_auc_matches_direct_computation() -> None:
    generator = np.random.default_rng(11)
    n = 400
    cells = generator.integers(0, 5, size=n)
    positive = generator.random(n) < 0.5
    scores = {"a": generator.normal(size=n), "b": generator.normal(size=n)}
    result = mod.per_cell_auc(cells, positive, scores, 6)
    for cell in range(6):
        mask = cells == cell
        for arm in scores:
            expected = mod.auc_from_scores(scores[arm][mask], positive[mask]) if mask.any() else float("nan")
            if np.isnan(expected):
                check(np.isnan(result[arm][cell]), "empty cells must stay nan")
            else:
                np.testing.assert_allclose(result[arm][cell], expected, atol=1e-12)


def test_spearman_matches_a_known_case() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(mod.spearman_rho(x, 2.0 * x + 1.0), 1.0, atol=1e-12)
    np.testing.assert_allclose(mod.spearman_rho(x, -x), -1.0, atol=1e-12)
    # Two adjacent swaps give sum d^2 = 4, so rho = 1 - 6*4/(5*24) = 0.8.
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    np.testing.assert_allclose(mod.spearman_rho(x, y), 0.8, atol=1e-12)


def test_weighted_spread_ignores_masked_cells() -> None:
    values = np.array([0.6, 0.4, np.nan])
    weights = np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(mod.weighted_spread(values, weights), 0.1, atol=1e-12)
    check(np.isnan(mod.weighted_spread(values, np.array([1.0, 0.0, 0.0]))), "one cell has no spread")


def test_exact_join_fails_on_duplicates() -> None:
    left = mod._structured_keys((np.array([0, 1]), np.array([0, 0]), np.array([1, 2])), ("a", "b", "c"))
    right = mod._structured_keys((np.array([0, 0]), np.array([0, 0]), np.array([1, 1])), ("a", "b", "c"))
    try:
        mod.exact_join_indices(left, right, "duplicate probe")
    except RuntimeError as error:
        check("duplicate" in str(error), "duplicate keys must be named in the failure")
    else:
        raise AssertionError("a duplicate right key must fail closed")


def test_restore_inverts_standardisation() -> None:
    stats = {"standardize": [True, False], "mean": [2.0, 99.0], "std": [3.0, 99.0]}
    values = np.array([[1.0, 7.0], [-1.0, 8.0]])
    np.testing.assert_allclose(mod.restore(values, stats, 0), [5.0, -1.0])
    np.testing.assert_allclose(mod.restore(values, stats, 1), [7.0, 8.0])


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
