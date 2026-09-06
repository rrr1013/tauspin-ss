#!/usr/bin/env python3
"""Unit tests for the non-spin information run.  Run with ``python3 -m pytest`` or directly."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import nonspin_information as ns


def test_average_ranks_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    for n in (1, 2, 5, 37, 1000):
        values = rng.integers(0, 4, n).astype(float)
        assert np.allclose(ns.average_ranks(values), rankdata(values))


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(2)
    values = rng.integers(0, 4, 60).astype(float)
    positive = rng.random(60) < 0.4
    pos, neg = values[positive], values[~positive]
    brute = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert abs(ns.auc(values, positive) - brute) < 1e-12


def test_auc_is_nan_for_a_single_class():
    values = np.arange(10.0)
    assert np.isnan(ns.auc(values, np.ones(10, dtype=bool)))


def test_conditional_auc_equals_auc_for_one_stratum():
    rng = np.random.default_rng(3)
    values = rng.normal(size=500)
    positive = rng.random(500) < 0.5
    stratum = np.zeros(500, dtype=np.int64)
    assert abs(ns.conditional_auc(values, positive, stratum) - ns.auc(values, positive)) < 1e-12


def test_conditional_auc_removes_a_confounder_that_explains_everything():
    rng = np.random.default_rng(4)
    n = 40000
    confounder = rng.normal(size=n)
    # the label depends only on the confounder; the score is a function of it plus
    # class-independent noise, so nothing is left inside a stratum
    positive = rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * confounder))
    score = 2.0 * confounder + rng.normal(scale=0.05, size=n)
    stratum = ns.quantile_strata(confounder, 40)
    assert ns.auc(score, positive) > 0.7
    assert abs(ns.conditional_auc(score, positive, stratum) - 0.5) < 0.02


def test_conditional_auc_keeps_an_independent_signal():
    rng = np.random.default_rng(5)
    n = 20000
    confounder = rng.normal(size=n)
    positive = rng.random(n) < 0.5
    score = confounder + 0.8 * positive
    stratum = ns.quantile_strata(confounder, 20)
    assert ns.conditional_auc(score, positive, stratum) > 0.68


def test_quantile_strata_are_balanced_and_monotone():
    rng = np.random.default_rng(6)
    values = rng.normal(size=10000)
    stratum = ns.quantile_strata(values, 10)
    counts = np.bincount(stratum)
    assert counts.size == 10
    assert counts.min() > 900
    means = [values[stratum == b].mean() for b in range(10)]
    assert np.all(np.diff(means) > 0)


def test_combine_strata_is_dense_and_multiplicative():
    a = np.array([0, 0, 1, 1, 2])
    b = np.array([0, 1, 0, 1, 1])
    combined = ns.combine_strata(a, b)
    assert combined.min() == 0
    assert set(combined.tolist()) == set(range(len(np.unique(combined))))
    assert len(np.unique(combined)) == 5


def test_fit_logistic_recovers_known_coefficients():
    rng = np.random.default_rng(7)
    n = 200000
    design = rng.normal(size=(n, 3))
    truth = np.array([0.4, -0.9, 0.3])
    prob = 1.0 / (1.0 + np.exp(-(0.2 + design @ truth)))
    target = (rng.random(n) < prob).astype(float)
    beta = ns.fit_logistic(design, target, ridge=1e-6)
    assert np.allclose(beta[1:], truth, atol=0.03)
    assert abs(beta[0] - 0.2) < 0.03


def test_crossfit_index_does_not_leak_across_folds():
    rng = np.random.default_rng(8)
    n = 4000
    fold = (np.arange(n) % 2).astype(np.int64)
    design = rng.normal(size=(n, 2))
    # the label is pure noise, so an out-of-fold index must not discriminate
    positive = rng.random(n) < 0.5
    index = ns.crossfit_index(design, positive, fold)
    assert abs(ns.auc(index, positive) - 0.5) < 0.05


def test_crossfit_index_finds_a_real_signal():
    rng = np.random.default_rng(9)
    n = 8000
    fold = (np.arange(n) % 2).astype(np.int64)
    positive = rng.random(n) < 0.5
    design = np.column_stack([rng.normal(loc=0.8 * positive), rng.normal(size=n)])
    index = ns.crossfit_index(design, positive, fold)
    assert ns.auc(index, positive) > 0.65


def test_build_design_adds_squares_and_standardises():
    table = {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([0.0, 10.0, 20.0, 30.0])}
    keep = np.ones(4, dtype=bool)
    design = ns.build_design(table, (("a", "lin"), ("b", "log")), keep)
    assert design.shape == (4, 4)
    assert np.allclose(design[:, :2].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(design[:, :2].std(axis=0), 1.0, atol=1e-12)
    assert np.allclose(design[:, 2:], design[:, :2] ** 2)


def test_build_design_log_transform_is_log1p():
    table = {"a": np.array([0.0, np.e - 1.0, 2.0, 5.0])}
    design = ns.build_design(table, (("a", "log"),), np.ones(4, dtype=bool))
    expected = np.log1p(table["a"])
    expected = (expected - expected.mean()) / expected.std()
    assert np.allclose(design[:, 0], expected)


def test_additive_interaction_recovers_a_known_product_term():
    edges = np.linspace(-1.0, 1.0, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(10)
    coefficient = -0.13
    rows, cols, values = [], [], []
    for i, u in enumerate(centres):
        for j, v in enumerate(centres):
            mean = 0.3 + 0.2 * u - 0.1 * v + coefficient * u * v
            for _ in range(40):
                rows.append(i)
                cols.append(j)
                values.append(mean + rng.normal(scale=1e-6))
    estimate = ns.additive_interaction(
        np.array(values), np.array(rows), np.array(cols), centres
    )
    assert abs(estimate - coefficient) < 1e-3


def test_additive_interaction_is_blind_to_additive_structure():
    edges = np.linspace(-1.0, 1.0, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows, cols, values = [], [], []
    for i, u in enumerate(centres):
        for j, v in enumerate(centres):
            rows.append(i)
            cols.append(j)
            values.append(0.5 + 0.7 * u - 0.4 * v)
    estimate = ns.additive_interaction(
        np.array(values), np.array(rows), np.array(cols), centres
    )
    assert abs(estimate) < 1e-9


def test_cluster_bootstrap_resamples_whole_clusters():
    cluster = np.repeat(np.arange(50), 20)
    seen = []

    def statistic(idx):
        seen.append(np.bincount(cluster[idx], minlength=50))
        return float(idx.size)

    lo, hi, sd = ns.cluster_bootstrap(statistic, cluster, replicates=25, seed=0)
    assert lo == hi == 1000.0 and sd == 0.0
    for counts in seen:
        assert set(np.unique(counts).tolist()) <= {0, 20, 40, 60, 80, 100, 120}


def test_frozen_closure_constants_are_the_published_ones():
    assert ns.FROZEN_ENSEMBLE_AUC["full_reco"] == 0.6150930579720123
    assert ns.FROZEN_ENSEMBLE_AUC["tau_object_only"] == 0.6008338304557016
    assert ns.FROZEN_ENSEMBLE_AUC["event_only"] == 0.5616469239203451


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
