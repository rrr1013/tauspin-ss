#!/usr/bin/env python3
"""Unit and regression tests for residual_substructure_decomposition.py."""

from pathlib import Path
import numpy as np

from residual_substructure_decomposition import (
    weighted_auc,
    weighted_conditional_auc,
    quantile_strata,
    combine_strata,
    fit_logistic,
    crossfit_index,
    load_data,
    FROZEN_CLOSURE,
)


def test_weighted_auc_perfect():
    score = np.array([0.1, 0.2, 0.8, 0.9])
    pos = np.array([False, False, True, True])
    weight = np.ones(4)
    assert weighted_auc(score, pos, weight) == 1.0


def test_weighted_auc_inverted():
    score = np.array([0.9, 0.8, 0.2, 0.1])
    pos = np.array([False, False, True, True])
    weight = np.ones(4)
    assert weighted_auc(score, pos, weight) == 0.0


def test_weighted_auc_ties():
    score = np.array([0.5, 0.5, 0.5, 0.5])
    pos = np.array([False, False, True, True])
    weight = np.ones(4)
    assert weighted_auc(score, pos, weight) == 0.5


def test_weighted_conditional_auc_pure():
    score = np.array([0.1, 0.9, 0.2, 0.8])
    pos = np.array([False, True, False, True])
    weight = np.ones(4)
    strata = np.array([0, 0, 1, 1])
    assert weighted_conditional_auc(score, pos, weight, strata) == 1.0


def test_quantile_strata():
    values = np.linspace(0, 100, 101)
    strata = quantile_strata(values, bins=10)
    assert len(strata) == 101
    assert strata.min() == 0
    assert strata.max() == 9


def test_fit_logistic_and_crossfit():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=(n, 2))
    logits = 0.5 + 1.2 * x[:, 0] - 0.8 * x[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logits))
    pos = rng.random(n) < prob
    fold = rng.integers(0, 2, size=n)

    index = crossfit_index(x, pos, fold)
    assert len(index) == n
    assert np.isfinite(index).all()


def test_load_data_and_closure():
    data_root = Path("/Users/ryunosuke/Projects")
    data = load_data(data_root)

    assert len(data["global_indices"]) == 11065
    assert len(np.unique(data["global_indices"])) == 11065
    assert np.isfinite(data["score_direct15-truth"]).all()
    assert np.isfinite(data["score_flow-full"]).all()

    pos = data["is_higgs"]
    w = data["overlap_weights"]

    # Closure assertions
    for arm, expected in FROZEN_CLOSURE.items():
        auc_val = weighted_auc(data["score_" + arm], pos, w)
        assert abs(auc_val - expected) < 1e-4, f"{arm} AUC {auc_val} deviates from expected {expected}"


if __name__ == "__main__":
    test_weighted_auc_perfect()
    test_weighted_auc_inverted()
    test_weighted_auc_ties()
    test_weighted_conditional_auc_pure()
    test_quantile_strata()
    test_fit_logistic_and_crossfit()
    test_load_data_and_closure()
    print("ALL 7 TESTS PASSED SUCCESSFULLY!")
