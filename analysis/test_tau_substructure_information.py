#!/usr/bin/env python3
"""Unit tests for the tau substructure run.  Run with ``python3 -m pytest`` or directly."""

from __future__ import annotations

import numpy as np

import tau_substructure_information as ts

try:  # awkward/uproot only exist where the ntuple lives, so those tests skip elsewhere.
    import extract_tau_substructure as ex
except ModuleNotFoundError:  # pragma: no cover - depends on the host
    ex = None


class Skipped(Exception):
    pass


def need_extraction():
    if ex is None:
        raise Skipped("awkward is not installed on this host")
    return ex


# ---------------------------------------------------------------------------- extraction


def test_safe_ratio_uses_the_fill_where_the_denominator_vanishes():
    out = need_extraction().safe_ratio(np.array([2.0, 5.0]), np.array([4.0, 0.0]), fill=-1.0)
    assert np.allclose(out, [0.5, -1.0])


def test_side_features_on_a_hand_built_one_prong_with_one_pi0():
    ak = need_extraction().ak
    tracks = {name: ak.Array([[value]]) for name, value in (
        ("pt", 30.0), ("eta", 0.0), ("phi", 0.0), ("dEta", 0.0), ("dPhi", 0.0),
        ("ptFraction", 0.6), ("d0", -0.2), ("z0SinTheta", 3.0), ("isFake", 0),
        ("passTrkSelector", 1), ("numberOfPixelHits", 4), ("numberOfSCTHits", 8),
        ("numberOfTRTHits", 30))}
    pfos = {name: ak.Array([[value]]) for name, value in (
        ("pt", 10.0), ("eta", 0.0), ("phi", 0.0), ("dEta", 0.03), ("dPhi", 0.04),
        ("ptFraction", 0.2), ("isPi0", 1), ("bdtPi0Score", 0.0), ("nPi0Proto", 1))}
    out = ex.side_features(tracks, pfos, np.array([40.0]), np.array([0.0]), np.array([0.0]))
    assert abs(out["f_charged"][0] - 0.75) < 1e-12
    assert abs(out["f_pi0"][0] - 0.25) < 1e-12
    assert abs(out["ups_pi0"][0] - 0.5) < 1e-12          # (30 - 10) / (30 + 10)
    assert abs(out["subl_trk_ptfrac"][0]) < 1e-12        # only one track
    assert abs(out["trk_pt_asym"][0] - 1.0) < 1e-12
    assert abs(out["pfo_width"][0] - 0.05) < 1e-12       # hypot(0.03, 0.04)
    assert abs(out["all_width"][0] - 10.0 * 0.05 / 40.0) < 1e-12
    assert abs(out["const_over_taupt"][0] - 1.0) < 1e-12
    assert abs(out["mean_pix"][0] - 4.0) < 1e-12
    assert abs(out["max_abs_d0"][0] - 0.2) < 1e-12


def test_side_features_with_no_neutral_pfo_falls_back_cleanly():
    ak = need_extraction().ak
    tracks = {name: ak.Array([[value]]) for name, value in (
        ("pt", 25.0), ("eta", 0.1), ("phi", 0.2), ("dEta", 0.0), ("dPhi", 0.0),
        ("ptFraction", 1.0), ("d0", 0.0), ("z0SinTheta", 0.0), ("isFake", 0),
        ("passTrkSelector", 1), ("numberOfPixelHits", 3), ("numberOfSCTHits", 8),
        ("numberOfTRTHits", 25))}
    pfos = {name: ak.Array([[]]) for name in ex.PFO_FIELDS}
    out = ex.side_features(tracks, pfos, np.array([25.0]), np.array([0.0]), np.array([1.0]))
    assert out["f_charged"][0] == 1.0
    assert out["f_pi0"][0] == 0.0
    assert out["ups_pi0"][0] == 1.0
    assert out["n_pfo"][0] == 0.0
    assert out["pfo_width"][0] == 0.0
    assert np.isfinite(list(out.values())).all()


def test_side_features_refuses_a_tau_without_tracks():
    ak = need_extraction().ak
    tracks = {name: ak.Array([[]]) for name in ex.TRACK_FIELDS}
    pfos = {name: ak.Array([[]]) for name in ex.PFO_FIELDS}
    try:
        ex.side_features(tracks, pfos, np.array([10.0]), np.array([0.0]), np.array([0.0]))
    except RuntimeError:
        return
    raise AssertionError("a tau with no core track must be refused")


def test_constituent_mass_of_two_back_to_back_massless_constituents():
    ak = need_extraction().ak
    tracks = {name: ak.Array([[value]]) for name, value in (
        ("pt", 10.0), ("eta", 0.0), ("phi", 0.0), ("dEta", 0.0), ("dPhi", 0.0),
        ("ptFraction", 1.0), ("d0", 0.0), ("z0SinTheta", 0.0), ("isFake", 0),
        ("passTrkSelector", 1), ("numberOfPixelHits", 0), ("numberOfSCTHits", 0),
        ("numberOfTRTHits", 0))}
    pfos = {name: ak.Array([[value]]) for name, value in (
        ("pt", 10.0), ("eta", 0.0), ("phi", np.pi), ("dEta", 0.0), ("dPhi", 0.0),
        ("ptFraction", 1.0), ("isPi0", 0), ("bdtPi0Score", 0.0), ("nPi0Proto", 0))}
    out = ex.side_features(tracks, pfos, np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert abs(out["m_const"][0] - 20.0) < 1e-9


# ----------------------------------------------------------------------------- analysis


def test_sided_expands_both_sides_in_order():
    assert ts.sided((("a", "lin"), ("b", "log"))) == (
        ("a_minus", "lin"), ("a_plus", "lin"), ("b_minus", "log"), ("b_plus", "log"))
    assert ts.sided((("a", "lin"),), sides=("plus",)) == (("a_plus", "lin"),)


def test_blocks_are_disjoint_and_s_all_is_their_union():
    columns = {name: {c for c, _ in ts.sided(ts.SIDED_BLOCKS[name])}
               for name in ts.SUBSTRUCTURE_ORDER}
    union: set[str] = set()
    for name, names in columns.items():
        assert not (union & names), f"{name} overlaps an earlier block"
        union |= names
    blocks = ts.build_blocks()
    assert {c for c, _ in blocks["S-all"]} == union
    assert len(union) == 2 * ts.sum_per_side()


def test_one_sided_blocks_partition_s_all():
    blocks = ts.build_blocks()
    minus = {c for c, _ in blocks["S-all (tau- only)"]}
    plus = {c for c, _ in blocks["S-all (tau+ only)"]}
    assert minus.isdisjoint(plus)
    assert minus | plus == {c for c, _ in blocks["S-all"]}
    assert all(c.endswith("_minus") for c in minus)


def test_transform_columns_applies_log1p_only_where_asked():
    table = {"a": np.array([0.0, 3.0, 8.0]), "b": np.array([0.0, 3.0, 8.0])}
    keep = np.ones(3, dtype=bool)
    out = ts.transform_columns(table, (("a", "lin"), ("b", "log")), keep)
    assert np.allclose(out[:, 0], [0.0, 3.0, 8.0])
    assert np.allclose(out[:, 1], np.log1p([0.0, 3.0, 8.0]))


def test_crossfit_index_is_out_of_fold_and_ignores_a_single_wild_outlier():
    rng = np.random.default_rng(11)
    n = 4000
    positive = rng.random(n) < 0.5
    raw = np.column_stack([rng.normal(positive * 0.4, 1.0, n), rng.normal(size=n)])
    fold = (np.arange(n) % 2).astype(np.int64)
    clean = ts.crossfit_index(raw.copy(), positive, fold)
    spoilt = raw.copy()
    spoilt[0, 0] = 1e9
    dirty = ts.crossfit_index(spoilt, positive, fold)
    assert ts.auc(clean, positive) > 0.55
    # Winsorisation must keep one absurd entry from dominating the standardisation.
    assert abs(ts.auc(dirty[1:], positive[1:]) - ts.auc(clean[1:], positive[1:])) < 0.01


def test_crossfit_index_has_no_leakage_when_the_label_is_pure_noise():
    rng = np.random.default_rng(12)
    n = 3000
    positive = rng.random(n) < 0.5
    raw = rng.normal(size=(n, 4))
    fold = (np.arange(n) % 2).astype(np.int64)
    index = ts.crossfit_index(raw, positive, fold)
    assert abs(ts.auc(index, positive) - 0.5) < 0.03


def test_conditioning_on_the_score_itself_removes_all_discrimination():
    rng = np.random.default_rng(13)
    n = 6000
    positive = rng.random(n) < 0.5
    score = rng.normal(positive * 0.5, 1.0, n)
    strata = ts.quantile_strata(score, 40)
    assert ts.auc(score, positive) > 0.6
    assert abs(ts.conditional_auc(score, positive, strata) - 0.5) < 0.02


def test_conditioning_on_an_independent_variable_changes_nothing():
    rng = np.random.default_rng(14)
    n = 6000
    positive = rng.random(n) < 0.5
    score = rng.normal(positive * 0.5, 1.0, n)
    strata = ts.quantile_strata(rng.normal(size=n), ts.STRATA_BINS)
    assert abs(ts.conditional_auc(score, positive, strata) - ts.auc(score, positive)) < 0.01


def test_frozen_closure_constants_are_the_ones_fixed_on_2026_08_25():
    assert ts.FROZEN_ENSEMBLE_AUC["full_reco"] == 0.6150930579720123
    assert ts.FROZEN_ENSEMBLE_AUC["tau_object_only"] == 0.6008338304557016
    assert ts.FROZEN_ENSEMBLE_AUC["event_only"] == 0.5616469239203451


def test_the_baseline_block_is_exactly_the_non_spin_run_p_reco_plus_d_reco():
    assert ts.build_blocks()["E-reco"] == ts.P_RECO + ts.D_RECO


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok   {name}")
            except Skipped as reason:
                print(f"skip {name}: {reason}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    print(f"{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
