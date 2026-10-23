"""Metrics, agreement, multiplicity and the reader study."""

from __future__ import annotations

import numpy as np
import pytest

from tearfilm_persistence.evaluation.ablation import delta, deltas, interaction_ratio
from tearfilm_persistence.evaluation.agreement import severity_kappa, weighted_kappa
from tearfilm_persistence.evaluation.budget import (
    DECLARED_PROFILE,
    count_flops,
    profile_checks,
    shares,
)
from tearfilm_persistence.evaluation.criteria import CriteriaThresholds, all_passed, evaluate
from tearfilm_persistence.evaluation.discrimination import (
    cluster_bootstrap,
    operating_point,
    paired_difference,
    roc_auc,
    sensitivity_at,
    specificity_at,
)
from tearfilm_persistence.evaluation.mechanistic import fold_change, percent_change
from tearfilm_persistence.evaluation.multiplicity import holm_stepdown
from tearfilm_persistence.evaluation.properties import (
    cross_site_change,
    expected_calibration_error,
    false_early_rate,
    prevalence,
)
from tearfilm_persistence.evaluation.reader_study import (
    experience_groups,
    orc_difference,
    reliance_summary,
    two_way_mean_squares,
    wilson_interval,
)


def test_auc_follows_the_rank_identity() -> None:
    assert roc_auc(
        np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([0.0, 0.0, 1.0, 1.0])
    ) == pytest.approx(1.0)
    assert roc_auc(
        np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([1.0, 0.0, 1.0, 0.0])
    ) == pytest.approx(0.25)


def test_auc_is_half_when_every_score_ties() -> None:
    assert roc_auc(np.ones(6), np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.5)


def test_auc_of_a_single_class_is_undefined() -> None:
    assert np.isnan(roc_auc(np.asarray([0.1, 0.2]), np.asarray([1.0, 1.0])))


def test_bootstrap_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(5)
    scores = np.concatenate([rng.normal(1.2, 1.0, 80), rng.normal(0.0, 1.0, 80)])
    labels = np.concatenate([np.ones(80), np.zeros(80)])
    clusters = np.asarray([f"p{index // 2}" for index in range(160)])
    result = cluster_bootstrap(scores, labels, clusters, n_boot=300, seed=1)
    assert result.lower <= result.auc <= result.upper
    assert result.n_clusters == 80


def test_bootstrap_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap(np.zeros(3), np.zeros(4), np.zeros(3))


def test_operating_point_respects_the_floor() -> None:
    rng = np.random.default_rng(6)
    scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    point = operating_point(scores, labels, 0.95)
    assert point.specificity >= 0.95 - 1e-9
    assert sensitivity_at(scores, labels, point.threshold) == pytest.approx(point.sensitivity)
    assert specificity_at(scores, labels, point.threshold) == pytest.approx(point.specificity)


def test_paired_difference_recovers_a_constant_shift() -> None:
    first = np.full(20, 0.9)
    second = np.full(20, 0.8)
    clusters = np.asarray([f"p{index}" for index in range(20)])
    mean, lower, upper = paired_difference(first, second, clusters, n_boot=100)
    assert mean == pytest.approx(0.1)
    assert lower == pytest.approx(0.1)
    assert upper == pytest.approx(0.1)


def test_false_early_needs_both_onsets() -> None:
    declared = np.asarray([1.0, 9.0, np.nan, 4.0])
    reference = np.asarray([5.0, 5.0, 5.0, 6.0])
    assert false_early_rate(declared, reference) == pytest.approx(1 / 3)


def test_cross_site_change_is_the_mean_absolute_difference() -> None:
    assert cross_site_change({"A": 0.9, "B": 0.8}, {"A": 0.95, "B": 0.82}) == pytest.approx(0.035)


def test_cross_site_change_rejects_mismatched_sites() -> None:
    with pytest.raises(ValueError):
        cross_site_change({"A": 0.9}, {"B": 0.9})


def test_calibration_error_is_zero_for_a_calibrated_sample() -> None:
    probabilities = np.asarray([0.0, 1.0, 0.0, 1.0])
    labels = np.asarray([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(probabilities, labels, bins=4) == pytest.approx(0.0)


def test_calibration_error_detects_overconfidence() -> None:
    probabilities = np.asarray([0.95] * 10)
    labels = np.zeros(10)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.95, abs=1e-9)


def test_prevalence_counts_positives() -> None:
    assert prevalence(np.asarray([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.5)


def test_weighted_kappa_penalises_a_one_level_shift_less_than_a_two_level_shift() -> None:
    near = weighted_kappa(
        ["mild", "mild", "moderate", "moderate"], ["mild", "moderate", "moderate", "severe"]
    )
    far = weighted_kappa(["mild", "mild", "moderate", "moderate"], ["severe"] * 4)
    assert 0.0 < near < 1.0
    assert far < near


def test_severity_kappa_uses_the_three_levels() -> None:
    assert severity_kappa(["mild", "moderate"], ["mild", "moderate"]) == pytest.approx(1.0)


def test_weighted_kappa_rejects_unknown_categories() -> None:
    with pytest.raises(ValueError):
        weighted_kappa(["mild"], ["grave"], ["mild", "moderate"])


def test_holm_adjusts_in_ascending_order() -> None:
    result = holm_stepdown({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    adjusted = dict(zip(result.names, result.adjusted, strict=True))
    assert adjusted["a"] == pytest.approx(0.04)
    assert adjusted["d"] == pytest.approx(0.06)
    assert result.rejected == (True, False, False, False)


def test_holm_rejects_an_empty_family() -> None:
    with pytest.raises(ValueError):
        holm_stepdown({})


def test_two_way_mean_squares_on_a_balanced_table() -> None:
    table = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    reader, case, residual = two_way_mean_squares(table)
    assert reader == pytest.approx(4.0)
    assert case == pytest.approx(1.0)
    assert residual == pytest.approx(0.0)


def test_orc_difference_is_positive_for_a_shift() -> None:
    rng = np.random.default_rng(8)
    base = rng.normal(size=(6, 20))
    result = orc_difference(base + 0.03, base)
    assert result.difference == pytest.approx(0.03)
    assert result.p_value < 0.01


def test_orc_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        orc_difference(np.zeros((3, 4)), np.zeros((3, 5)))


def test_reliance_summary_uses_the_discordant_denominator() -> None:
    summary = reliance_summary(20, 10, 318)
    assert summary.followed_rate == pytest.approx(20 / 318)
    assert summary.followed_interval[0] < summary.followed_rate < summary.followed_interval[1]


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    lower, upper = wilson_interval(0, 10)
    assert 0.0 <= lower <= upper <= 1.0


def test_experience_groups_split_at_the_printed_boundaries() -> None:
    groups = experience_groups([4, 5, 10, 11])
    assert groups == {"under_5": [0], "five_to_ten": [1, 2], "over_ten": [3]}


def test_interaction_ratio_is_one_for_additive_components() -> None:
    assert interaction_ratio(0.95, 0.92, 0.93, 0.90) == pytest.approx(1.0)


def test_delta_is_signed_from_the_full_model() -> None:
    entries = deltas(0.944, {"minus_fast": 0.897})
    assert entries[0].delta == pytest.approx(0.047)
    assert delta(0.944, 0.897) == pytest.approx(0.047)


def test_budget_profile_sums_to_the_declared_total() -> None:
    checks = profile_checks()
    assert checks["gflops_sum"] == pytest.approx(78.0)
    assert checks["share_sum"] == pytest.approx(1.0, abs=1e-3)
    assert len(shares()) == len(DECLARED_PROFILE)


def test_flop_counter_counts_a_convolution() -> None:
    import torch
    from torch import nn

    module = nn.Conv2d(3, 4, 3, padding=1)
    value = count_flops(module, (torch.zeros(1, 3, 8, 8),))
    expected = 2.0 * 4 * 8 * 8 * 3 * 9 / 1e9
    assert value == pytest.approx(expected)


def test_total_gflops_refuses_an_empty_profile() -> None:
    with pytest.raises(ValueError):
        shares([])


def test_criteria_pass_on_the_reported_levels() -> None:
    results = evaluate(0.936, 0.084, 0.8, 34.2)
    assert all_passed(results)
    assert {result.name for result in results} == {
        "assisted_auc",
        "assisted_gain",
        "false_early_rate",
        "descriptor_cov",
    }


def test_criteria_fail_below_the_threshold() -> None:
    results = evaluate(0.90, 0.02, 2.0, 50.0)
    assert not all_passed(results)


def test_criteria_thresholds_are_the_printed_ones() -> None:
    config = CriteriaThresholds()
    assert config.assisted_auc_min == 0.92
    assert config.assisted_gain_min == 0.05
    assert config.false_early_max_percent == 1.0
    assert config.descriptor_cov_max_percent == 45.0


def test_percent_change_of_a_doubling_is_one_hundred() -> None:
    estimate = percent_change([1.0, 1.1, 0.9], [2.0, 2.1, 1.9], n_boot=200)
    assert estimate.percent_change == pytest.approx(100.0, abs=5.0)
    assert estimate.replicates == 3


def test_fold_change_of_a_doubling_is_two() -> None:
    estimate = fold_change([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], n_boot=100)
    assert estimate.percent_change == pytest.approx(2.0)


def test_mechanistic_arm_needs_replicates() -> None:
    with pytest.raises(ValueError):
        percent_change([1.0], [2.0])
