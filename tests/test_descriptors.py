"""Descriptor vector, repeatability statistics and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from tearfilm_persistence.deformation.curve import CurveSummary
from tearfilm_persistence.descriptors.calibration import (
    LogisticCalibrator,
    VarianceComponentCalibrator,
)
from tearfilm_persistence.descriptors.repeatability import (
    bland_altman,
    coefficient_of_repeatability,
    coefficient_of_variation,
    icc_one_way,
    reliability,
    summarise_repeats,
    variance_shares,
    within_subject_sd,
)
from tearfilm_persistence.descriptors.vector import (
    COMPONENTS,
    DescriptorVector,
    Standardiser,
)


def test_component_order_is_the_printed_one() -> None:
    assert COMPONENTS == (
        "onset",
        "peak_instability",
        "curve_slope",
        "locus_entropy",
        "tmh",
        "tma",
    )


def test_vector_roundtrips_through_a_mapping() -> None:
    vector = DescriptorVector(3.2, 0.41, 0.8, 0.33, 0.21, 0.05)
    mapping = vector.as_dict()
    assert mapping["onset"] == pytest.approx(3.2)
    assert list(mapping) == list(COMPONENTS)
    assert vector.as_tensor().numel() == 6


def test_censored_vector_keeps_the_epoch_length() -> None:
    summary = CurveSummary(
        declared=False,
        onset_seconds=None,
        peak=0.1,
        slope=0.0,
        entropy=0.0,
        peak_time=0.0,
        n_subtests=1,
        epoch_seconds=17.9,
        per_subtest=(),
    )
    vector = DescriptorVector.from_summary(summary, 0.2, 0.04)
    assert vector.onset_seconds == pytest.approx(17.9)
    assert not vector.declared


def test_standardiser_centres_and_scales() -> None:
    matrix = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    standard = Standardiser.fit(matrix)
    transformed = standard.transform(matrix)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-12)


def test_within_subject_sd_pools_the_repeats() -> None:
    assert within_subject_sd([[1.0, 2.0, 3.0]]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        within_subject_sd([[1.0]])


def test_repeatability_coefficient_follows_the_printed_convention() -> None:
    assert coefficient_of_repeatability(1.21) == pytest.approx(3.3517, rel=1e-4)


def test_coefficient_of_variation_needs_a_positive_mean() -> None:
    assert coefficient_of_variation(1.0, 4.0) == pytest.approx(25.0)
    with pytest.raises(ValueError):
        coefficient_of_variation(1.0, 0.0)


def test_icc_is_one_when_repeats_agree() -> None:
    assert icc_one_way([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) == pytest.approx(1.0)


def test_icc_needs_repeated_eyes() -> None:
    with pytest.raises(ValueError):
        icc_one_way([[1.0], [2.0]])


def test_bland_altman_reports_bias_and_limits() -> None:
    summary = bland_altman([2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert summary.bias == pytest.approx(1.0)
    assert summary.lower_limit == pytest.approx(1.0)
    assert summary.n_pairs == 4


def test_variance_shares_close_to_one() -> None:
    values = [1.0, 1.2, 2.4, 2.6, 3.9, 4.1, 5.4, 5.6]
    shares = variance_shares(
        values,
        ["A"] * 4 + ["B"] * 4,
        ["s1", "s1", "s2", "s2"] * 2,
        ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"],
    )
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["site"] > 0


def test_variance_shares_reject_misaligned_input() -> None:
    with pytest.raises(ValueError):
        variance_shares([1.0, 2.0], ["A"], ["s1", "s1"], ["e1", "e1"])


def test_reliability_is_one_for_distinct_eyes_without_noise() -> None:
    assert reliability([1.0, 1.0, 5.0, 5.0], ["a", "a", "b", "b"]) == pytest.approx(1.0)


def test_summarise_repeats_returns_the_full_set() -> None:
    summary = summarise_repeats([[1.0, 1.2, 1.4], [2.0, 2.2, 2.4]])
    assert summary["n_eyes"] == 2.0
    assert summary["cor"] == pytest.approx(2.77 * summary["sw"])
    assert summary["cov_percent"] > 0


def test_calibrator_shrinks_toward_the_site_mean() -> None:
    matrix = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [3.2, 3.2, 3.2, 3.2, 3.2, 3.2],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [2.2, 2.2, 2.2, 2.2, 2.2, 2.2],
        ],
        dtype=np.float64,
    )
    eyes = ["a", "a", "b", "b", "c", "c"]
    calibrator = VarianceComponentCalibrator().fit(matrix, eyes, ["A"] * 6)
    calibrated = calibrator.transform(matrix, ["A"] * 6)
    centre = matrix.mean(axis=0)
    assert all(0.0 <= value <= 1.0 for value in calibrator.reliability)
    assert abs(calibrated[0, 0] - centre[0]) < abs(matrix[0, 0] - centre[0])
    assert abs(calibrated[0, 0] - centre[0]) > 0.0


def test_calibrator_state_roundtrip() -> None:
    matrix = np.asarray([[1.0] * 6, [1.1] * 6, [4.0] * 6, [4.2] * 6], dtype=np.float64)
    calibrator = VarianceComponentCalibrator().fit(matrix, ["a", "a", "b", "b"], ["A"] * 4)
    restored = VarianceComponentCalibrator.from_state(calibrator.state())
    assert restored.reliability == calibrator.reliability


def test_calibrator_requires_fitting() -> None:
    with pytest.raises(RuntimeError):
        VarianceComponentCalibrator().transform(np.zeros((1, 6)), ["A"])


def test_logistic_calibrator_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(11)
    logits = rng.normal(size=6000) * 1.4
    labels = (rng.random(6000) < 1.0 / (1.0 + np.exp(-(2.0 * logits + 0.5)))).astype(float)
    fitted = LogisticCalibrator().fit(logits, labels)
    assert fitted.slope == pytest.approx(2.0, abs=0.2)
    assert fitted.intercept == pytest.approx(0.5, abs=0.15)


def test_logistic_state_roundtrip() -> None:
    fitted = LogisticCalibrator(slope=1.3, intercept=-0.2)
    restored = LogisticCalibrator.from_state(fitted.state())
    assert restored.slope == pytest.approx(1.3)


def test_logistic_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError):
        LogisticCalibrator().fit(np.zeros(3), np.zeros(4))


def test_fit_and_apply_calibration_closes_the_loop(tmp_path) -> None:
    from tearfilm_persistence.cli.calibrate import main as calibrate_main

    def _table(path, scores, labels, eyes):
        rows = []
        for index, (score, label, eye) in enumerate(zip(scores, labels, eyes, strict=True)):
            components = [float(index % 3), float(index % 2), 0.5, 0.2, 0.15, 0.02]
            payload = {
                "eye_id": eye,
                "site": "Site A",
                "score": float(score),
                "label": float(label),
            }
            for name, value in zip(
                (
                    "onset",
                    "peak_instability",
                    "curve_slope",
                    "locus_entropy",
                    "tmh",
                    "tma",
                ),
                components,
                strict=True,
            ):
                payload[name] = value
            rows.append(payload)
        path.write_text(
            "\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        return path

    fitting = _table(
        tmp_path / "fit.jsonl",
        [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ["a", "a", "b", "b", "c", "c"],
    )
    applying = _table(
        tmp_path / "apply.jsonl",
        [0.15, 0.85, 0.25, 0.75],
        [0.0, 1.0, 0.0, 1.0],
        ["d", "d", "e", "e"],
    )
    exit_code = calibrate_main(
        ["--fit", str(fitting), "--apply", str(applying), "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0
    report = __import__("json").loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert report["outcome"]["n_records"] == 4
    assert report["outcome"]["n_eyes"] == 2
    assert 0.0 <= report["outcome"]["calibration_error_after"] <= 1.0
