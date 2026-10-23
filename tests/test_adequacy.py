"""Adequacy gate and the area grid."""

from __future__ import annotations

import pytest
import torch

from tearfilm_persistence.streams.adequacy import AdequacyConfig, assess, fixation_drift, gate_rate
from tearfilm_persistence.streams.masking import AreaConfig, band_pixels, build_area
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


def test_sampling_floor_rejects_slow_acquisition() -> None:
    decision = assess(make_clip(fps=25.0))
    assert not decision.accepted
    assert not decision.sampling_ok
    assert "sampling_rate_below_floor" in decision.reasons


def test_floor_is_inclusive() -> None:
    assert assess(make_clip(fps=30.0)).sampling_ok


def test_unstable_fixation_rejects() -> None:
    decision = assess(make_clip(fixation_spread=0.3))
    assert not decision.fixation_ok
    assert "fixation_stability_outside_band" in decision.reasons


def test_adequate_acquisition_is_accepted() -> None:
    assert assess(make_clip()).accepted


def test_gate_rate_counts_accepted_fraction() -> None:
    decisions = [assess(make_clip(fps=rate)) for rate in (60.0, 30.0, 20.0, 15.0)]
    assert gate_rate(decisions) == pytest.approx(0.5)


def test_fixation_drift_scales_with_spread() -> None:
    tight = fixation_drift(torch.zeros(10, 2), (32, 32))
    loose = fixation_drift(torch.randn(10, 2) * 0.2, (32, 32))
    assert tight < loose


def test_area_grid_has_the_declared_locus_count() -> None:
    area = build_area(AreaConfig(rows=5, cols=6))
    assert area.n_loci == 30


def test_area_rejects_out_of_range_locus_count() -> None:
    with pytest.raises(ValueError):
        AreaConfig(rows=3, cols=4).validated()


def test_pooling_keeps_the_masked_width() -> None:
    area = build_area(AreaConfig(rows=5, cols=5))
    feature = torch.ones(2, 4, 16, 16)
    pooled = area.pool(feature)
    assert pooled.shape == (2, 4, 25)
    assert float(pooled[0, 0].sum()) == pytest.approx(float(area.area))
    assert float(pooled[0, 0].max()) == pytest.approx(1.0)


def test_band_pixels_stay_inside_the_frame() -> None:
    top, bottom, left, right = band_pixels(AreaConfig(), (64, 48))
    assert 0 <= top < bottom <= 64
    assert 0 <= left < right <= 48


def test_adequacy_config_rejects_a_drift_above_one() -> None:
    with pytest.raises(ValueError):
        AdequacyConfig(max_fixation_drift=1.4).validated()
