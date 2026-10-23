"""Derivative, persistence operator and curve statistics."""

from __future__ import annotations

import pytest
import torch

from tearfilm_persistence.deformation.curve import locus_entropy, ols_slope, summarise_clip
from tearfilm_persistence.deformation.derivative import DerivativeConfig, TemporalDerivative
from tearfilm_persistence.deformation.persistence import (
    CurveSegment,
    PersistenceConfig,
    PersistenceOperator,
    area_fraction,
    first_passage,
    hard_occupancy,
    soft_occupancy,
    tau_frames,
)


def test_tau_discretises_by_frame_rate() -> None:
    assert tau_frames(0.17, 30.0) == 5
    assert tau_frames(0.17, 60.0) == 10


def test_derivative_of_a_ramp_is_exact_in_the_interior() -> None:
    field = torch.arange(12, dtype=torch.float32).view(1, 1, 12) * 0.4
    derivative = TemporalDerivative()(field, 25.0)
    assert float(derivative[0, 0, 6]) == pytest.approx(0.4 * 25.0, rel=1e-6)


def test_derivative_is_causal() -> None:
    first = torch.zeros(1, 1, 10)
    second = first.clone()
    second[:, :, 8:] = 5.0
    early = TemporalDerivative()(first, 30.0)
    later = TemporalDerivative()(second, 30.0)
    assert torch.allclose(early[:, :, :7], later[:, :, :7])


def test_derivative_window_is_bounded() -> None:
    with pytest.raises(ValueError):
        DerivativeConfig(window=6).validated()
    assert TemporalDerivative(DerivativeConfig(window=3)).kernel.numel() == 3


def test_single_frame_clip_has_zero_derivative() -> None:
    field = torch.ones(1, 2, 1)
    assert torch.equal(TemporalDerivative()(field, 30.0), torch.zeros_like(field))


def test_occupancy_requires_the_window_to_expire() -> None:
    field = torch.ones(1, 4, 10)
    occupancy = hard_occupancy(field, span=4)
    assert not bool(occupancy[0, 0, 3])
    assert bool(occupancy[0, 0, 4])


def test_occupancy_drops_a_reversed_locus() -> None:
    field = torch.ones(1, 2, 10)
    field[:, 1, 6:] = -1.0
    occupancy = hard_occupancy(field, span=3)
    assert bool(occupancy[0, 0, 8])
    assert not bool(occupancy[0, 1, 8])


def test_zero_derivative_is_never_occupied() -> None:
    assert not bool(hard_occupancy(torch.zeros(1, 2, 8), span=2).any())


def test_soft_occupancy_tracks_the_hard_one() -> None:
    field = torch.ones(1, 4, 10)
    hard = hard_occupancy(field, span=3).float()
    soft = soft_occupancy(field, span=3, gamma=20.0)
    assert torch.allclose(hard[:, :, 4:], soft[:, :, 4:], atol=0.02)


def test_soft_occupancy_without_persistence_is_amplitude_only() -> None:
    field = torch.zeros(1, 2, 6)
    field[:, :, 3] = 1.0
    soft = soft_occupancy(field, span=0, gamma=20.0)
    assert float(soft[0, 0, 3]) > 0.9
    assert float(soft[0, 0, 0]) < 0.1


def test_area_fraction_uses_the_masked_loci() -> None:
    occupancy = torch.ones(1, 8, 4)
    mask = torch.tensor([True] * 4 + [False] * 4)
    assert float(area_fraction(occupancy, mask)[0, 0]) == pytest.approx(1.0)
    occupancy[:, 4:] = 0.0
    assert float(area_fraction(occupancy, mask)[0, 0]) == pytest.approx(1.0)


def test_area_fraction_halves_when_half_the_mask_is_empty() -> None:
    occupancy = torch.zeros(1, 8, 4)
    occupancy[:, :4] = 1.0
    mask = torch.ones(8, dtype=torch.bool)
    assert float(area_fraction(occupancy, mask)[0, 0]) == pytest.approx(0.5)


def test_first_passage_reports_absence() -> None:
    curve = torch.tensor([[0.0, 0.01, 0.02]])
    index, found = first_passage(curve, 0.5)
    assert not bool(found.item())
    assert int(index.item()) == 3


def test_first_passage_is_the_first_crossing() -> None:
    curve = torch.tensor([[0.0, 0.3, 0.05, 0.6]])
    index, found = first_passage(curve, 0.2)
    assert bool(found.item())
    assert int(index.item()) == 1


def test_curve_requires_one_clip() -> None:
    operator = PersistenceOperator()
    with pytest.raises(ValueError):
        operator.curve(torch.ones(2, 4, 8), torch.ones(4, dtype=torch.bool), [(0, 8)], 30.0)


def test_operator_returns_one_segment_per_interval() -> None:
    operator = PersistenceOperator(PersistenceConfig(mode="hard"))
    derivative = torch.ones(1, 4, 12)
    segments = operator.curve(derivative, torch.ones(4, dtype=torch.bool), [(0, 6), (6, 12)], 30.0)
    assert len(segments) == 2
    assert segments[1].start_frame == 6


def test_ols_slope_is_per_second() -> None:
    assert ols_slope(torch.tensor([0.0, 0.1, 0.2, 0.3]), fps=10.0) == pytest.approx(1.0)
    assert ols_slope(torch.tensor([0.5]), fps=10.0) == 0.0


def test_entropy_of_a_single_locus_is_zero() -> None:
    occupancy = torch.zeros(4)
    occupancy[2] = 1.0
    assert locus_entropy(occupancy, torch.ones(4, dtype=torch.bool)) == pytest.approx(0.0)


def test_entropy_of_an_even_spread_is_one() -> None:
    occupancy = torch.ones(4)
    assert locus_entropy(occupancy, torch.ones(4, dtype=torch.bool)) == pytest.approx(1.0)


def _segment(values: list[float], loci: int = 4) -> CurveSegment:
    fractions = torch.tensor(values)
    occupancy = torch.ones(loci, len(values))
    return CurveSegment(fractions=fractions, occupancy=occupancy, start_frame=0, fps=30.0)


def test_summary_picks_the_earliest_declaring_subtest() -> None:
    late = _segment([0.0, 0.0, 0.0, 0.5, 0.6])
    early = _segment([0.0, 0.0, 0.3, 0.3, 0.3])
    summary = summarise_clip([late, early], torch.ones(4, dtype=torch.bool), 0.2)
    assert summary.declared
    assert summary.onset_seconds == pytest.approx(2 / 30.0)
    assert summary.n_subtests == 2


def test_summary_without_a_declaration_is_censored() -> None:
    summary = summarise_clip([_segment([0.0, 0.05, 0.05])], torch.ones(4, dtype=torch.bool), 0.5)
    assert not summary.declared
    assert summary.onset_seconds is None
    assert summary.peak == pytest.approx(0.05)


def test_summary_of_no_subtests_is_empty() -> None:
    summary = summarise_clip([], torch.ones(4, dtype=torch.bool), 0.2)
    assert summary.n_subtests == 0
    assert not summary.declared
