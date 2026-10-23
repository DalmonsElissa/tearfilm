"""Model, comparators and objectives."""

from __future__ import annotations

import pytest
import torch

from tearfilm_persistence.comparators import COMPARATORS, build_comparator
from tearfilm_persistence.comparators.adapter import ComparatorAdapter
from tearfilm_persistence.descriptors.vector import COMPONENTS
from tearfilm_persistence.models.head import DescriptorHead, HeadConfig
from tearfilm_persistence.models.teardyn import ModelConfig, TearDynModel, artefact_fraction
from tearfilm_persistence.objectives.composite import (
    ObjectiveWeights,
    composite_loss,
    decision_term,
    onset_term,
    repeatability_term,
)
from tearfilm_persistence.streams.acquisition import stack_clips
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


def test_forward_returns_one_row_per_clip(batch) -> None:
    model = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        output = model(batch)
    assert output.logit.shape == (3,)
    assert output.descriptor_matrix.shape == (3, len(COMPONENTS))
    assert output.artefact_rate.shape == (3,)
    assert len(output.descriptors) == 3


def test_descriptor_matrix_matches_the_vectors(batch) -> None:
    model = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        output = model(batch)
    for index, vector in enumerate(output.descriptors):
        assert output.descriptor_matrix[index, 0].item() == pytest.approx(vector.onset_seconds)


def test_removing_the_fast_branch_zeroes_the_dynamic_components(batch) -> None:
    model = TearDynModel(ModelConfig(use_fast=False)).eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.allclose(output.descriptor_matrix[:, 1:3], torch.zeros(3, 2), atol=1e-6)


def test_removing_the_structural_branch_zeroes_the_height_and_area(batch) -> None:
    model = TearDynModel(ModelConfig(use_structural=False)).eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.allclose(output.descriptor_matrix[:, 4:], torch.zeros(3, 2), atol=1e-6)


def test_gate_abstains_inside_the_model() -> None:
    good = make_clip(seed=0, eye_id="a")
    slow = make_clip(seed=1, eye_id="b", fps=12.0)
    model = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        output = model(stack_clips([good, slow]))
    assert output.abstain.tolist() == [False, True]
    assert output.gate_reasons[1] == ("sampling_rate_below_floor",)


def test_gate_can_be_withheld() -> None:
    slow = make_clip(seed=1, fps=12.0)
    model = TearDynModel(ModelConfig(use_gate=False)).eval()
    with torch.no_grad():
        output = model(stack_clips([slow]))
    assert not bool(output.abstain.any())


def test_mean_pool_substitution_flattens_the_curve(batch) -> None:
    model = TearDynModel(ModelConfig(persistence_substitution="mean_pool")).eval()
    with torch.no_grad():
        output = model(batch)
    curves = [summary.per_subtest[0].peak for summary in output.summaries if summary.per_subtest]
    assert len(curves) == len(output.summaries)


def test_conv3d_branch_runs(batch) -> None:
    model = TearDynModel(ModelConfig(fast_branch="conv3d")).eval()
    with torch.no_grad():
        output = model(batch)
    assert output.probability.shape == (3,)


def test_same_seed_gives_the_same_logits(batch) -> None:
    torch.manual_seed(4)
    first = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        a = first(batch).logit.clone()
    torch.manual_seed(4)
    second = TearDynModel(ModelConfig()).eval()
    with torch.no_grad():
        b = second(batch).logit.clone()
    assert torch.equal(a, b)


def test_gradients_reach_every_branch(batch) -> None:
    model = TearDynModel(ModelConfig())
    model.train()
    output = model(batch)
    (output.logit.sum() + output.soft_descriptor_matrix.sum()).backward()
    adapter = [p for p in model.backbone.adapter.parameters() if p.grad is not None]
    structural = [p for p in model.structural.parameters() if p.grad is not None]
    head = [p for p in model.head.parameters() if p.grad is not None]
    assert adapter and structural and head


def test_artefact_channel_is_bounded() -> None:
    derivative = torch.ones(1, 8, 12)
    derivative[:, :4, 2:] = -1.0
    value = artefact_fraction(derivative[0], torch.ones(8, dtype=torch.bool), 4)
    assert value == pytest.approx(0.5)


def test_head_returns_a_positive_onset() -> None:
    head = DescriptorHead(HeadConfig(appearance_dim=8)).eval()
    descriptors = torch.zeros(2, len(COMPONENTS))
    appearance = torch.zeros(2, 8)
    with torch.no_grad():
        output = head(descriptors, appearance)
    assert float(output.onset_seconds.min()) > 0


@pytest.mark.parametrize("name", COMPARATORS)
def test_comparator_returns_clip_level_pair(name: str, batch) -> None:
    model = build_comparator(name).eval()
    with torch.no_grad():
        logit, onset = model(batch.frames, batch.frame_mask)
    assert logit.shape == (3,)
    assert onset.shape == (3,)
    assert torch.isfinite(logit).all()


def test_comparator_adapter_matches_the_batch_interface(batch) -> None:
    adapter = ComparatorAdapter("siamese_pair").eval()
    with torch.no_grad():
        output = adapter(batch)
    assert output.logit.shape == (3,)
    assert output.descriptor_matrix.shape == (3, len(COMPONENTS))
    assert not bool(output.abstain.any())


def test_comparator_families_are_declared() -> None:
    families = {build_comparator(name).family for name in COMPARATORS}
    assert families == {
        "per-frame",
        "weak temporal",
        "moderate temporal",
        "pair-wise",
        "transfer",
        "full attention",
    }


def test_decision_term_masks_invalid_rows() -> None:
    logit = torch.tensor([0.2, -0.4, 1.0])
    label = torch.tensor([1.0, 0.0, 0.0])
    valid = torch.tensor([True, True, False])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logit[:2], label[:2])
    assert decision_term(logit, label, valid).item() == pytest.approx(expected.item())


def test_onset_term_is_zero_at_a_perfect_prediction() -> None:
    target = torch.tensor([2.0, 3.0])
    assert onset_term(target, target.clone(), torch.tensor([True, True])).item() == pytest.approx(
        0.0
    )


def test_repeatability_term_prefers_clean_repeats() -> None:
    clean = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
    noisy = torch.tensor([[1.0], [1.8], [2.0], [2.8]])
    groups = torch.tensor([0, 0, 1, 1])
    assert repeatability_term(clean, groups).item() < repeatability_term(noisy, groups).item()


def test_repeatability_term_needs_two_groups() -> None:
    values = torch.tensor([[1.0], [2.0]])
    value = repeatability_term(values, torch.tensor([0, 0]))
    assert value.item() == pytest.approx(0.0)


def test_composite_loss_is_the_weighted_sum() -> None:
    logit = torch.tensor([0.1, -0.2, 0.4])
    label = torch.tensor([1.0, 0.0, 1.0])
    valid = torch.ones(3, dtype=torch.bool)
    onset = torch.tensor([2.0, 2.5, 3.0])
    descriptors = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
    breakdown = composite_loss(
        logit,
        label,
        valid,
        onset,
        onset.clone(),
        valid,
        descriptors,
        torch.tensor([0, 0, 1, 1]),
        ObjectiveWeights(decision=0.5, onset=2.0, repeatability=0.0),
    )
    expected = 0.5 * breakdown.decision.item() + 2.0 * breakdown.onset.item()
    assert breakdown.total.item() == pytest.approx(expected)


def test_objective_weights_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        ObjectiveWeights(decision=-1.0).validated()
