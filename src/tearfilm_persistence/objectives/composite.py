"""The three objective terms.

Ref: Sec. 5.4. The integrated objective has a binary cross-entropy over the
break-up judgement, an onset regression term on the sub-tests that carry a
reference onset, and a repeatability penalty on the descriptor variance across
repeated acquisitions of the same eye.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def decision_term(
    logit: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy over the declared break-up judgement."""
    if logit.shape != label.shape or logit.shape != valid.shape:
        raise ValueError("logit, label and valid must share a shape")
    if not bool(valid.any()):
        return logit.sum() * 0.0
    weight = torch.where(
        label > 0.5, torch.full_like(logit, positive_weight), torch.ones_like(logit)
    )
    per_item = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    return (per_item * weight * valid.to(logit.dtype)).sum() / valid.to(logit.dtype).sum()


def onset_term(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Robust regression on the reference onset, over uncensored sub-tests.

    The paper names the term without fixing the norm; smooth L1 with a half-second
    transition is the release default and is recorded as such.
    """
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must share a shape")
    mask = valid.to(prediction.dtype)
    if float(mask.sum().item()) <= 0:
        return prediction.sum() * 0.0
    per_item = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (per_item * mask).sum() / mask.sum()


def repeatability_term(
    descriptors: torch.Tensor,
    groups: torch.Tensor,
    minimum_group_size: int = 2,
) -> torch.Tensor:
    """One minus the mean eye-level reliability of the descriptor components.

    Reliability is the between-eye share of total variance. It is scale free, so
    the term cannot be driven to zero by shrinking the descriptor, and it is
    bounded, so a single outlier repeat cannot dominate the objective.
    """
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be [B, C]")
    if groups.shape[0] != descriptors.shape[0]:
        raise ValueError("one group label per row is required")
    unique = torch.unique(groups)
    keep = torch.stack([(groups == value).sum() >= minimum_group_size for value in unique])
    selected = unique[keep]
    if selected.numel() < 2:
        return descriptors.sum() * 0.0
    means = torch.stack([descriptors[groups == value].mean(dim=0) for value in selected])
    sizes = torch.stack([(groups == value).sum().to(descriptors.dtype) for value in selected])
    total_count = sizes.sum()
    grand = (means * sizes.unsqueeze(1)).sum(dim=0) / total_count
    between = (sizes.unsqueeze(1) * (means - grand) ** 2).sum(dim=0) / total_count
    total = descriptors.var(dim=0, unbiased=False).clamp_min(1e-8)
    reliability = (between / total).clamp(0.0, 1.0)
    return 1.0 - reliability.mean()


@dataclass(frozen=True)
class ObjectiveWeights:
    decision: float = 1.0
    onset: float = 1.0
    repeatability: float = 0.1
    positive_weight: float = 1.0
    onset_beta: float = 0.5

    def validated(self) -> ObjectiveWeights:
        if self.decision < 0 or self.onset < 0 or self.repeatability < 0:
            raise ValueError("objective weights must be non-negative")
        if self.positive_weight <= 0:
            raise ValueError("positive weight must be positive")
        return self


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    decision: torch.Tensor
    onset: torch.Tensor
    repeatability: torch.Tensor


def composite_loss(
    logit: torch.Tensor,
    label: torch.Tensor,
    valid_decision: torch.Tensor,
    onset_prediction: torch.Tensor,
    onset_target: torch.Tensor,
    valid_onset: torch.Tensor,
    descriptors: torch.Tensor,
    groups: torch.Tensor,
    weights: ObjectiveWeights | None = None,
    use_repeatability: bool = True,
) -> LossBreakdown:
    cfg = (weights or ObjectiveWeights()).validated()
    decision = decision_term(logit, label, valid_decision, cfg.positive_weight)
    onset = onset_term(onset_prediction, onset_target, valid_onset, cfg.onset_beta)
    penalty = (
        repeatability_term(descriptors, groups) if use_repeatability else descriptors.sum() * 0.0
    )
    total = cfg.decision * decision + cfg.onset * onset + cfg.repeatability * penalty
    return LossBreakdown(total=total, decision=decision, onset=onset, repeatability=penalty)
