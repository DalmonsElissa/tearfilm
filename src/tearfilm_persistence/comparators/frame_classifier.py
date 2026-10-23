"""Per-frame break-up classifier.

Ref: Table 2, row "Frame classifier", family *per-frame*.

Every frame carries its own decision and the clip score is the mean of the frame
logits. The design has no horizon beyond one frame, which is exactly the
limitation the manuscript attributes to the frame-level literature.
"""

from __future__ import annotations

import torch
from torch import nn

from tearfilm_persistence.comparators.tower import FramePooledComparator, TowerConfig


class FrameClassifier(FramePooledComparator):
    family = "per-frame"

    def __init__(self, config: TowerConfig | None = None) -> None:
        super().__init__(config)
        self.head = nn.Linear(self.tower.out_channels, 2)

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, T, 3, H, W]`` -> clip logit and onset, both ``[B]``."""
        batch, steps = frames.shape[0], frames.shape[1]
        flat = frames.reshape(batch * steps, *frames.shape[2:])
        features = self.pool(self.tower(flat)).flatten(1)
        logits = self.head(features).reshape(batch, steps, 2)
        weights = frame_mask.to(logits.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        decision = (logits[..., 0] * weights).sum(dim=1) / count
        onset = (torch.nn.functional.softplus(logits[..., 1]) * weights).sum(dim=1) / count
        return decision, onset
