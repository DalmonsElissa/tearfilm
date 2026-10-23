"""Mean-pooled convolutional network.

Ref: Table 2, row "Mean-pooled CNN", family *weak temporal*.

Frames are pooled before the decision, so the network sees a temporal average
but no ordering. The difference from the frame classifier is only the pooling
site, which is what makes the pair a clean test of whether averaging alone buys
anything.
"""

from __future__ import annotations

import torch

from tearfilm_persistence.comparators.tower import ClipHead, FramePooledComparator, TowerConfig


class MeanPooledCNN(FramePooledComparator):
    family = "weak temporal"

    def __init__(self, config: TowerConfig | None = None) -> None:
        super().__init__(config)
        self.head = ClipHead(self.tower.out_channels)

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = frames.shape[0], frames.shape[1]
        flat = frames.reshape(batch * steps, *frames.shape[2:])
        features = self.pool(self.tower(flat)).flatten(1).reshape(batch, steps, -1)
        weights = frame_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logit, onset = self.head(pooled)
        return logit, onset
