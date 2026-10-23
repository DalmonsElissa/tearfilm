"""Frame-pair dual-task Siamese design.

Ref: Table 2, row "Frame-pair dual-task Siamese" [17], family *pair-wise*.

A shared frame encoder reads two adjacent frames; the pair decision is formed
from the second frame and the signed difference between them, under two heads
(a break-up judgement and an onset regression). The horizon is two frames by
construction, so a transient speck and a growing rupture that share a magnitude
are not separable by this design.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tearfilm_persistence.comparators.tower import ClipHead, FrameTower, TowerConfig


class SiamesePair(nn.Module):
    family = "pair-wise"

    def __init__(self, config: TowerConfig | None = None) -> None:
        super().__init__()
        self.tower = FrameTower(config)
        channels = self.tower.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.SiLU(inplace=True),
        )
        self.head = ClipHead(64)

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = frames.shape[0], frames.shape[1]
        if steps < 2:
            zeros = frames.new_zeros(batch)
            return zeros, F.softplus(zeros) + 0.05
        first = frames[:, :-1].reshape(batch * (steps - 1), *frames.shape[2:])
        second = frames[:, 1:].reshape(batch * (steps - 1), *frames.shape[2:])
        feature_first = self.pool(self.tower(first)).flatten(1)
        feature_second = self.pool(self.tower(second)).flatten(1)
        fused = self.fuse(torch.cat([feature_second, feature_second - feature_first], dim=1))
        fused = fused.reshape(batch, steps - 1, -1)
        weights = frame_mask[:, 1:].to(fused.dtype).unsqueeze(-1)
        pooled = (fused * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logit, onset = self.head(pooled)
        return logit, onset
