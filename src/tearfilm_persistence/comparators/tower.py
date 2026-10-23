"""Shared convolutional tower for the single-frame comparators.

The frame classifier and the mean-pooled network differ in where they collapse
the temporal axis, not in what reads a frame, so they share this tower and pass
different pooling to it. Keeping one tower avoids the two rows of the comparison
table drifting apart for any reason other than their pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TowerConfig:
    widths: tuple[int, int, int] = (32, 64, 96)

    def validated(self) -> TowerConfig:
        if len(self.widths) != 3:
            raise ValueError("the tower has three stages")
        return self


class FrameTower(nn.Module):
    """Maps one frame to a spatial feature map at one eighth of its size."""

    def __init__(self, config: TowerConfig | None = None) -> None:
        super().__init__()
        self.config = (config or TowerConfig()).validated()
        w0, w1, w2 = self.config.widths
        self.stages = nn.Sequential(
            nn.Conv2d(3, w0, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, w0),
            nn.SiLU(inplace=True),
            nn.Conv2d(w0, w1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, w1),
            nn.SiLU(inplace=True),
            nn.Conv2d(w1, w2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, w2),
            nn.SiLU(inplace=True),
        )

    @property
    def out_channels(self) -> int:
        return self.config.widths[-1]

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.stages(frame)
        return features


class ClipHead(nn.Module):
    """Shared decision and onset read-out for a pooled clip feature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.decision = nn.Linear(in_channels, 1)
        self.onset = nn.Linear(in_channels, 1)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit: torch.Tensor = self.decision(pooled).squeeze(-1)
        onset: torch.Tensor = F.softplus(self.onset(pooled)).squeeze(-1) + 0.05
        return logit, onset


class FramePooledComparator(nn.Module):
    """Shared construction for the two comparators that read single frames."""

    def __init__(self, config: TowerConfig | None = None) -> None:
        super().__init__()
        self.tower = FrameTower(config)
        self.pool = nn.AdaptiveAvgPool2d(1)
