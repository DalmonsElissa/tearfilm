"""Factorised (2+1)D convolutional design.

Ref: Table 2, row "(2+1)D factorised" [50], family *moderate temporal*.

Each block splits a 3x3x3 convolution into a spatial 3x3 and a temporal 3x1x1,
so the temporal filter is learned separately from the spatial one. This is the
2019-generation design the fast branch replaces by an adapted clip trunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tearfilm_persistence.comparators.tower import ClipHead


class FactorisedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(
            in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.spatial_norm = nn.GroupNorm(8, out_channels)
        self.temporal = nn.Conv3d(
            out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.temporal_norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut: nn.Module
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(8, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.shortcut(x)
        spatial: torch.Tensor = self.act(self.spatial_norm(self.spatial(x)))
        temporal: torch.Tensor = self.temporal_norm(self.temporal(spatial))
        output: torch.Tensor = self.act(temporal + residual)
        return output


class Factorised2Plus1(nn.Module):
    family = "moderate temporal"

    def __init__(self, widths: tuple[int, int] = (24, 48), blocks: int = 2) -> None:
        super().__init__()
        w0, w1 = widths
        self.entry = nn.Sequential(
            nn.Conv3d(3, w0, (3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.GroupNorm(8, w0),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[FactorisedBlock(w0, w0) for _ in range(blocks)])
        self.down = nn.Sequential(
            nn.Conv3d(w0, w1, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, w1),
            nn.SiLU(inplace=True),
        )
        self.head = ClipHead(w1)

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip = frames.transpose(1, 2)
        features = self.down(self.blocks(self.entry(clip)))
        weights = frame_mask.to(features.dtype).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        masked = (features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled = F.adaptive_avg_pool2d(masked, 1).flatten(1)
        logit, onset = self.head(pooled)
        return logit, onset
