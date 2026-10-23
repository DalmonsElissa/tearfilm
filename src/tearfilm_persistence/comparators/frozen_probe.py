"""Frozen video foundation model with a linear probe.

Ref: Table 2, row "Frozen video foundation model + probe", family *transfer*.

The trunk is frozen and only a linear probe is fitted. The trunk weights are not
distributed with this release, so a probe can only be fitted once a checkpoint
is supplied; without one the row reports no result rather than a probe over an
untrained trunk.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from tearfilm_persistence.deformation.adapter import WindowedClipTrunk


class FrozenVideoProbe(nn.Module):
    family = "transfer"

    def __init__(self, channels: int = 64, load: bool = True) -> None:
        super().__init__()
        self.trunk = WindowedClipTrunk(in_channels=3, widths=(24, 48, channels))
        self.probe = nn.Linear(channels, 2)
        self.pretrained_available = False
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        if not load:
            self.trunk.eval()

    def load_backbone(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.trunk.load_state_dict(state.get("trunk", state))
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        self.pretrained_available = True
        self.trunk.eval()

    def train(self, mode: bool = True) -> FrozenVideoProbe:
        super().train(mode)
        self.trunk.eval()
        self.probe.train(mode)
        return self

    def forward(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.trunk(frames.transpose(1, 2))
        # the trunk shortens the temporal axis, so the frame mask is resampled to it
        weights = (
            F.interpolate(
                frame_mask.to(features.dtype).unsqueeze(1), size=features.shape[2], mode="nearest"
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        pooled = (features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
        out = self.probe(pooled)
        return out[:, 0], F.softplus(out[:, 1]) + 0.05
