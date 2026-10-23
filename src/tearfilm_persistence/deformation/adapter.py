"""Temporal-difference adaptation of a clip trunk for the fast branch.

Ref: Sec. 5.4, the fast branch reads a pretrained spatio-temporal prior at
clip-level cost and is adjusted through a temporal-difference operator rather
than by full fine-tuning.

The trunk is windowed: it never sees more than five consecutive frames, which is
the same horizon the derivative uses. The adapter adds the frame difference back
onto the trunk response, so the branch is derivative-aware before the explicit
operator is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AdapterConfig:
    in_channels: int = 64
    hidden_channels: int = 64
    window: int = 5
    residual_scale: float = 0.5

    def validated(self) -> AdapterConfig:
        if not 2 <= self.window <= 5:
            raise ValueError("the clip window is at most five frames")
        return self


class WindowedClipTrunk(nn.Module):
    """A small 3D trunk with a causal temporal receptive field."""

    def __init__(self, in_channels: int = 3, widths: tuple[int, int, int] = (24, 48, 64)) -> None:
        super().__init__()
        w0, w1, w2 = widths
        self.entry = nn.Conv3d(in_channels, w0, (3, 5, 5), padding=(0, 2, 2), bias=False)
        self.entry_norm = nn.GroupNorm(8, w0)
        self.down_one = nn.Conv3d(
            w0, w1, (3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False
        )
        self.norm_one = nn.GroupNorm(8, w1)
        self.down_two = nn.Conv3d(
            w1, w2, (3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False
        )
        self.norm_two = nn.GroupNorm(8, w2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T, H, W] -> [B, C, T, h, w]`` with causal time padding."""
        x = F.pad(frames, (0, 0, 0, 0, 2, 0), mode="replicate")
        entered: torch.Tensor = self.act(self.entry_norm(self.entry(x)))
        first: torch.Tensor = self.act(self.norm_one(self.down_one(entered)))
        second: torch.Tensor = self.act(self.norm_two(self.down_two(first)))
        return second


class TemporalDifferenceAdapter(nn.Module):
    """Blends a trunk response with its own backward temporal difference."""

    def __init__(self, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = (config or AdapterConfig()).validated()
        self.project = nn.Conv3d(
            self.config.in_channels, self.config.hidden_channels, 1, bias=False
        )
        self.norm = nn.GroupNorm(8, self.config.hidden_channels)
        self.mix = nn.Parameter(torch.tensor(self.config.residual_scale))
        self.act = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[2] < 2:
            return features
        difference = torch.zeros_like(features)
        difference[:, :, 1:] = features[:, :, 1:] - features[:, :, :-1]
        delta = self.act(self.norm(self.project(difference)))
        adapted: torch.Tensor = features + torch.sigmoid(self.mix) * delta
        return adapted


class TemporalDifferenceBackbone(nn.Module):
    """Fast-branch backbone: windowed trunk plus its difference adapter."""

    def __init__(self, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = (config or AdapterConfig()).validated()
        self.trunk = WindowedClipTrunk(in_channels=3, widths=(24, 48, self.config.in_channels))
        self.adapter = TemporalDifferenceAdapter(self.config)
        self.field_head = nn.Conv3d(self.config.hidden_channels, 1, 1)
        self.field_head.bias = nn.Parameter(torch.zeros(1))

    def load_trunk(self, path: str | Path) -> None:
        """Load trunk weights and freeze the trunk; the adapter stays trainable."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.trunk.load_state_dict(state.get("trunk", state))
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, 3, T, H, W] -> field [B, 1, T, h, w], features [B, C, T, h, w]``."""
        if frames.ndim != 5:
            raise ValueError("expected [B, 3, T, H, W]")
        features = self.adapter(self.trunk(frames))
        field = torch.tanh(self.field_head(features))
        return field, features


class PlainClipBackbone(nn.Module):
    """The same windowed trunk without the difference adapter.

    This is the 3D-convolutional fast branch the ablation substitutes for the
    adapted one; it keeps the window and the head and drops the derivative-aware
    path, so the comparison isolates the adaptation.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = (config or AdapterConfig()).validated()
        self.trunk = WindowedClipTrunk(in_channels=3, widths=(24, 48, self.config.in_channels))
        self.field_head = nn.Conv3d(self.config.in_channels, 1, 1)
        self.field_head.bias = nn.Parameter(torch.zeros(1))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError("expected [B, 3, T, H, W]")
        features = self.trunk(frames)
        field = torch.tanh(self.field_head(features))
        return field, features
