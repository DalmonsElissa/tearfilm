"""Descriptor head: measurement vector to decision and onset.

Ref: Sec. 5.4 (a descriptor head assembles the vector), Sec. 5.4 (three
objective terms: a break-up decision, an onset regression, and a repeatability
penalty).

The head reads the six-component vector beside a pooled appearance feature. The
appearance path is what keeps the model usable when the temporal branches are
ablated, which is the behaviour the component table reports: removing a temporal
branch costs discrimination but does not silence the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    n_components: int = 6
    appearance_dim: int = 64
    hidden: int = 64
    dropout: float = 0.0

    def validated(self) -> HeadConfig:
        if self.n_components <= 0 or self.appearance_dim <= 0:
            raise ValueError("component and appearance widths must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must sit in [0, 1)")
        return self


@dataclass(frozen=True)
class HeadOutput:
    logit: torch.Tensor
    onset_seconds: torch.Tensor
    embedding: torch.Tensor


class DescriptorHead(nn.Module):
    """Maps a descriptor vector and a clip feature to a logit and an onset."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = (config or HeadConfig()).validated()
        width = self.config.n_components + self.config.appearance_dim
        self.input_norm = nn.BatchNorm1d(self.config.n_components)
        self.trunk = nn.Sequential(
            nn.Linear(width, self.config.hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.SiLU(inplace=True),
        )
        self.decision = nn.Linear(self.config.hidden, 1)
        self.onset = nn.Linear(self.config.hidden, 1)
        nn.init.zeros_(self.decision.bias)
        nn.init.zeros_(self.onset.bias)

    def forward(self, descriptors: torch.Tensor, appearance: torch.Tensor) -> HeadOutput:
        if descriptors.ndim != 2 or descriptors.shape[1] != self.config.n_components:
            raise ValueError("descriptors must be [B, 6]")
        normalised = self.input_norm(descriptors)
        embedding = self.trunk(torch.cat([normalised, appearance], dim=-1))
        logit = self.decision(embedding).squeeze(-1)
        onset = torch.nn.functional.softplus(self.onset(embedding).squeeze(-1)) + 0.05
        return HeadOutput(logit=logit, onset_seconds=onset, embedding=embedding)
