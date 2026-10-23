"""Adequacy gate inside the model.

Ref: Sec. 5.2 and Algorithm 1 line 1-3. The gate is the fifth component of the
ablation (C5) and is evaluated before any output exists.

The gate is not differentiable and is never applied during training: the
training clips are already inside the acceptance band, and the gate exists to
withhold an estimate that would be produced from an acquisition it cannot
support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tearfilm_persistence.streams.adequacy import AdequacyConfig, fixation_drift


@dataclass(frozen=True)
class GateOutput:
    accept: torch.Tensor
    reasons: tuple[tuple[str, ...], ...]


class AdequacyGate(nn.Module):
    """Batch-wise adequacy decision over sampling rate and fixation stability."""

    def __init__(self, config: AdequacyConfig | None = None) -> None:
        super().__init__()
        self.config = (config or AdequacyConfig()).validated()

    def forward(
        self,
        fps: torch.Tensor,
        fixation: torch.Tensor,
        frame_mask: torch.Tensor,
        frame_shape: tuple[int, int],
    ) -> GateOutput:
        accept: list[bool] = []
        reasons: list[tuple[str, ...]] = []
        for index in range(fps.shape[0]):
            valid = frame_mask[index]
            track = fixation[index][valid]
            if track.numel() == 0:
                reasons.append(("empty_acquisition",))
                accept.append(False)
                continue
            drift = fixation_drift(track, frame_shape)
            rate_ok = float(fps[index].item()) >= self.config.sampling_floor_hz
            stable_ok = drift <= self.config.max_fixation_drift
            local: list[str] = []
            if not rate_ok:
                local.append("sampling_rate_below_floor")
            if not stable_ok:
                local.append("fixation_stability_outside_band")
            accept.append(not local)
            reasons.append(tuple(local))
        return GateOutput(
            accept=torch.tensor(accept, dtype=torch.bool, device=fps.device),
            reasons=tuple(reasons),
        )
