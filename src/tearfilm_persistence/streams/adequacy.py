"""Acquisition-adequacy gate.

Ref: Sec. 5.2 (joint sampling and stability criterion), Algorithm 1 line 1-3.

The gate is evaluated before any model output exists, so a rejection here is an
abstention, not a low-confidence prediction. Both conditions must hold jointly:
the sampling floor and fixation stability inside the acceptance band.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tearfilm_persistence.streams.acquisition import AcquisitionClip

SAMPLING_FLOOR_HZ: float = 30.0
"""Inclusion floor from Sec. 5.2: 60 and 30 frames per second are
indistinguishable, lower rates degrade the measurement."""


@dataclass(frozen=True)
class AdequacyConfig:
    sampling_floor_hz: float = SAMPLING_FLOOR_HZ
    max_fixation_drift: float = 0.055
    min_valid_fraction: float = 0.90

    def validated(self) -> AdequacyConfig:
        if self.sampling_floor_hz <= 0:
            raise ValueError("sampling floor must be positive")
        if not 0 < self.max_fixation_drift < 1:
            raise ValueError("fixation drift is a fraction of the frame and must sit in (0, 1)")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("valid fraction must sit in (0, 1]")
        return self


@dataclass(frozen=True)
class AdequacyDecision:
    accepted: bool
    sampling_ok: bool
    fixation_ok: bool
    fixation_drift: float
    reasons: tuple[str, ...]

    @property
    def abstained(self) -> bool:
        return not self.accepted


def fixation_drift(fixation: torch.Tensor, frame_shape: tuple[int, int]) -> float:
    """Radial spread of the ocular-surface centroid, in frame-width units.

    The track is expressed in normalised coordinates, so the anisotropy of the
    frame is removed by scaling the vertical axis by the aspect ratio.
    """
    if fixation.shape[0] < 2:
        return 0.0
    height, width = frame_shape
    aspect = (width / height) if height else 1.0
    scaled = fixation.clone()
    scaled[:, 1] = scaled[:, 1] / aspect
    centre = scaled.median(dim=0).values
    radial = (scaled - centre).pow(2).sum(dim=1).sqrt()
    return float(radial.quantile(0.95).item())


def assess(clip: AcquisitionClip, config: AdequacyConfig | None = None) -> AdequacyDecision:
    cfg = (config or AdequacyConfig()).validated()
    sampling_ok = clip.fps >= cfg.sampling_floor_hz
    drift = fixation_drift(clip.fixation, clip.frame_shape)
    fixation_ok = drift <= cfg.max_fixation_drift
    reasons: list[str] = []
    if not sampling_ok:
        reasons.append("sampling_rate_below_floor")
    if not fixation_ok:
        reasons.append("fixation_stability_outside_band")
    if clip.n_frames < 2:
        reasons.append("clip_too_short")
    return AdequacyDecision(
        accepted=not reasons,
        sampling_ok=sampling_ok,
        fixation_ok=fixation_ok,
        fixation_drift=drift,
        reasons=tuple(reasons),
    )


def gate_rate(decisions: list[AdequacyDecision]) -> float:
    """Proportion of attempted acquisitions that met the joint criterion."""
    if not decisions:
        raise ValueError("no decisions to summarise")
    return sum(1.0 for decision in decisions if decision.accepted) / len(decisions)
