"""Blink segmentation into sub-test intervals.

Ref: Sec. 5.6, S_k = (b_k, b_{k+1}].

A blink is a joint event: the ocular-surface contrast collapses while the
inter-frame motion energy spikes. Requiring both keeps a slow contrast drift
and a single dropped frame out of the event list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tearfilm_persistence.streams.acquisition import AcquisitionClip


@dataclass(frozen=True)
class BlinkConfig:
    """Thresholds for the photometric blink detector.

    ``min_duration`` and ``max_duration`` are physical, in seconds, so the same
    values hold across sampling rates.
    """

    contrast_drop: float = 0.60
    motion_ratio: float = 5.0
    min_duration: float = 0.05
    max_duration: float = 0.80
    band: tuple[float, float] = (0.35, 0.95)

    def validated(self) -> BlinkConfig:
        if self.min_duration <= 0 or self.max_duration <= self.min_duration:
            raise ValueError("blink durations must satisfy 0 < min < max")
        if not 0.0 <= self.band[0] < self.band[1] <= 1.0:
            raise ValueError("band must sit inside [0, 1]")
        return self


def _contrast_collapse(contrast: torch.Tensor, drop: float) -> torch.Tensor:
    """Frames whose contrast falls to a fraction of the clip's upper reference.

    A global z-score is wrong here: a monotone contrast ramp inflates its own
    scale and hides the collapse. The upper quartile is taken as the reference
    the blink departs from.
    """
    reference = torch.quantile(contrast, 0.75)
    if float(reference) <= 1e-6:
        return torch.zeros_like(contrast, dtype=torch.bool)
    return contrast <= (1.0 - drop) * reference


def _motion_spike(motion: torch.Tensor, ratio: float) -> torch.Tensor:
    reference = motion.median().clamp_min(1e-6)
    return motion >= ratio * reference


def blink_signals(
    clip: AcquisitionClip, band: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-frame contrast and motion-energy signals of a clip."""
    gray = clip.to_grayscale()
    n_frames, height, _ = gray.shape
    top = round(band[0] * height)
    bottom = round(band[1] * height)
    strip = gray[:, top:bottom, :]
    contrast = strip.flatten(1).std(dim=1, unbiased=False)
    if n_frames == 1:
        motion = torch.zeros_like(contrast)
    else:
        steps = (gray[1:] - gray[:-1]).abs().flatten(1).mean(dim=1)
        motion = torch.cat([steps[:1], steps])
    return contrast, motion


def detect_blinks(clip: AcquisitionClip, config: BlinkConfig | None = None) -> torch.Tensor:
    """Return the frame indices at which blinks occur, ascending."""
    cfg = (config or BlinkConfig()).validated()
    contrast, motion = blink_signals(clip, cfg.band)
    if clip.n_frames < 3:
        return torch.empty(0, dtype=torch.long)
    collapse = _contrast_collapse(contrast, cfg.contrast_drop)
    spike = _motion_spike(motion, cfg.motion_ratio)
    runs = _runs(collapse)
    fps = clip.fps
    blinks: list[int] = []
    for start, stop in runs:
        duration = (stop - start + 1) / fps
        if not cfg.min_duration <= duration <= cfg.max_duration:
            continue
        boundary = max(start, 0)
        edges = [boundary, stop, min(stop + 1, int(spike.numel()) - 1)]
        if not any(bool(spike[index]) for index in edges):
            continue
        window = contrast[start : stop + 1]
        blinks.append(start + int(torch.argmin(window).item()))
    return torch.tensor(blinks, dtype=torch.long)


def _runs(flags: torch.Tensor) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(int(flags.numel())):
        active = bool(flags[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, int(flags.numel()) - 1))
    return runs


def subtest_intervals(blinks: torch.Tensor, n_frames: int) -> list[tuple[int, int]]:
    """Split a clip into half-open sub-test intervals ``(b_k, b_{k+1}]``.

    The reference index of each interval is its own blink frame, which is where
    the persistence clock starts.
    """
    if n_frames <= 0:
        return []
    boundaries = [int(index) for index in blinks.tolist() if 0 <= int(index) < n_frames]
    boundaries = sorted(set(boundaries))
    if not boundaries or boundaries[0] != 0:
        boundaries = [0, *boundaries]
    intervals: list[tuple[int, int]] = []
    for position, start in enumerate(boundaries):
        stop = boundaries[position + 1] if position + 1 < len(boundaries) else n_frames
        if stop - start >= 2:
            intervals.append((start, stop))
    return intervals
