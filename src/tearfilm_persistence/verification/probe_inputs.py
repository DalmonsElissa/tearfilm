"""In-process inputs for the execution checks.

Each builder returns a tensor of a known shape with a known structure: a frame
stack whose luminance steps once after a closure, a derivative field with a set
sign pattern, a short batch. They exist so the measurement chain can be
exercised without a cohort. Nothing here is a record, and nothing is written to
disk.
"""

from __future__ import annotations

import torch

from tearfilm_persistence.streams.acquisition import (
    AcquisitionClip,
    ClipMeta,
    PaddedClipBatch,
    stack_clips,
)

PROBE_SIZE = 32
PROBE_FPS = 30.0


def probe_clip(
    seed: int = 0,
    n_frames: int = 14,
    size: int = PROBE_SIZE,
    fps: float = PROBE_FPS,
    ramp: bool = True,
    eye_id: str = "eye-0",
    site: str = "Site A",
    reference_onset: float | None = 6.0,
    severity: str = "moderate",
    fixation_spread: float = 0.01,
) -> AcquisitionClip:
    """A 32-pixel clip whose band brightens once after a two-frame closure."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand((3, size, size), generator=generator) * 0.05
    band = torch.zeros((size, size))
    band[size // 2 : size // 2 + 5, :] = 1.0
    frames = []
    for index in range(n_frames):
        if index in (2, 3):
            frame = torch.full((3, size, size), 0.02)
        else:
            level = 0.55 + (
                0.20 * (index - 3) / max(n_frames - 4, 1) if ramp and index > 3 else 0.0
            )
            frame = base + level * band
        frames.append(frame)
    stack = torch.stack(frames)
    offsets = torch.randn((n_frames, 2), generator=generator) * fixation_spread
    meta = ClipMeta(
        eye_id=eye_id,
        patient_id=f"patient-{eye_id}",
        site=site,
        region="Region I",
        session="s1",
        repeat=1,
        arm="retrospective",
        reference_onset=reference_onset,
        reference_breakup=reference_onset,
        symptom_score=18.0,
        severity=severity,
        device_generation="g1",
    )
    return AcquisitionClip(frames=stack, fps=fps, fixation=offsets, meta=meta)


def probe_batch(count: int = 4, n_frames: int = 14, size: int = PROBE_SIZE) -> PaddedClipBatch:
    clips = []
    for index in range(count):
        clips.append(
            probe_clip(
                seed=index,
                n_frames=n_frames - index,
                size=size,
                eye_id=f"eye-{index}",
                site="Site A" if index % 2 == 0 else "Site B",
                reference_onset=6.0 if index < count // 2 else None,
                severity="mild" if index % 2 == 0 else "severe",
            )
        )
    return stack_clips(clips)


def step_derivative(shape: tuple[int, int, int], sign: float = 1.0, start: int = 0) -> torch.Tensor:
    """A derivative field that is zero before ``start`` and constant after."""
    batch, loci, steps = shape
    field = torch.zeros((batch, loci, steps))
    if start < steps:
        field[:, :, start:] = sign
    return field


def ramp_derivative(shape: tuple[int, int, int], slope: float = 0.5) -> torch.Tensor:
    batch, loci, steps = shape
    time = torch.arange(steps, dtype=torch.float32).view(1, 1, steps)
    return (slope * time).expand(batch, loci, steps).clone()


def flipping_derivative(shape: tuple[int, int, int], flip_at: int) -> torch.Tensor:
    batch, loci, steps = shape
    field = torch.ones((batch, loci, steps))
    if flip_at < steps:
        field[:, :, flip_at:] = -1.0
    return field


def alternating_derivative(shape: tuple[int, int, int], period: int = 2) -> torch.Tensor:
    batch, loci, steps = shape
    time = torch.arange(steps).view(1, 1, steps)
    sign = torch.where((time // period) % 2 == 0, 1.0, -1.0)
    return sign.expand(batch, loci, steps).clone()


def uniform_mask(loci: int) -> torch.Tensor:
    return torch.ones(loci, dtype=torch.bool)


def partial_mask(loci: int, keep: int) -> torch.Tensor:
    mask = torch.zeros(loci, dtype=torch.bool)
    mask[:keep] = True
    return mask
