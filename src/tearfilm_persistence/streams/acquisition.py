"""Acquisition container for one robotic anterior-segment stream.

Ref: Sec. 5.2 (acquisition), Sec. 5.6 (problem statement, X = {x_t}).

A clip is a frame stack sampled at a known rate plus the fixation track that the
adequacy criterion needs, and the clinical reference values the eye carries.
Nothing here resamples frames: the sampling floor is an inclusion criterion, so
a clip that arrives below it is rejected rather than repaired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

FRAME_KEYS: Final[tuple[str, ...]] = ("frames", "fps", "fixation")


@dataclass(frozen=True)
class ClipMeta:
    """Identity and clinical reference of one acquisition."""

    eye_id: str
    patient_id: str
    site: str
    region: str
    session: str
    repeat: int
    arm: str
    reference_onset: float | None = None
    reference_breakup: float | None = None
    symptom_score: float | None = None
    severity: str | None = None
    device_generation: str | None = None

    def with_reference(self, onset: float | None, breakup: float | None) -> ClipMeta:
        return ClipMeta(
            eye_id=self.eye_id,
            patient_id=self.patient_id,
            site=self.site,
            region=self.region,
            session=self.session,
            repeat=self.repeat,
            arm=self.arm,
            reference_onset=onset,
            reference_breakup=breakup,
            symptom_score=self.symptom_score,
            severity=self.severity,
            device_generation=self.device_generation,
        )


@dataclass
class AcquisitionClip:
    """A frame stack with its sampling rate and fixation track.

    ``frames`` is ``[T, 3, H, W]`` in the unit interval. ``fixation`` is the
    per-frame ocular-surface centroid in normalised coordinates, ``[T, 2]``,
    measured from the acquisition alone before any model output exists.
    """

    frames: torch.Tensor
    fps: float
    fixation: torch.Tensor
    meta: ClipMeta
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.ndim != 4:
            raise ValueError(f"frames must be [T, 3, H, W], got shape {tuple(self.frames.shape)}")
        if self.frames.shape[1] != 3:
            raise ValueError("frames must carry three colour channels")
        if self.fixation.shape != (self.frames.shape[0], 2):
            raise ValueError(
                f"fixation must be [T, 2] with T={self.frames.shape[0]}, "
                f"got {tuple(self.fixation.shape)}"
            )
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self.frames.shape[2]), int(self.frames.shape[3])

    def subset(self, start: int, stop: int) -> AcquisitionClip:
        """Crop the clip to a half-open frame range without touching its rate."""
        return AcquisitionClip(
            frames=self.frames[start:stop].clone(),
            fps=self.fps,
            fixation=self.fixation[start:stop].clone(),
            meta=self.meta,
            extras=dict(self.extras),
        )

    def to_grayscale(self) -> torch.Tensor:
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=self.frames.dtype)
        return torch.einsum("tchw,c->thw", self.frames, weights)


def _meta_from_mapping(payload: dict[str, Any]) -> ClipMeta:
    return ClipMeta(
        eye_id=str(payload["eye_id"]),
        patient_id=str(payload["patient_id"]),
        site=str(payload["site"]),
        region=str(payload.get("region", "")),
        session=str(payload.get("session", "s1")),
        repeat=int(payload.get("repeat", 1)),
        arm=str(payload.get("arm", "retrospective")),
        reference_onset=_optional_float(payload.get("reference_onset")),
        reference_breakup=_optional_float(payload.get("reference_breakup")),
        symptom_score=_optional_float(payload.get("symptom_score")),
        severity=_optional_str(payload.get("severity")),
        device_generation=_optional_str(payload.get("device_generation")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def load_clip(path: str | Path) -> AcquisitionClip:
    """Read one ``.npz`` clip plus its sidecar metadata.

    The array file holds ``frames``, ``fps`` and ``fixation``; the ``.json``
    sidecar beside it holds the clip identity. A clip without a sidecar carries
    anonymous identifiers so that a stream can be read before it is linked.
    """
    clip_path = Path(path)
    with np.load(clip_path) as handle:
        missing = [key for key in FRAME_KEYS if key not in handle]
        if missing:
            raise KeyError(f"{clip_path.name} is missing arrays: {', '.join(missing)}")
        frames = torch.from_numpy(np.asarray(handle["frames"], dtype=np.float32))
        fps = float(np.asarray(handle["fps"]).reshape(-1)[0])
        fixation = torch.from_numpy(np.asarray(handle["fixation"], dtype=np.float32))

    sidecar = clip_path.with_suffix(".json")
    if sidecar.exists():
        meta = _meta_from_mapping(json.loads(sidecar.read_text(encoding="utf-8")))
    else:
        meta = ClipMeta(
            eye_id=clip_path.stem,
            patient_id=clip_path.stem,
            site="unknown",
            region="",
            session="s1",
            repeat=1,
            arm="retrospective",
        )
    return AcquisitionClip(frames=frames, fps=fps, fixation=fixation, meta=meta)


def save_clip(clip: AcquisitionClip, path: str | Path) -> Path:
    """Write a clip in the layout :func:`load_clip` reads."""
    clip_path = Path(path)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        clip_path,
        frames=clip.frames.detach().cpu().numpy().astype(np.float32),
        fps=np.asarray([clip.fps], dtype=np.float32),
        fixation=clip.fixation.detach().cpu().numpy().astype(np.float32),
    )
    sidecar = clip_path.with_suffix(".json")
    sidecar.write_text(json.dumps(_meta_to_mapping(clip.meta), indent=2), encoding="utf-8")
    return clip_path


def _meta_to_mapping(meta: ClipMeta) -> dict[str, Any]:
    return {
        "eye_id": meta.eye_id,
        "patient_id": meta.patient_id,
        "site": meta.site,
        "region": meta.region,
        "session": meta.session,
        "repeat": meta.repeat,
        "arm": meta.arm,
        "reference_onset": meta.reference_onset,
        "reference_breakup": meta.reference_breakup,
        "symptom_score": meta.symptom_score,
        "severity": meta.severity,
        "device_generation": meta.device_generation,
    }


@dataclass
class PaddedClipBatch:
    """Variable-length clips stacked to a common frame count.

    ``frame_mask`` is ``[B, T]`` and is true for real frames; every temporal
    reduction in the pipeline multiplies by it so that padding never enters a
    derivative or an area fraction.
    """

    frames: torch.Tensor
    fixation: torch.Tensor
    fps: torch.Tensor
    frame_mask: torch.Tensor
    metas: list[ClipMeta]
    lengths: torch.Tensor


def stack_clips(clips: list[AcquisitionClip]) -> PaddedClipBatch:
    if not clips:
        raise ValueError("cannot stack an empty clip list")
    lengths = [clip.n_frames for clip in clips]
    max_len = max(lengths)
    channels, height, width = clips[0].frames.shape[1:]
    frames = clips[0].frames.new_zeros((len(clips), max_len, channels, height, width))
    fixation = clips[0].fixation.new_zeros((len(clips), max_len, 2))
    mask = clips[0].frames.new_zeros((len(clips), max_len), dtype=torch.bool)
    for index, clip in enumerate(clips):
        length = clip.n_frames
        frames[index, :length] = clip.frames
        fixation[index, :length] = clip.fixation
        mask[index, :length] = True
    return PaddedClipBatch(
        frames=frames,
        fixation=fixation,
        fps=torch.tensor([clip.fps for clip in clips], dtype=torch.float32),
        frame_mask=mask,
        metas=[clip.meta for clip in clips],
        lengths=torch.tensor(lengths, dtype=torch.long),
    )
