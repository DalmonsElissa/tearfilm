"""Acquisition container and batching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tearfilm_persistence.streams.acquisition import (
    AcquisitionClip,
    ClipMeta,
    load_clip,
    save_clip,
    stack_clips,
)
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


def test_clip_rejects_wrong_frame_rank() -> None:
    with pytest.raises(ValueError):
        AcquisitionClip(
            frames=torch.zeros(4, 32, 32),
            fps=30.0,
            fixation=torch.zeros(4, 2),
            meta=ClipMeta("e", "p", "A", "I", "s1", 1, "retrospective"),
        )


def test_clip_rejects_fixation_length_mismatch() -> None:
    with pytest.raises(ValueError):
        AcquisitionClip(
            frames=torch.zeros(4, 3, 32, 32),
            fps=30.0,
            fixation=torch.zeros(3, 2),
            meta=ClipMeta("e", "p", "A", "I", "s1", 1, "retrospective"),
        )


def test_clip_duration_uses_the_sampling_rate(clip: AcquisitionClip) -> None:
    assert clip.duration == pytest.approx(clip.n_frames / 30.0)


def test_subset_keeps_rate_and_meta(clip: AcquisitionClip) -> None:
    cropped = clip.subset(2, 8)
    assert cropped.n_frames == 6
    assert cropped.fps == clip.fps
    assert cropped.meta.eye_id == clip.meta.eye_id


def test_grayscale_reduces_channels(clip: AcquisitionClip) -> None:
    gray = clip.to_grayscale()
    assert gray.shape == (clip.n_frames, *clip.frame_shape)


def test_save_and_load_roundtrip(tmp_path, clip: AcquisitionClip) -> None:
    path = tmp_path / "clip.npz"
    save_clip(clip, path)
    restored = load_clip(path)
    assert restored.n_frames == clip.n_frames
    assert torch.allclose(restored.frames, clip.frames)
    assert restored.meta.eye_id == clip.meta.eye_id
    assert restored.meta.reference_onset == clip.meta.reference_onset


def test_load_without_sidecar_uses_anonymous_identity(tmp_path, clip: AcquisitionClip) -> None:
    path = tmp_path / "anon.npz"
    np.savez_compressed(
        path,
        frames=clip.frames.numpy(),
        fps=np.asarray([clip.fps], dtype=np.float32),
        fixation=clip.fixation.numpy(),
    )
    restored = load_clip(path)
    assert restored.meta.eye_id == "anon"
    assert restored.meta.site == "unknown"


def test_load_reports_missing_arrays(tmp_path) -> None:
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, frames=np.zeros((4, 3, 8, 8), dtype=np.float32))
    with pytest.raises(KeyError):
        load_clip(path)


def test_stack_pads_and_masks() -> None:
    short = make_clip(seed=0, n_frames=8)
    long = make_clip(seed=1, n_frames=12, eye_id="eye-1")
    stacked = stack_clips([short, long])
    assert stacked.frames.shape[1] == 12
    assert stacked.frame_mask.sum(dim=1).tolist() == [8, 12]
    assert int(stacked.lengths[0]) == 8


def test_stack_refuses_empty() -> None:
    with pytest.raises(ValueError):
        stack_clips([])
