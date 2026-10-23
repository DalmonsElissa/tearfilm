"""Blink segmentation and sub-test intervals."""

from __future__ import annotations

import itertools

import torch

from tearfilm_persistence.streams.blinks import (
    BlinkConfig,
    blink_signals,
    detect_blinks,
    subtest_intervals,
)
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


def test_blink_is_detected_at_the_collapse(clip) -> None:
    blinks = detect_blinks(clip)
    assert blinks.tolist() == [2]


def test_intervals_partition_the_clip(clip) -> None:
    blinks = detect_blinks(clip)
    intervals = subtest_intervals(blinks, clip.n_frames)
    assert intervals[0][0] == 0
    assert intervals[-1][1] == clip.n_frames
    for left, right in itertools.pairwise(intervals):
        assert left[1] == right[0]


def test_no_blink_yields_one_interval() -> None:
    clip = make_clip(n_frames=10)
    flat = clip.subset(0, 1)
    assert subtest_intervals(torch.empty(0, dtype=torch.long), flat.n_frames) == []


def test_short_runs_are_rejected() -> None:
    clip = make_clip()
    config = BlinkConfig(min_duration=0.5, max_duration=0.6)
    assert detect_blinks(clip, config).numel() == 0


def test_contrast_and_motion_signals_have_frame_length(clip) -> None:
    contrast, motion = blink_signals(clip, (0.35, 0.95))
    assert contrast.shape == (clip.n_frames,)
    assert motion.shape == (clip.n_frames,)
    assert float(contrast[2]) < float(contrast[8])


def test_config_rejects_inverted_durations() -> None:
    import pytest

    with pytest.raises(ValueError):
        BlinkConfig(min_duration=0.8, max_duration=0.2).validated()
