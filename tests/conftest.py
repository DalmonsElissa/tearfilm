"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from tearfilm_persistence.streams.acquisition import AcquisitionClip, PaddedClipBatch, stack_clips
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


@pytest.fixture()
def clip() -> AcquisitionClip:
    return make_clip()


@pytest.fixture()
def batch() -> PaddedClipBatch:
    return stack_clips(
        [
            make_clip(seed=0, eye_id="eye-0", n_frames=14),
            make_clip(seed=1, eye_id="eye-1", n_frames=11, site="Site B"),
            make_clip(seed=2, eye_id="eye-2", n_frames=12, reference_onset=None),
        ]
    )
