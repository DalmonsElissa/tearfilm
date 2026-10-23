"""Acquisition side of the measurement chain."""

from tearfilm_persistence.streams.adequacy import (
    AdequacyConfig,
    AdequacyDecision,
    assess,
    gate_rate,
)
from tearfilm_persistence.streams.blinks import BlinkConfig, detect_blinks, subtest_intervals
from tearfilm_persistence.streams.masking import AnalysedArea, AreaConfig, build_area

__all__ = [
    "AdequacyConfig",
    "AdequacyDecision",
    "AnalysedArea",
    "AreaConfig",
    "BlinkConfig",
    "assess",
    "build_area",
    "detect_blinks",
    "gate_rate",
    "subtest_intervals",
]
