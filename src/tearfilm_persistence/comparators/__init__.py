"""Comparator models bound to the rows of the detection table."""

from __future__ import annotations

from typing import Final

from torch import nn

from tearfilm_persistence.comparators.factorised_2plus1 import Factorised2Plus1
from tearfilm_persistence.comparators.frame_classifier import FrameClassifier
from tearfilm_persistence.comparators.frozen_probe import FrozenVideoProbe
from tearfilm_persistence.comparators.mean_pooled import MeanPooledCNN
from tearfilm_persistence.comparators.siamese_pair import SiamesePair
from tearfilm_persistence.comparators.video_transformer import VideoTransformer

COMPARATORS: Final[tuple[str, ...]] = (
    "frame_classifier",
    "mean_pooled",
    "factorised_2plus1",
    "siamese_pair",
    "frozen_probe",
    "video_transformer",
)


def build_comparator(name: str) -> nn.Module:
    builders: dict[str, type[nn.Module]] = {
        "frame_classifier": FrameClassifier,
        "mean_pooled": MeanPooledCNN,
        "factorised_2plus1": Factorised2Plus1,
        "siamese_pair": SiamesePair,
        "frozen_probe": FrozenVideoProbe,
        "video_transformer": VideoTransformer,
    }
    if name not in builders:
        raise KeyError(f"unknown comparator: {name}")
    return builders[name]()


__all__ = [
    "COMPARATORS",
    "Factorised2Plus1",
    "FrameClassifier",
    "FrozenVideoProbe",
    "MeanPooledCNN",
    "SiamesePair",
    "VideoTransformer",
    "build_comparator",
]
