"""The descriptor vector and its ordering.

Ref: Sec. 5.4, z_k = (onset, max_t u_k(t), slope u_k, locus entropy, TMH, TMA).

The order is fixed because the calibration map and the reported tables address
the components by position. A censored acquisition keeps the epoch length in the
onset slot so that the vector stays complete, and carries a flag saying that the
onset was not declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from tearfilm_persistence.deformation.curve import CurveSummary

COMPONENTS: Final[tuple[str, ...]] = (
    "onset",
    "peak_instability",
    "curve_slope",
    "locus_entropy",
    "tmh",
    "tma",
)
COMPONENT_UNITS: Final[tuple[str, ...]] = ("s", "fraction", "1/s", "nat", "mm", "mm2")


@dataclass(frozen=True)
class DescriptorVector:
    onset_seconds: float
    peak_instability: float
    curve_slope: float
    locus_entropy: float
    tmh_mm: float
    tma_mm2: float
    declared: bool = True

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.onset_seconds,
                self.peak_instability,
                self.curve_slope,
                self.locus_entropy,
                self.tmh_mm,
                self.tma_mm2,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        values = self.as_array()
        return {name: float(value) for name, value in zip(COMPONENTS, values, strict=True)}

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.as_array(), dtype=torch.float32)

    @classmethod
    def from_summary(
        cls,
        summary: CurveSummary,
        tmh_mm: float,
        tma_mm2: float,
    ) -> DescriptorVector:
        onset = (
            summary.onset_seconds if summary.onset_seconds is not None else summary.epoch_seconds
        )
        return cls(
            onset_seconds=float(onset),
            peak_instability=float(summary.peak),
            curve_slope=float(summary.slope),
            locus_entropy=float(summary.entropy),
            tmh_mm=float(tmh_mm),
            tma_mm2=float(tma_mm2),
            declared=summary.declared,
        )


@dataclass(frozen=True)
class Standardiser:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardiser:
        if matrix.ndim != 2 or matrix.shape[1] != len(COMPONENTS):
            raise ValueError(f"expected an [n, {len(COMPONENTS)}] matrix")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(
            mean=tuple(float(value) for value in mean), scale=tuple(float(value) for value in scale)
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean)
        scale = np.asarray(self.scale)
        result: np.ndarray = (matrix - mean) / scale
        return result

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device)
        scale = torch.tensor(self.scale, dtype=values.dtype, device=values.device)
        return (values - mean) / scale

    def state(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_state(cls, state: dict[str, list[float]]) -> Standardiser:
        return cls(mean=tuple(state["mean"]), scale=tuple(state["scale"]))
