"""Tear-meniscus geometry from a single frame.

Ref: Sec. 5.4 (structural branch, TMH and TMA from one frame), Sec. 5.2 (axial
resolution around 15 micrometres).

The branch is deliberately independent of the temporal path: it reads one frame
and returns a height and an area in physical units, so a low tear reserve and a
rapidly destabilising film remain separable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

AXIAL_UM: float = 15.0
"""Axial resolution quoted in Sec. 5.2."""


@dataclass(frozen=True)
class PixelGeometry:
    axial_um: float = AXIAL_UM
    lateral_um: float = AXIAL_UM

    @property
    def axial_mm(self) -> float:
        return self.axial_um / 1000.0

    @property
    def lateral_mm(self) -> float:
        return self.lateral_um / 1000.0

    @property
    def pixel_area_mm2(self) -> float:
        return self.axial_mm * self.lateral_mm

    def validated(self) -> PixelGeometry:
        if self.axial_um <= 0 or self.lateral_um <= 0:
            raise ValueError("pixel pitch must be positive")
        return self


@dataclass(frozen=True)
class StructuralReading:
    tmh_mm: float
    tma_mm2: float
    coverage: float


class MeniscusSegmenter(nn.Module):
    """Single-frame segmentation of the lower tear meniscus."""

    def __init__(self, width: int = 24, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
        ]
        channels = width
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
                    nn.GroupNorm(8, channels * 2),
                    nn.SiLU(inplace=True),
                ]
            )
            channels *= 2
        self.encoder = nn.Sequential(*layers)
        self.reduce = nn.Conv2d(channels, width * 2, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
        )
        self.logits = nn.Conv2d(width, 1, 1)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W] -> mask logits [B, 1, H, W]``."""
        height, width = frame.shape[-2:]
        features = self.reduce(self.encoder(frame))
        features = F.interpolate(features, size=(height // 2, width // 2), mode="nearest")
        decoded: torch.Tensor = self.decoder(features)
        upsampled = F.interpolate(decoded, size=(height, width), mode="nearest")
        logits: torch.Tensor = self.logits(upsampled)
        return logits


def read_geometry(
    probability: torch.Tensor,
    band: tuple[int, int],
    geometry: PixelGeometry,
) -> StructuralReading:
    """Turn a soft meniscus mask into height and area.

    Height is the mean column-wise thickness inside the band and area is the
    summed probability inside it; both use the axial and lateral pitch, so a
    change of frame size alone does not move the measurement.
    """
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("expected a [B, 1, H, W] mask")
    geom = geometry.validated()
    top, bottom = band
    strip = probability[:, :, top:bottom, :].clamp(0.0, 1.0)
    thickness = strip.sum(dim=2)  # [B, 1, W] of mask pixels per column
    height = float(thickness.mean().item()) * geom.axial_mm
    area = float(strip.sum().item() / strip.shape[0]) * geom.pixel_area_mm2
    coverage = float(strip.mean().item())
    return StructuralReading(tmh_mm=height, tma_mm2=area, coverage=coverage)


class StructuralBranch(nn.Module):
    """Segmenter plus its physical read-out, kept as one differentiable unit."""

    def __init__(self, geometry: PixelGeometry | None = None) -> None:
        super().__init__()
        self.geometry = (geometry or PixelGeometry()).validated()
        self.segmenter = MeniscusSegmenter()

    def forward(
        self, frame: torch.Tensor, band: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tmh_mm, tma_mm2)`` as differentiable scalars per sample."""
        probability = torch.sigmoid(self.segmenter(frame))
        strip = probability[:, :, band[0] : band[1], :]
        per_column = strip.sum(dim=2).squeeze(1) * self.geometry.axial_mm
        height = per_column.mean(dim=1)
        area = strip.sum(dim=(1, 2, 3)) * self.geometry.pixel_area_mm2
        return height, area
