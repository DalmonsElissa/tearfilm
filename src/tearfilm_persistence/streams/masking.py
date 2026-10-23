"""Analysed area and locus grid.

Ref: Sec. 5.2 (a structural descriptor resolved into 25 to 100 elements across
the tear meniscus), Sec. 5.4 (the area A over which the affected fraction is
taken).

The persistence operator counts loci, so the locus grid is the unit of area. The
grid is built once from the frame geometry and then reused for every frame and
every clip of the same acquisition geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AreaConfig:
    rows: int = 8
    cols: int = 8
    x_span: tuple[float, float] = (0.10, 0.90)
    y_span: tuple[float, float] = (0.55, 0.95)
    shape: str = "ellipse"

    def validated(self) -> AreaConfig:
        if not 25 <= self.rows * self.cols <= 100:
            raise ValueError("the locus count must sit between 25 and 100")
        if self.shape not in {"ellipse", "rectangle"}:
            raise ValueError(f"unknown area shape: {self.shape}")
        if not 0 <= self.x_span[0] < self.x_span[1] <= 1:
            raise ValueError("x span must sit inside the frame")
        if not 0 <= self.y_span[0] < self.y_span[1] <= 1:
            raise ValueError("y span must sit inside the frame")
        return self


@dataclass(frozen=True)
class AnalysedArea:
    """A rectangular or elliptical band resolved into a locus grid."""

    config: AreaConfig
    mask: torch.Tensor

    @property
    def rows(self) -> int:
        return self.config.rows

    @property
    def cols(self) -> int:
        return self.config.cols

    @property
    def n_loci(self) -> int:
        return self.rows * self.cols

    @property
    def area(self) -> int:
        return int(self.mask.sum().item())

    def pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Resample a feature map onto the locus grid, ``[B, C, h, w] -> [B, C, P]``.

        Loci outside the mask are zeroed rather than dropped, so that a mask can
        be applied downstream without changing the tensor width.
        """
        if feature_map.ndim != 4:
            raise ValueError("expected a [B, C, h, w] feature map")
        pooled = F.adaptive_avg_pool2d(feature_map, (self.rows, self.cols))
        flat = pooled.flatten(2)
        return flat * self.mask.flatten().to(flat.dtype).unsqueeze(0).unsqueeze(0)

    def to(self, device: torch.device | str) -> AnalysedArea:
        return AnalysedArea(config=self.config, mask=self.mask.to(device))


def build_area(config: AreaConfig | None = None) -> AnalysedArea:
    cfg = (config or AreaConfig()).validated()
    rows = torch.arange(cfg.rows, dtype=torch.float32) + 0.5
    cols = torch.arange(cfg.cols, dtype=torch.float32) + 0.5
    yy = (rows / cfg.rows).unsqueeze(1).expand(cfg.rows, cfg.cols)
    xx = (cols / cfg.cols).unsqueeze(0).expand(cfg.rows, cfg.cols)
    if cfg.shape == "rectangle":
        mask = torch.ones((cfg.rows, cfg.cols), dtype=torch.bool)
    else:
        radius = ((xx - 0.5) / 0.5).pow(2) + ((yy - 0.5) / 0.5).pow(2)
        mask = radius <= 1.0
    if int(mask.sum().item()) == 0:
        raise ValueError("the area mask is empty")
    return AnalysedArea(config=cfg, mask=mask)


def band_pixels(config: AreaConfig, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel bounds ``(top, bottom, left, right)`` of the band inside a frame."""
    height, width = frame_shape
    top = round(config.y_span[0] * height)
    bottom = round(config.y_span[1] * height)
    left = round(config.x_span[0] * width)
    right = round(config.x_span[1] * width)
    return top, bottom, left, right
