"""Causal temporal derivative over windows of at most five frames.

Ref: Sec. 5.4, D_t = d_t phi(x_t), evaluated on windows of up to five
consecutive frames; Algorithm 1 line 5.

The kernel is the first-derivative Savitzky-Golay stencil of a quadratic fit on
five points, (2, 1, 0, -1, -2) / 10 in time order. Shorter histories at the head
of a clip are handled by edge replication, which leaves the derivative at the
first frame at zero instead of inventing a slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

FIVE_POINT: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)


@dataclass(frozen=True)
class DerivativeConfig:
    window: int = 5

    def validated(self) -> DerivativeConfig:
        if not 2 <= self.window <= 5:
            raise ValueError("the derivative window is at most five frames")
        return self


def _kernel(window: int) -> torch.Tensor:
    if window == 5:
        return torch.tensor(FIVE_POINT, dtype=torch.float32).view(1, 1, 5)
    if window == 3:
        return torch.tensor((-0.5, 0.0, 0.5), dtype=torch.float32).view(1, 1, 3)
    if window == 2:
        return torch.tensor((-1.0, 1.0), dtype=torch.float32).view(1, 1, 2)
    raise ValueError(f"unsupported derivative window: {window}")


class TemporalDerivative(torch.nn.Module):
    """Per-second derivative of a locus field, ``[B, P, T] -> [B, P, T]``."""

    kernel: torch.Tensor

    def __init__(self, config: DerivativeConfig | None = None) -> None:
        super().__init__()
        self.config = (config or DerivativeConfig()).validated()
        self.register_buffer("kernel", _kernel(self.config.window), persistent=True)

    def forward(self, field: torch.Tensor, fps: float | torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError("field must be [B, P, T]")
        if field.shape[-1] < 2:
            return torch.zeros_like(field)
        width = self.config.window
        padded = F.pad(field, (width - 1, 0), mode="replicate")
        kernel = self.kernel.to(device=field.device, dtype=field.dtype)
        # one filter per locus: the derivative is taken along time, not across loci
        depthwise = kernel.repeat(field.shape[1], 1, 1)
        derivative = F.conv1d(padded, depthwise, groups=field.shape[1])
        rate = fps if isinstance(fps, torch.Tensor) else torch.tensor(fps, dtype=field.dtype)
        rate = rate.to(device=field.device, dtype=field.dtype).clamp_min(1e-6)
        # finite difference is per frame; physical time scales it by the frame rate
        return derivative * rate.view(-1, 1, 1) if rate.ndim == 1 else derivative * rate

    def extra_repr(self) -> str:
        return f"window={self.config.window}"
