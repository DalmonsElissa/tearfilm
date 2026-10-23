"""The persistence operator on a local deformation derivative.

Ref: Sec. 5.4 and Algorithm 1 line 9-13.

    Pi_tau(D | b_k : t) = { p : sign D_t(p) is constant on [max(b_k, t - tau_f), t]
                            and t >= b_k + tau_f }

The occupancy is a set of loci. Its size, divided by the examined area, is the
instability curve u_k(t); the declared onset is the first passage of u_k through
the affected-area fraction alpha.

Two evaluators are provided. ``hard`` is the operator as written and is what the
reported descriptor uses. ``soft`` replaces the sign-equality test by a
sigmoid of the consecutive derivative product, which is what the training
objective differentiates; the two agree as the sharpness grows, and the release
records the relaxation as a deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

TAU_SECONDS: float = 0.17
"""Persistence window fixed in Sec. 5.4 (tau* = 0.17 s)."""


@dataclass(frozen=True)
class PersistenceConfig:
    tau_seconds: float = TAU_SECONDS
    area_fraction: float = 0.10
    gamma: float = 20.0
    mode: str = "hard"

    def validated(self) -> PersistenceConfig:
        if self.tau_seconds <= 0:
            raise ValueError("the persistence window must be positive")
        if not 0 < self.area_fraction <= 1:
            raise ValueError("the affected-area fraction must sit in (0, 1]")
        if self.gamma <= 0:
            raise ValueError("the relaxation sharpness must be positive")
        if self.mode not in {"hard", "soft"}:
            raise ValueError(f"unknown persistence mode: {self.mode}")
        return self


def tau_frames(tau_seconds: float, fps: float) -> int:
    """Discretise the persistence window; at least two frame intervals."""
    return max(2, round(tau_seconds * fps))


def _run_length(signs: torch.Tensor) -> torch.Tensor:
    """Length of the same-sign run ending at each position, ``[B, P, T]``."""
    batch, loci, steps = signs.shape
    lengths = signs.new_zeros((batch, loci, steps), dtype=torch.long)
    if steps == 0:
        return lengths
    lengths[..., 0] = (signs[..., 0] != 0).long()
    for index in range(1, steps):
        same = (signs[..., index] == signs[..., index - 1]) & (signs[..., index] != 0)
        lengths[..., index] = torch.where(
            same, lengths[..., index - 1] + 1, (signs[..., index] != 0).long()
        )
    return lengths


def hard_occupancy(derivative: torch.Tensor, span: int) -> torch.Tensor:
    """Boolean occupancy for one sub-test, ``[B, P, T]``.

    ``span`` is tau_f in frames: the window ``[t - span, t]`` holds ``span + 1``
    samples, so a locus is occupied once its same-sign run reaches ``span + 1``
    and the persistence period has expired.
    """
    if derivative.ndim != 3:
        raise ValueError("derivative must be [B, P, T]")
    signs = torch.sign(derivative)
    lengths = _run_length(signs)
    elapsed = torch.arange(derivative.shape[-1], device=derivative.device).view(1, 1, -1)
    matured = elapsed >= span
    long_enough = lengths >= span + 1
    return long_enough & matured


def soft_occupancy(derivative: torch.Tensor, span: int, gamma: float) -> torch.Tensor:
    """Differentiable occupancy for one sub-test, ``[B, P, T]`` in ``[0, 1]``."""
    if derivative.ndim != 3:
        raise ValueError("derivative must be [B, P, T]")
    steps = derivative.shape[-1]
    if span <= 0:
        return 1.0 - torch.exp(-gamma * derivative.pow(2))
    if steps < 2:
        return derivative.new_zeros(derivative.shape)
    pair = torch.sigmoid(gamma * derivative[..., 1:] * derivative[..., :-1]).clamp_min(1e-7)
    log_pairs = pair.log()
    prefix = F.pad(torch.cumsum(log_pairs, dim=-1), (1, 0))
    positions = torch.arange(steps, device=derivative.device)
    lower = (positions - span).clamp_min(0)
    window_sum = prefix[..., positions] - prefix[..., lower]
    matured = (positions >= span).to(derivative.dtype)
    return torch.exp(window_sum) * matured.view(1, 1, -1)


def area_fraction(occupancy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fraction of the examined area that is occupied, ``[B, P, T] -> [B, T]``."""
    if occupancy.ndim != 3:
        raise ValueError("occupancy must be [B, P, T]")
    weights = mask.flatten().to(occupancy.dtype).unsqueeze(0).unsqueeze(-1)
    total = weights.sum().clamp_min(1.0)
    return (occupancy.to(occupancy.dtype) * weights).sum(dim=1) / total


def first_passage(curve: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First index at which ``curve`` reaches ``alpha``.

    Returns ``(index, found)``; ``index`` is clamped to the curve length when the
    threshold is never reached, and ``found`` says whether it was.
    """
    if curve.ndim != 2:
        raise ValueError("curve must be [B, T]")
    reached = curve >= alpha
    any_reached = reached.any(dim=-1)
    raw = reached.to(torch.int8).argmax(dim=-1)
    index = torch.where(any_reached, raw, torch.full_like(raw, curve.shape[-1]))
    return index, any_reached


@dataclass(frozen=True)
class CurveSegment:
    """Instability curve of one sub-test on its own frame grid.

    ``occupancy`` is kept beside the curve because the locus entropy is read at
    the peak time and recomputing the operator there would double the cost.
    """

    fractions: torch.Tensor
    occupancy: torch.Tensor
    start_frame: int
    fps: float

    @property
    def values(self) -> torch.Tensor:
        return self.fractions

    def time_grid(self) -> torch.Tensor:
        steps = self.fractions.shape[-1]
        return torch.arange(steps, dtype=self.fractions.dtype) / self.fps


class PersistenceOperator:
    """Applies the operator and returns curves for every sub-test of a clip."""

    def __init__(self, config: PersistenceConfig | None = None) -> None:
        self.config = (config or PersistenceConfig()).validated()

    def occupancy(self, derivative: torch.Tensor, span: int) -> torch.Tensor:
        if self.config.mode == "hard":
            return hard_occupancy(derivative, span).to(derivative.dtype)
        return soft_occupancy(derivative, span, self.config.gamma)

    def curve(
        self,
        derivative: torch.Tensor,
        mask: torch.Tensor,
        intervals: list[tuple[int, int]],
        fps: float,
    ) -> list[CurveSegment]:
        if derivative.ndim != 3:
            raise ValueError("derivative must be [B, P, T]")
        if derivative.shape[0] != 1:
            raise ValueError("one clip at a time: the epoch grid differs between clips")
        span = tau_frames(self.config.tau_seconds, fps)
        segments: list[CurveSegment] = []
        for start, stop in intervals:
            window = derivative[:, :, start:stop]
            occupancy = self.occupancy(window, span)
            fractions = area_fraction(occupancy, mask)[0]
            segments.append(
                CurveSegment(
                    fractions=fractions,
                    occupancy=occupancy[0],
                    start_frame=start,
                    fps=fps,
                )
            )
        return segments
