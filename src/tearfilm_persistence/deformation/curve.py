"""Descriptor statistics read off the instability curve.

Ref: Sec. 5.4, z_k = (onset, max_t u_k(t), slope u_k, locus entropy, TMH, TMA).

The four dynamic entries are taken from the curve alone: the declared onset is
its first passage through alpha, the peak is its maximum, the slope is taken
after the declaration (a curve that has not declared has no post-declaration
slope), and the locus entropy is the Shaw entropy of the occupied loci at the
peak, normalised by log P so that it does not depend on the grid size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tearfilm_persistence.deformation.persistence import CurveSegment, first_passage


@dataclass(frozen=True)
class SubtestStats:
    declared: bool
    onset_seconds: float | None
    peak: float
    slope: float
    entropy: float
    peak_time: float
    epoch_seconds: float


@dataclass(frozen=True)
class CurveSummary:
    declared: bool
    onset_seconds: float | None
    peak: float
    slope: float
    entropy: float
    peak_time: float
    n_subtests: int
    epoch_seconds: float
    per_subtest: tuple[SubtestStats, ...]


def ols_slope(values: torch.Tensor, fps: float, start: int = 0) -> float:
    """Least-squares slope per second of a curve from ``start`` to its end."""
    segment = values[start:]
    count = int(segment.numel())
    if count < 2 or fps <= 0:
        return 0.0
    index = torch.arange(count, dtype=segment.dtype) / fps
    centred = index - index.mean()
    denominator = (centred * centred).sum()
    if float(denominator) <= 1e-12:
        return 0.0
    centred_values = segment - segment.mean()
    return float((centred * centred_values).sum() / denominator)


def locus_entropy(occupancy: torch.Tensor, mask: torch.Tensor) -> float:
    """Normalised entropy of the occupied-locus distribution.

    A curve that occupies a single locus returns 0; an even spread over the
    examined area returns 1. When nothing is occupied the entropy is undefined
    and reported as 0 rather than as a maximum.
    """
    weights = occupancy.to(torch.float64) * mask.flatten().to(torch.float64)
    total = weights.sum()
    if float(total) <= 0:
        return 0.0
    share = weights / total
    positive = share[share > 0]
    entropy = -(positive * positive.log()).sum()
    normaliser = torch.log(torch.tensor(float(int(mask.sum().item())), dtype=torch.float64))
    if float(normaliser) <= 0:
        return 0.0
    return float((entropy / normaliser).clamp(0.0, 1.0))


def summarise_segment(segment: CurveSegment, mask: torch.Tensor, alpha: float) -> SubtestStats:
    fractions = segment.fractions
    epoch_seconds = float(fractions.numel() / segment.fps)
    index, found = first_passage(fractions.unsqueeze(0), alpha)
    declared = bool(found.item())
    onset_seconds: float | None = None
    if declared:
        onset_seconds = float(index.item()) / segment.fps
    peak_index = int(torch.argmax(fractions).item())
    slope = ols_slope(fractions, segment.fps, start=int(index.item()) if declared else 0)
    entropy = locus_entropy(segment.occupancy[:, peak_index], mask)
    return SubtestStats(
        declared=declared,
        onset_seconds=onset_seconds,
        peak=float(fractions.max().item()),
        slope=slope,
        entropy=entropy,
        peak_time=peak_index / segment.fps,
        epoch_seconds=epoch_seconds,
    )


def summarise_clip(segments: list[CurveSegment], mask: torch.Tensor, alpha: float) -> CurveSummary:
    """Reduce the sub-tests of one clip to the eye-level curve statistics.

    The declared onset is the earliest sub-test that declares, which is the
    first break-up after any blink. The peak, slope and entropy are read from
    that same sub-test so that they describe one instability event rather than a
    mixture of several.
    """
    if not segments:
        return CurveSummary(
            declared=False,
            onset_seconds=None,
            peak=0.0,
            slope=0.0,
            entropy=0.0,
            peak_time=0.0,
            n_subtests=0,
            epoch_seconds=0.0,
            per_subtest=(),
        )
    stats = [summarise_segment(segment, mask, alpha) for segment in segments]
    declaring = [entry for entry in stats if entry.declared]
    if declaring:
        chosen = min(declaring, key=lambda entry: entry.onset_seconds or 0.0)
        return CurveSummary(
            declared=True,
            onset_seconds=chosen.onset_seconds,
            peak=chosen.peak,
            slope=chosen.slope,
            entropy=chosen.entropy,
            peak_time=chosen.peak_time,
            n_subtests=len(stats),
            epoch_seconds=chosen.epoch_seconds,
            per_subtest=tuple(stats),
        )
    widest = max(stats, key=lambda entry: entry.peak)
    return CurveSummary(
        declared=False,
        onset_seconds=None,
        peak=widest.peak,
        slope=widest.slope,
        entropy=widest.entropy,
        peak_time=widest.peak_time,
        n_subtests=len(stats),
        epoch_seconds=max(entry.epoch_seconds for entry in stats),
        per_subtest=tuple(stats),
    )
