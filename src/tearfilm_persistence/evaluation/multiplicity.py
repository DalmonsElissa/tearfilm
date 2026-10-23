"""Holm step-down adjustment.

Ref: Sec. 5.8. The primary endpoint family carries four pre-specified criteria
and is corrected by Holm's step-down method; everything else is descriptive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HolmResult:
    names: tuple[str, ...]
    raw: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]


def holm_stepdown(pvalues: dict[str, float], alpha: float = 0.05) -> HolmResult:
    if not pvalues:
        raise ValueError("at least one p-value is required")
    names = tuple(pvalues)
    raw = np.asarray([pvalues[name] for name in names], dtype=np.float64)
    if np.any((raw < 0) | (raw > 1) | ~np.isfinite(raw)):
        raise ValueError("p-values must sit in [0, 1]")
    order = np.argsort(raw, kind="stable")
    count = raw.size
    adjusted_sorted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = (count - rank) * raw[position]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha
    return HolmResult(
        names=names,
        raw=tuple(float(value) for value in raw),
        adjusted=tuple(float(value) for value in adjusted),
        rejected=tuple(bool(value) for value in rejected),
    )
