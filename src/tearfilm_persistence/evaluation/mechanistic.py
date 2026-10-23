"""Mechanistic arm read-outs.

Ref: Sec. 2.9 and Sec. 5.10. Hyperosmolar against iso-osmolar corneal epithelial
cultures, three independent experiments, percent change in secreted mediators,
fold change in phospho-to-total ratio and nuclear translocation, and a decline in
transepithelial electrical resistance.

The arm is reported as partial: the manuscript prints its summary values without
the replicate table, so this module computes the statistics from replicates
supplied by the caller and never fills them in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectEstimate:
    name: str
    control_mean: float
    treated_mean: float
    percent_change: float
    lower: float
    upper: float
    replicates: int


def _bootstrap_interval(
    statistic: np.ndarray,
    level: float = 0.95,
) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(statistic, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _replicate_arrays(
    control: Sequence[float], treated: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(control, dtype=np.float64)
    b = np.asarray(treated, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("at least two replicates per arm are required")
    return a, b


def percent_change(
    control: Sequence[float],
    treated: Sequence[float],
    name: str = "readout",
    n_boot: int = 2000,
    seed: int = 0,
) -> EffectEstimate:
    """Percent change of treated over control with a percentile bootstrap."""
    a, b = _replicate_arrays(control, treated)
    if np.any(a == 0):
        raise ValueError("a zero control replicate has no percent change")
    point = float(100.0 * (b.mean() - a.mean()) / a.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample_a = rng.choice(a, size=a.size, replace=True)
        sample_b = rng.choice(b, size=b.size, replace=True)
        draws[index] = 100.0 * (sample_b.mean() - sample_a.mean()) / sample_a.mean()
    lower, upper = _bootstrap_interval(draws)
    return EffectEstimate(
        name=name,
        control_mean=float(a.mean()),
        treated_mean=float(b.mean()),
        percent_change=point,
        lower=lower,
        upper=upper,
        replicates=int(min(a.size, b.size)),
    )


def fold_change(
    control: Sequence[float],
    treated: Sequence[float],
    name: str = "fold",
    n_boot: int = 2000,
    seed: int = 0,
) -> EffectEstimate:
    """Ratio of treated to control, expressed as a fold change."""
    a, b = _replicate_arrays(control, treated)
    point = float(b.mean() / a.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        draws[index] = (
            rng.choice(b, size=b.size, replace=True).mean()
            / rng.choice(a, size=a.size, replace=True).mean()
        )
    lower, upper = _bootstrap_interval(draws)
    return EffectEstimate(
        name=name,
        control_mean=float(a.mean()),
        treated_mean=float(b.mean()),
        percent_change=point,
        lower=lower,
        upper=upper,
        replicates=int(min(a.size, b.size)),
    )


def summarise(
    readouts: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, EffectEstimate]:
    return {
        name: percent_change(control, treated, name=name, n_boot=n_boot, seed=seed + offset)
        for offset, (name, (control, treated)) in enumerate(readouts.items())
    }
