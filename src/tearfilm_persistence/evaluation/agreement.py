"""Agreement between a model grade and a reference grade.

Ref: Table 4 (severity kappa per site and per stratum), Table 6 (inter-reader
kappa), Sec. 5.8 (agreement measures reported with bias, limits of agreement and
an intraclass correlation).

Severity is ordinal, so the kappa is weighted linearly: a one-stratum
disagreement costs half of a two-stratum one, and a quadratic weighting would
overstate the cost of the extremes on a three-level scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_kappa(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str] | None = None,
    weighting: str = "linear",
) -> float:
    a = list(first)
    b = list(second)
    if len(a) != len(b):
        raise ValueError("the two ratings must cover the same units")
    if not a:
        raise ValueError("an empty rating set has no agreement")
    order = list(categories) if categories is not None else sorted(set(a) | set(b))
    if len(order) < 2:
        return 1.0
    index = {value: position for position, value in enumerate(order)}
    unknown = (set(a) | set(b)) - set(index)
    if unknown:
        raise ValueError(f"ratings outside the category list: {sorted(unknown)}")
    size = len(order)
    observed = np.zeros((size, size), dtype=np.float64)
    for left, right in zip(a, b, strict=True):
        observed[index[left], index[right]] += 1.0
    observed /= observed.sum()
    row = observed.sum(axis=1)
    column = observed.sum(axis=0)
    expected = np.outer(row, column)
    distances = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) / (size - 1)
    weights = 1.0 - distances if weighting == "linear" else 1.0 - distances**2
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if abs(1.0 - denominator) < 1e-12:
        return 1.0
    return (numerator - denominator) / (1.0 - denominator)


def severity_levels() -> tuple[str, str, str]:
    return ("mild", "moderate", "severe")


def severity_kappa(predicted: Sequence[str], reference: Sequence[str]) -> float:
    return weighted_kappa(predicted, reference, severity_levels())
