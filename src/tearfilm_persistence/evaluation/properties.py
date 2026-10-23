"""Properties the detection table reports beside discrimination.

Ref: Table 2 (false early, cross-site change), Sec. 2.7 (calibration), Sec.
5.9 (pre-specified thresholds).

False early is a property of two paired onsets, not of a score: an eye counts
when the declared onset precedes the reference by at least the margin. Cross-site
change is a property of the training partition: the model is fitted on three
sites and read on the fourth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARLY_MARGIN_SECONDS: float = 3.0
"""Table 2 definition: declared onset at least 3 s earlier than the reference."""


@dataclass(frozen=True)
class SiteDiscrimination:
    site: str
    eyes: int
    auc: float
    lower: float
    upper: float
    sensitivity: float
    specificity: float


def false_early_rate(
    declared: np.ndarray,
    reference: np.ndarray,
    margin: float = EARLY_MARGIN_SECONDS,
) -> float:
    """Share of scorable eyes declared at least ``margin`` seconds too early."""
    predicted = np.asarray(declared, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("declared and reference onsets must align")
    usable = np.isfinite(predicted) & np.isfinite(truth)
    if not usable.any():
        return float("nan")
    return float(((predicted[usable] - truth[usable]) <= -margin).mean())


def cross_site_change(holdout: dict[str, float], pooled: dict[str, float]) -> float:
    """Mean absolute AUC change between the leave-one-site-out and pooled fits."""
    if set(holdout) != set(pooled):
        raise ValueError("both mappings must cover the same sites")
    if not holdout:
        raise ValueError("at least one site is required")
    changes = [abs(holdout[site] - pooled[site]) for site in sorted(holdout)]
    return float(np.mean(changes))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    """Equal-width expected calibration error over the unit interval."""
    scores = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if scores.shape != truth.shape:
        raise ValueError("probabilities and labels must align")
    if scores.size == 0:
        raise ValueError("an empty sample has no calibration error")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        if not in_bin.any():
            continue
        gap = abs(float(truth[in_bin].mean()) - float(scores[in_bin].mean()))
        total += gap * float(in_bin.sum()) / float(scores.size)
    return float(total)


def calibration_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> list[tuple[float, float, int, float]]:
    """Per-bin ``(mean score, observed rate, count, gap)`` for inspection."""
    scores = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[tuple[float, float, int, float]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        if not in_bin.any():
            continue
        mean_score = float(scores[in_bin].mean())
        observed = float(truth[in_bin].mean())
        rows.append((mean_score, observed, int(in_bin.sum()), abs(observed - mean_score)))
    return rows


def prevalence(labels: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.float64) > 0.5
    return float(truth.mean())


def severity_strata(severities: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(severities):
        groups.setdefault(value, []).append(index)
    return groups
