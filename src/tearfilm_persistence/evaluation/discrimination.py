"""Discrimination with patient-level cluster bootstrap.

Ref: Sec. 5.8, discrimination is reported as the area under the curve with 95%
intervals from a patient-level cluster bootstrap.

The area is the Mann-Whitney statistic with mid-ranks for ties, so a score
vector with repeated values is not pushed toward either class. The bootstrap
resamples patients, not eyes, which is what the interval has to reflect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class DiscriminationResult:
    auc: float
    lower: float
    upper: float
    n_positive: int
    n_negative: int
    n_clusters: int


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    youden: float


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if values.shape != truth.shape or values.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional and aligned")
    positive = truth > 0.5
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(values)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _cluster_index(
    groups: np.ndarray, seed: int
) -> tuple[np.ndarray, list[np.ndarray], np.random.Generator]:
    """Unique clusters, their row indices, and a seeded generator."""
    unique = np.unique(groups)
    rows = [np.flatnonzero(groups == key) for key in unique]
    return unique, rows, np.random.default_rng(seed)


def cluster_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> DiscriminationResult:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(clusters)
    if not (values.shape == truth.shape == groups.shape) or values.ndim != 1:
        raise ValueError("scores, labels and clusters must be aligned vectors")
    unique, rows_by_cluster, rng = _cluster_index(groups, seed)
    draws: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, unique.size, size=unique.size)
        rows = np.concatenate([rows_by_cluster[index] for index in picked])
        estimate = roc_auc(values[rows], truth[rows])
        if np.isfinite(estimate):
            draws.append(estimate)
    point = roc_auc(values, truth)
    if not draws:
        return DiscriminationResult(
            auc=float(point),
            lower=float("nan"),
            upper=float("nan"),
            n_positive=int((truth > 0.5).sum()),
            n_negative=int((truth <= 0.5).sum()),
            n_clusters=int(unique.size),
        )
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(np.asarray(draws), [tail, 1.0 - tail])
    return DiscriminationResult(
        auc=float(point),
        lower=float(lower),
        upper=float(upper),
        n_positive=int((truth > 0.5).sum()),
        n_negative=int((truth <= 0.5).sum()),
        n_clusters=int(unique.size),
    )


def operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    specificity_floor: float = 0.90,
) -> OperatingPoint:
    """Operating point at the specificity floor, maximising sensitivity.

    The manuscript places the clinical threshold at the weighted specificity
    limit derived from the relative cost of the two errors; the floor is a
    released value because the two costs are not printed.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64) > 0.5
    if values.shape != truth.shape:
        raise ValueError("scores and labels must align")
    candidates = np.unique(values)
    best = OperatingPoint(threshold=float("inf"), sensitivity=0.0, specificity=1.0, youden=-1.0)
    for threshold in candidates:
        predicted = values >= threshold
        sensitivity = float((predicted & truth).sum() / max(int(truth.sum()), 1))
        specificity = float((~predicted & ~truth).sum() / max(int((~truth).sum()), 1))
        if specificity + 1e-9 < specificity_floor:
            continue
        youden = sensitivity + specificity - 1.0
        if youden > best.youden + 1e-12:
            best = OperatingPoint(
                threshold=float(threshold),
                sensitivity=sensitivity,
                specificity=specificity,
                youden=youden,
            )
    return best


def sensitivity_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predicted = np.asarray(scores) >= threshold
    truth = np.asarray(labels) > 0.5
    return float((predicted & truth).sum() / max(int(truth.sum()), 1))


def specificity_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predicted = np.asarray(scores) >= threshold
    truth = np.asarray(labels) > 0.5
    return float((~predicted & ~truth).sum() / max(int((~truth).sum()), 1))


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Paired mean difference with a cluster bootstrap interval."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    groups = np.asarray(clusters)
    if not (a.shape == b.shape == groups.shape):
        raise ValueError("paired vectors and clusters must align")
    unique, rows_by_cluster, rng = _cluster_index(groups, seed)
    draws: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, unique.size, size=unique.size)
        rows = np.concatenate([rows_by_cluster[index] for index in picked])
        draws.append(float((a[rows] - b[rows]).mean()))
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(np.asarray(draws), [tail, 1.0 - tail])
    return float((a - b).mean()), float(lower), float(upper)
