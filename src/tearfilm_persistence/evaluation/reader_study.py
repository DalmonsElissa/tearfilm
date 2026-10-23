"""Three-arm reader study.

Ref: Sec. 5.7 (fully crossed design, twelve readers, three arms, five-week
washout), Sec. 2.6, Sec. 5.8 (Obuchowski-Rockette style analysis with readers
and cases random).

The variance of an arm difference follows the Obuchowski-Rockette decomposition
for a fully crossed design: the case and reader main effects of the paired
difference enter with their own mean squares, and the interaction mean square
carries the residual degrees of freedom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from tearfilm_persistence.evaluation.discrimination import roc_auc


def wilson_interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("a proportion needs a denominator")
    if not 0 <= successes <= total:
        raise ValueError("successes must sit inside the denominator")
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    phat = successes / total
    denominator = 1.0 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return float(max(centre - spread, 0.0)), float(min(centre + spread, 1.0))


@dataclass(frozen=True)
class RelianceSummary:
    denominator: int
    followed_contradicting: int
    followed_rate: float
    followed_interval: tuple[float, float]
    declined_correct: int
    declined_rate: float
    declined_interval: tuple[float, float]


def reliance_summary(
    followed_contradicting: int,
    declined_correct: int,
    denominator: int,
) -> RelianceSummary:
    """Reliance and under-reliance over the discordant reads."""
    return RelianceSummary(
        denominator=denominator,
        followed_contradicting=followed_contradicting,
        followed_rate=followed_contradicting / denominator,
        followed_interval=wilson_interval(followed_contradicting, denominator),
        declined_correct=declined_correct,
        declined_rate=declined_correct / denominator,
        declined_interval=wilson_interval(declined_correct, denominator),
    )


def two_way_mean_squares(values: np.ndarray) -> tuple[float, float, float]:
    """Reader, case and residual mean squares for an ``[r, c]`` table."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("the table needs at least two readers and two cases")
    readers, cases = matrix.shape
    grand = float(matrix.mean())
    reader_means = matrix.mean(axis=1)
    case_means = matrix.mean(axis=0)
    ss_reader = cases * float(((reader_means - grand) ** 2).sum())
    ss_case = readers * float(((case_means - grand) ** 2).sum())
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_residual = max(ss_total - ss_reader - ss_case, 0.0)
    return (
        ss_reader / (readers - 1),
        ss_case / (cases - 1),
        ss_residual / ((readers - 1) * (cases - 1)),
    )


def orc_variance(
    reader_ms: float, case_ms: float, residual_ms: float, n_readers: int, n_cases: int
) -> float:
    """Obuchowski-Rockette variance of a mean arm difference."""
    if n_readers < 2 or n_cases < 2:
        raise ValueError("the design needs at least two readers and two cases")
    numerator = case_ms + reader_ms + (n_readers - 2) * residual_ms
    return max(numerator, 0.0) / (n_readers * n_cases)


@dataclass(frozen=True)
class OrcTest:
    difference: float
    standard_error: float
    statistic: float
    p_value: float
    n_readers: int
    n_cases: int


def orc_difference(first: np.ndarray, second: np.ndarray) -> OrcTest:
    """Paired arm difference over ``[reader, case]`` readings."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("both arms must be [reader, case] tables of the same shape")
    difference = a - b
    reader_ms, case_ms, residual_ms = two_way_mean_squares(difference)
    variance = orc_variance(reader_ms, case_ms, residual_ms, a.shape[0], a.shape[1])
    standard_error = float(np.sqrt(variance))
    mean = float(difference.mean())
    statistic = mean / standard_error if standard_error > 0 else 0.0
    p_value = float(2.0 * norm.sf(abs(statistic)))
    return OrcTest(
        difference=mean,
        standard_error=standard_error,
        statistic=float(statistic),
        p_value=p_value,
        n_readers=a.shape[0],
        n_cases=a.shape[1],
    )


def reader_case_auc(
    ratings: dict[tuple[str, str], float],
    labels: dict[str, int],
    readers: Sequence[str],
    cases: Sequence[str],
) -> np.ndarray:
    """Per-reader AUC over the shared case set, as an ``[r]`` vector."""
    values: list[float] = []
    for reader in readers:
        scores = np.asarray([ratings[(reader, case)] for case in cases], dtype=np.float64)
        truth = np.asarray([labels[case] for case in cases], dtype=np.float64)
        values.append(roc_auc(scores, truth))
    return np.asarray(values, dtype=np.float64)


def arm_auc_table(
    ratings: dict[tuple[str, str, str], float],
    labels: dict[str, int],
    readers: Sequence[str],
    cases: Sequence[str],
    arm: str,
) -> np.ndarray:
    """``[reader, case]`` score table for one arm, shaped for :func:`orc_difference`."""
    rows = []
    for reader in readers:
        rows.append([ratings[(reader, case, arm)] for case in cases])
    return np.asarray(rows, dtype=np.float64)


def experience_groups(years: Sequence[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"under_5": [], "five_to_ten": [], "over_ten": []}
    for index, value in enumerate(years):
        if value < 5:
            groups["under_5"].append(index)
        elif value <= 10:
            groups["five_to_ten"].append(index)
        else:
            groups["over_ten"].append(index)
    return groups
