"""Repeatability and agreement of a descriptor.

Ref: Sec. 2.2, Sec. 5.8 and Table 1. The coefficient of repeatability follows
the method-comparison convention printed in Table 1, CoR = 2.77 Sw; the
within-subject standard deviation pools the repeated acquisitions of the same
eye; the coefficient of variation divides that deviation by the mean.

The variance decomposition is the nested reading of Sec. 5.8: site, then session
within site, then eye within session, then repeat within eye.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

COR_FACTOR: float = 2.77
"""Table 1 prints CoR as 2.77 Sw under the standard method-comparison convention."""


def within_subject_sd(groups: Sequence[Sequence[float]]) -> float:
    """Pooled within-eye standard deviation over repeated acquisitions."""
    numerator = 0.0
    degrees = 0
    for group in groups:
        values = np.asarray(group, dtype=np.float64)
        if values.size < 2:
            continue
        numerator += float(((values - values.mean()) ** 2).sum())
        degrees += int(values.size - 1)
    if degrees == 0:
        raise ValueError("repeatability needs at least one eye measured twice")
    return float(np.sqrt(numerator / degrees))


def coefficient_of_repeatability(sw: float, factor: float = COR_FACTOR) -> float:
    return float(factor * sw)


def coefficient_of_variation(sw: float, grand_mean: float) -> float:
    if grand_mean <= 0:
        raise ValueError("the coefficient of variation needs a positive mean")
    return float(100.0 * sw / grand_mean)


def icc_one_way(groups: Sequence[Sequence[float]]) -> float:
    """ICC(1,1) from a one-way layout with eye as the random factor."""
    arrays = [np.asarray(group, dtype=np.float64) for group in groups]
    arrays = [values for values in arrays if values.size > 0]
    count = len(arrays)
    total = sum(int(values.size) for values in arrays)
    if count < 2 or total <= count:
        raise ValueError("ICC needs at least two eyes with repeated measurements")
    grand = float(np.concatenate(arrays).mean())
    between = sum(float(values.size) * (float(values.mean()) - grand) ** 2 for values in arrays)
    within = sum(float(((values - values.mean()) ** 2).sum()) for values in arrays)
    ms_between = between / (count - 1)
    ms_within = within / (total - count)
    sizes = np.asarray([values.size for values in arrays], dtype=np.float64)
    m0 = (total - float((sizes**2).sum()) / total) / (count - 1)
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator <= 0:
        return 0.0
    return float((ms_between - ms_within) / denominator)


@dataclass(frozen=True)
class AgreementSummary:
    bias: float
    lower_limit: float
    upper_limit: float
    sd_difference: float
    n_pairs: int


def bland_altman(first: Sequence[float], second: Sequence[float]) -> AgreementSummary:
    """Bias and 95% limits of agreement between two paired readings."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("agreement needs two paired vectors of the same length")
    difference = a - b
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    return AgreementSummary(
        bias=bias,
        lower_limit=bias - 1.96 * sd,
        upper_limit=bias + 1.96 * sd,
        sd_difference=sd,
        n_pairs=int(a.size),
    )


def variance_shares(
    values: Sequence[float],
    site: Sequence[str],
    session: Sequence[str],
    eye: Sequence[str],
) -> dict[str, float]:
    """Nested sum-of-squares shares for site, session, eye and repeat.

    On a balanced nested design these shares equal the variance-component
    shares; on an unbalanced one they are the sequential decomposition, which is
    the form the manuscript's percentages are reported in.
    """
    y = np.asarray(values, dtype=np.float64)
    if not (len(y) == len(site) == len(session) == len(eye)):
        raise ValueError("values and the three grouping vectors must align")
    if y.size < 2:
        raise ValueError("a decomposition needs at least two measurements")
    site_labels = np.asarray(site)
    session_labels = np.asarray(session)
    eye_labels = np.asarray(eye)
    grand = float(y.mean())
    total_ss = float(((y - grand) ** 2).sum())
    if total_ss <= 0:
        return {"site": 0.0, "session": 0.0, "eye": 0.0, "repeat": 0.0}

    between_site = 0.0
    between_session = 0.0
    between_eye = 0.0
    within = 0.0
    for site_value in np.unique(site_labels):
        site_mask = site_labels == site_value
        site_mean = float(y[site_mask].mean())
        between_site += float(site_mask.sum()) * (site_mean - grand) ** 2
        for session_value in np.unique(session_labels[site_mask]):
            session_mask = site_mask & (session_labels == session_value)
            session_mean = float(y[session_mask].mean())
            between_session += float(session_mask.sum()) * (session_mean - site_mean) ** 2
            for eye_value in np.unique(eye_labels[session_mask]):
                eye_mask = session_mask & (eye_labels == eye_value)
                eye_mean = float(y[eye_mask].mean())
                between_eye += float(eye_mask.sum()) * (eye_mean - session_mean) ** 2
                within += float(((y[eye_mask] - eye_mean) ** 2).sum())
    return {
        "site": between_site / total_ss,
        "session": between_session / total_ss,
        "eye": between_eye / total_ss,
        "repeat": within / total_ss,
    }


def reliability(values: Sequence[float], eye: Sequence[str]) -> float:
    """Share of total variance that is between-eye, in ``[0, 1]``.

    This is the quantity the repeatability-aware head maximises: a descriptor
    that moves between eyes and holds still within one has reliability near one.
    """
    y = np.asarray(values, dtype=np.float64)
    labels = np.asarray(eye)
    if y.size < 2:
        return 0.0
    grand = float(y.mean())
    total = float(((y - grand) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for value in np.unique(labels):
        mask = labels == value
        between += float(mask.sum()) * (float(y[mask].mean()) - grand) ** 2
    return float(min(max(between / total, 0.0), 1.0))


def summarise_repeats(groups: Iterable[Sequence[float]]) -> dict[str, float]:
    """CoR, Sw, CoV and ICC for a set of repeated-eye groups."""
    materialised = [list(group) for group in groups]
    sw = within_subject_sd(materialised)
    flat = np.concatenate([np.asarray(group, dtype=np.float64) for group in materialised])
    mean = float(flat.mean())
    return {
        "n_eyes": float(len(materialised)),
        "n_measurements": float(flat.size),
        "sw": sw,
        "cor": coefficient_of_repeatability(sw),
        "cov_percent": coefficient_of_variation(sw, mean) if mean > 0 else float("nan"),
        "icc": icc_one_way(materialised),
    }
