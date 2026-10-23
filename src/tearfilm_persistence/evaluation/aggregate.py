"""Aggregation of a scored eye table into the reported rows.

Ref: Table 2, Table 4, Table 5. Every row is computed from the same scored
table, so a site row and the pooled row are the same code path over a different
subset rather than two methods.

An abstention is excluded from every performance estimate and counted on its own,
which is the rule Sec. 2.7 states for the prospective arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tearfilm_persistence.evaluation.discrimination import (
    DiscriminationResult,
    cluster_bootstrap,
    operating_point,
    roc_auc,
)
from tearfilm_persistence.evaluation.properties import (
    SiteDiscrimination,
    expected_calibration_error,
    false_early_rate,
    prevalence,
)


@dataclass(frozen=True)
class ScoredEye:
    eye_id: str
    patient_id: str
    site: str
    region: str
    arm: str
    severity: str | None
    reference_onset: float | None
    declared_onset: float | None
    score: float
    abstained: bool
    descriptor: dict[str, float] | None = None


@dataclass(frozen=True)
class ArmReport:
    n_eyes: int
    n_patients: int
    n_abstained: int
    prevalence: float
    auc: float
    lower: float
    upper: float
    sensitivity: float
    specificity: float
    threshold: float
    false_early_percent: float
    calibration_error: float
    sites: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, float | int | str | list[str]]:
        return {
            "n_eyes": self.n_eyes,
            "n_patients": self.n_patients,
            "n_abstained": self.n_abstained,
            "prevalence": self.prevalence,
            "auc": self.auc,
            "auc_lower": self.lower,
            "auc_upper": self.upper,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "threshold": self.threshold,
            "false_early_percent": self.false_early_percent,
            "calibration_error": self.calibration_error,
            "sites": list(self.sites),
        }


def scorable(rows: list[ScoredEye]) -> list[ScoredEye]:
    return [row for row in rows if not row.abstained]


def _arrays(rows: list[ScoredEye]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray([row.score for row in rows], dtype=np.float64)
    labels = np.asarray(
        [1.0 if row.reference_onset is not None else 0.0 for row in rows], dtype=np.float64
    )
    clusters = np.asarray([row.patient_id for row in rows])
    return scores, labels, clusters


def summarise_arm(
    rows: list[ScoredEye],
    specificity_floor: float = 0.90,
    n_boot: int = 1000,
    seed: int = 0,
) -> ArmReport:
    usable = scorable(rows)
    if not usable:
        raise ValueError("no scorable eyes in this arm")
    scores, labels, clusters = _arrays(usable)
    discrimination: DiscriminationResult = cluster_bootstrap(
        scores, labels, clusters, n_boot=n_boot, seed=seed
    )
    point = operating_point(scores, labels, specificity_floor)
    declared = np.asarray(
        [row.declared_onset if row.declared_onset is not None else np.nan for row in usable],
        dtype=np.float64,
    )
    reference = np.asarray(
        [row.reference_onset if row.reference_onset is not None else np.nan for row in usable],
        dtype=np.float64,
    )
    return ArmReport(
        n_eyes=len(usable),
        n_patients=int(np.unique(clusters).size),
        n_abstained=len(rows) - len(usable),
        prevalence=prevalence(labels),
        auc=discrimination.auc,
        lower=discrimination.lower,
        upper=discrimination.upper,
        sensitivity=point.sensitivity,
        specificity=point.specificity,
        threshold=point.threshold,
        false_early_percent=100.0 * false_early_rate(declared, reference),
        calibration_error=expected_calibration_error(scores, labels),
        sites=tuple(sorted({row.site for row in usable})),
    )


def site_table(
    rows: list[ScoredEye],
    specificity_floor: float = 0.90,
    n_boot: int = 500,
    seed: int = 0,
) -> list[SiteDiscrimination]:
    table: list[SiteDiscrimination] = []
    for site in sorted({row.site for row in rows}):
        subset = [row for row in rows if row.site == site]
        usable = scorable(subset)
        if not usable:
            continue
        scores, labels, clusters = _arrays(usable)
        discrimination = cluster_bootstrap(scores, labels, clusters, n_boot=n_boot, seed=seed)
        point = operating_point(scores, labels, specificity_floor)
        table.append(
            SiteDiscrimination(
                site=site,
                eyes=len(usable),
                auc=discrimination.auc,
                lower=discrimination.lower,
                upper=discrimination.upper,
                sensitivity=point.sensitivity,
                specificity=point.specificity,
            )
        )
    return table


def stratum_table(
    rows: list[ScoredEye],
    field: str,
    specificity_floor: float = 0.90,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, ArmReport]:
    """Per-level reports for one grouping field (site, severity, generation)."""
    if field not in {"site", "severity", "region", "arm"}:
        raise ValueError(f"unsupported grouping field: {field}")
    groups: dict[str, list[ScoredEye]] = {}
    for row in rows:
        value = getattr(row, field)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    return {
        name: summarise_arm(subset, specificity_floor, n_boot=n_boot, seed=seed)
        for name, subset in sorted(groups.items())
        if scorable(subset)
    }


def heterogeneity(estimates: list[float], variances: list[float]) -> float:
    """I-squared for a set of site estimates, from the Q statistic.

    Ref: Table 4 prints an I2 across sites. The DerSimonian-Laird form is used
    because the site counts differ.
    """
    if len(estimates) != len(variances) or len(estimates) < 2:
        raise ValueError("at least two estimates with matching variances are required")
    weights = np.asarray([1.0 / max(value, 1e-12) for value in variances])
    values = np.asarray(estimates, dtype=np.float64)
    pooled = float((weights * values).sum() / weights.sum())
    q = float((weights * (values - pooled) ** 2).sum())
    degrees = len(values) - 1
    if q <= degrees:
        return 0.0
    return float((q - degrees) / q)


def pooled_auc(rows: list[ScoredEye]) -> float:
    usable = scorable(rows)
    if not usable:
        return float("nan")
    scores, labels, _ = _arrays(usable)
    return roc_auc(scores, labels)
