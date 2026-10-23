"""The four pre-specified success criteria.

Ref: Sec. 5.9. The primary assessment criterion is a clinician-fulfilled AUC of
at least 0.92, an assisted-minus-unaided difference of at least 0.05, a false
early rate of at most 1.0%, and a descriptor coefficient of variation of at most
45%. The thresholds are printed before any result and are not re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CRITERIA_ORDER: Final[tuple[str, ...]] = (
    "assisted_auc",
    "assisted_gain",
    "false_early_rate",
    "descriptor_cov",
)


@dataclass(frozen=True)
class CriteriaThresholds:
    assisted_auc_min: float = 0.92
    assisted_gain_min: float = 0.05
    false_early_max_percent: float = 1.0
    descriptor_cov_max_percent: float = 45.0

    def validated(self) -> CriteriaThresholds:
        if not 0.5 <= self.assisted_auc_min <= 1.0:
            raise ValueError("an AUC threshold sits in [0.5, 1]")
        if self.assisted_gain_min < 0:
            raise ValueError("a gain threshold is non-negative")
        if self.false_early_max_percent <= 0 or self.descriptor_cov_max_percent <= 0:
            raise ValueError("upper bounds must be positive")
        return self


@dataclass(frozen=True)
class CriterionResult:
    name: str
    value: float
    threshold: float
    direction: str
    passed: bool

    @property
    def margin(self) -> float:
        return (
            self.value - self.threshold
            if self.direction == "at_least"
            else self.threshold - self.value
        )


def evaluate(
    assisted_auc: float,
    assisted_gain: float,
    false_early_rate_percent: float,
    descriptor_cov_percent: float,
    thresholds: CriteriaThresholds | None = None,
) -> list[CriterionResult]:
    cfg = (thresholds or CriteriaThresholds()).validated()
    return [
        CriterionResult(
            name="assisted_auc",
            value=float(assisted_auc),
            threshold=cfg.assisted_auc_min,
            direction="at_least",
            passed=assisted_auc >= cfg.assisted_auc_min,
        ),
        CriterionResult(
            name="assisted_gain",
            value=float(assisted_gain),
            threshold=cfg.assisted_gain_min,
            direction="at_least",
            passed=assisted_gain >= cfg.assisted_gain_min,
        ),
        CriterionResult(
            name="false_early_rate",
            value=float(false_early_rate_percent),
            threshold=cfg.false_early_max_percent,
            direction="at_most",
            passed=false_early_rate_percent <= cfg.false_early_max_percent,
        ),
        CriterionResult(
            name="descriptor_cov",
            value=float(descriptor_cov_percent),
            threshold=cfg.descriptor_cov_max_percent,
            direction="at_most",
            passed=descriptor_cov_percent <= cfg.descriptor_cov_max_percent,
        ),
    ]


def all_passed(results: list[CriterionResult]) -> bool:
    return all(result.passed for result in results)


def as_mapping(results: list[CriterionResult]) -> dict[str, dict[str, float | bool | str]]:
    return {
        result.name: {
            "value": result.value,
            "threshold": result.threshold,
            "direction": result.direction,
            "passed": result.passed,
            "margin": result.margin,
        }
        for result in results
    }
