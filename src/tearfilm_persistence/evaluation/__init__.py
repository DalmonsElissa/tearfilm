"""Evaluation layers."""

from tearfilm_persistence.evaluation.aggregate import (
    ArmReport,
    ScoredEye,
    heterogeneity,
    pooled_auc,
    scorable,
    site_table,
    stratum_table,
    summarise_arm,
)
from tearfilm_persistence.evaluation.criteria import (
    CRITERIA_ORDER,
    CriteriaThresholds,
    CriterionResult,
    all_passed,
    as_mapping,
    evaluate,
)
from tearfilm_persistence.evaluation.discrimination import (
    DiscriminationResult,
    OperatingPoint,
    cluster_bootstrap,
    operating_point,
    paired_difference,
    roc_auc,
    sensitivity_at,
    specificity_at,
)
from tearfilm_persistence.evaluation.properties import (
    EARLY_MARGIN_SECONDS,
    SiteDiscrimination,
    cross_site_change,
    expected_calibration_error,
    false_early_rate,
    prevalence,
)

__all__ = [
    "CRITERIA_ORDER",
    "EARLY_MARGIN_SECONDS",
    "ArmReport",
    "CriteriaThresholds",
    "CriterionResult",
    "DiscriminationResult",
    "OperatingPoint",
    "ScoredEye",
    "SiteDiscrimination",
    "all_passed",
    "as_mapping",
    "cluster_bootstrap",
    "cross_site_change",
    "evaluate",
    "expected_calibration_error",
    "false_early_rate",
    "heterogeneity",
    "operating_point",
    "paired_difference",
    "pooled_auc",
    "prevalence",
    "roc_auc",
    "scorable",
    "sensitivity_at",
    "site_table",
    "specificity_at",
    "stratum_table",
    "summarise_arm",
]
