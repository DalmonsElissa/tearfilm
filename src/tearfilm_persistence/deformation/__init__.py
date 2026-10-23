"""Frame representation, derivative and persistence operator."""

from tearfilm_persistence.deformation.adapter import (
    AdapterConfig,
    PlainClipBackbone,
    TemporalDifferenceAdapter,
    TemporalDifferenceBackbone,
)
from tearfilm_persistence.deformation.curve import (
    CurveSummary,
    SubtestStats,
    locus_entropy,
    ols_slope,
    summarise_clip,
    summarise_segment,
)
from tearfilm_persistence.deformation.derivative import DerivativeConfig, TemporalDerivative
from tearfilm_persistence.deformation.persistence import (
    TAU_SECONDS,
    CurveSegment,
    PersistenceConfig,
    PersistenceOperator,
    area_fraction,
    first_passage,
    hard_occupancy,
    soft_occupancy,
    tau_frames,
)

__all__ = [
    "TAU_SECONDS",
    "AdapterConfig",
    "CurveSegment",
    "CurveSummary",
    "DerivativeConfig",
    "PersistenceConfig",
    "PersistenceOperator",
    "PlainClipBackbone",
    "SubtestStats",
    "TemporalDerivative",
    "TemporalDifferenceAdapter",
    "TemporalDifferenceBackbone",
    "area_fraction",
    "first_passage",
    "hard_occupancy",
    "locus_entropy",
    "ols_slope",
    "soft_occupancy",
    "summarise_clip",
    "summarise_segment",
    "tau_frames",
]
