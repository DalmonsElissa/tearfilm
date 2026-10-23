"""Post-collection calibration of the descriptor.

Ref: Sec. 5.5, calibration follows descriptor collection and is determined from
the variance of repeated acquisitions of the same eye; Sec. 5.4, the descriptor
head adjusts its measurement using variance components under one standard across
sites.

Two maps are applied in order. The variance-component map shrinks each component
toward its site mean by the eye-level reliability, which removes the part of a
reading that repeats do not support. The logistic map then sets the operating
probability, fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tearfilm_persistence.descriptors.repeatability import reliability
from tearfilm_persistence.descriptors.vector import COMPONENTS
from tearfilm_persistence.evaluation.properties import expected_calibration_error


@dataclass
class VarianceComponentCalibrator:
    """Site-wise empirical-Bayes shrinkage driven by eye-level reliability."""

    reliability: tuple[float, ...] = ()
    site_means: dict[str, tuple[float, ...]] = field(default_factory=dict)
    global_mean: tuple[float, ...] = ()

    def fit(
        self, matrix: np.ndarray, eye_ids: list[str], sites: list[str]
    ) -> VarianceComponentCalibrator:
        if matrix.ndim != 2 or matrix.shape[1] != len(COMPONENTS):
            raise ValueError(f"expected an [n, {len(COMPONENTS)}] matrix")
        if len(eye_ids) != matrix.shape[0] or len(sites) != matrix.shape[0]:
            raise ValueError("eye and site vectors must align with the matrix")
        self.reliability = tuple(
            reliability([float(value) for value in matrix[:, index]], eye_ids)
            for index in range(matrix.shape[1])
        )
        self.global_mean = tuple(float(value) for value in matrix.mean(axis=0))
        self.site_means = {}
        site_array = np.asarray(sites)
        for site in np.unique(site_array):
            mask = site_array == site
            self.site_means[str(site)] = tuple(float(value) for value in matrix[mask].mean(axis=0))
        return self

    def transform(self, matrix: np.ndarray, sites: list[str]) -> np.ndarray:
        if not self.reliability:
            raise RuntimeError("the calibrator must be fitted before it is applied")
        rho = np.asarray(self.reliability)
        calibrated = np.empty_like(matrix, dtype=np.float64)
        for row, site in enumerate(sites):
            centre = np.asarray(self.site_means.get(site, self.global_mean))
            calibrated[row] = centre + rho * (matrix[row] - centre)
        return calibrated

    def state(self) -> dict[str, object]:
        return {
            "reliability": list(self.reliability),
            "site_means": {key: list(value) for key, value in self.site_means.items()},
            "global_mean": list(self.global_mean),
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> VarianceComponentCalibrator:
        site_means = state.get("site_means")
        reliability_state = state.get("reliability")
        global_state = state.get("global_mean")
        if not isinstance(site_means, dict):
            raise ValueError("site_means must be a mapping")
        if not isinstance(reliability_state, list) or not isinstance(global_state, list):
            raise ValueError("reliability and global_mean must be sequences")
        return cls(
            reliability=tuple(float(value) for value in reliability_state),
            site_means={
                str(key): tuple(float(item) for item in value if isinstance(value, list))
                for key, value in site_means.items()
            },
            global_mean=tuple(float(value) for value in global_state),
        )


@dataclass
class LogisticCalibrator:
    """Two-parameter Platt map on the decision logit."""

    slope: float = 1.0
    intercept: float = 0.0
    iterations: int = 100
    tolerance: float = 1e-8

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> LogisticCalibrator:
        if logits.shape != labels.shape or logits.ndim != 1:
            raise ValueError("logits and labels must be one-dimensional and aligned")
        x = logits.astype(np.float64)
        y = labels.astype(np.float64)
        design = np.stack([x, np.ones_like(x)], axis=1)
        weights = np.array([self.slope, self.intercept], dtype=np.float64)
        for _ in range(self.iterations):
            logits_now = design @ weights
            probability = 1.0 / (1.0 + np.exp(-logits_now))
            gradient = design.T @ (y - probability)
            curvature = (design * (probability * (1.0 - probability))[:, None]).T @ design
            curvature += 1e-9 * np.eye(2)
            step = np.linalg.solve(curvature, gradient)
            weights = weights + step
            if float(np.abs(step).max()) < self.tolerance:
                break
        self.slope = float(weights[0])
        self.intercept = float(weights[1])
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.slope * logits + self.intercept)))

    def state(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_state(cls, state: dict[str, float]) -> LogisticCalibrator:
        return cls(slope=float(state["slope"]), intercept=float(state["intercept"]))


@dataclass(frozen=True)
class CalibrationOutcome:
    """What the fitted maps change on the table they are applied to."""

    n_records: int
    n_eyes: int
    reliability: tuple[float, ...]
    calibration_error_before: float
    calibration_error_after: float
    slope: float
    intercept: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_records": self.n_records,
            "n_eyes": self.n_eyes,
            "reliability": list(self.reliability),
            "calibration_error_before": self.calibration_error_before,
            "calibration_error_after": self.calibration_error_after,
            "slope": self.slope,
            "intercept": self.intercept,
        }


def _aligned(
    matrix: np.ndarray, eye_ids: list[str], sites: list[str], scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    if scores_array.shape != labels_array.shape or scores_array.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional vectors")
    if scores_array.size != matrix.shape[0]:
        raise ValueError("one score per descriptor row is required")
    if len(eye_ids) != matrix.shape[0] or len(sites) != matrix.shape[0]:
        raise ValueError("eye and site vectors must align with the matrix")
    return scores_array, labels_array, eye_ids, sites


def fit_calibration(
    matrix: np.ndarray,
    eye_ids: list[str],
    sites: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[VarianceComponentCalibrator, LogisticCalibrator]:
    """Fit both maps on a table of collected descriptors.

    Ref: Sec. 5.5. The variance-component map uses the repeats of the same eye;
    the logistic map uses the decision score against the reference label.
    """
    scores_array, labels_array, eyes, site_values = _aligned(matrix, eye_ids, sites, scores, labels)
    components = VarianceComponentCalibrator().fit(matrix, eyes, site_values)
    logistic = LogisticCalibrator().fit(logit(scores_array), labels_array)
    return components, logistic


def calibration_report(
    components: VarianceComponentCalibrator,
    logistic: LogisticCalibrator,
    matrix: np.ndarray,
    eye_ids: list[str],
    sites: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
) -> CalibrationOutcome:
    """Apply the fitted maps to a table and measure the calibration error there.

    The table passed here should not be the one the maps were fitted on; the
    caller decides, and the report records only what was applied.
    """
    scores_array, labels_array, eyes, _ = _aligned(matrix, eye_ids, sites, scores, labels)
    after = logistic.predict(logit(scores_array))
    return CalibrationOutcome(
        n_records=int(matrix.shape[0]),
        n_eyes=len(set(eyes)),
        reliability=components.reliability,
        calibration_error_before=expected_calibration_error(scores_array, labels_array),
        calibration_error_after=expected_calibration_error(after, labels_array),
        slope=logistic.slope,
        intercept=logistic.intercept,
    )


def logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))
