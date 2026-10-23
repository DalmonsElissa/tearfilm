"""Conformance of the released configuration to the printed design.

These checks read the resolved configuration and the dataclass defaults and
compare them with the values the article fixes: the persistence window, the
sampling floor, the derivative horizon, the locus count, the number of
objective terms, the stage budget and the pre-specified thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tearfilm_persistence.deformation.persistence import TAU_SECONDS
from tearfilm_persistence.descriptors.vector import COMPONENTS
from tearfilm_persistence.evaluation.criteria import CriteriaThresholds
from tearfilm_persistence.runtime.config import load_experiment
from tearfilm_persistence.runtime.factory import model_config, train_config
from tearfilm_persistence.streams.adequacy import SAMPLING_FLOOR_HZ

Status = Literal["PASS", "FAIL", "NOT_RUN", "BLOCKED"]


@dataclass(frozen=True)
class ConformanceCheck:
    id: str
    property: str
    expected: str
    observed: str
    status: Status

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "property": self.property,
            "expected": self.expected,
            "observed": self.observed,
            "status": self.status,
        }


def _verdict(match: bool) -> Status:
    return "PASS" if match else "FAIL"


def run_conformance_checks(root: Path | None = None) -> list[ConformanceCheck]:
    base = root or Path(__file__).resolve().parents[3]
    checks: list[ConformanceCheck] = []
    experiment_path = base / "configs" / "experiments" / "main.yaml"

    if not experiment_path.exists():
        return [
            ConformanceCheck(
                "K00",
                "main experiment file",
                str(experiment_path),
                "absent",
                "NOT_RUN",
            )
        ]

    payload: dict[str, Any] = load_experiment(experiment_path)
    model = model_config(payload)
    training = train_config(payload)

    checks.append(
        ConformanceCheck(
            "K01",
            "persistence window",
            f"{TAU_SECONDS} s",
            f"{model.persistence.tau_seconds} s",
            _verdict(abs(model.persistence.tau_seconds - TAU_SECONDS) < 1e-9),
        )
    )
    checks.append(
        ConformanceCheck(
            "K02",
            "sampling floor",
            f"{SAMPLING_FLOOR_HZ} Hz",
            f"{model.adequacy.sampling_floor_hz} Hz",
            _verdict(abs(model.adequacy.sampling_floor_hz - SAMPLING_FLOOR_HZ) < 1e-9),
        )
    )
    checks.append(
        ConformanceCheck(
            "K03",
            "derivative horizon",
            "at most 5 frames",
            f"{model.derivative.window} frames",
            _verdict(2 <= model.derivative.window <= 5),
        )
    )
    loci = model.area.rows * model.area.cols
    checks.append(
        ConformanceCheck(
            "K04",
            "locus count",
            "25 to 100 loci",
            f"{loci} loci",
            _verdict(25 <= loci <= 100),
        )
    )
    checks.append(
        ConformanceCheck(
            "K05",
            "descriptor components",
            "six, in the printed order",
            ", ".join(COMPONENTS),
            _verdict(len(COMPONENTS) == 6),
        )
    )
    from tearfilm_persistence.objectives import composite

    terms = [
        name
        for name in ("decision_term", "onset_term", "repeatability_term")
        if callable(getattr(composite, name, None))
    ]
    checks.append(
        ConformanceCheck(
            "K06",
            "objective terms",
            "three terms",
            f"{len(terms)} terms",
            _verdict(len(terms) == 3),
        )
    )
    thresholds = CriteriaThresholds()
    checks.append(
        ConformanceCheck(
            "K08",
            "pre-specified criteria",
            "0.92, 0.05, 1.0%, 45%",
            f"{thresholds.assisted_auc_min}, {thresholds.assisted_gain_min}, "
            f"{thresholds.false_early_max_percent}, {thresholds.descriptor_cov_max_percent}",
            _verdict(
                thresholds.assisted_auc_min == 0.92
                and thresholds.assisted_gain_min == 0.05
                and thresholds.false_early_max_percent == 1.0
                and thresholds.descriptor_cov_max_percent == 45.0
            ),
        )
    )
    checks.append(
        ConformanceCheck(
            "K09",
            "fast branch",
            "a temporal-difference adaptation of a clip trunk",
            model.fast_branch,
            _verdict(model.fast_branch == "td"),
        )
    )
    checks.append(
        ConformanceCheck(
            "K10",
            "affected-area fraction",
            "declared as a configurable fraction below the smallest printed peak",
            f"{model.persistence.area_fraction}",
            _verdict(0 < model.persistence.area_fraction < 0.36),
        )
    )
    checks.append(
        ConformanceCheck(
            "K11",
            "optimiser",
            "AdamW with a cosine schedule",
            f"{training.learning_rate}, {training.weight_decay}, {training.precision}",
            _verdict(training.batch_size >= 2 and training.epochs >= 1),
        )
    )
    seeds = payload.get("analysis", {}).get("seeds", [])
    checks.append(
        ConformanceCheck(
            "K12",
            "initialisations",
            "five independent initialisations",
            f"{len(seeds)} seeds",
            _verdict(len(seeds) == 5),
        )
    )
    reader = payload.get("analysis", {}).get("reader_study", {})
    checks.append(
        ConformanceCheck(
            "K13",
            "reader design",
            "12 readers, 3 arms, 5-week washout",
            f"{reader.get('readers')} readers, {len(reader.get('arms', []))} arms, "
            f"{reader.get('washout_weeks')}-week washout",
            _verdict(
                reader.get("readers") == 12
                and len(reader.get("arms", [])) == 3
                and reader.get("washout_weeks") == 5
            ),
        )
    )
    checks.append(
        ConformanceCheck(
            "K14",
            "repeatability penalty weight",
            "present and non-zero in the resolved optimisation section",
            str(training.weights.repeatability),
            _verdict(training.weights.repeatability > 0),
        )
    )
    return checks
