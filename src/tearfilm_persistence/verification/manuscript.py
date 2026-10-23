"""Arithmetic inside the manuscript's own tables.

The numbers below are transcribed from the article so that the relationships it
asserts can be checked without the private cohort: a coefficient of repeatability
against a within-subject standard deviation, a pooled AUC against its site rows,
an interaction ratio against two component losses. They are inputs to a
consistency test, not results of this release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Status = Literal["PASS", "FAIL", "NOT_RUN", "BLOCKED"]

COR_FACTOR = 2.77

# Table 1, p. 4. (CoR, CoV, Sw)
TABLE_1: dict[str, tuple[float, float, float]] = {
    "model_descriptor": (3.4, 34.2, 1.21),
    "reference": (19.79, 65.49, 7.14),
    "reference_low": (9.90, 51.90, 3.57),
    "meniscus_height": (0.058, 8.6, 0.021),
    "meniscus_area": (0.017, 12.4, 0.006),
    "symptom": (9.5, 21.7, 3.42),
}

# Table 2, p. 6.
TABLE_2: dict[str, dict[str, float]] = {
    "frame_classifier": {"auc": 0.812, "sens": 0.702, "spec": 0.788, "early": 7.2, "shift": 0.071},
    "mean_pooled": {"auc": 0.838, "sens": 0.724, "spec": 0.806, "early": 6.4, "shift": 0.063},
    "factorised": {"auc": 0.879, "sens": 0.762, "spec": 0.841, "early": 5.1, "shift": 0.055},
    "siamese": {"auc": 0.882, "sens": 0.769, "spec": 0.844, "early": 4.6, "shift": 0.062},
    "frozen_probe": {"auc": 0.861, "sens": 0.741, "spec": 0.828, "early": 5.8, "shift": 0.058},
    "video_transformer": {"auc": 0.947, "sens": 0.836, "spec": 0.901, "early": 3.9, "shift": 0.048},
    "teardyn": {"auc": 0.944, "sens": 0.834, "spec": 0.907, "early": 0.8, "shift": 0.011},
    "dynamic_reference": {"auc": 0.786, "sens": 0.712, "spec": 0.744, "early": 6.1},
    "clinician_grading": {"auc": 0.803, "sens": 0.734, "spec": 0.762, "early": 5.2},
}

# Table 3, p. 8.
TABLE_3: dict[str, dict[str, float]] = {
    "full": {"auc": 0.944, "early": 0.8, "ece": 0.031},
    "minus_fast": {"auc": 0.897, "early": 1.9, "ece": 0.044},
    "minus_persistence": {"auc": 0.912, "early": 4.4, "ece": 0.038},
    "minus_structural": {"auc": 0.936, "early": 0.9, "ece": 0.034},
    "minus_head": {"auc": 0.940, "early": 0.8, "ece": 0.052},
    "minus_gate": {"auc": 0.941, "early": 1.1, "ece": 0.036},
    "pooled_persistence": {"auc": 0.903, "early": 4.1, "ece": 0.041},
    "conv3d_fast": {"auc": 0.918, "early": 1.6, "ece": 0.040},
    "both_removed": {"auc": 0.838, "early": 5.3, "ece": 0.057},
}

# Table 4, p. 9.
TABLE_4: dict[str, Any] = {
    "sites": {
        "Site A": {"eyes": 2486, "auc": 0.947, "sens": 0.841, "spec": 0.910, "kappa": 0.76},
        "Site B": {"eyes": 2134, "auc": 0.943, "sens": 0.836, "spec": 0.906, "kappa": 0.74},
        "Site C": {"eyes": 2107, "auc": 0.938, "sens": 0.829, "spec": 0.899, "kappa": 0.72},
        "Site D": {"eyes": 1685, "auc": 0.941, "sens": 0.831, "spec": 0.903, "kappa": 0.73},
    },
    "pooled": {"eyes": 8412, "auc": 0.942, "sens": 0.834, "spec": 0.905, "kappa": 0.74},
    "strata": {
        "mild": {"eyes": 2877, "auc": 0.958, "sens": 0.851, "spec": 0.918, "kappa": 0.76},
        "moderate": {"eyes": 3499, "auc": 0.941, "sens": 0.833, "spec": 0.904, "kappa": 0.74},
        "severe": {"eyes": 2036, "auc": 0.881, "sens": 0.792, "spec": 0.871, "kappa": 0.71},
    },
    "external": {"eyes": 2147, "auc": 0.931, "sens": 0.822, "spec": 0.894, "kappa": 0.71},
}

# Table 5, p. 10.
TABLE_5: dict[str, Any] = {
    "auc": 0.938,
    "sites": {"Site A": 0.941, "Site B": 0.936, "Site C": 0.937},
    "sensitivity": 0.871,
    "specificity": 0.882,
    "kappa": 0.79,
    "referral": 0.918,
    "cov": 35.1,
    "false_early": 0.9,
    "adequacy": 97.5,
}

# Table 6, p. 12.
TABLE_6: dict[str, Any] = {
    "alone": {"auc": 0.852, "sens": 0.781, "spec": 0.842, "time": 62.4, "kappa": 0.61},
    "model": {"auc": 0.944, "sens": 0.834, "spec": 0.907, "time": 8.9},
    "assisted": {"auc": 0.936, "sens": 0.869, "spec": 0.877, "time": 51.8, "kappa": 0.74},
    "gain": 0.084,
    "gain_lower": 0.061,
    "gain_upper": 0.107,
    "reading_reduction_percent": 17.0,
    "reliance": {"count": 20, "denominator": 318, "rate": 6.3},
    "under_reliance": {"count": 10, "denominator": 318, "rate": 3.1},
    "experience": {
        "under_5": (0.830, 0.942, 0.112),
        "five_to_ten": (0.850, 0.944, 0.094),
        "over_ten": (0.871, 0.909, 0.038),
    },
    "tertiles": {
        "low": (0.824, 0.955, 0.131),
        "middle": (0.849, 0.931, 0.082),
        "high": (0.869, 0.895, 0.026),
    },
}

# Table 7, p. 16.
TABLE_7: dict[int, dict[str, float | None]] = {
    1: {"onset": 3.1, "peak": 0.41, "reference": 3.8},
    2: {"onset": 2.6, "peak": 0.55, "reference": 4.1},
    3: {"onset": None, "peak": None, "reference": 5.4},
    4: {"onset": 3.0, "peak": 0.62, "reference": 2.9},
    5: {"onset": 1.4, "peak": 0.38, "reference": 11.2},
    6: {"onset": 4.2, "peak": 0.44, "reference": 4.5},
    7: {"onset": 2.2, "peak": 0.48, "reference": 2.6},
    8: {"onset": 2.4, "peak": 0.51, "reference": 3.1},
    9: {"onset": 4.3, "peak": 0.36, "reference": 4.6},
}

# Table 8, p. 24.
TABLE_8: dict[str, Any] = {
    "stages": [1.2, 47.5, 10.4, 9.0, 7.3, 2.6],
    "shares": [1.5, 60.9, 13.3, 11.5, 9.4, 3.3],
    "total": 78.0,
    "milliseconds": 164.0,
}


@dataclass(frozen=True)
class ManuscriptCheck:
    id: str
    location: str
    statement: str
    computed: float | None
    printed: float | None
    tolerance: float
    status: Status
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "location": self.location,
            "statement": self.statement,
            "computed": self.computed,
            "printed": self.printed,
            "tolerance": self.tolerance,
            "status": self.status,
            "detail": self.detail,
        }


def _check(
    identifier: str,
    location: str,
    statement: str,
    computed: float,
    printed: float,
    tolerance: float,
    detail: str = "",
) -> ManuscriptCheck:
    status: Status = "PASS" if abs(computed - printed) <= tolerance else "FAIL"
    text = detail
    if status == "FAIL" and not text:
        text = f"computed {computed:.4f} against printed {printed:.4f}"
    return ManuscriptCheck(
        id=identifier,
        location=location,
        statement=statement,
        computed=float(computed),
        printed=float(printed),
        tolerance=float(tolerance),
        status=status,
        detail=text,
    )


def _direct(
    identifier: str,
    location: str,
    statement: str,
    first: float,
    second: float,
    tolerance: float,
    detail: str,
) -> ManuscriptCheck:
    status: Status = "PASS" if abs(first - second) <= tolerance else "FAIL"
    return ManuscriptCheck(
        id=identifier,
        location=location,
        statement=statement,
        computed=float(first),
        printed=float(second),
        tolerance=float(tolerance),
        status=status,
        detail=(
            detail
            if status == "PASS"
            else f"two printed values disagree: {first} against {second}. {detail}"
        ),
    )


def run_manuscript_checks() -> list[ManuscriptCheck]:
    checks: list[ManuscriptCheck] = []

    for name, (cor, _cov, sw) in TABLE_1.items():
        tolerance = 0.05 if cor > 1 else 0.0015
        checks.append(
            _check(
                f"M{len(checks) + 1:02d}",
                "Table 1, p. 4",
                f"CoR equals 2.77 Sw for the {name.replace('_', ' ')}",
                COR_FACTOR * sw,
                cor,
                tolerance,
            )
        )

    reference_midpoint = (TABLE_1["reference_low"][1] + TABLE_1["reference"][1]) / 2.0
    checks.append(
        _check(
            "M07",
            "Fig. 1, p. 5",
            "The reference bar sits at the midpoint of its printed coefficient-of-variation band",
            reference_midpoint,
            58.7,
            0.05,
        )
    )

    full_auc = TABLE_3["full"]["auc"]
    for offset, (row, gid) in enumerate(
        [
            ("minus_fast", "M08"),
            ("minus_persistence", "M09"),
            ("minus_structural", "M10"),
            ("minus_head", "M11"),
            ("minus_gate", "M12"),
        ]
    ):
        checks.append(
            _check(
                gid,
                "Table 3, p. 8",
                f"The AUC loss of the {row.replace('_', ' ')} row equals the published delta",
                full_auc - TABLE_3[row]["auc"],
                [0.047, 0.032, 0.008, 0.004, 0.003][offset],
                0.0005,
            )
        )

    individual = (full_auc - TABLE_3["minus_fast"]["auc"]) + (
        full_auc - TABLE_3["minus_persistence"]["auc"]
    )
    checks.append(
        _check(
            "M13",
            "Sec. 2.3, p. 6",
            "The sum of the two individual temporal losses is 0.079",
            individual,
            0.079,
            0.001,
        )
    )
    checks.append(
        _check(
            "M14",
            "Sec. 2.3, p. 6",
            "The joint loss of removing both temporal branches is 0.106",
            full_auc - TABLE_3["both_removed"]["auc"],
            0.106,
            0.001,
        )
    )
    checks.append(
        _check(
            "M15",
            "Sec. 2.3, p. 6",
            "The interaction ratio is the joint loss over the sum of individual losses",
            (full_auc - TABLE_3["both_removed"]["auc"]) / individual,
            1.34,
            0.005,
        )
    )
    checks.append(
        _check(
            "M16",
            "Sec. 2.3, p. 5",
            "The model's margin over the frame-pair design is 0.062",
            full_auc - TABLE_2["siamese"]["auc"],
            0.062,
            0.0005,
        )
    )
    checks.append(
        _check(
            "M17",
            "Sec. 2.3, p. 5",
            "The model's margin over the plain frame classifier is 0.132",
            full_auc - TABLE_2["frame_classifier"]["auc"],
            0.132,
            0.0005,
        )
    )
    checks.append(
        ManuscriptCheck(
            id="",
            location="Sec. 2.3, p. 5 against Table 2, p. 6",
            statement="The constrained configuration's AUC matches the table it is reported in",
            computed=float(TABLE_2["teardyn"]["auc"]),
            printed=0.947,
            tolerance=0.0,
            status="FAIL",
            detail="The prose of Sec. 2.3 assigns 0.947 to the constrained configuration and "
            "0.944 to the unconstrained attention baseline. Table 2, Table 3, Table 5, "
            "Table 6 and the abstract print 0.944 for the model and 0.947 for the "
            "baseline. The prose's own margins (0.062 over the frame-pair row and 0.132 "
            "over the frame classifier) are computed from 0.944, so 0.944 is the "
            "operative value and the two prose sentences are transposed.",
        )
    )

    site_eyes = sum(row["eyes"] for row in TABLE_4["sites"].values())
    checks.append(
        _check(
            "M19",
            "Table 4, p. 9",
            "The four site rows sum to the pooled retrospective count",
            float(site_eyes),
            float(TABLE_4["pooled"]["eyes"]),
            0.5,
        )
    )
    stratum_eyes = sum(row["eyes"] for row in TABLE_4["strata"].values())
    checks.append(
        _check(
            "M20",
            "Table 4, p. 9",
            "The three severity strata sum to the pooled retrospective count",
            float(stratum_eyes),
            float(TABLE_4["pooled"]["eyes"]),
            0.5,
        )
    )
    weighted = sum(row["eyes"] * row["auc"] for row in TABLE_4["sites"].values()) / site_eyes
    checks.append(
        _check(
            "M21",
            "Table 4, p. 9",
            "The pooled AUC is the eye-weighted mean of the site rows within rounding",
            weighted,
            TABLE_4["pooled"]["auc"],
            0.001,
        )
    )
    site_aucs = [row["auc"] for row in TABLE_4["sites"].values()]
    checks.append(
        _check(
            "M22",
            "Sec. 2.4, p. 8",
            "The printed site AUC range is 0.009",
            max(site_aucs) - min(site_aucs),
            0.009,
            0.0005,
        )
    )
    site_kappa = sum(row["kappa"] for row in TABLE_4["sites"].values()) / len(TABLE_4["sites"])
    checks.append(
        _check(
            "M23",
            "Table 4, p. 9",
            "The site kappa rows average to the pooled kappa within rounding",
            site_kappa,
            TABLE_4["pooled"]["kappa"],
            0.005,
        )
    )
    stratum_kappa = sum(row["kappa"] for row in TABLE_4["strata"].values()) / 3.0
    checks.append(
        _check(
            "M24",
            "Table 4, p. 9",
            "The severity-stratum kappa rows average to the pooled kappa within rounding",
            stratum_kappa,
            TABLE_4["pooled"]["kappa"],
            0.005,
        )
    )

    checks.append(
        _check(
            "M25",
            "Table 6, p. 12",
            "The assisted-minus-unaided AUC equals the printed difference",
            TABLE_6["assisted"]["auc"] - TABLE_6["alone"]["auc"],
            TABLE_6["gain"],
            0.0005,
        )
    )
    checks.append(
        _check(
            "M26",
            "Table 6, p. 12",
            "The sensitivity gain equals the printed difference",
            TABLE_6["assisted"]["sens"] - TABLE_6["alone"]["sens"],
            0.088,
            0.0005,
        )
    )
    checks.append(
        _check(
            "M27",
            "Table 6, p. 12",
            "The specificity gain equals the printed difference",
            TABLE_6["assisted"]["spec"] - TABLE_6["alone"]["spec"],
            0.035,
            0.0005,
        )
    )
    reduction = (
        100.0 * (TABLE_6["alone"]["time"] - TABLE_6["assisted"]["time"]) / TABLE_6["alone"]["time"]
    )
    checks.append(
        _check(
            "M28",
            "Sec. 2.6, p. 11",
            "Reading time falls by the printed percentage",
            reduction,
            TABLE_6["reading_reduction_percent"],
            0.05,
        )
    )
    checks.append(
        _check(
            "M29",
            "Table 6, p. 12",
            "The inter-reader kappa difference is 0.13",
            TABLE_6["assisted"]["kappa"] - TABLE_6["alone"]["kappa"],
            0.13,
            0.0005,
        )
    )
    for name, (unaided, assisted, printed) in TABLE_6["experience"].items():
        checks.append(
            _check(
                f"M{len(checks) + 1:02d}",
                "Table 6, p. 12",
                f"The {name.replace('_', ' ')} experience gain equals the printed difference",
                assisted - unaided,
                printed,
                0.0005,
            )
        )
    for name, (unaided, assisted, printed) in TABLE_6["tertiles"].items():
        checks.append(
            _check(
                f"M{len(checks) + 1:02d}",
                "Table 6, p. 12",
                f"The {name} agreement-tertile gain equals the printed difference",
                assisted - unaided,
                printed,
                0.0005,
            )
        )
    checks.append(
        _check(
            "M35",
            "Table 6, p. 12",
            "Reliance on a contradicting suggestion is the printed rate",
            100.0 * TABLE_6["reliance"]["count"] / TABLE_6["reliance"]["denominator"],
            TABLE_6["reliance"]["rate"],
            0.05,
        )
    )
    checks.append(
        _check(
            "M36",
            "Table 6, p. 12",
            "Declining a correct suggestion is the printed rate",
            100.0 * TABLE_6["under_reliance"]["count"] / TABLE_6["under_reliance"]["denominator"],
            TABLE_6["under_reliance"]["rate"],
            0.05,
        )
    )

    stage_sum = sum(float(value) for value in TABLE_8["stages"])
    checks.append(
        _check(
            "M37",
            "Table 8, p. 24",
            "The six stage costs sum to the printed total",
            stage_sum,
            78.0,
            0.05,
        )
    )
    share_sum = sum(float(value) for value in TABLE_8["shares"])
    checks.append(
        _check(
            "M38",
            "Table 8, p. 24",
            "The six stage shares sum to one hundred percent",
            share_sum,
            100.0,
            0.2,
        )
    )
    stages = list(TABLE_8["stages"])
    published = list(TABLE_8["shares"])
    for index, cost in enumerate(stages):
        checks.append(
            _check(
                f"M{39 + index:02d}",
                "Table 8, p. 24",
                f"Stage {index + 1} share equals its cost over the total",
                100.0 * float(cost) / 78.0,
                float(published[index]),
                0.06,
            )
        )

    prospective_mean = sum(TABLE_5["sites"].values()) / len(TABLE_5["sites"])
    checks.append(
        _check(
            "M45",
            "Table 5, p. 10",
            "The prospective pooled AUC is the mean of the three prospective site rows",
            prospective_mean,
            TABLE_5["auc"],
            0.0005,
        )
    )
    checks.append(
        _check(
            "M46",
            "Table 5, p. 10",
            "The prospective coefficient-of-variation interval is centred on its estimate",
            (32.9 + 37.4) / 2.0,
            TABLE_5["cov"],
            0.08,
        )
    )
    checks.append(
        _direct(
            "M47",
            "Sec. 2.1, p. 3 against Table 5, p. 10",
            "The acquisition-adequacy rate is the same quantity in both places",
            97.3,
            TABLE_5["adequacy"],
            0.05,
            "Sec. 2.1 prints 97.3% for the retrospective acquisition-adequacy rate; Sec. 2.7, "
            "Table 5 and Sec. 5.1 all use 97.5% for the same criterion.",
        )
    )
    checks.append(
        _check(
            "M48",
            "Sec. 5.1, p. 21",
            "Adequate acquisitions plus removed inadequate acquisitions account for the "
            "attempted total",
            97.3 + 2.3,
            100.0,
            0.05,
        )
    )
    checks.append(
        _direct(
            "M49",
            "Sec. 2.8, p. 13 against Table 7, p. 16",
            "Case 1's reference break-up time is the same in the prose and in the table",
            6.0,
            float(TABLE_7[1]["reference"] or 0.0),
            0.05,
            "The prose of case 1 reports a reference break-up time of 6 s and then states the "
            "comparison value as 3.8 s; Table 7 prints 3.8 s.",
        )
    )
    checks.append(
        _direct(
            "M50",
            "Sec. 2.8, p. 14 against Table 7, p. 16",
            "Case 5's instability peak is the same in the prose and in the table",
            0.48,
            float(TABLE_7[5]["peak"] or 0.0),
            0.005,
            "The prose of case 5 reports a sign-consistent derivative of 0.48; Table 7 prints a "
            "peak affected fraction of 0.38.",
        )
    )
    checks.append(
        _check(
            "M51",
            "Sec. 2.4, p. 8",
            "The external-cohort AUC change is 0.010",
            TABLE_4["sites"]["Site D"]["auc"] - TABLE_4["external"]["auc"],
            0.010,
            0.0005,
        )
    )
    checks.append(
        _check(
            "M52",
            "Sec. 2.2, p. 4",
            "The three variance shares close to one hundred percent",
            7.1 + 19.4 + 73.5,
            100.0,
            0.05,
        )
    )
    checks.append(
        _check(
            "M53",
            "Table 2, p. 6",
            "The interval of the constrained configuration is centred on its estimate",
            (0.938 + 0.950) / 2.0,
            TABLE_2["teardyn"]["auc"],
            0.0005,
        )
    )
    checks.append(
        _check(
            "M54",
            "Table 2, p. 6",
            "The interval of the attention baseline is centred on its estimate",
            (0.941 + 0.953) / 2.0,
            TABLE_2["video_transformer"]["auc"],
            0.0005,
        )
    )
    met = sum(
        [
            TABLE_6["assisted"]["auc"] >= 0.92,
            TABLE_6["gain"] >= 0.05,
            TABLE_1["model_descriptor"][1] <= 45.0,
            TABLE_3["full"]["early"] <= 1.0,
        ]
    )
    checks.append(
        _check(
            "M55",
            "Sec. 5.9, p. 25",
            "The four pre-specified criteria are met by the reported values",
            float(met),
            4.0,
            0.0,
        )
    )
    checks.append(
        _check(
            "M56",
            "Sec. 2.2, p. 4",
            "The descriptor's coefficient-of-variation interval contains its estimate",
            1.0 if 31.8 <= 34.2 <= 36.9 else 0.0,
            1.0,
            0.0,
        )
    )

    return [
        ManuscriptCheck(
            id=f"M{index:02d}",
            location=check.location,
            statement=check.statement,
            computed=check.computed,
            printed=check.printed,
            tolerance=check.tolerance,
            status=check.status,
            detail=check.detail,
        )
        for index, check in enumerate(checks, start=1)
    ]


def summary(checks: list[ManuscriptCheck]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for check in checks:
        counts[check.status] += 1
    return counts
