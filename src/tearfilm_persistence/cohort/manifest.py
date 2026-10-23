"""Cohort manifest: the record table the pipeline reads.

Ref: Sec. 5.1 (patient-level partition, site-wise documentation) and Sec. 2.1.

The table is supplied by the caller. Nothing here fabricates a record: a column
that the manifest does not carry stays absent, and a split that puts one patient
on both sides raises rather than being repaired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test", "external", "prospective")
ARMS: Final[tuple[str, ...]] = ("retrospective", "prospective", "external")


@dataclass(frozen=True)
class ClipRecord:
    patient_id: str
    eye_id: str
    site: str
    region: str
    session: str
    repeat: int
    arm: str
    split: str
    path: str
    reference_onset: float | None = None
    reference_breakup: float | None = None
    symptom_score: float | None = None
    severity: str | None = None
    device_generation: str | None = None

    def validated(self) -> ClipRecord:
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm: {self.arm}")
        if self.repeat < 1:
            raise ValueError("repeat index starts at one")
        return self


@dataclass(frozen=True)
class SplitAudit:
    patients_per_split: dict[str, int]
    eyes_per_split: dict[str, int]
    repeated_eyes: int
    sites: tuple[str, ...]
    leakage: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.leakage


@dataclass
class CohortTable:
    records: list[ClipRecord] = field(default_factory=list)

    def by_split(self, split: str) -> list[ClipRecord]:
        return [record for record in self.records if record.split == split]

    def by_arm(self, arm: str) -> list[ClipRecord]:
        return [record for record in self.records if record.arm == arm]

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({record.site for record in self.records}))

    def audit(self) -> SplitAudit:
        patients: dict[str, set[str]] = {}
        eyes: dict[str, set[str]] = {}
        for record in self.records:
            patients.setdefault(record.split, set()).add(record.patient_id)
            eyes.setdefault(record.split, set()).add(record.eye_id)
        leakage: list[str] = []
        splits = list(patients)
        for first in range(len(splits)):
            for second in range(first + 1, len(splits)):
                shared = patients[splits[first]] & patients[splits[second]]
                if shared:
                    leakage.append(f"patient shared by {splits[first]} and {splits[second]}")
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.eye_id] = counts.get(record.eye_id, 0) + 1
        return SplitAudit(
            patients_per_split={key: len(value) for key, value in patients.items()},
            eyes_per_split={key: len(value) for key, value in eyes.items()},
            repeated_eyes=sum(1 for value in counts.values() if value > 1),
            sites=self.sites(),
            leakage=tuple(leakage),
        )


def load_table(path: str | Path) -> CohortTable:
    """Read one JSON object per line from a manifest."""
    table = CohortTable()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload: dict[str, Any] = json.loads(stripped)
        try:
            record = ClipRecord(
                patient_id=str(payload["patient_id"]),
                eye_id=str(payload["eye_id"]),
                site=str(payload["site"]),
                region=str(payload.get("region", "")),
                session=str(payload.get("session", "s1")),
                repeat=int(payload.get("repeat", 1)),
                arm=str(payload.get("arm", "retrospective")),
                split=str(payload.get("split", "train")),
                path=str(payload["path"]),
                reference_onset=_maybe_float(payload.get("reference_onset")),
                reference_breakup=_maybe_float(payload.get("reference_breakup")),
                symptom_score=_maybe_float(payload.get("symptom_score")),
                severity=_maybe_text(payload.get("severity")),
                device_generation=_maybe_text(payload.get("device_generation")),
            )
        except KeyError as missing:
            raise KeyError(f"{path}:{number} is missing field {missing.args[0]}") from missing
        table.records.append(record.validated())
    if not table.records:
        raise ValueError(f"{path} contains no records")
    return table


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maybe_text(value: Any) -> str | None:
    return None if value is None else str(value)


def ensure_patient_level(table: CohortTable) -> SplitAudit:
    """Return the audit, refusing a table that shares a patient across splits."""
    audit = table.audit()
    if audit.leakage:
        raise ValueError("patient-level partition violated: " + "; ".join(audit.leakage))
    return audit
