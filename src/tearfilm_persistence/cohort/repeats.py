"""Repeat grouping for the repeatability penalty.

Ref: Sec. 5.4, the penalty regulates the descriptor variance across repeated
acquisitions of the same eye; Sec. 5.8, repeatability is separated into
within-eye, between-session and between-site components.

A group is one eye, not one patient: the two eyes of a patient are separate
measurement units throughout.
"""

from __future__ import annotations

import torch

from tearfilm_persistence.cohort.manifest import ClipRecord


def eye_group_ids(records: list[ClipRecord]) -> torch.Tensor:
    """Index each record by its eye, in first-appearance order."""
    order: dict[str, int] = {}
    ids: list[int] = []
    for record in records:
        if record.eye_id not in order:
            order[record.eye_id] = len(order)
        ids.append(order[record.eye_id])
    return torch.tensor(ids, dtype=torch.long)


def repeated_eye_count(records: list[ClipRecord]) -> int:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.eye_id] = counts.get(record.eye_id, 0) + 1
    return sum(1 for value in counts.values() if value > 1)


def repeat_groups(records: list[ClipRecord]) -> list[list[int]]:
    """Rows of each eye that carries more than one acquisition."""
    buckets: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        buckets.setdefault(record.eye_id, []).append(index)
    return [rows for rows in buckets.values() if len(rows) > 1]


def site_strata(records: list[ClipRecord]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.site, []).append(index)
    return groups


def session_strata(records: list[ClipRecord]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = f"{record.site}/{record.session}"
        groups.setdefault(key, []).append(index)
    return groups
