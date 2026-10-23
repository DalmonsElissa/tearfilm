"""Cohort manifest, repeat grouping and batching."""

from __future__ import annotations

import json

import pytest

from tearfilm_persistence.cohort.manifest import (
    SPLITS,
    ClipRecord,
    CohortTable,
    ensure_patient_level,
    load_table,
)
from tearfilm_persistence.cohort.repeats import (
    eye_group_ids,
    repeat_groups,
    repeated_eye_count,
    site_strata,
)
from tearfilm_persistence.cohort.sampling import ClipDataset, build_loader, labels_for
from tearfilm_persistence.streams.acquisition import save_clip
from tearfilm_persistence.verification.probe_inputs import probe_clip as make_clip


def _record(patient: str, eye: str, split: str = "train", repeat: int = 1) -> ClipRecord:
    return ClipRecord(
        patient_id=patient,
        eye_id=eye,
        site="Site A",
        region="Region I",
        session="s1",
        repeat=repeat,
        arm="retrospective",
        split=split,
        path=f"{eye}.npz",
        reference_onset=5.0,
    )


def test_table_audit_counts_patients_and_eyes() -> None:
    table = CohortTable([_record("p1", "e1"), _record("p1", "e2"), _record("p2", "e3")])
    audit = table.audit()
    assert audit.patients_per_split["train"] == 2
    assert audit.eyes_per_split["train"] == 3
    assert audit.clean


def test_patient_leakage_is_detected_and_refused() -> None:
    table = CohortTable([_record("p1", "e1", "train"), _record("p1", "e2", "test")])
    assert not table.audit().clean
    with pytest.raises(ValueError):
        ensure_patient_level(table)


def test_split_names_are_the_declared_ones() -> None:
    assert set(SPLITS) == {"train", "validation", "test", "external", "prospective"}


def test_record_rejects_an_unknown_split() -> None:
    with pytest.raises(ValueError):
        ClipRecord("p", "e", "A", "I", "s1", 1, "retrospective", "holdout", "x.npz").validated()


def test_load_table_reads_one_object_per_line(tmp_path) -> None:
    path = tmp_path / "cohort.jsonl"
    payload = {
        "patient_id": "p1",
        "eye_id": "e1",
        "site": "Site A",
        "split": "train",
        "path": "e1.npz",
        "reference_onset": 4.2,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    table = load_table(path)
    assert table.records[0].reference_onset == pytest.approx(4.2)


def test_load_table_names_the_missing_field(tmp_path) -> None:
    path = tmp_path / "cohort.jsonl"
    path.write_text(json.dumps({"patient_id": "p1"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_table(path)


def test_load_table_refuses_an_empty_file(tmp_path) -> None:
    path = tmp_path / "cohort.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_table(path)


def test_eye_groups_follow_first_appearance() -> None:
    ids = eye_group_ids([_record("p1", "e1"), _record("p1", "e2"), _record("p1", "e1")])
    assert ids.tolist() == [0, 1, 0]


def test_repeat_groups_keep_only_repeated_eyes() -> None:
    records = [_record("p1", "e1"), _record("p1", "e1", repeat=2), _record("p2", "e2")]
    assert repeat_groups(records) == [[0, 1]]
    assert repeated_eye_count(records) == 1


def test_site_strata_group_by_site() -> None:
    records = [_record("p1", "e1"), _record("p2", "e2")]
    assert len(site_strata(records)) == 1


def test_labels_mark_presence_and_onset() -> None:
    metas = [
        make_clip(eye_id="a", reference_onset=6.0).meta,
        make_clip(eye_id="b", reference_onset=None).meta,
    ]
    labels, targets, valid = labels_for(metas)
    assert labels.tolist() == [1.0, 0.0]
    assert targets.tolist() == [6.0, 0.0]
    assert valid.tolist() == [True, False]


def test_dataset_stamps_the_record_identity(tmp_path) -> None:
    clip = make_clip(eye_id="stored")
    save_clip(clip, tmp_path / "clip.npz")
    record = ClipRecord(
        "patient-x",
        "eye-x",
        "Site Z",
        "II",
        "s2",
        2,
        "prospective",
        "train",
        str(tmp_path / "clip.npz"),
    )
    dataset = ClipDataset([record])
    loaded = dataset[0]
    assert loaded.meta.eye_id == "eye-x"
    assert loaded.meta.site == "Site Z"


def test_dataset_truncates_to_the_requested_length(tmp_path) -> None:
    clip = make_clip(n_frames=20)
    save_clip(clip, tmp_path / "clip.npz")
    record = ClipRecord(
        "p", "e", "A", "I", "s1", 1, "retrospective", "train", str(tmp_path / "clip.npz")
    )
    assert ClipDataset([record], max_frames=8)[0].n_frames == 8


def test_loader_batches_with_padding(tmp_path) -> None:
    records = []
    for index, length in enumerate((10, 12)):
        clip = make_clip(seed=index, n_frames=length, eye_id=f"e{index}")
        path = tmp_path / f"c{index}.npz"
        save_clip(clip, path)
        records.append(
            ClipRecord(
                f"p{index}", f"e{index}", "A", "I", "s1", 1, "retrospective", "train", str(path)
            )
        )
    loader = build_loader(records, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch.frames.shape[0] == 2
    assert batch.frame_mask.sum(dim=1).tolist() == [10, 12]
    assert batch.fps.tolist() == [30.0, 30.0]


def test_dataset_refuses_an_empty_record_list() -> None:
    with pytest.raises(ValueError):
        ClipDataset([])
