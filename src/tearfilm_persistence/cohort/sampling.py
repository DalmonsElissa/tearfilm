"""Batching over an acquisition table.

Ref: Sec. 2.1 (clips of a median 18 s) and Sec. 5.6 (X is one acquisition).

Clips differ in length, so a batch is padded and every temporal reduction reads
the frame mask. The dataset keeps the record identity even when the clip sidecar
disagrees, because the table is the authority on which split a record belongs to.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader, Dataset

from tearfilm_persistence.cohort.manifest import ClipRecord
from tearfilm_persistence.streams.acquisition import (
    AcquisitionClip,
    ClipMeta,
    PaddedClipBatch,
    load_clip,
    stack_clips,
)


class ClipDataset(Dataset[AcquisitionClip]):
    """Reads one clip per record and stamps the record identity onto it."""

    def __init__(self, records: list[ClipRecord], max_frames: int | None = None) -> None:
        if not records:
            raise ValueError("a dataset needs at least one record")
        self.records = records
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AcquisitionClip:
        record = self.records[index]
        clip = load_clip(record.path)
        meta = ClipMeta(
            eye_id=record.eye_id,
            patient_id=record.patient_id,
            site=record.site,
            region=record.region,
            session=record.session,
            repeat=record.repeat,
            arm=record.arm,
            reference_onset=record.reference_onset,
            reference_breakup=record.reference_breakup,
            symptom_score=record.symptom_score,
            severity=record.severity,
            device_generation=record.device_generation,
        )
        clip = replace(clip, meta=meta)
        if self.max_frames is not None and clip.n_frames > self.max_frames:
            clip = _uniform_crop(clip, self.max_frames)
        return clip


def _uniform_crop(clip: AcquisitionClip, length: int) -> AcquisitionClip:
    start = (clip.n_frames - length) // 2
    return clip.subset(start, start + length)


def collate(clips: list[AcquisitionClip]) -> PaddedClipBatch:
    return stack_clips(clips)


def build_loader(
    records: list[ClipRecord],
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    max_frames: int | None = None,
) -> DataLoader[AcquisitionClip]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = ClipDataset(records, max_frames=max_frames)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )


def labels_for(
    metas: list[ClipMeta], onset_threshold: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decision label, onset target and onset availability for a batch.

    The decision label is the presence of a reference onset; the onset target is
    that onset and is only usable when the reference recorded one.
    """
    labels: list[float] = []
    targets: list[float] = []
    valid: list[bool] = []
    for meta in metas:
        onset = meta.reference_onset
        has_reference = onset is not None and onset > onset_threshold
        labels.append(1.0 if has_reference else 0.0)
        targets.append(float(onset) if onset is not None and has_reference else 0.0)
        valid.append(has_reference)
    return (
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(valid, dtype=torch.bool),
    )
