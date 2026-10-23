"""Adapter that gives a comparator the batch interface of the main model.

The comparison table needs every row trained and scored under one protocol, so
the comparators are wrapped into the same output record the evaluation path
already reads. A comparator has no descriptor, which is the point of the
comparison: it reports a decision and an onset and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn

from tearfilm_persistence.comparators import build_comparator
from tearfilm_persistence.descriptors.vector import COMPONENTS, DescriptorVector
from tearfilm_persistence.models.teardyn import ClipResult, ModelOutput
from tearfilm_persistence.streams.acquisition import AcquisitionClip, PaddedClipBatch, stack_clips


class ComparatorAdapter(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.comparator = build_comparator(name)

    def forward(self, batch: PaddedClipBatch) -> ModelOutput:
        logit, onset = self.comparator(batch.frames, batch.frame_mask)
        zeros = logit.new_zeros((logit.shape[0], len(COMPONENTS)))
        descriptors = [
            DescriptorVector(
                onset_seconds=float(onset[index].item()),
                peak_instability=0.0,
                curve_slope=0.0,
                locus_entropy=0.0,
                tmh_mm=0.0,
                tma_mm2=0.0,
                declared=False,
            )
            for index in range(logit.shape[0])
        ]
        return ModelOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            onset_prediction=onset,
            descriptor_matrix=zeros,
            soft_descriptor_matrix=zeros,
            abstain=torch.zeros_like(logit, dtype=torch.bool),
            gate_reasons=tuple(() for _ in range(logit.shape[0])),
            artefact_rate=torch.zeros_like(logit),
            descriptors=descriptors,
            summaries=[],
            appearance=logit.new_zeros((logit.shape[0], 1)),
        )

    @torch.no_grad()
    def infer(self, clip: AcquisitionClip) -> ClipResult:
        self.eval()
        output = self.forward(stack_clips([clip]))
        return ClipResult(
            probability=float(output.probability[0].item()),
            onset_prediction=float(output.onset_prediction[0].item()),
            descriptor=output.descriptors[0],
            abstain=False,
            reasons=(),
            artefact_rate=0.0,
            n_subtests=0,
        )
