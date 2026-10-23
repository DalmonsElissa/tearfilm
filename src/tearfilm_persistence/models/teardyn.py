"""The two-timescale model.

Ref: Sec. 5.4 and Algorithm 1.

One forward pass is the normative system of Sec. 5.4: adequacy appraisal,
frame-to-tear-film representation, derivative evaluation, persistence
integration, and descriptor adjustment. The morphological components can be
withheld one at a time, which is what the component table reports; a withheld
component zeroes its own contribution rather than silently substituting another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tearfilm_persistence.deformation.adapter import (
    AdapterConfig,
    PlainClipBackbone,
    TemporalDifferenceBackbone,
)
from tearfilm_persistence.deformation.curve import CurveSummary, summarise_clip
from tearfilm_persistence.deformation.derivative import DerivativeConfig, TemporalDerivative
from tearfilm_persistence.deformation.persistence import (
    CurveSegment,
    PersistenceConfig,
    PersistenceOperator,
    area_fraction,
    soft_occupancy,
    tau_frames,
)
from tearfilm_persistence.descriptors.vector import COMPONENTS, DescriptorVector
from tearfilm_persistence.models.gate import AdequacyGate, GateOutput
from tearfilm_persistence.models.head import DescriptorHead, HeadConfig
from tearfilm_persistence.streams.acquisition import (
    AcquisitionClip,
    PaddedClipBatch,
    stack_clips,
)
from tearfilm_persistence.streams.adequacy import AdequacyConfig
from tearfilm_persistence.streams.blinks import BlinkConfig, detect_blinks, subtest_intervals
from tearfilm_persistence.streams.masking import AreaConfig, band_pixels, build_area
from tearfilm_persistence.structure.meniscus import PixelGeometry, StructuralBranch

FAST_BRANCHES: Final[tuple[str, ...]] = ("td", "conv3d")


@dataclass(frozen=True)
class ModelConfig:
    area: AreaConfig = field(default_factory=AreaConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    adequacy: AdequacyConfig = field(default_factory=AdequacyConfig)
    derivative: DerivativeConfig = field(default_factory=DerivativeConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    geometry: PixelGeometry = field(default_factory=PixelGeometry)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    fast_branch: str = "td"
    use_fast: bool = True
    use_persistence: bool = True
    use_structural: bool = True
    use_repeatability: bool = True
    use_gate: bool = True
    persistence_substitution: str = "none"
    freeze_trunk: bool = False

    def validated(self) -> ModelConfig:
        if self.fast_branch not in FAST_BRANCHES:
            raise ValueError(f"unknown fast branch: {self.fast_branch}")
        if self.persistence_substitution not in {"none", "mean_pool"}:
            raise ValueError(f"unknown persistence substitution: {self.persistence_substitution}")
        return self


@dataclass
class ModelOutput:
    logit: torch.Tensor
    probability: torch.Tensor
    onset_prediction: torch.Tensor
    descriptor_matrix: torch.Tensor
    soft_descriptor_matrix: torch.Tensor
    abstain: torch.Tensor
    gate_reasons: tuple[tuple[str, ...], ...]
    artefact_rate: torch.Tensor
    descriptors: list[DescriptorVector]
    summaries: list[CurveSummary]
    appearance: torch.Tensor


@dataclass
class ClipResult:
    probability: float
    onset_prediction: float
    descriptor: DescriptorVector
    abstain: bool
    reasons: tuple[str, ...]
    artefact_rate: float
    n_subtests: int


def _pool_field(field_map: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """``[B, 1, T, h, w] -> [B, P, T]`` on the locus grid."""
    batch, _, steps, height, width = field_map.shape
    flat = field_map.reshape(batch * steps, 1, height, width)
    pooled = F.adaptive_avg_pool2d(flat, (rows, cols)).reshape(batch, steps, rows * cols)
    return pooled.transpose(1, 2)


def artefact_fraction(derivative: torch.Tensor, mask: torch.Tensor, span: int) -> float:
    """Share of loci whose derivative reverses inside the early window.

    Ref: Algorithm 1 line 13. The channel is reported, not discarded: a locus
    that reverses inside one blink cycle is the artefact class the persistence
    requirement is designed to refuse.
    """
    if derivative.shape[-1] < 3 or span < 1:
        return 0.0
    window = derivative[:, : max(span + 1, 3)]
    signs = torch.sign(window)
    changes = (signs[:, 1:] != signs[:, :-1]).any(dim=1)
    weights = mask.flatten().to(changes.dtype)
    total = weights.sum().clamp_min(1.0)
    return float((changes.to(weights.dtype) * weights).sum().item() / float(total.item()))


def _soft_description(
    derivative: torch.Tensor,
    mask: torch.Tensor,
    fps: float,
    span: int,
    alpha: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable dynamic descriptor and the soft curve behind it."""
    occupancy = soft_occupancy(derivative, span, gamma)
    curve = area_fraction(occupancy, mask)[0]
    steps = int(curve.numel())
    if steps < 2:
        return curve.new_zeros(4), curve
    crossing = torch.sigmoid(gamma * (curve - alpha))
    survival = torch.cumprod(
        torch.cat([torch.ones(1, dtype=curve.dtype, device=curve.device), 1.0 - crossing[:-1]]),
        dim=0,
    )
    crossing_pmf = crossing * survival
    positions = torch.arange(steps, dtype=curve.dtype, device=curve.device) / fps
    mass = crossing_pmf.sum()
    soft_onset = torch.where(
        mass > 1e-6,
        (crossing_pmf * positions).sum() / mass.clamp_min(1e-6),
        torch.full_like(mass, steps / fps),
    )
    soft_peak = curve.max()
    time_index = positions
    centred = time_index - time_index.mean()
    denominator = (centred * centred).sum().clamp_min(1e-9)
    soft_slope = (centred * (curve - curve.mean())).sum() / denominator
    peak_index = int(torch.argmax(curve).item())
    shares = occupancy[:, peak_index].clamp_min(0.0)
    total = shares.sum().clamp_min(1e-6)
    share = shares / total
    entropy = -(share * (share + 1e-9).log()).sum() / float(np.log(max(int(mask.sum().item()), 2)))
    dynamic = torch.stack([soft_onset, soft_peak, soft_slope, entropy.clamp(0.0, 1.0)])
    return dynamic, curve


class TearDynModel(nn.Module):
    """Two-timescale tear-film measurement model."""

    area_mask: torch.Tensor

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = (config or ModelConfig()).validated()
        if self.config.fast_branch == "td":
            self.backbone: nn.Module = TemporalDifferenceBackbone(self.config.adapter)
        else:
            self.backbone = PlainClipBackbone(self.config.adapter)
        if self.config.freeze_trunk and isinstance(self.backbone, TemporalDifferenceBackbone):
            for parameter in self.backbone.trunk.parameters():
                parameter.requires_grad_(False)
        self.derivative = TemporalDerivative(self.config.derivative)
        self.operator = PersistenceOperator(
            PersistenceConfig(
                tau_seconds=self.config.persistence.tau_seconds,
                area_fraction=self.config.persistence.area_fraction,
                gamma=self.config.persistence.gamma,
                mode="hard",
            )
        )
        self.structural = StructuralBranch(self.config.geometry)
        self.gate = AdequacyGate(self.config.adequacy)
        self.head = DescriptorHead(
            HeadConfig(
                n_components=len(COMPONENTS),
                appearance_dim=self.config.adapter.hidden_channels,
                hidden=self.config.head.hidden,
                dropout=self.config.head.dropout,
            )
        )
        self.area = build_area(self.config.area)
        self.register_buffer("area_mask", self.area.mask.flatten(), persistent=True)

    def train(self, mode: bool = True) -> TearDynModel:
        super().train(mode)
        return self

    def _encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, T, 3, H, W] -> locus field [B, P, T], appearance [B, C]``."""
        clip_frames = frames.transpose(1, 2)
        field_map, features = self.backbone(clip_frames)
        locus_field = _pool_field(field_map, self.area.rows, self.area.cols)
        locus_field = locus_field * self.area.mask.flatten().to(locus_field.dtype).view(1, -1, 1)
        appearance = features.mean(dim=2).mean(dim=(2, 3))
        return locus_field, appearance

    def _reference_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames[:, 0]

    def forward(self, batch: PaddedClipBatch) -> ModelOutput:
        frames = batch.frames
        locus_field, appearance = self._encode(frames)
        mask = self.area_mask.to(locus_field.device)
        band = band_pixels(self.config.area, (frames.shape[-2], frames.shape[-1]))

        soft_rows: list[torch.Tensor] = []
        descriptor_rows: list[torch.Tensor] = []
        descriptors: list[DescriptorVector] = []
        summaries: list[CurveSummary] = []
        artefacts: list[float] = []

        tmh, tma = self.structural(self._reference_frames(frames), (band[0], band[1]))

        for index, meta in enumerate(batch.metas):
            length = int(batch.lengths[index].item())
            fps = float(batch.fps[index].item())
            clip = AcquisitionClip(
                frames=frames[index, :length],
                fps=fps,
                fixation=batch.fixation[index, :length],
                meta=meta,
            )
            blinks = detect_blinks(clip, self.config.blink)
            intervals = subtest_intervals(blinks, length)
            derivative = self.derivative(locus_field[index : index + 1, :, :length], fps)
            if not self.config.use_fast:
                derivative = torch.zeros_like(derivative)
            span = tau_frames(self.config.persistence.tau_seconds, fps)
            if not self.config.use_persistence:
                span = 0

            segments = self.operator.curve(derivative, mask, intervals, fps)
            if self.config.persistence_substitution == "mean_pool":
                segments = [
                    CurveSegment(
                        fractions=segment.fractions.mean().expand_as(segment.fractions),
                        occupancy=segment.occupancy.mean(dim=-1, keepdim=True).expand_as(
                            segment.occupancy
                        ),
                        start_frame=segment.start_frame,
                        fps=segment.fps,
                    )
                    for segment in segments
                ]
            summary = summarise_clip(segments, mask, self.config.persistence.area_fraction)
            summaries.append(summary)

            if self.config.use_structural:
                structural_hard = torch.stack([tmh[index].detach(), tma[index].detach()])
                structural_soft = torch.stack([tmh[index], tma[index]])
            else:
                structural_hard = locus_field.new_zeros(2)
                structural_soft = locus_field.new_zeros(2)
            vector = DescriptorVector.from_summary(
                summary,
                tmh_mm=float(structural_hard[0].item()),
                tma_mm2=float(structural_hard[1].item()),
            )
            descriptors.append(vector)
            descriptor_rows.append(
                torch.cat(
                    [
                        vector.as_tensor()[:4].to(locus_field.dtype),
                        structural_hard.to(locus_field.dtype),
                    ]
                )
            )

            start = intervals[0][0] if intervals else 0
            soft_dynamic, _ = _soft_description(
                derivative[:, :, start:],
                mask,
                fps,
                span,
                self.config.persistence.area_fraction,
                self.config.persistence.gamma,
            )
            soft_rows.append(torch.cat([soft_dynamic, structural_soft.to(soft_dynamic.dtype)]))

            if intervals:
                artefacts.append(
                    max(
                        artefact_fraction(derivative[:, :, begin:stop][0], mask, span)
                        for begin, stop in intervals
                    )
                )
            else:
                artefacts.append(0.0)

        descriptor_matrix = torch.stack(descriptor_rows)
        soft_matrix = torch.stack(soft_rows)
        gate_output = self._gate(batch, (int(frames.shape[-2]), int(frames.shape[-1])))
        head_input = soft_matrix if self.training else descriptor_matrix
        head_output = self.head(head_input, appearance)
        probability = torch.sigmoid(head_output.logit)
        abstain = (
            ~gate_output.accept if self.config.use_gate else torch.zeros_like(gate_output.accept)
        )
        return ModelOutput(
            logit=head_output.logit,
            probability=probability,
            onset_prediction=head_output.onset_seconds,
            descriptor_matrix=descriptor_matrix,
            soft_descriptor_matrix=soft_matrix,
            abstain=abstain,
            gate_reasons=gate_output.reasons,
            artefact_rate=torch.tensor(
                artefacts, dtype=probability.dtype, device=probability.device
            ),
            descriptors=descriptors,
            summaries=summaries,
            appearance=appearance,
        )

    def _gate(self, batch: PaddedClipBatch, frame_shape: tuple[int, int]) -> GateOutput:
        decision: GateOutput = self.gate(batch.fps, batch.fixation, batch.frame_mask, frame_shape)
        return decision

    @torch.no_grad()
    def infer(self, clip: AcquisitionClip) -> ClipResult:
        self.eval()
        output = self.forward(stack_clips([clip]))
        return ClipResult(
            probability=float(output.probability[0].item()),
            onset_prediction=float(output.onset_prediction[0].item()),
            descriptor=output.descriptors[0],
            abstain=bool(output.abstain[0].item()),
            reasons=output.gate_reasons[0],
            artefact_rate=float(output.artefact_rate[0].item()),
            n_subtests=output.summaries[0].n_subtests,
        )


def describe_components() -> tuple[str, ...]:
    return COMPONENTS
