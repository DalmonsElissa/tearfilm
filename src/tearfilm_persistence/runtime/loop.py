"""Training and scoring loops.

The loop carries the three objective terms through one backward pass per
gradient-accumulation window, and it records the seed and the resolved config in
the checkpoint so a resumed run is the same run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tearfilm_persistence.cohort.sampling import labels_for
from tearfilm_persistence.evaluation.aggregate import ScoredEye
from tearfilm_persistence.objectives.composite import ObjectiveWeights, composite_loss
from tearfilm_persistence.runtime.checkpoint import save_checkpoint
from tearfilm_persistence.streams.acquisition import ClipMeta, PaddedClipBatch

LOGGER = logging.getLogger("tearfilm_persistence.train")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    grad_accumulation: int = 1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16"
    specificity_floor: float = 0.90
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def validated(self) -> TrainConfig:
        if self.epochs < 1:
            raise ValueError("at least one epoch")
        if self.batch_size < 2:
            raise ValueError("the descriptor head normalises over the batch, so batch >= 2")
        if self.grad_accumulation < 1:
            raise ValueError("accumulation must be at least one")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unknown precision: {self.precision}")
        return self


@dataclass
class EpochRecord:
    epoch: int
    total: float
    decision: float
    onset: float
    repeatability: float
    steps: int


@dataclass
class TrainingHistory:
    epochs: list[EpochRecord] = field(default_factory=list)
    checkpoint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": [record.__dict__ for record in self.epochs],
            "checkpoint": self.checkpoint,
            "final_total": self.epochs[-1].total if self.epochs else float("nan"),
        }


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(config.warmup_steps, 1)
    horizon = max(total_steps, warmup + 1)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(horizon - warmup, 1)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def group_tensor(metas: list[ClipMeta]) -> torch.Tensor:
    order: dict[str, int] = {}
    identifiers: list[int] = []
    for meta in metas:
        if meta.eye_id not in order:
            order[meta.eye_id] = len(order)
        identifiers.append(order[meta.eye_id])
    return torch.tensor(identifiers, dtype=torch.long)


def _autocast_context(precision: str, device: torch.device) -> Any:
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_model(
    model: nn.Module,
    loader: DataLoader[Any],
    config: TrainConfig,
    device: torch.device,
    seed: int,
    checkpoint_path: str | Path,
    use_repeatability: bool = True,
    resolved: Mapping[str, Any] | None = None,
) -> TrainingHistory:
    cfg = config.validated()
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps=len(loader) * cfg.epochs)
    history = TrainingHistory()
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        totals = {"total": 0.0, "decision": 0.0, "onset": 0.0, "repeatability": 0.0}
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            batch = _to_device(batch, device)
            with _autocast_context(cfg.precision, device):
                output = model(batch)
                label, onset_target, onset_valid = labels_for(batch.metas)
                label = label.to(device)
                onset_target = onset_target.to(device)
                onset_valid = onset_valid.to(device)
                valid = (
                    (~output.abstain)
                    if bool(output.abstain.any())
                    else torch.ones_like(label, dtype=torch.bool)
                )
                breakdown = composite_loss(
                    logit=output.logit,
                    label=label,
                    valid_decision=valid,
                    onset_prediction=output.onset_prediction,
                    onset_target=onset_target,
                    valid_onset=onset_valid & valid,
                    descriptors=output.soft_descriptor_matrix,
                    groups=group_tensor(batch.metas).to(device),
                    weights=cfg.weights,
                    use_repeatability=use_repeatability,
                )
                loss = breakdown.total / cfg.grad_accumulation
            loss.backward()  # type: ignore[no-untyped-call]
            totals["total"] += float(breakdown.total.item())
            totals["decision"] += float(breakdown.decision.item())
            totals["onset"] += float(breakdown.onset.item())
            totals["repeatability"] += float(breakdown.repeatability.item())
            batches += 1
            if (batch_index + 1) % cfg.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        history.epochs.append(
            EpochRecord(
                epoch=epoch,
                total=totals["total"] / max(batches, 1),
                decision=totals["decision"] / max(batches, 1),
                onset=totals["onset"] / max(batches, 1),
                repeatability=totals["repeatability"] / max(batches, 1),
                steps=step,
            )
        )
        LOGGER.info(
            "epoch %d/%d loss=%.4f decision=%.4f onset=%.4f repeat=%.4f",
            epoch + 1,
            cfg.epochs,
            history.epochs[-1].total,
            history.epochs[-1].decision,
            history.epochs[-1].onset,
            history.epochs[-1].repeatability,
        )
    state = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=cfg.epochs,
        seed=seed,
        config=resolved or {},
        metrics={"final_loss": history.epochs[-1].total if history.epochs else float("nan")},
    )
    history.checkpoint = state.path
    return history


def _to_device(batch: PaddedClipBatch, device: torch.device) -> PaddedClipBatch:
    return PaddedClipBatch(
        frames=batch.frames.to(device),
        fixation=batch.fixation.to(device),
        fps=batch.fps.to(device),
        frame_mask=batch.frame_mask.to(device),
        metas=batch.metas,
        lengths=batch.lengths.to(device),
    )


@torch.no_grad()
def score_loader(
    model: nn.Module,
    loader: Iterable[PaddedClipBatch],
    device: torch.device,
) -> list[ScoredEye]:
    model.eval()
    rows: list[ScoredEye] = []
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(batch)
        for index, meta in enumerate(batch.metas):
            rows.append(
                ScoredEye(
                    eye_id=meta.eye_id,
                    patient_id=meta.patient_id,
                    site=meta.site,
                    region=meta.region,
                    arm=meta.arm,
                    severity=meta.severity,
                    reference_onset=meta.reference_onset,
                    declared_onset=(
                        output.descriptors[index].onset_seconds
                        if output.descriptors[index].declared
                        else None
                    ),
                    score=float(output.probability[index].item()),
                    abstained=bool(output.abstain[index].item()),
                    descriptor=output.descriptors[index].as_dict(),
                )
            )
    return rows
