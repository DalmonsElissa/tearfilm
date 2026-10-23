"""Checkpoint writing and reading.

Checkpoints are written to a temporary file in the destination directory and
moved into place, so an interrupted save cannot leave a half-written artefact
that a later run would load as valid.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointState:
    path: str
    epoch: int
    seed: int
    digest: str
    config: Mapping[str, Any]
    metrics: Mapping[str, float]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    seed: int,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> CheckpointState:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_payload: dict[str, Any] = dict(config or {})
    metrics_payload: dict[str, float] = {
        key: float(value) for key, value in (metrics or {}).items()
    }
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "seed": int(seed),
        "config": config_payload,
        "metrics": metrics_payload,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return CheckpointState(
        path=str(destination),
        epoch=int(epoch),
        seed=int(seed),
        digest=sha256(destination),
        config=config_payload,
        metrics=metrics_payload,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> CheckpointState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = int(payload.get("format_version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(f"checkpoint format {version} is not {FORMAT_VERSION}")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    config_payload: dict[str, Any] = dict(payload.get("config") or {})
    metrics_payload: dict[str, float] = {
        str(key): float(value) for key, value in (payload.get("metrics") or {}).items()
    }
    return CheckpointState(
        path=str(path),
        epoch=int(payload["epoch"]),
        seed=int(payload["seed"]),
        digest=sha256(path),
        config=config_payload,
        metrics=metrics_payload,
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
