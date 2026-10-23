"""Resolved configuration to typed configs.

The YAML trees are the interface a reviewer edits; the dataclasses are the
interface the code reads. Every conversion happens here so that a missing key
fails once, at startup, with the section named.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tearfilm_persistence.comparators.adapter import ComparatorAdapter
from tearfilm_persistence.deformation.adapter import AdapterConfig
from tearfilm_persistence.deformation.derivative import DerivativeConfig
from tearfilm_persistence.deformation.persistence import PersistenceConfig
from tearfilm_persistence.models.head import HeadConfig
from tearfilm_persistence.models.teardyn import ModelConfig, TearDynModel
from tearfilm_persistence.objectives.composite import ObjectiveWeights
from tearfilm_persistence.runtime.loop import TrainConfig
from tearfilm_persistence.streams.adequacy import AdequacyConfig
from tearfilm_persistence.streams.blinks import BlinkConfig
from tearfilm_persistence.streams.masking import AreaConfig
from tearfilm_persistence.structure.meniscus import PixelGeometry


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name} must be a mapping")
    return dict(value)


def model_config(payload: Mapping[str, Any]) -> ModelConfig:
    model = _section(payload, "model")
    acquisition = _section(payload, "acquisition")
    area = AreaConfig(**_section(acquisition, "area"))
    adequacy = AdequacyConfig(**_section(acquisition, "adequacy"))
    blink = BlinkConfig(**_section(acquisition, "blink"))
    geometry = PixelGeometry(**_section(acquisition, "geometry"))
    persistence = PersistenceConfig(**_section(model, "persistence"))
    adapter = AdapterConfig(**_section(model, "adapter"))
    derivative = DerivativeConfig(**_section(model, "derivative"))
    head = HeadConfig(**_section(model, "head"))
    return ModelConfig(
        area=area,
        adapter=adapter,
        persistence=persistence,
        adequacy=adequacy,
        derivative=derivative,
        head=head,
        geometry=geometry,
        blink=blink,
        fast_branch=str(model.get("fast_branch", "td")),
        use_fast=bool(model.get("use_fast", True)),
        use_persistence=bool(model.get("use_persistence", True)),
        use_structural=bool(model.get("use_structural", True)),
        use_repeatability=bool(model.get("use_repeatability", True)),
        use_gate=bool(model.get("use_gate", True)),
        persistence_substitution=str(model.get("persistence_substitution", "none")),
        freeze_trunk=bool(model.get("freeze_trunk", False)),
    ).validated()


def train_config(payload: Mapping[str, Any]) -> TrainConfig:
    optimisation = _section(payload, "optimisation")
    weights = ObjectiveWeights(**_section(optimisation, "weights"))
    return TrainConfig(
        epochs=int(optimisation.get("epochs", 40)),
        learning_rate=float(optimisation.get("learning_rate", 3e-4)),
        weight_decay=float(optimisation.get("weight_decay", 1e-2)),
        batch_size=int(optimisation.get("batch_size", 8)),
        grad_accumulation=int(optimisation.get("grad_accumulation", 1)),
        warmup_steps=int(optimisation.get("warmup_steps", 500)),
        grad_clip=float(optimisation.get("grad_clip", 1.0)),
        precision=str(optimisation.get("precision", "bf16")),
        specificity_floor=float(optimisation.get("specificity_floor", 0.90)),
        weights=weights,
    ).validated()


def build_model(payload: Mapping[str, Any]) -> TearDynModel | ComparatorAdapter:
    """Build the configured model, or the comparator named in the same section."""
    model = _section(payload, "model")
    comparator = model.get("comparator")
    if comparator:
        return ComparatorAdapter(str(comparator))
    return TearDynModel(model_config(payload))


def analysis_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _section(payload, "analysis")
