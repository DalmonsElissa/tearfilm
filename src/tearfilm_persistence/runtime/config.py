"""Configuration loading.

Experiment files declare the component files they build on and the sections they
override. The merge is recursive and strict about unknown sections, so a typo in
a config file fails at load rather than silently running the defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

KNOWN_SECTIONS = frozenset(
    {
        "name",
        "extends",
        "model",
        "acquisition",
        "optimisation",
        "analysis",
        "aim",
        "notes",
        "requires",
    }
)


def read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a mapping at the top level")
    unknown = set(payload) - KNOWN_SECTIONS
    if unknown:
        raise ValueError(f"{path} has unknown sections: {sorted(unknown)}")
    return dict(payload)


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(cast(dict[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Resolve an experiment file and every file it extends, in order."""
    experiment = read_yaml(path)
    root = Path(path).resolve().parent
    resolved: dict[str, Any] = {}
    for relative in experiment.get("extends", []):
        component = (root / str(relative)).resolve()
        resolved = deep_merge(resolved, read_yaml(component))
    return deep_merge(resolved, experiment)


def experiment_name(path: str | Path) -> str:
    return Path(path).stem


def flatten(config: Mapping[str, Any]) -> dict[str, Any]:
    """Dotted-path view of a nested mapping, for logging and diffing."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            for inner_key, inner_value in flatten(value).items():
                flat[f"{key}.{inner_key}"] = inner_value
        else:
            flat[str(key)] = value
    return flat
