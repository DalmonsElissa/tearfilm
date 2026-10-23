"""Shared CLI plumbing.

Only I/O and argument handling live here: the commands stay thin so that the
library is testable without a subprocess.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, help="experiment YAML file")
    parser.add_argument("--seed", type=int, default=0, help="seed for this run")
    parser.add_argument("--device", default="cpu", help="torch device, for example cpu or cuda:0")
    parser.add_argument("--output-dir", default="artifacts", help="directory for run artefacts")
    parser.add_argument("--verbose", action="store_true", help="log at debug level")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but none is available")
    return device


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def namespace_to_dict(namespace: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(namespace))


def parse(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> argparse.Namespace:
    return parser.parse_args(argv)
