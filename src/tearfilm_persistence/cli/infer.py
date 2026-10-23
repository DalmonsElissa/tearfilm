"""Single-acquisition inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tearfilm_persistence.cli.common import (
    add_common,
    configure_logging,
    resolve_device,
)
from tearfilm_persistence.runtime.checkpoint import load_checkpoint
from tearfilm_persistence.runtime.config import load_experiment
from tearfilm_persistence.runtime.factory import build_model
from tearfilm_persistence.streams.acquisition import load_clip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure one acquisition.")
    add_common(parser)
    parser.add_argument("--clip", required=True, help=".npz clip with its .json sidecar")
    parser.add_argument("--checkpoint", required=True, help="checkpoint to read")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    payload = load_experiment(args.experiment)
    model = build_model(payload)
    load_checkpoint(args.checkpoint, model)
    model.to(resolve_device(args.device))
    clip = load_clip(Path(args.clip))
    result = model.infer(clip)
    payload = {
        "eye_id": clip.meta.eye_id,
        "site": clip.meta.site,
        "abstained": result.abstain,
        "reasons": list(result.reasons),
        "probability": result.probability,
        "predicted_onset_seconds": result.onset_prediction,
        "declared_onset_seconds": result.descriptor.onset_seconds,
        "declared": result.descriptor.declared,
        "descriptor": result.descriptor.as_dict(),
        "artefact_rate": result.artefact_rate,
        "n_subtests": result.n_subtests,
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
