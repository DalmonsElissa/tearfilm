"""Training entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from tearfilm_persistence.cli.common import (
    add_common,
    configure_logging,
    resolve_device,
    write_json,
)
from tearfilm_persistence.cohort.manifest import load_table
from tearfilm_persistence.cohort.sampling import build_loader
from tearfilm_persistence.runtime.config import flatten, load_experiment
from tearfilm_persistence.runtime.factory import build_model, train_config
from tearfilm_persistence.runtime.loop import train_model
from tearfilm_persistence.runtime.seeding import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the two-timescale tear-film model.")
    add_common(parser)
    parser.add_argument("--cohort", required=True, help="JSONL table of acquisition clips")
    parser.add_argument("--checkpoint", default="artifacts/model.pt", help="checkpoint destination")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="truncate long clips, 0 keeps all"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("tearfilm_persistence.cli.train")
    payload = load_experiment(args.experiment)
    config = train_config(payload)
    table = load_table(args.cohort)
    records = table.by_split("train")
    if not records:
        raise ValueError("the table has no records in the train split")
    logger.info(
        "training on %d records from %d sites", len(records), len({r.site for r in records})
    )
    set_seed(args.seed)
    model = build_model(payload)
    loader = build_loader(
        records,
        batch_size=config.batch_size,
        shuffle=True,
        seed=args.seed,
        max_frames=args.max_frames or None,
    )
    history = train_model(
        model,
        loader,
        config,
        device=resolve_device(args.device),
        seed=args.seed,
        checkpoint_path=Path(args.output_dir) / Path(args.checkpoint).name,
        use_repeatability=bool(payload.get("model", {}).get("use_repeatability", True)),
        resolved=flatten(payload),
    )
    write_json(Path(args.output_dir) / "training_history.json", history.as_dict())
    logger.info("checkpoint written to %s", history.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
