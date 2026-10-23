"""Evaluation entry point."""

from __future__ import annotations

import argparse
import json
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
from tearfilm_persistence.evaluation.aggregate import (
    ScoredEye,
    scorable,
    site_table,
    stratum_table,
    summarise_arm,
)
from tearfilm_persistence.runtime.checkpoint import load_checkpoint
from tearfilm_persistence.runtime.config import load_experiment
from tearfilm_persistence.runtime.factory import analysis_config, build_model, train_config
from tearfilm_persistence.runtime.loop import score_loader
from tearfilm_persistence.runtime.seeding import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a held-out split and write its report.")
    add_common(parser)
    parser.add_argument("--cohort", required=True, help="JSONL table of acquisition clips")
    parser.add_argument("--checkpoint", required=True, help="checkpoint to evaluate")
    parser.add_argument("--split", default="test", help="split to score")
    parser.add_argument("--name", default="evaluation", help="stem of the report file")
    parser.add_argument(
        "--dump-descriptors",
        default="",
        help="write one JSON object per scored eye, for the calibration step",
    )
    return parser


def _descriptor_row(row: ScoredEye) -> dict[str, object]:
    payload: dict[str, object] = {
        "eye_id": row.eye_id,
        "site": row.site,
        "score": row.score,
        "label": 1.0 if row.reference_onset is not None else 0.0,
        "abstained": row.abstained,
    }
    if row.descriptor is not None:
        payload.update({name: float(value) for name, value in row.descriptor.items()})
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    logger = logging.getLogger("tearfilm_persistence.cli.evaluate")
    payload = load_experiment(args.experiment)
    training = train_config(payload)
    analysis = analysis_config(payload)
    set_seed(args.seed)
    model = build_model(payload)
    state = load_checkpoint(args.checkpoint, model)
    logger.info("loaded checkpoint epoch %d seed %d", state.epoch, state.seed)
    table = load_table(args.cohort)
    records = table.by_split(args.split)
    if not records:
        raise ValueError(f"the table has no records in the {args.split} split")
    loader = build_loader(records, batch_size=training.batch_size, shuffle=False, seed=args.seed)
    rows = score_loader(model, loader, resolve_device(args.device))
    report = {
        "split": args.split,
        "n_eyes": len(rows),
        "n_abstained": len(rows) - len(scorable(rows)),
        "arm": summarise_arm(
            rows,
            specificity_floor=training.specificity_floor,
            n_boot=int(analysis.get("bootstrap", 1000)),
            seed=args.seed,
        ).as_dict(),
        "sites": [entry.__dict__ for entry in site_table(rows, training.specificity_floor)],
        "severity": {
            name: value.as_dict() for name, value in stratum_table(rows, "severity").items()
        },
    }
    write_json(Path(args.output_dir) / f"{args.name}.json", report)
    if args.dump_descriptors:
        destination = Path(args.dump_descriptors)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_descriptor_row(row)) + "\n")
    logger.info("wrote %s.json", args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
