"""Post-collection calibration entry point.

Ref: Sec. 5.5. The calibration is fitted on the collected descriptors of one
split and applied to another; the report states both calibration errors so that
the effect is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tearfilm_persistence.cli.common import write_json
from tearfilm_persistence.descriptors.calibration import calibration_report, fit_calibration
from tearfilm_persistence.descriptors.vector import COMPONENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit and apply the descriptor calibration.")
    parser.add_argument("--fit", required=True, help="JSONL descriptor table used to fit the maps")
    parser.add_argument(
        "--apply", required=True, help="JSONL descriptor table the maps are applied to"
    )
    parser.add_argument(
        "--output-dir", default="artifacts", help="directory for the calibration state"
    )
    return parser


def read_table(path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        missing = [
            name
            for name in (*COMPONENTS, "eye_id", "site", "score", "label")
            if name not in payload
        ]
        if missing:
            raise KeyError(f"{path}:{number} is missing fields: {', '.join(missing)}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path} contains no descriptor rows")
    matrix = np.asarray(
        [[float(row[name]) for name in COMPONENTS] for row in rows], dtype=np.float64
    )
    return {
        "matrix": matrix,
        "eyes": [str(row["eye_id"]) for row in rows],
        "sites": [str(row["site"]) for row in rows],
        "scores": np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
        "labels": np.asarray([float(row["label"]) for row in rows], dtype=np.float64),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fitting = read_table(args.fit)
    applying = read_table(args.apply)
    components, logistic = fit_calibration(
        fitting["matrix"], fitting["eyes"], fitting["sites"], fitting["scores"], fitting["labels"]
    )
    outcome = calibration_report(
        components,
        logistic,
        applying["matrix"],
        applying["eyes"],
        applying["sites"],
        applying["scores"],
        applying["labels"],
    )
    write_json(
        Path(args.output_dir) / "calibration.json",
        {
            "outcome": outcome.as_dict(),
            "variance_components": components.state(),
            "logistic": logistic.state(),
            "fit_records": int(fitting["matrix"].shape[0]),
            "applied_records": int(applying["matrix"].shape[0]),
        },
    )
    sys.stdout.write(json.dumps(outcome.as_dict(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
