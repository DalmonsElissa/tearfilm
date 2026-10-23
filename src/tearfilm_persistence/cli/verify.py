"""Verification entry point.

Runs both layers, writes the claim map, the verification report and the
integrity manifest, and exits non-zero when a code check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tearfilm_persistence.verification.report import collect, summary_text, write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-derive the release verification reports.")
    parser.add_argument("--root", default=None, help="release root; defaults to the package parent")
    parser.add_argument("--twice", action="store_true", help="write the manifest a second time")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    bundle = collect(root)
    written = write_reports(root, bundle)
    if args.twice:
        written = write_reports(root, bundle)
    sys.stdout.write(summary_text(bundle))
    sys.stdout.write("\n" + "\n".join(f"{key}: {path}" for key, path in written.items()) + "\n")
    report = json.loads((root / "verification_report.json").read_text(encoding="utf-8"))
    return 0 if report["summary"]["code_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
