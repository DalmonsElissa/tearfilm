#!/usr/bin/env bash
# Validate a cohort table before any run reads it.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <clip-directory> <cohort.jsonl>" >&2
    exit 2
fi

clip_dir="$1"
table="$2"

if [ ! -d "$clip_dir" ]; then
    echo "clip directory not found: $clip_dir" >&2
    exit 1
fi
if [ ! -f "$table" ]; then
    echo "cohort table not found: $table" >&2
    exit 1
fi

PYTHONPATH="${PYTHONPATH:-src}" python3 - "$table" <<'PY'
import sys
from pathlib import Path

from tearfilm_persistence.cohort.manifest import ensure_patient_level, load_table

table = load_table(sys.argv[1])
audit = ensure_patient_level(table)
missing = [record.path for record in table.records if not Path(record.path).exists()]
if missing:
    raise SystemExit(f"{len(missing)} clips referenced by the table are absent, first: {missing[0]}")
print(f"records    {len(table.records)}")
print(f"splits     {audit.patients_per_split}")
print(f"eyes       {audit.eyes_per_split}")
print(f"repeated   {audit.repeated_eyes}")
print(f"sites      {list(audit.sites)}")
PY
