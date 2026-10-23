#!/usr/bin/env bash
# Re-derive the three reports. The manifest is written twice so that the
# second write digests the tree the first write produced.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m tearfilm_persistence.cli.verify --twice
