#!/usr/bin/env bash
# Score one split. Usage: evaluate.sh <experiment.yaml> <cohort.jsonl> <weights.pt> <split> <name>
set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "usage: $0 <experiment.yaml> <cohort.jsonl> <weights.pt> <split> <name>" >&2
    exit 2
fi

experiment="$1"
cohort="$2"
weights="$3"
split="$4"
name="$5"

export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m tearfilm_persistence.cli.evaluate \
    --experiment "$experiment" \
    --cohort "$cohort" \
    --checkpoint "$weights" \
    --split "$split" \
    --name "$name" \
    --output-dir "artifacts"
