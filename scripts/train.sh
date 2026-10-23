#!/usr/bin/env bash
# Fit one configuration. Usage: train.sh <experiment.yaml> <cohort.jsonl> [seed]
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <experiment.yaml> <cohort.jsonl> [seed]" >&2
    exit 2
fi

experiment="$1"
cohort="$2"
seed="${3:-0}"

export PYTHONPATH="${PYTHONPATH:-src}"
python3 -m tearfilm_persistence.cli.train \
    --experiment "$experiment" \
    --cohort "$cohort" \
    --seed "$seed" \
    --output-dir "artifacts/$(basename "${experiment%.yaml}")"
