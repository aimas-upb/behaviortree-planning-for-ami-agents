#!/bin/bash

# First argument: strategy to use (agentic_discovery, reasoning_cot, etc.)
STRATEGY=$1

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Start simulator in the background
uv run python -m homebench.smart_home_simulator --data-dir data/homebench/hmas/home_description --home 96 --port 8080 &

# Run experiment
uv run python -m src.runner \
    --config experiments/configs/${STRATEGY}.yaml \
    --goal "Increase the interval of the aromatherapy device by 25 seconds in the living room." \
    --home 96 \
    --output experiments/results/home96_aroma_therapy_living \
    --verbose

# Find the name of the generated result file
RESULT_FILE=$(ls experiments/results/home96_aroma_therapy_living/${STRATEGY}_*.json | head -n 1)

# The name of the trace file
TRACE_FILE=$(basename $RESULT_FILE .json)_trace.html

# Generate trace view
uv run python -m viewers.trace_viewer $RESULT_FILE \
    --html experiments/results/home96_aroma_therapy_living/${TRACE_FILE}

# Kill the simulator
pkill -f smart_home_simulator.py > /dev/null 2>&1 || true
