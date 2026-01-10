#!/bin/bash

# First argument: strategy to use (agentic_discovery, reasoning_cot, etc.)
STRATEGY=$1

set -e

# Start simulator in the background
uv run python homebench/smart_home_simulator.py --data-dir datasets/HomeBench/hmas_format/home_description --home 96 --port 8080 &

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
uv run python trace_viewer.py $RESULT_FILE \
    --html experiments/results/home96_aroma_therapy_living/${TRACE_FILE}

# Kill the simulator
pkill -f smart_home_simulator.py > /dev/null 2>&1 || true
