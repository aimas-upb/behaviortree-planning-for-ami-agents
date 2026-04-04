#!/bin/bash
# Worker entrypoint script for experience-reuse experiments.
# Starts the simulator, then processes experiments sequentially.

set -e

WORKER_ID=${WORKER_ID:-0}
SIMULATOR_PORT=${SIMULATOR_PORT:-8080}
WORK_DIR=${WORK_DIR:-/work}
RESULTS_DIR=${RESULTS_DIR:-/results}
MODEL=${MODEL:-gpt-4o}
RPM_LIMIT=${RPM_LIMIT:-500}
TPM_LIMIT=${TPM_LIMIT:-500000}
CONFIG_NAME=${CONFIG_NAME:-experience_reuse}
REASONING_EFFORT=${REASONING_EFFORT:-}
SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD:-0.85}
CLEAR_EXPERIENCE=${CLEAR_EXPERIENCE:-true}
ONTOLOGY=${ONTOLOGY:-ontologies/homeont.ttl}
HOME_CONFIG=${HOME_CONFIG:-}
SOURCE_FILE_FILTER=${SOURCE_FILE_FILTER:-}
STRUCTURED_GOAL=${STRUCTURED_GOAL:-false}
NEUROSYMBOLIC=${NEUROSYMBOLIC:-false}

echo "=========================================="
echo "Experience Worker $WORKER_ID starting..."
echo "  Config: $CONFIG_NAME"
echo "  Simulator port: $SIMULATOR_PORT"
echo "  Work directory: $WORK_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Model: $MODEL"
echo "  Reasoning effort: ${REASONING_EFFORT:-default}"
echo "  Rate limits: $RPM_LIMIT RPM, $TPM_LIMIT TPM"
echo "  Similarity threshold: $SIMILARITY_THRESHOLD"
echo "  Clear experience: $CLEAR_EXPERIENCE"
echo "  Home config: ${HOME_CONFIG:-none (loading all homes)}"
echo "=========================================="

# Start the simulator in the background
echo "[Worker $WORKER_ID] Starting simulator..."
SIMULATOR_CMD="python -m homebench.smart_home_simulator --port $SIMULATOR_PORT --data-dir /app/data/homebench/hmas/home_description"
if [ -n "$HOME_CONFIG" ]; then
    SIMULATOR_CMD="$SIMULATOR_CMD --home-config $HOME_CONFIG"
fi
$SIMULATOR_CMD &
SIMULATOR_PID=$!

# Wait for simulator to be ready
echo "[Worker $WORKER_ID] Waiting for simulator to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s "http://localhost:$SIMULATOR_PORT/workspaces" > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[Worker $WORKER_ID] ERROR: Simulator failed to start after $MAX_RETRIES attempts"
        exit 1
    fi
    sleep 1
done
echo "[Worker $WORKER_ID] Simulator is ready!"

# Function to cleanup on exit
cleanup() {
    echo "[Worker $WORKER_ID] Shutting down..."
    kill $SIMULATOR_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Build optional arguments
EXTRA_ARGS=""
if [ -n "$REASONING_EFFORT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --reasoning-effort $REASONING_EFFORT"
fi

if [ "$CLEAR_EXPERIENCE" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --clear-experience"
fi

if [ -n "$SOURCE_FILE_FILTER" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --source-file-filter $SOURCE_FILE_FILTER"
fi

if [ "$STRUCTURED_GOAL" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --structured-goal"
fi

if [ "$NEUROSYMBOLIC" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --neurosymbolic"
fi

# Process experiments from the work queue
echo "[Worker $WORKER_ID] Starting experiment processing in $CONFIG_NAME mode..."
python /app/docker/worker_experience_reuse.py \
    --worker-id $WORKER_ID \
    --work-dir $WORK_DIR \
    --results-dir $RESULTS_DIR \
    --simulator-url "http://localhost:$SIMULATOR_PORT" \
    --model $MODEL \
    --config-name $CONFIG_NAME \
    --rpm-limit $RPM_LIMIT \
    --tpm-limit $TPM_LIMIT \
    --ontology $ONTOLOGY \
    --similarity-threshold $SIMILARITY_THRESHOLD \
    $EXTRA_ARGS

echo "[Worker $WORKER_ID] All work completed!"
cleanup
