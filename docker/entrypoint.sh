#!/bin/bash
# Worker entrypoint script
# Starts the simulator, then processes experiments from the work queue

set -e

WORKER_ID=${WORKER_ID:-0}
SIMULATOR_PORT=${SIMULATOR_PORT:-8080}
WORK_DIR=${WORK_DIR:-/work}
RESULTS_DIR=${RESULTS_DIR:-/results}
MODEL=${MODEL:-gpt-5-nano}
RPM_LIMIT=${RPM_LIMIT:-500}
TPM_LIMIT=${TPM_LIMIT:-500000}
WORKER_MODE=${WORKER_MODE:-ablation}

echo "=========================================="
echo "Worker $WORKER_ID starting..."
echo "  Mode: $WORKER_MODE"
echo "  Simulator port: $SIMULATOR_PORT"
echo "  Work directory: $WORK_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Model: $MODEL"
echo "  Rate limits: $RPM_LIMIT RPM, $TPM_LIMIT TPM"
echo "=========================================="

# Start the simulator in the background
echo "[Worker $WORKER_ID] Starting simulator..."
python -m homebench.smart_home_simulator --port $SIMULATOR_PORT --data-dir /app/datasets/HomeBench/hmas_format/home_description &
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

# Process experiments from the work queue
echo "[Worker $WORKER_ID] Starting experiment processing loop in $WORKER_MODE mode..."
python /app/docker/worker.py \
    --worker-id $WORKER_ID \
    --work-dir $WORK_DIR \
    --results-dir $RESULTS_DIR \
    --simulator-url "http://localhost:$SIMULATOR_PORT" \
    --model $MODEL \
    --rpm-limit $RPM_LIMIT \
    --tpm-limit $TPM_LIMIT \
    --mode $WORKER_MODE

echo "[Worker $WORKER_ID] All work completed!"
cleanup
