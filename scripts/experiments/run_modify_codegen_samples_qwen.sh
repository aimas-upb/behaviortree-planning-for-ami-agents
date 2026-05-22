#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (shell or .env)}"
: "${MODIFY_CODEGEN_BACKEND:=fep_qwen_local}"
: "${MODIFY_CODEGEN_BASE_MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-Coder-3B-Instruct}"
: "${MODIFY_CODEGEN_ADAPTER_PATH:=${ADAPTER_PATH:-outputs/sft/qwen25-coder-3b-instruct-lora-homebench/best_adapter}}"
: "${MODIFY_CODEGEN_DEVICE_MAP:=auto}"
: "${MODIFY_CODEGEN_TORCH_DTYPE:=bfloat16}"
: "${MODIFY_CODEGEN_LOCAL_FILES_ONLY:=1}"
: "${MODIFY_CODEGEN_PROMPT_STYLE:=planner_exact}"
: "${MODIFY_CODEGEN_MAX_NEW_TOKENS:=2048}"
: "${MODIFY_CODEGEN_TIMEOUT_SECONDS:=180}"
: "${MODEL:=gpt-4o}"
: "${SIMULATOR_PORT:=8080}"
: "${RESULT_LABEL:=qwen}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/modify_codegen_${RESULT_LABEL}_samples_${RUN_ID}}"

export MODIFY_CODEGEN_BACKEND
export MODIFY_CODEGEN_BASE_MODEL_NAME_OR_PATH
export MODIFY_CODEGEN_ADAPTER_PATH
export MODIFY_CODEGEN_DEVICE_MAP
export MODIFY_CODEGEN_TORCH_DTYPE
export MODIFY_CODEGEN_LOCAL_FILES_ONLY
export MODIFY_CODEGEN_PROMPT_STYLE
export MODIFY_CODEGEN_TIMEOUT_SECONDS
export MODIFY_CODEGEN_MAX_NEW_TOKENS

python -m scripts.experiments.run_modify_codegen_samples_qwen_cluster \
  --model "$MODEL" \
  --data data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference.json \
  --home-config data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference_homes.json \
  --simulator-port "$SIMULATOR_PORT" \
  --output "$OUTPUT_DIR"
