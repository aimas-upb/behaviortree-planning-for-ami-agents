#!/bin/bash
#SBATCH --job-name=codegemma-7b-it-samples
#SBATCH -A student
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$HOME/behaviortree-planning-for-ami-agents}"
# Reuse the shared SFT image and venv (same Python/CUDA/transformers stack as Qwen).
export IMAGE_PATH="${IMAGE_PATH:-$HOME/images/qwen25-coder-sft.sif}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/qwen25-coder-sft}"
export BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-google/codegemma-7b-it}"
export ADAPTER_PATH="${ADAPTER_PATH:-outputs/sft/codegemma-7b-it-lora-homebench/best_adapter}"
export RESULT_LABEL="${RESULT_LABEL:-codegemma_7b_it}"
export OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/modify_codegen_${RESULT_LABEL}_samples_${SLURM_JOB_ID}}"
export MODEL="${MODEL:-gpt-4o}"
export MODIFY_CODEGEN_DEVICE_MAP="${MODIFY_CODEGEN_DEVICE_MAP:-auto}"
export MODIFY_CODEGEN_TORCH_DTYPE="${MODIFY_CODEGEN_TORCH_DTYPE:-bfloat16}"
export MODIFY_CODEGEN_LOCAL_FILES_ONLY="${MODIFY_CODEGEN_LOCAL_FILES_ONLY:-1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

resolve_project_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$PROJECT_DIR/$path"
  fi
}

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[codegemma-samples] ERROR: Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "[codegemma-samples] ERROR: Container image not found: $IMAGE_PATH" >&2
  exit 1
fi

cd "$PROJECT_DIR"
ADAPTER_PATH="$(resolve_project_path "$ADAPTER_PATH")"

if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "[codegemma-samples] ERROR: Missing adapter_config.json at $ADAPTER_PATH" >&2
  echo "[codegemma-samples] If training used a timestamped OUTPUT_DIR, submit with that exact adapter path." >&2
  find "$PROJECT_DIR/outputs" -maxdepth 5 -type f -name adapter_config.json -printf "  %h\n" 2>/dev/null | sort >&2 || true
  exit 1
fi

export MODIFY_CODEGEN_BACKEND=fep_qwen_local
export MODIFY_CODEGEN_BASE_MODEL_NAME_OR_PATH="$BASE_MODEL_NAME_OR_PATH"
export MODIFY_CODEGEN_ADAPTER_PATH="$ADAPTER_PATH"

echo "[codegemma-samples] Hostname: $(hostname)"
echo "[codegemma-samples] Project: $PROJECT_DIR"
echo "[codegemma-samples] Image: $IMAGE_PATH"
echo "[codegemma-samples] Adapter: $ADAPTER_PATH"
echo "[codegemma-samples] Output: $OUTPUT_DIR"
echo "[codegemma-samples] Model: $MODEL"

srun apptainer exec --nv \
  --bind "$HOME:$HOME" \
  "$IMAGE_PATH" \
  bash -lc '
    set -euo pipefail
    cd "$PROJECT_DIR"
    if [ -d "$VENV_DIR" ]; then
      echo "[codegemma-samples] Activating cluster runtime at $VENV_DIR"
      source "$VENV_DIR/bin/activate"
    fi
    bash scripts/experiments/run_modify_codegen_samples_qwen.sh
  '
