#!/bin/bash
#SBATCH --job-name=deepseek-coder-1.3b-sft-lora-2gpu
#SBATCH -A student
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

export PROJECT_DIR="$HOME/behaviortree-planning-for-ami-agents"
# Reuse the shared SFT image and venv from the qwen prepare/training jobs.
export IMAGE_PATH="${IMAGE_PATH:-$HOME/images/qwen25-coder-sft.sif}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/qwen25-coder-sft}"
export CONFIG_PATH="${CONFIG_PATH:-configs/training/deepseek_coder_1_3b_instruct_lora_homebench.yaml}"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$HOME/wandb"

# Pull WANDB_API_KEY (and any other secrets) from the cluster .env if present.
if [ -f "$PROJECT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  . "$PROJECT_DIR/.env"
  set +o allexport
fi

# Default to online wandb logging into the requested project; the user can
# override any of these at submit time with sbatch --export=ALL,...
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-deepseek-coder-fine-tune}"
export WANDB_ENTITY="${WANDB_ENTITY:-r-vulpe25-universitatea-politehnica-din-bucuresti}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "$HF_HOME" "$WANDB_DIR"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi

if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[run] WARNING: WANDB_MODE=online but WANDB_API_KEY is not set;" \
       "wandb will fall back to offline mode." >&2
fi

srun apptainer exec --nv \
  --bind "$HOME:$HOME" \
  "$IMAGE_PATH" \
  bash -lc '
    set -euo pipefail
    if [ -f "$VENV_DIR/bin/activate" ]; then
      echo "[run] Activating cluster runtime at $VENV_DIR"
      source "$VENV_DIR/bin/activate"
    else
      echo "[run] Using Python from $IMAGE_PATH"
    fi
    echo "[run] python=$(command -v python)"
    python -c "import sys; print(\"[run] sys.executable=\" + sys.executable)"
    echo "[run] HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
    echo "[run] WANDB_MODE=${WANDB_MODE:-unset} WANDB_PROJECT=${WANDB_PROJECT:-unset} WANDB_ENTITY=${WANDB_ENTITY:-unset}"
    cd "$PROJECT_DIR"
    args=(--config "$CONFIG_PATH")
    if [ -n "${OUTPUT_DIR:-}" ]; then
      args+=(--output-dir "$OUTPUT_DIR")
    fi
    if [ -n "${LOGGING_DIR:-}" ]; then
      args+=(--logging-dir "$LOGGING_DIR")
    fi
    if [ -n "${RUN_NAME:-}" ]; then
      args+=(--run-name "$RUN_NAME")
    fi
    if [ -n "${REPORT_TO:-}" ]; then
      args+=(--report-to "$REPORT_TO")
    fi
    if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
      args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
    fi
    if [ "${EVAL_ONLY:-0}" = "1" ]; then
      args+=(--eval-only)
    fi
    python -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
      scripts/training/train_sft_lora.py "${args[@]}"
  '
