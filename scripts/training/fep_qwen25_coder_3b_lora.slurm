#!/bin/bash
#SBATCH --job-name=qwen25-3b-sft-lora-2gpu
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
export IMAGE_PATH="$HOME/images/qwen25-coder-sft.sif"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/qwen25-coder-sft}"
export CONFIG_PATH="${CONFIG_PATH:-configs/training/qwen25_coder_3b_instruct_lora_homebench.yaml}"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "$HF_HOME" "$WANDB_DIR"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
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
    if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
      args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
    fi
    python -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
      scripts/training/train_sft_lora.py "${args[@]}"
  '
