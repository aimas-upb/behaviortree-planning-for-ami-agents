#!/bin/bash
#SBATCH --job-name=qwen25-3b-sft-prepare
#SBATCH -A student
#SBATCH --partition=xl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

export PROJECT_DIR="$HOME/behaviortree-planning-for-ami-agents"
export IMAGE_DIR="$HOME/images"
export IMAGE_PATH="$IMAGE_DIR/qwen25-coder-sft.sif"
export VENV_DIR="$HOME/.venvs/qwen25-coder-sft"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export APPTAINER_CACHEDIR="$HOME/.cache/apptainer"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$IMAGE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$APPTAINER_CACHEDIR" "$(dirname "$VENV_DIR")"
cd "$PROJECT_DIR"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi

if [ -f "$IMAGE_PATH" ] && ! apptainer exec "$IMAGE_PATH" python -V >/dev/null 2>&1; then
  echo "[prepare] Existing image at $IMAGE_PATH is not usable; removing it."
  rm -f "$IMAGE_PATH"
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "[prepare] Pulling base PyTorch image to $IMAGE_PATH"
  apptainer pull "$IMAGE_PATH" docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
else
  echo "[prepare] Reusing existing image at $IMAGE_PATH"
fi

apptainer exec \
  --bind "$HOME:$HOME" \
  "$IMAGE_PATH" \
  bash -lc '
    set -euo pipefail
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --no-cache-dir -r "$PROJECT_DIR/docker/fep-qwen-sft-requirements.txt"
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-Coder-3B-Instruct\", token=os.environ.get(\"HF_TOKEN\"))"
  '
