#!/bin/bash
#SBATCH --job-name=codegemma-7b-it-sft-prepare
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
# Reuse the shared SFT image and venv built by fep_prepare_qwen25_coder_sft.slurm
# (same Python/CUDA/transformers/trl stack); all training jobs are compatible.
export IMAGE_PATH="${IMAGE_PATH:-$IMAGE_DIR/qwen25-coder-sft.sif}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/qwen25-coder-sft}"
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
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
      echo "[prepare] Creating venv at $VENV_DIR"
      python -m venv --system-site-packages "$VENV_DIR"
      source "$VENV_DIR/bin/activate"
      python -m pip install --upgrade pip setuptools wheel
      python -m pip install --no-cache-dir -r "$PROJECT_DIR/docker/fep-qwen-sft-requirements.txt"
    else
      echo "[prepare] Reusing existing venv at $VENV_DIR"
      source "$VENV_DIR/bin/activate"
    fi
    python -c "import dotenv, httpx, openai, pydantic, py_trees, rdflib, requests, tqdm; import accelerate, datasets, fastapi, peft, transformers, trl, uvicorn, yaml"
    # google/codegemma-7b-it is a gated repo on the Hugging Face Hub; HF_TOKEN
    # must belong to an account that has accepted the CodeGemma license.
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(\"google/codegemma-7b-it\", token=os.environ.get(\"HF_TOKEN\"))"
  '
