# FEP Qwen2.5-Coder SFT Launch

This guide launches a LoRA fine-tune of `Qwen/Qwen2.5-Coder-3B-Instruct` on the HomeBench conversational SFT dataset at:

`data/homebench/benchmarks/modify_codegen/sft/home_disjoint/conversational/planner_exact`

The repo assets for this run are:

- training entrypoint: `scripts/training/train_sft_lora.py`
- config: `configs/training/qwen25_coder_3b_instruct_lora_homebench.yaml`
- batch script: `scripts/training/fep_qwen25_coder_3b_lora.slurm`
- plotting helper: `scripts/analysis/plot_sft_metrics.py`
- container definition: `docker/fep-qwen-sft.def`

The batch script is pinned to the live FEP account and partition values discovered on April 13, 2026:

- account: `student`
- partition: `dgxh100`
- GPUs: `2`
- time limit: `12:00:00`

The checked-in config already targets the adapted run:

- `num_train_epochs: 3.0`
- `report_to: [tensorboard, wandb]`
- `metric_for_best_model: eval_loss`
- `packing: false` until a supported flash-attention backend is configured

## 1. Prepare the runtime on FEP

If you do not want to build the image locally, use the repo's prepare job on FEP. It does three things directly on the cluster:

- pulls the base PyTorch `.sif` to `~/images/qwen25-coder-sft.sif`
- installs the training plus direct-inference runtime stack into `~/.venvs/qwen25-coder-sft`
- pre-downloads `Qwen/Qwen2.5-Coder-3B-Instruct` into your Hugging Face cache

If you want authenticated Hub access for the prepare step, set `HF_TOKEN` in the FEP shell or place the token at `$HF_HOME/token` before submitting. The prepare and training batch scripts will automatically reuse it when present.

First sync the repo:

```bash
rsync -avzh --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  ./ fep:~/behaviortree-planning-for-ami-agents/
```

Then submit the prepare job:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch scripts/training/fep_prepare_qwen25_coder_sft.slurm'
```

Monitor it with:

```bash
ssh fep 'squeue -u $USER'
ssh fep 'tail -f ~/behaviortree-planning-for-ami-agents/slurm-qwen25-3b-sft-prepare-<job_id>.out'
```

After that job succeeds, submit the 2x H100 training job from section 6. The training batch script will automatically activate `~/.venvs/qwen25-coder-sft` when it exists.

## 2. Optional: build the full image on FEP

If you specifically want a self-contained `.sif` built from `docker/fep-qwen-sft.def`, do it from an interactive compute allocation, not from the login node:

```bash
ssh fep
srun -A student -p xl --cpus-per-task=8 --mem=32G --time=04:00:00 --pty /bin/bash
export APPTAINER_CACHEDIR="$HOME/.cache/apptainer"
export TMPDIR="$HOME/.cache/apptainer/tmp"
mkdir -p "$HOME/images" "$APPTAINER_CACHEDIR" "$TMPDIR"
cd ~/behaviortree-planning-for-ami-agents
apptainer build --fakeroot "$HOME/images/qwen25-coder-sft.sif" docker/fep-qwen-sft.def
exit
```

If `--fakeroot` is denied on FEP, fall back to the prepare-job workflow above. That path does not require `apptainer build`.

## 3. Optional: build the runtime image locally

Run this from the repository root:

```bash
mkdir -p build
apptainer build --fakeroot build/qwen25-coder-sft.sif docker/fep-qwen-sft.def
```

## 4. Sync the repo and image to FEP

```bash
rsync -avzh --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  ./ fep:~/behaviortree-planning-for-ami-agents/

ssh fep "mkdir -p ~/images"
rsync -avzh build/qwen25-coder-sft.sif fep:~/images/qwen25-coder-sft.sif
```

## 5. Pre-stage the base model

This avoids depending on internet access from the compute node.

```bash
ssh fep 'export HF_HOME="$HOME/.cache/huggingface"; mkdir -p "$HF_HOME"; apptainer exec --bind "$HOME:$HOME" "$HOME/images/qwen25-coder-sft.sif" bash -lc '\''if [ -f "$HOME/.venvs/qwen25-coder-sft/bin/activate" ]; then source "$HOME/.venvs/qwen25-coder-sft/bin/activate"; fi; python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-Coder-3B-Instruct\", token=os.environ.get(\"HF_TOKEN\"))"'\'''
```

If you used `fep_prepare_qwen25_coder_sft.slurm`, the Hugging Face Python packages live in `~/.venvs/qwen25-coder-sft`, not in the bare base image. The command above activates that venv when present before calling `snapshot_download`.

If you want online Weights & Biases logging, export `WANDB_API_KEY` on the cluster and set `WANDB_MODE=online` before submission. The training job defaults to `HF_HUB_OFFLINE=1`, reusing the pre-staged Hugging Face cache instead of making Hub requests from the compute node.

## 6. Submit the job

The simplest submission uses the config paths as-is:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch scripts/training/fep_qwen25_coder_3b_lora.slurm'
```

For a fresh run directory and run name, pass explicit overrides at submit time:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && RUN_ID=$(date +%Y%m%d-%H%M%S) && sbatch --export=ALL,OUTPUT_DIR=outputs/sft/qwen25-coder-3b-instruct-lora-homebench-$RUN_ID,LOGGING_DIR=outputs/sft/qwen25-coder-3b-instruct-lora-homebench-$RUN_ID/tensorboard,RUN_NAME=qwen25-coder-3b-instruct-lora-homebench-$RUN_ID scripts/training/fep_qwen25_coder_3b_lora.slurm'
```

If you use the timestamped `OUTPUT_DIR` override, keep that exact run directory for every later TensorBoard, plotting, and inference command. The adapter you serve later will live under `outputs/sft/qwen25-coder-3b-instruct-lora-homebench-$RUN_ID/best_adapter`.

The batch script launches the trainer with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/training/train_sft_lora.py --config configs/training/qwen25_coder_3b_instruct_lora_homebench.yaml
```

## 7. Monitor the job

```bash
ssh fep 'squeue -u $USER'
ssh fep 'scontrol show job <job_id>'
ssh fep 'tail -f ~/behaviortree-planning-for-ami-agents/slurm-qwen25-3b-sft-lora-2gpu-<job_id>.out ~/behaviortree-planning-for-ami-agents/slurm-qwen25-3b-sft-lora-2gpu-<job_id>.err'
```

For live metric tracking with TensorBoard, open a tunnel from your local machine:

```bash
ssh -L 6006:localhost:6006 fep
```

Then, in the remote shell (recommended, works with the prepare-job runtime):

```bash
cd ~/behaviortree-planning-for-ami-agents
apptainer exec --bind "$HOME:$HOME" "$HOME/images/qwen25-coder-sft.sif" \
  bash -lc 'source "$HOME/.venvs/qwen25-coder-sft/bin/activate" && tensorboard --logdir outputs/sft/qwen25-coder-3b-instruct-lora-homebench/tensorboard --host localhost --port 6006'
```

Open `http://localhost:6006` locally.

Why this is needed: the base `.sif` may not include `tensorboard` on its own PATH. The prepare job installs it into `~/.venvs/qwen25-coder-sft`, so activating that venv inside the container makes the command available.

If your image already contains TensorBoard globally, this shorter variant also works:

```bash
cd ~/behaviortree-planning-for-ami-agents
apptainer exec --bind "$HOME:$HOME" "$HOME/images/qwen25-coder-sft.sif" \
  tensorboard --logdir outputs/sft/qwen25-coder-3b-instruct-lora-homebench/tensorboard \
  --host localhost --port 6006
```

## 8. Create train and eval plots

After the run finishes, or against the latest checkpoint during training, generate PNG plots plus a CSV export:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && apptainer exec --bind "$HOME:$HOME" "$HOME/images/qwen25-coder-sft.sif" python scripts/analysis/plot_sft_metrics.py --run-dir outputs/sft/qwen25-coder-3b-instruct-lora-homebench'
```

If you launched with a timestamped `OUTPUT_DIR`, use that path for `--run-dir`.

If you prepared the runtime with `fep_prepare_qwen25_coder_sft.slurm`, run the plotting helper inside that venv:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && apptainer exec --bind "$HOME:$HOME" "$HOME/images/qwen25-coder-sft.sif" bash -lc "source \"$HOME/.venvs/qwen25-coder-sft/bin/activate\" && python scripts/analysis/modify_codegen/plot_sft_metrics.py --run-dir outputs/sft/qwen25-coder-3b-instruct-lora-homebench"'
```

The script writes:

- `plots/loss_curve.png`
- `plots/perplexity_curve.png`
- `plots/learning_rate_curve.png`
- `plots/metric_history.csv`

It automatically falls back to the latest `checkpoint-*/trainer_state.json`, so you can rerun it mid-training to refresh the plots.

## Outputs

If you keep the default config paths, the trainer keeps rich training state and exports the best adapter weights at the end:

- step checkpoints: `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/checkpoint-*`
- best adapter export: `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/best_adapter`
- TensorBoard logs: `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/tensorboard`
- generated plots: `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/plots`
- metric files: `train_results.json`, `validation_results.json`, `test_results.json`
- aggregated summary: `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/run_summary.json`

If you launched with a timestamped `OUTPUT_DIR`, substitute that run directory everywhere above.

The training configuration uses validation loss as the tracked metric for `load_best_model_at_end`, so the `best_adapter` export corresponds to the best validation checkpoint seen during the run.
