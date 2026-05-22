# FEP DeepSeek Coder SFT Launch

This guide launches a LoRA fine-tune of `deepseek-ai/deepseek-coder-1.3b-instruct`
on the same HomeBench conversational SFT dataset used for the Qwen2.5-Coder and
CodeGemma runs:

`data/homebench/benchmarks/modify_codegen/sft/home_disjoint/conversational/planner_exact`

It mirrors [`FEP_QWEN25_CODER_SFT.md`](FEP_QWEN25_CODER_SFT.md) and
[`FEP_CODEGEMMA_SFT.md`](FEP_CODEGEMMA_SFT.md) and reuses the same apptainer
image and venv. Only the model, output directory, run name, and Weights &
Biases project differ.

The repo assets for this run are:

- training entrypoint: `scripts/training/train_sft_lora.py`
- config: `configs/training/deepseek_coder_1_3b_instruct_lora_homebench.yaml`
- prepare batch script: `scripts/training/fep_prepare_deepseek_coder_1_3b_sft.slurm`
- training batch script: `scripts/training/fep_deepseek_coder_1_3b_lora.slurm`
- shared image definition: `docker/fep-qwen-sft.def`

The training batch script is pinned to the same FEP account/partition values
as the other runs (`student` / `dgxh100`, 2 GPUs, 12h wall time).

DeepSeek-Coder-1.3b-Instruct ships its own chat template, so the trainer uses
the tokenizer template directly with no `chat_template_source` clone. No new
special tokens are added; the trainer fine-tunes the existing weights with
LoRA only.

## 1. Prepare the runtime on FEP

Sync the repo to FEP:

```bash
rsync -avzh --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'outputs/' \
  ./ fep:~/behaviortree-planning-for-ami-agents/
```

Then submit the prepare job. It reuses `~/images/qwen25-coder-sft.sif` and
`~/.venvs/qwen25-coder-sft` if they already exist (built by the Qwen prepare
job) and pre-downloads `deepseek-ai/deepseek-coder-1.3b-instruct` into the HF
cache:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch scripts/training/fep_prepare_deepseek_coder_1_3b_sft.slurm'
```

Monitor:

```bash
ssh fep 'squeue -u $USER'
ssh fep 'tail -f ~/behaviortree-planning-for-ami-agents/slurm-deepseek-coder-1.3b-sft-prepare-<job_id>.out'
```

## 2. Make the wandb API key available to the job

The training slurm script sources `~/behaviortree-planning-for-ami-agents/.env`
before launching the trainer. Place a single line containing the key there:

```bash
ssh fep 'cat >> ~/behaviortree-planning-for-ami-agents/.env <<EOF
WANDB_API_KEY=<your wandb api key>
EOF'
```

(If you already keep secrets in that file, just confirm `WANDB_API_KEY` is
present.) The training script defaults to:

- `WANDB_MODE=online`
- `WANDB_ENTITY=r-vulpe25-universitatea-politehnica-din-bucuresti`
- `WANDB_PROJECT=deepseek-coder-fine-tune`

If the compute node cannot reach `api.wandb.ai`, override at submit time
with `--export=ALL,WANDB_MODE=offline` and run `wandb sync` from a login
node afterwards.

## 3. Submit the training job

The simplest submission uses the config paths as-is:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch scripts/training/fep_deepseek_coder_1_3b_lora.slurm'
```

For a fresh run directory and run name, pass explicit overrides:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && RUN_ID=$(date +%Y%m%d-%H%M%S) && sbatch --export=ALL,OUTPUT_DIR=outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench-$RUN_ID,LOGGING_DIR=outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench-$RUN_ID/tensorboard,RUN_NAME=deepseek-coder-1.3b-instruct-lora-homebench-$RUN_ID scripts/training/fep_deepseek_coder_1_3b_lora.slurm'
```

The batch script launches:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/training/train_sft_lora.py \
  --config configs/training/deepseek_coder_1_3b_instruct_lora_homebench.yaml
```

To recover metrics from a run whose post-training evaluation crashed, submit
the same job with `EVAL_ONLY=1` so the launcher passes `--eval-only` to the
trainer (it skips training and re-runs validation/test against
`<output_dir>/best_adapter`):

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch --export=ALL,EVAL_ONLY=1 scripts/training/fep_deepseek_coder_1_3b_lora.slurm'
```

## 4. Monitor the job

```bash
ssh fep 'squeue -u $USER'
ssh fep 'tail -f ~/behaviortree-planning-for-ami-agents/slurm-deepseek-coder-1.3b-sft-lora-2gpu-<job_id>.out ~/behaviortree-planning-for-ami-agents/slurm-deepseek-coder-1.3b-sft-lora-2gpu-<job_id>.err'
```

Live metrics:

- Weights & Biases:
  https://wandb.ai/r-vulpe25-universitatea-politehnica-din-bucuresti/deepseek-coder-fine-tune
- TensorBoard (over an SSH tunnel, see Qwen guide section 7):
  `outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench/tensorboard`

## Outputs

If you keep the default config paths, the trainer writes:

- step checkpoints: `outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench/checkpoint-*`
- best adapter export: `outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench/best_adapter`
- TensorBoard logs: `outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench/tensorboard`
- metric files: `train_results.json`, `validation_results.json`, `test_results.json`
- aggregated summary: `outputs/sft/deepseek-coder-1.3b-instruct-lora-homebench/run_summary.json`

If you launched with a timestamped `OUTPUT_DIR`, substitute that run
directory everywhere above.
