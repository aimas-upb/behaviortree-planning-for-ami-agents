# FEP Qwen2.5-Coder Sample Inference Evaluation

This guide runs the sampled modify-codegen benchmark entirely on the FEP
cluster. The fine-tuned Qwen model is loaded directly inside the benchmark
process on the compute node, so there is no FastAPI server and no SSH tunnel.

It assumes you already have:

- the FEP runtime prepared from [docs/guides/FEP_QWEN25_CODER_SFT.md](docs/guides/FEP_QWEN25_CODER_SFT.md)
- a trained adapter directory on FEP, typically `outputs/sft/qwen25-coder-3b-instruct-lora-homebench/best_adapter`
- an `OPENAI_API_KEY` available on the cluster, either in the repo `.env` or exported in the submission shell

If your `~/.venvs/qwen25-coder-sft` was created before the direct-inference
dependencies were added to the prepare job, rerun the prepare step once before
submitting the sampled benchmark.

## 1. Sync the repo

```bash
rsync -avzh --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'outputs/' \
  ./ fep:~/behaviortree-planning-for-ami-agents/
```

The `outputs/` tree is excluded on purpose so repeated syncs do not delete
on-cluster checkpoints or adapter exports.

## 2. Confirm the adapter path on FEP

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && find outputs -maxdepth 5 -name adapter_config.json -printf "%h\n"'
```

Use the directory that contains `adapter_config.json` as `ADAPTER_PATH`.

## 3. Ensure the cluster job can see `OPENAI_API_KEY`

The sampled benchmark still uses OpenAI for intent extraction. The easiest path
is to keep `OPENAI_API_KEY` in `~/behaviortree-planning-for-ami-agents/.env`
on FEP because `scripts/experiments/run_modify_codegen_samples_qwen.sh`
automatically sources that file.

If you do not want to store the key in `.env`, export it in the shell before
`sbatch` or pass it through `--export=ALL,...`.

## 4. Submit the sampled benchmark job

The checked-in SLURM wrapper starts the HomeBench simulator locally on the
compute node and runs the sampled benchmark directly against the in-process
Qwen backend:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch --export=ALL,ADAPTER_PATH=outputs/sft/qwen25-coder-3b-instruct-lora-homebench-<run_id>/best_adapter scripts/experiments/fep/fep_run_modify_codegen_samples_qwen.slurm'
```

Useful overrides:

```bash
ssh fep 'cd ~/behaviortree-planning-for-ami-agents && sbatch --export=ALL,ADAPTER_PATH=outputs/sft/qwen25-coder-3b-instruct-lora-homebench-<run_id>/best_adapter,MODEL=gpt-4o,OUTPUT_DIR=experiments/results/modify_codegen_qwen_samples_<run_id> scripts/experiments/fep/fep_run_modify_codegen_samples_qwen.slurm'
```

The job wrapper exports:

- `MODIFY_CODEGEN_BACKEND=fep_qwen_local`
- `MODIFY_CODEGEN_BASE_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-Coder-3B-Instruct`
- `MODIFY_CODEGEN_ADAPTER_PATH=<resolved absolute adapter path>`

so the Qwen model is loaded directly inside the sampled benchmark process.
Inside the container, the wrapper activates `~/.venvs/qwen25-coder-sft` when it
exists. The inner runner prefers `uv` when available, but on FEP compute nodes
it now automatically falls back to the active `python` from that venv.

## 5. Monitor the job

```bash
ssh fep 'squeue -u $USER'
ssh fep 'sacct -j <job_id> --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList -P'
ssh fep 'tail -f ~/behaviortree-planning-for-ami-agents/slurm-qwen25-3b-samples-<job_id>.out ~/behaviortree-planning-for-ami-agents/slurm-qwen25-3b-samples-<job_id>.err'
```

You should see the simulator start, then the direct benchmark runner begin
evaluating the sampled tests.

## 6. Outputs

By default the job writes to:

`experiments/results/modify_codegen_qwen_samples_<job_id>`

That directory contains:

- `results/`: per-test JSON results
- `traces/`: reshaped JSON traces plus HTML trace pages
- `summary.json`: top-level run summary
- `metrics_summary.json`: aggregated evaluation metrics
- `results_<timestamp>.json`: detailed aggregated result dump
- `eval_report.html`: aggregate HTML report
- `experience_store.json`: stored infeasible-intent experience state

## 7. Run the same sampled benchmark manually inside an interactive allocation

If you want to debug without `sbatch`, open an interactive GPU shell and run
the same wrapper directly:

```bash
ssh fep
srun -A student -p dgxh100 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --pty /bin/bash
cd ~/behaviortree-planning-for-ami-agents
source ~/.venvs/qwen25-coder-sft/bin/activate
export MODIFY_CODEGEN_ADAPTER_PATH=outputs/sft/qwen25-coder-3b-instruct-lora-homebench-<run_id>/best_adapter
bash scripts/experiments/run_modify_codegen_samples_qwen.sh
```

This uses the same direct local backend as the SLURM job and does not require
any SSH port forwarding. You do not need `uv` installed on the compute node for
this path as long as the cluster venv is activated.
