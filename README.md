# Behavior Tree Planning for AMI Agents

This repository contains the current HMAS/HomeBench planning pipeline under `src/`, experiment runners under `scripts/experiments/`, viewers under `viewers/`, and older standalone agents under `legacy/standalone/`.

## Repository Layout

- `src/`: current discovery, planning, execution, and experience-reuse implementation.
- `scripts/experiments/`: canonical experiment runners.
- `scripts/analysis/`: analysis, repair, and dataset-prep utilities.
- `scripts/dev/`: ad hoc testing and local developer helpers.
- `viewers/`: terminal/HTML viewers for traces and evaluation reports.
- `legacy/standalone/`: archived pre-`src` standalone agents kept for reference.
- `data/`: centralized datasets, converted artifacts, curated benchmarks, and retry sets.
- `docs/`: architecture notes, guides, experiment docs, reports, and working notes.
- `docker/`: worker image, entrypoints, and Docker-specific worker code.

## Primary Entrypoints

Run these from the repository root.

- HomeBench evaluation:
  `uv run python -m scripts.experiments.run_homebench --help`
- Parallel HomeBench evaluation:
  `uv run python -m scripts.experiments.run_parallel_homebench --help`
- Experience-reuse evaluation:
  `uv run python -m scripts.experiments.run_experience_homebench --help`
- Parallel experience-reuse evaluation:
  `uv run python -m scripts.experiments.run_parallel_experience_homebench --help`
- Ablation runners:
  `uv run python -m scripts.experiments.run_ablation --help`
  `uv run python -m scripts.experiments.run_homebench_ablation --help`
- Trace viewer:
  `uv run python -m viewers.trace_viewer --help`
- Experience trace viewer:
  `uv run python -m viewers.experience_trace_viewer --help`
- Evaluation report viewer:
  `uv run python -m viewers.eval_viewer --help`
- Legacy standalone BT agent:
  `uv run python -m legacy.standalone.bt_agent --help`
- Legacy standalone direct agent:
  `uv run python -m legacy.standalone.agent`

## Analysis Utilities

- Test statistics:
  `uv run python -m scripts.analysis.compute_test_statistics --help`
- Episode selection:
  `uv run python -m scripts.analysis.select_test_episodes --help`
- HomeBench benchmark preprocessing:
  `uv run python -m scripts.analysis.preprocess_homebench --help`
- Output schema inspection:
  `uv run python -m scripts.analysis.extract_output_schema --help`
- Experience report repair:
  `uv run python -m scripts.analysis.fix_experience_reporting --help`

## Data Layout

All repo-managed datasets and benchmark inputs now live under `data/`.

- `data/homebench/raw/`: original HomeBench JSONL files.
- `data/homebench/converted/`: converted HomeBench JSON plus curated benchmark files.
- `data/homebench/hmas/home_description/`: canonical simulator-ready HomeBench TTL/state files.
- `data/homebench/benchmarks/action_groups/`: sampled action-group benchmark files and statistics.
- `data/homebench/benchmarks/repeated_query/`: repeated-query evaluation sets and derived stats.
- `data/homebench/benchmarks/experience_reuse/`: experience-reuse benchmark files and home subsets.
- `data/homebench/retries/action_groups/`: retry-specific benchmark slices.
- `data/blocksworld/raw/`: original Blocksworld PDDL instances.
- `data/blocksworld/hmas/`: simulator-ready Blocksworld HMAS exports.

## Docker

The worker image definition now lives at `docker/Dockerfile`.

- Manual build:
  `docker build -t bt-planning-worker -f docker/Dockerfile .`

The parallel runners in `scripts/experiments/` use this Dockerfile directly.

## Documentation

- Architecture:
  `docs/architecture/ARCHITECTURE.md`
  `docs/architecture/SYSTEM_DOCUMENTATION.md`
- Integration guide:
  `docs/guides/LLM_AGENT_INTEGRATION.md`
- Experiment notes and command catalogs:
  `docs/experiments/README_EXPERIMENTS.md`
  `docs/experiments/experiment_commands.txt`
  `docs/experiments/experience_reuse_experiment_commands.txt`

## Notes

- The canonical shared HMAS client is `src/hmas_client.py`.
- Use `data/` paths in new scripts and docs; the root-level `datasets/`, `experiments_data/`, and `retry_experiments_data/` layout is no longer canonical.
