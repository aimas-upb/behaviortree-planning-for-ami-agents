#!/usr/bin/env python3
"""
Run the sampled modify-codegen Qwen benchmark directly on a cluster node.

This runner starts the HomeBench simulator locally, loads the
fine-tuned Qwen model once, and evaluates the sampled neurosymbolic tests
sequentially in the same job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from docker.worker_experience_reuse import (
    create_config,
    resolve_modify_codegen_config_from_env,
    run_ns_test,
)
from scripts.common import PROJECT_ROOT, resolve_repo_path
from scripts.analysis.fix_modify_codegen_reporting import (
    fix_results_dir as fix_modify_codegen_results_dir,
)
from scripts.experiments.run_parallel_experience_homebench import (
    aggregate_results,
    generate_eval_report,
)
from src.experience.engine import ExperienceEngine
from src.experience.intent import IntentExtractor
from src.experience.neurosymbolic_runner import (
    NeuroSymbolicRunner,
    QwenModifyCodegenNeuroSymbolicRunner,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DATA = (
    "data/homebench/benchmarks/modify_codegen/home_disjoint/"
    "samples_for_inference.json"
)
DEFAULT_HOME_CONFIG = (
    "data/homebench/benchmarks/modify_codegen/home_disjoint/"
    "samples_for_inference_homes.json"
)
DEFAULT_SIMULATOR_DATA_DIR = "data/homebench/hmas/home_description"


def _normalize_home_id(raw_home_id, test_id: str) -> str:
    if raw_home_id is not None:
        return str(raw_home_id).replace("home", "")

    parts = str(test_id).split("_")
    if parts:
        return parts[0].replace("home", "")
    raise ValueError(f"Could not derive home_id from test id: {test_id}")


def _load_tests(data_path: Path, limit: int | None = None) -> list[dict]:
    with open(data_path) as f:
        tests = json.load(f)
    if limit is not None:
        tests = tests[:limit]
    return tests


def _wait_for_simulator(
    simulator_url: str,
    process: subprocess.Popen,
    timeout_seconds: float = 60.0,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "simulator not ready yet"
    health_url = f"{simulator_url}/health"

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Simulator exited before becoming ready "
                f"(exit code {process.returncode})"
            )

        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)

        time.sleep(1.0)

    raise TimeoutError(
        f"Simulator did not become ready within {timeout_seconds:.0f}s: "
        f"{last_error}"
    )


def _start_simulator(
    simulator_data_dir: Path,
    home_config: Path | None,
    port: int,
) -> tuple[subprocess.Popen, str]:
    simulator_host = "127.0.0.1"
    simulator_url = f"http://{simulator_host}:{port}"
    command = [
        sys.executable,
        "-m",
        "homebench.smart_home_simulator",
        "--data-dir",
        str(simulator_data_dir),
        "--host",
        simulator_host,
        "--port",
        str(port),
    ]
    if home_config is not None:
        command.extend(["--home-config", str(home_config)])

    logger.info("Starting simulator: %s", " ".join(command))
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        _wait_for_simulator(simulator_url, process)
    except Exception:
        _stop_process(process)
        raise
    return process, simulator_url


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_modify_codegen_samples_qwen_cluster(
    data_path: str = DEFAULT_DATA,
    home_config: str | None = DEFAULT_HOME_CONFIG,
    model: str = "gpt-4o",
    reasoning_effort: str | None = None,
    ontology: str = "ontologies/homeont.ttl",
    output_dir: str | None = None,
    limit: int | None = None,
    clear_experience: bool = True,
    simulator_data_dir: str = DEFAULT_SIMULATOR_DATA_DIR,
    simulator_port: int = 8080,
    generate_traces: bool = True,
    generate_report: bool = True,
    recompute_metrics: bool = True,
    progress: bool = True,
) -> Path:
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key or not openai_base_url:
        raise RuntimeError(
            "OPENAI_API_KEY and OPENAI_BASE_URL must be set in the environment"
        )

    resolved_data_path = resolve_repo_path(data_path)
    resolved_home_config = (
        resolve_repo_path(home_config) if home_config else None
    )
    resolved_ontology_path = resolve_repo_path(ontology)
    resolved_simulator_data_dir = resolve_repo_path(simulator_data_dir)

    tests = _load_tests(resolved_data_path, limit=limit)
    if not tests:
        raise RuntimeError(f"No tests found in {resolved_data_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_dir = resolve_repo_path(output_dir)
    else:
        base_dir = resolve_repo_path(
            f"experiments/results/modify_codegen_qwen_samples_{timestamp}"
        )
    base_dir.mkdir(parents=True, exist_ok=True)

    results_dir = base_dir / "results"
    traces_dir = base_dir / "traces"
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    experience_store_path = base_dir / "experience_store.json"
    if clear_experience and experience_store_path.exists():
        experience_store_path.unlink()

    client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
    ontology_text = resolved_ontology_path.read_text()
    engine = ExperienceEngine(persistence_path=str(experience_store_path))
    intent_extractor = IntentExtractor(ontology_text=ontology_text)
    modify_codegen_config = resolve_modify_codegen_config_from_env()

    ns_exp_config = create_config(
        name="neurosymbolic_modify_codegen_qwen_samples",
        output_format="python_code",
        prompt_strategy="detailed_structured_modify_only",
        model=model,
        reasoning_effort=reasoning_effort,
        experience_store=str(experience_store_path),
        experience_enabled=True,
        modify_codegen_backend=modify_codegen_config.backend,
        modify_codegen_base_url=modify_codegen_config.base_url,
        modify_codegen_base_model_name_or_path=(
            modify_codegen_config.base_model_name_or_path
        ),
        modify_codegen_adapter_path=modify_codegen_config.adapter_path,
        modify_codegen_device_map=modify_codegen_config.device_map,
        modify_codegen_torch_dtype=modify_codegen_config.torch_dtype,
        modify_codegen_local_files_only=(
            modify_codegen_config.local_files_only
        ),
        modify_codegen_prompt_style=modify_codegen_config.prompt_style,
        modify_codegen_timeout_seconds=modify_codegen_config.timeout_seconds,
        modify_codegen_max_new_tokens=modify_codegen_config.max_new_tokens,
        modify_codegen_temperature=modify_codegen_config.temperature,
        modify_codegen_top_p=modify_codegen_config.top_p,
    )

    runner_cls = (
        QwenModifyCodegenNeuroSymbolicRunner
        if ns_exp_config.modify_codegen.backend
        in {"fep_qwen_http", "fep_qwen_local"}
        else NeuroSymbolicRunner
    )
    ns_runner = runner_cls(
        config=ns_exp_config,
        client=client,
        engine=engine,
        intent_extractor=intent_extractor,
    )

    config_metadata = {
        "config_name": "neurosymbolic",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "needs_experience": True,
        "structured_goal": False,
        "neurosymbolic": True,
        "modify_codegen_backend": modify_codegen_config.backend,
        "modify_codegen_base_model_name_or_path": (
            modify_codegen_config.base_model_name_or_path
        ),
        "modify_codegen_adapter_path": modify_codegen_config.adapter_path,
        "modify_codegen_prompt_style": modify_codegen_config.prompt_style,
        "modify_codegen_max_new_tokens": (modify_codegen_config.max_new_tokens),
        "modify_codegen_torch_dtype": modify_codegen_config.torch_dtype,
        "modify_codegen_device_map": modify_codegen_config.device_map,
        "modify_codegen_local_files_only": (
            modify_codegen_config.local_files_only
        ),
    }

    print("\n" + "=" * 60)
    print("MODIFY-CODEGEN QWEN SAMPLE RUNNER")
    print("=" * 60)
    print(f"Data: {resolved_data_path}")
    print(f"Home config: {resolved_home_config or 'all homes'}")
    print(f"Simulator data: {resolved_simulator_data_dir}")
    print(f"Tests: {len(tests)}")
    print(f"Model: {model}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Modify backend: {modify_codegen_config.backend}")
    print(f"Adapter: {modify_codegen_config.adapter_path or '(none)'}")
    print(f"Output: {base_dir}")
    print("=" * 60 + "\n")

    simulator_process: subprocess.Popen | None = None
    simulator_url = ""
    start_time = time.time()

    try:
        simulator_process, simulator_url = _start_simulator(
            simulator_data_dir=resolved_simulator_data_dir,
            home_config=resolved_home_config,
            port=simulator_port,
        )

        iterator = tqdm(tests, desc="Evaluating") if progress else tests
        for item in iterator:
            work_item = dict(item)
            work_item["test_id"] = work_item["id"]
            work_item["home_id"] = _normalize_home_id(
                work_item.get("home_id"), work_item["id"]
            )
            work_item["config"] = config_metadata

            run_ns_test(
                work_item=work_item,
                simulator_url=simulator_url,
                results_dir=results_dir,
                client=client,
                worker_id=0,
                ns_runner=ns_runner,
                engine=engine,
                rate_limiter=None,
            )
    finally:
        _stop_process(simulator_process)

    elapsed = time.time() - start_time

    print("\nAggregating results...")
    metrics, experience_metrics = aggregate_results(
        results_dir=results_dir,
        output_dir=base_dir,
        config_name="neurosymbolic",
        generate_traces=generate_traces,
        experiment_config=config_metadata,
    )

    corrected_data_file: Path | None = None
    if recompute_metrics:
        print("\nRecomputing modify-codegen metrics...")
        corrected_data_file = fix_modify_codegen_results_dir(
            base_dir=base_dir,
            sample_file=resolved_data_path,
            dry_run=False,
            regenerate_report=False,
        )
        metrics_files = sorted(base_dir.glob("metrics_*.json"))
        if metrics_files:
            with open(metrics_files[-1]) as f:
                metrics_doc = json.load(f)
            metrics = metrics_doc.get("metrics", metrics)
            experience_metrics = metrics_doc.get(
                "experience_metrics", experience_metrics
            )

    if generate_report and metrics:
        generate_eval_report(
            base_dir,
            test_data=corrected_data_file or resolved_data_path,
        )

    summary_corrected_data_file = (
        str(corrected_data_file) if corrected_data_file else None
    )

    summary = {
        "timestamp": timestamp,
        "data_file": str(resolved_data_path),
        "home_config": (
            str(resolved_home_config) if resolved_home_config else None
        ),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "modify_backend": modify_codegen_config.backend,
        "adapter_path": modify_codegen_config.adapter_path,
        "base_model_name_or_path": (
            modify_codegen_config.base_model_name_or_path
        ),
        "total_tests": len(tests),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "experience_metrics": experience_metrics,
    }
    if summary_corrected_data_file:
        summary["corrected_data_file"] = summary_corrected_data_file
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {base_dir}")
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sampled Qwen modify-codegen benchmark on-cluster"
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--home-config", default=DEFAULT_HOME_CONFIG)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium"],
        default=None,
    )
    parser.add_argument("--ontology", default="ontologies/homeont.ttl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-clear-experience",
        action="store_true",
        help="Reuse any existing experience store in the output directory",
    )
    parser.add_argument(
        "--simulator-data-dir",
        default=DEFAULT_SIMULATOR_DATA_DIR,
    )
    parser.add_argument("--simulator-port", type=int, default=8080)
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--no-metric-fix",
        action="store_true",
        help="Skip modify-codegen post-processing that recomputes metrics for mislabeled error_input cases",
    )
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()

    run_modify_codegen_samples_qwen_cluster(
        data_path=args.data,
        home_config=args.home_config,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        ontology=args.ontology,
        output_dir=args.output,
        limit=args.limit,
        clear_experience=not args.no_clear_experience,
        simulator_data_dir=args.simulator_data_dir,
        simulator_port=args.simulator_port,
        generate_traces=not args.no_traces,
        generate_report=not args.no_report,
        recompute_metrics=not args.no_metric_fix,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
