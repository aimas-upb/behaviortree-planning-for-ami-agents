#!/usr/bin/env python3
"""
Parallel HomeBench Test Runner using Docker Worker Pool.

Runs HomeBench test files (single_feasible_action_tests.json, etc.) in parallel
using Docker containers, each with its own simulator instance.

Generates eval_report.html and traces compatible with run_homebench.py output.

Usage:
    python run_parallel_homebench.py --data experiments_data/single_feasible_action_tests.json
    python run_parallel_homebench.py --data experiments_data/*.json --workers 8
"""

import json
import os
import sys
import time
import shutil
import argparse
import subprocess
import glob as globlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


# Default configuration
DEFAULT_WORKERS = 4
DEFAULT_MODEL = "gpt-4o"
DEFAULT_IMAGE_NAME = "bt-planning-worker"

# Default rate limits (Tier 1 OpenAI limits)
DEFAULT_TPM_LIMIT = 500000
DEFAULT_RPM_LIMIT = 500

# Available models
AVAILABLE_MODELS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "description": "Fastest, cheapest"},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "description": "Balance of cost and capability"},
    "gpt-4o-mini": {"input": 0.60, "output": 2.40, "description": "Legacy mini model"},
    "gpt-4o": {"input": 5.00, "output": 20.00, "description": "Full GPT-4o capabilities"},
    "gpt-5": {"input": 1.25, "output": 10.00, "description": "Latest flagship model"},
}


def build_docker_image(image_name: str, force_rebuild: bool = False) -> bool:
    """Build the Docker image for workers."""
    print("\n" + "=" * 60)
    print("Building Docker image...")
    print("=" * 60)

    if not force_rebuild:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"Image '{image_name}' already exists. Use --rebuild to force rebuild.")
            return True

    try:
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)
        print(f"Successfully built image: {image_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build image: {e}")
        return False


def load_test_cases(data_paths: list[str]) -> list[dict]:
    """Load test cases from one or more JSON files."""
    all_tests = []

    for data_path in data_paths:
        # Handle glob patterns
        files = globlib.glob(data_path)
        if not files:
            print(f"Warning: No files matching {data_path}")
            continue

        for file_path in files:
            print(f"Loading tests from {file_path}...")
            with open(file_path) as f:
                data = json.load(f)

            # Add source file info to each test
            for item in data:
                item['source_file'] = os.path.basename(file_path)
                # Extract home_id from test id (e.g., "home96_one_123" -> 96)
                parts = item['id'].split('_')
                item['home_id'] = parts[0].replace('home', '')
                all_tests.append(item)

    return all_tests


def create_work_queue(
    tests: list[dict],
    work_dir: Path,
    config: dict,
) -> int:
    """Create the work queue file with all test cases."""
    queue = []

    for test in tests:
        queue.append({
            'id': test['id'],
            'test_id': test['id'],
            'home_id': test['home_id'],
            'input': test['input'],
            'output': test['output'],
            'source_file': test.get('source_file', ''),
            'config': config,
            'status': 'pending',
            'worker_id': None,
            'started_at': None,
            'completed_at': None,
        })

    queue_file = work_dir / "queue.json"
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)

    # Also save the config for reference
    config_file = work_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created work queue with {len(queue)} test cases")
    return len(queue)


def start_worker(
    worker_id: int,
    image_name: str,
    work_dir: Path,
    results_dir: Path,
    openai_api_key: str,
    model: str,
    rpm_limit: int,
    tpm_limit: int,
    mode: str = "homebench",
) -> subprocess.Popen:
    """Start a worker container."""
    container_name = f"bt-worker-{worker_id}"

    # Remove existing container if any
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )

    cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "-e", f"WORKER_ID={worker_id}",
        "-e", f"OPENAI_API_KEY={openai_api_key}",
        "-e", f"MODEL={model}",
        "-e", f"RPM_LIMIT={rpm_limit}",
        "-e", f"TPM_LIMIT={tpm_limit}",
        "-e", f"WORKER_MODE={mode}",
        "-v", f"{work_dir.absolute()}:/work",
        "-v", f"{results_dir.absolute()}:/results",
        "-e", "WORK_DIR=/work",
        "-e", "RESULTS_DIR=/results",
        image_name,
    ]

    print(f"Starting worker {worker_id}...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process


def monitor_progress(work_dir: Path, total: int) -> dict:
    """Monitor the progress of tests."""
    queue_file = work_dir / "queue.json"

    if not queue_file.exists():
        return {'pending': total, 'running': 0, 'completed': 0, 'failed': 0}

    with open(queue_file, 'r') as f:
        queue = json.load(f)

    stats = {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}
    for item in queue:
        status = item.get('status', 'pending')
        if status in stats:
            stats[status] += 1
        else:
            stats['pending'] += 1

    return stats


def stream_worker_logs(process: subprocess.Popen, worker_id: int):
    """Stream logs from a worker process."""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    process.wait()


def aggregate_results(results_dir: Path, output_dir: Path) -> dict:
    """Aggregate individual test results into metrics and generate report."""
    from dataclasses import dataclass, field
    from typing import Literal

    results = []
    for result_file in results_dir.glob("*.json"):
        if result_file.name in ('queue.json', 'experiment_config.json', 'rate_limit.json'):
            continue
        with open(result_file) as f:
            results.append(json.load(f))

    if not results:
        print("No results found to aggregate")
        return {}

    # Calculate metrics (same structure as run_homebench.py)
    metrics = {
        'total_tests': len(results),
        'successful_tests': 0,
        'quantifiable_tests': 0,
        'failed_tests': 0,
        'plans_generated': 0,
        'total_expected_actions': 0,
        'total_matched_actions': 0,
        'total_missing_actions': 0,
        'total_extra_actions': 0,
        'total_properties_checked': 0,
        'total_properties_matched': 0,
        'total_expected_impossible': 0,
        'total_detected_impossible': 0,
        'total_duration': 0.0,
        'failures_by_type': {
            'parse_error': 0,
            'compilation_error': 0,
            'execution_error': 0,
            'action_mismatch': 0,
            'property_mismatch': 0,
            'error_input_not_detected': 0,
            'other': 0,
        }
    }

    for r in results:
        success = r.get('success', 'False')
        if success == 'True':
            metrics['successful_tests'] += 1
        elif success == 'Quantifiable':
            metrics['quantifiable_tests'] += 1
        else:
            metrics['failed_tests'] += 1

        if r.get('plan_generated'):
            metrics['plans_generated'] += 1

        metrics['total_expected_actions'] += len(r.get('expected_actions', []))
        metrics['total_matched_actions'] += len(r.get('matched_actions', []))
        metrics['total_missing_actions'] += len(r.get('missing_actions', []))
        metrics['total_extra_actions'] += len(r.get('extra_actions', []))
        metrics['total_properties_checked'] += r.get('properties_checked', 0)
        metrics['total_properties_matched'] += r.get('properties_matched', 0)
        metrics['total_expected_impossible'] += r.get('expected_impossible', 0)
        metrics['total_detected_impossible'] += len(r.get('detected_impossible', []))
        metrics['total_duration'] += r.get('duration', 0.0)

        failure_type = r.get('failure_type')
        if failure_type and failure_type in metrics['failures_by_type']:
            metrics['failures_by_type'][failure_type] += 1

    # Calculate rates
    total = metrics['total_tests']
    metrics['success_rate'] = metrics['successful_tests'] / total if total > 0 else 0.0
    metrics['quantifiable_rate'] = metrics['quantifiable_tests'] / total if total > 0 else 0.0
    metrics['success_or_quantifiable_rate'] = (
        (metrics['successful_tests'] + metrics['quantifiable_tests']) / total if total > 0 else 0.0
    )

    matched = metrics['total_matched_actions']
    extra = metrics['total_extra_actions']
    expected = metrics['total_expected_actions']
    metrics['action_precision'] = matched / (matched + extra) if (matched + extra) > 0 else 0.0
    metrics['action_recall'] = matched / expected if expected > 0 else 0.0
    p, r = metrics['action_precision'], metrics['action_recall']
    metrics['action_f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    props_checked = metrics['total_properties_checked']
    props_matched = metrics['total_properties_matched']
    metrics['property_accuracy'] = props_matched / props_checked if props_checked > 0 else 0.0

    exp_impossible = metrics['total_expected_impossible']
    det_impossible = metrics['total_detected_impossible']
    metrics['impossible_detection_rate'] = min(1.0, det_impossible / exp_impossible) if exp_impossible > 0 else 1.0

    metrics['avg_duration'] = metrics['total_duration'] / total if total > 0 else 0.0

    # Save aggregated metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump({'metrics': metrics}, f, indent=2)

    # Save detailed results file (compatible with eval_viewer)
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Copy individual traces to traces directory
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    for r in results:
        test_id = r.get('test_id', r.get('id', 'unknown'))
        trace_data = r.get('trace', r.get('raw_result', {}))
        if trace_data:
            trace_file = traces_dir / f"{test_id}.json"
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)

    return metrics


def generate_eval_report(output_dir: Path):
    """Generate eval_report.html using eval_viewer.py."""
    print("\nGenerating evaluation report...")
    # Get project root (where eval_viewer.py is located)
    project_root = Path(__file__).parent
    eval_viewer = project_root / "eval_viewer.py"

    if not eval_viewer.exists():
        print(f"Warning: eval_viewer.py not found at {eval_viewer}")
        return False

    try:
        subprocess.run(
            ["python", str(eval_viewer), str(output_dir.absolute())],
            check=True,
            cwd=project_root,
        )
        report_file = output_dir / "eval_report.html"
        if report_file.exists():
            print(f"Report generated: {report_file}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate eval report: {e}")
    except FileNotFoundError:
        print("Warning: eval_viewer.py not found")
    return False


def run_parallel_homebench(
    data_paths: list[str],
    num_workers: int = DEFAULT_WORKERS,
    model: str = DEFAULT_MODEL,
    discovery_affordances: str = "agentic",
    discovery_state: str = "relevant",
    planning_reasoning: str = "none",
    planning_output: str = "json_ir",
    prompt_strategy: str = "detailed",
    execution_mode: str = "behavior_tree",
    output_dir: str | None = None,
    image_name: str = DEFAULT_IMAGE_NAME,
    rebuild_image: bool = False,
    rpm_limit: int = DEFAULT_RPM_LIMIT,
    tpm_limit: int = DEFAULT_TPM_LIMIT,
    limit: int | None = None,
    generate_traces: bool = True,
    generate_report: bool = True,
):
    """Run HomeBench tests in parallel using Docker workers."""

    # Check for API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Build config dict to pass to workers
    config = {
        'model': model,
        'discovery_affordances': discovery_affordances,
        'discovery_state': discovery_state,
        'planning_reasoning': planning_reasoning,
        'planning_output': planning_output,
        'prompt_strategy': prompt_strategy,
        'execution_mode': execution_mode,
        'generate_traces': generate_traces,
    }

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_dir = Path(output_dir)
    else:
        # Generate name based on config
        config_name = f"{discovery_affordances}_{discovery_state}_{planning_output}"
        base_dir = Path(f"experiments/tests/parallel_{config_name}_{timestamp}")

    work_dir = base_dir / "work"
    results_dir = base_dir / "results"

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PARALLEL HOMEBENCH TEST RUNNER")
    print("=" * 60)
    print(f"Data files: {data_paths}")
    print(f"Workers: {num_workers}")
    print(f"Model: {model}")
    print(f"Discovery - Affordances: {discovery_affordances}")
    print(f"Discovery - State: {discovery_state}")
    print(f"Planning - Reasoning: {planning_reasoning}")
    print(f"Planning - Output: {planning_output}")
    print(f"Execution Mode: {execution_mode}")
    print(f"Rate limits: {rpm_limit} RPM, {tpm_limit:,} TPM")
    print(f"Output: {base_dir}")
    print("=" * 60)

    # Build Docker image
    if not build_docker_image(image_name, rebuild_image):
        print("Failed to build Docker image. Exiting.")
        sys.exit(1)

    # Load test cases
    print("\nLoading test cases...")
    tests = load_test_cases(data_paths)
    if limit:
        tests = tests[:limit]
    print(f"Loaded {len(tests)} test cases")

    if not tests:
        print("No test cases to run!")
        sys.exit(0)

    # Create work queue
    total_tests = create_work_queue(tests, work_dir, config)

    print(f"\nTotal tests to run: {total_tests}")
    print(f"Estimated time with {num_workers} workers: "
          f"~{total_tests * 30 / num_workers / 60:.1f} minutes")

    # Start workers
    print("\n" + "=" * 60)
    print("Starting workers...")
    print("=" * 60)

    workers = []
    for i in range(num_workers):
        process = start_worker(
            worker_id=i,
            image_name=image_name,
            work_dir=work_dir,
            results_dir=results_dir,
            openai_api_key=openai_api_key,
            model=model,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            mode="homebench",
        )
        workers.append((i, process))
        time.sleep(2)  # Stagger startup

    # Monitor progress and stream logs
    print("\n" + "=" * 60)
    print("Running tests...")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        log_futures = [
            executor.submit(stream_worker_logs, proc, wid)
            for wid, proc in workers
        ]

        start_time = time.time()
        last_completed = 0

        while True:
            all_done = all(proc.poll() is not None for _, proc in workers)

            stats = monitor_progress(work_dir, total_tests)
            completed = stats['completed'] + stats['failed']

            if completed != last_completed:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = total_tests - completed
                eta = remaining / rate if rate > 0 else 0

                print(f"\n[Progress] Completed: {completed}/{total_tests} "
                      f"({completed/total_tests*100:.1f}%) | "
                      f"Running: {stats['running']} | "
                      f"Failed: {stats['failed']} | "
                      f"Rate: {rate*60:.1f}/min | "
                      f"ETA: {eta/60:.1f} min")
                last_completed = completed

            if all_done:
                break

            time.sleep(5)

        for future in log_futures:
            future.result()

    # Final stats
    elapsed = time.time() - start_time
    stats = monitor_progress(work_dir, total_tests)

    print("\n" + "=" * 60)
    print("TEST RUN COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")

    # Aggregate results
    print("\nAggregating results...")
    metrics = aggregate_results(results_dir, base_dir)

    if metrics:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total tests: {metrics['total_tests']}")
        print(f"Successful: {metrics['successful_tests']} ({metrics['success_rate']:.1%})")
        print(f"Quantifiable: {metrics['quantifiable_tests']} ({metrics['quantifiable_rate']:.1%})")
        print(f"Failed: {metrics['failed_tests']}")
        print()
        print("Action Matching:")
        print(f"  Precision: {metrics['action_precision']:.1%}")
        print(f"  Recall: {metrics['action_recall']:.1%}")
        print(f"  F1: {metrics['action_f1']:.1%}")
        print()
        print(f"Property Accuracy: {metrics['property_accuracy']:.1%}")
        print(f"Avg duration: {metrics['avg_duration']:.1f}s")
        print("=" * 60)

    # Generate eval report
    if generate_report:
        generate_eval_report(base_dir)

    # Save summary
    summary = {
        'timestamp': timestamp,
        'data_files': data_paths,
        'num_workers': num_workers,
        'config': config,
        'total_tests': total_tests,
        'completed': stats['completed'],
        'failed': stats['failed'],
        'elapsed_seconds': elapsed,
        'metrics': metrics,
    }
    with open(base_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {base_dir}")
    return base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run HomeBench tests in parallel using Docker workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single test file
  python run_parallel_homebench.py --data experiments_data/single_feasible_action_tests.json

  # Run all test files
  python run_parallel_homebench.py --data "experiments_data/*_tests.json"

  # Run with specific config
  python run_parallel_homebench.py --data experiments_data/*.json \\
      --discovery-affordances agentic --discovery-state relevant \\
      --workers 8 --model gpt-4o
"""
    )

    # Data
    parser.add_argument(
        "--data", "-d",
        nargs="+",
        required=True,
        help="Path(s) to test data JSON files (supports glob patterns)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of tests to run"
    )

    # Workers
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )

    # Discovery config
    parser.add_argument(
        "--discovery-affordances",
        choices=["exhaustive", "agentic", "relevant"],
        default="agentic",
        help="Affordance discovery strategy (default: agentic)"
    )
    parser.add_argument(
        "--discovery-state",
        choices=["all", "relevant", "agentic", "none"],
        default="relevant",
        help="State discovery strategy (default: relevant)"
    )

    # Planning config
    parser.add_argument(
        "--planning-reasoning",
        choices=["none", "chain_of_thought", "multi_turn", "reflection"],
        default="none",
        help="Reasoning strategy (default: none)"
    )
    parser.add_argument(
        "--planning-output",
        choices=["json_ir", "python_code", "python_code_unconstrained"],
        default="json_ir",
        help="Output format (default: json_ir)"
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=["baseline", "detailed", "few_shot", "icl"],
        default="detailed",
        help="Prompt strategy (default: detailed)"
    )

    # Execution mode
    parser.add_argument(
        "--execution-mode",
        choices=["behavior_tree", "direct_agent"],
        default="behavior_tree",
        help="Execution mode (default: behavior_tree)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--generate-traces",
        action="store_true",
        default=True,
        help="Generate individual trace files (default: True)"
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Disable trace generation"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate eval_report.html (default: True)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable report generation"
    )

    # Docker
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild Docker image"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_NAME,
        help=f"Docker image name (default: {DEFAULT_IMAGE_NAME})"
    )

    # Rate limiting
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=DEFAULT_RPM_LIMIT,
        help=f"Requests per minute limit (default: {DEFAULT_RPM_LIMIT})"
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=DEFAULT_TPM_LIMIT,
        help=f"Tokens per minute limit (default: {DEFAULT_TPM_LIMIT:,})"
    )

    # Info
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:\n")
        print(f"{'Model':<15} {'Input/1M':<12} {'Output/1M':<12} Description")
        print("-" * 70)
        for model, info in AVAILABLE_MODELS.items():
            print(f"{model:<15} ${info['input']:<11.2f} ${info['output']:<11.2f} {info['description']}")
        print(f"\nDefault: {DEFAULT_MODEL}")
        sys.exit(0)

    run_parallel_homebench(
        data_paths=args.data,
        num_workers=args.workers,
        model=args.model,
        discovery_affordances=args.discovery_affordances,
        discovery_state=args.discovery_state,
        planning_reasoning=args.planning_reasoning,
        planning_output=args.planning_output,
        prompt_strategy=args.prompt_strategy,
        execution_mode=args.execution_mode,
        output_dir=args.output,
        image_name=args.image,
        rebuild_image=args.rebuild,
        rpm_limit=args.rpm_limit,
        tpm_limit=args.tpm_limit,
        limit=args.limit,
        generate_traces=not args.no_traces,
        generate_report=not args.no_report,
    )


if __name__ == "__main__":
    main()
