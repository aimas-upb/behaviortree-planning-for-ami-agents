#!/usr/bin/env python3
"""
Parallel Experience-Reuse HomeBench Test Runner using Docker Worker Pool.

Runs experience-reuse experiments using Docker containers. Supports three configs:
- semantic_nav: agentic affordance strategy (baseline, no experience)
- semantic_query: agentic_query affordance strategy (baseline, no experience)
- experience_reuse: agentic_query + experience matching pipeline

For experience_reuse, each category (single_feasible, single_unfeasible,
multi_feasible, multi_mixed) runs in a single dedicated container to preserve
sequential experience accumulation.

For non-experience configs (semantic_nav, semantic_query), multiple workers
can process tests in parallel.

Usage:
    # Experience reuse (1 worker per category)
    python run_parallel_experience_homebench.py \\
        --data experiments_data/experience_reuse/single_feasible_tests.json \\
        --config-name experience_reuse --model gpt-4o

    # Semantic query baseline (parallel workers)
    python run_parallel_experience_homebench.py \\
        --data "experiments_data/experience_reuse/*_tests.json" \\
        --config-name semantic_query --workers 4 --model gpt-4o
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
DEFAULT_IMAGE_NAME = "bt-planning-worker-experience"

# Default rate limits
DEFAULT_TPM_LIMIT = 500000
DEFAULT_RPM_LIMIT = 500

# Available models
AVAILABLE_MODELS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "description": "Fastest, cheapest (reasoning)"},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "description": "Balance of cost and capability (reasoning)"},
    "gpt-4o-mini": {"input": 0.60, "output": 2.40, "description": "Legacy mini model"},
    "gpt-4o": {"input": 5.00, "output": 20.00, "description": "Full GPT-4o capabilities"},
}

# Ablation configs
ABLATION_CONFIGS = {
    "semantic_nav": {"needs_experience": False},
    "semantic_query": {"needs_experience": False},
    "experience_reuse": {"needs_experience": True},
    "neurosymbolic": {"needs_experience": True},
}

# Test categories (for experience_reuse, each gets 1 worker)
EXPERIENCE_CATEGORIES = [
    "single_feasible",
    "single_unfeasible",
    "multi_feasible",
    "multi_mixed",
]


def build_docker_image(image_name: str, force_rebuild: bool = False) -> bool:
    """Build the Docker image for experience workers."""
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

    # Build using the same Dockerfile but with experience entrypoint
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
        files = globlib.glob(data_path)
        if not files:
            print(f"Warning: No files matching {data_path}")
            continue

        for file_path in files:
            print(f"Loading tests from {file_path}...")
            with open(file_path) as f:
                data = json.load(f)

            for item in data:
                item['source_file'] = os.path.basename(file_path)
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

    config_file = work_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created work queue with {len(queue)} test cases")
    return len(queue)


def start_experience_worker(
    worker_id: int,
    image_name: str,
    work_dir: Path,
    results_dir: Path,
    openai_api_key: str,
    model: str,
    config_name: str,
    rpm_limit: int,
    tpm_limit: int,
    reasoning_effort: str | None = None,
    similarity_threshold: float = 0.85,
    clear_experience: bool = True,
    home_config: Path | None = None,
    source_file_filter: str | None = None,
    traces_dir: Path | None = None,
    structured_goal: bool = False,
    neurosymbolic: bool = False,
) -> subprocess.Popen:
    """Start an experience worker container."""
    container_name = f"bt-exp-worker-{worker_id}"

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )

    cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"WORKER_ID={worker_id}",
        "-e", f"OPENAI_API_KEY={openai_api_key}",
        "-e", f"MODEL={model}",
        "-e", f"CONFIG_NAME={config_name}",
        "-e", f"RPM_LIMIT={rpm_limit}",
        "-e", f"TPM_LIMIT={tpm_limit}",
        "-e", f"SIMILARITY_THRESHOLD={similarity_threshold}",
        "-e", f"CLEAR_EXPERIENCE={'true' if clear_experience else 'false'}",
        "-e", f"STRUCTURED_GOAL={'true' if structured_goal else 'false'}",
        "-e", f"NEUROSYMBOLIC={'true' if neurosymbolic else 'false'}",
        "-v", f"{work_dir.absolute()}:/work",
        "-v", f"{results_dir.absolute()}:/results",
        "-e", "WORK_DIR=/work",
        "-e", "RESULTS_DIR=/results",
        "-e", "TRACES_DIR=/traces",
    ]

    if traces_dir:
        cmd.extend(["-v", f"{traces_dir.absolute()}:/traces"])

    if home_config:
        cmd.extend([
            "-v", f"{home_config.absolute()}:/app/home_config.json:ro",
            "-e", "HOME_CONFIG=/app/home_config.json",
        ])

    if reasoning_effort:
        cmd.extend(["-e", f"REASONING_EFFORT={reasoning_effort}"])

    if source_file_filter:
        cmd.extend(["-e", f"SOURCE_FILE_FILTER={source_file_filter}"])

    # Use experience entrypoint
    cmd.extend([
        "--entrypoint", "/app/docker/entrypoint_experience.sh",
        image_name,
    ])

    print(f"Starting experience worker {worker_id} (config={config_name})...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process


def monitor_progress(work_dir: Path, total: int) -> dict:
    """Monitor the progress of tests."""
    import fcntl

    queue_file = work_dir / "queue.json"
    lock_file = work_dir / "queue.lock"

    if not queue_file.exists():
        return {'pending': total, 'running': 0, 'completed': 0, 'failed': 0}

    try:
        lock_file.touch(exist_ok=True)
        with open(lock_file, 'r') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, IOError, PermissionError, OSError) as e:
        print(f"Warning: Could not read queue file: {e}")
        return {'pending': total, 'running': 0, 'completed': 0, 'failed': 0}

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


def _reshape_trace_for_viewer(raw_result: dict, test_id: str, config: dict) -> dict:
    """
    Reshape an ExperienceRunResult trace dict into the format expected
    by the experience trace viewer.

    Includes both the standard trace_viewer keys (config, goal, discovery,
    planning, execution) and experience-specific keys (intents,
    match_results, matched_plan_traces, SPARQL queries in exploration_trace).
    """
    if not isinstance(raw_result, dict):
        return {"config_name": f"experience_{test_id}"}

    trace: dict = {
        "config_name": f"experience_{test_id}",
        "config": config,
        "success": raw_result.get("success", False),
        "duration_seconds": raw_result.get("duration_seconds", 0),
        "error": raw_result.get("error"),
        # Experience-specific data
        "intents": raw_result.get("intents"),
        "match_results": raw_result.get("match_results"),
        "matched_plan_traces": raw_result.get("matched_plan_traces"),
        "matched_plan_time_seconds": raw_result.get("matched_plan_time_seconds"),
        "unmatched_plan_time_seconds": raw_result.get("unmatched_plan_time_seconds"),
        "detected_impossible": raw_result.get("detected_impossible"),
    }

    # Extract discovery and planning from unmatched_plan_traces
    unmatched_traces = raw_result.get("unmatched_plan_traces", [])
    if unmatched_traces and isinstance(unmatched_traces[0], dict):
        first_trace = unmatched_traces[0]
        trace["discovery"] = first_trace.get("discovery", {})
        trace["planning"] = first_trace.get("planning", {})
    else:
        trace["discovery"] = {}
        trace["planning"] = {}

    # Execution
    trace["execution"] = raw_result.get("execution") or {}

    # Goal: reconstruct from intents
    intents = raw_result.get("intents", [])
    if intents:
        intent_texts = [
            i.get("text_intent", "") if isinstance(i, dict) else ""
            for i in intents
        ]
        trace["goal"] = ", ".join(t for t in intent_texts if t)

    return trace


def _reshape_ns_trace_for_viewer(raw_result: dict, test_id: str, config: dict) -> dict:
    """
    Reshape a NeuroSymbolicRunResult trace dict into the format expected
    by the experience trace viewer.

    Passes through the NS-specific keys (resolution_results, set_count,
    modify_count, impossible_count, etc.) directly alongside the standard
    viewer keys.
    """
    if not isinstance(raw_result, dict):
        return {"config_name": f"neurosymbolic_{test_id}"}

    trace: dict = {
        "config_name": f"neurosymbolic_{test_id}",
        "config": config,
        "success": raw_result.get("success", False),
        "duration_seconds": raw_result.get("duration_seconds", 0),
        "error": raw_result.get("error"),
        # Intent extraction (shared with experience viewer)
        "intents": raw_result.get("intents"),
        "intent_extraction_trace": raw_result.get("intent_extraction_trace"),
        # NS-specific routing data
        "resolution_results": raw_result.get("resolution_results"),
        "set_count": raw_result.get("set_count", 0),
        "modify_count": raw_result.get("modify_count", 0),
        "impossible_count": raw_result.get("impossible_count", 0),
        "impossible_details": raw_result.get("impossible_details"),
        # Plan IRs for reference
        "set_actions_tree_ir": raw_result.get("set_actions_tree_ir"),
        "modify_actions_tree_ir": raw_result.get("modify_actions_tree_ir"),
        "combined_plan_ir": raw_result.get("combined_plan_ir"),
        # Modify-planning trace (contains discovery + planning sub-sections)
        "modify_plan_trace": raw_result.get("modify_plan_trace"),
        "modify_plan_time_seconds": raw_result.get("modify_plan_time_seconds", 0.0),
        # Detected impossible
        "detected_impossible": raw_result.get("detected_impossible"),
        # Execution
        "execution": raw_result.get("execution") or {},
    }

    # Expose modify planning sub-sections at top level for standard viewer cards
    modify_trace = raw_result.get("modify_plan_trace") or {}
    planning_sub = modify_trace.get("planning") or {}
    trace["planning"] = planning_sub
    # Lift the discovery context built for modify-branch planning (if any) so
    # the trace viewer can render "Planning Context (What the Model Sees)".
    trace["discovery"] = modify_trace.get("discovery") or {}

    # Reconstruct goal from intents
    intents = raw_result.get("intents", [])
    if intents:
        intent_texts = [
            i.get("text_intent", "") if isinstance(i, dict) else ""
            for i in intents
        ]
        trace["goal"] = ", ".join(t for t in intent_texts if t)

    return trace


def _reshape_standard_trace_for_viewer(raw_result: dict, test_id: str, config: dict) -> dict:
    """
    Reshape a standard (non-experience) trace dict into the format expected
    by the trace viewer.
    """
    if not isinstance(raw_result, dict):
        return {"config_name": f"standard_{test_id}"}

    trace = raw_result.copy()
    trace["config_name"] = f"standard_{test_id}"
    # Only set config if the raw result doesn't already have the structured
    # ExperimentConfig format (with nested model.name, discovery.affordances, etc.)
    if "config" not in trace or not isinstance(trace["config"], dict) or "model" not in trace["config"]:
        trace["config"] = config
    return trace


def aggregate_results(
    results_dir: Path,
    output_dir: Path,
    config_name: str,
    generate_traces: bool = True,
    experiment_config: dict | None = None,
) -> dict:
    """Aggregate individual test results into metrics, including experience metrics."""
    results = []
    for result_file in results_dir.glob("*.json"):
        if result_file.name in ('queue.json', 'experiment_config.json', 'rate_limit.json'):
            continue
        if result_file.name.startswith('experience_store_') or result_file.name.startswith('experience_stats_'):
            continue
        with open(result_file) as f:
            results.append(json.load(f))

    if not results:
        print("No results found to aggregate")
        return {}

    # Standard metrics
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
        },
    }

    # Experience metrics
    experience_metrics = {
        'total_intents_extracted': 0,
        'total_matched': 0,
        'total_unmatched': 0,
        'total_matched_infeasible': 0,
        'match_count_per_test': [],
        'avg_similarity_per_test': [],
        'matched_planning_times': [],
        'unmatched_planning_times': [],
        'end_to_end_times': [],
        'total_new_experiences_stored': 0,
        'experience_size_over_time': [],
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

        # Experience metrics
        exp_matched = r.get('experience_matched', 0)
        exp_unmatched = r.get('experience_unmatched', 0)
        exp_intents = r.get('experience_intents_extracted', 0)
        exp_infeasible = r.get('experience_matched_infeasible', 0)

        experience_metrics['total_intents_extracted'] += exp_intents
        experience_metrics['total_matched'] += exp_matched
        experience_metrics['total_unmatched'] += exp_unmatched
        experience_metrics['total_matched_infeasible'] += exp_infeasible
        experience_metrics['match_count_per_test'].append(exp_matched)
        experience_metrics['avg_similarity_per_test'].append(
            r.get('experience_avg_similarity', 0.0)
        )
        experience_metrics['matched_planning_times'].append(
            r.get('experience_matched_plan_time', 0.0)
        )
        experience_metrics['unmatched_planning_times'].append(
            r.get('experience_unmatched_plan_time', 0.0)
        )
        experience_metrics['end_to_end_times'].append(r.get('duration', 0.0))
        experience_metrics['total_new_experiences_stored'] += r.get('experience_new_stored', 0)
        experience_metrics['experience_size_over_time'].append(
            r.get('experience_store_size', 0)
        )

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
    p, r_val = metrics['action_precision'], metrics['action_recall']
    metrics['action_f1'] = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0

    props_checked = metrics['total_properties_checked']
    props_matched = metrics['total_properties_matched']
    metrics['property_accuracy'] = props_matched / props_checked if props_checked > 0 else 0.0

    exp_impossible = metrics['total_expected_impossible']
    det_impossible = metrics['total_detected_impossible']
    metrics['impossible_detection_rate'] = min(1.0, det_impossible / exp_impossible) if exp_impossible > 0 else 1.0
    metrics['avg_duration'] = metrics['total_duration'] / total if total > 0 else 0.0

    # Experience computed metrics
    matched_times = experience_metrics['matched_planning_times']
    unmatched_times = experience_metrics['unmatched_planning_times']
    match_counts = experience_metrics['match_count_per_test']

    experience_metrics['avg_match_count'] = (
        sum(match_counts) / len(match_counts) if match_counts else 0.0
    )
    experience_metrics['avg_matched_planning_time'] = (
        sum(matched_times) / len(matched_times) if matched_times else 0.0
    )
    experience_metrics['avg_unmatched_planning_time'] = (
        sum(unmatched_times) / len(unmatched_times) if unmatched_times else 0.0
    )

    # Save aggregated metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'experience_metrics': experience_metrics,
        }, f, indent=2)

    # Save detailed results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save individual traces (reshaped for trace viewer compatibility)
    is_neurosymbolic = config_name == "neurosymbolic"
    is_experience = ABLATION_CONFIGS.get(config_name, {}).get('needs_experience', False)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    for r in results:
        test_id = r.get('test_id', r.get('id', 'unknown'))
        trace_data = r.get('trace', r.get('raw_result', {}))
        if trace_data:
            # Reshape trace for the appropriate viewer
            if is_neurosymbolic:
                reshaped = _reshape_ns_trace_for_viewer(
                    trace_data, test_id, experiment_config or {}
                )
            elif is_experience:
                reshaped = _reshape_trace_for_viewer(
                    trace_data, test_id, experiment_config or {}
                )
            else:
                reshaped = _reshape_standard_trace_for_viewer(
                    trace_data, test_id, experiment_config or {}
                )

            trace_file = traces_dir / f"{test_id}.json"
            with open(trace_file, 'w') as f:
                json.dump(reshaped, f, indent=2, default=str)

    # Generate HTML trace files for each test
    if generate_traces:
        print("\nGenerating HTML traces...")
        generate_html_traces(output_dir, config_name)

    # Collect experience stats files
    experience_stats = []
    for stats_file in results_dir.glob("experience_stats_*.json"):
        with open(stats_file) as f:
            experience_stats.append(json.load(f))
    if experience_stats:
        stats_output = output_dir / "experience_engine_stats.json"
        with open(stats_output, 'w') as f:
            json.dump(experience_stats, f, indent=2)

    # Copy experience stores
    for store_file in results_dir.glob("experience_store_*.json"):
        shutil.copy2(store_file, output_dir / store_file.name)

    return metrics, experience_metrics


def generate_html_traces(output_dir: Path, config_name: str) -> int:
    """
    (Re)generate HTML trace files from already-reshaped JSON traces in
    output_dir/traces/.  Works on any existing experiment directory.
    Returns the number of HTML files written.
    """
    # neurosymbolic uses the experience trace viewer (intent/routing sections)
    is_experience = ABLATION_CONFIGS.get(config_name, {}).get('needs_experience', False)
    traces_dir = output_dir / "traces"

    if not traces_dir.exists():
        print(f"Warning: traces directory not found at {traces_dir}")
        return 0

    trace_files = list(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"Warning: no JSON trace files found in {traces_dir}")
        return 0

    html_count = 0
    if is_experience:
        try:
            from experience_trace_viewer import export_experience_html
            for trace_file in trace_files:
                try:
                    with open(trace_file) as f:
                        trace = json.load(f)
                    export_experience_html(trace, str(trace_file.with_suffix(".html")))
                    html_count += 1
                except Exception as e:
                    print(f"  Warning: failed to generate HTML for {trace_file.name}: {e}")
        except ImportError:
            print("Warning: experience_trace_viewer not found, skipping HTML traces")
    else:
        try:
            from trace_viewer import export_html
            for trace_file in trace_files:
                try:
                    with open(trace_file) as f:
                        trace = json.load(f)
                    export_html(trace, str(trace_file.with_suffix(".html")))
                    html_count += 1
                except Exception as e:
                    print(f"  Warning: failed to generate HTML for {trace_file.name}: {e}")
        except ImportError:
            print("Warning: trace_viewer not found, skipping HTML traces")

    print(f"Generated {html_count} HTML traces in {traces_dir}/")
    return html_count


def generate_eval_report(output_dir: Path):
    """Generate eval_report.html using eval_viewer.py."""
    print("\nGenerating evaluation report...")
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


def run_parallel_experience_homebench(
    data_paths: list[str],
    config_name: str = "experience_reuse",
    num_workers: int = DEFAULT_WORKERS,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = None,
    output_dir: str | None = None,
    image_name: str = DEFAULT_IMAGE_NAME,
    rebuild_image: bool = False,
    rpm_limit: int = DEFAULT_RPM_LIMIT,
    tpm_limit: int = DEFAULT_TPM_LIMIT,
    limit: int | None = None,
    similarity_threshold: float = 0.85,
    clear_experience: bool = True,
    generate_traces: bool = True,
    generate_report: bool = True,
    home_config: str | None = None,
    structured_goal: bool = False,
    neurosymbolic: bool = False,
):
    """Run experience-reuse HomeBench tests using Docker workers."""

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    config_info = ABLATION_CONFIGS.get(config_name)
    if not config_info:
        print(f"ERROR: Unknown config '{config_name}'. Available: {list(ABLATION_CONFIGS.keys())}")
        sys.exit(1)

    needs_experience = config_info['needs_experience']

    # For experience_reuse, check if per-category parallelism is possible
    # Resolve input file paths to get the actual file list
    resolved_data_files = []
    for data_path in data_paths:
        resolved_data_files.extend(globlib.glob(data_path))

    per_category = False
    if needs_experience:
        if num_workers > 1 and num_workers == len(resolved_data_files):
            per_category = True
            print(f"NOTE: Per-category mode enabled — {num_workers} workers, "
                  f"each processing one input file sequentially")
        elif num_workers > 1:
            print(f"NOTE: Forcing 1 worker for experience_reuse config (was {num_workers}). "
                  f"To parallelize, set --workers equal to number of input files ({len(resolved_data_files)}).")
            num_workers = 1

    # Build config dict for metadata
    config = {
        'config_name': config_name,
        'model': model,
        'reasoning_effort': reasoning_effort,
        'needs_experience': needs_experience,
        'similarity_threshold': similarity_threshold,
        'clear_experience': clear_experience,
        'structured_goal': structured_goal,
        'neurosymbolic': neurosymbolic,
    }

    # Setup directories (use absolute paths to avoid any ambiguity)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_dir = Path(output_dir).resolve()
    else:
        base_dir = Path(f"experiments/results/experience_{config_name}_{model}_{timestamp}").resolve()

    work_dir = base_dir / "work"
    results_dir = base_dir / "results"
    traces_dir = base_dir / "traces"

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PARALLEL EXPERIENCE-REUSE HOMEBENCH RUNNER")
    print("=" * 60)
    print(f"Data files: {data_paths}")
    print(f"Config: {config_name}")
    print(f"Workers: {num_workers}")
    print(f"Model: {model}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Experience reuse: {needs_experience}")
    if needs_experience:
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Clear experience: {clear_experience}")
    if neurosymbolic:
        print("Strategy: neurosymbolic (SPARQL resolution + programmatic BT)")
    print(f"Rate limits: {rpm_limit} RPM, {tpm_limit:,} TPM")
    if home_config:
        print(f"Home config: {home_config}")
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
    est_per_test = 45 if needs_experience else 30
    print(f"Estimated time with {num_workers} workers: "
          f"~{total_tests * est_per_test / num_workers / 60:.1f} minutes")

    # Start workers
    print("\n" + "=" * 60)
    print("Starting workers...")
    print("=" * 60)

    # Build per-worker source file assignments for per-category mode
    worker_source_files = {}
    if per_category:
        for i, data_file in enumerate(resolved_data_files):
            worker_source_files[i] = os.path.basename(data_file)
        print("Worker → category assignments:")
        for wid, sf in worker_source_files.items():
            print(f"  Worker {wid} → {sf}")

    workers = []
    for i in range(num_workers):
        process = start_experience_worker(
            worker_id=i,
            image_name=image_name,
            work_dir=work_dir,
            results_dir=results_dir,
            openai_api_key=openai_api_key,
            model=model,
            config_name=config_name,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            reasoning_effort=reasoning_effort,
            similarity_threshold=similarity_threshold,
            clear_experience=clear_experience,
            home_config=Path(home_config) if home_config else None,
            source_file_filter=worker_source_files.get(i),
            traces_dir=traces_dir,
            structured_goal=structured_goal,
            neurosymbolic=neurosymbolic,
        )
        workers.append((i, process))
        time.sleep(2)

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

    # Wait for filesystem sync after Docker containers exit
    time.sleep(3)

    # Final stats
    elapsed = time.time() - start_time
    stats = monitor_progress(work_dir, total_tests)

    print("\n" + "=" * 60)
    print("TEST RUN COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")

    # Debug: check what files exist in the results directory
    print(f"\nDiagnostics:")
    print(f"  base_dir: {base_dir} (absolute: {base_dir.absolute()})")
    print(f"  work_dir: {work_dir} (absolute: {work_dir.absolute()})")
    print(f"  results_dir: {results_dir} (absolute: {results_dir.absolute()})")
    print(f"  results_dir exists: {results_dir.exists()}")
    if results_dir.exists():
        all_files = list(results_dir.iterdir())
        json_files = [f for f in all_files if f.suffix == '.json']
        print(f"  total files in results_dir: {len(all_files)}")
        print(f"  JSON files in results_dir: {len(json_files)}")
        if json_files:
            print(f"  first 5 JSON files: {[f.name for f in json_files[:5]]}")
        if all_files and not json_files:
            print(f"  non-JSON files: {[f.name for f in all_files[:10]]}")
    else:
        print(f"  WARNING: results_dir does not exist!")

    # Also check work queue status
    queue_file = work_dir / "queue.json"
    if queue_file.exists():
        with open(queue_file) as f:
            queue_data = json.load(f)
        q_completed = sum(1 for q in queue_data if q.get('status') == 'completed')
        q_failed = sum(1 for q in queue_data if q.get('status') == 'failed')
        q_running = sum(1 for q in queue_data if q.get('status') == 'running')
        q_pending = sum(1 for q in queue_data if q.get('status') == 'pending')
        print(f"  queue.json status: completed={q_completed}, failed={q_failed}, running={q_running}, pending={q_pending}")
    else:
        print(f"  WARNING: queue.json not found at {queue_file}")

    # Aggregate results
    print("\nAggregating results...")
    metrics = {}
    experience_metrics = {}

    agg_result = aggregate_results(
        results_dir, base_dir, config_name,
        generate_traces=generate_traces,
        experiment_config=config,
    )

    if agg_result and isinstance(agg_result, tuple):
        metrics, experience_metrics = agg_result

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

        if needs_experience:
            print()
            print("Experience Metrics:")
            print(f"  Total intents extracted: {experience_metrics['total_intents_extracted']}")
            print(f"  Total matched (feasible): {experience_metrics['total_matched']}")
            print(f"  Total matched (infeasible): {experience_metrics['total_matched_infeasible']}")
            print(f"  Total unmatched (novel): {experience_metrics['total_unmatched']}")
            print(f"  New experiences stored: {experience_metrics['total_new_experiences_stored']}")
            print(f"  Avg match count/test: {experience_metrics['avg_match_count']:.2f}")
            print(f"  Avg matched planning time: {experience_metrics['avg_matched_planning_time']:.2f}s")
            print(f"  Avg unmatched planning time: {experience_metrics['avg_unmatched_planning_time']:.2f}s")

        print("=" * 60)
    else:
        print("WARNING: No results to aggregate. Check worker logs for errors.")

    # Generate eval report
    if generate_report and metrics:
        generate_eval_report(base_dir)

    # Save summary
    base_dir.mkdir(parents=True, exist_ok=True)
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
        'experience_metrics': experience_metrics,
    }
    with open(base_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {base_dir}")
    return base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run experience-reuse HomeBench tests using Docker workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Experience reuse on single_feasible tests
  python run_parallel_experience_homebench.py \\
      --data experiments_data/experience_reuse/single_feasible_tests.json \\
      --config-name experience_reuse --model gpt-4o

  # Semantic query baseline (parallel)
  python run_parallel_experience_homebench.py \\
      --data "experiments_data/experience_reuse/*_tests.json" \\
      --config-name semantic_query --workers 4 --model gpt-4o

  # All categories for experience reuse
  python run_parallel_experience_homebench.py \\
      --data experiments_data/experience_reuse/single_feasible_tests.json \\
            experiments_data/experience_reuse/single_unfeasible_tests.json \\
            experiments_data/experience_reuse/multi_feasible_tests.json \\
            experiments_data/experience_reuse/multi_mixed_tests.json \\
      --config-name experience_reuse --model gpt-5-mini --reasoning-effort low
"""
    )

    # Data
    parser.add_argument(
        "--data", "-d",
        nargs="+",
        default=None,
        help="Path(s) to test data JSON files (required unless --regenerate-traces is used)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of tests"
    )

    # Config
    parser.add_argument(
        "--config-name", "-c",
        required=True,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation config to run"
    )

    # Workers
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}; forced to 1 for experience_reuse)"
    )

    # Model
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium"],
        default=None,
        help="Reasoning effort for reasoning models (gpt-5-mini, gpt-5-nano)"
    )

    # Experience options
    parser.add_argument(
        "--structured-goal",
        action="store_true",
        help="Use structured intent format for discovery and planning prompts (experience_reuse only)"
    )
    parser.add_argument(
        "--neurosymbolic",
        action="store_true",
        help="Use NeuroSymbolicRunner strategy (requires --config-name neurosymbolic)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Embedding similarity threshold for matching (default: 0.85)"
    )
    parser.add_argument(
        "--no-clear-experience",
        action="store_true",
        help="Do not clear experience store before starting"
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

    # Simulator
    parser.add_argument(
        "--home-config",
        default=None,
        help="JSON file with list of home IDs to load in simulator (reduces memory usage)"
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
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=DEFAULT_TPM_LIMIT,
    )

    # Info
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    # Trace regeneration (standalone mode — no experiment is run)
    parser.add_argument(
        "--regenerate-traces",
        metavar="EXPERIMENT_DIR",
        default=None,
        help="Regenerate HTML traces from an existing experiment directory and exit. "
             "Requires --config-name to know which viewer to use."
    )

    args = parser.parse_args()

    if args.regenerate_traces:
        output_dir = Path(args.regenerate_traces).resolve()
        if not output_dir.exists():
            print(f"ERROR: directory not found: {output_dir}")
            sys.exit(1)
        config_name = args.config_name
        print(f"Regenerating HTML traces in {output_dir}/traces/ (config={config_name})...")
        count = generate_html_traces(output_dir, config_name)
        print(f"Done. {count} HTML file(s) written.")
        sys.exit(0)

    if not args.regenerate_traces and not args.data:
        parser.error("--data is required unless --regenerate-traces is specified")

    if args.list_models:
        print("Available models:\n")
        print(f"{'Model':<15} {'Input/1M':<12} {'Output/1M':<12} Description")
        print("-" * 70)
        for model_name, info in AVAILABLE_MODELS.items():
            print(f"{model_name:<15} ${info['input']:<11.2f} ${info['output']:<11.2f} {info['description']}")
        print(f"\nDefault: {DEFAULT_MODEL}")
        sys.exit(0)

    run_parallel_experience_homebench(
        data_paths=args.data,
        config_name=args.config_name,
        num_workers=args.workers,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=args.output,
        image_name=args.image,
        rebuild_image=args.rebuild,
        rpm_limit=args.rpm_limit,
        tpm_limit=args.tpm_limit,
        limit=args.limit,
        similarity_threshold=args.similarity_threshold,
        clear_experience=not args.no_clear_experience,
        generate_traces=not args.no_traces,
        generate_report=not args.no_report,
        home_config=args.home_config,
        structured_goal=args.structured_goal,
        neurosymbolic=args.neurosymbolic,
    )


if __name__ == "__main__":
    main()
