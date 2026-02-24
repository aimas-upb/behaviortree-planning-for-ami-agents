#!/usr/bin/env python3
"""
Worker script for experience-reuse experiments.

Processes test cases sequentially within a single Docker container,
accumulating experience across tests. Each container handles one
test category (e.g., single_feasible, single_unfeasible) to ensure
experience reuse is not hindered by parallel execution.

Supports three ablation configs:
- semantic_nav: agentic affordance strategy (no experience)
- semantic_query: agentic_query affordance strategy (no experience)
- experience_reuse: agentic_query + experience matching pipeline
"""

import json
import os
import sys
import time
import argparse
import fcntl
import random
import httpx
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, Literal
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, '/app')

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError

load_dotenv()


# ============================================================================
# Retry Configuration
# ============================================================================

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 120.0  # seconds
JITTER_FACTOR = 0.5


# ============================================================================
# Rate Limiting Configuration
# ============================================================================

ESTIMATED_TOKENS_PER_EXPERIMENT = 10000
DEFAULT_TPM_LIMIT = 500000
DEFAULT_RPM_LIMIT = 500


class DistributedRateLimiter:
    """File-based distributed rate limiter for coordinating across Docker containers."""

    def __init__(self, work_dir: Path, rpm_limit: int = DEFAULT_RPM_LIMIT,
                 tpm_limit: int = DEFAULT_TPM_LIMIT):
        self.work_dir = Path(work_dir)
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rate_file = self.work_dir / "rate_limit.json"
        self.lock_file = self.work_dir / "rate_limit.lock"
        self.lock_file.touch(exist_ok=True)

    def _load_state(self) -> dict:
        if not self.rate_file.exists():
            return {"requests": [], "tokens": []}
        try:
            with open(self.rate_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"requests": [], "tokens": []}

    def _save_state(self, state: dict):
        with open(self.rate_file, 'w') as f:
            json.dump(state, f)

    def _clean_old_entries(self, entries: list, window_seconds: float = 60.0) -> list:
        cutoff = time.time() - window_seconds
        return [e for e in entries if e["timestamp"] > cutoff]

    def wait_for_capacity(self, estimated_tokens: int = ESTIMATED_TOKENS_PER_EXPERIMENT,
                          worker_id: int = 0) -> float:
        wait_start = time.time()
        max_wait = 300

        while True:
            with open(self.lock_file, 'r+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    state = self._load_state()
                    state["requests"] = self._clean_old_entries(state["requests"])
                    state["tokens"] = self._clean_old_entries(state["tokens"])

                    current_rpm = len(state["requests"])
                    current_tpm = sum(e["tokens"] for e in state["tokens"])

                    rpm_ok = current_rpm < self.rpm_limit
                    tpm_ok = (current_tpm + estimated_tokens) <= self.tpm_limit

                    if rpm_ok and tpm_ok:
                        now = time.time()
                        state["requests"].append({"timestamp": now, "worker": worker_id})
                        state["tokens"].append({"timestamp": now, "tokens": estimated_tokens, "worker": worker_id})
                        self._save_state(state)

                        waited = time.time() - wait_start
                        if waited > 1:
                            print(f"[Worker {worker_id}] Rate limit: waited {waited:.1f}s for capacity")
                        return waited

                    if not rpm_ok:
                        oldest_request = min(e["timestamp"] for e in state["requests"])
                        wait_time = 60 - (time.time() - oldest_request) + 0.5
                    else:
                        oldest_token = min(e["timestamp"] for e in state["tokens"])
                        wait_time = 60 - (time.time() - oldest_token) + 0.5

                    wait_time = max(0.5, min(wait_time, 10))
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

            if time.time() - wait_start > max_wait:
                print(f"[Worker {worker_id}] Warning: Rate limit wait exceeded {max_wait}s, proceeding anyway")
                return time.time() - wait_start

            time.sleep(wait_time)


def retry_with_exponential_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    retryable_exceptions: tuple = (RateLimitError, APIError, APIConnectionError, APITimeoutError),
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(f"[Retry] Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * JITTER_FACTOR * random.random()
                    sleep_time = delay + jitter

                    error_type = type(e).__name__
                    print(f"[Retry] {error_type} on attempt {attempt + 1}/{max_retries + 1}. "
                          f"Retrying in {sleep_time:.1f}s...")

                    if isinstance(e, RateLimitError) and hasattr(e, 'response'):
                        retry_after = e.response.headers.get('retry-after')
                        if retry_after:
                            sleep_time = max(sleep_time, float(retry_after))
                            print(f"[Retry] Server requested retry-after: {retry_after}s")

                    time.sleep(sleep_time)
            raise last_exception
        return wrapper
    return decorator


from src.config import (
    ExperimentConfig, ExperimentMeta, DiscoveryConfig, PlanningConfig,
    ExecutionConfig, ModelConfig, TracingConfig, AffordanceConfig,
    StateConfig, ReasoningConfig, OutputConfig, ExperienceConfig,
)
from src.runner import run_experiment
from src.experience.intent import IntentExtractor
from src.experience.engine import ExperienceEngine
from src.experience.matching import ExperienceMatcher
from src.experience.adaptation import ExperienceAdapter
from src.experience.runner import ExperiencePipelineRunner, ExperienceRunResult
from src.experience.neurosymbolic_runner import NeuroSymbolicRunner, NeuroSymbolicRunResult
from src.experience.bt_serialization import extract_action_urls


# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS_NO_EXPERIENCE = {
    "semantic_nav": {
        "affordance_strategy": "agentic",
        "state_strategy": "all",
        "reasoning_enabled": False,
        "output_format": "python_code",
        "prompt_strategy": "detailed",
    },
    "semantic_query": {
        "affordance_strategy": "agentic_query",
        "state_strategy": "all",
        "reasoning_enabled": False,
        "output_format": "python_code",
        "prompt_strategy": "detailed",
    },
}

ABLATION_CONFIGS_EXPERIENCE = {
    "experience_reuse": {
        "affordance_strategy": "agentic_query",
        "state_strategy": "all",
        "reasoning_enabled": False,
        "output_format": "python_code",
        "prompt_strategy": "detailed",
        "similarity_threshold": 0.85,
        "ontology": "ontologies/homeont.ttl",
        "clear_experience": True,
    },
    "neurosymbolic": {
        # Discovery and full planning are bypassed; these document the strategy.
        # The modify branch uses "detailed_structured_modify_only" / "python_code"
        # hardcoded inside NeuroSymbolicRunner._build_modify_tree().
        "output_format": "python_code",
        "prompt_strategy": "detailed_structured_modify_only",
        "ontology": "ontologies/homeont.ttl",
        "clear_experience": True,
    },
}


# ============================================================================
# Config creation
# ============================================================================

def create_config(
    name: str,
    affordance_strategy: str = "agentic_query",
    state_strategy: str = "all",
    reasoning_enabled: bool = False,
    reasoning_strategy: str = "chain_of_thought",
    output_format: str = "python_code",
    prompt_strategy: str = "detailed",
    model: str = "gpt-4o",
    reasoning_effort: Optional[str] = None,
    experience_store: str = "experience_store.json",
    similarity_threshold: float = 0.85,
    experience_enabled: bool = False,
) -> ExperimentConfig:
    """Create an experiment config."""
    return ExperimentConfig(
        experiment=ExperimentMeta(name=name, description=f"Experience Reuse Ablation: {name}"),
        discovery=DiscoveryConfig(
            affordances=AffordanceConfig(strategy=affordance_strategy),
            state=StateConfig(strategy=state_strategy),
        ),
        planning=PlanningConfig(
            reasoning=ReasoningConfig(
                enabled=reasoning_enabled,
                strategy=reasoning_strategy,
                max_turns=3,
            ),
            output=OutputConfig(format=output_format),
            prompt_strategy=prompt_strategy,
        ),
        execution=ExecutionConfig(max_ticks=10),
        model=ModelConfig(name=model, temperature=0.0, reasoning_effort=reasoning_effort),
        tracing=TracingConfig(enabled=True, output_dir="traces/", verbose=False),
        experience=ExperienceConfig(
            enabled=experience_enabled,
            persistence_path=experience_store,
            similarity_threshold=similarity_threshold,
        ),
    )


# ============================================================================
# HomeBench Test Result (matches run_homebench.py and worker.py)
# ============================================================================

@dataclass
class HomeBenchTestResult:
    """Result of running a single HomeBench test case."""
    test_id: str
    success: Literal["True", "False", "Quantifiable"] = "False"

    # Planning metrics
    plan_generated: bool = False
    plan_format: str = ""
    actions_in_plan: list = field(default_factory=list)
    params_in_plan: dict = field(default_factory=dict)

    # Execution metrics
    execution_success: bool = False
    execution_ticks: int = 0

    # Comparison metrics
    expected_actions: list = field(default_factory=list)
    expected_params: dict = field(default_factory=dict)
    matched_actions: list = field(default_factory=list)
    missing_actions: list = field(default_factory=list)
    extra_actions: list = field(default_factory=list)
    params_correct: bool = False

    # Property verification
    properties_checked: int = 0
    properties_matched: int = 0
    property_results: list = field(default_factory=list)

    # Impossible sub-goal detection
    expected_impossible: int = 0
    detected_impossible: list = field(default_factory=list)
    is_error_input_only: bool = False
    handled_correctly: bool = False

    # Experience-specific metrics
    experience_intents_extracted: int = 0
    experience_matched: int = 0
    experience_unmatched: int = 0
    experience_matched_infeasible: int = 0
    experience_avg_similarity: float = 0.0
    experience_matched_plan_time: float = 0.0
    experience_unmatched_plan_time: float = 0.0
    experience_new_stored: int = 0
    experience_store_size: int = 0

    # Timing and errors
    duration_seconds: float = 0.0
    failure_type: Optional[str] = None
    error: Optional[str] = None
    raw_result: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "success": self.success,
            "plan_generated": self.plan_generated,
            "plan_format": self.plan_format,
            "actions_in_plan": self.actions_in_plan,
            "params_in_plan": self.params_in_plan,
            "execution_success": self.execution_success,
            "execution_ticks": self.execution_ticks,
            "expected_actions": self.expected_actions,
            "expected_params": self.expected_params,
            "matched_actions": self.matched_actions,
            "missing_actions": self.missing_actions,
            "extra_actions": self.extra_actions,
            "params_correct": self.params_correct,
            "properties_checked": self.properties_checked,
            "properties_matched": self.properties_matched,
            "property_results": self.property_results,
            "expected_impossible": self.expected_impossible,
            "detected_impossible": self.detected_impossible,
            "is_error_input_only": self.is_error_input_only,
            "handled_correctly": self.handled_correctly,
            "experience_intents_extracted": self.experience_intents_extracted,
            "experience_matched": self.experience_matched,
            "experience_unmatched": self.experience_unmatched,
            "experience_matched_infeasible": self.experience_matched_infeasible,
            "experience_avg_similarity": self.experience_avg_similarity,
            "experience_matched_plan_time": self.experience_matched_plan_time,
            "experience_unmatched_plan_time": self.experience_unmatched_plan_time,
            "experience_new_stored": self.experience_new_stored,
            "experience_store_size": self.experience_store_size,
            "duration": self.duration_seconds,
            "failure_type": self.failure_type,
            "error": self.error,
            "trace": self.raw_result,
        }


# ============================================================================
# Helpers
# ============================================================================

def extract_actions_from_plan(plan: dict) -> list[str]:
    """Extract action URLs from a plan."""
    actions = []
    if not plan:
        return actions

    content = plan.get("content", plan)
    if isinstance(content, str):
        import re
        urls = re.findall(r'http://[^"\'>\s]+', content)
        actions = [u for u in urls if '/properties/' not in u]
    else:
        _extract_actions_recursive(content, actions)

    return actions


def _extract_actions_recursive(node: dict, actions: list):
    if not isinstance(node, dict):
        return
    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            actions.append(url)
    for child in node.get("children", []):
        _extract_actions_recursive(child, actions)


def extract_params_from_plan(plan: dict) -> dict:
    params = {}
    content = plan.get("content", plan)
    if isinstance(content, dict):
        _extract_params_recursive(content, params)
    return params


def _extract_params_recursive(node: dict, params: dict):
    if not isinstance(node, dict):
        return
    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            params[url] = node.get("parameters", {})
    for child in node.get("children", []):
        _extract_params_recursive(child, params)


def verify_property(simulator_url: str, property_url: str, expected_value) -> tuple[bool, any]:
    try:
        response = httpx.get(property_url, timeout=30.0)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        actual = response.json()
        matched = actual == expected_value
        return matched, actual
    except Exception as e:
        return False, str(e)


def classify_failure(result: HomeBenchTestResult, exp_result: dict) -> str:
    error_msg = exp_result.get("error", "") or ""
    planning = exp_result.get("planning", {})
    execution = exp_result.get("execution", {})

    if "JSON parse error" in error_msg or "JSON decode" in error_msg:
        return "parse_error"

    plan_explanation = planning.get("plan", {}).get("explanation", "")
    if "parse error" in plan_explanation.lower() or ("json" in plan_explanation.lower() and "error" in plan_explanation.lower()):
        return "parse_error"

    exec_error = execution.get("error", "") or ""
    if "Unknown node type" in exec_error or "Compilation" in error_msg:
        return "compilation_error"
    if "KeyError" in exec_error or "missing" in exec_error.lower():
        return "compilation_error"

    if result.plan_generated and not result.execution_success:
        return "execution_error"

    if result.execution_success and result.properties_checked > 0 and result.properties_matched < result.properties_checked:
        return "property_mismatch"

    if not result.plan_generated:
        if "parse" in error_msg.lower() or "json" in error_msg.lower():
            return "parse_error"
        return "other"

    return "other"


def classify_experience_failure(result: HomeBenchTestResult, exp_result: ExperienceRunResult) -> str:
    error_msg = exp_result.error or ""

    if not result.plan_generated:
        if "parse" in error_msg.lower() or "json" in error_msg.lower():
            return "parse_error"
        return "other"

    if result.plan_generated and not result.execution_success:
        return "execution_error"

    if result.execution_success and result.properties_checked > 0 and result.properties_matched < result.properties_checked:
        return "property_mismatch"

    if result.missing_actions or result.extra_actions:
        return "action_mismatch"

    return "other"


def generate_trace_html(result: dict, results_dir: Path, config_name: str,
                        experiment_config: dict) -> None:
    """Write a reshaped JSON trace and its HTML file for a single test result."""
    test_id = result.get('test_id', result.get('id', 'unknown'))
    trace_data = result.get('trace', result.get('raw_result'))
    if not trace_data:
        return

    # Use TRACES_DIR env var if set (needed when results_dir is a top-level
    # Docker mount like /results, where .parent would be /), otherwise fall
    # back to a sibling directory of results_dir on the host.
    traces_dir_env = os.environ.get("TRACES_DIR")
    if traces_dir_env:
        traces_dir = Path(traces_dir_env)
    else:
        traces_dir = results_dir.parent / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    is_experience = config_name in ABLATION_CONFIGS_EXPERIENCE

    # Reshape trace
    if is_experience:
        reshaped = _reshape_experience_trace(trace_data, test_id, experiment_config)
    else:
        reshaped = _reshape_standard_trace(trace_data, test_id, experiment_config)

    trace_file = traces_dir / f"{test_id}.json"
    with open(trace_file, 'w') as f:
        json.dump(reshaped, f, indent=2, default=str)

    # Generate HTML
    if is_experience:
        try:
            from experience_trace_viewer import export_experience_html
            export_experience_html(reshaped, str(trace_file.with_suffix(".html")))
        except Exception:
            pass
    else:
        try:
            from trace_viewer import export_html
            export_html(reshaped, str(trace_file.with_suffix(".html")))
        except Exception:
            pass


def _reshape_experience_trace(raw_result: dict, test_id: str, config: dict) -> dict:
    trace: dict = {
        "config_name": f"experience_{test_id}",
        "config": config,
        "success": raw_result.get("success", False),
        "duration_seconds": raw_result.get("duration_seconds", 0),
        "error": raw_result.get("error"),
        "intents": raw_result.get("intents"),
        "match_results": raw_result.get("match_results"),
        "matched_plan_traces": raw_result.get("matched_plan_traces"),
        "matched_plan_time_seconds": raw_result.get("matched_plan_time_seconds"),
        "unmatched_plan_time_seconds": raw_result.get("unmatched_plan_time_seconds"),
        "detected_impossible": raw_result.get("detected_impossible"),
    }
    unmatched_traces = raw_result.get("unmatched_plan_traces", [])
    if unmatched_traces and isinstance(unmatched_traces[0], dict):
        first_trace = unmatched_traces[0]
        trace["discovery"] = first_trace.get("discovery", {})
        trace["planning"] = first_trace.get("planning", {})
    else:
        trace["discovery"] = {}
        trace["planning"] = {}
    trace["execution"] = raw_result.get("execution") or {}
    intents = raw_result.get("intents", [])
    if intents:
        intent_texts = [
            i.get("text_intent", "") if isinstance(i, dict) else "" for i in intents
        ]
        trace["goal"] = ", ".join(t for t in intent_texts if t)
    return trace


def _reshape_standard_trace(raw_result: dict, test_id: str, config: dict) -> dict:
    trace = raw_result.copy()
    trace["config_name"] = f"standard_{test_id}"
    if "config" not in trace or not isinstance(trace["config"], dict) or "model" not in trace["config"]:
        trace["config"] = config
    return trace


def reset_simulator(simulator_url: str, home_id: str) -> bool:
    try:
        response = httpx.post(
            f"{simulator_url}/reset",
            json={"home": str(home_id)},
            timeout=30.0
        )
        if response.status_code == 200:
            return True
        response = httpx.post(
            f"{simulator_url}/workspaces/home{home_id}/reset",
            timeout=30.0
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Failed to reset home{home_id}: {e}")
        return False


# ============================================================================
# Work queue management
# ============================================================================

def claim_work_item(work_dir: Path, worker_id: int,
                    source_file_filter: str | None = None) -> dict | None:
    queue_file = work_dir / "queue.json"
    lock_file = work_dir / "queue.lock"
    lock_file.touch(exist_ok=True)

    with open(lock_file, 'r+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not queue_file.exists():
                return None

            with open(queue_file, 'r') as f:
                queue = json.load(f)

            for item in queue:
                if item.get('status') != 'pending':
                    continue
                # If a source file filter is set, only claim items from that file
                if source_file_filter and item.get('source_file') != source_file_filter:
                    continue

                item['status'] = 'running'
                item['worker_id'] = worker_id
                item['started_at'] = datetime.now().isoformat()

                with open(queue_file, 'w') as f:
                    json.dump(queue, f, indent=2)

                return item

            return None
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def mark_work_complete(work_dir: Path, item_id: str, success: bool):
    queue_file = work_dir / "queue.json"
    lock_file = work_dir / "queue.lock"

    with open(lock_file, 'r+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with open(queue_file, 'r') as f:
                queue = json.load(f)

            for item in queue:
                if item.get('id') == item_id:
                    item['status'] = 'completed' if success else 'failed'
                    item['completed_at'] = datetime.now().isoformat()
                    break

            with open(queue_file, 'w') as f:
                json.dump(queue, f, indent=2)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# ============================================================================
# Standard (no-experience) test runner
# ============================================================================

@retry_with_exponential_backoff(max_retries=MAX_RETRIES)
def run_experiment_with_retry(config, goal: str, entry_point: str, client: OpenAI) -> dict:
    return run_experiment(
        config=config,
        goal=goal,
        entry_point=entry_point,
        client=client,
    )


def run_standard_test(
    work_item: dict,
    simulator_url: str,
    results_dir: Path,
    client: OpenAI,
    worker_id: int,
    config_name: str,
    config_params: dict,
    model: str,
    reasoning_effort: Optional[str] = None,
    rate_limiter: Optional[DistributedRateLimiter] = None,
) -> bool:
    """Run a single test case without experience reuse (semantic_nav or semantic_query)."""
    test_id = work_item['test_id']
    home_id = work_item['home_id']
    goal = work_item['input']
    expected_outputs = work_item['output']

    expected_successes = [o for o in expected_outputs if o.get("execution") == "success"]
    expected_errors = [o for o in expected_outputs if o.get("execution") == "error_input"]

    result = HomeBenchTestResult(test_id=test_id)
    result.expected_actions = [o.get("affordance", "") for o in expected_successes if o.get("affordance")]
    result.expected_params = {o.get("affordance"): o.get("params", {}) for o in expected_successes if o.get("affordance")}
    result.expected_impossible = len(expected_errors)
    result.is_error_input_only = len(expected_successes) == 0 and len(expected_errors) > 0

    entry_point = f"{simulator_url}/workspaces/home{home_id}#workspace"

    if rate_limiter:
        rate_limiter.wait_for_capacity(worker_id=worker_id)

    print(f"[Worker {worker_id}] Running standard test ({config_name}): {test_id}")

    start_time = datetime.now()

    try:
        reset_simulator(simulator_url, home_id)

        exp_config = create_config(
            name=f"{config_name}_{test_id}",
            affordance_strategy=config_params.get('affordance_strategy', 'agentic'),
            state_strategy=config_params.get('state_strategy', 'all'),
            reasoning_enabled=config_params.get('reasoning_enabled', False),
            output_format=config_params.get('output_format', 'python_code'),
            prompt_strategy=config_params.get('prompt_strategy', 'detailed'),
            model=model,
            reasoning_effort=reasoning_effort,
        )

        exp_result = run_experiment_with_retry(
            config=exp_config,
            goal=goal,
            entry_point=entry_point,
            client=client,
        )
        result.raw_result = exp_result

        # Check planning
        if exp_result.get("planning"):
            planning = exp_result["planning"]
            if planning.get("success"):
                result.plan_generated = True
                result.plan_format = planning.get("plan", {}).get("format", "")
                plan_content = planning.get("plan", {}).get("content", {})
                result.actions_in_plan = extract_actions_from_plan({"content": plan_content})
                result.params_in_plan = extract_params_from_plan({"content": plan_content})

                detected = planning.get("plan", {}).get("detected_impossible", [])
                if detected:
                    result.detected_impossible = detected

        # Check execution
        if exp_result.get("execution"):
            execution = exp_result["execution"]
            result.execution_success = execution.get("success", False)
            result.execution_ticks = execution.get("ticks", 0)

        # Compare actions
        expected_set = set(result.expected_actions)
        actual_set = set(result.actions_in_plan)
        result.matched_actions = list(expected_set & actual_set)
        result.missing_actions = list(expected_set - actual_set)
        result.extra_actions = list(actual_set - expected_set)

        # Verify properties
        if result.execution_success and not result.is_error_input_only:
            for expected in expected_successes:
                test_spec = expected.get("test", {})
                if test_spec:
                    prop_url = test_spec.get("property")
                    exp_val = test_spec.get("expected_value")
                    if prop_url:
                        matched, actual = verify_property(simulator_url, prop_url, exp_val)
                        result.properties_checked += 1
                        if matched:
                            result.properties_matched += 1
                        result.property_results.append({
                            "property": prop_url,
                            "expected": exp_val,
                            "actual": actual,
                            "matched": matched,
                        })

        # Determine success
        if result.is_error_input_only:
            detected_as_impossible = (
                len(result.detected_impossible) > 0 or
                not result.plan_generated or
                len(result.actions_in_plan) == 0
            )
            result.success = "True" if detected_as_impossible else "False"
        else:
            is_plan_executed = result.plan_generated and result.execution_success

            if not is_plan_executed:
                result.success = "False"
            else:
                no_extra_actions = len(result.extra_actions) == 0
                all_properties_matched = result.properties_matched == result.properties_checked

                result.handled_correctly = no_extra_actions and all_properties_matched

                if result.handled_correctly:
                    result.success = "True"
                elif result.matched_actions:
                    result.success = "Quantifiable"
                else:
                    result.success = "False"

        if result.success == "False":
            if result.is_error_input_only:
                result.failure_type = "error_input_not_detected"
            else:
                result.failure_type = classify_failure(result, exp_result)

    except Exception as e:
        result.error = str(e)
        result.failure_type = "other"
        print(f"[Worker {worker_id}] Test {test_id} failed with exception: {e}")

    result.duration_seconds = (datetime.now() - start_time).total_seconds()

    # Save result
    result_dict = result.to_dict()
    result_file = results_dir / f"{test_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    generate_trace_html(result_dict, results_dir, config_name,
                        work_item.get('config', {}))

    status_icon = "✓" if result.success == "True" else ("◐" if result.success == "Quantifiable" else "✗")
    print(f"[Worker {worker_id}] {status_icon} {test_id}: {result.success} "
          f"(matched: {len(result.matched_actions)}/{len(result.expected_actions)}, "
          f"props: {result.properties_matched}/{result.properties_checked})")

    return result.success != "False"


# ============================================================================
# Experience-reuse test runner
# ============================================================================

def run_experience_test(
    work_item: dict,
    simulator_url: str,
    results_dir: Path,
    client: OpenAI,
    worker_id: int,
    pipeline_runner: ExperiencePipelineRunner,
    engine: ExperienceEngine,
    config_name: str = "experience_reuse",
    rate_limiter: Optional[DistributedRateLimiter] = None,
) -> bool:
    """Run a single test case with experience reuse pipeline."""
    test_id = work_item['test_id']
    home_id = work_item['home_id']
    goal = work_item['input']
    expected_outputs = work_item['output']

    expected_successes = [o for o in expected_outputs if o.get("execution") == "success"]
    expected_errors = [o for o in expected_outputs if o.get("execution") == "error_input"]

    result = HomeBenchTestResult(test_id=test_id)
    result.expected_actions = [o.get("affordance", "") for o in expected_successes if o.get("affordance")]
    result.expected_params = {o.get("affordance"): o.get("params", {}) for o in expected_successes if o.get("affordance")}
    result.expected_impossible = len(expected_errors)
    result.is_error_input_only = len(expected_successes) == 0 and len(expected_errors) > 0

    entry_point = f"{simulator_url}/workspaces/home{home_id}#workspace"

    if rate_limiter:
        rate_limiter.wait_for_capacity(worker_id=worker_id)

    print(f"[Worker {worker_id}] Running experience test: {test_id}")

    start_time = datetime.now()

    try:
        reset_simulator(simulator_url, home_id)

        # Run experience pipeline
        exp_result = pipeline_runner.run(
            goal=goal,
            entry_point=entry_point,
            home_id=str(home_id),
            test_id=test_id,
        )

        result.raw_result = exp_result.to_dict()

        # Map ExperienceRunResult to TestResult fields
        if exp_result.combined_plan_ir:
            result.plan_generated = True
            result.plan_format = "json_ir"
            action_urls = extract_action_urls(exp_result.combined_plan_ir)
            result.actions_in_plan = action_urls
            result.params_in_plan = _extract_params_from_ir(exp_result.combined_plan_ir)
        elif exp_result.matched_infeasible_intents and not exp_result.matched_plans:
            result.plan_generated = False

        # Track detected impossible sub-goals
        result.detected_impossible = exp_result.detected_impossible

        # Execution result
        if exp_result.execution_result:
            result.execution_success = exp_result.execution_result.success
            result.execution_ticks = exp_result.execution_result.ticks

        # Compare actions
        expected_set = set(result.expected_actions)
        actual_set = set(result.actions_in_plan)
        result.matched_actions = list(expected_set & actual_set)
        result.missing_actions = list(expected_set - actual_set)
        result.extra_actions = list(actual_set - expected_set)

        # Verify properties
        if result.execution_success and not result.is_error_input_only:
            for expected in expected_successes:
                test_spec = expected.get("test", {})
                if test_spec:
                    prop_url = test_spec.get("property")
                    exp_val = test_spec.get("expected_value")
                    if prop_url:
                        matched, actual = verify_property(simulator_url, prop_url, exp_val)
                        result.properties_checked += 1
                        if matched:
                            result.properties_matched += 1
                        result.property_results.append({
                            "property": prop_url,
                            "expected": exp_val,
                            "actual": actual,
                            "matched": matched,
                        })

        # Determine success
        if result.is_error_input_only:
            detected_as_impossible = (
                len(result.detected_impossible) > 0 or
                not result.plan_generated or
                len(result.actions_in_plan) == 0
            )
            result.success = "True" if detected_as_impossible else "False"
        else:
            is_plan_executed = result.plan_generated and result.execution_success

            if not is_plan_executed:
                if (exp_result.matched_infeasible_intents
                        and not exp_result.matched_plans
                        and not exp_result.unmatched_plans):
                    if result.is_error_input_only:
                        result.success = "True"
                    else:
                        result.success = "False"
                else:
                    result.success = "False"
            else:
                no_extra_actions = len(result.extra_actions) == 0
                all_properties_matched = result.properties_matched == result.properties_checked

                result.handled_correctly = no_extra_actions and all_properties_matched

                if result.handled_correctly:
                    result.success = "True"
                elif result.matched_actions:
                    result.success = "Quantifiable"
                else:
                    result.success = "False"

        if result.success == "False":
            if result.is_error_input_only:
                result.failure_type = "error_input_not_detected"
            else:
                result.failure_type = classify_experience_failure(result, exp_result)

        # Learning: store experiences on success
        if result.success == "True" and exp_result.intents:
            stored = pipeline_runner.store_experiences(
                intents=exp_result.intents,
                match_results=exp_result.match_results,
                combined_plan_ir=exp_result.combined_plan_ir,
                test_id=test_id,
            )
            exp_result.new_experiences_stored = stored

            # Store confirmed infeasible intents from ground truth
            _store_infeasible_from_ground_truth(
                pipeline_runner, expected_outputs, exp_result, test_id,
                home_id=str(home_id),
            )

        # Experience-specific metrics
        n_intents = len(exp_result.intents)
        n_matched = sum(1 for m in exp_result.match_results if m.matched and not m.is_infeasible)
        n_infeasible = len(exp_result.matched_infeasible_intents)
        n_unmatched = n_intents - n_matched - n_infeasible

        result.experience_intents_extracted = n_intents
        result.experience_matched = n_matched
        result.experience_unmatched = n_unmatched
        result.experience_matched_infeasible = n_infeasible

        matched_sims = [m.similarity_score for m in exp_result.match_results if m.matched]
        result.experience_avg_similarity = (
            sum(matched_sims) / len(matched_sims) if matched_sims else 0.0
        )
        result.experience_matched_plan_time = exp_result.matched_plan_time_seconds
        result.experience_unmatched_plan_time = exp_result.unmatched_plan_time_seconds
        result.experience_new_stored = exp_result.new_experiences_stored
        result.experience_store_size = engine.size()

    except Exception as e:
        result.error = str(e)
        result.failure_type = "other"
        print(f"[Worker {worker_id}] Test {test_id} failed with exception: {e}")

    result.duration_seconds = (datetime.now() - start_time).total_seconds()

    # Save result
    result_dict = result.to_dict()
    result_file = results_dir / f"{test_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    generate_trace_html(result_dict, results_dir, config_name,
                        work_item.get('config', {}))

    status_icon = "✓" if result.success == "True" else ("◐" if result.success == "Quantifiable" else "✗")
    print(f"[Worker {worker_id}] {status_icon} {test_id}: {result.success} "
          f"(matched: {len(result.matched_actions)}/{len(result.expected_actions)}, "
          f"exp_matched: {result.experience_matched}, "
          f"store_size: {result.experience_store_size})")

    return result.success != "False"


def _reshape_ns_trace(raw_result: dict, test_id: str, config: dict) -> dict:
    """Reshape a NeuroSymbolicRunResult dict for the experience trace viewer."""
    trace: dict = {
        "config_name": f"neurosymbolic_{test_id}",
        "config": config,
        "success": raw_result.get("success", False),
        "duration_seconds": raw_result.get("duration_seconds", 0),
        "error": raw_result.get("error"),
        "intents": raw_result.get("intents"),
        "intent_extraction_trace": raw_result.get("intent_extraction_trace"),
        "resolution_results": raw_result.get("resolution_results"),
        "set_count": raw_result.get("set_count", 0),
        "modify_count": raw_result.get("modify_count", 0),
        "impossible_count": raw_result.get("impossible_count", 0),
        "impossible_details": raw_result.get("impossible_details"),
        "set_actions_tree_ir": raw_result.get("set_actions_tree_ir"),
        "modify_actions_tree_ir": raw_result.get("modify_actions_tree_ir"),
        "combined_plan_ir": raw_result.get("combined_plan_ir"),
        "modify_plan_trace": raw_result.get("modify_plan_trace"),
        "modify_plan_time_seconds": raw_result.get("modify_plan_time_seconds", 0.0),
        "detected_impossible": raw_result.get("detected_impossible"),
        "execution": raw_result.get("execution") or {},
    }
    modify_trace = raw_result.get("modify_plan_trace") or {}
    trace["planning"] = modify_trace.get("planning") or {}
    # Lift the discovery context built for modify-branch planning (if any) so
    # the trace viewer can render "Planning Context (What the Model Sees)".
    trace["discovery"] = modify_trace.get("discovery") or {}
    intents = raw_result.get("intents", [])
    if intents:
        intent_texts = [
            i.get("text_intent", "") if isinstance(i, dict) else "" for i in intents
        ]
        trace["goal"] = ", ".join(t for t in intent_texts if t)
    return trace


# ============================================================================
# Neuro-symbolic test runner
# ============================================================================

def run_ns_test(
    work_item: dict,
    simulator_url: str,
    results_dir: Path,
    client: OpenAI,
    worker_id: int,
    ns_runner: NeuroSymbolicRunner,
    engine: ExperienceEngine,
    rate_limiter: Optional[DistributedRateLimiter] = None,
) -> bool:
    """Run a single test case with the NeuroSymbolicRunner."""
    test_id = work_item['test_id']
    home_id = work_item['home_id']
    goal = work_item['input']
    expected_outputs = work_item['output']

    expected_successes = [o for o in expected_outputs if o.get("execution") == "success"]
    expected_errors = [o for o in expected_outputs if o.get("execution") == "error_input"]

    result = HomeBenchTestResult(test_id=test_id)
    result.expected_actions = [o.get("affordance", "") for o in expected_successes if o.get("affordance")]
    result.expected_params = {o.get("affordance"): o.get("params", {}) for o in expected_successes if o.get("affordance")}
    result.expected_impossible = len(expected_errors)
    result.is_error_input_only = len(expected_successes) == 0 and len(expected_errors) > 0

    entry_point = f"{simulator_url}/workspaces/home{home_id}#workspace"

    if rate_limiter:
        rate_limiter.wait_for_capacity(worker_id=worker_id)

    print(f"[Worker {worker_id}] Running neurosymbolic test: {test_id}")

    start_time = datetime.now()

    try:
        reset_simulator(simulator_url, home_id)

        ns_result: NeuroSymbolicRunResult = ns_runner.run(
            goal=goal,
            entry_point=entry_point,
            home_id=str(home_id),
            test_id=test_id,
        )

        result.raw_result = ns_result.to_dict()

        # Plan produced?
        if ns_result.combined_plan_ir:
            result.plan_generated = True
            result.plan_format = "json_ir"
            result.actions_in_plan = extract_action_urls(ns_result.combined_plan_ir)
            result.params_in_plan = _extract_params_from_ir(ns_result.combined_plan_ir)

        result.detected_impossible = ns_result.detected_impossible

        # Execution
        if ns_result.execution_result:
            result.execution_success = ns_result.execution_result.success
            result.execution_ticks = ns_result.execution_result.ticks

        # Compare actions
        expected_set = set(result.expected_actions)
        actual_set = set(result.actions_in_plan)
        result.matched_actions = list(expected_set & actual_set)
        result.missing_actions = list(expected_set - actual_set)
        result.extra_actions = list(actual_set - expected_set)

        # Verify properties
        if result.execution_success and not result.is_error_input_only:
            for expected in expected_successes:
                test_spec = expected.get("test", {})
                if test_spec:
                    prop_url = test_spec.get("property")
                    exp_val = test_spec.get("expected_value")
                    if prop_url:
                        matched, actual = verify_property(simulator_url, prop_url, exp_val)
                        result.properties_checked += 1
                        if matched:
                            result.properties_matched += 1
                        result.property_results.append({
                            "property": prop_url,
                            "expected": exp_val,
                            "actual": actual,
                            "matched": matched,
                        })

        # Determine success
        if result.is_error_input_only:
            detected_as_impossible = (
                len(result.detected_impossible) > 0 or
                not result.plan_generated or
                len(result.actions_in_plan) == 0
            )
            result.success = "True" if detected_as_impossible else "False"
        else:
            is_plan_executed = result.plan_generated and result.execution_success
            if not is_plan_executed:
                result.success = "False"
            else:
                no_extra = len(result.extra_actions) == 0
                all_props = result.properties_matched == result.properties_checked
                result.handled_correctly = no_extra and all_props
                if result.handled_correctly:
                    result.success = "True"
                elif result.matched_actions:
                    result.success = "Quantifiable"
                else:
                    result.success = "False"

        if result.success == "False":
            result.failure_type = (
                "error_input_not_detected" if result.is_error_input_only
                else "execution_error" if result.plan_generated
                else "other"
            )

        # Store confirmed-infeasible intents in the engine
        if ns_result.intents and expected_errors:
            error_positions: set[int] = set()
            for idx, output in enumerate(expected_outputs):
                if output.get("execution") == "error_input":
                    error_positions.add(idx)
            for intent in ns_result.intents:
                if intent.original_index in error_positions:
                    ns_runner.store_infeasible(intent, test_id, home_id=str(home_id))

        # NS-specific metrics (reuse experience fields for aggregation compatibility)
        result.experience_intents_extracted = len(ns_result.intents)
        result.experience_matched = len(ns_result.set_intents)
        result.experience_unmatched = len(ns_result.modify_intents)
        result.experience_matched_infeasible = len(ns_result.impossible_intents)
        result.experience_unmatched_plan_time = ns_result.modify_plan_time_seconds
        result.experience_store_size = engine.size()

    except Exception as e:
        result.error = str(e)
        result.failure_type = "other"
        print(f"[Worker {worker_id}] Test {test_id} failed with exception: {e}")

    result.duration_seconds = (datetime.now() - start_time).total_seconds()

    result_dict = result.to_dict()
    result_file = results_dir / f"{test_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # Reshape and write trace + HTML
    raw = result_dict.get('trace') or {}
    reshaped = _reshape_ns_trace(raw, test_id, work_item.get('config', {}))
    traces_dir_env = os.environ.get("TRACES_DIR")
    traces_dir = Path(traces_dir_env) if traces_dir_env else results_dir.parent / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_file = traces_dir / f"{test_id}.json"
    with open(trace_file, 'w') as f:
        json.dump(reshaped, f, indent=2, default=str)
    try:
        from experience_trace_viewer import export_experience_html
        export_experience_html(reshaped, str(trace_file.with_suffix(".html")))
    except Exception:
        pass

    status_icon = "✓" if result.success == "True" else ("◐" if result.success == "Quantifiable" else "✗")
    set_c = result.experience_matched
    mod_c = result.experience_unmatched
    print(f"[Worker {worker_id}] {status_icon} {test_id}: {result.success} "
          f"(matched: {len(result.matched_actions)}/{len(result.expected_actions)}, "
          f"set: {set_c}, modify: {mod_c})")

    return result.success != "False"


def _extract_params_from_ir(json_ir: dict) -> dict:
    params = {}
    _collect_params(json_ir, params)
    return params


def _collect_params(node: dict, params: dict):
    if not isinstance(node, dict):
        return
    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            params[url] = node.get("parameters", {})
    for child in node.get("children", []):
        _collect_params(child, params)


def _store_infeasible_from_ground_truth(
    pipeline_runner: ExperiencePipelineRunner,
    expected_outputs: list[dict],
    exp_result: ExperienceRunResult,
    test_id: str,
    home_id: str = "",
):
    """Store infeasible intents confirmed by ground truth."""
    if not exp_result.intents:
        return

    error_positions: set[int] = set()
    for i, output in enumerate(expected_outputs):
        if output.get("execution") == "error_input":
            error_positions.add(i)

    if not error_positions:
        return

    for intent in exp_result.intents:
        if intent.original_index in error_positions:
            pipeline_runner.store_infeasible(intent, test_id, home_id=home_id)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experience reuse experiment worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--simulator-url", default="http://localhost:8080")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--config-name", required=True,
                        choices=list(ABLATION_CONFIGS_NO_EXPERIENCE.keys()) + list(ABLATION_CONFIGS_EXPERIENCE.keys()),
                        help="Ablation config to run")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium"],
                        help="Reasoning effort for reasoning models (gpt-5-mini, gpt-5-nano)")
    parser.add_argument("--rpm-limit", type=int, default=DEFAULT_RPM_LIMIT)
    parser.add_argument("--tpm-limit", type=int, default=DEFAULT_TPM_LIMIT)
    parser.add_argument("--no-rate-limit", action="store_true")
    parser.add_argument("--ontology", default="ontologies/homeont.ttl",
                        help="Path to homeont.ttl ontology file")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Embedding similarity threshold for experience matching")
    parser.add_argument("--clear-experience", action="store_true",
                        help="Clear experience store before starting")
    parser.add_argument("--source-file-filter", type=str, default=None,
                        help="Only process tests from this source file (for per-category parallelism)")
    parser.add_argument("--structured-goal", action="store_true",
                        help="Use structured intent format for discovery and planning prompts")
    parser.add_argument("--neurosymbolic", action="store_true",
                        help="Use NeuroSymbolicRunner (requires config_name=neurosymbolic)")

    args = parser.parse_args()

    worker_id = args.worker_id
    work_dir = args.work_dir
    results_dir = args.results_dir
    config_name = args.config_name

    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"[Worker {worker_id}] ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Setup rate limiter
    rate_limiter = None
    if not args.no_rate_limit:
        rate_limiter = DistributedRateLimiter(
            work_dir=work_dir,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
        )
        print(f"[Worker {worker_id}] Rate limiting enabled: "
              f"{args.rpm_limit} RPM, {args.tpm_limit} TPM")

    # Determine if this is an experience config
    is_experience = config_name in ABLATION_CONFIGS_EXPERIENCE
    is_neurosymbolic = config_name == "neurosymbolic" or args.neurosymbolic

    # Setup experience pipeline / NS runner if needed
    pipeline_runner = None
    ns_runner = None
    engine = None

    if is_experience:
        exp_params = ABLATION_CONFIGS_EXPERIENCE[config_name]
        ontology_path = args.ontology
        similarity_threshold = args.similarity_threshold

        # Experience store path scoped to this worker
        experience_store_path = str(results_dir / f"experience_store_worker{worker_id}.json")

        if args.clear_experience:
            store_path = Path(experience_store_path)
            if store_path.exists():
                store_path.unlink()
                print(f"[Worker {worker_id}] Cleared experience store: {experience_store_path}")

        ontology_text = Path(ontology_path).read_text()
        engine = ExperienceEngine(persistence_path=experience_store_path)
        intent_extractor = IntentExtractor(ontology_text=ontology_text)

        if is_neurosymbolic:
            # Config for the NS runner: model + execution are used at runtime;
            # output_format and prompt_strategy document the modify-branch strategy.
            ns_exp_config = create_config(
                name=f"neurosymbolic_{config_name}",
                output_format=exp_params.get("output_format", "python_code"),
                prompt_strategy=exp_params.get("prompt_strategy", "detailed_structured_modify_only"),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                experience_store=experience_store_path,
                experience_enabled=True,
            )
            ns_runner = NeuroSymbolicRunner(
                config=ns_exp_config,
                client=client,
                engine=engine,
                intent_extractor=intent_extractor,
            )
            print(f"[Worker {worker_id}] NeuroSymbolicRunner initialized "
                  f"(ontology={ontology_path})")
        else:
            matcher = ExperienceMatcher(similarity_threshold=similarity_threshold)
            adapter = ExperienceAdapter()

            exp_config = create_config(
                name=f"experience_reuse_{config_name}",
                affordance_strategy=exp_params.get('affordance_strategy', 'agentic_query'),
                state_strategy=exp_params.get('state_strategy', 'all'),
                reasoning_enabled=exp_params.get('reasoning_enabled', False),
                output_format=exp_params.get('output_format', 'python_code'),
                prompt_strategy=exp_params.get('prompt_strategy', 'detailed'),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                experience_store=experience_store_path,
                similarity_threshold=similarity_threshold,
                experience_enabled=True,
            )

            pipeline_runner = ExperiencePipelineRunner(
                config=exp_config,
                client=client,
                engine=engine,
                matcher=matcher,
                intent_extractor=intent_extractor,
                adapter=adapter,
                structured_goal=args.structured_goal,
            )

            print(f"[Worker {worker_id}] Experience pipeline initialized "
                  f"(threshold={similarity_threshold}, ontology={ontology_path})")

    source_file_filter = args.source_file_filter
    if source_file_filter:
        print(f"[Worker {worker_id}] Filtering to source file: {source_file_filter}")

    print(f"[Worker {worker_id}] Ready in {config_name} mode, looking for work...")

    # Process work items
    experiments_run = 0
    while True:
        work_item = claim_work_item(work_dir, worker_id, source_file_filter)

        if work_item is None:
            print(f"[Worker {worker_id}] No more work available. "
                  f"Completed {experiments_run} experiments.")
            break

        item_id = work_item['id']

        if is_neurosymbolic:
            success = run_ns_test(
                work_item=work_item,
                simulator_url=args.simulator_url,
                results_dir=results_dir,
                client=client,
                worker_id=worker_id,
                ns_runner=ns_runner,
                engine=engine,
                rate_limiter=rate_limiter,
            )
        elif is_experience:
            success = run_experience_test(
                work_item=work_item,
                simulator_url=args.simulator_url,
                results_dir=results_dir,
                client=client,
                worker_id=worker_id,
                pipeline_runner=pipeline_runner,
                engine=engine,
                config_name=config_name,
                rate_limiter=rate_limiter,
            )
        else:
            config_params = ABLATION_CONFIGS_NO_EXPERIENCE[config_name]
            success = run_standard_test(
                work_item=work_item,
                simulator_url=args.simulator_url,
                results_dir=results_dir,
                client=client,
                worker_id=worker_id,
                config_name=config_name,
                config_params=config_params,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                rate_limiter=rate_limiter,
            )

        mark_work_complete(work_dir, item_id, success)
        experiments_run += 1

    # Save final experience store stats
    if engine is not None:
        stats_file = results_dir / f"experience_stats_worker{worker_id}.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "worker_id": worker_id,
                "config_name": config_name,
                "final_store_size": engine.size(),
                "engine_stats": engine.stats(),
                "experiments_run": experiments_run,
            }, f, indent=2)

    print(f"[Worker {worker_id}] Shutting down.")


if __name__ == "__main__":
    main()
