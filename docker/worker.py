#!/usr/bin/env python3
"""
Worker script that processes experiments from a shared work queue.
Runs inside each Docker container.

Supports two modes:
- "ablation": Runs ablation study experiments (default)
- "homebench": Runs HomeBench test cases with full evaluation metrics

Features:
- Atomic work queue claiming with file locking
- Exponential backoff retry for API rate limits
- Graceful error handling for transient failures
"""

import argparse
import fcntl
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Literal, Optional

import httpx

# Add project root to path
sys.path.insert(0, "/app")

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()


# ============================================================================
# Retry Configuration
# ============================================================================

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 120.0  # seconds
JITTER_FACTOR = 0.5  # Add randomness to prevent thundering herd

# ============================================================================
# Rate Limiting Configuration (Proactive TPM/RPM management)
# ============================================================================

# Estimated tokens per experiment (conservative estimate)
ESTIMATED_TOKENS_PER_EXPERIMENT = 10000  # ~10K tokens per experiment

# Default limits (Tier 1 OpenAI limits for GPT-5-nano)
DEFAULT_TPM_LIMIT = 500000  # 500K tokens per minute
DEFAULT_RPM_LIMIT = 500  # 500 requests per minute


class DistributedRateLimiter:
    """
    File-based distributed rate limiter for coordinating across Docker containers.
    Uses a shared file to track request timestamps and enforce global limits.
    """

    def __init__(
        self,
        work_dir: Path,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
    ):
        self.work_dir = Path(work_dir)
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rate_file = self.work_dir / "rate_limit.json"
        self.lock_file = self.work_dir / "rate_limit.lock"
        self.lock_file.touch(exist_ok=True)

    def _load_state(self) -> dict:
        """Load rate limit state from file."""
        if not self.rate_file.exists():
            return {"requests": [], "tokens": []}
        try:
            with open(self.rate_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"requests": [], "tokens": []}

    def _save_state(self, state: dict):
        """Save rate limit state to file."""
        with open(self.rate_file, "w") as f:
            json.dump(state, f)

    def _clean_old_entries(
        self, entries: list, window_seconds: float = 60.0
    ) -> list:
        """Remove entries older than the window."""
        cutoff = time.time() - window_seconds
        return [e for e in entries if e["timestamp"] > cutoff]

    def wait_for_capacity(
        self,
        estimated_tokens: int = ESTIMATED_TOKENS_PER_EXPERIMENT,
        worker_id: int = 0,
    ) -> float:
        """
        Wait until there's capacity for another request.
        Returns the time waited in seconds.
        """
        wait_start = time.time()
        max_wait = 300  # Max 5 minutes wait

        while True:
            with open(self.lock_file, "r+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    state = self._load_state()

                    # Clean old entries (older than 60 seconds)
                    state["requests"] = self._clean_old_entries(
                        state["requests"]
                    )
                    state["tokens"] = self._clean_old_entries(state["tokens"])

                    # Calculate current usage
                    current_rpm = len(state["requests"])
                    current_tpm = sum(e["tokens"] for e in state["tokens"])

                    # Check if we have capacity
                    rpm_ok = current_rpm < self.rpm_limit
                    tpm_ok = (current_tpm + estimated_tokens) <= self.tpm_limit

                    if rpm_ok and tpm_ok:
                        # Record this request
                        now = time.time()
                        state["requests"].append(
                            {"timestamp": now, "worker": worker_id}
                        )
                        state["tokens"].append(
                            {
                                "timestamp": now,
                                "tokens": estimated_tokens,
                                "worker": worker_id,
                            }
                        )
                        self._save_state(state)

                        waited = time.time() - wait_start
                        if waited > 1:
                            print(
                                f"[Worker {worker_id}] Rate limit: waited {waited:.1f}s for capacity"
                            )
                        return waited

                    # Calculate how long to wait
                    if not rpm_ok:
                        oldest_request = min(
                            e["timestamp"] for e in state["requests"]
                        )
                        wait_time = 60 - (time.time() - oldest_request) + 0.5
                    else:
                        oldest_token = min(
                            e["timestamp"] for e in state["tokens"]
                        )
                        wait_time = 60 - (time.time() - oldest_token) + 0.5

                    wait_time = max(0.5, min(wait_time, 10))  # Clamp to 0.5-10s

                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

            # Check if we've waited too long
            if time.time() - wait_start > max_wait:
                print(
                    f"[Worker {worker_id}] Warning: Rate limit wait exceeded {max_wait}s, proceeding anyway"
                )
                return time.time() - wait_start

            # Wait and retry
            time.sleep(wait_time)

    def record_actual_tokens(self, actual_tokens: int, worker_id: int = 0):
        """
        Update the token count with actual usage (call after experiment completes).
        This helps refine the estimate for better rate limiting.
        """
        with open(self.lock_file, "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                state = self._load_state()
                # Find and update the most recent entry from this worker
                for entry in reversed(state["tokens"]):
                    if entry.get("worker") == worker_id:
                        entry["tokens"] = actual_tokens
                        break
                self._save_state(state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def retry_with_exponential_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    retryable_exceptions: tuple = (
        RateLimitError,
        APIError,
        APIConnectionError,
        APITimeoutError,
    ),
):
    """
    Decorator that retries a function with exponential backoff.

    Handles:
    - RateLimitError (429): Too many requests
    - APIError (5xx): Server errors
    - APIConnectionError: Network issues
    - APITimeoutError: Request timeouts
    """

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
                        print(
                            f"[Retry] Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Calculate delay with exponential backoff + jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = delay * JITTER_FACTOR * random.random()
                    sleep_time = delay + jitter

                    error_type = type(e).__name__
                    print(
                        f"[Retry] {error_type} on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {sleep_time:.1f}s..."
                    )

                    # For rate limits, check if we have a retry-after header
                    if isinstance(e, RateLimitError) and hasattr(e, "response"):
                        retry_after = e.response.headers.get("retry-after")
                        if retry_after:
                            sleep_time = max(sleep_time, float(retry_after))
                            print(
                                f"[Retry] Server requested retry-after: {retry_after}s"
                            )

                    time.sleep(sleep_time)

            raise last_exception

        return wrapper

    return decorator


from src.config import (
    AffordanceConfig,
    DiscoveryConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    OutputConfig,
    PlanningConfig,
    ReasoningConfig,
    StateConfig,
    TracingConfig,
)
from src.discovery import create_discovery_pipeline
from src.execution import DirectAgentExecutor
from src.runner import run_experiment

# Ablation configurations (same as run_homebench_ablation.py)
ABLATION_CONFIGS = {
    "baseline": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "none",
    },
    "baseline_with_state": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "reflection": {
        "reasoning_enabled": True,
        "reasoning_strategy": "reflection",
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "multi_turn": {
        "reasoning_enabled": True,
        "reasoning_strategy": "multi_turn",
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "agentic": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },
    "agentic_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },
    "fully_agentic": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },
    "fully_agentic_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },
    "python_direct": {
        "reasoning_enabled": False,
        "output_format": "python_code",
        "state_strategy": "none",
    },
    "python_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code",
        "state_strategy": "all",
    },
    "python_unconstrained": {
        "reasoning_enabled": False,
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },
    "python_unconstrained_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },
}


def create_config(
    name: str,
    affordance_strategy: str = "exhaustive",
    state_strategy: str = "none",
    reasoning_enabled: bool = False,
    reasoning_strategy: str = "chain_of_thought",
    output_format: str = "json_ir",
    prompt_strategy: str = "detailed",
    model: str = "gpt-4o",
) -> ExperimentConfig:
    """Create an experiment config."""
    return ExperimentConfig(
        experiment=ExperimentMeta(
            name=name, description=f"HomeBench Ablation: {name}"
        ),
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
        model=ModelConfig(name=model, temperature=0.0),
        tracing=TracingConfig(
            enabled=True, output_dir="traces/", verbose=False
        ),
    )


def evaluate_result(result: dict, expected_output: list[dict]) -> dict:
    """Evaluate experiment result against expected output."""
    expected_actions = []
    for out in expected_output:
        if out.get("execution") == "success":
            expected_actions.append(
                {
                    "affordance": out["affordance"],
                    "params": out.get("params", {}),
                    "test": out.get("test", {}),
                }
            )

    executed_actions = []
    if result.get("execution") and result["execution"].get("action_history"):
        for action in result["execution"]["action_history"]:
            executed_actions.append(
                {
                    "affordance": action.get("affordance", ""),
                    "params": action.get("params", {}),
                }
            )

    correct = 0
    for expected in expected_actions:
        for executed in executed_actions:
            if expected["affordance"] == executed["affordance"]:
                if expected["params"] == executed["params"]:
                    correct += 1
                    break

    total_expected = len(expected_actions)
    total_executed = len(executed_actions)

    return {
        "expected_actions": expected_actions,
        "executed_actions": executed_actions,
        "correct": correct,
        "total_expected": total_expected,
        "total_executed": total_executed,
        "precision": correct / total_executed if total_executed > 0 else 0.0,
        "recall": correct / total_expected if total_expected > 0 else 0.0,
    }


def claim_work_item(work_dir: Path, worker_id: int) -> dict | None:
    """
    Atomically claim a work item from the queue.
    Uses file locking to prevent race conditions.
    Returns the work item or None if no work available.
    """
    queue_file = work_dir / "queue.json"
    lock_file = work_dir / "queue.lock"

    # Create lock file if it doesn't exist
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as lock:
        # Acquire exclusive lock
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        try:
            if not queue_file.exists():
                return None

            with open(queue_file, "r") as f:
                queue = json.load(f)

            # Find first unclaimed item
            for item in queue:
                if item.get("status") == "pending":
                    # Claim it
                    item["status"] = "running"
                    item["worker_id"] = worker_id
                    item["started_at"] = datetime.now().isoformat()

                    # Write back
                    with open(queue_file, "w") as f:
                        json.dump(queue, f, indent=2)

                    return item

            return None  # No pending items

        finally:
            # Release lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def mark_work_complete(work_dir: Path, item_id: str, success: bool):
    """Mark a work item as complete."""
    queue_file = work_dir / "queue.json"
    lock_file = work_dir / "queue.lock"

    with open(lock_file, "r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        try:
            with open(queue_file, "r") as f:
                queue = json.load(f)

            for item in queue:
                if item.get("id") == item_id:
                    item["status"] = "completed" if success else "failed"
                    item["completed_at"] = datetime.now().isoformat()
                    break

            with open(queue_file, "w") as f:
                json.dump(queue, f, indent=2)

        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def reset_simulator(simulator_url: str, home_id: str) -> bool:
    """Reset the simulator home to initial state."""
    try:
        # Try the /reset endpoint first
        response = httpx.post(
            f"{simulator_url}/reset", json={"home": str(home_id)}, timeout=30.0
        )
        if response.status_code == 200:
            return True
        # Fallback to workspace-specific reset
        response = httpx.post(
            f"{simulator_url}/workspaces/home{home_id}/reset", timeout=30.0
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Failed to reset home{home_id}: {e}")
        return False


# ============================================================================
# HomeBench Test Execution (homebench mode)
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
            "duration": self.duration_seconds,
            "failure_type": self.failure_type,
            "error": self.error,
            "trace": self.raw_result,
        }


def extract_actions_from_plan(plan: dict) -> list[str]:
    """Extract action URLs from a plan."""
    actions = []
    if not plan:
        return actions

    content = plan.get("content", plan)
    if isinstance(content, str):
        # Python code - extract URLs
        import re

        urls = re.findall(r'http://[^"\'>\s]+', content)
        actions = [u for u in urls if "/properties/" not in u]
    else:
        # JSON IR - traverse tree
        _extract_actions_recursive(content, actions)

    return actions


def _extract_actions_recursive(node: dict, actions: list):
    """Recursively extract action URLs from JSON IR tree."""
    if not isinstance(node, dict):
        return

    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            actions.append(url)

    for child in node.get("children", []):
        _extract_actions_recursive(child, actions)


def extract_params_from_plan(plan: dict) -> dict:
    """Extract parameters for each action from plan."""
    params = {}
    content = plan.get("content", plan)
    if isinstance(content, dict):
        _extract_params_recursive(content, params)
    return params


def _extract_params_recursive(node: dict, params: dict):
    """Recursively extract action parameters from JSON IR tree."""
    if not isinstance(node, dict):
        return

    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            params[url] = node.get("parameters", {})

    for child in node.get("children", []):
        _extract_params_recursive(child, params)


def verify_property(
    simulator_url: str, property_url: str, expected_value
) -> tuple[bool, any]:
    """Verify a property matches expected value."""
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
    """Classify the type of failure for debugging and analysis."""
    error_msg = exp_result.get("error", "") or ""
    planning = exp_result.get("planning", {})
    execution = exp_result.get("execution", {})

    if "JSON parse error" in error_msg or "JSON decode" in error_msg:
        return "parse_error"

    plan_explanation = planning.get("plan", {}).get("explanation", "")
    if "parse error" in plan_explanation.lower() or (
        "json" in plan_explanation.lower()
        and "error" in plan_explanation.lower()
    ):
        return "parse_error"

    exec_error = execution.get("error", "") or ""
    if "Unknown node type" in exec_error or "Compilation" in error_msg:
        return "compilation_error"
    if "KeyError" in exec_error or "missing" in exec_error.lower():
        return "compilation_error"

    if result.plan_generated and not result.execution_success:
        return "execution_error"

    if (
        result.execution_success
        and result.properties_checked > 0
        and result.properties_matched < result.properties_checked
    ):
        return "property_mismatch"

    if not result.plan_generated:
        if "parse" in error_msg.lower() or "json" in error_msg.lower():
            return "parse_error"
        return "other"

    return "other"


def run_homebench_test(
    work_item: dict,
    simulator_url: str,
    results_dir: Path,
    client: OpenAI,
    worker_id: int,
    rate_limiter: Optional[DistributedRateLimiter] = None,
) -> bool:
    """Run a single HomeBench test case with full evaluation metrics."""
    test_id = work_item["test_id"]
    home_id = work_item["home_id"]
    goal = work_item["input"]
    expected_outputs = work_item["output"]
    config = work_item.get("config", {})

    # Parse expected outputs
    expected_successes = [
        o for o in expected_outputs if o.get("execution") == "success"
    ]
    expected_errors = [
        o for o in expected_outputs if o.get("execution") == "error_input"
    ]

    result = HomeBenchTestResult(test_id=test_id)
    result.expected_actions = [
        o.get("affordance", "")
        for o in expected_successes
        if o.get("affordance")
    ]
    result.expected_params = {
        o.get("affordance"): o.get("params", {})
        for o in expected_successes
        if o.get("affordance")
    }
    result.expected_impossible = len(expected_errors)
    result.is_error_input_only = (
        len(expected_successes) == 0 and len(expected_errors) > 0
    )

    entry_point = f"{simulator_url}/workspaces/home{home_id}#workspace"

    # Wait for rate limit capacity
    if rate_limiter:
        rate_limiter.wait_for_capacity(
            estimated_tokens=ESTIMATED_TOKENS_PER_EXPERIMENT,
            worker_id=worker_id,
        )

    print(f"[Worker {worker_id}] Running HomeBench test: {test_id}")

    start_time = datetime.now()

    try:
        # Reset home to initial state
        reset_simulator(simulator_url, home_id)

        # Get config parameters
        model = config.get("model", "gpt-4o")
        execution_mode = config.get("execution_mode", "behavior_tree")

        if execution_mode == "direct_agent":
            # =====================================================
            # Direct Agent Mode - LLM makes tool calls directly
            # =====================================================
            exp_config = create_config(
                name=f"homebench_{test_id}",
                affordance_strategy=config.get(
                    "discovery_affordances", "agentic"
                ),
                state_strategy=config.get("discovery_state", "relevant"),
                reasoning_enabled=False,
                output_format="json_ir",  # Not used in direct agent mode
                prompt_strategy=config.get("prompt_strategy", "detailed"),
                model=model,
            )

            # Run discovery phase only
            discovery_pipeline = create_discovery_pipeline(
                config=exp_config.discovery,
                client=client,
                model_config=exp_config.model,
            )
            discovery_result = discovery_pipeline.discover(entry_point, goal)

            # Create direct agent executor
            executor = DirectAgentExecutor(
                client=client,
                model_config=exp_config.model,
                max_iterations=20,
            )

            # Execute with direct agent
            agent_result = executor.execute(
                goal=goal,
                affordances=discovery_result.affordances,
                state=discovery_result.state,
            )

            # Map results
            result.plan_generated = (
                True  # Direct agent doesn't generate a "plan"
            )
            result.plan_format = "direct_agent"
            result.execution_success = agent_result.success
            result.execution_ticks = agent_result.iterations

            # Extract actions from agent execution
            result.actions_in_plan = [
                a["url"] for a in agent_result.actions_executed
            ]
            result.params_in_plan = {
                a["url"]: a.get("params", {})
                for a in agent_result.actions_executed
            }
            result.detected_impossible = agent_result.impossible_reported

            # Store trace
            result.raw_result = {
                "discovery": discovery_result.to_dict(),
                "execution": {
                    "mode": "direct_agent",
                    "success": agent_result.success,
                    "iterations": agent_result.iterations,
                    "actions_executed": agent_result.actions_executed,
                    "properties_read": agent_result.properties_read,
                    "impossible_reported": agent_result.impossible_reported,
                    "summary": agent_result.summary,
                    "trace": agent_result.trace,
                },
            }
        else:
            # =====================================================
            # Behavior Tree Mode - Generate and execute BT plan
            # =====================================================
            reasoning_enabled = (
                config.get("planning_reasoning", "none") != "none"
            )

            exp_config = create_config(
                name=f"homebench_{test_id}",
                affordance_strategy=config.get(
                    "discovery_affordances", "agentic"
                ),
                state_strategy=config.get("discovery_state", "relevant"),
                reasoning_enabled=reasoning_enabled,
                reasoning_strategy=(
                    config.get("planning_reasoning", "chain_of_thought")
                    if reasoning_enabled
                    else "chain_of_thought"
                ),
                output_format=config.get("planning_output", "json_ir"),
                prompt_strategy=config.get("prompt_strategy", "detailed"),
                model=model,
            )

            # Run experiment
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
                    result.plan_format = planning.get("plan", {}).get(
                        "format", ""
                    )

                    plan_content = planning.get("plan", {}).get("content", {})
                    result.actions_in_plan = extract_actions_from_plan(
                        {"content": plan_content}
                    )
                    result.params_in_plan = extract_params_from_plan(
                        {"content": plan_content}
                    )

                    detected = planning.get("plan", {}).get(
                        "detected_impossible", []
                    )
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

        # Verify properties (only if execution succeeded and not error_input case)
        if result.execution_success and not result.is_error_input_only:
            for expected in expected_successes:
                test_spec = expected.get("test", {})
                if test_spec:
                    prop_url = test_spec.get("property")
                    exp_val = test_spec.get("expected_value")
                    if prop_url:
                        matched, actual = verify_property(
                            simulator_url, prop_url, exp_val
                        )
                        result.properties_checked += 1
                        if matched:
                            result.properties_matched += 1
                        result.property_results.append(
                            {
                                "property": prop_url,
                                "expected": exp_val,
                                "actual": actual,
                                "matched": matched,
                            }
                        )

        # Determine overall success
        if result.is_error_input_only:
            detected_as_impossible = (
                len(result.detected_impossible) > 0
                or not result.plan_generated
                or len(result.actions_in_plan) == 0
            )
            result.success = "True" if detected_as_impossible else "False"
        else:
            is_plan_executed = (
                result.plan_generated and result.execution_success
            )

            if not is_plan_executed:
                result.success = "False"
            else:
                no_extra_actions = len(result.extra_actions) == 0
                all_properties_matched = (
                    result.properties_matched == result.properties_checked
                )

                result.handled_correctly = (
                    no_extra_actions and all_properties_matched
                )

                if result.handled_correctly:
                    result.success = "True"
                elif result.matched_actions:
                    result.success = "Quantifiable"
                else:
                    result.success = "False"

        # Classify failure type
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
    result_file = results_dir / f"{test_id}.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    status_icon = (
        "✓"
        if result.success == "True"
        else ("◐" if result.success == "Quantifiable" else "✗")
    )
    print(
        f"[Worker {worker_id}] {status_icon} {test_id}: {result.success} "
        f"(matched: {len(result.matched_actions)}/{len(result.expected_actions)}, "
        f"props: {result.properties_matched}/{result.properties_checked})"
    )

    return result.success != "False"


@retry_with_exponential_backoff(max_retries=MAX_RETRIES)
def run_experiment_with_retry(
    config, goal: str, entry_point: str, client: OpenAI
) -> dict:
    """
    Wrapper around run_experiment with automatic retry for API errors.
    """
    return run_experiment(
        config=config,
        goal=goal,
        entry_point=entry_point,
        client=client,
    )


def run_single_experiment(
    work_item: dict,
    simulator_url: str,
    results_dir: Path,
    client: OpenAI,
    model: str,
    worker_id: int,
    rate_limiter: DistributedRateLimiter | None = None,
) -> bool:
    """Run a single experiment and save results."""

    prompt_id = work_item["prompt_id"]
    config_name = work_item["config_name"]
    goal = work_item["goal"]
    expected = work_item["expected_output"]
    home_id = work_item["home_id"]

    entry_point = f"{simulator_url}/workspaces/home{home_id}#workspace"

    # Wait for rate limit capacity before starting
    if rate_limiter:
        rate_limiter.wait_for_capacity(
            estimated_tokens=ESTIMATED_TOKENS_PER_EXPERIMENT,
            worker_id=worker_id,
        )

    print(f"[Worker {worker_id}] Running: {prompt_id} / {config_name}")

    try:
        # Reset simulator state for this home
        reset_simulator(simulator_url, home_id)

        # Create config
        config_params = ABLATION_CONFIGS[config_name]
        config = create_config(
            name=config_name,
            model=model,
            **config_params,
        )

        # Run experiment with automatic retry for transient API errors
        result = run_experiment_with_retry(
            config=config,
            goal=goal,
            entry_point=entry_point,
            client=client,
        )

        # Evaluate
        evaluation = evaluate_result(result, expected)

        # Add metadata
        result["prompt_id"] = prompt_id
        result["config_name"] = config_name
        result["goal"] = goal
        result["expected_output"] = expected
        result["evaluation"] = evaluation
        result["worker_id"] = worker_id

        # Save result
        result_file = results_dir / f"{prompt_id}_{config_name}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        status = "SUCCESS" if result.get("success") else "FAILED"
        print(
            f"[Worker {worker_id}] {status}: {prompt_id} / {config_name} "
            f"(correct: {evaluation['correct']}/{evaluation['total_expected']})"
        )

        return True

    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: {prompt_id} / {config_name}: {e}")

        # Save error result
        error_result = {
            "prompt_id": prompt_id,
            "config_name": config_name,
            "goal": goal,
            "success": False,
            "error": str(e),
            "worker_id": worker_id,
        }

        result_file = results_dir / f"{prompt_id}_{config_name}.json"
        with open(result_file, "w") as f:
            json.dump(error_result, f, indent=2)

        return False


def main():
    parser = argparse.ArgumentParser(description="Experiment worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--simulator-url", default="http://localhost:8080")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=DEFAULT_RPM_LIMIT,
        help=f"Requests per minute limit (default: {DEFAULT_RPM_LIMIT})",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=DEFAULT_TPM_LIMIT,
        help=f"Tokens per minute limit (default: {DEFAULT_TPM_LIMIT})",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable proactive rate limiting",
    )
    parser.add_argument(
        "--mode",
        choices=["ablation", "homebench"],
        default="ablation",
        help="Worker mode: ablation (default) or homebench",
    )

    args = parser.parse_args()

    # Also check environment variable for mode (set by orchestrator)
    mode = os.environ.get("WORKER_MODE", args.mode)

    worker_id = args.worker_id
    work_dir = args.work_dir
    results_dir = args.results_dir

    # Ensure results dir exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"[Worker {worker_id}] ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Setup rate limiter (shared across all workers via file)
    rate_limiter = None
    if not args.no_rate_limit:
        rate_limiter = DistributedRateLimiter(
            work_dir=work_dir,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
        )
        print(
            f"[Worker {worker_id}] Rate limiting enabled: "
            f"{args.rpm_limit} RPM, {args.tpm_limit} TPM"
        )

    print(f"[Worker {worker_id}] Ready in {mode} mode, looking for work...")

    # Process work items until none left
    experiments_run = 0
    while True:
        work_item = claim_work_item(work_dir, worker_id)

        if work_item is None:
            # No more work
            print(
                f"[Worker {worker_id}] No more work available. "
                f"Completed {experiments_run} experiments."
            )
            break

        item_id = work_item["id"]

        # Choose runner based on mode
        if mode == "homebench":
            success = run_homebench_test(
                work_item=work_item,
                simulator_url=args.simulator_url,
                results_dir=results_dir,
                client=client,
                worker_id=worker_id,
                rate_limiter=rate_limiter,
            )
        else:
            success = run_single_experiment(
                work_item=work_item,
                simulator_url=args.simulator_url,
                results_dir=results_dir,
                client=client,
                model=args.model,
                worker_id=worker_id,
                rate_limiter=rate_limiter,
            )

        mark_work_complete(work_dir, item_id, success)
        experiments_run += 1

    print(f"[Worker {worker_id}] Shutting down.")


if __name__ == "__main__":
    main()
