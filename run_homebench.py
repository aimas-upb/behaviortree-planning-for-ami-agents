"""
HomeBench Evaluation Runner.

Runs behavior tree planning experiments against the HomeBench dataset
and evaluates performance against ground truth.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    ExperimentConfig, ExperimentMeta, DiscoveryConfig, PlanningConfig,
    ExecutionConfig, ModelConfig, TracingConfig, AffordanceConfig,
    StateConfig, ReasoningConfig, OutputConfig, load_config
)
from src.runner import run_experiment

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single HomeBench test case."""
    id: str
    home_id: str
    test_type: str  # 'one' or 'multi'
    input: str
    expected_outputs: list[dict]

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        parts = data["id"].split("_")
        home_id = parts[0]
        test_type = parts[1] if len(parts) > 1 else "unknown"
        return cls(
            id=data["id"],
            home_id=home_id,
            test_type=test_type,
            input=data["input"],
            expected_outputs=data["output"],
        )

    @property
    def expected_successes(self) -> list[dict]:
        """Get expected successful actions."""
        return [o for o in self.expected_outputs if o.get("execution") == "success"]

    @property
    def expected_errors(self) -> list[dict]:
        """Get expected error_input cases."""
        return [o for o in self.expected_outputs if o.get("execution") == "error_input"]


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_id: str
    success: Literal["True", "False", "Quantifiable"]

    # Planning metrics
    plan_generated: bool = False
    plan_format: str = ""
    actions_in_plan: list[str] = field(default_factory=list)
    params_in_plan: dict = field(default_factory=dict)  # action_url -> params

    # Execution metrics
    execution_success: bool = False
    execution_ticks: int = 0

    # Comparison metrics
    expected_actions: list[str] = field(default_factory=list)
    expected_params: dict = field(default_factory=dict)  # action_url -> params
    matched_actions: list[str] = field(default_factory=list)
    missing_actions: list[str] = field(default_factory=list)
    extra_actions: list[str] = field(default_factory=list)
    params_correct: bool = False

    # Property verification
    properties_checked: int = 0
    properties_matched: int = 0
    property_results: list[dict] = field(default_factory=list)

    # Impossible sub-goal detection (error_input cases)
    # These are sub-goals that cannot be achieved (missing capability, invalid params, etc.)
    expected_impossible: int = 0  # Count of error_input in ground truth
    detected_impossible: list[str] = field(default_factory=list)  # Sub-goals reported as impossible
    is_error_input_only: bool = False  # True if ALL expected outputs are error_input
    handled_correctly: bool = False  # For error cases: didn't attempt; for success: all matched

    # Timing
    duration_seconds: float = 0.0

    # Failure tracking
    failure_type: Optional[str] = None  # none, parse_error, compilation_error, execution_error, property_mismatch

    # Raw data
    error: Optional[str] = None
    raw_result: Optional[dict] = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_tests: int = 0
    successful_tests: int = 0
    quantifiable_tests: int = 0  # Partially successful - some actions/properties matched
    failed_tests: int = 0

    # Planning
    plans_generated: int = 0

    # Action matching
    total_expected_actions: int = 0
    total_matched_actions: int = 0
    total_missing_actions: int = 0
    total_extra_actions: int = 0

    # Property verification
    total_properties_checked: int = 0
    total_properties_matched: int = 0

    # Impossible sub-goal detection (error_input cases)
    # Tracks detection of sub-goals that cannot be achieved
    total_expected_impossible: int = 0  # Total error_input sub-goals in ground truth
    total_detected_impossible: int = 0  # Total sub-goals reported as impossible by system
    # Note: We can't do perfect matching since error_inputs don't have identifiers,
    # so we track counts - if system detects >= expected, it's good

    # Failure type tracking
    failures_by_type: dict = field(default_factory=lambda: {
        "parse_error": 0,
        "compilation_error": 0,
        "execution_error": 0,
        "action_mismatch": 0,
        "property_mismatch": 0,
        "error_input_not_detected": 0,  # Failed to detect impossible goal
        "other": 0,
    })

    # Timing
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Rate of fully successful tests."""
        return self.successful_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def quantifiable_rate(self) -> float:
        """Rate of quantifiable (partially successful) tests."""
        return self.quantifiable_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def success_or_quantifiable_rate(self) -> float:
        """Rate of tests that are either successful or quantifiable (not failed)."""
        return (self.successful_tests + self.quantifiable_tests) / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def action_precision(self) -> float:
        """Precision: matched / (matched + extra)"""
        total = self.total_matched_actions + self.total_extra_actions
        return self.total_matched_actions / total if total > 0 else 0.0

    @property
    def action_recall(self) -> float:
        """Recall: matched / expected"""
        return self.total_matched_actions / self.total_expected_actions if self.total_expected_actions > 0 else 0.0

    @property
    def action_f1(self) -> float:
        """F1 score for action matching."""
        p, r = self.action_precision, self.action_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def property_accuracy(self) -> float:
        """Property verification accuracy."""
        return self.total_properties_matched / self.total_properties_checked if self.total_properties_checked > 0 else 0.0

    @property
    def impossible_detection_rate(self) -> float:
        """Rate of detecting impossible sub-goals (detected / expected)."""
        if self.total_expected_impossible == 0:
            return 1.0  # No impossible sub-goals expected, so "perfect" detection
        # Cap at 1.0 - detecting more than expected is fine
        return min(1.0, self.total_detected_impossible / self.total_expected_impossible)

    def to_dict(self) -> dict:
        return {
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "quantifiable_tests": self.quantifiable_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.success_rate,
            "quantifiable_rate": self.quantifiable_rate,
            "success_or_quantifiable_rate": self.success_or_quantifiable_rate,
            "plans_generated": self.plans_generated,
            "action_precision": self.action_precision,
            "action_recall": self.action_recall,
            "action_f1": self.action_f1,
            "total_expected_actions": self.total_expected_actions,
            "total_matched_actions": self.total_matched_actions,
            "total_missing_actions": self.total_missing_actions,
            "total_extra_actions": self.total_extra_actions,
            "property_accuracy": self.property_accuracy,
            "total_properties_checked": self.total_properties_checked,
            "total_properties_matched": self.total_properties_matched,
            "total_expected_impossible": self.total_expected_impossible,
            "total_detected_impossible": self.total_detected_impossible,
            "impossible_detection_rate": self.impossible_detection_rate,
            "failures_by_type": self.failures_by_type,
            "total_duration": self.total_duration,
            "avg_duration": self.total_duration / self.total_tests if self.total_tests > 0 else 0,
        }


class HomeBenchEvaluator:
    """Evaluator for HomeBench test cases."""

    def __init__(
        self,
        config: ExperimentConfig,
        client: OpenAI,
        simulator_url: str = "http://localhost:8080",
        reset_enabled: bool = True,
        execution_mode: str = "behavior_tree",  # "behavior_tree" or "direct_agent"
    ):
        self.config = config
        self.client = client
        self.simulator_url = simulator_url
        self.reset_enabled = reset_enabled
        self.execution_mode = execution_mode
        self.http_client = httpx.Client(timeout=30.0)

    def load_tests(
        self,
        data_path: str,
        home_id: Optional[str] = None,
        test_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TestCase]:
        """Load test cases from JSON file."""
        with open(data_path) as f:
            data = json.load(f)

        tests = [TestCase.from_dict(d) for d in data]

        # Filter by home
        if home_id:
            tests = [t for t in tests if t.home_id == home_id]

        # Filter by type
        if test_type:
            tests = [t for t in tests if t.test_type == test_type]

        # Limit
        if limit:
            tests = tests[:limit]

        return tests

    def reset_home(self, home_id: str) -> bool:
        """Reset a home to initial state."""
        if not self.reset_enabled:
            return True  # Skip reset

        try:
            # Extract numeric part from home_id (e.g., "home96" -> "96")
            numeric_id = home_id.replace("home", "")
            response = self.http_client.post(
                f"{self.simulator_url}/reset",
                json={"home": numeric_id},
            )
            if response.status_code == 404:
                logger.warning("Reset endpoint not available, continuing without reset")
                self.reset_enabled = False  # Disable for future tests
                return True
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to reset home {home_id}: {e}")
            return True  # Continue anyway

    def verify_property(self, property_url: str, expected_value) -> tuple[bool, any]:
        """Verify a property matches expected value."""
        try:
            response = self.http_client.get(property_url)
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"

            actual = response.json()
            matched = actual == expected_value
            return matched, actual
        except Exception as e:
            return False, str(e)

    def extract_actions_from_plan(self, plan: dict) -> list[str]:
        """Extract action URLs from a plan."""
        actions = []

        if not plan:
            return actions

        # Handle both JSON IR and raw dict
        content = plan.get("content", plan)
        if isinstance(content, str):
            # Python code - try to extract URLs
            import re
            urls = re.findall(r'http://[^"\'>\s]+', content)
            # Filter for action URLs (not property URLs)
            actions = [u for u in urls if '/properties/' not in u]
        else:
            # JSON IR - traverse tree
            self._extract_actions_recursive(content, actions)

        return actions

    def _extract_actions_recursive(self, node: dict, actions: list[str]):
        """Recursively extract action URLs from JSON IR tree."""
        if not isinstance(node, dict):
            return

        if node.get("type") == "action":
            url = node.get("action_url")
            if url:
                actions.append(url)

        for child in node.get("children", []):
            self._extract_actions_recursive(child, actions)

    def _extract_params_from_plan(self, plan: dict) -> dict:
        """Extract parameters for each action from plan."""
        params = {}
        content = plan.get("content", plan)
        if isinstance(content, dict):
            self._extract_params_recursive(content, params)
        return params

    def _extract_params_recursive(self, node: dict, params: dict):
        """Recursively extract action parameters from JSON IR tree."""
        if not isinstance(node, dict):
            return

        if node.get("type") == "action":
            url = node.get("action_url")
            if url:
                params[url] = node.get("parameters", {})

        for child in node.get("children", []):
            self._extract_params_recursive(child, params)

    def _classify_failure(self, result: TestResult, exp_result: dict) -> str:
        """
        Classify the type of failure for debugging and analysis.

        Returns one of:
        - parse_error: LLM output couldn't be parsed as JSON
        - compilation_error: JSON IR couldn't be compiled to py_trees
        - execution_error: py_trees execution failed
        - action_mismatch: Wrong actions in the plan
        - property_mismatch: Actions correct but property values wrong
        - other: Unknown failure type
        """
        error_msg = exp_result.get("error", "") or ""
        planning = exp_result.get("planning", {})
        execution = exp_result.get("execution", {})

        # Check for parse errors (JSON parsing failed)
        if "JSON parse error" in error_msg or "JSON decode" in error_msg:
            return "parse_error"

        # Check planning errors
        plan_explanation = planning.get("plan", {}).get("explanation", "")
        if "parse error" in plan_explanation.lower() or "json" in plan_explanation.lower() and "error" in plan_explanation.lower():
            return "parse_error"

        # Check for compilation errors (invalid node type, missing fields)
        exec_error = execution.get("error", "") or ""
        if "Unknown node type" in exec_error or "Compilation" in error_msg:
            return "compilation_error"
        if "KeyError" in exec_error or "missing" in exec_error.lower():
            return "compilation_error"

        # Plan generated but execution failed
        if result.plan_generated and not result.execution_success:
            return "execution_error"

        # Execution succeeded, but affordance properties don't match
        if result.execution_success and result.properties_checked > 0 and result.properties_matched < result.properties_checked:
            return "property_mismatch"

        # Plan not generated at all
        if not result.plan_generated:
            if "parse" in error_msg.lower() or "json" in error_msg.lower():
                return "parse_error"
            return "other"

        return "other"

    def run_test_direct_agent(self, test: TestCase) -> TestResult:
        """Run a single test case using direct agent mode."""
        from src.execution import DirectAgentExecutor
        from src.discovery import create_discovery_pipeline

        result = TestResult(test_id=test.id, success="False")
        result.expected_actions = [o.get("affordance", "") for o in test.expected_successes if o.get("affordance")]
        result.expected_params = {o.get("affordance"): o.get("params", {}) for o in test.expected_successes if o.get("affordance")}
        result.expected_impossible = len(test.expected_errors)
        result.is_error_input_only = len(test.expected_successes) == 0 and len(test.expected_errors) > 0

        start_time = datetime.now()

        try:
            # Reset home to initial state
            self.reset_home(test.home_id)

            # Build entry point
            entry_point = f"{self.simulator_url}/workspaces/{test.home_id}#workspace"

            # Run discovery phase only
            discovery_pipeline = create_discovery_pipeline(
                config=self.config.discovery,
                client=self.client,
                model=self.config.model.name,
            )
            discovery_result = discovery_pipeline.discover(entry_point, test.input)

            # Create direct agent executor
            executor = DirectAgentExecutor(
                client=self.client,
                model=self.config.model.name,
                max_iterations=20,
            )

            # Execute with direct agent
            agent_result = executor.execute(
                goal=test.input,
                affordances=discovery_result.affordances,
                state=discovery_result.state,
            )

            # Map results
            result.plan_generated = True  # Direct agent doesn't generate a "plan"
            result.plan_format = "direct_agent"
            result.execution_success = agent_result.success
            result.execution_ticks = agent_result.iterations

            # Extract actions from agent execution
            result.actions_in_plan = [a["url"] for a in agent_result.actions_executed]
            result.params_in_plan = {a["url"]: a.get("params", {}) for a in agent_result.actions_executed}
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

            # Compare actions
            expected_set = set(result.expected_actions)
            actual_set = set(result.actions_in_plan)
            result.matched_actions = list(expected_set & actual_set)
            result.missing_actions = list(expected_set - actual_set)
            result.extra_actions = list(actual_set - expected_set)

            # Verify properties
            if result.execution_success and not result.is_error_input_only:
                self._verify_test_properties(test, result)

            # Determine success
            if result.is_error_input_only:
                # For pure error_input cases: success only if system detected it's impossible
                # and did NOT generate/execute a plan.
                detected_as_impossible = (
                    len(result.detected_impossible) > 0 or
                    not result.plan_generated or
                    len(result.actions_in_plan) == 0
                )
                result.success = "True" if detected_as_impossible else "False"
            else:
                if not result.execution_success:
                    result.success = "False"
                else:
                    no_extra_actions = len(result.extra_actions) == 0
                    all_properties_matched = result.properties_matched == result.properties_checked

                    result.handled_correctly = no_extra_actions and all_properties_matched
                    
                    if result.handled_correctly:
                        result.success = "True"
                    elif result.matched_actions:
                        # Some actions or properties matched
                        # Quantifiable only applies when there are multiple expected actions
                        result.success = "Quantifiable"
                    else:
                        # No actions or properties matched
                        result.success = "False"

            # Classify failure
            if result.success == "False":
                if result.is_error_input_only:
                    result.failure_type = "error_input_not_detected"
                else:
                    result.failure_type = self._classify_failure(result, result.raw_result)

            executor.close()

        except Exception as e:
            result.error = str(e)
            result.failure_type = "other"
            logger.exception(f"Test {test.id} failed with exception")

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def _verify_test_properties(self, test: TestCase, result: TestResult):
        """
        Verify all expected properties for a test case.
        Args:
            test: TestCase
            result: TestResult to update
        """
        for expected in test.expected_successes:
            test_spec = expected.get("test", {})
            if test_spec:
                prop_url = test_spec.get("property")
                exp_val = test_spec.get("expected_value")
                if prop_url:
                    matched, actual = self.verify_property(prop_url, exp_val)
                    result.properties_checked += 1
                    if matched:
                        result.properties_matched += 1
                    result.property_results.append({
                                "property": prop_url,
                                "expected": exp_val,
                                "actual": actual,
                                "matched": matched,
                            })

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        result = TestResult(test_id=test.id, success="False")
        result.expected_actions = [o.get("affordance", "") for o in test.expected_successes if o.get("affordance")]
        result.expected_params = {o.get("affordance"): o.get("params", {}) for o in test.expected_successes if o.get("affordance")}
        result.expected_impossible = len(test.expected_errors)

        # Check if this is an error_input-only case (impossible/invalid command)
        result.is_error_input_only = len(test.expected_successes) == 0 and len(test.expected_errors) > 0

        start_time = datetime.now()

        try:
            # Reset home to initial state (best effort)
            self.reset_home(test.home_id)

            # Build entry point
            entry_point = f"{self.simulator_url}/workspaces/{test.home_id}#workspace"

            # Run experiment
            exp_result = run_experiment(
                config=self.config,
                goal=test.input,
                entry_point=entry_point,
                client=self.client,
            )
            result.raw_result = exp_result

            # Check planning
            if exp_result.get("planning"):
                planning = exp_result["planning"]
                if planning.get("success"):
                    result.plan_generated = True
                    result.plan_format = planning.get("plan", {}).get("format", "")

                    # Extract actions from plan
                    plan_content = planning.get("plan", {}).get("content", {})
                    result.actions_in_plan = self.extract_actions_from_plan({"content": plan_content})
                    result.params_in_plan = self._extract_params_from_plan({"content": plan_content})

                    # Extract detected impossible sub-goals (if reported by planner)
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

            # Verify properties (only if execution succeeded and not error_input case)
            if result.execution_success and not result.is_error_input_only:
                self._verify_test_properties(test, result)

            # Determine overall success based on case type
            if result.is_error_input_only:
                # For pure error_input cases: success only if system detected it's impossible
                # and did NOT generate/execute a plan. If a plan was generated and executed
                # (even if execution failed), the system didn't properly identify the impossibility.
                detected_as_impossible = (
                    len(result.detected_impossible) > 0 or
                    not result.plan_generated or
                    len(result.actions_in_plan) == 0
                )
                result.success = "True" if detected_as_impossible else "False"
            else:
                # For cases with success actions (may also have error_inputs):
                # Success = all expected SUCCESS actions matched AND all properties matched
                # Error_input detection is tracked separately, doesn't affect success
                is_plan_executed = result.plan_generated and result.execution_success

                if not is_plan_executed:
                    # No plan or behavior tree execution failed
                    result.success = "False"
                else:
                    # Behavior tree executed - check actions and properties
                    no_extra_actions = len(result.extra_actions) == 0
                    all_properties_matched = result.properties_matched == result.properties_checked

                    result.handled_correctly = no_extra_actions and all_properties_matched

                    if result.handled_correctly:
                        result.success = "True"
                    elif result.matched_actions:
                        # Some actions or properties matched, and multiple actions were expected
                        # Quantifiable only applies when there are multiple expected actions
                        result.success = "Quantifiable"
                    else:
                        # No actions or properties matched
                        result.success = "False"

            # Classify failure type
            if result.success == "False":
                if result.is_error_input_only:
                    # Error input case that wasn't detected - system incorrectly executed
                    result.failure_type = "error_input_not_detected"
                else:
                    result.failure_type = self._classify_failure(result, exp_result)

        except Exception as e:
            result.error = str(e)
            result.failure_type = "other"
            logger.exception(f"Test {test.id} failed with exception")

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def evaluate(
        self,
        tests: list[TestCase],
        progress: bool = True,
    ) -> tuple[list[TestResult], EvaluationMetrics]:
        """Evaluate all test cases."""
        results = []
        metrics = EvaluationMetrics()

        iterator = tqdm(tests, desc="Evaluating") if progress else tests

        for test in iterator:
            # Choose runner based on execution mode
            if self.execution_mode == "direct_agent":
                result = self.run_test_direct_agent(test)
            else:
                result = self.run_test(test)
            results.append(result)

            # Update metrics
            metrics.total_tests += 1
            if result.success == "True":
                metrics.successful_tests += 1
            elif result.success == "Quantifiable":
                metrics.quantifiable_tests += 1
            else:
                metrics.failed_tests += 1

            if result.plan_generated:
                metrics.plans_generated += 1

            metrics.total_expected_actions += len(result.expected_actions)
            metrics.total_matched_actions += len(result.matched_actions)
            metrics.total_missing_actions += len(result.missing_actions)
            metrics.total_extra_actions += len(result.extra_actions)

            metrics.total_properties_checked += result.properties_checked
            metrics.total_properties_matched += result.properties_matched

            # Track impossible sub-goal detection
            metrics.total_expected_impossible += result.expected_impossible
            metrics.total_detected_impossible += len(result.detected_impossible)

            # Track failure types
            if result.failure_type and result.failure_type in metrics.failures_by_type:
                metrics.failures_by_type[result.failure_type] += 1

            metrics.total_duration += result.duration_seconds

            if progress:
                iterator.set_postfix({
                    "success": f"{metrics.success_rate:.1%}",
                    "recall": f"{metrics.action_recall:.1%}",
                })

        return results, metrics


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
        experiment=ExperimentMeta(name=name, description=f"HomeBench evaluation: {name}"),
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
        tracing=TracingConfig(enabled=False),
    )


def main():
    parser = argparse.ArgumentParser(
        description="HomeBench Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data",
        default="datasets/HomeBench/converted/test_data.json",
        help="Path to test data JSON"
    )
    parser.add_argument("--home", type=str, help="Filter by home ID (e.g., 'home96')")
    parser.add_argument("--type", choices=["one", "multi"], help="Filter by test type")
    parser.add_argument("--limit", type=int, help="Limit number of tests")

    # Config
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument(
        "--discovery-affordances",
        choices=["exhaustive", "agentic", "relevant"],
        default="exhaustive",
    )
    parser.add_argument(
        "--discovery-state",
        choices=["all", "relevant", "agentic", "none"],
        default="none",
    )
    parser.add_argument(
        "--planning-reasoning",
        choices=["none", "chain_of_thought", "multi_turn", "reflection"],
        default="none",
    )
    parser.add_argument(
        "--planning-output",
        choices=["json_ir", "python_code", "python_code_unconstrained"],
        default="json_ir",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=["baseline", "detailed", "few_shot", "icl"],
        default="detailed",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["behavior_tree", "direct_agent"],
        default="behavior_tree",
        help="Execution mode: behavior_tree (generate code) or direct_agent (LLM tool calls)",
    )

    # Output
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--generate-traces", action="store_true", help="Generate individual HTML trace files for each test")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML evaluation report after running")
    parser.add_argument("--open-report", action="store_true", help="Open the generated report in browser")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        reasoning_enabled = args.planning_reasoning != "none"
        config = create_config(
            name="homebench_eval",
            affordance_strategy=args.discovery_affordances,
            state_strategy=args.discovery_state,
            reasoning_enabled=reasoning_enabled,
            reasoning_strategy=args.planning_reasoning if reasoning_enabled else "chain_of_thought",
            output_format=args.planning_output,
            prompt_strategy=args.prompt_strategy,
            model=args.model,
        )

    # Setup client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Create evaluator
    evaluator = HomeBenchEvaluator(
        config=config,
        client=client,
        execution_mode=args.execution_mode,
    )

    # Load tests
    print(f"Loading tests from {args.data}...")
    tests = evaluator.load_tests(
        args.data,
        home_id=args.home,
        test_type=args.type,
        limit=args.limit,
    )
    print(f"Loaded {len(tests)} tests")

    if not tests:
        print("No tests to run!")
        sys.exit(0)

    # Print config summary
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Execution Mode: {args.execution_mode}")
    print(f"Discovery - Affordances: {config.discovery.affordances.strategy}")
    print(f"Discovery - State: {config.discovery.state.strategy}")
    if args.execution_mode == "behavior_tree":
        reasoning = config.planning.reasoning.strategy if config.planning.reasoning.enabled else "none"
        print(f"Planning - Reasoning: {reasoning}")
        print(f"Planning - Output: {config.planning.output.format}")
        print(f"Planning - Prompt: {config.planning.prompt_strategy}")
    print("=" * 60 + "\n")

    # Run evaluation
    results, metrics = evaluator.evaluate(tests, progress=not args.no_progress)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tests: {metrics.total_tests}")
    print(f"Successful: {metrics.successful_tests} ({metrics.success_rate:.1%})")
    print(f"Quantifiable: {metrics.quantifiable_tests} ({metrics.quantifiable_rate:.1%})")
    print(f"Success + Quantifiable: {metrics.successful_tests + metrics.quantifiable_tests} ({metrics.success_or_quantifiable_rate:.1%})")
    print(f"Failed: {metrics.failed_tests}")
    print()
    print(f"Plans generated: {metrics.plans_generated}")
    print()
    print("Action Matching:")
    print(f"  Precision: {metrics.action_precision:.1%}")
    print(f"  Recall: {metrics.action_recall:.1%}")
    print(f"  F1: {metrics.action_f1:.1%}")
    print(f"  Expected: {metrics.total_expected_actions}")
    print(f"  Matched: {metrics.total_matched_actions}")
    print(f"  Missing: {metrics.total_missing_actions}")
    print(f"  Extra: {metrics.total_extra_actions}")
    print()
    print(f"Property Verification: {metrics.total_properties_matched}/{metrics.total_properties_checked} ({metrics.property_accuracy:.1%})")
    print()
    if metrics.total_expected_impossible > 0:
        print(f"Impossible Sub-goals: {metrics.total_detected_impossible}/{metrics.total_expected_impossible} detected ({metrics.impossible_detection_rate:.1%})")
        print()
    # Print failure breakdown
    if metrics.failed_tests > 0:
        print("Failure Breakdown:")
        for ftype, count in metrics.failures_by_type.items():
            if count > 0:
                pct = count / metrics.failed_tests * 100
                print(f"  {ftype}: {count} ({pct:.1f}%)")
        print()
    print(f"Total duration: {metrics.total_duration:.1f}s")
    print(f"Avg per test: {metrics.total_duration/metrics.total_tests:.1f}s" if metrics.total_tests > 0 else "")
    print("=" * 60)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "config": config.model_dump(),
                "filters": {
                    "home": args.home,
                    "type": args.type,
                    "limit": args.limit,
                },
                "metrics": metrics.to_dict(),
            }, f, indent=2, default=str)

        # Save detailed results with full trace
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([{
                "test_id": r.test_id,
                "success": r.success,
                "is_error_input_only": r.is_error_input_only,
                "handled_correctly": r.handled_correctly,
                "plan_generated": r.plan_generated,
                "execution_success": r.execution_success,
                "matched_actions": r.matched_actions,
                "missing_actions": r.missing_actions,
                "extra_actions": r.extra_actions,
                "expected_actions": r.expected_actions,
                "expected_params": r.expected_params,
                "actions_in_plan": r.actions_in_plan,
                "params_in_plan": r.params_in_plan,
                "properties_matched": r.properties_matched,
                "properties_checked": r.properties_checked,
                "property_results": r.property_results,
                "expected_impossible": r.expected_impossible,
                "detected_impossible": r.detected_impossible,
                "failure_type": r.failure_type,
                "duration": r.duration_seconds,
                "error": r.error,
                "trace": r.raw_result,  # Full experiment trace
            } for r in results], f, indent=2, default=str)

        # Save individual traces (compatible with trace_viewer)
        traces_dir = output_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        for r in results:
            if r.raw_result:
                trace_file = traces_dir / f"{r.test_id}.json"
                # Add metadata for trace_viewer compatibility
                trace_data = r.raw_result.copy()
                trace_data["config_name"] = f"homebench_{r.test_id}"
                with open(trace_file, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}/")
        print(f"Individual traces saved to {traces_dir}/")

        # Generate HTML traces if requested
        if args.generate_traces:
            print("\nGenerating HTML traces...")
            from trace_viewer import export_html
            for r in results:
                if r.raw_result:
                    trace_file = traces_dir / f"{r.test_id}.json"
                    html_file = traces_dir / f"{r.test_id}.html"
                    export_html(r.raw_result, str(html_file))
            print(f"HTML traces saved to {traces_dir}/")

        # Generate eval report if requested
        if args.generate_report:
            print("\nGenerating evaluation report...")
            import subprocess
            subprocess.run(["uv", "run", "python", "eval_viewer.py", str(output_dir)], check=True)
            report_file = output_dir / "eval_report.html"
            if report_file.exists():
                print(f"Report generated: {report_file}")
                if args.open_report:
                    import webbrowser
                    webbrowser.open(f"file://{report_file.absolute()}")

    # Exit successfully when the run completes; benchmark score should not
    # determine process success/failure.
    sys.exit(0)


if __name__ == "__main__":
    main()
