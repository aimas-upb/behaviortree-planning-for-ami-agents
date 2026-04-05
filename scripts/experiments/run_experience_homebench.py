"""
Experience-Based HomeBench Evaluation Runner.

Runs behavior tree planning experiments using experience accumulation
and reuse across sequential test runs. Measures the impact of experience
matching on planning efficiency and success rates.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

from scripts.common import resolve_repo_path

# Reuse TestCase, TestResult, EvaluationMetrics from run_homebench
from scripts.experiments.run_homebench import (
    EvaluationMetrics,
    TestCase,
    TestResult,
)
from src.config import (
    AffordanceConfig,
    DiscoveryConfig,
    ExecutionConfig,
    ExperienceConfig,
    ExperimentConfig,
    ExperimentMeta,
    ModelConfig,
    OutputConfig,
    PlanningConfig,
    ReasoningConfig,
    StateConfig,
    TracingConfig,
)
from src.experience.adaptation import ExperienceAdapter
from src.experience.bt_serialization import extract_action_urls
from src.experience.engine import ExperienceEngine
from src.experience.intent import IntentExtractor
from src.experience.matching import ExperienceMatcher
from src.experience.runner import ExperiencePipelineRunner, ExperienceRunResult

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperienceMetrics:
    """Experience-specific metrics tracked across the evaluation."""

    total_intents_extracted: int = 0
    total_matched: int = 0
    total_unmatched: int = 0
    total_matched_infeasible: int = 0
    match_count_per_test: list[int] = field(default_factory=list)
    avg_similarity_per_test: list[float] = field(default_factory=list)
    matched_planning_times: list[float] = field(default_factory=list)
    unmatched_planning_times: list[float] = field(default_factory=list)
    end_to_end_times: list[float] = field(default_factory=list)
    experience_size_over_time: list[int] = field(default_factory=list)
    total_new_experiences_stored: int = 0

    def to_dict(self) -> dict:
        return {
            "total_intents_extracted": self.total_intents_extracted,
            "total_matched": self.total_matched,
            "total_unmatched": self.total_unmatched,
            "total_matched_infeasible": self.total_matched_infeasible,
            "match_count_per_test": self.match_count_per_test,
            "avg_similarity_per_test": self.avg_similarity_per_test,
            "matched_planning_times": self.matched_planning_times,
            "unmatched_planning_times": self.unmatched_planning_times,
            "end_to_end_times": self.end_to_end_times,
            "experience_size_over_time": self.experience_size_over_time,
            "total_new_experiences_stored": self.total_new_experiences_stored,
            "avg_match_count": (
                sum(self.match_count_per_test) / len(self.match_count_per_test)
                if self.match_count_per_test
                else 0.0
            ),
            "avg_matched_planning_time": (
                sum(self.matched_planning_times)
                / len(self.matched_planning_times)
                if self.matched_planning_times
                else 0.0
            ),
            "avg_unmatched_planning_time": (
                sum(self.unmatched_planning_times)
                / len(self.unmatched_planning_times)
                if self.unmatched_planning_times
                else 0.0
            ),
        }


class ExperienceHomeBenchEvaluator:
    """
    Evaluator for HomeBench test cases with experience accumulation.

    Mirrors HomeBenchEvaluator but uses the experience pipeline runner
    instead of the stateless pipeline. Experience accumulates within
    a single evaluation run (one pass through a test JSON file).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client: OpenAI,
        ontology_path: str = "ontologies/homeont.ttl",
        experience_store_path: Optional[str] = None,
        similarity_threshold: float = 0.85,
        clear_experience: bool = False,
        simulator_url: str = "http://localhost:8080",
        reset_enabled: bool = True,
    ):
        self.config = config
        self.client = client
        self.simulator_url = simulator_url
        self.reset_enabled = reset_enabled
        self.http_client = httpx.Client(timeout=30.0)

        # Load ontology
        ontology_text = Path(ontology_path).read_text()

        # Initialize experience components
        if clear_experience and experience_store_path:
            store_path = Path(experience_store_path)
            if store_path.exists():
                store_path.unlink()
                logger.info(
                    f"Cleared experience store: {experience_store_path}"
                )

        self.engine = ExperienceEngine(persistence_path=experience_store_path)
        self.matcher = ExperienceMatcher(
            similarity_threshold=similarity_threshold,
        )
        self.intent_extractor = IntentExtractor(ontology_text=ontology_text)
        self.adapter = ExperienceAdapter()

        self.pipeline_runner = ExperiencePipelineRunner(
            config=config,
            client=client,
            engine=self.engine,
            matcher=self.matcher,
            intent_extractor=self.intent_extractor,
            adapter=self.adapter,
        )

    def load_tests(
        self,
        data_path: str,
        home_id: Optional[str] = None,
        test_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TestCase]:
        """Load test cases from JSON file."""
        with open(resolve_repo_path(data_path)) as f:
            data = json.load(f)

        tests = [TestCase.from_dict(d) for d in data]

        if home_id:
            tests = [t for t in tests if t.home_id == home_id]
        if test_type:
            tests = [t for t in tests if t.test_type == test_type]
        if limit:
            tests = tests[:limit]

        return tests

    def reset_home(self, home_id: str) -> bool:
        """Reset a home to initial state."""
        if not self.reset_enabled:
            return True

        try:
            numeric_id = home_id.replace("home", "")
            response = self.http_client.post(
                f"{self.simulator_url}/reset",
                json={"home": numeric_id},
            )
            if response.status_code == 404:
                logger.warning(
                    "Reset endpoint not available, continuing without reset"
                )
                self.reset_enabled = False
                return True
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to reset home {home_id}: {e}")
            return True

    def verify_property(
        self, property_url: str, expected_value
    ) -> tuple[bool, any]:
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

    def _verify_test_properties(self, test: TestCase, result: TestResult):
        """Verify all expected properties for a test case."""
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
                    result.property_results.append(
                        {
                            "property": prop_url,
                            "expected": exp_val,
                            "actual": actual,
                            "matched": matched,
                        }
                    )

    def run_test(
        self, test: TestCase
    ) -> tuple[TestResult, ExperienceRunResult]:
        """
        Run a single test case using the experience pipeline.

        Returns:
            (TestResult, ExperienceRunResult) tuple.
        """
        result = TestResult(test_id=test.id, success="False")
        result.expected_actions = [
            o.get("affordance", "")
            for o in test.expected_successes
            if o.get("affordance")
        ]
        result.expected_params = {
            o.get("affordance"): o.get("params", {})
            for o in test.expected_successes
            if o.get("affordance")
        }
        result.expected_impossible = len(test.expected_errors)
        result.is_error_input_only = (
            len(test.expected_successes) == 0 and len(test.expected_errors) > 0
        )

        start_time = datetime.now()

        try:
            # Reset home to initial state
            self.reset_home(test.home_id)

            # Build entry point
            entry_point = (
                f"{self.simulator_url}/workspaces/{test.home_id}#workspace"
            )

            # Run experience pipeline
            exp_result = self.pipeline_runner.run(
                goal=test.input,
                entry_point=entry_point,
                home_id=test.home_id,
                test_id=test.id,
            )

            result.raw_result = exp_result.to_dict()

            # Map ExperienceRunResult to TestResult fields
            if exp_result.combined_plan_ir:
                result.plan_generated = True
                result.plan_format = "json_ir"

                # Extract actions from the combined plan
                action_urls = extract_action_urls(exp_result.combined_plan_ir)
                result.actions_in_plan = action_urls

                # Extract params from the combined plan
                result.params_in_plan = self._extract_params_from_ir(
                    exp_result.combined_plan_ir
                )
            elif (
                exp_result.matched_infeasible_intents
                and not exp_result.matched_plans
            ):
                # All intents were infeasible — no plan generated is expected
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
                self._verify_test_properties(test, result)

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
                    # Check if all intents were matched as infeasible
                    if (
                        exp_result.matched_infeasible_intents
                        and not exp_result.matched_plans
                        and not exp_result.unmatched_plans
                    ):
                        # All matched as infeasible — treat as success if
                        # that matches ground truth
                        if result.is_error_input_only:
                            result.success = "True"
                        else:
                            result.success = "False"
                    else:
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
                    result.failure_type = self._classify_failure(
                        result, exp_result
                    )

            # Learning: store experiences on success
            if result.success == "True" and exp_result.intents:
                stored = self.pipeline_runner.store_experiences(
                    intents=exp_result.intents,
                    match_results=exp_result.match_results,
                    combined_plan_ir=exp_result.combined_plan_ir,
                    test_id=test.id,
                )
                exp_result.new_experiences_stored = stored

                # Store confirmed infeasible intents
                self._store_infeasible_from_ground_truth(test, exp_result)

        except Exception as e:
            result.error = str(e)
            result.failure_type = "other"
            logger.exception(f"Test {test.id} failed with exception")
            exp_result = ExperienceRunResult(error=str(e))

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result, exp_result

    def _store_infeasible_from_ground_truth(
        self,
        test: TestCase,
        exp_result: ExperienceRunResult,
    ):
        """
        Store infeasible intents confirmed by ground truth.

        Aligns extracted intents with ground truth by splitting the
        test input by comma to get positional alignment.
        """
        if not test.expected_errors or not exp_result.intents:
            return

        # Split input by comma for positional alignment
        commands = [c.strip() for c in test.input.split(",")]

        # Find positions of error_input in ground truth
        error_positions: set[int] = set()
        for i, output in enumerate(test.expected_outputs):
            if output.get("execution") == "error_input":
                error_positions.add(i)

        if not error_positions:
            return

        # For each extracted intent, check if its original_index
        # corresponds to an error_input position
        for intent in exp_result.intents:
            if intent.original_index in error_positions:
                self.pipeline_runner.store_infeasible(
                    intent, test.id, home_id=test.home_id
                )

    def _classify_failure(
        self,
        result: TestResult,
        exp_result: ExperienceRunResult,
    ) -> str:
        """Classify the type of failure."""
        error_msg = exp_result.error or ""

        if not result.plan_generated:
            if "parse" in error_msg.lower() or "json" in error_msg.lower():
                return "parse_error"
            return "other"

        if result.plan_generated and not result.execution_success:
            return "execution_error"

        if (
            result.execution_success
            and result.properties_checked > 0
            and result.properties_matched < result.properties_checked
        ):
            return "property_mismatch"

        if result.missing_actions or result.extra_actions:
            return "action_mismatch"

        return "other"

    def _extract_params_from_ir(self, json_ir: dict) -> dict:
        """Extract action_url -> params mapping from JSON-IR."""
        params = {}
        self._collect_params(json_ir, params)
        return params

    def _collect_params(self, node: dict, params: dict):
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            url = node.get("action_url")
            if url:
                params[url] = node.get("parameters", {})
        for child in node.get("children", []):
            self._collect_params(child, params)

    def evaluate(
        self,
        tests: list[TestCase],
        progress: bool = True,
    ) -> tuple[list[TestResult], EvaluationMetrics, ExperienceMetrics]:
        """
        Evaluate all test cases sequentially (order matters for experience).

        Returns:
            (results, standard_metrics, experience_metrics)
        """
        results = []
        metrics = EvaluationMetrics()
        exp_metrics = ExperienceMetrics()

        iterator = tqdm(tests, desc="Evaluating") if progress else tests

        for test in iterator:
            result, exp_result = self.run_test(test)
            results.append(result)

            # Standard metrics
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

            metrics.total_expected_impossible += result.expected_impossible
            metrics.total_detected_impossible += len(result.detected_impossible)

            if (
                result.failure_type
                and result.failure_type in metrics.failures_by_type
            ):
                metrics.failures_by_type[result.failure_type] += 1

            metrics.total_duration += result.duration_seconds

            # Experience-specific metrics
            n_intents = len(exp_result.intents)
            n_matched = sum(
                1
                for m in exp_result.match_results
                if m.matched and not m.is_infeasible
            )
            n_infeasible = len(exp_result.matched_infeasible_intents)
            n_unmatched = n_intents - n_matched - n_infeasible

            exp_metrics.total_intents_extracted += n_intents
            exp_metrics.total_matched += n_matched
            exp_metrics.total_unmatched += n_unmatched
            exp_metrics.total_matched_infeasible += n_infeasible

            exp_metrics.match_count_per_test.append(n_matched)

            # Average similarity for matched intents
            matched_sims = [
                m.similarity_score
                for m in exp_result.match_results
                if m.matched
            ]
            avg_sim = (
                sum(matched_sims) / len(matched_sims) if matched_sims else 0.0
            )
            exp_metrics.avg_similarity_per_test.append(avg_sim)

            # Timing
            exp_metrics.matched_planning_times.append(
                exp_result.matched_plan_time_seconds
            )
            exp_metrics.unmatched_planning_times.append(
                exp_result.unmatched_plan_time_seconds
            )
            exp_metrics.end_to_end_times.append(exp_result.duration_seconds)

            # Experience store size
            exp_metrics.experience_size_over_time.append(self.engine.size())
            exp_metrics.total_new_experiences_stored += (
                exp_result.new_experiences_stored
            )

            if progress:
                iterator.set_postfix(
                    {
                        "success": f"{metrics.success_rate:.1%}",
                        "exp_size": self.engine.size(),
                        "matched": n_matched,
                    }
                )

        return results, metrics, exp_metrics


def _reshape_trace_for_viewer(
    raw_result: dict,
    test_id: str,
    config: ExperimentConfig,
) -> dict:
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
        "config": config.model_dump(),
        "success": raw_result.get("success", False),
        "duration_seconds": raw_result.get("duration_seconds", 0),
        "error": raw_result.get("error"),
        # Experience-specific data
        "intents": raw_result.get("intents"),
        "match_results": raw_result.get("match_results"),
        "matched_plan_traces": raw_result.get("matched_plan_traces"),
        "matched_plan_time_seconds": raw_result.get(
            "matched_plan_time_seconds"
        ),
        "unmatched_plan_time_seconds": raw_result.get(
            "unmatched_plan_time_seconds"
        ),
        "detected_impossible": raw_result.get("detected_impossible"),
    }

    # Extract discovery and planning from unmatched_plan_traces
    # (the first entry contains the full discovery+planning for
    # unmatched intents, which is the most informative trace).
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


def create_config(
    name: str,
    affordance_strategy: str = "agentic_query",
    state_strategy: str = "all",
    reasoning_enabled: bool = False,
    reasoning_strategy: str = "chain_of_thought",
    output_format: str = "python_code",
    prompt_strategy: str = "detailed",
    model: str = "gpt-4o",
    experience_store: str = "experience_store.json",
    similarity_threshold: float = 0.85,
) -> ExperimentConfig:
    """Create an experiment config for experience evaluation."""
    return ExperimentConfig(
        experiment=ExperimentMeta(
            name=name,
            description=f"Experience-based HomeBench evaluation: {name}",
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
        tracing=TracingConfig(enabled=False),
        experience=ExperienceConfig(
            enabled=True,
            persistence_path=experience_store,
            similarity_threshold=similarity_threshold,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experience-Based HomeBench Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data",
        default="data/homebench/benchmarks/repeated_query/selected_test_single_feasible.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--home", type=str, help="Filter by home ID (e.g., 'home96')"
    )
    parser.add_argument(
        "--type", choices=["one", "multi"], help="Filter by test type"
    )
    parser.add_argument("--limit", type=int, help="Limit number of tests")

    # Config
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument(
        "--discovery-affordances",
        choices=["exhaustive", "agentic", "relevant", "agentic_query"],
        default="agentic_query",
    )
    parser.add_argument(
        "--discovery-state",
        choices=["all", "relevant", "agentic", "none"],
        default="all",
    )
    parser.add_argument(
        "--planning-reasoning",
        choices=["none", "chain_of_thought", "multi_turn", "reflection"],
        default="none",
    )
    parser.add_argument(
        "--planning-output",
        choices=["json_ir", "python_code", "python_code_unconstrained"],
        default="python_code",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=["baseline", "detailed", "few_shot", "icl"],
        default="detailed",
    )

    # Experience options
    parser.add_argument(
        "--experience-store",
        type=str,
        default=None,
        help="Path for experience persistence (default: auto-scoped by data file)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Embedding similarity threshold for matching (default: 0.85)",
    )
    parser.add_argument(
        "--clear-experience",
        action="store_true",
        help="Clear experience store before starting",
    )
    parser.add_argument(
        "--ontology",
        default="ontologies/homeont.ttl",
        help="Path to homeont.ttl ontology file",
    )

    # Output
    parser.add_argument(
        "--output", type=str, help="Output directory for results"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--generate-traces",
        action="store_true",
        help="Generate individual HTML trace files for each test",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML evaluation report after running",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated report in browser",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    data_path = resolve_repo_path(args.data)
    config_path = resolve_repo_path(args.config) if args.config else None
    ontology_path = resolve_repo_path(args.ontology)
    resolved_output_dir = (
        resolve_repo_path(args.output) if args.output else None
    )

    # Determine experience store path (auto-scope by data file)
    if args.experience_store:
        experience_store_path = resolve_repo_path(args.experience_store)
    else:
        data_stem = data_path.stem
        base_output_dir = resolved_output_dir or resolve_repo_path(".")
        experience_store_path = (
            base_output_dir / f"experience_store_{data_stem}.json"
        )

    # Load or create config
    if config_path:
        from src.config import load_config

        config = load_config(str(config_path))
    else:
        reasoning_enabled = args.planning_reasoning != "none"
        config = create_config(
            name="experience_homebench_eval",
            affordance_strategy=args.discovery_affordances,
            state_strategy=args.discovery_state,
            reasoning_enabled=reasoning_enabled,
            reasoning_strategy=(
                args.planning_reasoning
                if reasoning_enabled
                else "chain_of_thought"
            ),
            output_format=args.planning_output,
            prompt_strategy=args.prompt_strategy,
            model=args.model,
            experience_store=str(experience_store_path),
            similarity_threshold=args.similarity_threshold,
        )

    # Setup client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Create evaluator
    evaluator = ExperienceHomeBenchEvaluator(
        config=config,
        client=client,
        ontology_path=str(ontology_path),
        experience_store_path=str(experience_store_path),
        similarity_threshold=args.similarity_threshold,
        clear_experience=args.clear_experience,
    )

    # Load tests
    print(f"Loading tests from {data_path}...")
    tests = evaluator.load_tests(
        str(data_path),
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
    print("CONFIGURATION (Experience-Based)")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Discovery - Affordances: {config.discovery.affordances.strategy}")
    print(f"Discovery - State: {config.discovery.state.strategy}")
    reasoning = (
        config.planning.reasoning.strategy
        if config.planning.reasoning.enabled
        else "none"
    )
    print(f"Planning - Reasoning: {reasoning}")
    print(f"Planning - Output: {config.planning.output.format}")
    print(f"Planning - Prompt: {config.planning.prompt_strategy}")
    print(f"Experience Store: {experience_store_path}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Clear Experience: {args.clear_experience}")
    print("=" * 60 + "\n")

    # Run evaluation (sequential — order matters)
    results, metrics, exp_metrics = evaluator.evaluate(
        tests, progress=not args.no_progress
    )

    # Print standard summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tests: {metrics.total_tests}")
    print(
        f"Successful: {metrics.successful_tests} ({metrics.success_rate:.1%})"
    )
    print(
        f"Quantifiable: {metrics.quantifiable_tests} ({metrics.quantifiable_rate:.1%})"
    )
    print(
        f"Success + Quantifiable: "
        f"{metrics.successful_tests + metrics.quantifiable_tests} "
        f"({metrics.success_or_quantifiable_rate:.1%})"
    )
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
    print(
        f"Property Verification: "
        f"{metrics.total_properties_matched}/{metrics.total_properties_checked} "
        f"({metrics.property_accuracy:.1%})"
    )
    print()
    if metrics.total_expected_impossible > 0:
        print(
            f"Impossible Sub-goals: "
            f"{metrics.total_detected_impossible}/{metrics.total_expected_impossible} "
            f"detected ({metrics.impossible_detection_rate:.1%})"
        )
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
    if metrics.total_tests > 0:
        print(
            f"Avg per test: {metrics.total_duration / metrics.total_tests:.1f}s"
        )

    # Print experience summary
    print("\n" + "=" * 60)
    print("EXPERIENCE METRICS")
    print("=" * 60)
    print(f"Total intents extracted: {exp_metrics.total_intents_extracted}")
    print(f"Total matched (feasible): {exp_metrics.total_matched}")
    print(f"Total matched (infeasible): {exp_metrics.total_matched_infeasible}")
    print(f"Total unmatched (novel): {exp_metrics.total_unmatched}")
    print(f"New experiences stored: {exp_metrics.total_new_experiences_stored}")
    print(f"Final experience store size: {evaluator.engine.size()}")
    print()

    if exp_metrics.matched_planning_times:
        avg_matched = sum(exp_metrics.matched_planning_times) / len(
            exp_metrics.matched_planning_times
        )
        print(f"Avg matched planning time: {avg_matched:.2f}s")
    if exp_metrics.unmatched_planning_times:
        avg_unmatched = sum(exp_metrics.unmatched_planning_times) / len(
            exp_metrics.unmatched_planning_times
        )
        print(f"Avg unmatched planning time: {avg_unmatched:.2f}s")
    if exp_metrics.end_to_end_times:
        avg_e2e = sum(exp_metrics.end_to_end_times) / len(
            exp_metrics.end_to_end_times
        )
        print(f"Avg end-to-end time: {avg_e2e:.2f}s")

    print()
    engine_stats = evaluator.engine.stats()
    print(f"Engine stats: {json.dumps(engine_stats, indent=2)}")
    print("=" * 60)

    # Save results
    if args.output:
        output_dir = resolved_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics (prefix must be "metrics_" for eval_viewer compatibility)
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "config": config.model_dump(),
                    "filters": {
                        "home": args.home,
                        "type": args.type,
                        "limit": args.limit,
                    },
                    "metrics": metrics.to_dict(),
                    "experience_metrics": exp_metrics.to_dict(),
                    "engine_stats": engine_stats,
                },
                f,
                indent=2,
                default=str,
            )

        # Save detailed results (prefix must be "results_" for eval_viewer compatibility)
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                [
                    {
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
                        "trace": r.raw_result,
                    }
                    for r in results
                ],
                f,
                indent=2,
                default=str,
            )

        # Save individual traces (reshaped for trace_viewer compatibility)
        traces_dir = output_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        for r in results:
            if r.raw_result:
                trace_file = traces_dir / f"{r.test_id}.json"
                trace_data = _reshape_trace_for_viewer(
                    r.raw_result, r.test_id, config
                )
                with open(trace_file, "w") as f:
                    json.dump(trace_data, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}/")
        print(f"Individual traces saved to {traces_dir}/")
        print(f"Experience store: {experience_store_path}")

        # Generate HTML traces if requested
        if args.generate_traces:
            print("\nGenerating HTML traces...")
            from viewers.experience_trace_viewer import export_experience_html

            for r in results:
                if r.raw_result:
                    trace = _reshape_trace_for_viewer(
                        r.raw_result, r.test_id, config
                    )
                    html_file = traces_dir / f"{r.test_id}.html"
                    try:
                        export_experience_html(trace, str(html_file))
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate HTML trace for {r.test_id}: {e}"
                        )
            print(f"HTML traces saved to {traces_dir}/")

        # Generate eval report if requested
        if args.generate_report:
            print("\nGenerating evaluation report...")
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "viewers.eval_viewer", str(output_dir)],
                check=True,
            )
            report_file = output_dir / "eval_report.html"
            if report_file.exists():
                print(f"Report generated: {report_file}")
                if args.open_report:
                    import webbrowser

                    webbrowser.open(f"file://{report_file.absolute()}")

    sys.exit(0 if metrics.success_rate > 0.5 else 1)


if __name__ == "__main__":
    main()
