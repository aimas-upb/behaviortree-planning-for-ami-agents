"""
Experience Pipeline Runner.

Orchestrates the full experience-based planning pipeline:
  1. Intent Extraction — parse goal into structured intents
  2. Experience Matching — match intents against stored experiences
  3. Matched Experience Planning — adapt matched experiences (lightweight)
  4. Unmatched Experience Planning — full discovery + planning for novel intents
  5. Concatenation — combine matched + unmatched BTs
  6. Execution — run the combined BT
"""

import json
import logging
import time
import uuid
import ast
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import py_trees

from ..config import (
    ExperimentConfig,
    ModelConfig,
    DiscoveryConfig,
    AffordanceConfig,
    StateConfig,
)
from ..discovery import create_discovery_pipeline, DiscoveryResult
from ..planning import create_planner, Plan
from ..execution import create_executor, ExecutionResult, IRExecutor, CodeExecutor
from .adaptation import ExperienceAdapter
from .bt_serialization import (
    combine_trees_parallel,
    extract_leaf_nodes_json_ir,
    py_tree_to_json_ir,
)
from .engine import ExperienceEngine, ExperienceEntry
from .intent import IntentExtractor, StructuredIntent
from .matching import ExperienceMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class ExperienceRunResult:
    """Result of a single experience-based pipeline run."""

    # Intent extraction
    intents: list[StructuredIntent] = field(default_factory=list)
    intent_extraction_trace: Optional[dict] = None

    # Matching
    match_results: list[MatchResult] = field(default_factory=list)

    # Matched experience planning
    matched_plans: list[dict] = field(default_factory=list)  # adapted JSON-IR dicts
    matched_infeasible_intents: list[StructuredIntent] = field(default_factory=list)
    matched_plan_traces: list[dict] = field(default_factory=list)
    matched_plan_time_seconds: float = 0.0

    # Unmatched (full pipeline) planning
    unmatched_plans: list[dict] = field(default_factory=list)  # JSON-IR or code plans
    unmatched_plan_traces: list[dict] = field(default_factory=list)
    unmatched_plan_time_seconds: float = 0.0

    # Combined plan & execution
    combined_plan_ir: Optional[dict] = None
    execution_result: Optional[ExecutionResult] = None
    executed_tree: Optional[py_trees.behaviour.Behaviour] = None

    # Learning
    new_experiences_stored: int = 0

    # Overall
    success: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0

    # Impossible sub-goal tracking
    detected_impossible: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intents": [i.to_dict() for i in self.intents],
            "intent_extraction_trace": self.intent_extraction_trace,
            "match_results": [m.to_dict() for m in self.match_results],
            "matched_plans": self.matched_plans,
            "matched_infeasible_count": len(self.matched_infeasible_intents),
            "matched_plan_traces": self.matched_plan_traces,
            "matched_plan_time_seconds": self.matched_plan_time_seconds,
            "unmatched_plans": self.unmatched_plans,
            "unmatched_plan_traces": self.unmatched_plan_traces,
            "unmatched_plan_time_seconds": self.unmatched_plan_time_seconds,
            "combined_plan_ir": self.combined_plan_ir,
            "execution": self.execution_result.to_dict() if self.execution_result else None,
            "new_experiences_stored": self.new_experiences_stored,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "detected_impossible": self.detected_impossible,
        }


class ExperiencePipelineRunner:
    """
    Orchestrates the experience-based planning pipeline.

    The pipeline accumulates knowledge across sequential test runs.
    On subsequent similar requests, matched experiences skip expensive
    discovery+planning and adapt stored BT leaf nodes instead.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client,
        engine: ExperienceEngine,
        matcher: ExperienceMatcher,
        intent_extractor: IntentExtractor,
        adapter: ExperienceAdapter,
    ):
        self.config = config
        self.client = client
        self.engine = engine
        self.matcher = matcher
        self.intent_extractor = intent_extractor
        self.adapter = adapter

    def run(
        self,
        goal: str,
        entry_point: str,
        home_id: str,
        test_id: str,
    ) -> ExperienceRunResult:
        """
        Run the full experience pipeline for a single goal.

        Steps:
          1. Extract intents from the goal
          2. Match intents against stored experiences
          3. Partition into matched-feasible, matched-infeasible, unmatched
          4. Adapt matched experiences (lightweight)
          5. Full discovery + planning for unmatched intents
          6. Combine and execute

        Returns:
            ExperienceRunResult with all pipeline traces.
        """
        result = ExperienceRunResult()
        start_time = time.time()

        try:
            # Step 1: Intent Extraction
            logger.info("=" * 60)
            logger.info("STEP 1: Intent Extraction")
            logger.info("=" * 60)

            intents, intent_trace = self.intent_extractor.extract(
                goal=goal,
                client=self.client,
                model_config=self.config.model,
            )
            result.intents = intents
            result.intent_extraction_trace = intent_trace

            if not intents:
                result.error = "No intents extracted from goal"
                logger.warning(result.error)
                return result

            logger.info(f"Extracted {len(intents)} intents")

            # Step 2: Experience Matching
            logger.info("=" * 60)
            logger.info("STEP 2: Experience Matching")
            logger.info("=" * 60)

            match_results = self.matcher.match(intents, self.engine)
            result.match_results = match_results

            # Partition intents
            matched_feasible: list[tuple[StructuredIntent, MatchResult]] = []
            matched_infeasible: list[StructuredIntent] = []
            unmatched: list[StructuredIntent] = []

            for intent, match in zip(intents, match_results):
                if match.matched:
                    if match.is_infeasible:
                        matched_infeasible.append(intent)
                    else:
                        matched_feasible.append((intent, match))
                else:
                    unmatched.append(intent)

            result.matched_infeasible_intents = matched_infeasible

            logger.info(
                f"Partition: {len(matched_feasible)} matched-feasible, "
                f"{len(matched_infeasible)} matched-infeasible, "
                f"{len(unmatched)} unmatched"
            )

            # Track infeasible intents detected via matching
            for intent in matched_infeasible:
                result.detected_impossible.append(intent.text_intent)

            # Step 3: Matched Experience Planning (lightweight adaptation)
            logger.info("=" * 60)
            logger.info("STEP 3: Matched Experience Adaptation")
            logger.info("=" * 60)

            matched_t0 = time.time()
            adapted_irs: list[dict] = []

            for intent, match in matched_feasible:
                logger.info(
                    f"Adapting experience {match.experience.id} "
                    f"for intent: {intent.text_intent!r}"
                )

                # Lightweight state discovery scoped to the target device
                try:
                    discovery_result = self._discover_for_adaptation(
                        entry_point, goal
                    )
                except Exception as e:
                    logger.warning(
                        f"Discovery for adaptation failed: {e}; "
                        f"moving intent to unmatched"
                    )
                    unmatched.append(intent)
                    continue

                adapted_ir, adapt_trace = self.adapter.adapt(
                    experience=match.experience,
                    intent=intent,
                    discovery_result=discovery_result,
                    client=self.client,
                    model_config=self.config.model,
                )
                result.matched_plan_traces.append(adapt_trace)

                if adapted_ir is not None:
                    adapted_irs.append(adapted_ir)
                    logger.info(
                        f"Successfully adapted: {adapted_ir.get('action_url', 'N/A')}"
                    )
                else:
                    logger.warning(
                        f"Adaptation failed for intent {intent.text_intent!r}; "
                        f"moving to unmatched"
                    )
                    unmatched.append(intent)

            result.matched_plans = adapted_irs
            result.matched_plan_time_seconds = time.time() - matched_t0

            logger.info(
                f"Matched planning: {len(adapted_irs)} adapted, "
                f"{result.matched_plan_time_seconds:.2f}s"
            )

            # Step 4: Unmatched Experience Planning (full pipeline)
            logger.info("=" * 60)
            logger.info("STEP 4: Unmatched (Full) Planning")
            logger.info("=" * 60)

            unmatched_t0 = time.time()
            unmatched_irs: list[dict] = []

            if unmatched:
                # Build the full goal from unmatched intents for discovery
                unmatched_goal = ", ".join(i.text_intent for i in unmatched)
                logger.info(
                    f"Running full pipeline for {len(unmatched)} unmatched intents"
                )

                try:
                    # Full discovery
                    discovery_pipeline = create_discovery_pipeline(
                        config=self.config.discovery,
                        client=self.client,
                        model_config=self.config.model,
                    )
                    discovery_result = discovery_pipeline.discover(
                        entry_point, unmatched_goal
                    )

                    # Planning
                    planner = create_planner(self.config.planning)
                    planning_result = planner.plan(
                        goal=unmatched_goal,
                        discovery=discovery_result,
                        client=self.client,
                        model_config=self.config.model,
                    )

                    result.unmatched_plan_traces.append({
                        "discovery": discovery_result.to_dict(),
                        "planning": planning_result.to_dict(),
                    })

                    if planning_result.success:
                        plan = planning_result.plan

                        # Track impossible sub-goals detected during planning
                        if plan.detected_impossible:
                            result.detected_impossible.extend(
                                plan.detected_impossible
                            )

                        if plan.is_json_ir and isinstance(plan.content, dict):
                            if plan.content:
                                unmatched_irs.append(plan.content)
                        elif plan.is_python_code:
                            # If planning marked the goal as impossible and produced
                            # only comments (no executable tree), treat as a normal
                            # infeasible/no-op outcome and skip execution.
                            if (
                                plan.detected_impossible
                                and not self._code_defines_tree_or_builder(
                                    plan.content
                                )
                            ):
                                logger.info(
                                    "Plan contains only impossible sub-goals and "
                                    "no executable tree; skipping execution"
                                )
                            else:
                                # Execute code to get the tree, then serialize
                                executor = create_executor(
                                    output_format=plan.format,
                                    max_ticks=self.config.execution.max_ticks,
                                )
                                exec_result = executor.execute(plan)

                                if exec_result.success:
                                    # Re-build the tree (without ticking) and
                                    # convert to JSON-IR so downstream code
                                    # (store_experiences, concatenation) gets a
                                    # proper dict instead of a code string.
                                    code_plan_json_ir = None
                                    try:
                                        assert isinstance(executor, CodeExecutor)
                                        tree = executor._execute_code(
                                            plan.content,
                                            unconstrained=(
                                                plan.format
                                                == "python_code_unconstrained"
                                            ),
                                        )
                                        code_plan_json_ir = py_tree_to_json_ir(
                                            tree
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to convert code plan to "
                                            f"JSON-IR: {e}"
                                        )

                                    unmatched_irs.append({
                                        "_code_plan": True,
                                        "_plan": plan.to_dict(),
                                        "_execution": exec_result.to_dict(),
                                        "_json_ir": code_plan_json_ir,
                                    })
                                else:
                                    logger.warning(
                                        f"Unmatched plan execution failed: "
                                        f"{exec_result.error}"
                                    )
                    else:
                        logger.warning(
                            f"Unmatched planning failed: {planning_result.error}"
                        )

                except Exception as e:
                    logger.error(f"Unmatched planning pipeline failed: {e}")
                    result.unmatched_plan_traces.append({"error": str(e)})

            result.unmatched_plans = unmatched_irs
            result.unmatched_plan_time_seconds = time.time() - unmatched_t0

            logger.info(
                f"Unmatched planning: {len(unmatched_irs)} plans, "
                f"{result.unmatched_plan_time_seconds:.2f}s"
            )

            # Step 5: Concatenation
            logger.info("=" * 60)
            logger.info("STEP 5: Concatenation")
            logger.info("=" * 60)

            # Separate JSON-IR plans from code plans that were already executed
            json_irs = adapted_irs + [
                ir for ir in unmatched_irs if not ir.get("_code_plan")
            ]
            code_plans = [
                ir for ir in unmatched_irs if ir.get("_code_plan")
            ]

            # If we only have code plans (no JSON-IR), use the already-
            # completed execution result from Step 4 directly.
            if not json_irs and code_plans:
                if not matched_infeasible:
                    # Use the code plan execution result
                    code_plan_data = code_plans[0]
                    exec_data = code_plan_data["_execution"]

                    # Use the JSON-IR converted from the tree instance
                    # (not the raw code string from plan.content)
                    result.combined_plan_ir = code_plan_data.get("_json_ir")
                    result.execution_result = ExecutionResult(
                        success=exec_data.get("success", False),
                        tree_name=exec_data.get("tree_name", ""),
                        ticks=exec_data.get("ticks", 0),
                        final_status=exec_data.get("final_status", ""),
                        tick_history=exec_data.get("tick_history", []),
                        error=exec_data.get("error"),
                    )

                    if result.execution_result.success:
                        logger.info(
                            f"Using code plan execution: SUCCESS in "
                            f"{result.execution_result.ticks} ticks"
                        )
                        result.success = True
                    else:
                        logger.warning(
                            f"Code plan execution FAILED: "
                            f"{result.execution_result.error or result.execution_result.final_status}"
                        )
                        result.error = result.execution_result.error
                else:
                    # Code plans succeeded + some infeasible intents
                    logger.info(
                        "Code plan executed; some intents matched as infeasible"
                    )
                    code_plan_data = code_plans[0]
                    exec_data = code_plan_data["_execution"]

                    # Use the JSON-IR converted from the tree instance
                    result.combined_plan_ir = code_plan_data.get("_json_ir")
                    result.execution_result = ExecutionResult(
                        success=exec_data.get("success", False),
                        tree_name=exec_data.get("tree_name", ""),
                        ticks=exec_data.get("ticks", 0),
                        final_status=exec_data.get("final_status", ""),
                        tick_history=exec_data.get("tick_history", []),
                        error=exec_data.get("error"),
                    )
                    result.success = result.execution_result.success

            elif not json_irs and not code_plans:
                if matched_infeasible or result.detected_impossible:
                    # All intents were infeasible/impossible
                    logger.info(
                        "All intents infeasible/impossible — skipping execution"
                    )
                    result.success = True
                else:
                    result.error = "No plans generated (all intents failed)"
                    logger.warning(result.error)
                return result

            else:
                # We have JSON-IR plans (possibly mixed with code plans).
                # If there are also code plans, we can't easily combine them
                # with JSON-IR, so log a warning and use only the JSON-IRs.
                if code_plans:
                    logger.warning(
                        f"Dropping {len(code_plans)} code plan(s) — "
                        f"cannot combine with JSON-IR plans"
                    )

                # Build combined JSON-IR
                if len(json_irs) == 1:
                    combined_ir = json_irs[0]
                else:
                    combined_ir = {
                        "type": "parallel",
                        "name": "CombinedPlan",
                        "policy": "success_on_one",
                        "children": json_irs,
                    }

                result.combined_plan_ir = combined_ir

                logger.info(
                    f"Combined plan: {len(json_irs)} subtrees"
                )

                # Step 6: Execution
                logger.info("=" * 60)
                logger.info("STEP 6: Execution")
                logger.info("=" * 60)

                plan = Plan(
                    format="json_ir",
                    content=combined_ir,
                    explanation="Experience-adapted combined plan",
                )

                executor = IRExecutor(max_ticks=self.config.execution.max_ticks)
                exec_result = executor.execute(plan)
                result.execution_result = exec_result

                if exec_result.success:
                    logger.info(
                        f"Execution SUCCESS in {exec_result.ticks} ticks"
                    )
                    result.success = True
                else:
                    logger.warning(
                        f"Execution FAILED: "
                        f"{exec_result.error or exec_result.final_status}"
                    )
                    result.error = exec_result.error

        except Exception as e:
            logger.exception("Experience pipeline failed")
            result.error = str(e)

        result.duration_seconds = time.time() - start_time
        return result

    def store_experiences(
        self,
        intents: list[StructuredIntent],
        match_results: list[MatchResult],
        combined_plan_ir: Optional[dict],
        test_id: str,
    ) -> int:
        """
        Store newly learned experiences after a successful test.

        Only stores experiences for intents that were unmatched (novel)
        and successfully planned. Infeasible intents are stored separately
        by the evaluator.

        Args:
            intents: All extracted intents for this test.
            match_results: Match results (same order as intents).
            combined_plan_ir: The combined JSON-IR that was executed.
            test_id: Test identifier for provenance.

        Returns:
            Number of new experiences stored.
        """
        if combined_plan_ir is None:
            return 0

        # Extract all action leaf nodes from the combined plan
        action_leaves = self._extract_all_action_leaves(combined_plan_ir)

        stored = 0

        for intent, match in zip(intents, match_results):
            # Only store for unmatched intents
            if match.matched:
                continue

            # Skip if identical experience already exists
            if self.engine.has_identical(intent):
                continue

            # Find the action leaf that corresponds to this intent
            # Match by the affordance_type pattern in action URLs/names
            leaf_ir = self._find_leaf_for_intent(intent, action_leaves)

            if leaf_ir is None:
                logger.debug(
                    f"No leaf found for intent {intent.text_intent!r} — skipping"
                )
                continue

            entry = ExperienceEntry(
                id=str(uuid.uuid4()),
                text_intent=intent.text_intent,
                affordance_type=intent.affordance_type,
                artifact_type=intent.artifact_type,
                workspace_type=intent.workspace_type,
                verb=intent.verb,
                bt_leaf_json_ir=leaf_ir,
                is_infeasible=False,
                created_at=datetime.now().isoformat(),
                source_test_id=test_id,
            )
            self.engine.add(entry)
            stored += 1

            logger.info(
                f"Stored experience {entry.id}: "
                f"{intent.text_intent!r} -> {leaf_ir.get('action_url', 'N/A')}"
            )

        if stored > 0:
            self.engine.save()

        return stored

    def store_infeasible(
        self,
        intent: StructuredIntent,
        test_id: str,
    ) -> bool:
        """
        Store a confirmed infeasible intent.

        Args:
            intent: The infeasible intent.
            test_id: Test identifier for provenance.

        Returns:
            True if stored (not a duplicate).
        """
        if self.engine.has_identical(intent):
            return False

        self.engine.add_infeasible(intent, test_id)
        self.engine.save()
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_for_adaptation(
        self,
        entry_point: str,
        goal: str,
    ) -> DiscoveryResult:
        """
        Run lightweight discovery for experience adaptation.

        Uses the state strategy 'all' to get current property values
        for parameter adaptation.
        """
        adaptation_discovery_config = DiscoveryConfig(
            affordances=AffordanceConfig(
                strategy=self.config.discovery.affordances.strategy,
            ),
            state=StateConfig(strategy="all"),
        )

        pipeline = create_discovery_pipeline(
            config=adaptation_discovery_config,
            client=self.client,
            model_config=self.config.model,
        )
        return pipeline.discover(entry_point, goal)

    @staticmethod
    def _extract_all_action_leaves(json_ir: dict) -> list[dict]:
        """Recursively extract all action-type leaf nodes from JSON-IR."""
        leaves: list[dict] = []
        ExperiencePipelineRunner._collect_leaves(json_ir, leaves)
        return leaves

    @staticmethod
    def _collect_leaves(node: dict, out: list[dict]) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "action":
            out.append(node)
            return
        for child in node.get("children", []):
            ExperiencePipelineRunner._collect_leaves(child, out)

    @staticmethod
    def _find_leaf_for_intent(
        intent: StructuredIntent,
        leaves: list[dict],
    ) -> Optional[dict]:
        """
        Find the action leaf that best corresponds to the given intent.

        Primary: match by semantic_type (ontology class name).
        Fallback: match by action URL pattern.
        """
        aff_type_lower = intent.affordance_type.lower()

        # Primary: match by semantic_type
        for leaf in leaves:
            leaf_type = leaf.get("semantic_type", "")
            if leaf_type and leaf_type.lower() == aff_type_lower:
                return leaf

        # Fallback: match by action URL pattern (for leaves without semantic_type)
        aff = intent.affordance_type
        if aff.endswith("Command"):
            aff = aff[: -len("Command")]

        # CamelCase -> snake_case
        parts: list[str] = []
        current: list[str] = []
        for ch in aff:
            if ch.isupper() and current:
                parts.append("".join(current).lower())
                current = [ch]
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).lower())

        snake_pattern = "_".join(parts)
        flat_pattern = "".join(parts)

        for leaf in leaves:
            url = leaf.get("action_url", "").lower()
            name = leaf.get("name", "").lower()
            url_flat = url.replace("-", "").replace("_", "")
            name_flat = name.replace("-", "").replace("_", "")

            if (
                snake_pattern in url
                or flat_pattern in url_flat
                or snake_pattern in name
                or flat_pattern in name_flat
            ):
                return leaf

        # Last resort: if there's only one leaf, use it
        if len(leaves) == 1:
            return leaves[0]

        return None

    @staticmethod
    def _code_defines_tree_or_builder(code: str) -> bool:
        """
        Return True if code defines an executable BT entrypoint.

        Recognized entrypoints:
        - assignment to variable named 'tree'
        - function definition named 'build_tree'
        """
        if not code or not code.strip():
            return False

        try:
            module = ast.parse(code)
        except SyntaxError:
            # Conservative fallback for malformed snippets
            return "tree" in code or "build_tree" in code

        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "tree":
                        return True
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "tree":
                    return True
            elif isinstance(node, ast.FunctionDef) and node.name == "build_tree":
                return True

        return False
