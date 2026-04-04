"""
Neuro-Symbolic Planning Runner.

A new strategy that combines:
  - Structured intent extraction (neuro step) with ontology-grounded LLM parsing
  - Direct SPARQL-based device/affordance resolution (symbolic step) for "set" intents
  - LLM-based BT code generation scoped only to "modify" intents

Pipeline:
  1. Intent Extraction — extended StructuredIntent with parameter & value fields
  2. SPARQL Resolution — query the /sparql endpoint directly per intent
  3. Routing:
       a. No result → impossible (stored per home_id in experience engine)
       b. Result + verb="set" → build BT node programmatically (no LLM planning)
       c. Result + verb="modify" → LLM-based BT code generation (modify-only prompt)
    4. Assemble & Execute — SetActions‖ModifyActions Parallel tree

Notes:
    - JSON-IR snapshots are persisted for traceability/viewing.
    - Runtime execution uses the merged in-memory py_trees object directly,
        avoiding lossy IR recompilation for custom inline compute nodes.
"""

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import py_trees
import requests

from ..config import ExperimentConfig
from ..execution import ExecutionResult, CodeExecutor, create_executor
from ..planning import create_planner
from .bt_serialization import py_tree_to_json_ir
from .engine import ExperienceEngine, ExperienceEntry
from .intent import IntentExtractor, StructuredIntent

from behavior_trees.affordance_nodes import ActionAffordanceNode

logger = logging.getLogger(__name__)


class _AnySuccessElseAllFailureParallel(py_trees.composites.Parallel):
    """
    Deterministic parallel semantics for neuro-symbolic execution.

    - RUNNING while any child is still non-terminal (RUNNING/INVALID)
    - SUCCESS once all children are terminal and at least one succeeded
    - FAILURE once all children are terminal and all failed

    This avoids early collapse from one failing sibling, allows slower siblings
    to complete, and still guarantees termination when no branch can succeed.
    """

    def __init__(self, name: str, children: list[py_trees.behaviour.Behaviour]):
        super().__init__(
            name=name,
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
            children=children,
        )

    def tick(self):
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        self.validate_policy_configuration()

        if self.status != py_trees.common.Status.RUNNING:
            self.logger.debug("%s.tick(): re-initialising" % self.__class__.__name__)
            for child in self.children:
                if child.status != py_trees.common.Status.INVALID:
                    child.stop(py_trees.common.Status.INVALID)
            self.current_child = None
            self.initialise()

        if not self.children:
            self.current_child = None
            self.stop(py_trees.common.Status.SUCCESS)
            yield self
            return

        for child in self.children:
            if self.policy.synchronise and child.status == py_trees.common.Status.SUCCESS:
                continue
            for node in child.tick():
                yield node

        statuses = [child.status for child in self.children]

        if any(s in (py_trees.common.Status.RUNNING, py_trees.common.Status.INVALID) for s in statuses):
            new_status = py_trees.common.Status.RUNNING
            self.current_child = self.children[-1]
        elif any(s == py_trees.common.Status.SUCCESS for s in statuses):
            new_status = py_trees.common.Status.SUCCESS
            self.current_child = next(
                (child for child in reversed(self.children) if child.status == py_trees.common.Status.SUCCESS),
                self.children[-1],
            )
        else:
            new_status = py_trees.common.Status.FAILURE
            self.current_child = self.children[-1]

        if new_status != py_trees.common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_home_id(entry_point: str) -> str:
    """Extract home ID number from entry point URI.

    e.g. 'http://localhost:8080/workspaces/home0#workspace' -> '0'
    """
    match = re.search(r"/workspaces/home(\d+)", entry_point)
    if not match:
        raise ValueError(f"Cannot extract home_id from entry point: {entry_point}")
    return match.group(1)


def _derive_sparql_url(entry_point: str) -> str:
    """Derive the SPARQL endpoint URL from the entry point.

    e.g. 'http://localhost:8080/workspaces/home0#workspace' -> 'http://localhost:8080/sparql'
    """
    parsed = urlparse(entry_point)
    return f"{parsed.scheme}://{parsed.netloc}/sparql"


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase (or camelCase) to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase."""
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def _cast_value(raw_value: str, schema_type_uri: str):
    """Cast a string value to the appropriate Python type based on JSON-Schema type URI."""
    if not raw_value:
        return raw_value
    schema_type = schema_type_uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower()
    try:
        if schema_type in ("integerschema", "integer"):
            return int(raw_value)
        if schema_type in ("numberschema", "number"):
            return float(raw_value)
        if schema_type in ("booleanschema", "boolean"):
            return raw_value.lower() in ("true", "1", "yes", "on")
    except (ValueError, TypeError):
        pass
    # Default: keep as string
    return raw_value


# ---------------------------------------------------------------------------
# SPARQL query template
# ---------------------------------------------------------------------------

_SPARQL_QUERY_TEMPLATE = """\
PREFIX ex: <http://example.org/>
PREFIX hctl: <https://www.w3.org/2019/wot/hypermedia#>
PREFIX hmas: <https://purl.org/hmas/>
PREFIX http: <http://www.w3.org/2011/http#>
PREFIX jsonschema: <https://www.w3.org/2019/wot/json-schema#>
PREFIX td: <https://www.w3.org/2019/wot/td#>

SELECT ?workspace ?artifact ?affordance_name ?target_uri ?parameter_name ?parameter_schema_type
WHERE {{
    ?workspace a {workspace_type} ;
              hmas:contains ?artifact .
    ?artifact a {artifact_type} ;
              td:hasActionAffordance ?affordance .
    ?affordance a {affordance_type} ;
                 td:name ?affordance_name ;
                 td:hasForm ?form .
    ?form hctl:hasTarget ?target_uri .
    {parameter_block}
}}"""

_PARAMETER_OPTIONAL_BLOCK = """\
    OPTIONAL {{
        ?affordance td:hasInputSchema ?inputSchema .
        ?inputSchema jsonschema:properties ?property .
        ?property jsonschema:propertyName ?parameter_name .
        ?property a ?parameter_schema_type .
        FILTER (?parameter_name = "{parameter_name}")
    }}"""

_PARAMETER_EMPTY_BLOCK = ""


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SparqlResolutionResult:
    """Result of running the SPARQL query for a single intent."""

    intent: StructuredIntent
    success: bool = False  # True if query returned at least one binding
    workspace_uri: Optional[str] = None
    artifact_uri: Optional[str] = None
    affordance_name: Optional[str] = None
    target_uri: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_schema_type: Optional[str] = None
    query: str = ""
    raw_bindings: list = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "success": self.success,
            "workspace_uri": self.workspace_uri,
            "artifact_uri": self.artifact_uri,
            "affordance_name": self.affordance_name,
            "target_uri": self.target_uri,
            "parameter_name": self.parameter_name,
            "parameter_schema_type": self.parameter_schema_type,
            "query": self.query,
            "bindings_count": len(self.raw_bindings),
            "bindings": self.raw_bindings,
            "error": self.error,
        }


@dataclass
class NeuroSymbolicRunResult:
    """Result of a single neuro-symbolic pipeline run."""

    # Intent extraction
    intents: list[StructuredIntent] = field(default_factory=list)
    intent_extraction_trace: Optional[dict] = None

    # SPARQL resolution
    resolution_results: list[SparqlResolutionResult] = field(default_factory=list)

    # Routing
    impossible_intents: list[StructuredIntent] = field(default_factory=list)
    # For intents flagged impossible via experience lookup, the matched entry
    # (None for intents that were routed impossible via SPARQL 0-bindings)
    impossible_matches: list[Optional[object]] = field(default_factory=list)
    set_intents: list[tuple[StructuredIntent, SparqlResolutionResult]] = field(
        default_factory=list
    )
    modify_intents: list[tuple[StructuredIntent, SparqlResolutionResult]] = field(
        default_factory=list
    )

    # Set-action subtree (built programmatically)
    set_actions_tree_ir: Optional[dict] = None

    # Modify-action subtree (built via LLM)
    modify_plan_trace: Optional[dict] = None
    modify_plan_time_seconds: float = 0.0
    modify_actions_tree_ir: Optional[dict] = None

    # Combined plan & execution
    combined_plan_ir: Optional[dict] = None
    execution_result: Optional[ExecutionResult] = None
    execution_backend: Optional[str] = None

    # Detected impossible (from SPARQL no-result and from LLM planning)
    detected_impossible: list[str] = field(default_factory=list)

    # Overall
    success: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        # Serialise impossible intents paired with their matched experience entry (if any)
        impossible_details = []
        for intent, match in zip(
            self.impossible_intents,
            self.impossible_matches + [None] * max(0, len(self.impossible_intents) - len(self.impossible_matches)),
        ):
            impossible_details.append({
                "intent": intent.to_dict(),
                "reason": "experience_match" if match is not None else "sparql_no_result",
                "matched_experience": match.to_dict() if match is not None else None,
            })

        return {
            "intents": [i.to_dict() for i in self.intents],
            "intent_extraction_trace": self.intent_extraction_trace,
            "resolution_results": [r.to_dict() for r in self.resolution_results],
            "impossible_count": len(self.impossible_intents),
            "impossible_details": impossible_details,
            "set_count": len(self.set_intents),
            "modify_count": len(self.modify_intents),
            "set_actions_tree_ir": self.set_actions_tree_ir,
            "modify_plan_trace": self.modify_plan_trace,
            "modify_plan_time_seconds": self.modify_plan_time_seconds,
            "modify_actions_tree_ir": self.modify_actions_tree_ir,
            "combined_plan_ir": self.combined_plan_ir,
            "execution_backend": self.execution_backend,
            "execution": (
                self.execution_result.to_dict() if self.execution_result else None
            ),
            "detected_impossible": self.detected_impossible,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class NeuroSymbolicRunner:
    """
    Neuro-Symbolic planning runner.

    Uses ontology-grounded intent extraction + direct SPARQL resolution
    to avoid the full discovery pipeline for "set" intents.  Only "modify"
    (relative-change) intents fall back to LLM-based BT code generation.

    Experience is used only to remember impossible requests per home_id.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client,
        engine: ExperienceEngine,
        intent_extractor: IntentExtractor,
    ):
        self.config = config
        self.client = client
        self.engine = engine
        self.intent_extractor = intent_extractor
        logger.info("NeuroSymbolicRunner initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        entry_point: str,
        home_id: str,
        test_id: str,
    ) -> NeuroSymbolicRunResult:
        """
        Run the neuro-symbolic planning pipeline for a single goal.

        Steps:
          1. Extract intents (with parameter & value fields)
          2. Check experience engine for previously-known impossible intents
          3. Resolve each non-impossible intent via SPARQL
          4. Route: impossible / set / modify
          5. Build set-actions BT programmatically
          6. Build modify-actions BT via LLM (if any)
          7. Combine & execute

        Returns:
            NeuroSymbolicRunResult with full pipeline traces.
        """
        result = NeuroSymbolicRunResult()
        start_time = time.time()

        logger.info(f"Starting neuro-symbolic pipeline [test_id={test_id}]")

        try:
            sparql_url = _derive_sparql_url(entry_point)
            home_id_num = _extract_home_id(entry_point)

            # ----------------------------------------------------------------
            # Step 1: Intent Extraction
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # Step 2: Check experience engine for known-impossible intents
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 2: Experience — Impossible Intent Lookup")
            logger.info("=" * 60)

            active_intents: list[StructuredIntent] = []
            for intent in intents:
                matched_entry = self._is_known_impossible(intent, home_id)
                if matched_entry is not None:
                    logger.info(
                        f"Intent known-impossible from experience: {intent.text_intent!r} "
                        f"(matched entry from test {matched_entry.source_test_id!r}, "
                        f"home={matched_entry.home_id!r})"
                    )
                    result.impossible_intents.append(intent)
                    result.impossible_matches.append(matched_entry)
                    result.detected_impossible.append(intent.text_intent)
                else:
                    active_intents.append(intent)

            # ----------------------------------------------------------------
            # Step 3: SPARQL Resolution
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 3: SPARQL Resolution")
            logger.info("=" * 60)

            resolution_results: list[SparqlResolutionResult] = []
            for intent in active_intents:
                res = self._resolve_intent(intent, sparql_url, home_id_num)
                resolution_results.append(res)
                logger.info(
                    f"  Intent {intent.text_intent!r}: "
                    f"success={res.success}, target={res.target_uri or '(none)'}"
                )

            result.resolution_results = resolution_results

            # ----------------------------------------------------------------
            # Step 4: Route intents
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 4: Routing")
            logger.info("=" * 60)

            for intent, res in zip(active_intents, resolution_results):
                if not res.success:
                    logger.info(f"IMPOSSIBLE (no SPARQL result): {intent.text_intent!r}")
                    result.impossible_intents.append(intent)
                    result.impossible_matches.append(None)  # SPARQL-based, no experience entry
                    result.detected_impossible.append(intent.text_intent)
                elif intent.verb == "set":
                    result.set_intents.append((intent, res))
                else:  # "modify"
                    result.modify_intents.append((intent, res))

            logger.info(
                f"Routing: {len(result.impossible_intents)} impossible, "
                f"{len(result.set_intents)} set, "
                f"{len(result.modify_intents)} modify"
            )

            # Early exit: all intents impossible
            if not result.set_intents and not result.modify_intents:
                logger.info("All intents impossible — skipping execution")
                result.success = True
                result.duration_seconds = time.time() - start_time
                return result

            # ----------------------------------------------------------------
            # Step 5: Build set-actions BT (programmatic)
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 5: Build Set-Actions BT")
            logger.info("=" * 60)

            set_tree: Optional[py_trees.behaviour.Behaviour] = None
            if result.set_intents:
                set_tree = self._build_set_tree(result.set_intents)
                result.set_actions_tree_ir = py_tree_to_json_ir(set_tree)
                logger.info(
                    f"Built set-actions tree with {len(result.set_intents)} nodes"
                )

            # ----------------------------------------------------------------
            # Step 6: Build modify-actions BT (LLM)
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 6: Build Modify-Actions BT (LLM)")
            logger.info("=" * 60)

            modify_tree: Optional[py_trees.behaviour.Behaviour] = None
            if result.modify_intents:
                modify_t0 = time.time()
                modify_tree, modify_trace = self._build_modify_tree(
                    result.modify_intents, entry_point, goal
                )
                result.modify_plan_trace = modify_trace
                result.modify_plan_time_seconds = time.time() - modify_t0

                if modify_tree is not None:
                    result.modify_actions_tree_ir = py_tree_to_json_ir(modify_tree)
                    logger.info("Modify-actions BT built successfully")

                # Collect impossible sub-goals reported by LLM
                if modify_trace and modify_trace.get("detected_impossible"):
                    result.detected_impossible.extend(
                        modify_trace["detected_impossible"]
                    )

            # ----------------------------------------------------------------
            # Step 7: Combine & Execute
            # ----------------------------------------------------------------
            logger.info("=" * 60)
            logger.info("STEP 7: Combine & Execute")
            logger.info("=" * 60)

            trace_subtrees = [t for t in [set_tree, modify_tree] if t is not None]

            if not trace_subtrees:
                modify_error = None
                if result.modify_plan_trace and isinstance(result.modify_plan_trace, dict):
                    modify_error = result.modify_plan_trace.get("error")

                if modify_error:
                    result.error = modify_error
                    logger.warning(
                        "No executable subtrees due to modify planning/code error: %s",
                        modify_error,
                    )
                elif result.impossible_intents or result.detected_impossible:
                    logger.info(
                        "No executable subtrees (all impossible) — skipping execution"
                    )
                    result.success = True
                else:
                    result.error = "No executable BT produced"
                    logger.warning(result.error)
                result.duration_seconds = time.time() - start_time
                return result

            # Build combined JSON-IR snapshot without re-parenting runtime nodes.
            if len(trace_subtrees) > 1:
                children_ir = []
                if result.set_actions_tree_ir is not None:
                    children_ir.append(result.set_actions_tree_ir)
                if result.modify_actions_tree_ir is not None:
                    children_ir.append(result.modify_actions_tree_ir)
                result.combined_plan_ir = {
                    "type": "parallel",
                    "name": "CombinedPlan",
                    "policy": "success_on_one",
                    "children": children_ir,
                }
            else:
                # Single subtree: keep existing serialization
                result.combined_plan_ir = (
                    result.set_actions_tree_ir
                    if result.set_actions_tree_ir is not None
                    else result.modify_actions_tree_ir
                )

            # Build execution tree from the in-memory subtrees.
            # Then recursively retune every Parallel(SuccessOnOne) to deterministic
            # wait-all semantics, without re-parenting nodes.
            exec_subtrees = [t for t in [set_tree, modify_tree] if t is not None]
            if len(exec_subtrees) > 1:
                exec_tree = py_trees.composites.Parallel(
                    name="CombinedPlan",
                    policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
                    children=exec_subtrees,
                )
            else:
                exec_tree = exec_subtrees[0]

            changed = self._apply_deterministic_policy_recursively(exec_tree)
            if changed:
                logger.info(
                    "Applied deterministic policy to %d Parallel(SuccessOnOne) node(s)",
                    changed,
                )

            # Execute directly from the in-memory py_trees object.
            # This avoids JSON-IR recompilation issues for custom compute nodes
            # (inline Behaviour subclasses) that are not representable in IR.
            exec_result = self._execute_tree_direct(
                exec_tree,
                max_ticks=self.config.execution.max_ticks,
            )
            result.execution_backend = "direct_tree"
            result.execution_result = exec_result

            if exec_result.success:
                logger.info(f"Execution SUCCESS in {exec_result.ticks} ticks")
                result.success = True
            else:
                logger.warning(
                    f"Execution FAILED: {exec_result.error or exec_result.final_status}"
                )
                result.error = exec_result.error

        except Exception as e:
            logger.exception("Neuro-symbolic pipeline failed")
            result.error = str(e)

        result.duration_seconds = time.time() - start_time
        return result

    @staticmethod
    def _execute_tree_direct(
        tree: py_trees.behaviour.Behaviour,
        max_ticks: int,
    ) -> ExecutionResult:
        """Execute a py_trees root directly and return ExecutionResult."""
        tree.setup_with_descendants()

        exec_result = ExecutionResult(
            success=False,
            tree_name=tree.name,
            ticks=0,
            tick_history=[],
        )

        try:
            for tick in range(max_ticks):
                exec_result.ticks = tick + 1
                tree.tick_once()

                status_name = tree.status.name
                exec_result.tick_history.append(status_name)

                if tree.status == py_trees.common.Status.SUCCESS:
                    exec_result.final_status = "SUCCESS"
                    exec_result.success = True
                    break
                if tree.status == py_trees.common.Status.FAILURE:
                    exec_result.final_status = "FAILURE"
                    exec_result.success = False
                    break
            else:
                exec_result.final_status = "RUNNING (max ticks reached)"
        except Exception as exc:
            exec_result.success = False
            exec_result.error = str(exc)
            exec_result.final_status = "ERROR"
        finally:
            tree.shutdown()

        return exec_result

    @classmethod
    def _apply_deterministic_policy_recursively(
        cls,
        tree: py_trees.behaviour.Behaviour,
    ) -> int:
        """
        Recursively retune every Parallel(SuccessOnOne) node in-place.

        This updates semantics without creating new parents, preventing the
        "already has parent" errors from node reattachment.

        Returns:
            Number of parallel nodes retuned.
        """
        changed = 0

        if (
            isinstance(tree, py_trees.composites.Parallel)
            and isinstance(tree.policy, py_trees.common.ParallelPolicy.SuccessOnOne)
            and not isinstance(tree, _AnySuccessElseAllFailureParallel)
        ):
            tree.__class__ = _AnySuccessElseAllFailureParallel
            changed += 1

        if isinstance(tree, py_trees.composites.Composite):
            for child in tree.children:
                changed += cls._apply_deterministic_policy_recursively(child)

        return changed

    def store_infeasible(
        self,
        intent: StructuredIntent,
        test_id: str,
        home_id: str = "",
    ) -> bool:
        """
        Store a confirmed infeasible intent in the experience engine.

        Only infeasible intents are stored — this runner does not store
        successful experiences (those are resolved symbolically each time).

        Returns:
            True if stored (not a duplicate).
        """
        if self.engine.has_identical(intent):
            return False
        self.engine.add_infeasible(intent, test_id, home_id=home_id)
        self.engine.save()
        return True

    # ------------------------------------------------------------------
    # Internal: experience lookup
    # ------------------------------------------------------------------

    def _is_known_impossible(
        self, intent: StructuredIntent, home_id: str
    ) -> Optional[ExperienceEntry]:
        """
        Return the matching infeasible ExperienceEntry if one exists for this
        home, or None otherwise.

        Scoping rules:
          - An entry stored with a specific home_id only matches that home.
          - An entry stored with home_id="" matches any home (global infeasible).
          - The lookup home_id must always be provided; if it is empty the
            check falls through to text_intent equality only (no home filter).
        """
        for entry in self.engine.get_by_slot_key(intent.slot_key()):
            if not entry.is_infeasible:
                continue
            # Skip entries scoped to a different home
            if entry.home_id and home_id and entry.home_id != home_id:
                continue
            if entry.text_intent == intent.text_intent:
                return entry
        return None

    # ------------------------------------------------------------------
    # Internal: SPARQL resolution
    # ------------------------------------------------------------------

    def _build_sparql_query(self, intent: StructuredIntent) -> str:
        """Construct the SPARQL query for a single structured intent."""
        if intent.parameter:
            param_block = _PARAMETER_OPTIONAL_BLOCK.format(
                parameter_name=intent.parameter
            )
        else:
            param_block = _PARAMETER_EMPTY_BLOCK

        return _SPARQL_QUERY_TEMPLATE.format(
            workspace_type=intent.workspace_type,
            artifact_type=intent.artifact_type,
            affordance_type=intent.affordance_type,
            parameter_block=param_block,
        )

    def _resolve_intent(
        self,
        intent: StructuredIntent,
        sparql_url: str,
        home_id_num: str,
    ) -> SparqlResolutionResult:
        """Execute the SPARQL query for an intent and parse the first binding."""
        query = self._build_sparql_query(intent)
        res = SparqlResolutionResult(intent=intent, query=query)

        try:
            resp = requests.post(
                sparql_url,
                json={"home_id": home_id_num, "query": query},
                timeout=30,
            )
            resp.raise_for_status()
            sparql_result = resp.json()
        except requests.RequestException as exc:
            error_msg = str(exc)
            try:
                error_msg = resp.json().get("detail", error_msg)
            except Exception:
                pass
            logger.warning(
                f"SPARQL query failed for intent {intent.text_intent!r}: {error_msg}"
            )
            res.error = error_msg
            return res

        bindings = sparql_result.get("results", {}).get("bindings", [])
        res.raw_bindings = bindings

        if not bindings:
            logger.info(f"No SPARQL bindings for intent: {intent.text_intent!r}")
            return res

        # Use the first binding
        b = bindings[0]
        res.success = True
        res.workspace_uri = b.get("workspace", {}).get("value")
        res.artifact_uri = b.get("artifact", {}).get("value")
        res.affordance_name = b.get("affordance_name", {}).get("value")
        res.target_uri = b.get("target_uri", {}).get("value")
        res.parameter_name = b.get("parameter_name", {}).get("value")
        res.parameter_schema_type = b.get("parameter_schema_type", {}).get("value")

        return res

    # ------------------------------------------------------------------
    # Internal: set-actions BT construction
    # ------------------------------------------------------------------

    def _build_set_tree(
        self,
        set_intents: list[tuple[StructuredIntent, SparqlResolutionResult]],
    ) -> py_trees.behaviour.Behaviour:
        """
        Build a Parallel(SuccessOnOne) tree from all "set" intents.

        Each intent becomes a single ActionAffordanceNode with either:
          - no parameters (parameterless commands like TurnOn)
          - static parameters cast to the correct type
        """
        nodes: list[py_trees.behaviour.Behaviour] = []

        for intent, res in set_intents:
            node_name = _snake_to_camel(
                res.affordance_name or intent.affordance_type
            )
            target_uri = res.target_uri

            if res.parameter_name and intent.value is not None:
                # Cast value to correct type
                param_value = _cast_value(
                    intent.value,
                    res.parameter_schema_type or "",
                )
                node = ActionAffordanceNode(
                    name=node_name,
                    action_url=target_uri,
                    parameters={res.parameter_name: param_value},
                )
                logger.info(
                    f"Set node: {node_name} @ {target_uri} "
                    f"[{res.parameter_name}={param_value!r}]"
                )
            else:
                # Parameterless action (e.g. TurnOn, Close)
                node = ActionAffordanceNode(
                    name=node_name,
                    action_url=target_uri,
                )
                logger.info(f"Set node (parameterless): {node_name} @ {target_uri}")

            nodes.append(node)

        if len(nodes) == 1:
            return nodes[0]

        return py_trees.composites.Parallel(
            name="SetActions",
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
            children=nodes,
        )

    # ------------------------------------------------------------------
    # Internal: modify-actions BT via LLM
    # ------------------------------------------------------------------

    def _build_modify_tree(
        self,
        modify_intents: list[tuple[StructuredIntent, SparqlResolutionResult]],
        entry_point: str,
        original_goal: str,
    ) -> tuple[Optional[py_trees.behaviour.Behaviour], dict]:
        """
        Build a BT for modify intents using LLM code generation.

        The DiscoveryResult is constructed directly from the SPARQL bindings
        already stored in SparqlResolutionResult — no second discovery pass is
        needed.  We only make targeted hmas_client calls to enrich each
        artifact with its property URIs and to read current property values.

        Returns (tree or None, trace dict).
        """
        trace: dict = {
            "phase": "modify_planning",
            "intents": [i.to_dict() for i, _ in modify_intents],
            "planning": None,
            "detected_impossible": [],
            "error": None,
        }

        # Build a structured goal string for the modify intents
        goal_lines: list[str] = []
        for idx, (intent, res) in enumerate(modify_intents, start=1):
            goal_lines.append(f"**Intent {idx}**")
            goal_lines.append(f"text_intent: {intent.text_intent}")
            goal_lines.append(f"action.affordance_type: {intent.affordance_type}")
            goal_lines.append(f"action.verb: {intent.verb}")
            goal_lines.append(f"action.parameter: {intent.parameter}")
            goal_lines.append(f"action.value: {intent.value}")
            goal_lines.append(f"target.artifact_type: {intent.artifact_type}")
            goal_lines.append(f"target.workspace_type: {intent.workspace_type}")
            if res.target_uri:
                goal_lines.append(f"resolved.action_url: {res.target_uri}")
            if res.artifact_uri:
                goal_lines.append(f"resolved.artifact_uri: {res.artifact_uri}")
            goal_lines.append("")
        modify_goal = "\n".join(goal_lines).rstrip()

        # Build DiscoveryResult from the SPARQL bindings already at hand
        try:
            discovery_result = self._build_discovery_result_from_resolutions(
                modify_intents, entry_point
            )
        except Exception as exc:
            logger.error(f"Failed to build discovery result from SPARQL bindings: {exc}")
            trace["error"] = str(exc)
            return None, trace

        # Plan using modify-only prompt
        from ..config import PlanningConfig, ReasoningConfig, OutputConfig

        planning_config = PlanningConfig(
            reasoning=ReasoningConfig(enabled=False),
            output=OutputConfig(format="python_code"),
            prompt_strategy="detailed_structured_modify_only",
        )

        planner = create_planner(planning_config)
        try:
            planning_result = planner.plan(
                goal=modify_goal,
                discovery=discovery_result,
                client=self.client,
                model_config=self.config.model,
            )
        except Exception as exc:
            logger.error(f"Planning for modify intents failed: {exc}")
            trace["error"] = str(exc)
            return None, trace

        trace["planning"] = planning_result.to_dict()
        trace["discovery"] = discovery_result.to_dict()

        if not planning_result.success:
            logger.warning(f"Modify planning failed: {planning_result.error}")
            trace["error"] = planning_result.error
            return None, trace

        plan = planning_result.plan

        # Generated code is non-deterministic; defensively normalize a known
        # formatting glitch where top-level `tree = ...` is emitted with
        # accidental leading indentation.
        if plan.is_python_code and isinstance(plan.content, str):
            normalized_code = self._normalize_top_level_tree_assignment(plan.content)
            if normalized_code != plan.content:
                logger.info(
                    "Normalized generated modify code by de-indenting top-level tree assignment"
                )
                plan.content = normalized_code
                trace["planning"] = planning_result.to_dict()

        # Collect impossible sub-goals from the plan
        if plan.detected_impossible:
            trace["detected_impossible"] = plan.detected_impossible

        if not plan.is_python_code:
            logger.warning("Modify planning produced non-code output")
            trace["error"] = "Unexpected non-code plan format"
            return None, trace

        # Skip if only impossible comments, no executable tree
        if (
            plan.detected_impossible
            and not self._code_defines_tree_or_builder(plan.content)
        ):
            logger.info(
                "Modify plan contains only impossible sub-goals — no executable tree"
            )
            return None, trace

        # Execute code to get tree
        try:
            executor = create_executor(
                output_format=plan.format,
                max_ticks=self.config.execution.max_ticks,
            )
            # Validate by constructing the tree object only (no ticking here).
            # A plan can be executable as code while still failing when actually
            # ticked against environment state; that should be handled at final
            # execution stage, not collapsed into "No executable BT produced".
            assert isinstance(executor, CodeExecutor)
            tree = executor._execute_code(plan.content, unconstrained=False)
            return tree, trace

        except Exception as exc:
            logger.error(f"Failed to build modify tree from code: {exc}")
            trace["error"] = str(exc)
            return None, trace

    # ------------------------------------------------------------------
    # Internal: DiscoveryResult construction from SPARQL bindings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_discovery_result_from_resolutions(
        intents_with_results: list[tuple[StructuredIntent, SparqlResolutionResult]],
        entry_point: str,
    ):
        """
        Construct a DiscoveryResult directly from already-resolved SPARQL bindings.

        For each resolved intent we know the workspace URI, artifact URI, and
        action affordance URI.  We call hmas_client to enrich each unique
        artifact with its property affordances and action schemas, then read
        all current property values — exactly what the "all" state strategy
        would do, but restricted to the artifacts we already know about.

        This avoids re-running the LLM-based SPARQL query generation step.
        """
        from datetime import datetime

        from ..discovery.base import (
            Affordance,
            Artifact,
            CapabilityModel,
            EnvironmentState,
            DiscoveryResult,
        )
        from ..hmas_client import (
            list_actions,
            list_properties,
            get_artifact_semantic_type,
            get_workspace_semantic_type,
            get_property_by_uri,
            GetPropertyError,
        )

        capability_model = CapabilityModel(entry_point=entry_point)

        for intent, res in intents_with_results:
            if not res.success or not res.artifact_uri:
                continue

            art_uri = res.artifact_uri
            ws_uri = res.workspace_uri or entry_point

            # Register workspace
            ws = capability_model.get_or_create_workspace(ws_uri)
            if art_uri not in ws.artifact_uris:
                ws.artifact_uris.append(art_uri)

            # Register artifact (once per URI)
            if art_uri not in capability_model.artifacts:
                art_name = art_uri.split("/")[-1].replace("#artifact", "")
                capability_model.artifacts[art_uri] = Artifact(
                    name=art_name,
                    uri=art_uri,
                    workspace=ws_uri,
                )

            artifact = capability_model.artifacts[art_uri]

            # Add the resolved action affordance
            existing_uris = {a.uri for a in artifact.actions}
            if res.target_uri and res.target_uri not in existing_uris:
                artifact.actions.append(Affordance(
                    name=res.affordance_name or "",
                    uri=res.target_uri,
                    command=intent.text_intent,
                    semantic_type=intent.affordance_type,
                ))

        # Enrich each artifact with semantic type, full action schemas, and properties
        for art_uri, artifact in capability_model.artifacts.items():
            # Workspace semantic type
            ws = capability_model.workspaces.get(artifact.workspace)
            if ws and not ws.semantic_type:
                try:
                    ws.semantic_type = get_workspace_semantic_type(artifact.workspace)
                except Exception as exc:
                    logger.warning(f"Could not fetch workspace type for {artifact.workspace}: {exc}")

            # Artifact semantic type
            if not artifact.semantic_type:
                try:
                    artifact.semantic_type = get_artifact_semantic_type(art_uri)
                except Exception as exc:
                    logger.warning(f"Could not fetch artifact type for {art_uri}: {exc}")

            # Property affordances (needed so LLM can read current values)
            try:
                props = list_properties(art_uri)
                artifact.properties = [
                    Affordance(
                        name=p["name"],
                        uri=p["uri"],
                        schema=p.get("output_schema", {}),
                        semantic_type=p.get("semantic_type"),
                    )
                    for p in props
                ]
            except Exception as exc:
                logger.warning(f"Could not fetch properties for {art_uri}: {exc}")

            # Enrich action schemas
            try:
                all_actions = list_actions(art_uri)
                schema_map = {a["uri"]: a.get("input_schema", {}) for a in all_actions}
                name_map = {a["uri"]: a["name"] for a in all_actions}
                for action in artifact.actions:
                    if action.uri in schema_map:
                        action.schema = schema_map[action.uri]
                    if action.uri in name_map:
                        action.name = name_map[action.uri]
            except Exception as exc:
                logger.warning(f"Could not fetch action schemas for {art_uri}: {exc}")

        # Read current property values for all discovered properties
        state = EnvironmentState(timestamp=datetime.now().isoformat())
        for artifact in capability_model.artifacts.values():
            for prop in artifact.properties:
                try:
                    state.property_values[prop.uri] = get_property_by_uri(prop.uri)
                except GetPropertyError as exc:
                    state.errors[prop.uri] = str(exc)
                except Exception as exc:
                    state.errors[prop.uri] = str(exc)

        logger.info(
            f"Built discovery result from SPARQL bindings: "
            f"{len(capability_model.artifacts)} artifacts, "
            f"{len(state.property_values)} property values"
        )
        return DiscoveryResult(affordances=capability_model, state=state)

    # ------------------------------------------------------------------
    # Internal: code analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_top_level_tree_assignment(code: str) -> str:
        """
        Normalize a common LLM formatting glitch for Python code generation.

        If parsing fails with "unexpected indent" and the offending line is an
        indented top-level `tree = ...` assignment, strip indentation from that
        line and return the fixed code. Otherwise return code unchanged.
        """
        if not code or not code.strip():
            return code

        try:
            ast.parse(code)
            return code
        except SyntaxError as exc:
            if "unexpected indent" not in str(exc):
                return code

            line_no = exc.lineno or 0
            lines = code.splitlines()
            if line_no < 1 or line_no > len(lines):
                return code

            offending = lines[line_no - 1]
            if not re.match(r"^[ \t]+tree\s*=", offending):
                return code

            lines[line_no - 1] = offending.lstrip()
            fixed = "\n".join(lines)
            if code.endswith("\n"):
                fixed += "\n"

            try:
                ast.parse(fixed)
                return fixed
            except SyntaxError:
                return code

    @staticmethod
    def _code_defines_tree_or_builder(code: str) -> bool:
        """Return True if code defines an executable BT entrypoint."""
        if not code or not code.strip():
            return False
        try:
            module = ast.parse(code)
        except SyntaxError:
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
