"""
Agentic query affordance discovery strategy.

Uses an LLM to generate SPARQL queries based on ontology documentation,
then executes them programmatically against the smart home simulator's
/sparql endpoint. The LLM decomposes user goals into individual commands
and produces a SPARQL query per command.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from ...config import ModelConfig, get_model_kwargs
from ...hmas_client import (
    get_artifact_semantic_type,
    get_workspace_semantic_type,
    list_actions,
    list_properties,
)
from ..base import Affordance, Artifact, CapabilityModel

logger = logging.getLogger(__name__)

# Path to the semantic query prompt
SEMANTIC_QUERY_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "ontologies"
    / "semantic-query-prompt.txt"
)

# Path to the structured semantic query prompt (used with structured_goal)
SEMANTIC_QUERY_STRUCTURED_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "ontologies"
    / "semantic-query-structured-prompt.txt"
)


def _extract_home_id(entry_point: str) -> str:
    """Extract home ID number from entry point URI.

    e.g. 'http://localhost:8080/workspaces/home0#workspace' -> '0'
    """
    match = re.search(r"/workspaces/home(\d+)", entry_point)
    if not match:
        raise ValueError(
            f"Cannot extract home_id from entry point: {entry_point}"
        )
    return match.group(1)


def _derive_sparql_url(entry_point: str) -> str:
    """Derive the SPARQL endpoint URL from the entry point.

    e.g. 'http://localhost:8080/workspaces/home0#workspace' -> 'http://localhost:8080/sparql'
    """
    parsed = urlparse(entry_point)
    return f"{parsed.scheme}://{parsed.netloc}/sparql"


def _parse_llm_queries(response_text: str) -> list[dict]:
    """Parse the LLM response to extract the queries JSON.

    Handles both raw JSON and markdown-fenced JSON blocks.
    Returns list of dicts with 'command' and 'query' keys.
    """
    # Try to extract from markdown code block
    code_block = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
    )
    json_str = (
        code_block.group(1).strip() if code_block else response_text.strip()
    )

    parsed = json.loads(json_str)

    if isinstance(parsed, dict) and "queries" in parsed:
        return parsed["queries"]
    if isinstance(parsed, list):
        return parsed

    raise ValueError(f"Unexpected JSON structure: {type(parsed)}")


class AgenticQueryAffordanceDiscovery:
    """
    Agentic query discovery strategy.

    Two-phase approach:
    1. LLM generates SPARQL queries from the goal using ontology documentation
    2. Queries are executed programmatically against the /sparql endpoint
    """

    def __init__(
        self,
        client: OpenAI,
        model_config: ModelConfig,
        semantic_query_prompt_path: Optional[Path] = None,
    ):
        self.client = client
        self.model_config = model_config
        self.model = model_config.name
        self.exploration_trace: list[dict] = []

        # Load the semantic query prompt (use custom path if provided)
        prompt_path = semantic_query_prompt_path or SEMANTIC_QUERY_PROMPT_PATH
        self.semantic_query_prompt = Path(prompt_path).read_text()

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,
    ) -> CapabilityModel:
        """
        Build capability model by generating and executing SPARQL queries.

        Args:
            entry_point: Root workspace URI
            goal: The user's goal (required)

        Returns:
            CapabilityModel with discovered affordances grouped by workspace/artifact
        """
        if not goal:
            raise ValueError("Agentic query discovery requires a goal")

        logger.info(f"Starting agentic query discovery for goal: {goal}")
        capability_model = CapabilityModel(entry_point=entry_point)
        self.exploration_trace = []

        home_id = _extract_home_id(entry_point)
        sparql_url = _derive_sparql_url(entry_point)

        # Phase 1: Generate SPARQL queries via LLM
        queries = self._generate_queries(goal)

        # Phase 2: Execute each query and build capability model
        for query_entry in queries:
            command = query_entry.get("command", "")
            query = query_entry.get("query", "")

            if not query:
                logger.warning(f"Empty query for command: {command}")
                continue

            self._execute_and_process(
                sparql_url, home_id, command, query, capability_model
            )

        # Enrich discovered artifacts with full schemas and properties
        self._enrich_artifacts(capability_model)

        logger.info(
            f"Query discovery complete: {len(capability_model.artifacts)} artifacts, "
            f"{len(capability_model.infeasible_commands)} infeasible commands"
        )
        return capability_model

    def get_exploration_trace(self) -> list[dict]:
        """Return the exploration trace for visualization."""
        return self.exploration_trace

    def _generate_queries(self, goal: str) -> list[dict]:
        """Call the LLM to decompose the goal into SPARQL queries.

        Returns:
            List of dicts with 'command' and 'query' keys.
        """
        logger.info("Generating SPARQL queries from goal")

        api_kwargs = get_model_kwargs(
            self.model, model_config=self.model_config
        )
        api_kwargs.update(
            {
                "messages": [
                    {"role": "system", "content": self.semantic_query_prompt},
                    {"role": "user", "content": goal},
                ],
            }
        )

        response = self.client.chat.completions.create(**api_kwargs)
        response_text = response.choices[0].message.content

        # Record the LLM response in the trace
        self.exploration_trace.append(
            {
                "phase": "query_generation",
                "goal": goal,
                "llm_response": response_text,
                "queries": None,  # filled below
            }
        )

        try:
            queries = _parse_llm_queries(response_text)
            self.exploration_trace[-1]["queries"] = queries
            logger.info(
                f"Generated {len(queries)} queries for {len(queries)} commands"
            )
            return queries
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM query response: {e}")
            self.exploration_trace[-1]["error"] = str(e)
            return []

    def _execute_and_process(
        self,
        sparql_url: str,
        home_id: str,
        command: str,
        query: str,
        model: CapabilityModel,
    ) -> None:
        """Execute a SPARQL query and process its results into the model."""
        logger.debug(f"Executing SPARQL query for command: {command}")

        trace_entry = {
            "phase": "query_execution",
            "command": command,
            "query": query,
            "result": None,
        }

        try:
            resp = requests.post(
                sparql_url,
                json={"home_id": home_id, "query": query},
                timeout=30,
            )
            resp.raise_for_status()
            sparql_result = resp.json()
        except requests.RequestException as e:
            error_msg = str(e)
            try:
                error_msg = resp.json().get("detail", error_msg)
            except Exception:
                pass
            logger.warning(
                f"SPARQL query failed for command '{command}': {error_msg}"
            )
            trace_entry["result"] = {"error": error_msg, "bindings_count": 0}
            trace_entry["error"] = error_msg
            self.exploration_trace.append(trace_entry)
            model.mark_infeasible(
                command, query, reason=f"query_error: {error_msg}"
            )
            return

        bindings = sparql_result.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info(f"Zero bindings for command: {command}")
            model.mark_infeasible(command, query)
            trace_entry["result"] = {
                "bindings_count": 0,
                "message": "No matching devices or affordances found",
            }
            self.exploration_trace.append(trace_entry)
            return

        # Process bindings into capability model
        self._process_bindings(bindings, command, model)

        trace_entry["result"] = {
            "bindings_count": len(bindings),
            "bindings": bindings,
        }
        self.exploration_trace.append(trace_entry)

    def _process_bindings(
        self,
        bindings: list[dict],
        command: str,
        model: CapabilityModel,
    ) -> None:
        """Process SPARQL result bindings into the capability model,
        grouped by workspace then by artifact."""
        for binding in bindings:
            ws_uri = binding.get("workspace", {}).get("value")
            art_uri = binding.get("artifact", {}).get("value")
            affordance_name = binding.get("affordance_name", {}).get("value")
            target_uri = binding.get("target_uri", {}).get("value")

            if not art_uri or not affordance_name or not target_uri:
                logger.warning(f"Incomplete binding, skipping: {binding}")
                continue

            # Derive workspace URI if not provided
            if not ws_uri:
                parts = art_uri.split("/artifacts/")
                ws_uri = (
                    parts[0] + "#workspace"
                    if len(parts) == 2
                    else model.entry_point
                )

            # Register workspace -> artifact mapping
            ws = model.get_or_create_workspace(ws_uri)
            if art_uri not in ws.artifact_uris:
                ws.artifact_uris.append(art_uri)

            # Create or update artifact
            if art_uri not in model.artifacts:
                art_name = art_uri.split("/")[-1].replace("#artifact", "")
                model.artifacts[art_uri] = Artifact(
                    name=art_name,
                    uri=art_uri,
                    workspace=ws_uri,
                )

            artifact = model.artifacts[art_uri]

            # Add action affordance if not already present
            existing_action_uris = {a.uri for a in artifact.actions}
            if target_uri not in existing_action_uris:
                artifact.actions.append(
                    Affordance(
                        name=affordance_name,
                        uri=target_uri,
                        command=command,
                    )
                )

    def _enrich_artifacts(self, model: CapabilityModel) -> None:
        """Enrich discovered artifacts with full action schemas, property affordances, and semantic types."""
        # Enrich workspace semantic types
        for ws_uri, workspace in model.workspaces.items():
            if not workspace.semantic_type:
                try:
                    workspace.semantic_type = get_workspace_semantic_type(
                        ws_uri
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch workspace type for {ws_uri}: {e}"
                    )

        for art_uri, artifact in model.artifacts.items():
            # Fetch artifact semantic type
            if not artifact.semantic_type:
                try:
                    artifact.semantic_type = get_artifact_semantic_type(art_uri)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch artifact type for {art_uri}: {e}"
                    )

            # Fetch all property affordances
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
            except Exception as e:
                logger.warning(f"Failed to fetch properties for {art_uri}: {e}")

            # Enrich action schemas and semantic types
            try:
                all_actions = list_actions(art_uri)
                action_schema_map = {
                    a["uri"]: a.get("input_schema", {}) for a in all_actions
                }
                action_name_map = {a["uri"]: a["name"] for a in all_actions}
                action_type_map = {
                    a["uri"]: a.get("semantic_type") for a in all_actions
                }
                for action in artifact.actions:
                    if action.uri in action_schema_map:
                        action.schema = action_schema_map[action.uri]
                    if action.uri in action_name_map:
                        action.name = action_name_map[action.uri]
                    if action.uri in action_type_map:
                        action.semantic_type = action_type_map[action.uri]
            except Exception as e:
                logger.warning(
                    f"Failed to fetch action schemas for {art_uri}: {e}"
                )
