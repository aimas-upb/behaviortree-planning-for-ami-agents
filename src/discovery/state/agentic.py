"""
Agentic state gathering strategy.

Uses LLM to interactively explore and read properties based on the goal.
"""

import json
from typing import Optional
import logging
from datetime import datetime

from openai import OpenAI

from ..base import CapabilityModel, EnvironmentState
from hmas_client import get_property_by_uri, GetPropertyError, list_properties

logger = logging.getLogger(__name__)


STATE_DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_artifact_properties",
            "description": "List all available properties of an artifact. Use this to discover what properties can be read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_uri": {
                        "type": "string",
                        "description": "The artifact URI to list properties from",
                    }
                },
                "required": ["artifact_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_property",
            "description": "Read the current value of a property from the environment. Use EXACT URIs from list_artifact_properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_uri": {
                        "type": "string",
                        "description": "The EXACT property URI to read (from list_artifact_properties)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you want to read this property",
                    }
                },
                "required": ["property_uri", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_gathering",
            "description": "Call this when you have gathered enough state information for the goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what state was gathered and why it's sufficient",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

STATE_DISCOVERY_SYSTEM_PROMPT = """You are gathering state information from a smart home environment to help plan actions for a goal.

You have tools to:
1. list_artifact_properties - Discover what properties an artifact has (ALWAYS use this first!)
2. read_property - Read the current value of a property (use EXACT URIs from list_artifact_properties)
3. done_gathering - Signal you've gathered enough state information

STRATEGY:
1. For each artifact relevant to the goal, FIRST call list_artifact_properties to discover its properties
2. Then call read_property with the EXACT URIs returned (do NOT guess or modify URIs!)
3. Read properties that are directly relevant to the goal (e.g., current brightness, state, temperature)
4. Stop when you have enough information to plan effectively

IMPORTANT:
- ALWAYS use list_artifact_properties before read_property
- Use EXACT property URIs from the list - do NOT modify or guess URIs
- If a property read fails, check that you're using the correct URI

Goal: {goal}

Available artifacts to explore:
{artifacts}

Gather state efficiently - only what's needed for planning."""


class AgenticStateGathering:
    """
    Agentic state gathering strategy.

    Uses LLM to decide which properties to read based on goal and intermediate results.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        max_iterations: int = 10,
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.state_trace: list[dict] = []  # Track reads for visualization

    def gather(
        self,
        affordances: CapabilityModel,
        goal: Optional[str] = None,
    ) -> EnvironmentState:
        """
        Gather state through LLM-guided property reading.

        Args:
            affordances: Discovered capability model
            goal: The goal to guide state gathering

        Returns:
            EnvironmentState with gathered property values
        """
        if not goal:
            logger.info("No goal provided, gathering all state")
            return self._gather_all(affordances)

        logger.info(f"Starting agentic state gathering for goal: {goal}")
        self.state_trace = []  # Reset trace

        # Build artifacts list for context
        artifacts_list = []
        for artifact in affordances.artifacts.values():
            artifacts_list.append({
                "uri": artifact.uri,
                "name": artifact.name,
                "workspace": artifact.workspace,
            })

        if not artifacts_list:
            logger.warning("No artifacts discovered, returning empty state")
            return EnvironmentState()

        # Track known properties (discovered via list_artifact_properties)
        self._known_properties: dict[str, list[dict]] = {}

        artifacts_json = json.dumps(artifacts_list, indent=2)
        system_prompt = STATE_DISCOVERY_SYSTEM_PROMPT.format(
            goal=goal,
            artifacts=artifacts_json,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Gather the state needed to: {goal}"},
        ]

        state = EnvironmentState(timestamp=datetime.now().isoformat())

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=STATE_DISCOVERY_TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                logger.info(f"State gathering ended after {iteration + 1} iterations (no tool call)")
                break

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Log tool call to trace
                trace_entry = {
                    "iteration": iteration + 1,
                    "function": fn_name,
                    "arguments": fn_args,
                    "result": None,
                }

                if fn_name == "done_gathering":
                    trace_entry["result"] = {"summary": fn_args.get("summary", "done")}
                    self.state_trace.append(trace_entry)
                    logger.info(f"State gathering complete: {fn_args.get('summary', 'done')}")
                    return state

                elif fn_name == "list_artifact_properties":
                    artifact_uri = fn_args["artifact_uri"]

                    try:
                        # Call actual API to list properties
                        properties = list_properties(artifact_uri)
                        # Store known properties for validation
                        self._known_properties[artifact_uri] = properties
                        result = {
                            "properties": [
                                {"name": p["name"], "uri": p["uri"]}
                                for p in properties
                            ],
                            "success": True,
                        }
                        logger.debug(f"Listed {len(properties)} properties for {artifact_uri}")
                    except Exception as e:
                        result = {"error": str(e), "success": False, "properties": []}
                        logger.warning(f"Failed to list properties for {artifact_uri}: {e}")

                    trace_entry["result"] = result
                    self.state_trace.append(trace_entry)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

                elif fn_name == "read_property":
                    prop_uri = fn_args["property_uri"]
                    reason = fn_args.get("reason", "")

                    # Validate that this property was discovered via list_artifact_properties
                    is_known = any(
                        any(p["uri"] == prop_uri for p in props)
                        for props in self._known_properties.values()
                    )
                    if not is_known:
                        # Warn but still try to read - might be a valid URI
                        logger.warning(f"Property {prop_uri} was not discovered via list_artifact_properties")

                    try:
                        value = get_property_by_uri(prop_uri)
                        state.property_values[prop_uri] = value
                        result = {"value": value, "success": True}
                        logger.debug(f"Read property {prop_uri}: {value}")
                    except GetPropertyError as e:
                        state.errors[prop_uri] = str(e)
                        result = {"error": str(e), "success": False}
                        logger.warning(f"Failed to read property {prop_uri}: {e}")
                    except Exception as e:
                        state.errors[prop_uri] = str(e)
                        result = {"error": str(e), "success": False}
                        logger.warning(f"Unexpected error reading property {prop_uri}: {e}")

                    trace_entry["result"] = result
                    self.state_trace.append(trace_entry)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

        logger.info(
            f"State gathering complete: {len(state.property_values)} values "
            f"(agentic, {self.max_iterations} max iterations)"
        )
        return state

    def get_state_trace(self) -> list[dict]:
        """Return the state gathering trace for visualization."""
        return self.state_trace

    def _gather_all(self, affordances: CapabilityModel) -> EnvironmentState:
        """Fallback to gather all properties."""
        from .all import AllStateGathering
        return AllStateGathering().gather(affordances)
