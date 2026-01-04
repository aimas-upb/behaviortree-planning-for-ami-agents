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
from hmas_client import get_property_by_uri, GetPropertyError

logger = logging.getLogger(__name__)


STATE_DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_property",
            "description": "Read the current value of a property from the environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_uri": {
                        "type": "string",
                        "description": "The property URI to read",
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
1. read_property - Read the current value of a device property
2. done_gathering - Signal you've gathered enough state information

STRATEGY:
- Based on the goal, identify which properties need to be checked
- Read properties that are directly relevant to the goal
- Read properties that could affect how actions should be executed
- Stop when you have enough information to plan effectively
- Don't read everything - be selective based on the goal

Goal: {goal}

Available properties:
{properties}

Read efficiently - only what's needed for planning."""


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

        # Build property list for context
        properties_list = []
        for artifact in affordances.artifacts.values():
            for prop in artifact.properties:
                prop_info = {
                    "uri": prop.uri,
                    "name": prop.name,
                    "artifact": artifact.name,
                }
                if prop.schema:
                    prop_info["type"] = prop.schema.get("type", "unknown")
                    if "enum" in prop.schema:
                        prop_info["possible_values"] = prop.schema["enum"]
                properties_list.append(prop_info)

        if not properties_list:
            return EnvironmentState()

        properties_json = json.dumps(properties_list, indent=2)
        system_prompt = STATE_DISCOVERY_SYSTEM_PROMPT.format(
            goal=goal,
            properties=properties_json,
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

                elif fn_name == "read_property":
                    prop_uri = fn_args["property_uri"]
                    reason = fn_args.get("reason", "")

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
