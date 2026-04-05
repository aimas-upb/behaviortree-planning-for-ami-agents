"""
Direct agent execution strategy.

Instead of generating code/behavior trees, the LLM directly makes tool calls
to interact with the environment to achieve the goal.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx
from openai import OpenAI

from ..config import ModelConfig, get_model_kwargs
from ..discovery.base import CapabilityModel, EnvironmentState
from .base import ExecutionResult

logger = logging.getLogger(__name__)


# Tools for direct environment interaction
DIRECT_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_property",
            "description": "Read the current value of a device property.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_url": {
                        "type": "string",
                        "description": "The URL of the property to read",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you are reading this property",
                    },
                },
                "required": ["property_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": "Execute an action on a device. This makes a POST request to the action URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_url": {
                        "type": "string",
                        "description": "The URL of the action to execute",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters to pass to the action (as JSON object)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you are executing this action",
                    },
                },
                "required": ["action_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_impossible",
            "description": "Report that a sub-goal cannot be achieved with the available devices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal": {
                        "type": "string",
                        "description": "Description of the sub-goal that is impossible",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this sub-goal cannot be achieved",
                    },
                },
                "required": ["subgoal", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that you have completed all achievable sub-goals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


DIRECT_AGENT_SYSTEM_PROMPT = """You are a smart home control agent. Your task is to achieve the user's goal by interacting with devices in the environment.

## Tools Available
1. read_property - Read the current value of a device property
2. execute_action - Execute an action on a device (POST request with parameters)
3. report_impossible - Report if a sub-goal cannot be achieved (device missing, capability unavailable)
4. done - Signal completion when all achievable sub-goals are done

## CRITICAL: Parameter Handling

Actions require SPECIFIC parameters. You MUST:
1. Look at the action's parameter schema to know what parameters are needed
2. For RELATIVE changes (increase/decrease by X), you MUST:
   a. First read_property to get the current value
   b. Calculate the new value (current + increment or current - decrement)
   c. Execute the action with the COMPUTED value as the parameter

### Example: "Increase brightness by 20"
WRONG: execute_action(url, {{}})  // Missing parameter!
WRONG: execute_action(url, {{"increase": 20}})  // Wrong parameter name!
CORRECT WORKFLOW:
1. read_property(brightness_url) -> returns 50
2. Calculate: 50 + 20 = 70
3. execute_action(set_brightness_url, {{"brightness": 70}})

### Example: "Set temperature to 25"
CORRECT: execute_action(set_temperature_url, {{"temperature": 25}})

### Example: "Turn on the light"
CORRECT: execute_action(turn_on_url, {{}})  // No parameters needed for toggle actions

## Action Parameter Rules
- Parameter names must EXACTLY match the schema (e.g., "brightness" not "level")
- For set_X actions, the parameter is usually named X (set_brightness -> brightness, set_speed -> speed)
- For turn_on/turn_off, no parameters are needed
- Numeric values must respect min/max constraints from the schema
- Enum values must be one of the allowed values from the schema

## Strategy
1. Parse the goal into sub-goals
2. For each sub-goal:
   - Find the relevant device and action
   - If it's a relative change (increase/decrease BY X), first read the current value
   - Compute the correct parameter value
   - Execute with proper parameters
3. Only report_impossible when a device or capability truly doesn't exist
4. If an action fails with HTTP 400, check if you passed the wrong parameters
5. Call done when finished

## Available Devices and Their Capabilities
{capability_model}

## Current Device State
{current_state}

## Verification
- The goal of the user can implicitly request control of non-existent devices or affordances.
- DO NOT assume the existence of any device or affordance;
- Ensure that each device and capability is verified through the provided capability model and current state.
"""


@dataclass
class DirectAgentResult:
    """Result of direct agent execution."""

    success: bool
    actions_executed: list[dict] = field(default_factory=list)
    properties_read: list[dict] = field(default_factory=list)
    impossible_reported: list[str] = field(default_factory=list)
    summary: str = ""
    error: Optional[str] = None
    iterations: int = 0
    trace: list[dict] = field(default_factory=list)

    def to_execution_result(self) -> ExecutionResult:
        """Convert to standard ExecutionResult."""
        return ExecutionResult(
            success=self.success,
            tree_name="DirectAgent",
            ticks=self.iterations,
            final_status="SUCCESS" if self.success else "FAILURE",
            tick_history=["SUCCESS" if self.success else "FAILURE"],
            error=self.error,
        )


class DirectAgentExecutor:
    """
    Direct agent executor.

    Uses LLM with tool calls to directly interact with the environment
    instead of generating code or behavior trees.
    """

    def __init__(
        self,
        client: OpenAI,
        model_config: ModelConfig,
        max_iterations: int = 20,
        http_timeout: float = 30.0,
    ):
        self.client = client
        self.model_config = model_config
        self.model = model_config.name
        self.max_iterations = max_iterations
        self.http_client = httpx.Client(timeout=http_timeout)

    def execute(
        self,
        goal: str,
        affordances: CapabilityModel,
        state: EnvironmentState,
    ) -> DirectAgentResult:
        """
        Execute goal using direct LLM agent with tool calls.

        Args:
            goal: The user's goal
            affordances: Discovered capability model
            state: Current environment state

        Returns:
            DirectAgentResult with execution details
        """
        logger.info(f"Starting direct agent execution for goal: {goal}")

        result = DirectAgentResult(success=False)

        # Build capability context
        capability_context = self._format_capabilities(affordances)
        state_context = self._format_state(state)

        system_prompt = DIRECT_AGENT_SYSTEM_PROMPT.format(
            capability_model=capability_context,
            current_state=state_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Achieve this goal: {goal}"},
        ]

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1

            api_kwargs = get_model_kwargs(
                self.model, model_config=self.model_config
            )
            api_kwargs.update(
                {
                    "messages": messages,
                    "tools": DIRECT_AGENT_TOOLS,
                    "tool_choice": "auto",
                }
            )
            response = self.client.chat.completions.create(**api_kwargs)

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                # No more tool calls - check if done was called
                logger.info(
                    f"Agent finished after {iteration + 1} iterations (no tool call)"
                )
                if not result.summary:
                    result.summary = (
                        message.content
                        or "Completed without explicit done signal"
                    )
                    result.success = (
                        len(result.actions_executed) > 0
                        or len(result.impossible_reported) > 0
                    )
                break

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                trace_entry = {
                    "iteration": iteration + 1,
                    "function": fn_name,
                    "arguments": fn_args,
                    "result": None,
                }

                if fn_name == "done":
                    result.summary = fn_args.get("summary", "Completed")
                    result.success = True
                    trace_entry["result"] = {
                        "status": "completed",
                        "summary": result.summary,
                    }
                    result.trace.append(trace_entry)
                    logger.info(f"Agent completed: {result.summary}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"status": "completed"}),
                        }
                    )
                    return result

                elif fn_name == "read_property":
                    prop_url = fn_args.get("property_url", "")
                    try:
                        resp = self.http_client.get(prop_url)
                        if resp.status_code == 200:
                            value = resp.json()
                            tool_result = {"success": True, "value": value}
                            result.properties_read.append(
                                {
                                    "url": prop_url,
                                    "value": value,
                                }
                            )
                        else:
                            tool_result = {
                                "success": False,
                                "error": f"HTTP {resp.status_code}",
                            }
                    except Exception as e:
                        tool_result = {"success": False, "error": str(e)}

                    trace_entry["result"] = tool_result
                    result.trace.append(trace_entry)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

                elif fn_name == "execute_action":
                    action_url = fn_args.get("action_url", "")
                    params = fn_args.get("parameters", {})
                    reason = fn_args.get("reason", "")

                    try:
                        if params:
                            resp = self.http_client.post(
                                action_url, json=params
                            )
                        else:
                            resp = self.http_client.post(action_url)

                        if resp.status_code == 200:
                            tool_result = {"success": True}
                            result.actions_executed.append(
                                {
                                    "url": action_url,
                                    "params": params,
                                    "reason": reason,
                                }
                            )
                            logger.info(f"Executed action: {action_url}")
                        elif resp.status_code == 400:
                            # Bad request - likely wrong parameters
                            error_body = ""
                            try:
                                error_body = resp.text
                            except Exception:
                                pass
                            tool_result = {
                                "success": False,
                                "error": "HTTP 400 Bad Request - Check your parameters!",
                                "hint": 'The action requires specific parameters. Check the schema and ensure you\'re passing the correct parameter names and values. For set_X actions, the parameter is usually named X (e.g., set_brightness needs {"brightness": value}, set_interval needs {"interval": value}).',
                                "params_sent": params,
                                "response": (
                                    error_body[:200] if error_body else None
                                ),
                            }
                            logger.warning(
                                f"Action {action_url} failed with 400. Params sent: {params}"
                            )
                        else:
                            tool_result = {
                                "success": False,
                                "error": f"HTTP {resp.status_code}",
                            }
                    except Exception as e:
                        tool_result = {"success": False, "error": str(e)}

                    trace_entry["result"] = tool_result
                    result.trace.append(trace_entry)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

                elif fn_name == "report_impossible":
                    subgoal = fn_args.get("subgoal", "")
                    reason = fn_args.get("reason", "")
                    result.impossible_reported.append(f"{subgoal}: {reason}")
                    tool_result = {"acknowledged": True}
                    trace_entry["result"] = tool_result
                    result.trace.append(trace_entry)
                    logger.info(f"Reported impossible: {subgoal}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

        if not result.success and result.iterations >= self.max_iterations:
            result.error = f"Max iterations ({self.max_iterations}) reached"
            logger.warning(result.error)

        return result

    def _format_capabilities(self, affordances: CapabilityModel) -> str:
        """Format capabilities for the prompt with detailed parameter schemas."""
        lines = []
        for workspace_uri, workspace in affordances.workspaces.items():
            ws_name = workspace_uri.split("/")[-1].replace("#workspace", "")
            lines.append(f"\n### {ws_name}")
            for artifact_uri in workspace.artifact_uris:
                artifact = affordances.artifacts.get(artifact_uri)
                if artifact:
                    lines.append(f"\n**{artifact.name}**")
                    if artifact.actions:
                        lines.append("Actions:")
                        for action in artifact.actions:
                            lines.append(f"  - {action.name}: `{action.uri}`")
                            if action.schema:
                                params = action.schema.get("properties", {})
                                required = action.schema.get("required", [])
                                if params:
                                    for (
                                        param_name,
                                        param_spec,
                                    ) in params.items():
                                        param_type = param_spec.get(
                                            "type", "any"
                                        )
                                        req_marker = (
                                            " (REQUIRED)"
                                            if param_name in required
                                            else ""
                                        )
                                        constraints = []
                                        if "minimum" in param_spec:
                                            constraints.append(
                                                f"min={param_spec['minimum']}"
                                            )
                                        if "maximum" in param_spec:
                                            constraints.append(
                                                f"max={param_spec['maximum']}"
                                            )
                                        if "enum" in param_spec:
                                            constraints.append(
                                                f"values={param_spec['enum']}"
                                            )
                                        constraint_str = (
                                            f" [{', '.join(constraints)}]"
                                            if constraints
                                            else ""
                                        )
                                        lines.append(
                                            f"      Parameter: {param_name} ({param_type}){constraint_str}{req_marker}"
                                        )
                                else:
                                    lines.append("      No parameters required")
                            else:
                                lines.append("      No parameters required")
                    if artifact.properties:
                        lines.append("Properties (for reading current values):")
                        for prop in artifact.properties:
                            type_info = ""
                            if prop.schema:
                                prop_type = prop.schema.get("type", "")
                                if prop_type:
                                    type_info = f" (returns: {prop_type})"
                            lines.append(
                                f"  - {prop.name}: `{prop.uri}`{type_info}"
                            )
        return "\n".join(lines)

    def _format_state(self, state: EnvironmentState) -> str:
        """Format current state for the prompt."""
        if not state.property_values:
            return "No state information available."

        lines = []
        for prop_url, value in state.property_values.items():
            # Extract property name from URL
            prop_name = prop_url.split("/")[-1]
            lines.append(f"- {prop_name}: {value} ({prop_url})")
        return "\n".join(lines)

    def close(self):
        """Close HTTP client."""
        self.http_client.close()
