"""
Agentic affordance discovery strategy.

Uses LLM to guide exploration based on the goal.
"""

import json
from typing import Optional
import logging

from openai import OpenAI

from ..base import CapabilityModel, Artifact, Affordance
from ...config import ModelConfig, get_model_kwargs
from ...hmas_client import (
    list_workspaces,
    list_artifacts,
    list_properties,
    list_actions,
    get_artifact_name,
    get_artifact_semantic_type,
    get_workspace_semantic_type,
)

logger = logging.getLogger(__name__)


DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore_workspace",
            "description": "Explore a workspace to see what's inside. Returns sub-workspaces and artifacts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_uri": {
                        "type": "string",
                        "description": "The workspace URI to explore",
                    }
                },
                "required": ["workspace_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_artifact",
            "description": "Inspect an artifact to discover its actions and properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_uri": {
                        "type": "string",
                        "description": "The artifact URI to inspect",
                    }
                },
                "required": ["artifact_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_exploring",
            "description": "Call this when you have discovered enough capabilities to accomplish the goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're done exploring",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

DISCOVERY_SYSTEM_PROMPT = """You are exploring a smart home environment to discover capabilities needed for a goal.

You have tools to:
1. explore_workspace - See what's in a workspace (rooms, devices)
2. inspect_artifact - See what a device can do (actions, properties)
3. done_exploring - Signal you've found enough

STRATEGY:
- Start by exploring the entry point workspace to see available rooms
- Based on the goal, explore relevant rooms (e.g., "bathroom" for bathroom light)
- Inspect artifacts that seem relevant to the goal
- Stop when you have enough information to accomplish the goal
- Don't explore everything - be selective based on the goal

Entry point: {entry_point}
Goal: {goal}

VERIFICATION:
- The goal can implicitly request control of non-existent devices or affordances.
- DO NOT assume the existence of any device or affordance not verified through exploration.

Explore efficiently - only what's needed for the goal."""


class AgenticAffordanceDiscovery:
    """
    Agentic discovery strategy.

    Uses LLM to decide which workspaces/artifacts to explore based on goal.
    """

    def __init__(
        self,
        client: OpenAI,
        model_config: ModelConfig,
        max_iterations: int = 15,
    ):
        self.client = client
        self.model_config = model_config
        self.model = model_config.name
        self.max_iterations = max_iterations
        self.exploration_trace: list[dict] = []  # Track tool calls for visualization

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,
    ) -> CapabilityModel:
        """
        Build capability model through LLM-guided exploration.

        Args:
            entry_point: Root workspace URI
            goal: The goal to guide exploration (required for agentic mode)

        Returns:
            CapabilityModel with discovered affordances
        """
        if not goal:
            raise ValueError("Agentic discovery requires a goal")

        logger.info(f"Starting agentic discovery for goal: {goal}")
        capability_model = CapabilityModel(entry_point=entry_point)
        self.exploration_trace = []  # Reset trace

        system_prompt = DISCOVERY_SYSTEM_PROMPT.format(entry_point=entry_point, goal=goal)
        user_message = f"Find capabilities needed to: {goal}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for iteration in range(self.max_iterations):
            api_kwargs = get_model_kwargs(self.model, model_config=self.model_config)
            api_kwargs.update({
                "messages": messages,
                "tools": DISCOVERY_TOOLS,
                "tool_choice": "auto",
            })
            response = self.client.chat.completions.create(**api_kwargs)

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                logger.info(f"Discovery ended after {iteration + 1} iterations (no tool call)")
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

                if fn_name == "done_exploring":
                    trace_entry["result"] = {"reason": fn_args.get("reason", "done")}
                    self.exploration_trace.append(trace_entry)
                    logger.info(f"Discovery complete: {fn_args.get('reason', 'done')}")
                    return capability_model

                elif fn_name == "explore_workspace":
                    result = self._explore_workspace(
                        fn_args["workspace_uri"],
                        capability_model,
                    )
                    trace_entry["result"] = result
                    self.exploration_trace.append(trace_entry)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

                elif fn_name == "inspect_artifact":
                    result = self._inspect_artifact(
                        fn_args["artifact_uri"],
                        capability_model,
                    )
                    trace_entry["result"] = result
                    self.exploration_trace.append(trace_entry)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

        logger.info(
            f"Discovery complete: {len(capability_model.artifacts)} artifacts "
            f"(agentic, {self.max_iterations} max iterations)"
        )
        return capability_model

    def get_exploration_trace(self) -> list[dict]:
        """Return the exploration trace for visualization."""
        return self.exploration_trace

    def _explore_workspace(
        self,
        workspace_uri: str,
        model: CapabilityModel,
    ) -> dict:
        """Explore a workspace and update capability model."""
        ws_name = workspace_uri.split("/")[-1].replace("#workspace", "")
        logger.debug(f"Exploring workspace: {ws_name}")

        try:
            # Get sub-workspaces
            sub_workspaces = []
            try:
                subs = list_workspaces(workspace_uri)
                for s in subs:
                    name = s.split("/")[-1].replace("#workspace", "")
                    sub_workspaces.append({"uri": s, "name": name})
            except Exception:
                pass

            # Get artifacts
            artifacts = []
            try:
                arts = list_artifacts(workspace_uri)
                ws_type = get_workspace_semantic_type(workspace_uri)
                ws = model.get_or_create_workspace(workspace_uri, semantic_type=ws_type)
                ws.artifact_uris = arts
                for a in arts:
                    try:
                        name = get_artifact_name(a)
                    except Exception:
                        name = a.split("/")[-1].replace("#artifact", "")
                    artifacts.append({"uri": a, "name": name})
            except Exception:
                pass

            return {
                "workspace": ws_name,
                "sub_workspaces": sub_workspaces,
                "artifacts": artifacts,
            }

        except Exception as e:
            return {"error": str(e)}

    def _inspect_artifact(
        self,
        artifact_uri: str,
        model: CapabilityModel,
    ) -> dict:
        """Inspect an artifact and update capability model."""
        art_name = artifact_uri.split("/")[-1].replace("#artifact", "")
        logger.debug(f"Inspecting artifact: {art_name}")

        try:
            name = get_artifact_name(artifact_uri)
            art_type = get_artifact_semantic_type(artifact_uri)
            actions = []
            properties = []

            for a in list_actions(artifact_uri):
                actions.append(Affordance(
                    name=a["name"],
                    uri=a["uri"],
                    schema=a.get("input_schema", {}),
                    semantic_type=a.get("semantic_type"),
                ))

            for p in list_properties(artifact_uri):
                properties.append(Affordance(
                    name=p["name"],
                    uri=p["uri"],
                    schema=p.get("output_schema", {}),
                    semantic_type=p.get("semantic_type"),
                ))

            # Find workspace for this artifact
            ws_uri = "/".join(artifact_uri.split("/")[:-2]) + "#workspace"

            model.artifacts[artifact_uri] = Artifact(
                name=name,
                uri=artifact_uri,
                workspace=ws_uri,
                actions=actions,
                properties=properties,
                semantic_type=art_type,
            )

            return {
                "name": name,
                "actions": [{"name": a.name, "uri": a.uri} for a in actions],
                "properties": [{"name": p.name, "uri": p.uri} for p in properties],
            }

        except Exception as e:
            return {"error": str(e)}
