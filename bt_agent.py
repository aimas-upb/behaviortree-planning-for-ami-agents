"""
Behavior Tree Planning Agent for HMAS environments.

This agent:
1. Explores the environment to build a capability model
2. Uses LLM to generate behavior tree specifications from goals
3. Compiles BT specs to executable py_trees
4. Executes trees with monitoring

Compare with agent.py which uses direct action invocation.

Ablation flags:
    --strategy: Prompting strategy (baseline, detailed, few_shot, icl, icl_verbose)
    --discovery: Discovery mode (exhaustive, agentic, relevant)
    --trace: Enable tracing
    --trace-dir: Directory for trace files

Discovery modes:
    exhaustive: Explore all workspaces/artifacts upfront (current behavior)
    agentic: LLM-guided exploration - decides what to explore based on goal
    relevant: Exhaustive discovery, but filter to goal-relevant capabilities in prompt
"""

import json
import os
from typing import Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
import py_trees
from py_trees.common import Status

from hmas_client import (
    list_workspaces,
    list_artifacts,
    list_properties,
    list_actions,
    get_artifact_name,
)

from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator,
)

from tracing import Tracer, TraceEventType, get_tracer, set_tracer
from prompts import get_strategy, list_strategies, get_strategy_descriptions, PromptStrategy
from src.config import get_model_kwargs

load_dotenv()

DEFAULT_ENTRY_POINT = "http://localhost:8080/workspaces/home0#workspace"


# =============================================================================
# Phase 1: Capability Model
# =============================================================================

@dataclass
class Affordance:
    """Represents an action or property affordance."""
    name: str
    uri: str
    schema: dict = field(default_factory=dict)


@dataclass
class Artifact:
    """Represents a discovered artifact with its affordances."""
    name: str
    uri: str
    workspace: str
    actions: list[Affordance] = field(default_factory=list)
    properties: list[Affordance] = field(default_factory=list)


@dataclass
class CapabilityModel:
    """Complete model of discovered environment capabilities."""
    entry_point: str
    workspaces: dict[str, list[str]] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a concise summary for the LLM."""
        lines = ["# Available Devices and Capabilities\n"]

        for ws_uri, artifact_uris in self.workspaces.items():
            ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
            lines.append(f"\n## {ws_name}")

            for art_uri in artifact_uris:
                art = self.artifacts.get(art_uri)
                if not art:
                    continue

                lines.append(f"\n### {art.name}")
                lines.append(f"URI: `{art_uri}`")

                if art.actions:
                    lines.append("**Actions:**")
                    for action in art.actions:
                        schema_info = ""
                        if action.schema:
                            params = action.schema.get("properties", {})
                            if params:
                                schema_info = f" (params: {list(params.keys())})"
                        lines.append(f"  - `{action.name}`: `{action.uri}`{schema_info}")

                if art.properties:
                    lines.append("**Properties:**")
                    for prop in art.properties:
                        lines.append(f"  - `{prop.name}`: `{prop.uri}`")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dict for tracing (summary stats)."""
        return {
            "entry_point": self.entry_point,
            "workspace_count": len(self.workspaces),
            "artifact_count": len(self.artifacts),
            "action_count": sum(len(a.actions) for a in self.artifacts.values()),
            "property_count": sum(len(a.properties) for a in self.artifacts.values()),
        }

    def to_full_dict(self) -> dict:
        """Convert to full dict with all discovered capabilities for tracing."""
        workspaces_data = {}
        for ws_uri, artifact_uris in self.workspaces.items():
            ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
            artifacts_data = []
            for art_uri in artifact_uris:
                art = self.artifacts.get(art_uri)
                if art:
                    artifacts_data.append({
                        "name": art.name,
                        "uri": art.uri,
                        "actions": [
                            {"name": a.name, "uri": a.uri, "schema": a.schema}
                            for a in art.actions
                        ],
                        "properties": [
                            {"name": p.name, "uri": p.uri, "schema": p.schema}
                            for p in art.properties
                        ],
                    })
            workspaces_data[ws_name] = artifacts_data

        return {
            "entry_point": self.entry_point,
            "workspaces": workspaces_data,
            "stats": self.to_dict(),
        }


def build_capability_model(
    entry_point: str,
    max_workspaces: int = 5,
    tracer: Optional[Tracer] = None,
) -> CapabilityModel:
    """
    Explore environment and build capability model.

    Args:
        entry_point: Root workspace URI
        max_workspaces: Limit exploration depth
        tracer: Optional tracer for logging
    """
    tracer = tracer or get_tracer()
    tracer.log(TraceEventType.CAPABILITY_MODEL_START, {"entry_point": entry_point})

    print("Building capability model...")
    model = CapabilityModel(entry_point=entry_point)

    try:
        workspaces = list_workspaces(entry_point)[:max_workspaces]
    except Exception as e:
        print(f"  Failed to list workspaces: {e}")
        tracer.log(TraceEventType.ERROR, {"error": str(e), "phase": "list_workspaces"})
        return model

    for ws_uri in workspaces:
        ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
        print(f"  Exploring {ws_name}...")

        try:
            artifact_uris = list_artifacts(ws_uri)
            model.workspaces[ws_uri] = artifact_uris

            tracer.log(TraceEventType.WORKSPACE_DISCOVERED, {
                "workspace": ws_name,
                "artifact_count": len(artifact_uris),
            })

            for art_uri in artifact_uris:
                try:
                    name = get_artifact_name(art_uri)
                    actions = []
                    properties = []

                    for a in list_actions(art_uri):
                        actions.append(Affordance(
                            name=a["name"],
                            uri=a["uri"],
                            schema=a.get("input_schema", {})
                        ))

                    for p in list_properties(art_uri):
                        properties.append(Affordance(
                            name=p["name"],
                            uri=p["uri"],
                            schema=p.get("output_schema", {})
                        ))

                    model.artifacts[art_uri] = Artifact(
                        name=name,
                        uri=art_uri,
                        workspace=ws_uri,
                        actions=actions,
                        properties=properties
                    )

                    tracer.log(TraceEventType.ARTIFACT_DISCOVERED, {
                        "artifact": name,
                        "actions": [a.name for a in actions],
                        "properties": [p.name for p in properties],
                    })

                except Exception as e:
                    print(f"    Failed to inspect {art_uri}: {e}")
                    tracer.log(TraceEventType.ERROR, {"error": str(e), "artifact": art_uri})

        except Exception as e:
            print(f"  Failed to list artifacts in {ws_name}: {e}")
            tracer.log(TraceEventType.ERROR, {"error": str(e), "workspace": ws_name})

    artifact_count = len(model.artifacts)
    action_count = sum(len(a.actions) for a in model.artifacts.values())
    print(f"  Found {artifact_count} artifacts with {action_count} actions")

    tracer.log(TraceEventType.CAPABILITY_MODEL_END, model.to_dict())

    return model


# Alias for clarity
build_capability_model_exhaustive = build_capability_model


# =============================================================================
# Phase 1b: Agentic Discovery (LLM-guided exploration)
# =============================================================================

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

Explore efficiently - only what's needed for the goal."""


def build_capability_model_agentic(
    entry_point: str,
    goal: str,
    client: OpenAI,
    model: str,
    max_iterations: int = 15,
    tracer: Optional[Tracer] = None,
) -> CapabilityModel:
    """
    Build capability model through LLM-guided exploration.

    The LLM decides which workspaces/artifacts to explore based on the goal,
    rather than exhaustively exploring everything.
    """
    tracer = tracer or get_tracer()
    tracer.log(TraceEventType.CAPABILITY_MODEL_START, {
        "entry_point": entry_point,
        "mode": "agentic",
        "goal": goal,
    })

    print(f"Building capability model (agentic for: '{goal}')...")
    capability_model = CapabilityModel(entry_point=entry_point)

    system_prompt = DISCOVERY_SYSTEM_PROMPT.format(entry_point=entry_point, goal=goal)
    user_message = f"Find capabilities needed to: {goal}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Log the initial discovery prompt
    tracer.log(TraceEventType.LLM_PROMPT_CONTENT, {
        "phase": "discovery",
        "system_prompt": system_prompt,
        "user_message": user_message,
        "tools": [t["function"]["name"] for t in DISCOVERY_TOOLS],
    })

    for iteration in range(max_iterations):
        api_kwargs = get_model_kwargs(model)
        api_kwargs.update({
            "messages": messages,
            "tools": DISCOVERY_TOOLS,
            "tool_choice": "auto",
        })
        response = client.chat.completions.create(**api_kwargs)

        message = response.choices[0].message
        messages.append(message)

        # Log the LLM turn with all tool calls
        tool_calls_data = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        tracer.log(TraceEventType.DISCOVERY_LLM_TURN, {
            "iteration": iteration + 1,
            "content": message.content,
            "tool_calls": tool_calls_data,
        })

        if not message.tool_calls:
            # LLM stopped without calling done_exploring
            print(f"  Discovery ended after {iteration + 1} iterations (no tool call)")
            break

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Log tool call
            tracer.log(TraceEventType.DISCOVERY_TOOL_CALL, {
                "tool_call_id": tool_call.id,
                "function": fn_name,
                "arguments": fn_args,
            })

            if fn_name == "done_exploring":
                print(f"  Discovery complete: {fn_args.get('reason', 'done')}")
                # Log tool result for done_exploring
                tracer.log(TraceEventType.DISCOVERY_TOOL_RESULT, {
                    "tool_call_id": tool_call.id,
                    "function": fn_name,
                    "result": {"status": "discovery_complete", "reason": fn_args.get("reason")},
                })
                tracer.log(TraceEventType.CAPABILITY_MODEL_END, {
                    **capability_model.to_dict(),
                    "mode": "agentic",
                    "iterations": iteration + 1,
                    "reason": fn_args.get("reason"),
                })
                return capability_model

            elif fn_name == "explore_workspace":
                ws_uri = fn_args["workspace_uri"]
                ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
                print(f"  Exploring {ws_name}...")

                try:
                    # Get sub-workspaces
                    sub_workspaces = []
                    try:
                        subs = list_workspaces(ws_uri)
                        for s in subs:
                            name = s.split("/")[-1].replace("#workspace", "")
                            sub_workspaces.append({"uri": s, "name": name})
                    except Exception:
                        pass

                    # Get artifacts
                    artifacts = []
                    try:
                        arts = list_artifacts(ws_uri)
                        capability_model.workspaces[ws_uri] = arts
                        for a in arts:
                            try:
                                name = get_artifact_name(a)
                            except Exception:
                                name = a.split("/")[-1].replace("#artifact", "")
                            artifacts.append({"uri": a, "name": name})
                    except Exception:
                        pass

                    result = {
                        "workspace": ws_name,
                        "sub_workspaces": sub_workspaces,
                        "artifacts": artifacts,
                    }

                    tracer.log(TraceEventType.WORKSPACE_DISCOVERED, result)

                except Exception as e:
                    result = {"error": str(e)}
                    tracer.log(TraceEventType.ERROR, {"error": str(e), "workspace": ws_uri})

                # Log tool result
                tracer.log(TraceEventType.DISCOVERY_TOOL_RESULT, {
                    "tool_call_id": tool_call.id,
                    "function": fn_name,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            elif fn_name == "inspect_artifact":
                art_uri = fn_args["artifact_uri"]
                print(f"    Inspecting {art_uri.split('/')[-1].replace('#artifact', '')}...")

                try:
                    name = get_artifact_name(art_uri)
                    actions = []
                    properties = []

                    for a in list_actions(art_uri):
                        actions.append(Affordance(
                            name=a["name"],
                            uri=a["uri"],
                            schema=a.get("input_schema", {})
                        ))

                    for p in list_properties(art_uri):
                        properties.append(Affordance(
                            name=p["name"],
                            uri=p["uri"],
                            schema=p.get("output_schema", {})
                        ))

                    # Find workspace for this artifact
                    ws_uri = "/".join(art_uri.split("/")[:-2]) + "#workspace"

                    capability_model.artifacts[art_uri] = Artifact(
                        name=name,
                        uri=art_uri,
                        workspace=ws_uri,
                        actions=actions,
                        properties=properties
                    )

                    result = {
                        "name": name,
                        "actions": [{"name": a.name, "uri": a.uri} for a in actions],
                        "properties": [{"name": p.name, "uri": p.uri} for p in properties],
                    }

                    tracer.log(TraceEventType.ARTIFACT_DISCOVERED, {
                        "artifact": name,
                        "actions": [a.name for a in actions],
                        "properties": [p.name for p in properties],
                    })

                except Exception as e:
                    result = {"error": str(e)}
                    tracer.log(TraceEventType.ERROR, {"error": str(e), "artifact": art_uri})

                # Log tool result
                tracer.log(TraceEventType.DISCOVERY_TOOL_RESULT, {
                    "tool_call_id": tool_call.id,
                    "function": fn_name,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

    artifact_count = len(capability_model.artifacts)
    action_count = sum(len(a.actions) for a in capability_model.artifacts.values())
    print(f"  Found {artifact_count} artifacts with {action_count} actions (agentic)")

    tracer.log(TraceEventType.CAPABILITY_MODEL_END, {
        **capability_model.to_dict(),
        "mode": "agentic",
        "iterations": max_iterations,
    })

    return capability_model


# =============================================================================
# Phase 1c: Relevant Filtering (post-discovery filter)
# =============================================================================

def filter_capabilities_for_goal(
    model: CapabilityModel,
    goal: str,
    client: OpenAI,
    llm_model: str,
    tracer: Optional[Tracer] = None,
) -> CapabilityModel:
    """
    Filter a capability model to only include goal-relevant capabilities.

    Uses LLM to determine which artifacts are relevant to the goal.
    """
    tracer = tracer or get_tracer()

    # Build list of all artifacts
    artifact_list = []
    for art_uri, art in model.artifacts.items():
        ws_name = art.workspace.split("/")[-1].replace("#workspace", "")
        artifact_list.append({
            "uri": art_uri,
            "name": art.name,
            "workspace": ws_name,
            "actions": [a.name for a in art.actions],
        })

    if not artifact_list:
        return model

    # Ask LLM which are relevant
    filter_prompt = f"""Given this goal: "{goal}"

Which of these devices are relevant? Return a JSON array of URIs.

Devices:
{json.dumps(artifact_list, indent=2)}

Return ONLY a JSON array of relevant URIs, nothing else."""

    api_kwargs = get_model_kwargs(llm_model)
    api_kwargs["messages"] = [{"role": "user", "content": filter_prompt}]

    response = client.chat.completions.create(**api_kwargs)

    try:
        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        relevant_uris = set(json.loads(content))
    except Exception as e:
        print(f"  Filter failed ({e}), using all capabilities")
        tracer.log(TraceEventType.ERROR, {"error": f"Filter failed: {e}"})
        return model

    # Build filtered model
    filtered = CapabilityModel(entry_point=model.entry_point)

    for ws_uri, art_uris in model.workspaces.items():
        filtered_arts = [u for u in art_uris if u in relevant_uris]
        if filtered_arts:
            filtered.workspaces[ws_uri] = filtered_arts

    for art_uri, art in model.artifacts.items():
        if art_uri in relevant_uris:
            filtered.artifacts[art_uri] = art

    original_count = len(model.artifacts)
    filtered_count = len(filtered.artifacts)
    print(f"  Filtered: {original_count} → {filtered_count} artifacts")

    tracer.log(TraceEventType.CAPABILITY_MODEL_SUMMARY, {
        "mode": "relevant_filter",
        "original_count": original_count,
        "filtered_count": filtered_count,
        "relevant_uris": list(relevant_uris),
    })

    return filtered


# =============================================================================
# Phase 2: BT Compiler (JSON -> py_trees)
# =============================================================================

OPERATOR_MAP = {
    "==": ComparisonOperator.EQUAL,
    "!=": ComparisonOperator.NOT_EQUAL,
    ">": ComparisonOperator.GREATER_THAN,
    ">=": ComparisonOperator.GREATER_THAN_OR_EQUAL,
    "<": ComparisonOperator.LESS_THAN,
    "<=": ComparisonOperator.LESS_THAN_OR_EQUAL,
    "in": ComparisonOperator.IN,
    "not_in": ComparisonOperator.NOT_IN,
    "contains": ComparisonOperator.CONTAINS,
}


def compile_bt(spec: dict) -> py_trees.behaviour.Behaviour:
    """
    Compile a JSON behavior tree specification to py_trees.

    Args:
        spec: JSON specification with type, name, children, etc.

    Returns:
        Executable py_trees behavior
    """
    node_type = spec.get("type")
    name = spec.get("name", "unnamed")

    if node_type == "sequence":
        children = [compile_bt(child) for child in spec.get("children", [])]
        return py_trees.composites.Sequence(name=name, memory=True, children=children)

    elif node_type == "selector":
        children = [compile_bt(child) for child in spec.get("children", [])]
        return py_trees.composites.Selector(name=name, memory=False, children=children)

    elif node_type == "parallel":
        children = [compile_bt(child) for child in spec.get("children", [])]
        policy_name = spec.get("policy", "success_on_all")
        if policy_name == "success_on_one":
            policy = py_trees.common.ParallelPolicy.SuccessOnOne()
        else:
            policy = py_trees.common.ParallelPolicy.SuccessOnAll()
        return py_trees.composites.Parallel(name=name, policy=policy, children=children)

    elif node_type == "action":
        return ActionAffordanceNode(
            name=name,
            action_url=spec["action_url"],
            parameters=spec.get("parameters", {}),
        )

    elif node_type == "condition":
        operator = spec.get("operator")
        if operator and operator != "==":
            return ComparisonPropertyConditionNode(
                name=name,
                property_url=spec["property_url"],
                expected_value=spec["expected_value"],
                operator=OPERATOR_MAP.get(operator, ComparisonOperator.EQUAL),
                value_path=spec.get("value_path"),
            )
        else:
            return PropertyConditionNode(
                name=name,
                property_url=spec["property_url"],
                expected_value=spec["expected_value"],
                value_path=spec.get("value_path"),
            )

    else:
        raise ValueError(f"Unknown node type: {node_type}")


def execute_bt(
    tree: py_trees.behaviour.Behaviour,
    max_ticks: int = 10,
    tracer: Optional[Tracer] = None,
) -> dict:
    """
    Execute a behavior tree and return results.

    Args:
        tree: Compiled py_trees behavior
        max_ticks: Maximum tick iterations
        tracer: Optional tracer for logging
    """
    tracer = tracer or get_tracer()
    tree.setup_with_descendants()

    results = {
        "tree_name": tree.name,
        "ticks": 0,
        "final_status": None,
        "success": False,
        "tick_history": [],
    }

    for tick in range(max_ticks):
        results["ticks"] = tick + 1
        tree.tick_once()

        status_name = tree.status.name
        print(f"  Tick {tick + 1}: {status_name}")

        results["tick_history"].append(status_name)
        tracer.log(TraceEventType.BT_TICK, {
            "tick": tick + 1,
            "status": status_name,
        })

        if tree.status == Status.SUCCESS:
            results["final_status"] = "SUCCESS"
            results["success"] = True
            break
        elif tree.status == Status.FAILURE:
            results["final_status"] = "FAILURE"
            results["success"] = False
            break
    else:
        results["final_status"] = "RUNNING (max ticks reached)"

    tracer.log(TraceEventType.BT_EXECUTION_END, results)
    tree.shutdown()
    return results


# =============================================================================
# Phase 3: LLM Agent (with modular prompting)
# =============================================================================

def build_tools(strategy: PromptStrategy) -> list[dict]:
    """Build tool definitions using the strategy's tool description."""
    return [
        {
            "type": "function",
            "function": {
                "name": "generate_behavior_tree",
                "description": strategy.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tree": {
                            "type": "object",
                            "description": "Behavior tree spec",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of the plan"
                        }
                    },
                    "required": ["tree", "explanation"],
                },
            },
        },
    ]


def run_bt_agent(
    goal: str,
    client: OpenAI,
    model: str,
    capability_model: CapabilityModel,
    strategy: Optional[PromptStrategy] = None,
    tracer: Optional[Tracer] = None,
) -> str:
    """
    Run the BT planning agent for a goal.

    Args:
        goal: User's goal
        client: OpenAI client
        model: Model name
        capability_model: Discovered capabilities
        strategy: Prompting strategy (default: detailed)
        tracer: Optional tracer
    """
    tracer = tracer or get_tracer()
    strategy = strategy or get_strategy("detailed")

    # Build prompt
    system_prompt = strategy.system_prompt.format(
        capability_model=capability_model.to_summary()
    )
    tools = build_tools(strategy)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]

    # Log capability model (full) for analysis
    tracer.log(TraceEventType.CAPABILITY_MODEL_SUMMARY, {
        "discovered_capabilities": capability_model.to_full_dict(),
        "prompt_representation": capability_model.to_summary(),
    })

    # Log LLM request with full prompt for reproducibility
    tracer.log(TraceEventType.LLM_REQUEST, {
        "goal": goal,
        "model": model,
        "strategy": strategy.name,
        "system_prompt_length": len(system_prompt),
    })

    # Log full prompt content for analysis
    tracer.log(TraceEventType.LLM_PROMPT_CONTENT, {
        "system_prompt": system_prompt,
        "user_message": goal,
        "tool_description": strategy.tool_description,
    })

    api_kwargs = get_model_kwargs(model)
    api_kwargs.update({
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "generate_behavior_tree"}},
    })
    response = client.chat.completions.create(**api_kwargs)

    message = response.choices[0].message

    # Log LLM response
    tracer.log(TraceEventType.LLM_RESPONSE, {
        "has_tool_calls": bool(message.tool_calls),
        "finish_reason": response.choices[0].finish_reason,
    })

    if not message.tool_calls:
        return f"No behavior tree generated. Response: {message.content}"

    tool_call = message.tool_calls[0]
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        tracer.log(TraceEventType.ERROR, {"error": f"JSON decode: {e}"})
        return f"Invalid JSON from LLM: {e}\nRaw: {tool_call.function.arguments[:500]}"

    if "tree" not in args:
        tracer.log(TraceEventType.ERROR, {"error": "Missing tree field"})
        return f"Missing 'tree' in response. Got: {list(args.keys())}"

    tree_spec = args["tree"]
    explanation = args.get("explanation", "")

    tracer.log(TraceEventType.BT_SPEC_GENERATED, {
        "tree_spec": tree_spec,
        "explanation": explanation,
    })

    print(f"\n--- Plan ---")
    print(f"{explanation}")
    print(f"\n--- Tree Spec ---")
    print(json.dumps(tree_spec, indent=2))

    # Compile
    print(f"\n--- Compiling ---")
    try:
        tree = compile_bt(tree_spec)
        print(f"Compiled: {tree.name}")
        print(py_trees.display.unicode_tree(tree, show_status=True))
        tracer.log(TraceEventType.BT_COMPILED, {"tree_name": tree.name})
    except Exception as e:
        tracer.log(TraceEventType.ERROR, {"error": f"Compilation: {e}"})
        return f"Compilation failed: {e}"

    # Execute
    print(f"\n--- Executing ---")
    try:
        result = execute_bt(tree, tracer=tracer)
    except Exception as e:
        tracer.log(TraceEventType.ERROR, {"error": f"Execution: {e}"})
        return f"Execution failed: {e}"

    status = "SUCCESS" if result["success"] else "FAILED"
    return f"Tree '{result['tree_name']}' {status} after {result['ticks']} tick(s)"


def interactive_mode(
    client: OpenAI,
    model: str,
    capability_model: CapabilityModel,
    strategy: PromptStrategy,
    discovery_mode: str,
    tracer: Tracer,
):
    """Run interactive mode."""
    print("\n" + "=" * 60)
    print("Behavior Tree Planning Agent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Strategy: {strategy.name} - {strategy.description}")
    print(f"Discovery: {discovery_mode}")
    print(f"Tracing: {'enabled' if tracer.enabled else 'disabled'}")
    print(f"Artifacts: {len(capability_model.artifacts)}")
    print(f"Actions: {sum(len(a.actions) for a in capability_model.artifacts.values())}")
    print("\nCommands:")
    print("  /model      - Show capability model")
    print("  /strategies - List available strategies")
    print("  /quit       - Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break
        elif user_input == "/model":
            print(capability_model.to_summary())
            continue
        elif user_input == "/strategies":
            print("\nAvailable strategies:")
            for name, desc in get_strategy_descriptions().items():
                marker = " *" if name == strategy.name else ""
                print(f"  {name}: {desc}{marker}")
            print()
            continue

        print("\nAgent:")
        tracer.start_trace(
            goal=user_input,
            model=model,
            entry_point=capability_model.entry_point,
            ablation_config={"strategy": strategy.name, "discovery": discovery_mode},
        )

        try:
            result = run_bt_agent(
                user_input, client, model, capability_model,
                strategy=strategy, tracer=tracer
            )
            tracer.end_trace(success="SUCCESS" in result, result=result)
            print(f"\n{result}\n")
        except Exception as e:
            tracer.end_trace(success=False, result=str(e))
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BT Planning Agent with ablation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation strategies (--strategy):
  baseline    - Minimal prompting (tests raw LLM capability)
  detailed    - Comprehensive node type documentation (default)
  few_shot    - Example-based learning with diverse patterns
  icl         - In-context learning with reasoning traces
  icl_verbose - Detailed reasoning with methodology

Discovery modes (--discovery):
  exhaustive  - Explore all workspaces/artifacts upfront (default)
  agentic     - LLM-guided exploration, discovers only goal-relevant artifacts
  relevant    - Exhaustive discovery, then filter to goal-relevant in prompt

Examples:
  uv run python bt_agent.py "Turn on bathroom light"
  uv run python bt_agent.py --home 5 "Turn on bathroom light"   # Use home 5
  uv run python bt_agent.py --strategy few_shot "Turn on lights"
  uv run python bt_agent.py --discovery agentic "Turn on bathroom light"
  uv run python bt_agent.py --discovery relevant "Turn on the AC"
  uv run python bt_agent.py --trace --trace-dir ./traces "Turn on AC"
        """
    )

    # Model settings
    parser.add_argument("--model", default="gpt-4o",
                        help="Model to use (gpt-4o recommended)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None,
                        help="API key")

    # Environment settings
    parser.add_argument("--home", type=int, default=0,
                        help="Home ID to use (0-99, default: 0). Sets entry point to home{N}")
    parser.add_argument("--entry", default=None,
                        help="Entry point URI (overrides --home if specified)")
    parser.add_argument("--max-workspaces", type=int, default=5,
                        help="Max workspaces to explore")

    # Ablation settings
    parser.add_argument("--strategy", default="detailed",
                        choices=list_strategies(),
                        help="Prompting strategy for ablation")
    parser.add_argument("--discovery", default="exhaustive",
                        choices=["exhaustive", "agentic", "relevant"],
                        help="Discovery mode: exhaustive (all), agentic (LLM-guided), relevant (filtered)")
    parser.add_argument("--trace", action="store_true",
                        help="Enable tracing")
    parser.add_argument("--trace-dir", default="traces",
                        help="Directory for trace files")
    parser.add_argument("--trace-verbose", action="store_true",
                        help="Print trace events to console")

    parser.add_argument("goal", nargs="?",
                        help="Goal (interactive if not provided)")

    args = parser.parse_args()

    # Determine entry point (--entry overrides --home)
    if args.entry:
        entry_point = args.entry
    else:
        entry_point = f"http://localhost:8080/workspaces/home{args.home}#workspace"

    # Setup API client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.base_url:
        print("Error: Set OPENAI_API_KEY in .env or provide --api-key")
        return

    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=args.base_url,
    )

    # Setup tracer
    tracer = Tracer(enabled=args.trace, verbose=args.trace_verbose)
    set_tracer(tracer)

    # Get prompting strategy
    strategy = get_strategy(args.strategy)
    print(f"Using strategy: {strategy.name}")
    print(f"Discovery mode: {args.discovery}")
    print(f"Entry point: {entry_point}")

    # Start trace BEFORE discovery (so we capture discovery events)
    if args.goal:
        tracer.start_trace(
            goal=args.goal,
            model=args.model,
            entry_point=entry_point,
            ablation_config={"strategy": strategy.name, "discovery": args.discovery, "home": args.home},
        )

    # Build capability model based on discovery mode
    if args.discovery == "agentic":
        # Agentic discovery requires a goal upfront
        if not args.goal:
            print("Error: Agentic discovery requires a goal (--discovery agentic needs a goal argument)")
            print("Use --discovery exhaustive for interactive mode")
            return
        capability_model = build_capability_model_agentic(
            entry_point=entry_point,
            goal=args.goal,
            client=client,
            model=args.model,
            tracer=tracer,
        )
    else:
        # Exhaustive or relevant: start with full discovery
        capability_model = build_capability_model(
            entry_point, args.max_workspaces, tracer=tracer
        )

    if not capability_model.artifacts:
        print("No artifacts found. Is the simulator running?")
        return

    # Apply relevant filtering if requested
    if args.discovery == "relevant" and args.goal:
        capability_model = filter_capabilities_for_goal(
            model=capability_model,
            goal=args.goal,
            client=client,
            llm_model=args.model,
            tracer=tracer,
        )

    if args.goal:
        # Single goal mode (trace already started before discovery)
        result = run_bt_agent(
            args.goal, client, args.model, capability_model,
            strategy=strategy, tracer=tracer
        )
        print(result)

        tracer.end_trace(success="SUCCESS" in result, result=result)

        if args.trace:
            trace_path = tracer.save(args.trace_dir)
            print(f"\nTrace saved to: {trace_path}")
    else:
        # Interactive mode
        interactive_mode(client, args.model, capability_model, strategy, args.discovery, tracer)

        if args.trace:
            trace_path = tracer.save(args.trace_dir)
            if trace_path:
                print(f"\nTrace saved to: {trace_path}")


if __name__ == "__main__":
    main()
