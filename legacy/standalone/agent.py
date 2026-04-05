"""
Minimal LLM Agent for exploring and interacting with HMAS environments.

The agent discovers the environment dynamically through hypermedia exploration -
no hardcoded knowledge of rooms, devices, or actions.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.hmas_client import (
    GetPropertyError,
    InvokeActionError,
    get_artifact_name,
    get_property_by_uri,
    invoke_action_by_uri,
    list_actions,
    list_artifacts,
    list_properties,
    list_workspaces,
)

load_dotenv()

# The ONLY thing the agent knows is the entry point URI
DEFAULT_ENTRY_POINT = "http://localhost:8080/workspaces/home0#workspace"

# Primitive tools for hypermedia exploration
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore_workspace",
            "description": "Explore a workspace URI to discover what it contains. Returns sub-workspaces and artifacts (things/devices) contained within. This is your primary discovery tool - start here and follow the URIs you find.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_uri": {
                        "type": "string",
                        "description": "The workspace URI to explore (must include #workspace fragment)",
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
            "description": "Inspect an artifact (device/thing) to discover its capabilities. Returns the artifact's name, available actions (things you can do), and observable properties (things you can read). Use the returned URIs to interact with the artifact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_uri": {
                        "type": "string",
                        "description": "The artifact URI to inspect (must include #artifact fragment)",
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
            "description": "Read the current value of a property. Use the property URI obtained from inspect_artifact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_uri": {
                        "type": "string",
                        "description": "The property URI to read",
                    }
                },
                "required": ["property_uri"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoke_action",
            "description": "Invoke an action on an artifact. Use the action URI from inspect_artifact. Check the input_schema to know what parameters are required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_uri": {
                        "type": "string",
                        "description": "The action URI to invoke",
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the action as specified in input_schema",
                        "default": {},
                    },
                },
                "required": ["action_uri"],
            },
        },
    },
]


def handle_tool_call(name: str, args: dict) -> dict[str, Any]:
    """Execute a tool and return the result."""
    try:
        if name == "explore_workspace":
            workspace_uri = args["workspace_uri"]

            # Get sub-workspaces
            sub_workspaces = []
            try:
                subs = list_workspaces(workspace_uri)
                for s in subs:
                    # Extract readable name from URI
                    readable_name = s.split("/")[-1].replace("#workspace", "")
                    sub_workspaces.append({"uri": s, "name": readable_name})
            except Exception:
                pass

            # Get artifacts
            artifacts = []
            try:
                arts = list_artifacts(workspace_uri)
                for a in arts:
                    try:
                        name = get_artifact_name(a)
                    except Exception:
                        name = a.split("/")[-1].replace("#artifact", "")
                    artifacts.append({"uri": a, "name": name})
            except Exception:
                pass

            return {
                "workspace_uri": workspace_uri,
                "sub_workspaces": sub_workspaces,
                "artifacts": artifacts,
                "hint": "Use explore_workspace on sub_workspaces to go deeper, or inspect_artifact on artifacts to see their capabilities",
            }

        elif name == "inspect_artifact":
            artifact_uri = args["artifact_uri"]

            # Get name
            try:
                artifact_name = get_artifact_name(artifact_uri)
            except Exception:
                artifact_name = artifact_uri.split("/")[-1].replace(
                    "#artifact", ""
                )

            # Get actions with their schemas
            actions = []
            try:
                for a in list_actions(artifact_uri):
                    actions.append(
                        {
                            "name": a["name"],
                            "uri": a["uri"],
                            "input_schema": a.get("input_schema", {}),
                        }
                    )
            except Exception as e:
                actions = [{"error": str(e)}]

            # Get properties with their schemas
            properties = []
            try:
                for p in list_properties(artifact_uri):
                    properties.append(
                        {
                            "name": p["name"],
                            "uri": p["uri"],
                            "output_schema": p.get("output_schema", {}),
                        }
                    )
            except Exception as e:
                properties = [{"error": str(e)}]

            return {
                "artifact_uri": artifact_uri,
                "name": artifact_name,
                "actions": actions,
                "properties": properties,
                "hint": "Use read_property with property URIs to check state, invoke_action with action URIs to control the artifact",
            }

        elif name == "read_property":
            value = get_property_by_uri(args["property_uri"])
            return {"property_uri": args["property_uri"], "value": value}

        elif name == "invoke_action":
            params = args.get("params", {})
            success = invoke_action_by_uri(args["action_uri"], params)
            return {
                "action_uri": args["action_uri"],
                "params": params,
                "success": success,
            }

        else:
            return {"error": f"Unknown tool: {name}"}

    except (GetPropertyError, InvokeActionError) as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


SYSTEM_PROMPT = """You are an agent that explores and interacts with a hypermedia environment.

You start with ONLY an entry point URI. You must discover everything else by exploration:

1. explore_workspace - Discover what's inside a workspace (sub-workspaces and artifacts)
2. inspect_artifact - Learn what an artifact can do (actions) and what you can observe (properties)
3. read_property - Read the current value of a property
4. invoke_action - Execute an action (check input_schema for required parameters)

WORKFLOW:
1. Start by exploring the entry point workspace
2. Explore sub-workspaces to find what you're looking for
3. Inspect artifacts to understand their capabilities
4. Read properties to check current state
5. Invoke actions to make changes

IMPORTANT:
- You don't know what exists until you explore it
- URIs you discover are your navigation links - follow them to learn more
- Action input_schema tells you what parameters are needed
- Be systematic: explore, understand, then act

Entry point: {entry_point}

Be concise. After completing a task, summarize what you discovered and did."""


def run_agent(
    user_goal: str, client: OpenAI, model: str, entry_point: str
) -> str:
    """Run the agent loop for a given user goal."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(entry_point=entry_point),
        },
        {"role": "user", "content": user_goal},
    ]

    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or "Task completed."

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Pretty print tool call
            args_str = json.dumps(fn_args, indent=2) if fn_args else ""
            print(f"  → {fn_name}")
            if args_str:
                for line in args_str.split("\n"):
                    print(f"    {line}")

            result = handle_tool_call(fn_name, fn_args)

            # Pretty print result (truncated)
            result_str = json.dumps(result, indent=2)
            print(f"  ← ", end="")
            lines = result_str.split("\n")
            if len(lines) > 8:
                print("\n".join(f"    {l}" for l in lines[:8]))
                print(f"    ... ({len(lines) - 8} more lines)")
            else:
                print(
                    "\n".join(
                        f"    {l}" if i > 0 else l for i, l in enumerate(lines)
                    )
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return "Max iterations reached. Task may be incomplete."


def interactive_mode(client: OpenAI, model: str, entry_point: str):
    """Run interactive mode."""
    print("\n" + "=" * 60)
    print("HMAS Explorer Agent")
    print("=" * 60)
    print(f"Entry point: {entry_point}")
    print(f"Model: {model}")
    print("\nThe agent discovers everything by exploration.")
    print("Try: 'What's available?' or 'Turn on a light'")
    print("\nCommands:")
    print("  /entry <uri>  - Change entry point")
    print("  /quit         - Exit")
    print("=" * 60 + "\n")

    current_entry = entry_point

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
        elif user_input.startswith("/entry "):
            current_entry = user_input[7:].strip()
            print(f"Entry point changed to: {current_entry}")
            continue

        print("\nAgent:")
        try:
            result = run_agent(user_input, client, model, current_entry)
            print(f"\n{result}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HMAS Explorer Agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (or set OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--entry", default=DEFAULT_ENTRY_POINT, help="Entry point URI"
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="Goal to accomplish (interactive if not provided)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.base_url:
        print("Error: Set OPENAI_API_KEY in .env or provide --api-key")
        print(
            "For local models: --base-url http://localhost:11434/v1 --api-key dummy"
        )
        return

    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=args.base_url,
    )

    if args.goal:
        result = run_agent(args.goal, client, args.model, args.entry)
        print(result)
    else:
        interactive_mode(client, args.model, args.entry)


if __name__ == "__main__":
    main()
