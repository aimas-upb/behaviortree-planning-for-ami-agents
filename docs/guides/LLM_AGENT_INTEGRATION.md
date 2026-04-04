# LLM Agent Integration Guide

> Repository note: after the repository cleanup, the reusable environment client lives at `src/hmas_client.py`. Use the root `README.md` for the current repo layout and entrypoints.

This document outlines how to build an LLM agent that interacts with the simulated environments in this repository.

## Repository Overview

This framework enables LLM agents to interact with simulated environments using:
- **Behavior Trees** (py-trees) for hierarchical task planning and execution
- **W3C Thing Description (TD)** for semantic device descriptions
- **HMAS (Hypermedia Multi-Agent Systems)** for workspace/artifact discovery
- **HTTP REST APIs** for action invocation and property reading

## Available Environments

### 1. HomeBench (Smart Home)
- **URL**: `http://localhost:8080`
- **Devices**: Lights, ACs, Heaters, Fans, Garage Doors, Blinds
- **Actions**: turn_on, turn_off, set_brightness, set_temperature, set_color, etc.
- **Structure**: Workspaces (rooms) → Artifacts (devices) → Affordances (actions/properties)

### 2. Blocksworld (Planning Domain)
- **URL**: `http://localhost:8081`
- **Actions**: pickup, putdown, stack, unstack
- **State**: Block positions, hand status (empty/holding)

---

## Key Components for LLM Integration

### 1. Environment Client (`src/hmas_client.py`)

The standalone client provides discovery and interaction functions:

```python
from src.hmas_client import (
    list_workspaces,      # Discover rooms/areas
    list_artifacts,       # Discover devices in a workspace
    list_properties,      # Get readable properties of a device
    list_actions,         # Get available actions for a device
    get_property,         # Read a property value (HTTP GET)
    invoke_action         # Execute an action (HTTP POST)
)
```

**Discovery Flow**:
```python
# 1. List all rooms
workspaces = list_workspaces("http://localhost:8080/workspaces/home0#workspace")
# → ['http://localhost:8080/workspaces/home0/master_bedroom#workspace', ...]

# 2. List devices in a room
artifacts = list_artifacts(workspaces[0])
# → ['http://localhost:8080/workspaces/home0/master_bedroom/artifacts/light#artifact', ...]

# 3. Discover device capabilities
actions = list_actions(artifacts[0])
# → [{"name": "turnOn", "uri": "...", "input_schema": {...}}, ...]

properties = list_properties(artifacts[0])
# → [{"name": "brightness", "uri": "...", "output_schema": {...}}, ...]
```

**Interaction**:
```python
# Read current state
brightness = get_property(artifact_uri, "brightness")  # → 75

# Execute action
success = invoke_action(artifact_uri, "set_brightness", {"brightness": 50})
```

### 2. Behavior Tree Nodes (`behavior_trees/`)

Pre-built nodes for environment interaction:

| Node | Purpose | Returns |
|------|---------|---------|
| `ActionAffordanceNode` | Execute actions (HTTP POST) | SUCCESS/FAILURE |
| `PropertyAffordanceNode` | Read properties (HTTP GET) | SUCCESS/FAILURE |
| `PropertyConditionNode` | Check property == expected value | SUCCESS/FAILURE |
| `ComparisonPropertyConditionNode` | Compare with operators (>, <, in, etc.) | SUCCESS/FAILURE |

**Example: Turn on light if off**
```python
from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode
)
import py_trees

light_base = "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight"

# Condition: is light already on?
is_on = PropertyConditionNode(
    name="IsLightOn",
    property_url=f"{light_base}/properties/state",
    expected_value="on"
)

# Action: turn it on
turn_on = ActionAffordanceNode(
    name="TurnOn",
    action_url=f"{light_base}/turn_on"
)

# Selector: succeed if already on, otherwise turn on
tree = py_trees.composites.Selector(
    name="EnsureLightOn",
    children=[is_on, turn_on]
)
```

---

## LLM Agent Architecture Options

### Option A: Direct API Interaction

The LLM directly calls `hmas_client` functions based on natural language goals.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  LLM Agent  │───▶│ hmas_client │───▶ Environment
│   Goal      │    │  (planner)  │    │   (HTTP)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Pros**: Simple, low latency
**Cons**: No built-in failure recovery, no parallel execution

### Option B: Behavior Tree Generation

The LLM generates behavior trees that are then executed.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  LLM Agent  │───▶│  BT Builder │───▶│  py_trees   │───▶ Env
│   Goal      │    │ (generates) │    │ (compiles)  │    │  (executes) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Pros**: Built-in retry logic, parallel execution, hierarchical plans
**Cons**: More complex, requires BT → code translation

### Option C: Hybrid (Recommended)

LLM uses high-level planning with behavior trees for execution patterns.

```python
# LLM receives goal: "Set living room temperature to 22°C"

# 1. LLM discovers available devices
artifacts = list_artifacts("living_room")
ac = find_artifact_by_type(artifacts, "AirConditioner")

# 2. LLM checks current state
current_temp = get_property(ac, "temperature")

# 3. LLM generates plan as behavior tree
if current_temp != 22:
    tree = py_trees.composites.Sequence([
        # Precondition: AC must be on
        py_trees.composites.Selector([
            PropertyConditionNode("IsACOn", f"{ac}/properties/state", "on"),
            ActionAffordanceNode("TurnOnAC", f"{ac}/turn_on")
        ]),
        # Main action
        ActionAffordanceNode("SetTemp", f"{ac}/set_temperature", {"temperature": 22})
    ])

# 4. Execute tree with monitoring
tree.tick_once()
```

---

## Implementation Steps

### Step 1: Start the Simulator

```bash
# HomeBench
cd homebench
python smart_home_simulator.py

# Or Blocksworld
cd blocksworld
python blocksworld_simulator.py
```

### Step 2: Build LLM Tool Definitions

Define tools the LLM can call:

```python
TOOLS = [
    {
        "name": "discover_rooms",
        "description": "List all rooms/workspaces in the environment",
        "parameters": {"home_uri": "string"}
    },
    {
        "name": "discover_devices",
        "description": "List all devices in a room",
        "parameters": {"room_uri": "string"}
    },
    {
        "name": "get_device_capabilities",
        "description": "Get available actions and properties for a device",
        "parameters": {"device_uri": "string"}
    },
    {
        "name": "read_property",
        "description": "Read current value of a device property",
        "parameters": {"device_uri": "string", "property_name": "string"}
    },
    {
        "name": "execute_action",
        "description": "Execute an action on a device",
        "parameters": {"device_uri": "string", "action_name": "string", "params": "object"}
    }
]
```

### Step 3: Implement Tool Handlers

```python
from src.hmas_client import (
    list_workspaces, list_artifacts, list_properties,
    list_actions, get_property, invoke_action
)

def handle_tool_call(tool_name: str, params: dict) -> dict:
    if tool_name == "discover_rooms":
        return {"rooms": list_workspaces(params["home_uri"])}

    elif tool_name == "discover_devices":
        return {"devices": list_artifacts(params["room_uri"])}

    elif tool_name == "get_device_capabilities":
        return {
            "actions": list_actions(params["device_uri"]),
            "properties": list_properties(params["device_uri"])
        }

    elif tool_name == "read_property":
        value = get_property(params["device_uri"], params["property_name"])
        return {"value": value}

    elif tool_name == "execute_action":
        success = invoke_action(
            params["device_uri"],
            params["action_name"],
            params.get("params", {})
        )
        return {"success": success}
```

### Step 4: Build the Agent Loop

```python
import openai  # or anthropic, etc.

def agent_loop(user_goal: str, home_uri: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal}
    ]

    while True:
        response = llm.chat(messages, tools=TOOLS)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = handle_tool_call(tool_call.name, tool_call.params)
                messages.append({"role": "tool", "content": json.dumps(result)})
        else:
            # No more tool calls - agent is done
            return response.content
```

---

## Example: Complete Agent Interaction

**User Goal**: "Turn on all lights in the bathroom and set brightness to 80%"

**Agent Execution**:

```
1. discover_rooms(home_uri="http://localhost:8080/workspaces/home0#workspace")
   → Found: [..., "bathroom#workspace", ...]

2. discover_devices(room_uri="bathroom#workspace")
   → Found: ["bathroomLight#artifact", "bathroomFan#artifact"]

3. get_device_capabilities(device_uri="bathroomLight#artifact")
   → Actions: ["turnOn", "turnOff", "setBrightness", "setColor"]
   → Properties: ["state", "brightness", "color"]

4. read_property(device_uri="bathroomLight", property_name="state")
   → "off"

5. execute_action(device_uri="bathroomLight", action_name="turn_on", params={})
   → success: true

6. execute_action(device_uri="bathroomLight", action_name="set_brightness", params={"brightness": 80})
   → success: true

Agent: "Done! I turned on the bathroom light and set brightness to 80%."
```

---

## Advanced: Using Behavior Trees for Complex Plans

For multi-step goals with dependencies:

```python
from behavior_trees.affordance_nodes import *
import py_trees

def create_morning_routine_tree():
    """Turn on lights, set comfortable temperature, open blinds"""

    bedroom = "http://localhost:8080/workspaces/home0/master_bedroom/artifacts"

    return py_trees.composites.Parallel(
        name="MorningRoutine",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
        children=[
            # Light sequence
            py_trees.composites.Sequence([
                ActionAffordanceNode("LightOn", f"{bedroom}/light/turn_on"),
                ActionAffordanceNode("SetBrightness", f"{bedroom}/light/set_brightness",
                                    parameters={"brightness": 70})
            ]),

            # AC sequence
            py_trees.composites.Sequence([
                ActionAffordanceNode("ACOn", f"{bedroom}/ac/turn_on"),
                ActionAffordanceNode("SetTemp", f"{bedroom}/ac/set_temperature",
                                    parameters={"temperature": 22})
            ]),

            # Blinds
            ActionAffordanceNode("OpenBlinds", f"{bedroom}/blinds/open")
        ]
    )

# Execute
tree = create_morning_routine_tree()
tree.setup_with_descendants()
tree.tick_once()  # All three run in parallel
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/hmas_client.py` | Standalone client for environment interaction |
| `behavior_trees/affordance_nodes.py` | Behavior tree node implementations |
| `behavior_trees/blackboard_keys.py` | Standardized keys for data sharing |
| `behavior_trees/examples/basic_usage.py` | 7 complete usage examples |
| `homebench/smart_home_simulator.py` | Smart home FastAPI server |
| `blocksworld/blocksworld_simulator.py` | Blocksworld FastAPI server |

---

## Dependencies

```bash
pip install -r requirements.txt
# Key: py_trees, httpx, fastapi, uvicorn, rdflib
```

---

## Summary

To build an LLM agent for this framework:

1. **Use `src/hmas_client.py`** for discovery and simple interactions
2. **Use behavior tree nodes** for complex, multi-step plans with retry logic
3. **Define LLM tools** that wrap the client functions
4. **Build an agent loop** that handles tool calls and tracks state
5. **Start with simple goals** (single device actions) before complex routines

The framework handles the low-level HTTP communication and semantic parsing, letting the LLM focus on high-level planning and goal decomposition.
