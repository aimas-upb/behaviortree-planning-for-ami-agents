# HMAS Agent Architecture

This document explains how the agent system works and the path toward behavior tree-based planning.

## Overview

The system has three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Agent (Planning)                     │
│  - Receives user goals                                       │
│  - Explores environment to understand capabilities           │
│  - Generates execution plans                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Behavior Trees (Execution)                  │
│  - Hierarchical task decomposition                          │
│  - Built-in retry logic and failure handling                │
│  - Parallel execution support                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 HMAS Client (Communication)                  │
│  - HTTP requests to simulator                               │
│  - RDF/Turtle parsing for discovery                         │
│  - Action invocation and property reading                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Simulator (Environment)                   │
│  - HomeBench: Smart home devices                            │
│  - Blocksworld: Classical planning domain                   │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: The Environment

### Hypermedia Multi-Agent Systems (HMAS)

The environment follows W3C Thing Description (TD) standards. Everything is discoverable through URIs:

```
Workspace (home0)
├── Sub-workspace (bathroom)
│   ├── Artifact (bathroomLight)
│   │   ├── Action: turnOn (POST /turn_on)
│   │   ├── Action: turnOff (POST /turn_off)
│   │   ├── Action: setColor (POST /set_color, params: {color: [r,g,b]})
│   │   └── Property: state (GET /properties/state → "on"|"off")
│   └── Artifact (bathroomFan)
│       └── ...
├── Sub-workspace (kitchen)
│   └── ...
└── ...
```

### Key Concepts

| Term | Meaning |
|------|---------|
| **Workspace** | A container (room, area) that holds other workspaces or artifacts |
| **Artifact** | A thing/device with actions and observable properties |
| **ActionAffordance** | Something you can do (HTTP POST) |
| **PropertyAffordance** | Something you can observe (HTTP GET) |
| **Thing Description** | RDF/Turtle document describing an artifact's capabilities |

### Discovery Flow

```python
# 1. Start at entry point
workspace_uri = "http://localhost:8080/workspaces/home0#workspace"

# 2. Fetch RDF to discover contents
GET http://localhost:8080/workspaces/home0
# Returns Turtle with hmas:contains links to sub-workspaces and artifacts

# 3. Follow links to explore
GET http://localhost:8080/workspaces/home0/bathroom
# Returns Turtle with artifacts in bathroom

# 4. Inspect artifact to get affordances
GET http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight
# Returns Turtle with td:hasActionAffordance and td:hasPropertyAffordance

# 5. Interact
POST http://localhost:8080/.../bathroomLight/turn_on  # Action
GET http://localhost:8080/.../bathroomLight/properties/state  # Property
```

## Layer 2: HMAS Client (`hmas_client.py`)

Wraps the HTTP/RDF complexity into simple Python functions:

```python
from hmas_client import (
    list_workspaces,      # Discover sub-workspaces
    list_artifacts,       # Discover artifacts in a workspace
    list_actions,         # Get available actions on an artifact
    list_properties,      # Get observable properties
    get_property_by_uri,  # Read a property value
    invoke_action_by_uri, # Execute an action
)

# Example: Turn on bathroom light
workspaces = list_workspaces("http://localhost:8080/workspaces/home0#workspace")
bathroom = [w for w in workspaces if "bathroom" in w][0]

artifacts = list_artifacts(bathroom)
light = [a for a in artifacts if "Light" in a][0]

actions = list_actions(light)
# [{"name": "turnOn", "uri": "http://...turn_on", "input_schema": {}}, ...]

invoke_action_by_uri(actions[0]["uri"], {})
```

## Layer 3: Behavior Trees (`behavior_trees/`)

Behavior trees provide structured execution with:
- **Sequences**: Execute children in order, fail on first failure
- **Selectors**: Try children until one succeeds (fallback pattern)
- **Parallel**: Execute children concurrently
- **Conditions**: Check state before proceeding
- **Actions**: Execute affordances

### Node Types

```python
from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,           # Execute an action (HTTP POST)
    PropertyAffordanceNode,         # Read a property (HTTP GET)
    PropertyConditionNode,          # Check property == expected
    ComparisonPropertyConditionNode # Check with operators (>, <, in, etc.)
)
```

### Example: Ensure Light is On

```python
import py_trees

light_url = "http://localhost:8080/.../bathroomLight"

# Condition: Is light already on?
is_on = PropertyConditionNode(
    name="IsLightOn",
    property_url=f"{light_url}/properties/state",
    expected_value="on"
)

# Action: Turn it on
turn_on = ActionAffordanceNode(
    name="TurnOn",
    action_url=f"{light_url}/turn_on"
)

# Selector: If already on, succeed. Otherwise turn on.
tree = py_trees.composites.Selector(
    name="EnsureLightOn",
    children=[is_on, turn_on]
)

# Execute
tree.setup_with_descendants()
tree.tick_once()
```

### Why Behavior Trees?

| Feature | Direct API Calls | Behavior Trees |
|---------|-----------------|----------------|
| Retry on failure | Manual | Built-in |
| Parallel execution | Manual threading | Native support |
| Precondition checking | Manual | Condition nodes |
| Hierarchical plans | Flat | Natural structure |
| Reusability | Copy-paste | Composable subtrees |

## Layer 4: LLM Agent

### Current Implementation (`agent.py`)

The current agent directly calls the HMAS client through tools:

```
User: "Turn on the bathroom light"
  │
  ▼
LLM explores workspace → finds bathroom → finds light → inspects → invokes turnOn
  │
  ▼
Direct HTTP calls via hmas_client
```

This works but lacks:
- Failure recovery
- Parallel execution
- Precondition validation
- Reusable plans

### Target Architecture: BT-Planning Agent

The goal is an agent that:
1. **Explores** the environment to understand capabilities
2. **Plans** by generating behavior trees
3. **Executes** the trees for robust task completion

```
User: "Turn on all lights and set AC to 22°C"
  │
  ▼
LLM explores environment
  │
  ▼
LLM generates behavior tree:
  │
  │   Parallel
  │   ├── Sequence (lights)
  │   │   ├── EnsureLightOn(bathroom)
  │   │   ├── EnsureLightOn(kitchen)
  │   │   └── EnsureLightOn(bedroom)
  │   └── Sequence (AC)
  │       ├── Selector
  │       │   ├── IsACOn
  │       │   └── TurnOnAC
  │       └── SetTemperature(22)
  │
  ▼
Execute behavior tree with retry/recovery
```

## File Structure

```
behaviortree-planning-for-ami-agents/
├── agent.py                 # Current simple agent (direct calls)
├── hmas_client.py          # Low-level environment client
│
├── behavior_trees/
│   ├── affordance_nodes.py # BT node implementations
│   ├── http_client.py      # HTTP layer for nodes
│   ├── blackboard_keys.py  # Shared state keys
│   └── examples/           # Usage examples
│
├── homebench/
│   ├── smart_home_simulator.py    # FastAPI simulator
│   └── smart_home_to_td_converter.py
│
├── blocksworld/
│   ├── blocksworld_simulator.py   # FastAPI simulator
│   └── blocksworld_pddl_to_td_converter.py
│
└── datasets/
    ├── HomeBench/          # Smart home data (100 homes)
    └── Blocksworld/        # Planning problems
```

## Running the System

### 1. Start Simulator

```bash
# HomeBench (smart home)
uv run python homebench/smart_home_simulator.py \
    --data-dir datasets/HomeBench/hmas_format/home_description

# Blocksworld (planning domain)
uv run python blocksworld/blocksworld_simulator.py \
    --data-dir datasets/Blocksworld
```

### 2. Run Agent

```bash
# Set API key in .env
echo "OPENAI_API_KEY=sk-..." > .env

# Interactive mode
uv run python agent.py

# Single goal
uv run python agent.py "Turn on all lights in the kitchen"
```

## Next Steps: BT-Planning Agent

The next phase is to build an agent that generates and executes behavior trees.

### Architecture

```
User Goal: "Turn on lights and set AC to 22°C"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: EXPLORATION                                        │
│  - Explore workspace hierarchy                               │
│  - Discover artifacts and their affordances                  │
│  - Build capability model (what can be done, with what)      │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: PLANNING                                           │
│  - LLM analyzes goal against capability model                │
│  - Generates behavior tree specification (JSON)              │
│  - Specifies: nodes, structure, parameters, conditions       │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: EXECUTION                                          │
│  - Convert BT spec to py_trees code                          │
│  - Execute tree with tick loop                               │
│  - Monitor status, handle failures                           │
└──────────────────────────────────────────────────────────────┘
```

### Behavior Tree Specification Format

The LLM will generate a JSON specification that maps to py_trees:

```json
{
  "name": "SetupRoom",
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {
      "name": "EnsureLightOn",
      "type": "selector",
      "children": [
        {
          "name": "IsLightOn",
          "type": "condition",
          "property_url": "http://.../properties/state",
          "expected_value": "on"
        },
        {
          "name": "TurnOnLight",
          "type": "action",
          "action_url": "http://.../turn_on",
          "parameters": {}
        }
      ]
    },
    {
      "name": "ConfigureAC",
      "type": "sequence",
      "children": [
        {
          "name": "TurnOnAC",
          "type": "action",
          "action_url": "http://.../turn_on"
        },
        {
          "name": "SetTemp",
          "type": "action",
          "action_url": "http://.../set_temperature",
          "parameters": {"temperature": 22}
        }
      ]
    }
  ]
}
```

### Node Type Mapping

| JSON type | py_trees class | Purpose |
|-----------|---------------|---------|
| `"sequence"` | `py_trees.composites.Sequence` | Execute in order, fail on first failure |
| `"selector"` | `py_trees.composites.Selector` | Try until one succeeds |
| `"parallel"` | `py_trees.composites.Parallel` | Execute concurrently |
| `"action"` | `ActionAffordanceNode` | HTTP POST to action URL |
| `"condition"` | `PropertyConditionNode` | Check property value |
| `"comparison"` | `ComparisonPropertyConditionNode` | Compare with operators |

### Common Patterns

**1. Ensure State (Idempotent)**
```json
{
  "type": "selector",
  "children": [
    {"type": "condition", "expected_value": "on"},
    {"type": "action", "action_url": ".../turn_on"}
  ]
}
```

**2. Conditional Action**
```json
{
  "type": "sequence",
  "children": [
    {"type": "condition", "expected_value": true},
    {"type": "action", "action_url": ".../do_something"}
  ]
}
```

**3. Multi-Device Parallel**
```json
{
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {"type": "action", "action_url": ".../device1/turn_on"},
    {"type": "action", "action_url": ".../device2/turn_on"}
  ]
}
```

### Implementation Plan

1. **Capability Model Builder**
   - Explore environment on startup
   - Cache discovered workspaces, artifacts, affordances
   - Provide structured summary to LLM

2. **BT Spec Generator**
   - LLM tool that returns JSON behavior tree spec
   - Validate spec against discovered capabilities
   - Handle goal decomposition

3. **BT Compiler**
   - Convert JSON spec to py_trees objects
   - Wire up nodes with proper parameters
   - Return executable tree

4. **Execution Engine**
   - Run tree with tick loop
   - Collect results and status
   - Report back to user

See `behavior_trees/examples/basic_usage.py` for patterns the LLM should generate.
