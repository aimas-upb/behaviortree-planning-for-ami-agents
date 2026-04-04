# Behavior Tree Planning Agent - Architecture Documentation

> Repository note: this document was written before the repository reorganization. Canonical entrypoints now live under `scripts/experiments/`, `viewers/`, `legacy/standalone/`, and `src/`. Use the root `README.md` for the current command paths.

This document provides a complete technical overview of the system for context preservation across sessions.

## Table of Contents

1. [System Overview](#system-overview)
2. [HMAS Environment & Client](#hmas-environment--client)
3. [Agent Architectures](#agent-architectures)
4. [Capability Discovery](#capability-discovery)
5. [Behavior Tree Generation](#behavior-tree-generation)
6. [JSON-to-py_trees Compilation](#json-to-py_trees-compilation)
7. [Execution Engine](#execution-engine)
8. [Tracing System](#tracing-system)
9. [Prompting Strategies](#prompting-strategies)
10. [File Structure](#file-structure)

---

## System Overview

This project implements LLM agents that interact with **Hypermedia Multi-Agent Systems (HMAS)** environments. The key insight is that the environment exposes its capabilities through hypermedia links (like a REST API with discoverable endpoints), and the agent must:

1. **Discover** what's available (workspaces, artifacts, actions, properties)
2. **Plan** how to achieve a goal (either reactively or via behavior trees)
3. **Execute** the plan by making HTTP requests to the environment

There are two agent implementations:

| Agent | File | Approach |
|-------|------|----------|
| **Direct Agent** | `agent.py` | Reactive, one action at a time via LLM tool calls |
| **BT Agent** | `bt_agent.py` | Planned, generates behavior tree then executes |

---

## HMAS Environment & Client

### What is HMAS?

HMAS (Hypermedia Multi-Agent Systems) is a framework where environments are described using:
- **RDF/Turtle** format for semantic descriptions
- **W3C Thing Description** vocabulary for device capabilities
- **Hypermedia links** for discoverability (no hardcoded endpoints)

### Simulator

The environment runs as a local HTTP server (default: `http://localhost:8080`).

**Starting the simulator:**
```bash
uv run python -m homebench.smart_home_simulator --data-dir data/homebench/hmas/home_description
```

This serves a smart home environment with rooms (workspaces) containing devices (artifacts).

### Hypermedia Structure

```
Entry Point: http://localhost:8080/workspaces/home0#workspace
    │
    ├── contains → http://localhost:8080/workspaces/home0/bathroom#workspace
    │                   │
    │                   └── contains → http://localhost:8080/.../bathroomLight#artifact
    │                                       │
    │                                       ├── actions → turn_on, turn_off
    │                                       └── properties → state
    │
    ├── contains → http://localhost:8080/workspaces/home0/kitchen#workspace
    │                   └── ...
    └── ...
```

### The hmas_client.py Module

This module provides the HTTP/RDF interface to the environment. Key functions:

```python
# List sub-workspaces from a workspace URI
list_workspaces(workspace_uri: str) -> list[str]

# List artifacts in a workspace
list_artifacts(workspace_uri: str) -> list[str]

# Get artifact's human-readable name
get_artifact_name(artifact_uri: str) -> str

# List actions an artifact can perform
list_actions(artifact_uri: str) -> list[dict]
# Returns: [{"name": "turnOn", "uri": "http://.../turn_on", "input_schema": {...}}, ...]

# List properties an artifact exposes
list_properties(artifact_uri: str) -> list[dict]
# Returns: [{"name": "state", "uri": "http://.../properties/state", "output_schema": {...}}, ...]

# Read a property value
read_property(property_uri: str) -> Any

# Invoke an action
invoke_action(action_uri: str, payload: dict = None) -> dict
```

**RDF Parsing:** The client uses `rdflib` to parse Turtle responses and extract:
- `hmas:contains` links for hierarchy
- `td:hasActionAffordance` for actions
- `td:hasPropertyAffordance` for properties
- `schema:name` for display names

---

## Agent Architectures

### Direct Agent (`agent.py`)

**Philosophy:** Reactive, tool-calling agent. The LLM decides each action one at a time.

**Tools exposed to LLM:**
```python
explore_workspace(workspace_uri: str) -> dict
# Returns sub-workspaces and artifacts

inspect_artifact(artifact_uri: str) -> dict
# Returns actions and properties

read_property(property_uri: str) -> Any
# Returns current value

invoke_action(action_uri: str, payload: dict = None) -> dict
# Executes action, returns result
```

**Flow:**
```
User: "Turn on the bathroom light"
    │
    ▼
LLM: explore_workspace(entry_point)
    │ ← Returns list of rooms
    ▼
LLM: explore_workspace(bathroom_uri)
    │ ← Returns bathroom artifacts
    ▼
LLM: inspect_artifact(bathroomLight_uri)
    │ ← Returns actions: [turn_on, turn_off]
    ▼
LLM: invoke_action(turn_on_uri)
    │ ← Returns success
    ▼
LLM: "Done! Light is on."
```

**Characteristics:**
- Simple implementation
- No parallel execution
- Many LLM calls (one per decision)
- Discovery happens during execution

### BT Agent (`bt_agent.py`)

**Philosophy:** Plan first, execute later. Generate a complete behavior tree, then run it.

**Three Phases:**

```
┌─────────────────────────────────────────┐
│  Phase 1: CAPABILITY DISCOVERY          │
│  - Explore environment                  │
│  - Build CapabilityModel                │
│  - Modes: exhaustive, agentic, relevant │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 2: LLM PLANNING                  │
│  - Inject capabilities into prompt      │
│  - LLM generates JSON behavior tree     │
│  - Single LLM call                      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Phase 3: COMPILE & EXECUTE             │
│  - JSON → py_trees objects              │
│  - Execute with tick loop               │
│  - Built-in retry, parallel, conditions │
└─────────────────────────────────────────┘
```

**Characteristics:**
- Full plan before any execution
- Parallel execution via Parallel nodes
- Precondition checking via Condition nodes
- Single LLM call (lower latency for complex goals)
- Debuggable (can inspect tree before execution)

---

## Capability Discovery

The BT Agent needs to know what's available before planning. Three discovery modes:

### 1. Exhaustive (`--discovery exhaustive`)

**Implementation:** `build_capability_model()` in bt_agent.py

```python
def build_capability_model(entry_point, max_workspaces=5):
    model = CapabilityModel(entry_point)

    for workspace in list_workspaces(entry_point):
        for artifact in list_artifacts(workspace):
            actions = list_actions(artifact)
            properties = list_properties(artifact)
            model.artifacts[artifact] = Artifact(...)

    return model
```

**Behavior:** Explores ALL workspaces and ALL artifacts upfront.

**Trade-off:** Complete information, but many HTTP calls and long prompt.

### 2. Agentic (`--discovery agentic`)

**Implementation:** `build_capability_model_agentic()` in bt_agent.py

```python
def build_capability_model_agentic(entry_point, goal, client, model):
    # LLM-guided exploration loop
    messages = [{"role": "system", "content": DISCOVERY_SYSTEM_PROMPT}]

    while iteration < max_iterations:
        response = client.chat.completions.create(tools=DISCOVERY_TOOLS, ...)

        for tool_call in response.tool_calls:
            if tool_call.name == "explore_workspace":
                # Explore and add to capability model
            elif tool_call.name == "inspect_artifact":
                # Inspect and add to capability model
            elif tool_call.name == "done_exploring":
                return capability_model  # LLM decided it has enough
```

**Tools available to discovery LLM:**
- `explore_workspace` - See what's in a workspace
- `inspect_artifact` - See what a device can do
- `done_exploring` - Signal completion

**Behavior:** LLM decides what to explore based on the goal. For "turn on bathroom light", it might only explore: `home0 → bathroom → bathroomLight`.

**Trade-off:** Fewer HTTP calls, goal-focused, but may miss edge cases.

### 3. Relevant (`--discovery relevant`)

**Implementation:** `filter_capabilities_for_goal()` in bt_agent.py

```python
def filter_capabilities_for_goal(model, goal, client, llm_model):
    # First: exhaustive discovery
    # Then: ask LLM to filter

    filter_prompt = f"""Given goal: "{goal}"
    Which devices are relevant? Return JSON array of URIs.
    Devices: {json.dumps(artifact_list)}"""

    relevant_uris = llm.complete(filter_prompt)

    # Build filtered model with only relevant artifacts
    return filtered_model
```

**Behavior:** Exhaustive discovery, then LLM filters to goal-relevant devices before injecting into planning prompt.

**Trade-off:** Complete discovery (no missed devices), but shorter planning prompt.

### CapabilityModel Data Structure

```python
@dataclass
class Affordance:
    name: str           # "turnOn"
    uri: str            # "http://.../turn_on"
    schema: dict        # Input/output JSON schema

@dataclass
class Artifact:
    name: str           # "Bathroomlight"
    uri: str            # "http://.../bathroomLight#artifact"
    workspace: str      # "http://.../bathroom#workspace"
    actions: list[Affordance]
    properties: list[Affordance]

@dataclass
class CapabilityModel:
    entry_point: str
    workspaces: dict[str, list[str]]  # workspace_uri → artifact_uris
    artifacts: dict[str, Artifact]     # artifact_uri → Artifact

    def to_summary(self) -> str:
        # Generates markdown for LLM prompt:
        # ## bathroom
        # ### Bathroomlight
        # **Actions:** turnOn, turnOff
        # **Properties:** state
```

---

## Behavior Tree Generation

### What the LLM Generates

The LLM is given:
1. **System prompt** with node type documentation (from prompts.py)
2. **Capability model** as markdown summary
3. **User goal** as the user message

It must call the `generate_behavior_tree` tool with:
```json
{
  "tree": { /* JSON behavior tree spec */ },
  "explanation": "Why this structure was chosen"
}
```

### JSON Behavior Tree Specification

```json
{
  "name": "TurnOnBathroomLight",
  "type": "selector",
  "children": [
    {
      "name": "CheckAlreadyOn",
      "type": "condition",
      "property_url": "http://.../properties/state",
      "expected_value": "on",
      "operator": "=="
    },
    {
      "name": "TurnOn",
      "type": "action",
      "action_url": "http://.../turn_on",
      "parameters": {}
    }
  ]
}
```

### Node Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `sequence` | Run children in order, fail on first failure | `children` |
| `selector` | Run children until one succeeds | `children` |
| `parallel` | Run children concurrently | `children`, optional `policy` |
| `action` | HTTP POST to action URL | `action_url`, optional `parameters` |
| `condition` | Check property value | `property_url`, `expected_value`, optional `operator` |

### Common Patterns

**Idempotent "Ensure" Pattern (Selector):**
```json
{
  "type": "selector",
  "children": [
    {"type": "condition", "expected_value": "on"},  // Already on? SUCCESS
    {"type": "action", "action_url": ".../turn_on"} // Not on? Turn on
  ]
}
```

**Sequential Steps (Sequence):**
```json
{
  "type": "sequence",
  "children": [
    {"type": "action", "action_url": ".../turn_on"},
    {"type": "action", "action_url": ".../set_temperature", "parameters": {"temp": 22}}
  ]
}
```

**Parallel Execution:**
```json
{
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {"type": "action", "action_url": ".../light1/turn_on"},
    {"type": "action", "action_url": ".../light2/turn_on"}
  ]
}
```

---

## JSON-to-py_trees Compilation

### The compile_bt() Function

Located in `bt_agent.py`, this function recursively converts JSON specs to py_trees objects:

```python
def compile_bt(spec: dict) -> py_trees.behaviour.Behaviour:
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
        policy = get_policy(spec.get("policy", "success_on_all"))
        return py_trees.composites.Parallel(name=name, policy=policy, children=children)

    elif node_type == "action":
        return ActionAffordanceNode(
            name=name,
            action_url=spec["action_url"],
            parameters=spec.get("parameters", {}),
        )

    elif node_type == "condition":
        return PropertyConditionNode(
            name=name,
            property_url=spec["property_url"],
            expected_value=spec["expected_value"],
        )
```

### Custom py_trees Nodes

Located in `behavior_trees/affordance_nodes.py`:

**ActionAffordanceNode:**
```python
class ActionAffordanceNode(py_trees.behaviour.Behaviour):
    def __init__(self, name, action_url, parameters=None):
        self.action_url = action_url
        self.parameters = parameters or {}

    def update(self):
        try:
            result = invoke_action(self.action_url, self.parameters)
            return Status.SUCCESS
        except Exception:
            return Status.FAILURE
```

**PropertyConditionNode:**
```python
class PropertyConditionNode(py_trees.behaviour.Behaviour):
    def __init__(self, name, property_url, expected_value):
        self.property_url = property_url
        self.expected_value = expected_value

    def update(self):
        current = read_property(self.property_url)
        if current == self.expected_value:
            return Status.SUCCESS
        return Status.FAILURE
```

**ComparisonPropertyConditionNode:**
```python
class ComparisonPropertyConditionNode(PropertyConditionNode):
    # Supports operators: ==, !=, >, >=, <, <=, in, not_in, contains
    def __init__(self, name, property_url, expected_value, operator):
        self.operator = operator  # ComparisonOperator enum

    def update(self):
        current = read_property(self.property_url)
        return Status.SUCCESS if self._compare(current) else Status.FAILURE
```

---

## Execution Engine

### The execute_bt() Function

```python
def execute_bt(tree, max_ticks=10, tracer=None):
    tree.setup_with_descendants()

    results = {"ticks": 0, "success": False, "tick_history": []}

    for tick in range(max_ticks):
        tree.tick_once()
        results["tick_history"].append(tree.status.name)

        if tree.status == Status.SUCCESS:
            results["success"] = True
            break
        elif tree.status == Status.FAILURE:
            break

    tree.shutdown()
    return results
```

### py_trees Execution Model

- **Tick:** One traversal of the tree
- **Status:** Each node returns `SUCCESS`, `FAILURE`, or `RUNNING`
- **Composite behavior:**
  - **Sequence:** Succeeds if ALL children succeed (in order)
  - **Selector:** Succeeds if ANY child succeeds (tries until one works)
  - **Parallel:** Runs all children, success policy determines outcome

### Example Execution

Tree:
```
[Selector] TurnOnLight
    ├── [Condition] IsLightOn? (checks state == "on")
    └── [Action] TurnOn (POST to turn_on)
```

**Case 1: Light already on**
```
Tick 1:
  - Selector tries first child
  - Condition checks state → "on" → SUCCESS
  - Selector returns SUCCESS (first child succeeded)
Result: SUCCESS after 1 tick
```

**Case 2: Light is off**
```
Tick 1:
  - Selector tries first child
  - Condition checks state → "off" → FAILURE
  - Selector tries second child
  - Action POSTs to turn_on → SUCCESS
  - Selector returns SUCCESS
Result: SUCCESS after 1 tick
```

---

## Tracing System

### Purpose

Capture all agent activity for:
- Debugging
- Ablation analysis
- Reproducibility
- Visualization

### Implementation (`tracing.py`)

```python
class TraceEventType(Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    CAPABILITY_MODEL_START = "capability_model_start"
    CAPABILITY_MODEL_END = "capability_model_end"
    CAPABILITY_MODEL_SUMMARY = "capability_model_summary"
    WORKSPACE_DISCOVERED = "workspace_discovered"
    ARTIFACT_DISCOVERED = "artifact_discovered"
    # Discovery conversation events (for agentic mode)
    DISCOVERY_LLM_TURN = "discovery_llm_turn"      # LLM response with tool calls
    DISCOVERY_TOOL_CALL = "discovery_tool_call"    # Individual tool call
    DISCOVERY_TOOL_RESULT = "discovery_tool_result" # Tool result
    # BT generation events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_PROMPT_CONTENT = "llm_prompt_content"      # Full prompts for analysis
    BT_SPEC_GENERATED = "bt_spec_generated"
    BT_COMPILED = "bt_compiled"
    BT_TICK = "bt_tick"
    BT_EXECUTION_END = "bt_execution_end"
    ERROR = "error"

class Tracer:
    def start_trace(self, goal, model, entry_point, ablation_config): ...
    def log(self, event_type: TraceEventType, data: dict): ...
    def end_trace(self, success: bool, result: str): ...
    def save(self, output_dir: str) -> str: ...
```

### Trace File Structure

```json
{
  "trace_id": "20251222_234414_896156",
  "goal": "Turn on the bathroom light",
  "model": "gpt-4o",
  "entry_point": "http://localhost:8080/workspaces/home0#workspace",
  "start_time": "2025-12-22T23:44:14.896156",
  "end_time": "2025-12-22T23:44:18.123456",
  "success": true,
  "ablation_config": {
    "strategy": "detailed",
    "discovery": "relevant"
  },
  "events": [
    {
      "event_type": "agent_start",
      "timestamp": "2025-12-22T23:44:14.896200",
      "elapsed_ms": 0.04,
      "data": {"goal": "Turn on the bathroom light", "model": "gpt-4o"}
    },
    {
      "event_type": "capability_model_summary",
      "timestamp": "...",
      "elapsed_ms": 150.23,
      "data": {
        "discovered_capabilities": {
          "workspaces": {"bathroom": [...]},
          "stats": {"artifact_count": 1, "action_count": 2}
        },
        "prompt_representation": "# Available Devices..."
      }
    },
    {
      "event_type": "bt_spec_generated",
      "timestamp": "...",
      "elapsed_ms": 2500.00,
      "data": {
        "tree_spec": {"type": "selector", "children": [...]},
        "explanation": "Using selector pattern..."
      }
    },
    {
      "event_type": "bt_tick",
      "timestamp": "...",
      "elapsed_ms": 45.00,
      "data": {"tick": 1, "status": "SUCCESS"}
    }
  ],
  "summary": {
    "total_events": 8,
    "event_counts": {"agent_start": 1, "bt_tick": 1, ...},
    "total_time_ms": 3111.73
  }
}
```

### Usage

```bash
# Enable tracing
uv run python -m legacy.standalone.bt_agent --trace "Turn on bathroom light"

# Traces saved to traces/trace_YYYYMMDD_HHMMSS_ffffff.json

# View traces in human-readable format
uv run python -m viewers.trace_viewer --latest
uv run python -m viewers.trace_viewer traces/trace_*.json
```

### Trace Viewer (`trace_viewer.py`)

Formats traces as human-readable conversation logs:

```
[DISCOVERY LLM TURN 1] +1406ms
LLM calls 1 tool(s):
  → explore_workspace({"workspace_uri": "http://...#workspace"})
------------------------------------------------------------
[TOOL RESULT] explore_workspace
  Workspace: home0
  Sub-workspaces: ['balcony', 'bathroom', 'corridor', ...]
------------------------------------------------------------
[DISCOVERY LLM TURN 2] +733ms
LLM calls 1 tool(s):
  → explore_workspace({"workspace_uri": ".../bathroom#workspace"})
...
[BT SPEC GENERATED] +0ms
  Explanation: Using selector pattern for idempotent turn-on...

  GENERATED TREE:
  ----------------------------------------
  {
    "name": "TurnOnBathroomLight",
    "type": "selector",
    ...
  }
```

Shows the complete agent "internal monologue":
- Discovery conversation (LLM turns, tool calls, results)
- Full prompts (system prompt, user message)
- Discovered capabilities (what the LLM will see)
- Generated behavior tree JSON
- Execution ticks and results

---

## Prompting Strategies

### Purpose

Different prompting approaches for ablation studies. Defined in `prompts.py`.

### Available Strategies

| Strategy | Description | System Prompt Size |
|----------|-------------|-------------------|
| `baseline` | Minimal, just node types | ~500 chars |
| `detailed` | Full documentation of each node | ~3000 chars |
| `few_shot` | 4 example behavior trees | ~4000 chars |
| `icl` | In-context learning with reasoning | ~4500 chars |
| `icl_verbose` | Detailed methodology steps | ~5000 chars |

### Structure

```python
@dataclass
class PromptStrategy:
    name: str
    description: str
    system_prompt: str      # Contains {capability_model} placeholder
    tool_description: str   # Description for generate_behavior_tree tool

STRATEGIES = {
    "baseline": PromptStrategy(
        name="baseline",
        description="Minimal prompting",
        system_prompt="""You generate behavior trees.
Node types: sequence, selector, parallel, action, condition.
{capability_model}""",
        tool_description="Generate a behavior tree JSON spec."
    ),
    "detailed": PromptStrategy(...),
    # ...
}
```

### Usage

```bash
uv run python -m legacy.standalone.bt_agent --strategy baseline "Turn on light"
uv run python -m legacy.standalone.bt_agent --strategy detailed "Turn on light"
uv run python -m legacy.standalone.bt_agent --strategy few_shot "Turn on light"
```

---

## File Structure

```
behaviortree-planning-for-ami-agents/
│
├── agent.py                 # Direct agent (reactive, tool-calling)
├── bt_agent.py              # BT planning agent (main)
├── hmas_client.py           # HTTP/RDF client for HMAS environments
├── tracing.py               # Trace capture system
├── trace_viewer.py          # Human-readable trace viewer
├── prompts.py               # Ablation prompt strategies
│
├── behavior_trees/
│   ├── __init__.py
│   ├── affordance_nodes.py  # ActionAffordanceNode, PropertyConditionNode
│   ├── templates.py         # Pre-built tree templates (unused by LLM)
│   └── examples/
│       └── basic_usage.py   # Example patterns
│
├── traces/                  # Saved experiment traces (JSON)
│   └── trace_*.json
│
├── yggdrasil-hmas-rdf-server/  # HMAS simulator (submodule/separate repo)
│   ├── main.py              # Simulator entry point
│   └── datasets/
│       └── HomeBench/       # Smart home environment definitions
│
├── README.md                # Project overview
├── README_AGENT.md          # Direct agent documentation
├── README_EXPERIMENTS.md    # Experiment/ablation documentation
├── ARCHITECTURE.md          # This file
│
└── pyproject.toml           # Dependencies (uv project)
```

---

## Quick Reference Commands

```bash
# Start simulator (loads all 100 homes)
cd homebench
uv run python smart_home_simulator.py

# Start simulator with specific home(s)
uv run python smart_home_simulator.py --home 5           # Load only home 5
uv run python smart_home_simulator.py --home 0,1,5       # Load homes 0, 1, and 5

# Run BT agent (in project root, uses home 0 by default)
uv run python -m legacy.standalone.bt_agent "Turn on bathroom light"

# Run against a specific home
uv run python -m legacy.standalone.bt_agent --home 5 "Turn on bathroom light"

# With tracing
uv run python -m legacy.standalone.bt_agent --trace "Turn on bathroom light"
uv run python -m legacy.standalone.bt_agent --home 5 --trace "Turn on bathroom light"

# View traces (human-readable format)
uv run python -m viewers.trace_viewer --latest
uv run python -m viewers.trace_viewer traces/trace_*.json

# Different discovery modes
uv run python -m legacy.standalone.bt_agent --discovery agentic "Turn on bathroom light"
uv run python -m legacy.standalone.bt_agent --discovery relevant "Turn on bathroom light"

# Different prompting strategies
uv run python -m legacy.standalone.bt_agent --strategy few_shot "Turn on bathroom light"

# Custom entry point (overrides --home)
uv run python -m legacy.standalone.bt_agent --entry "http://localhost:8080/workspaces/home5#workspace" "Turn on light"

# Interactive mode
uv run python -m legacy.standalone.bt_agent

# Run direct agent for comparison
uv run python -m legacy.standalone.agent "Turn on bathroom light"
```

### HomeBench Dataset

The dataset contains 100 different home configurations:

```
data/homebench/hmas/home_description/
├── home_0.ttl           # RDF/Turtle description of home 0
├── home_0_state.json    # Initial device states for home 0
├── home_1.ttl
├── home_1_state.json
├── ...
├── home_99.ttl
└── home_99_state.json
```

Each home has different rooms, devices, and configurations. The `--home` flag lets you:
- **Simulator:** Control which homes to load (reduces memory for testing)
- **BT Agent:** Set which home's entry point to use for the agent

---

## Key Insights & Gotchas

1. **Selector vs Sequence for "turn on":**
   - **Selector** = idempotent (check if on, else turn on) → preferred
   - **Sequence** = conditional (only turn on if off) → fails if already on

2. **gpt-4o vs gpt-4o-mini:**
   - gpt-4o-mini struggles with complex structured output
   - gpt-4o is recommended for reliable tree generation

3. **few_shot/icl strategies:**
   - Sometimes confuse the model (it copies examples too literally)
   - `detailed` strategy works most reliably

4. **Agentic discovery:**
   - Requires goal upfront (can't use interactive mode)
   - LLM may miss devices if goal is ambiguous

5. **Property values:**
   - Usually strings: `"on"`, `"off"`, `"22"`
   - Check trace's `capability_model_summary` to see actual values

6. **Trace analysis:**
   - `elapsed_ms` is time since previous event
   - `total_time_ms` in summary is sum of all elapsed times
   - LLM response typically dominates (~3000ms)
