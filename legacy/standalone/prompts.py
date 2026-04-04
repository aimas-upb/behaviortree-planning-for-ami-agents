"""
Prompting strategies for behavior tree generation.

Provides modular prompt configurations for ablation studies:
- baseline: Minimal prompting
- detailed: Comprehensive node type documentation
- few_shot: Example-based learning
- icl: In-context learning with reasoning traces

Each strategy provides:
- system_prompt: The system message template
- tool_description: The tool description for generate_behavior_tree
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptStrategy:
    """A complete prompting strategy configuration."""
    name: str
    system_prompt: str
    tool_description: str
    description: str  # Human-readable description for logging


# =============================================================================
# Strategy: BASELINE
# Minimal prompting - tests raw LLM capability
# =============================================================================

BASELINE_SYSTEM = """You are a behavior tree planning agent. Generate behavior tree specifications for the given goal.

Use the generate_behavior_tree tool with both 'tree' and 'explanation' fields.
Use EXACT URIs from the capability model.

## Available Devices
{capability_model}
"""

BASELINE_TOOL = """Generate a behavior tree specification.

Node types: sequence, selector, parallel, action, condition
- Composites need 'children' array
- Actions need 'action_url'
- Conditions need 'property_url' and 'expected_value'

Every node needs 'name' and 'type'."""


# =============================================================================
# Strategy: DETAILED
# Comprehensive documentation of node types and semantics
# =============================================================================

DETAILED_SYSTEM = """You are a behavior tree planning agent. You generate executable behavior tree specifications.

## Behavior Tree Concepts

A behavior tree is a hierarchical structure where:
- **Composite nodes** control execution flow
- **Leaf nodes** perform actions or check conditions
- Execution follows a tick-based model

## Node Types

### Composite Nodes (have 'children' array)

1. **sequence**: Execute children left-to-right. Returns SUCCESS if all succeed, FAILURE on first failure.
   Use for: ordered steps that must all complete.

2. **selector**: Try children left-to-right. Returns SUCCESS on first success, FAILURE if all fail.
   Use for: fallback patterns, "try this, else try that".

3. **parallel**: Execute all children simultaneously.
   - policy: "success_on_all" (default) or "success_on_one"
   Use for: independent concurrent operations.

### Leaf Nodes

4. **action**: Execute an HTTP POST to action_url.
   Required: action_url
   Optional: parameters (object)

5. **condition**: Check if property matches expected value via HTTP GET.
   Required: property_url, expected_value
   Optional: operator ("==", "!=", ">", "<", ">=", "<="), value_path (for nested JSON)

## Rules
1. ALWAYS use generate_behavior_tree with 'tree' and 'explanation'
2. Use EXACT URIs from the capability model
3. Every node MUST have 'name' and 'type'
4. Use selector pattern for idempotent operations: check-before-act

## Available Devices
{capability_model}
"""

DETAILED_TOOL = """Generate a behavior tree specification that will be compiled to py_trees and executed.

## Node Schema

Composite nodes:
```
{
  "name": "NodeName",
  "type": "sequence" | "selector" | "parallel",
  "children": [...],
  "policy": "success_on_all" | "success_on_one"  // only for parallel
}
```

Action nodes:
```
{
  "name": "ActionName",
  "type": "action",
  "action_url": "http://...",
  "parameters": {}  // optional
}
```

Condition nodes:
```
{
  "name": "ConditionName",
  "type": "condition",
  "property_url": "http://.../properties/...",
  "expected_value": "on" | 25 | true,
  "operator": "==" | "!=" | ">" | "<"  // optional, default "=="
}
```

## Execution Semantics
- sequence: All must succeed, stops on first failure
- selector: Returns on first success (fallback pattern)
- parallel: Runs all children concurrently
- action: HTTP POST, SUCCESS on 2xx response
- condition: HTTP GET, SUCCESS if value matches expected"""


# =============================================================================
# Strategy: FEW_SHOT
# Example-based learning with diverse patterns
# =============================================================================

FEW_SHOT_SYSTEM = """You are a behavior tree planning agent. Learn from these examples to generate behavior trees.

## Example 1: Turn on a single device (idempotent)
Goal: "Turn on the bathroom light"
```json
{{
  "name": "EnsureBathroomLightOn",
  "type": "selector",
  "children": [
    {{"name": "IsLightOn", "type": "condition", "property_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/properties/state", "expected_value": "on"}},
    {{"name": "TurnOnLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/turn_on"}}
  ]
}}
```
Explanation: Selector checks if already on, only turns on if needed.

## Example 2: Multiple devices in parallel
Goal: "Turn on lights in bathroom and corridor"
```json
{{
  "name": "TurnOnMultipleLights",
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {{"name": "BathroomLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/turn_on"}},
    {{"name": "CorridorLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/corridor/artifacts/corridorLight/turn_on"}}
  ]
}}
```
Explanation: Parallel executes both actions concurrently.

## Example 3: Sequential with parameters
Goal: "Turn on AC and set to 22 degrees"
```json
{{
  "name": "ConfigureAC",
  "type": "sequence",
  "children": [
    {{"name": "TurnOnAC", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/turn_on"}},
    {{"name": "SetTemp", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/set_temperature", "parameters": {{"temperature": 22}}}}
  ]
}}
```
Explanation: Sequence ensures AC is on before setting temperature.

## Example 4: Conditional action
Goal: "If temperature is above 25, turn on AC"
```json
{{
  "name": "ConditionalCooling",
  "type": "sequence",
  "children": [
    {{"name": "IsHot", "type": "condition", "property_url": "http://localhost:8080/.../properties/temperature", "expected_value": 25, "operator": ">"}},
    {{"name": "TurnOnAC", "type": "action", "action_url": "http://localhost:8080/.../turn_on"}}
  ]
}}
```
Explanation: Sequence with condition as guard - action only runs if condition passes.

## Rules
- Use EXACT URIs from capability model
- Every node needs 'name' and 'type'
- Composites need 'children'

## Available Devices
{capability_model}
"""

FEW_SHOT_TOOL = """Generate a behavior tree following the patterns shown in the examples.

Node types:
- sequence: {type:"sequence", children:[...]} - All must succeed
- selector: {type:"selector", children:[...]} - First success wins
- parallel: {type:"parallel", policy:"success_on_all", children:[...]}
- action: {type:"action", action_url:"...", parameters:{}}
- condition: {type:"condition", property_url:"...", expected_value:...}

Use EXACT URIs from the capability model."""


# =============================================================================
# Strategy: ICL (In-Context Learning)
# Reasoning traces showing step-by-step planning
# =============================================================================

ICL_SYSTEM = """You are a behavior tree planning agent. Think step-by-step to generate optimal trees.

## Planning Process

When generating a behavior tree, follow this reasoning:

1. **Identify the goal components**: What devices/actions are needed?
2. **Check dependencies**: Do any actions depend on others?
3. **Choose structure**:
   - Independent actions → parallel
   - Dependent actions → sequence
   - Idempotent operation → selector (check-then-act)
4. **Map to URIs**: Find exact URIs from capability model
5. **Build tree bottom-up**: Start with leaf nodes, compose into tree

## Reasoning Example

Goal: "Turn on bathroom light and set brightness to 80"

Reasoning:
1. Goal components: turn on light, set brightness
2. Dependencies: Must be ON before setting brightness → sequence
3. But turning on should be idempotent → selector for turn-on part
4. URIs:
   - State check: .../bathroomLight/properties/state
   - Turn on: .../bathroomLight/turn_on
   - Brightness: .../bathroomLight/set_brightness
5. Tree structure:
   sequence [
     selector [ condition(state=on), action(turn_on) ]  // idempotent on
     action(set_brightness, brightness=80)              // then set
   ]

Result:
```json
{{
  "name": "ConfigureBathroomLight",
  "type": "sequence",
  "children": [
    {{
      "name": "EnsureOn",
      "type": "selector",
      "children": [
        {{"name": "IsOn", "type": "condition", "property_url": ".../properties/state", "expected_value": "on"}},
        {{"name": "TurnOn", "type": "action", "action_url": ".../turn_on"}}
      ]
    }},
    {{"name": "SetBrightness", "type": "action", "action_url": ".../set_brightness", "parameters": {{"brightness": 80}}}}
  ]
}}
```

## Rules
- Use EXACT URIs from capability model
- Every node needs 'name' and 'type'
- Think through dependencies before building tree

## Available Devices
{capability_model}
"""

ICL_TOOL = """Generate a behavior tree by reasoning through the problem.

Think about:
1. What devices/actions are needed?
2. What dependencies exist between actions?
3. What structure best fits? (parallel, sequence, selector)
4. Should operations be idempotent? (use selector pattern)

Node types:
- sequence: ordered, dependent actions
- selector: fallback/idempotent pattern
- parallel: independent concurrent actions
- action: HTTP POST with optional parameters
- condition: HTTP GET and compare value

Include your reasoning in the explanation field."""


# =============================================================================
# Strategy: ICL_VERBOSE
# Even more detailed reasoning with failure handling
# =============================================================================

ICL_VERBOSE_SYSTEM = """You are an expert behavior tree planning agent. Use detailed reasoning to generate robust trees.

## Behavior Tree Theory

Behavior trees originated in game AI. Key concepts:
- **Tick**: One execution cycle through the tree
- **Status**: Each node returns SUCCESS, FAILURE, or RUNNING
- **Memory**: Some nodes remember state between ticks

### Node Semantics (detailed)

**Sequence** (memory=True):
- Ticks children left-to-right
- On SUCCESS: move to next child
- On FAILURE: return FAILURE (short-circuit)
- All succeed → return SUCCESS
- Use for: ordered dependencies

**Selector** (memory=False):
- Ticks children left-to-right
- On SUCCESS: return SUCCESS immediately
- On FAILURE: try next child
- All fail → return FAILURE
- Use for: fallbacks, "try until one works"

**Parallel** (policy):
- Ticks ALL children every tick
- success_on_all: SUCCESS when all succeed
- success_on_one: SUCCESS when any succeeds
- Use for: concurrent independent actions

## Planning Methodology

### Step 1: Decompose the goal
- List all required state changes
- Identify target devices
- Note any parameters

### Step 2: Analyze dependencies
- Which actions must complete before others?
- Which can run in parallel?
- Which should be idempotent?

### Step 3: Design tree structure
- Draw the tree mentally
- Consider failure modes
- Apply patterns:
  * Check-then-act: selector[condition, action]
  * Ordered steps: sequence[action1, action2]
  * Fan-out: parallel[action1, action2, action3]

### Step 4: Map to concrete URIs
- Find exact URIs from capability model
- Verify action parameters match schema
- Ensure property paths are correct

## Example with Full Reasoning

Goal: "Set up bedroom for sleep: dim lights to 20%, turn off AC, close blinds"

**Step 1: Decompose**
- Light: set brightness to 20
- AC: turn off
- Blinds: close

**Step 2: Dependencies**
- These are independent operations
- Light dimming is a single action (not turn on + set)
- All can run in parallel

**Step 3: Structure**
```
parallel [
  action(set_brightness, 20)
  action(turn_off_ac)
  action(close_blinds)
]
```

**Step 4: URIs** (from capability model)
- Light: .../masterBedroomLight/set_brightness
- AC: .../masterBedroomAirConditioner/turn_off
- Blinds: .../masterBedroomBlinds/close

Final tree:
```json
{{
  "name": "BedroomSleepMode",
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {{"name": "DimLights", "type": "action", "action_url": ".../set_brightness", "parameters": {{"brightness": 20}}}},
    {{"name": "TurnOffAC", "type": "action", "action_url": ".../turn_off"}},
    {{"name": "CloseBlinds", "type": "action", "action_url": ".../close"}}
  ]
}}
```

## Available Devices
{capability_model}
"""

ICL_VERBOSE_TOOL = """Generate a behavior tree using detailed reasoning.

Follow the 4-step methodology:
1. Decompose the goal into required state changes
2. Analyze dependencies between actions
3. Design tree structure using appropriate patterns
4. Map to concrete URIs from capability model

Patterns:
- Idempotent: selector[condition(state=X), action(set_to_X)]
- Sequential: sequence[action1, action2, ...]
- Concurrent: parallel[action1, action2, ...]
- Conditional: sequence[condition(guard), action]

Include your full reasoning in the explanation field."""


# =============================================================================
# Registry and Factory
# =============================================================================

STRATEGIES: dict[str, PromptStrategy] = {
    "baseline": PromptStrategy(
        name="baseline",
        system_prompt=BASELINE_SYSTEM,
        tool_description=BASELINE_TOOL,
        description="Minimal prompting - tests raw LLM capability",
    ),
    "detailed": PromptStrategy(
        name="detailed",
        system_prompt=DETAILED_SYSTEM,
        tool_description=DETAILED_TOOL,
        description="Comprehensive node type documentation",
    ),
    "few_shot": PromptStrategy(
        name="few_shot",
        system_prompt=FEW_SHOT_SYSTEM,
        tool_description=FEW_SHOT_TOOL,
        description="Example-based learning with diverse patterns",
    ),
    "icl": PromptStrategy(
        name="icl",
        system_prompt=ICL_SYSTEM,
        tool_description=ICL_TOOL,
        description="In-context learning with reasoning traces",
    ),
    "icl_verbose": PromptStrategy(
        name="icl_verbose",
        system_prompt=ICL_VERBOSE_SYSTEM,
        tool_description=ICL_VERBOSE_TOOL,
        description="Detailed reasoning with methodology",
    ),
}


def get_strategy(name: str) -> PromptStrategy:
    """Get a prompting strategy by name."""
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGIES[name]


def list_strategies() -> list[str]:
    """List available strategy names."""
    return list(STRATEGIES.keys())


def get_strategy_descriptions() -> dict[str, str]:
    """Get descriptions of all strategies."""
    return {name: s.description for name, s in STRATEGIES.items()}
