# Behavior Tree Planning System - Technical Documentation

This document provides exhaustive documentation of the modular behavior tree planning system for HMAS (Hypermedia Multi-Agent Systems) environments. It is designed to be self-contained so that future sessions can understand the system fully by reading this file.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Configuration System](#3-configuration-system)
4. [Discovery Module](#4-discovery-module)
5. [Planning Module](#5-planning-module)
6. [Execution Module](#6-execution-module)
7. [Prompts Module](#7-prompts-module)
8. [Runner Module](#8-runner-module)
9. [Data Flow](#9-data-flow)
10. [Extending the System](#10-extending-the-system)
11. [Ablation Dimensions](#11-ablation-dimensions)
12. [File Reference](#12-file-reference)

---

## 1. System Overview

### Purpose

This system generates and executes behavior trees for controlling smart home devices in HMAS environments. It supports multiple strategies at each phase for ablation studies comparing different approaches.

### Core Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Discovery  │───▶│  Planning   │───▶│  Execution  │
│             │    │             │    │             │
│ Affordances │    │ Reasoning   │    │ Compile &   │
│ + State     │    │ + Generate  │    │ Run Tree    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Design Principles

1. **YAML-Driven Configuration**: All experiment parameters are in YAML files for reproducibility
2. **Strategy Pattern**: Each phase has pluggable strategies (e.g., exhaustive vs agentic discovery)
3. **Clean Separation**: Discovery, Planning, and Execution are independent modules
4. **Three Output Formats**: Supports JSON IR (compiled), constrained Python code (template nodes), and unconstrained Python code (custom behaviors)

---

## 2. Architecture

### Directory Structure

```
src/
├── __init__.py              # Package exports ExperimentConfig, load_config
├── config.py                # Pydantic models for YAML configuration
├── runner.py                # CLI entry point and experiment orchestration
│
├── discovery/               # Phase 1: Environment exploration
│   ├── __init__.py          # Exports DiscoveryResult, CapabilityModel, etc.
│   ├── base.py              # Core dataclasses (CapabilityModel, EnvironmentState, etc.)
│   ├── pipeline.py          # DiscoveryPipeline orchestrates affordance + state
│   ├── affordances/         # Affordance discovery strategies
│   │   ├── __init__.py      # Factory: create_affordance_strategy()
│   │   ├── exhaustive.py    # ExhaustiveAffordanceDiscovery
│   │   ├── agentic.py       # AgenticAffordanceDiscovery
│   │   └── relevant.py      # RelevantAffordanceDiscovery
│   └── state/               # State gathering strategies
│       ├── __init__.py      # Factory: create_state_strategy()
│       ├── none.py          # NoStateGathering
│       ├── all.py           # AllStateGathering
│       └── relevant.py      # RelevantStateGathering
│
├── planning/                # Phase 2: Behavior tree generation
│   ├── __init__.py          # Exports Plan, PlanningResult, create_planner
│   ├── base.py              # Core dataclasses (Plan, PlanningResult)
│   ├── planner.py           # Main Planner class orchestrating reasoning + output
│   ├── reasoning/           # Pre-generation reasoning strategies
│   │   ├── __init__.py      # Factory: create_reasoning_strategy()
│   │   ├── none.py          # NoReasoning (direct generation)
│   │   ├── chain_of_thought.py  # ChainOfThoughtReasoning
│   │   ├── multi_turn.py    # MultiTurnReasoning
│   │   └── reflection.py    # ReflectionReasoning
│   └── output/              # Output format generators
│       ├── __init__.py      # Factory: create_output_generator()
│       ├── json_ir.py       # JsonIRGenerator
│       └── python_code.py   # PythonCodeGenerator
│
├── execution/               # Phase 3: Tree execution
│   ├── __init__.py          # Exports ExecutionResult, create_executor
│   ├── base.py              # ExecutionResult dataclass
│   ├── ir_executor.py       # IRExecutor (JSON → py_trees → execute)
│   └── code_executor.py     # CodeExecutor (safety check → exec → execute)
│
└── prompts/                 # Prompt templates for LLM
    ├── __init__.py          # Exports get_prompt, list_strategies
    ├── registry.py          # Prompt lookup by strategy + format
    ├── ir/                  # Prompts for JSON IR generation
    │   ├── __init__.py      # PROMPTS dict
    │   ├── baseline.py
    │   ├── detailed.py
    │   ├── few_shot.py
    │   └── icl.py
    └── code/                # Prompts for Python code generation
        ├── __init__.py      # PROMPTS dict
        ├── baseline.py
        └── detailed.py

experiments/
├── configs/                 # YAML experiment configurations
│   ├── baseline.yaml
│   ├── reasoning_cot.yaml
│   ├── reasoning_multi_turn.yaml
│   ├── reasoning_reflection.yaml
│   ├── python_code_direct.yaml
│   ├── python_code_with_reasoning.yaml
│   └── agentic_discovery.yaml
└── scenarios/               # Test scenarios
    └── homebench/
        └── scenario_001.yaml
```

### Module Dependencies

```
runner.py
    ├── config.py
    ├── discovery/
    │   ├── base.py
    │   ├── pipeline.py
    │   │   ├── affordances/{exhaustive,agentic,relevant}.py
    │   │   └── state/{none,all,relevant}.py
    │   └── (uses hmas_client.py for HTTP/RDF)
    ├── planning/
    │   ├── base.py
    │   ├── planner.py
    │   │   ├── reasoning/{none,chain_of_thought,multi_turn,reflection}.py
    │   │   └── output/{json_ir,python_code}.py
    │   └── (output generators use prompts/)
    └── execution/
        ├── base.py
        ├── ir_executor.py (uses behavior_trees/affordance_nodes.py)
        └── code_executor.py (uses behavior_trees/affordance_nodes.py)
```

---

## 3. Configuration System

### Location
`src/config.py`

### Pydantic Models

```python
ExperimentConfig
├── experiment: ExperimentMeta
│   ├── name: str
│   └── description: str
├── discovery: DiscoveryConfig
│   ├── affordances: AffordanceConfig
│   │   ├── strategy: "exhaustive" | "agentic" | "relevant"
│   │   └── max_workspaces: int (default: 10)
│   └── state: StateConfig
│       └── strategy: "all" | "relevant" | "none"
├── planning: PlanningConfig
│   ├── reasoning: ReasoningConfig
│   │   ├── enabled: bool
│   │   ├── strategy: "chain_of_thought" | "multi_turn" | "reflection"
│   │   └── max_turns: int (for multi_turn)
│   ├── output: OutputConfig
│   │   └── format: "json_ir" | "python_code" | "python_code_unconstrained"
│   └── prompt_strategy: str ("baseline" | "detailed" | "few_shot" | "icl")
├── execution: ExecutionConfig
│   └── max_ticks: int (default: 10)
├── model: ModelConfig
│   ├── name: str (default: "gpt-4o")
│   ├── temperature: float (default: 0.0)
│   ├── base_url: Optional[str]
│   └── api_key: Optional[str]
└── tracing: TracingConfig
    ├── enabled: bool
    ├── output_dir: str
    └── verbose: bool
```

### YAML Example

```yaml
experiment:
  name: reasoning_cot
  description: Chain-of-thought reasoning before JSON IR generation

discovery:
  affordances:
    strategy: exhaustive
    max_workspaces: 10
  state:
    strategy: all

planning:
  reasoning:
    enabled: true
    strategy: chain_of_thought
    max_turns: 3
  output:
    format: json_ir
  prompt_strategy: detailed

execution:
  max_ticks: 10

model:
  name: gpt-4o
  temperature: 0.0
```

### Loading Configs

```python
from src.config import load_config, create_default_config

# From file
config = load_config("experiments/configs/baseline.yaml")

# Default config
config = create_default_config("my_experiment")

# Programmatic modification
config.planning.reasoning.enabled = True
config.planning.reasoning.strategy = "reflection"
```

---

## 4. Discovery Module

### Location
`src/discovery/`

### Purpose
Explore the HMAS environment to discover:
1. **Affordances**: What devices exist and what actions/properties they have
2. **State**: Current values of properties (optional)

### Core Dataclasses (`base.py`)

#### Affordance
```python
@dataclass
class Affordance:
    name: str       # e.g., "turn_on"
    uri: str        # e.g., "http://localhost:8080/.../turn_on"
    schema: dict    # Input/output schema from Thing Description
```

#### Artifact
```python
@dataclass
class Artifact:
    name: str                    # e.g., "bathroomLight"
    uri: str                     # Artifact URI with #artifact fragment
    workspace: str               # Parent workspace URI
    actions: list[Affordance]    # Available actions
    properties: list[Affordance] # Observable properties
```

#### CapabilityModel
```python
@dataclass
class CapabilityModel:
    entry_point: str                           # Root workspace URI
    workspaces: dict[str, list[str]]           # workspace_uri -> artifact_uris
    artifacts: dict[str, Artifact]             # artifact_uri -> Artifact

    def to_summary(self) -> str:
        """Generate markdown summary for LLM prompt."""

    def get_all_property_uris(self) -> list[str]:
        """Get all property URIs for state gathering."""
```

#### EnvironmentState
```python
@dataclass
class EnvironmentState:
    property_values: dict[str, Any]  # property_uri -> value
    timestamp: str
    errors: dict[str, str]           # property_uri -> error message

    def to_summary(self) -> str:
        """Generate summary for LLM prompt."""
```

#### DiscoveryResult
```python
@dataclass
class DiscoveryResult:
    affordances: CapabilityModel
    state: EnvironmentState

    def to_prompt_context(self) -> str:
        """Combine affordances + state for LLM prompt."""
```

### What the Model Sees

The discovery context passed to the LLM includes:

**1. Affordances (always)**:
```markdown
## bathroom

### Bathroomlight
URI: `http://localhost:8080/.../bathroomLight#artifact`
**Actions:**
  - `turnOff`: `http://localhost:8080/.../bathroomLight/turn_off`
  - `setBrightness`: `http://localhost:8080/.../set_brightness`
      Parameters: brightness: integer, range: 0-100
  - `turnOn`: `http://localhost:8080/.../turn_on`
**Properties:**
  - `state`: `http://localhost:8080/.../properties/state` (type: string, values: ['off', 'on'])
  - `brightness`: `http://localhost:8080/.../properties/brightness` (type: integer)
```

**2. Current State (when state_strategy != 'none')**:
```markdown
# Current State

- `state`: `off` (http://localhost:8080/.../bathroomLight/properties/state)
- `brightness`: `50` (http://localhost:8080/.../bathroomLight/properties/brightness)
```

**3. Environment Interaction (in prompts)**:

The prompts also explain how to interact with the environment:
- **Actions**: HTTP POST to action URL with JSON body containing parameters
- **Properties**: HTTP GET from property URL returns JSON value

For unconstrained mode, the model has direct access to `http_client` (httpx.Client).

### Affordance Discovery Strategies

#### 1. Exhaustive (`affordances/exhaustive.py`)
- Explores ALL workspaces (up to `max_workspaces`)
- Discovers ALL artifacts and their affordances
- No LLM calls - pure HTTP/RDF exploration
- **Use when**: You want complete environment knowledge

```python
class ExhaustiveAffordanceDiscovery:
    def __init__(self, max_workspaces: int = 10):
        ...

    def discover(self, entry_point: str, goal: str | None) -> CapabilityModel:
        # Iterates through workspaces using hmas_client functions:
        # - list_workspaces(workspace_uri)
        # - list_artifacts(workspace_uri)
        # - list_actions(artifact_uri)
        # - list_properties(artifact_uri)
        # - get_artifact_name(artifact_uri)
```

#### 2. Agentic (`affordances/agentic.py`)
- LLM-guided exploration based on goal
- Uses tool-calling to decide what to explore
- Stops when LLM calls `done_exploring`
- **Use when**: You want goal-focused, efficient discovery

```python
class AgenticAffordanceDiscovery:
    def __init__(self, client: OpenAI, model: str, max_iterations: int = 15):
        ...

    def discover(self, entry_point: str, goal: str) -> CapabilityModel:
        # LLM uses tools:
        # - explore_workspace(workspace_uri) -> sub_workspaces, artifacts
        # - inspect_artifact(artifact_uri) -> actions, properties
        # - done_exploring(reason) -> stop exploration
```

#### 3. Relevant (`affordances/relevant.py`)
- Hybrid: Exhaustive discovery THEN LLM filtering
- Discovers everything, asks LLM "which artifacts are relevant?"
- **Use when**: You want complete discovery but focused planning context

```python
class RelevantAffordanceDiscovery:
    def __init__(self, client: OpenAI, model: str, max_workspaces: int = 10):
        self.exhaustive = ExhaustiveAffordanceDiscovery(max_workspaces)

    def discover(self, entry_point: str, goal: str) -> CapabilityModel:
        full_model = self.exhaustive.discover(entry_point, goal)
        return self._filter_for_goal(full_model, goal)  # LLM filtering
```

### State Gathering Strategies

#### 1. None (`state/none.py`)
- Returns empty `EnvironmentState`
- State will be read during execution via condition nodes
- **Use when**: You don't need pre-planning state knowledge

#### 2. All (`state/all.py`)
- Reads ALL property values from discovered artifacts
- Uses `hmas_client.get_property_by_uri()` for each property
- **Use when**: You want full state context for planning

#### 3. Relevant (`state/relevant.py`)
- Asks LLM which properties are relevant to the goal
- Only reads those properties
- **Use when**: You want state context but efficiency

### Discovery Pipeline (`pipeline.py`)

Orchestrates affordance discovery + state gathering:

```python
class DiscoveryPipeline:
    def __init__(self, affordance_strategy, state_strategy):
        ...

    def discover(self, entry_point: str, goal: str | None) -> DiscoveryResult:
        affordances = self.affordance_strategy.discover(entry_point, goal)
        state = self.state_strategy.gather(affordances, goal)
        return DiscoveryResult(affordances=affordances, state=state)

# Factory function
def create_discovery_pipeline(
    config: DiscoveryConfig,
    client: OpenAI | None,
    model: str
) -> DiscoveryPipeline:
    affordance_strategy = create_affordance_strategy(config.affordances.strategy, ...)
    state_strategy = create_state_strategy(config.state.strategy, ...)
    return DiscoveryPipeline(affordance_strategy, state_strategy)
```

---

## 5. Planning Module

### Location
`src/planning/`

### Purpose
Generate a behavior tree plan from the discovery context. Supports:
1. **Reasoning strategies**: Pre-generation thinking
2. **Output formats**: JSON IR or Python code

### Core Dataclasses (`base.py`)

#### Plan
```python
@dataclass
class Plan:
    format: Literal["json_ir", "python_code"]
    content: str | dict              # JSON dict or Python code string
    explanation: str                 # LLM's explanation of the plan
    reasoning_trace: list[str]       # Captured reasoning steps

    @property
    def is_json_ir(self) -> bool: ...

    @property
    def is_python_code(self) -> bool: ...
```

#### PlanningResult
```python
@dataclass
class PlanningResult:
    plan: Plan
    success: bool
    error: str | None
    llm_calls: int
    total_tokens: int
```

### Reasoning Strategies

#### 1. None (`reasoning/none.py`)
- No reasoning, passes context through unchanged
- Direct generation from discovery context
- **Use when**: Testing baseline LLM capability

```python
class NoReasoning:
    def reason(self, goal, context, client, model) -> tuple[str, list[str]]:
        return context, []  # Unchanged context, empty trace
```

#### 2. Chain of Thought (`reasoning/chain_of_thought.py`)
- Single LLM call for step-by-step analysis
- Adds structured analysis to context before generation
- **Use when**: You want single-turn thinking

```python
class ChainOfThoughtReasoning:
    def reason(self, goal, context, client, model) -> tuple[str, list[str]]:
        # System prompt asks for:
        # 1. Goal Decomposition
        # 2. Device Identification
        # 3. Action Mapping
        # 4. Dependency Analysis
        # 5. Pattern Selection
        # 6. Edge Cases

        reasoning = llm_call(...)
        enhanced_context = f"{context}\n\n## Planning Analysis\n\n{reasoning}"
        return enhanced_context, [reasoning]
```

#### 3. Multi-Turn (`reasoning/multi_turn.py`)
- Multiple LLM turns with tool-calling
- LLM uses analysis tools to explore the problem
- Stops when LLM calls `done_reasoning`
- **Use when**: You want deeper exploration

```python
class MultiTurnReasoning:
    def __init__(self, max_turns: int = 3):
        ...

    def reason(self, goal, context, client, model) -> tuple[str, list[str]]:
        # LLM uses tools:
        # - analyze_goal(sub_goals, required_devices)
        # - identify_dependencies(sequential, parallel, conditional)
        # - consider_edge_cases(edge_cases)
        # - propose_structure(root_type, structure_description, reasoning)
        # - done_reasoning(summary)

        # Returns enhanced context with all tool call results
```

#### 4. Reflection (`reasoning/reflection.py`)
- Three-phase reasoning: Generate → Critique → Refine
- Each phase is a separate LLM call
- **Use when**: You want self-improvement loop

```python
class ReflectionReasoning:
    def reason(self, goal, context, client, model) -> tuple[str, list[str]]:
        # Phase 1: Initial analysis
        initial_analysis = llm_call(INITIAL_ANALYSIS_PROMPT)

        # Phase 2: Critique
        critique = llm_call(CRITIQUE_PROMPT, initial_analysis)

        # Phase 3: Refinement
        refined = llm_call(REFINEMENT_PROMPT, initial_analysis, critique)

        # Enhanced context includes all three phases
        return enhanced_context, [initial, critique, refined]
```

### Output Generators

#### 1. JSON IR (`output/json_ir.py`)
- Uses LLM tool-calling with `generate_behavior_tree` tool
- Returns structured JSON that gets compiled to py_trees
- **Output**: `Plan(format="json_ir", content={...dict...})`

```python
class JsonIRGenerator:
    def generate(self, goal, context, client, model, prompt_strategy) -> Plan:
        system_prompt, tool_desc = get_prompt(prompt_strategy, "json_ir")

        response = client.chat.completions.create(
            model=model,
            messages=[...],
            tools=[GENERATE_BT_TOOL],
            tool_choice={"type": "function", "function": {"name": "generate_behavior_tree"}},
        )

        # Extract tree spec and explanation from tool call
        return Plan(format="json_ir", content=tree_spec, explanation=explanation)
```

**JSON IR Schema**:
```json
{
  "name": "NodeName",
  "type": "sequence | selector | parallel | action | condition",
  "children": [...],           // for composites
  "policy": "success_on_all",  // for parallel
  "action_url": "...",         // for action
  "parameters": {},            // for action (optional)
  "property_url": "...",       // for condition
  "expected_value": "...",     // for condition
  "operator": "==",            // for condition (optional)
  "value_path": [...]          // for condition (optional)
}
```

#### 2. Python Code - Constrained (`output/python_code.py`)
- Uses LLM tool-calling with `generate_behavior_tree_code` tool
- Returns Python code string using predefined template nodes
- **Output**: `Plan(format="python_code", content="...python code...")`

```python
class PythonCodeGenerator:
    def generate(self, goal, context, client, model, prompt_strategy) -> Plan:
        system_prompt, tool_desc = get_prompt(prompt_strategy, "python_code")

        # Similar to JSON IR but expects code string
        return Plan(format="python_code", content=code_string, explanation=explanation)
```

**Expected Code Structure** (uses template nodes):
```python
# Code must define 'tree' variable or 'build_tree()' function
tree = py_trees.composites.Selector(
    name="EnsureLightOn",
    memory=False,
    children=[
        PropertyConditionNode("IsOn", "http://.../properties/state", "on"),
        ActionAffordanceNode("TurnOn", "http://.../turn_on"),
    ]
)
```

#### 3. Python Code - Unconstrained (`output/python_code.py`)
- Allows custom py_trees behaviors with direct HTTP access
- Model can subclass `py_trees.behaviour.Behaviour` and implement custom logic
- Has access to `http_client` (httpx.Client) for environment interaction
- **Output**: `Plan(format="python_code_unconstrained", content="...python code...")`

```python
class UnconstrainedPythonCodeGenerator:
    def generate(self, goal, context, client, model, prompt_strategy) -> Plan:
        system_prompt, tool_desc = get_prompt(prompt_strategy, "python_code_unconstrained")
        return Plan(format="python_code_unconstrained", content=code_string, explanation=explanation)
```

**Expected Code Structure** (custom behaviors):
```python
class EnsureLightOn(py_trees.behaviour.Behaviour):
    def __init__(self, name, state_url, action_url):
        super().__init__(name)
        self.state_url = state_url
        self.action_url = action_url

    def update(self):
        # Check current state via HTTP GET
        resp = http_client.get(self.state_url)
        if resp.status_code == 200 and resp.json() == "on":
            return Status.SUCCESS  # Already on

        # Turn it on via HTTP POST
        resp = http_client.post(self.action_url)
        return Status.SUCCESS if resp.status_code == 200 else Status.FAILURE

tree = EnsureLightOn("TurnOnLight", "http://.../properties/state", "http://.../turn_on")
```

### Planner Class (`planner.py`)

Orchestrates reasoning + output generation:

```python
class Planner:
    def __init__(self, reasoning_strategy, output_generator, prompt_strategy):
        ...

    def plan(self, goal, discovery, client, model) -> PlanningResult:
        # Get context from discovery
        context = discovery.to_prompt_context()

        # Phase 1: Reasoning (optional)
        enhanced_context, reasoning_trace = self.reasoning_strategy.reason(
            goal, context, client, model
        )

        # Phase 2: Output generation
        plan = self.output_generator.generate(
            goal, enhanced_context, client, model, self.prompt_strategy
        )

        plan.reasoning_trace = reasoning_trace
        return PlanningResult(plan=plan, success=True, ...)

# Factory function
def create_planner(config: PlanningConfig) -> Planner:
    reasoning = create_reasoning_strategy(config.reasoning.strategy, ...)
    output = create_output_generator(config.output.format)
    return Planner(reasoning, output, config.prompt_strategy)
```

---

## 6. Execution Module

### Location
`src/execution/`

### Purpose
Execute the generated plan:
1. **IR Executor**: Compile JSON → py_trees → execute
2. **Code Executor**: Safety check → exec → execute

### ExecutionResult (`base.py`)

```python
@dataclass
class ExecutionResult:
    success: bool
    tree_name: str
    ticks: int                    # Number of ticks executed
    final_status: str             # "SUCCESS", "FAILURE", or "RUNNING (max ticks)"
    tick_history: list[str]       # Status at each tick
    error: str | None
```

### IR Executor (`ir_executor.py`)

Compiles JSON IR to py_trees and executes:

```python
class IRExecutor:
    def __init__(self, max_ticks: int = 10):
        ...

    def execute(self, plan: Plan) -> ExecutionResult:
        tree = self._compile(plan.content)  # JSON → py_trees
        return self._execute_tree(tree)

    def _compile(self, spec: dict) -> py_trees.behaviour.Behaviour:
        node_type = spec.get("type")

        if node_type == "sequence":
            children = [self._compile(c) for c in spec["children"]]
            return py_trees.composites.Sequence(name=..., children=children)

        elif node_type == "selector":
            children = [self._compile(c) for c in spec["children"]]
            return py_trees.composites.Selector(name=..., children=children)

        elif node_type == "parallel":
            children = [self._compile(c) for c in spec["children"]]
            policy = ...  # Based on spec["policy"]
            return py_trees.composites.Parallel(name=..., policy=policy, children=children)

        elif node_type == "action":
            return ActionAffordanceNode(
                name=spec["name"],
                action_url=spec["action_url"],
                parameters=spec.get("parameters", {})
            )

        elif node_type == "condition":
            # Uses PropertyConditionNode or ComparisonPropertyConditionNode
            # based on whether operator is specified
            ...

    def _execute_tree(self, tree) -> ExecutionResult:
        tree.setup_with_descendants()

        for tick in range(self.max_ticks):
            tree.tick_once()

            if tree.status == Status.SUCCESS:
                return ExecutionResult(success=True, ...)
            elif tree.status == Status.FAILURE:
                return ExecutionResult(success=False, ...)

        return ExecutionResult(success=False, final_status="RUNNING (max ticks)")
```

### Code Executor (`code_executor.py`)

Executes Python code with safety checks. Supports two modes:
1. **Constrained** (`python_code`): Uses predefined template nodes only
2. **Unconstrained** (`python_code_unconstrained`): Allows custom behaviors with HTTP access

```python
class CodeExecutor:
    def __init__(self, max_ticks: int = 10, unconstrained: bool = False):
        self.unconstrained = unconstrained
        self._http_client = None  # Lazy-initialized for unconstrained mode
        ...

    def execute(self, plan: Plan) -> ExecutionResult:
        is_unconstrained = plan.format == "python_code_unconstrained"
        self._check_safety(plan.content, unconstrained=is_unconstrained)
        tree = self._execute_code(plan.content, unconstrained=is_unconstrained)
        return self._execute_tree(tree)

    def _check_safety(self, code: str, unconstrained: bool = False) -> None:
        # 1. Regex pattern matching for forbidden patterns (same for both modes)
        FORBIDDEN_PATTERNS = [
            r'\bos\.(remove|unlink|rmdir|rmtree|system|popen|exec)',
            r'\bsubprocess\.',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            # ... more patterns
        ]

        # 2. AST-based import checking
        # Only allows: py_trees (both modes)

        # Raises CodeSafetyError if any check fails

    def _execute_code(self, code: str, unconstrained: bool = False) -> py_trees.behaviour.Behaviour:
        if unconstrained:
            # Unconstrained mode: full py_trees + http_client
            safe_globals = {
                "__builtins__": {...builtins with __build_class__...},
                "py_trees": py_trees,
                "Status": py_trees.common.Status,
                "http_client": self.http_client,  # httpx.Client for HTTP access
            }
        else:
            # Constrained mode: template nodes only
            safe_globals = {
                "__builtins__": {...limited builtins...},
                "py_trees": py_trees,
                "ActionAffordanceNode": ActionAffordanceNode,
                "PropertyConditionNode": PropertyConditionNode,
                "ComparisonPropertyConditionNode": ComparisonPropertyConditionNode,
                "ComparisonOperator": ComparisonOperator,
            }

        local_namespace = {}
        exec(code, safe_globals, local_namespace)

        # Extract 'tree' variable or call 'build_tree()'
        # Also handles py_trees.trees.BehaviourTree wrapper
        if "tree" in local_namespace:
            tree = local_namespace["tree"]
            if isinstance(tree, py_trees.trees.BehaviourTree):
                tree = tree.root  # Extract root from wrapper
            return tree
```

### Safety Checks

The code executor blocks:
- File system operations (`os.remove`, `shutil.rmtree`, etc.)
- Process execution (`subprocess`, `os.system`, `os.popen`)
- Dynamic code execution (`eval`, `exec`, `compile`)
- Dangerous imports (only `py_trees` is allowed)
- Attribute access tricks (`__builtins__`, `__class__`, etc.)

---

## 7. Prompts Module

### Location
`src/prompts/`

### Purpose
Provide prompt templates for different strategies and output formats.

### Registry (`registry.py`)

```python
def get_prompt(strategy: str, output_format: str) -> tuple[str, str]:
    """
    Returns (system_prompt, tool_description) for the given strategy and format.

    Args:
        strategy: "baseline" | "detailed" | "few_shot" | "icl"
        output_format: "json_ir" | "python_code" | "python_code_unconstrained"
    """
    if output_format == "json_ir":
        prompts = ir_prompts.PROMPTS
    elif output_format == "python_code":
        prompts = code_prompts.PROMPTS
    elif output_format == "python_code_unconstrained":
        prompts = code_prompts.UNCONSTRAINED_PROMPTS

    return prompts[strategy]  # (system_prompt, tool_description)
```

### Prompt Strategies

#### 1. Baseline
- Minimal prompting
- Tests raw LLM capability
- Just lists available devices and basic node types

#### 2. Detailed
- Comprehensive documentation
- Explains all node types, their semantics, and schemas
- Includes execution semantics (sequence stops on failure, etc.)

#### 3. Few-Shot
- Example-based learning
- Shows 4 diverse examples:
  1. Idempotent turn-on (selector pattern)
  2. Multiple devices in parallel
  3. Sequential with parameters
  4. Conditional action

#### 4. ICL (In-Context Learning)
- Shows reasoning traces
- Demonstrates step-by-step planning process
- Emphasizes dependency analysis and pattern selection

### Unconstrained Prompts (`code/unconstrained.py`)

For `python_code_unconstrained` format, prompts include:
- Full py_trees documentation (composite nodes, behavior lifecycle)
- Custom behavior creation patterns (subclassing `py_trees.behaviour.Behaviour`)
- HTTP client usage (`http_client.get()`, `http_client.post()`)
- Common patterns: retry, polling, idempotent actions
- Environment interaction documentation

### Prompt Structure

Each prompt has two parts:

1. **System Prompt**: Contains `{capability_model}` placeholder for discovery context
2. **Tool Description**: Describes the tool schema and expected output

```python
# Example from prompts/ir/detailed.py
DETAILED_SYSTEM = """You are a behavior tree planning agent...

## Node Types
...

## Available Devices
{capability_model}
"""

DETAILED_TOOL = """Generate a behavior tree specification...

## Node Schema
...
"""
```

---

## 8. Runner Module

### Location
`src/runner.py`

### Purpose
CLI entry point that orchestrates the complete pipeline.

### Main Function Flow

```python
def run_experiment(config, goal, entry_point, client) -> dict:
    # Phase 1: Discovery
    discovery_pipeline = create_discovery_pipeline(config.discovery, client, model)
    discovery_result = discovery_pipeline.discover(entry_point, goal)

    # Phase 2: Planning
    planner = create_planner(config.planning)
    planning_result = planner.plan(goal, discovery_result, client, model)

    # Phase 3: Execution
    executor = create_executor(config.planning.output.format, config.execution.max_ticks)
    execution_result = executor.execute(planning_result.plan)

    return {
        "config": config.model_dump(),
        "discovery": discovery_result.to_dict(),
        "planning": planning_result.to_dict(),
        "execution": execution_result.to_dict(),
        "success": execution_result.success,
    }
```

### CLI Arguments

```bash
uv run python -m src.runner [OPTIONS]

# Config
--config PATH           # YAML config file

# Goal/Environment
--goal TEXT            # Goal to accomplish (required)
--home INT             # Home ID (0-99), default: 0
--entry URI            # Entry point URI (overrides --home)

# Model
--model NAME           # Model name, default: gpt-4o
--base-url URL         # API base URL
--api-key KEY          # API key

# Discovery overrides
--discovery-affordances {exhaustive,agentic,relevant}
--discovery-state {all,relevant,none}

# Planning overrides
--planning-reasoning {none,chain_of_thought,multi_turn,reflection}
--planning-output {json_ir,python_code,python_code_unconstrained}
--prompt-strategy {baseline,detailed,few_shot,icl}

# Output
--output DIR           # Save results to directory
--verbose              # Enable debug logging
```

### Usage Examples

```bash
# Basic usage with config
uv run python -m src.runner \
    --config experiments/configs/baseline.yaml \
    --goal "Turn on bathroom light"

# Override config options
uv run python -m src.runner \
    --config experiments/configs/baseline.yaml \
    --goal "Turn on bathroom light" \
    --planning-reasoning chain_of_thought \
    --discovery-state all

# Full custom run
uv run python -m src.runner \
    --goal "Set up bedroom for sleep" \
    --home 0 \
    --discovery-affordances exhaustive \
    --discovery-state all \
    --planning-reasoning reflection \
    --planning-output json_ir \
    --prompt-strategy icl \
    --output results/
```

---

## 9. Data Flow

### Complete Pipeline Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                           CONFIGURATION                                 │
│  ExperimentConfig (from YAML or CLI)                                   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASE 1: DISCOVERY                                │
│                                                                        │
│  Entry Point URI + Goal                                                │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                   │
│  │ Affordance Strategy │    │ State Strategy      │                   │
│  │ (exhaustive/agentic/│    │ (none/all/relevant) │                   │
│  │  relevant)          │    │                     │                   │
│  └─────────────────────┘    └─────────────────────┘                   │
│         │                            │                                 │
│         ▼                            ▼                                 │
│  CapabilityModel              EnvironmentState                         │
│  - workspaces                 - property_values                        │
│  - artifacts                  - errors                                 │
│    - actions                                                           │
│    - properties                                                        │
│         │                            │                                 │
│         └──────────┬─────────────────┘                                 │
│                    ▼                                                   │
│            DiscoveryResult                                             │
│            - affordances: CapabilityModel                              │
│            - state: EnvironmentState                                   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: PLANNING                                 │
│                                                                        │
│  DiscoveryResult.to_prompt_context() → context string                  │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────┐                                              │
│  │ Reasoning Strategy  │ (optional)                                   │
│  │ (none/cot/multi/    │                                              │
│  │  reflection)        │                                              │
│  └─────────────────────┘                                              │
│         │                                                              │
│         ▼                                                              │
│  enhanced_context + reasoning_trace                                    │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────┐                                              │
│  │ Output Generator    │                                              │
│  │ (json_ir/           │                                              │
│  │  python_code)       │                                              │
│  └─────────────────────┘                                              │
│         │                                                              │
│         ▼                                                              │
│  Plan                                                                  │
│  - format: "json_ir" | "python_code"                                  │
│  - content: dict | str                                                │
│  - explanation: str                                                   │
│  - reasoning_trace: list[str]                                         │
│         │                                                              │
│         ▼                                                              │
│  PlanningResult                                                        │
│  - plan: Plan                                                         │
│  - success: bool                                                      │
│  - llm_calls: int                                                     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: EXECUTION                                │
│                                                                        │
│  Plan                                                                  │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                   │
│  │ IR Executor         │ OR │ Code Executor       │                   │
│  │ - _compile()        │    │ - _check_safety()   │                   │
│  │ - _execute_tree()   │    │ - _execute_code()   │                   │
│  └─────────────────────┘    │ - _execute_tree()   │                   │
│                             └─────────────────────┘                   │
│         │                            │                                 │
│         └──────────┬─────────────────┘                                 │
│                    ▼                                                   │
│  py_trees.behaviour.Behaviour (compiled tree)                          │
│         │                                                              │
│         ▼                                                              │
│  Tick loop (up to max_ticks)                                          │
│  - tree.tick_once()                                                   │
│  - Check status: SUCCESS/FAILURE/RUNNING                              │
│         │                                                              │
│         ▼                                                              │
│  ExecutionResult                                                       │
│  - success: bool                                                      │
│  - ticks: int                                                         │
│  - final_status: str                                                  │
│  - tick_history: list[str]                                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Extending the System

### Adding a New Affordance Discovery Strategy

1. Create `src/discovery/affordances/my_strategy.py`:
```python
from ..base import CapabilityModel

class MyAffordanceDiscovery:
    def __init__(self, ...):
        ...

    def discover(self, entry_point: str, goal: str | None) -> CapabilityModel:
        ...
```

2. Register in `src/discovery/affordances/__init__.py`:
```python
from .my_strategy import MyAffordanceDiscovery

def create_affordance_strategy(strategy: str, ...):
    ...
    elif strategy == "my_strategy":
        return MyAffordanceDiscovery(...)
```

3. Add to config validation in `src/config.py`:
```python
class AffordanceConfig(BaseModel):
    strategy: Literal["exhaustive", "agentic", "relevant", "my_strategy"]
```

### Adding a New Reasoning Strategy

1. Create `src/planning/reasoning/my_reasoning.py`:
```python
class MyReasoning:
    def reason(self, goal, context, client, model) -> tuple[str, list[str]]:
        # Return (enhanced_context, reasoning_trace)
        ...
```

2. Register in `src/planning/reasoning/__init__.py`

3. Add to config validation in `src/config.py`

### Adding a New Prompt Strategy

1. Add prompts for JSON IR in `src/prompts/ir/my_prompt.py`:
```python
MY_SYSTEM = """..."""
MY_TOOL = """..."""
```

2. Add prompts for Python code in `src/prompts/code/my_prompt.py`

3. Register in the respective `__init__.py` files:
```python
PROMPTS = {
    ...
    "my_prompt": (MY_SYSTEM, MY_TOOL),
}
```

### Adding a New Output Format

1. Create `src/planning/output/my_format.py`:
```python
from ..base import Plan

class MyFormatGenerator:
    def generate(self, goal, context, client, model, prompt_strategy) -> Plan:
        ...
```

2. Create `src/execution/my_format_executor.py`:
```python
from .base import ExecutionResult

class MyFormatExecutor:
    def execute(self, plan: Plan) -> ExecutionResult:
        ...
```

3. Update factories in respective `__init__.py` files

4. Add to config validation

---

## 11. Ablation Dimensions

The system supports the following ablation dimensions, all configurable via YAML:

| Dimension | Options | Config Path |
|-----------|---------|-------------|
| **Affordance Discovery** | `exhaustive`, `agentic`, `relevant` | `discovery.affordances.strategy` |
| **State Gathering** | `none`, `all`, `relevant`, `agentic` | `discovery.state.strategy` |
| **Reasoning** | `none`, `chain_of_thought`, `multi_turn`, `reflection` | `planning.reasoning.strategy` |
| **Output Format** | `json_ir`, `python_code`, `python_code_unconstrained` | `planning.output.format` |
| **Prompt Strategy** | `baseline`, `detailed`, `few_shot`, `icl` | `planning.prompt_strategy` |
| **Model** | Any OpenAI-compatible model | `model.name` |

### Ablation Matrix

For a full ablation study, you could test:
- 3 affordance strategies × 4 state strategies = 12 discovery configs
- 4 reasoning strategies × 3 output formats × 4 prompt strategies = 48 planning configs
- Total: 12 × 48 = 576 configurations

Example configs provided cover key combinations:
- `baseline.yaml`: exhaustive/none + no reasoning + json_ir + detailed
- `reasoning_cot.yaml`: exhaustive/all + chain_of_thought + json_ir + detailed
- `reasoning_multi_turn.yaml`: exhaustive/all + multi_turn + json_ir + detailed
- `reasoning_reflection.yaml`: exhaustive/all + reflection + json_ir + detailed
- `python_code_direct.yaml`: exhaustive/none + no reasoning + python_code + detailed
- `python_code_with_reasoning.yaml`: exhaustive/all + chain_of_thought + python_code + detailed
- `agentic_discovery.yaml`: agentic/relevant + no reasoning + json_ir + detailed

Additional ablation configs in `run_ablation.py`:
- `python_unconstrained`: exhaustive/all + no reasoning + python_code_unconstrained
- `python_unconstrained_cot`: exhaustive/all + chain_of_thought + python_code_unconstrained
- `fully_agentic`: agentic affordances + agentic state + json_ir
- `fully_agentic_cot`: agentic affordances + agentic state + chain_of_thought + json_ir

---

## 12. File Reference

### Core Files

| File | Purpose |
|------|---------|
| `src/__init__.py` | Package exports |
| `src/config.py` | Pydantic models for configuration |
| `src/runner.py` | CLI and experiment orchestration |

### Discovery Module

| File | Purpose |
|------|---------|
| `src/discovery/__init__.py` | Module exports |
| `src/discovery/base.py` | CapabilityModel, EnvironmentState, DiscoveryResult |
| `src/discovery/pipeline.py` | DiscoveryPipeline, create_discovery_pipeline() |
| `src/discovery/affordances/exhaustive.py` | ExhaustiveAffordanceDiscovery |
| `src/discovery/affordances/agentic.py` | AgenticAffordanceDiscovery |
| `src/discovery/affordances/relevant.py` | RelevantAffordanceDiscovery |
| `src/discovery/state/none.py` | NoStateGathering |
| `src/discovery/state/all.py` | AllStateGathering |
| `src/discovery/state/relevant.py` | RelevantStateGathering |

### Planning Module

| File | Purpose |
|------|---------|
| `src/planning/__init__.py` | Module exports |
| `src/planning/base.py` | Plan, PlanningResult |
| `src/planning/planner.py` | Planner, create_planner() |
| `src/planning/reasoning/none.py` | NoReasoning |
| `src/planning/reasoning/chain_of_thought.py` | ChainOfThoughtReasoning |
| `src/planning/reasoning/multi_turn.py` | MultiTurnReasoning |
| `src/planning/reasoning/reflection.py` | ReflectionReasoning |
| `src/planning/output/json_ir.py` | JsonIRGenerator |
| `src/planning/output/python_code.py` | PythonCodeGenerator |

### Execution Module

| File | Purpose |
|------|---------|
| `src/execution/__init__.py` | Module exports, create_executor() |
| `src/execution/base.py` | ExecutionResult |
| `src/execution/ir_executor.py` | IRExecutor (JSON → py_trees) |
| `src/execution/code_executor.py` | CodeExecutor (Python code) |

### Prompts Module

| File | Purpose |
|------|---------|
| `src/prompts/__init__.py` | Module exports |
| `src/prompts/registry.py` | get_prompt(), list_strategies() |
| `src/prompts/ir/*.py` | JSON IR prompt templates |
| `src/prompts/code/baseline.py` | Minimal Python code prompt |
| `src/prompts/code/detailed.py` | Comprehensive Python code prompt (constrained) |
| `src/prompts/code/unconstrained.py` | Custom behavior prompts (unconstrained) |

### External Dependencies

| File | Purpose |
|------|---------|
| `hmas_client.py` | HTTP/RDF client for HMAS environments |
| `behavior_trees/affordance_nodes.py` | ActionAffordanceNode, PropertyConditionNode, etc. |

---

## Quick Reference: Common Operations

### Run an experiment
```bash
uv run python -m src.runner --config experiments/configs/baseline.yaml --goal "Turn on light"
```

### Load config programmatically
```python
from src.config import load_config
config = load_config("experiments/configs/baseline.yaml")
```

### Run pipeline programmatically
```python
from openai import OpenAI
from src.config import load_config
from src.discovery import create_discovery_pipeline
from src.planning import create_planner
from src.execution import create_executor

config = load_config("baseline.yaml")
client = OpenAI()

discovery = create_discovery_pipeline(config.discovery, client, config.model.name)
discovery_result = discovery.discover("http://localhost:8080/workspaces/home0#workspace", "Turn on light")

planner = create_planner(config.planning)
planning_result = planner.plan("Turn on light", discovery_result, client, config.model.name)

executor = create_executor(config.planning.output.format, config.execution.max_ticks)
execution_result = executor.execute(planning_result.plan)
```

### Add a new config
```yaml
# experiments/configs/my_experiment.yaml
experiment:
  name: my_experiment
  description: My custom experiment

discovery:
  affordances:
    strategy: exhaustive
  state:
    strategy: all

planning:
  reasoning:
    enabled: true
    strategy: reflection
  output:
    format: json_ir
  prompt_strategy: icl

execution:
  max_ticks: 15

model:
  name: gpt-4o
```

---

*Last updated: January 2026*
