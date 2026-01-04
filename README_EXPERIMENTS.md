# Behavior Tree Planning Agent - Experiments & Findings

This document describes the experimental setup, agent architectures, ablation configurations, and how to analyze results.

## Overview

We have two agent architectures for interacting with HMAS (Hypermedia Multi-Agent Systems) environments:

| Agent | File | Approach | Planning |
|-------|------|----------|----------|
| **Direct Agent** | `agent.py` | Tool calls → Direct HTTP | None (reactive) |
| **BT Agent** | `bt_agent.py` | Tool call → JSON spec → py_trees → Execute | Behavior tree generation |

## Architecture Comparison

### Direct Agent (`agent.py`)

```
User Goal
    │
    ▼
┌─────────────────┐
│  LLM decides    │ ◄─── Tools: explore_workspace, inspect_artifact,
│  next action    │       read_property, invoke_action
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Execute HTTP   │ ◄─── Direct call via hmas_client
│  request        │
└─────────────────┘
    │
    ▼
  Repeat until done
```

**Characteristics:**
- Reactive: One action at a time
- No parallel execution
- No precondition checking
- No retry logic
- Discovery happens during execution

### BT Agent (`bt_agent.py`)

```
User Goal
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1: BUILD CAPABILITY MODEL        │
│  - Explore all workspaces upfront       │
│  - Discover all artifacts & affordances │
│  - Cache for prompt injection           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 2: LLM PLANNING                  │
│  - Receive capability model in prompt   │
│  - Generate behavior tree JSON spec     │
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
- Proactive: Full plan before execution
- Parallel execution via Parallel nodes
- Precondition checking via Condition nodes
- Idempotent patterns via Selector nodes
- All capabilities known upfront

## Ablation Configurations

### 1. Prompting Strategies (`--strategy`)

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| `baseline` | Minimal prompting | Tests raw LLM capability |
| `detailed` | Full node documentation | Better structured output |
| `few_shot` | 4 example patterns | Pattern matching helps |
| `icl` | Reasoning traces | Step-by-step improves planning |
| `icl_verbose` | Detailed methodology | Most guidance, highest token cost |

**Run different strategies:**
```bash
uv run python bt_agent.py --strategy baseline "Turn on light"
uv run python bt_agent.py --strategy few_shot "Turn on light"
uv run python bt_agent.py --strategy icl "Turn on light"
```

### 2. Capability Discovery Modes (`--discovery`)

| Mode | Description | Trade-off |
|------|-------------|-----------|
| `exhaustive` | Explore all workspaces/artifacts upfront | Complete info, many HTTP calls |
| `agentic` | LLM-guided exploration based on goal | Fewer calls, goal-focused |
| `relevant` | Exhaustive + filter to goal-relevant | Complete discovery, focused prompt |

**Run different discovery modes:**
```bash
# Default: explore everything
uv run python bt_agent.py --discovery exhaustive "Turn on bathroom light"

# LLM decides what to explore (goal-aware)
uv run python bt_agent.py --discovery agentic "Turn on bathroom light"

# Explore all, but filter prompt to relevant devices
uv run python bt_agent.py --discovery relevant "Turn on bathroom light"
```

**Agentic Discovery Example:**
```
Building capability model (agentic for: 'Turn on the bathroom light')...
  Exploring home0...
  Exploring bathroom...
    Inspecting bathroomLight...
  Discovery complete: Found turnOn action for bathroomLight
```

**Relevant Filtering Example:**
```
Building capability model...
  Found 11 artifacts with 29 actions
  Filtered: 11 → 1 artifacts
```

### 3. Model Comparison (`--model`)

```bash
uv run python bt_agent.py --model gpt-4o "Turn on light"
uv run python bt_agent.py --model gpt-4o-mini "Turn on light"
uv run python bt_agent.py --base-url http://localhost:11434/v1 --model llama3.2 "Turn on light"
```

## Trace Analysis

### What's Captured

Each trace (`traces/trace_*.json`) contains:

```json
{
  "trace_id": "20251222_233328_787257",
  "goal": "Turn on the bathroom light",
  "model": "gpt-4o",
  "ablation_config": {"strategy": "detailed", "discovery": "exhaustive"},

  "events": [
    {
      "event_type": "capability_model_summary",
      "data": {
        "discovered_capabilities": {
          "workspaces": {
            "bathroom": [
              {
                "name": "Bathroomlight",
                "actions": [{"name": "turnOn", "uri": "..."}],
                "properties": [{"name": "state", "uri": "..."}]
              }
            ]
          }
        },
        "prompt_representation": "# Available Devices..."
      }
    },
    {
      "event_type": "llm_prompt_content",
      "data": {
        "system_prompt": "...",
        "user_message": "Turn on the bathroom light",
        "tool_description": "..."
      }
    },
    {
      "event_type": "bt_spec_generated",
      "data": {
        "tree_spec": {...},
        "explanation": "..."
      }
    },
    {
      "event_type": "bt_tick",
      "data": {"tick": 1, "status": "SUCCESS"}
    }
  ],

  "summary": {
    "total_events": 10,
    "total_time_ms": 3111.73
  }
}
```

### Key Analysis Points

1. **Capability Discovery**: Compare `discovered_capabilities` with what the LLM actually used
2. **Prompt Length**: Check `system_prompt_length` across strategies
3. **Tree Complexity**: Analyze `tree_spec` structure (depth, node count)
4. **Execution Time**: Compare `total_time_ms` across configurations
5. **Success Rate**: Check `success` field across runs

### Analyzing Traces

```python
import json
from pathlib import Path

def load_traces(trace_dir="traces"):
    traces = []
    for f in Path(trace_dir).glob("trace_*.json"):
        with open(f) as fp:
            traces.append(json.load(fp))
    return traces

def analyze_strategy_performance(traces):
    """Group traces by strategy and compute stats."""
    by_strategy = {}
    for t in traces:
        strategy = t["ablation_config"].get("strategy", "unknown")
        if strategy not in by_strategy:
            by_strategy[strategy] = {"success": 0, "total": 0, "times": []}
        by_strategy[strategy]["total"] += 1
        if t["success"]:
            by_strategy[strategy]["success"] += 1
        by_strategy[strategy]["times"].append(t["summary"]["total_time_ms"])

    for s, data in by_strategy.items():
        avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
        print(f"{s}: {data['success']}/{data['total']} success, {avg_time:.0f}ms avg")

def get_capability_usage(trace):
    """Check which discovered capabilities were actually used."""
    # Get discovered
    cap_event = next((e for e in trace["events"]
                      if e["event_type"] == "capability_model_summary"), None)
    if not cap_event:
        return None

    discovered = set()
    for ws, artifacts in cap_event["data"]["discovered_capabilities"]["workspaces"].items():
        for art in artifacts:
            for action in art["actions"]:
                discovered.add(action["uri"])

    # Get used
    bt_event = next((e for e in trace["events"]
                     if e["event_type"] == "bt_spec_generated"), None)
    if not bt_event:
        return None

    def extract_uris(node):
        uris = []
        if "action_url" in node:
            uris.append(node["action_url"])
        if "property_url" in node:
            uris.append(node["property_url"])
        for child in node.get("children", []):
            uris.extend(extract_uris(child))
        return uris

    used = set(extract_uris(bt_event["data"]["tree_spec"]))

    return {
        "discovered": len(discovered),
        "used": len(used),
        "usage_ratio": len(used) / len(discovered) if discovered else 0
    }
```

## Running Experiments

### Basic Experiment

```bash
# Run with tracing
uv run python bt_agent.py --trace --strategy detailed "Turn on bathroom light"
uv run python bt_agent.py --trace --strategy few_shot "Turn on bathroom light"
uv run python bt_agent.py --trace --strategy icl "Turn on bathroom light"

# Check traces
ls -la traces/
```

### Batch Experiment

```bash
#!/bin/bash
GOALS=(
    "Turn on the bathroom light"
    "Turn on all lights in the corridor"
    "Set the AC temperature to 22 degrees"
    "Turn on lights and set brightness to 50"
)

STRATEGIES=(baseline detailed few_shot icl icl_verbose)

for goal in "${GOALS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "Running: $strategy - $goal"
        uv run python bt_agent.py --trace --strategy "$strategy" "$goal"
    done
done
```

### Compare with Direct Agent

```bash
# BT Agent (planned execution)
uv run python bt_agent.py --trace "Turn on bathroom light"

# Direct Agent (reactive execution)
uv run python agent.py "Turn on bathroom light"
```

## Key Findings

### What Works Well

1. **`detailed` strategy**: Most reliable for gpt-4o, good balance of guidance and flexibility
2. **Selector pattern**: LLM naturally generates idempotent check-then-act patterns
3. **Parallel execution**: LLM correctly uses parallel nodes for independent operations
4. **Upfront discovery**: Having all capabilities in context helps accurate URI usage

### Challenges

1. **`few_shot`/`icl` with smaller models**: gpt-4o-mini sometimes fails to generate proper tree structure
2. **Long prompts**: Full capability model can be verbose; may need filtering for large environments
3. **Complex goals**: Multi-step goals sometimes generate overly nested trees

### Trade-offs

| Dimension | Direct Agent | BT Agent |
|-----------|--------------|----------|
| Latency to first action | Low | High (builds model first) |
| Parallel execution | No | Yes |
| Error recovery | Manual | Built into tree |
| Token usage | Low per call, many calls | High single call |
| Debuggability | Hard (reactive) | Easy (inspect tree spec) |

## File Structure

```
├── agent.py              # Direct agent (baseline)
├── bt_agent.py           # BT planning agent (main)
├── tracing.py            # Trace capture system
├── prompts.py            # Ablation prompt strategies
├── hmas_client.py        # HTTP/RDF client
├── behavior_trees/       # py_trees node implementations
├── traces/               # Saved experiment traces
└── README_EXPERIMENTS.md # This file
```

## Next Steps

1. **Selective capability injection**: Only include goal-relevant devices in prompt
2. **Multi-turn planning**: Let LLM ask for more details before generating tree
3. **Failure recovery**: Re-plan on execution failure
4. **Hierarchical goals**: Decompose complex goals into subtrees
