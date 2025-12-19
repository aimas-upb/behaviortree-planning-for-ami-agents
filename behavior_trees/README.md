# Behavior Tree Templates for HMAS Affordances

This module provides **py-trees** based behavior tree node templates.

## Overview

The templates enable you to:

1. **Execute Actions** - Invoke action affordances (e.g., turn on lights, stack blocks) via HTTP
2. **Read Properties** - Query property affordances (e.g., device state, temperature)
3. **Check Conditions** - Evaluate property values against expected values
4. **Build Complex Trees** - Compose nodes into sophisticated behavior trees

## Installation

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Basic Action Node

```python
from behavior_trees import ActionAffordanceNode

# Create an action node to turn on a light
turn_on = ActionAffordanceNode(
    name="TurnOnLight",
    action_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/turn_on"
)

# Create an action with parameters
set_brightness = ActionAffordanceNode(
    name="SetBrightness",
    action_url="http://localhost:8080/workspaces/home0/corridor/artifacts/corridorLight/set_brightness",
    parameters={"brightness": 75}
)
```

### Property Condition Node

```python
from behavior_trees import PropertyConditionNode

# Check if a light is on
is_light_on = PropertyConditionNode(
    name="IsLightOn",
    property_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/properties/state",
    expected_value="on"
)
```

### Building a Behavior Tree

```python
import py_trees
from behavior_trees import ActionAffordanceNode, PropertyConditionNode

# Create nodes
is_on = PropertyConditionNode(
    name="IsLightOn",
    property_url="http://localhost:8080/artifacts/light/properties/state",
    expected_value="on"
)

turn_on = ActionAffordanceNode(
    name="TurnOnLight",
    action_url="http://localhost:8080/artifacts/light/turn_on"
)

# Build tree: If light is not on, turn it on
ensure_light_on = py_trees.composites.Selector(
    name="EnsureLightOn",
    memory=False,
    children=[is_on, turn_on]
)

# Run the tree
ensure_light_on.setup_with_descendants()
ensure_light_on.tick_once()
```

## Node Types

### ActionAffordanceNode

Executes action affordances via HTTP POST requests.

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Node name for the behavior tree |
| `action_url` | str | HTTP endpoint URL for the action |
| `parameters` | Dict[str, Any] | Static parameters to send with the action |
| `parameter_keys` | Dict[str, str] | Blackboard keys for dynamic parameters |
| `store_result` | bool | Whether to store results on blackboard (default: True) |
| `result_key` | str | Custom blackboard key for the result |

**Return Status:**
- `SUCCESS` - Action completed successfully (HTTP 2xx)
- `FAILURE` - Action failed (HTTP error or exception)

**Example with all features:**

```python
from behavior_trees import ActionAffordanceNode

# Action with dynamic parameters from blackboard
action = ActionAffordanceNode(
    name="SetColor",
    action_url="http://localhost:8080/artifacts/light/set_color",
    parameters={"brightness": 100},  # Static param
    parameter_keys={"color": "user/selected_color"},  # Dynamic param from blackboard
)
```

### PropertyAffordanceNode

Reads property affordances via HTTP GET requests.

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Node name for the behavior tree |
| `property_url` | str | HTTP endpoint URL for the property |
| `store_result` | bool | Whether to store value on blackboard (default: True) |
| `result_key` | str | Custom blackboard key for the value |
| `property_name` | str | Name of the property (auto-extracted if not provided) |

**Return Status:**
- `SUCCESS` - Property read successfully
- `FAILURE` - Read failed

### PropertyConditionNode

Checks if a property value matches an expected value.

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Node name for the behavior tree |
| `property_url` | str | HTTP endpoint URL for the property |
| `expected_value` | Any | Value to compare against |
| `expected_value_key` | str | Blackboard key for dynamic expected value |
| `value_path` | List[str] | Path to navigate nested response objects |
| `negate` | bool | If True, succeed when values DON'T match |

**Return Status:**
- `SUCCESS` - Comparison matches (or doesn't match if `negate=True`)
- `FAILURE` - Comparison fails or property cannot be read

**Example with nested values (Blocksworld):**

```python
# Check if hand is empty in blocksworld
# The state response is: {"hand": "empty", "blocks": [...]}
hand_empty = PropertyConditionNode(
    name="IsHandEmpty",
    property_url="http://localhost:8080/workspaces/blocksworld/artifacts/world-1/properties/state",
    expected_value="empty",
    value_path=["hand"]  # Navigate to state.hand
)
```

### ComparisonPropertyConditionNode

Extended condition node supporting various comparison operators.

**Additional Parameter:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | ComparisonOperator | The comparison operator to use |

**Available Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `EQUAL` | Equal to (==) | `value == expected` |
| `NOT_EQUAL` | Not equal to (!=) | `value != expected` |
| `GREATER_THAN` | Greater than (>) | `value > expected` |
| `GREATER_THAN_OR_EQUAL` | Greater than or equal (>=) | `value >= expected` |
| `LESS_THAN` | Less than (<) | `value < expected` |
| `LESS_THAN_OR_EQUAL` | Less than or equal (<=) | `value <= expected` |
| `IN` | Value in collection | `value in [a, b, c]` |
| `NOT_IN` | Value not in collection | `value not in [a, b]` |
| `CONTAINS` | Contains element/substring | `"hello" in value` |
| `MATCHES` | Regex match | `re.match(pattern, value)` |

**Example:**

```python
from behavior_trees import ComparisonPropertyConditionNode, ComparisonOperator

# Check if temperature is above threshold
is_hot = ComparisonPropertyConditionNode(
    name="IsHot",
    property_url="http://localhost:8080/artifacts/ac/properties/temperature",
    expected_value=25,
    operator=ComparisonOperator.GREATER_THAN
)

# Check if mode is one of several values
valid_mode = ComparisonPropertyConditionNode(
    name="IsValidMode",
    property_url="http://localhost:8080/artifacts/ac/properties/mode",
    expected_value=["cool", "auto", "fan_only"],
    operator=ComparisonOperator.IN
)
```

## Blackboard Integration

Nodes store results on the py-trees blackboard for sharing data.

```python
from behavior_trees import BlackboardKeys
import py_trees

# Standard blackboard keys
BlackboardKeys.LAST_ACTION_RESULT      # Last action result
BlackboardKeys.LAST_PROPERTY_VALUE     # Last property value
BlackboardKeys.LAST_ACTION_ERROR       # Last action error message

# Property-specific keys
key = BlackboardKeys.property_value_key("brightness")  # "affordance/properties/brightness"

# Artifact-specific keys
key = BlackboardKeys.artifact_property_key("light1", "state")  # "affordance/artifacts/light1/state"

# Access blackboard in custom code
blackboard = py_trees.blackboard.Client()
blackboard.register_key(key=BlackboardKeys.LAST_ACTION_RESULT, access=py_trees.common.Access.READ)
result = blackboard.get(BlackboardKeys.LAST_ACTION_RESULT)
```

## Complete Examples

### HomeBench: Smart Home Room Setup

```python
import py_trees
from behavior_trees import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator
)

def create_room_comfort_tree(room: str = "living_room"):
    """Set up a room for comfort: lights on, AC at comfortable temp."""
    
    base = f"http://localhost:8080/workspaces/home0/{room}/artifacts"
    
    # Light control
    is_light_on = PropertyConditionNode(
        name="IsLightOn",
        property_url=f"{base}/{room}Light/properties/state",
        expected_value="on"
    )
    
    turn_on_light = ActionAffordanceNode(
        name="TurnOnLight",
        action_url=f"{base}/{room}Light/turn_on"
    )
    
    ensure_light = py_trees.composites.Selector(
        name="EnsureLightOn",
        children=[is_light_on, turn_on_light]
    )
    
    # Temperature control
    is_comfortable = ComparisonPropertyConditionNode(
        name="IsComfortable",
        property_url=f"{base}/{room}AirConditioner/properties/temperature",
        expected_value=24,
        operator=ComparisonOperator.LESS_THAN_OR_EQUAL
    )
    
    set_temperature = ActionAffordanceNode(
        name="SetComfortTemp",
        action_url=f"{base}/{room}AirConditioner/set_temperature",
        parameters={"temperature": 22}
    )
    
    ensure_temp = py_trees.composites.Selector(
        name="EnsureComfortTemp",
        children=[is_comfortable, set_temperature]
    )
    
    # Run both in parallel
    room_setup = py_trees.composites.Parallel(
        name="RoomSetup",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
        children=[ensure_light, ensure_temp]
    )
    
    return room_setup
```

### Blocksworld: Stack Block Sequence

```python
import py_trees
from behavior_trees import ActionAffordanceNode, PropertyConditionNode

def create_move_block_tree(
    world_id: str,
    block: str,
    target: str
):
    """Move a block from wherever it is onto another block."""
    
    base = f"http://localhost:8080/workspaces/blocksworld/artifacts/{world_id}"
    
    # Pre-condition: Hand must be empty
    hand_empty = PropertyConditionNode(
        name="IsHandEmpty",
        property_url=f"{base}/properties/state",
        expected_value="empty",
        value_path=["hand"]
    )
    
    # If block is on table, pick it up
    pickup = ActionAffordanceNode(
        name=f"Pickup_{block}",
        action_url=f"{base}/pickup",
        parameters={"target_block": block}
    )
    
    # Stack onto target
    stack = ActionAffordanceNode(
        name=f"Stack_{block}_on_{target}",
        action_url=f"{base}/stack",
        parameters={"target_block": block, "to_block": target}
    )
    
    # Full sequence
    move_sequence = py_trees.composites.Sequence(
        name=f"Move_{block}_to_{target}",
        memory=True,
        children=[hand_empty, pickup, stack]
    )
    
    return move_sequence
```

## Running with Simulators

Before running behavior trees, start the appropriate simulator:

### HomeBench Simulator

```bash
cd homebench
python smart_home_simulator.py --state-file ../datasets/HomeBench/hmas_format/home_0_state.json
```

### Blocksworld Simulator

```bash
cd blocksworld
python blocksworld_simulator.py --data-dir ../datasets/Blocksworld/hmas_format/generated_basic --port 8081
```

## Architecture

```
behavior_trees/
├── __init__.py              # Package exports
├── affordance_nodes.py      # Core behavior tree nodes
│   ├── ActionAffordanceNode
│   ├── PropertyAffordanceNode
│   ├── PropertyConditionNode
│   └── ComparisonPropertyConditionNode
├── http_client.py           # httpx-based HTTP communication
│   ├── HTTPClient
│   ├── HTTPClientConfig
│   └── HTTPResponse
├── blackboard_keys.py       # Standardized blackboard keys
└── examples/
    ├── __init__.py
    └── basic_usage.py       # Example implementations
```
