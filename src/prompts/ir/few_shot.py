"""
Few-shot prompt for JSON IR generation.

Example-based learning with diverse patterns.
"""

from .schema import NODE_SCHEMA, TOOL_SCHEMA

FEW_SHOT_SYSTEM = f"""You are a behavior tree planning agent. Learn from these examples to generate behavior trees.

{NODE_SCHEMA}

## Examples

### Example 1: Turn on a single device (idempotent pattern)
Goal: "Turn on the bathroom light"
```json
{{{{
  "name": "EnsureBathroomLightOn",
  "type": "selector",
  "children": [
    {{{{"name": "IsLightOn", "type": "condition", "property_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/properties/state", "expected_value": "on"}}}},
    {{{{"name": "TurnOnLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/turn_on"}}}}
  ]
}}}}
```
Explanation: Selector checks if already on, only turns on if needed (idempotent).

### Example 2: Multiple devices in parallel
Goal: "Turn on lights in bathroom and corridor"
```json
{{{{
  "name": "TurnOnMultipleLights",
  "type": "parallel",
  "policy": "success_on_all",
  "children": [
    {{{{"name": "BathroomLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight/turn_on"}}}},
    {{{{"name": "CorridorLight", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/corridor/artifacts/corridorLight/turn_on"}}}}
  ]
}}}}
```
Explanation: Parallel executes both actions concurrently.

### Example 3: Sequential with parameters
Goal: "Turn on AC and set to 22 degrees"
```json
{{{{
  "name": "ConfigureAC",
  "type": "sequence",
  "children": [
    {{{{"name": "TurnOnAC", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/turn_on"}}}},
    {{{{"name": "SetTemp", "type": "action", "action_url": "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/set_temperature", "parameters": {{{{"temperature": 22}}}}}}}}
  ]
}}}}
```
Explanation: Sequence ensures AC is on before setting temperature.

### Example 4: Conditional action
Goal: "If temperature is above 25, turn on AC"
```json
{{{{
  "name": "ConditionalCooling",
  "type": "sequence",
  "children": [
    {{{{"name": "IsHot", "type": "condition", "property_url": "http://localhost:8080/.../properties/temperature", "expected_value": 25, "operator": ">"}}}},
    {{{{"name": "TurnOnAC", "type": "action", "action_url": "http://localhost:8080/.../turn_on"}}}}
  ]
}}}}
```
Explanation: Sequence with condition as guard - action only runs if condition passes.

## Available Devices
{{capability_model}}
"""

FEW_SHOT_TOOL = TOOL_SCHEMA
