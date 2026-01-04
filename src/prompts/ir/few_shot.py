"""
Few-shot prompt for JSON IR generation.

Example-based learning with diverse patterns.
"""

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
