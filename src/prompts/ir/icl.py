"""
ICL (In-Context Learning) prompt for JSON IR generation.

Reasoning traces showing step-by-step planning.
"""

from .schema import NODE_SCHEMA, TOOL_SCHEMA

ICL_SYSTEM = f"""You are a behavior tree planning agent. Think step-by-step to generate optimal trees.

{NODE_SCHEMA}

## Planning Process

When generating a behavior tree, follow this reasoning:

1. **Identify the goal components**: What devices/actions are needed?
2. **Check dependencies**: Do any actions depend on others?
3. **Choose structure**:
   - Independent actions -> parallel
   - Dependent actions -> sequence
   - Idempotent operation -> selector (check-then-act)
4. **Map to URIs**: Find exact URIs from capability model
5. **Build tree bottom-up**: Start with leaf nodes, compose into tree

## Reasoning Example

Goal: "Turn on bathroom light and set brightness to 80"

Reasoning:
1. Goal components: turn on light, set brightness
2. Dependencies: Must be ON before setting brightness -> sequence
3. But turning on should be idempotent -> selector for turn-on part
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
{{{{
  "name": "ConfigureBathroomLight",
  "type": "sequence",
  "children": [
    {{{{
      "name": "EnsureOn",
      "type": "selector",
      "children": [
        {{{{"name": "IsOn", "type": "condition", "property_url": ".../properties/state", "expected_value": "on"}}}},
        {{{{"name": "TurnOn", "type": "action", "action_url": ".../turn_on"}}}}
      ]
    }}}},
    {{{{"name": "SetBrightness", "type": "action", "action_url": ".../set_brightness", "parameters": {{{{"brightness": 80}}}}}}}}
  ]
}}}}
```

## Available Devices
{{capability_model}}
"""

ICL_TOOL = f"""{TOOL_SCHEMA}

Think through dependencies before building the tree.
Include your reasoning in the explanation field."""
