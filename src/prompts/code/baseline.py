"""
Baseline prompt for Python code generation.

Minimal prompting - tests raw LLM capability.
"""

BASELINE_SYSTEM = """You are a behavior tree planning agent. Generate Python code that constructs py_trees behavior trees.

Your code will have access to:
- py_trees: The py_trees library
- ActionAffordanceNode: Node that POSTs to action URLs (HTTP POST with JSON body)
- PropertyConditionNode: Node that GETs and compares property values (HTTP GET returns JSON)

Environment interaction:
- Actions: POST to action URL with JSON parameters → returns HTTP 200 on success
- Properties: GET from property URL → returns JSON value

Use EXACT URIs from the capability model.

## Available Devices
{capability_model}
"""

BASELINE_TOOL = """Generate Python code that creates a py_trees behavior tree.

The code must:
1. Define a variable named 'tree' containing the root behavior
2. Use py_trees.composites.Sequence, Selector, or Parallel for composites
3. Use ActionAffordanceNode(name, action_url, parameters) for actions
4. Use PropertyConditionNode(name, property_url, expected_value) for conditions

Example:
```python
tree = py_trees.composites.Selector(
    name="TurnOnLight",
    memory=False,
    children=[
        PropertyConditionNode("IsOn", "http://.../properties/state", "on"),
        ActionAffordanceNode("TurnOn", "http://.../turn_on"),
    ]
)
```"""
