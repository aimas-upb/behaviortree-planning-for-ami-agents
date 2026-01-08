"""
Shared schema documentation for JSON IR behavior trees.

All IR prompts should include this to ensure valid compilable output.
"""

# Node type documentation - required for all prompts
NODE_SCHEMA = """## Node Types

The behavior tree compiler supports exactly 5 node types:

### Composite Nodes (have 'children' array)

1. **sequence**: Execute children left-to-right.
   - Returns SUCCESS if ALL children succeed
   - Returns FAILURE on first child failure (stops execution)
   - Use for: ordered steps that must all complete

2. **selector**: Try children left-to-right.
   - Returns SUCCESS on first child success (stops trying)
   - Returns FAILURE only if ALL children fail
   - Use for: fallback patterns, idempotent "check-then-act"

3. **parallel**: Execute all children simultaneously.
   - Required: "children" array
   - Optional: "policy" - "success_on_all" (default) or "success_on_one"
   - Use for: independent concurrent operations

### Leaf Nodes (no children)

4. **action**: Execute HTTP POST to action_url.
   - Required: "action_url" (string) - exact URI from capability model
   - Optional: "parameters" (object) - JSON body for POST request
   - Returns SUCCESS on HTTP 2xx, FAILURE otherwise

5. **condition**: Check property value via HTTP GET.
   - Required: "property_url" (string) - exact URI from capability model
   - Required: "expected_value" (any) - value to compare against
   - Optional: "operator" (string) - "==", "!=", ">", "<", ">=", "<=" (default: "==")
   - Optional: "value_path" (string) - JSON path for nested values
   - Returns SUCCESS if value matches, FAILURE otherwise

## Schema Reference

```json
// Composite node
{{
  "name": "NodeName",
  "type": "sequence" | "selector" | "parallel",
  "children": [...],
  "policy": "success_on_all" | "success_on_one"  // parallel only
}}

// Action node
{{
  "name": "ActionName",
  "type": "action",
  "action_url": "http://...",
  "parameters": {{}}  // optional
}}

// Condition node
{{
  "name": "ConditionName",
  "type": "condition",
  "property_url": "http://.../properties/...",
  "expected_value": "on" | 25 | true,
  "operator": "=="  // optional
}}
```

## Rules
1. Every node MUST have "name" (string) and "type" (string)
2. Use EXACT URIs from the capability model - do not modify them
3. Composite nodes MUST have "children" array (can be empty)
4. Action/condition nodes MUST NOT have "children"
"""

# Tool description for generate_behavior_tree function
TOOL_SCHEMA = """Generate an executable behavior tree specification.

The tree will be compiled to py_trees and executed via HTTP calls.

IMPORTANT - Valid node types:
- sequence: {{"type": "sequence", "children": [...]}} - all must succeed
- selector: {{"type": "selector", "children": [...]}} - first success wins
- parallel: {{"type": "parallel", "children": [...], "policy": "success_on_all"}}
- action: {{"type": "action", "action_url": "http://...", "parameters": {{}}}}
- condition: {{"type": "condition", "property_url": "http://...", "expected_value": ...}}

Every node requires "name" and "type" fields.
Use EXACT URIs from the capability model."""
