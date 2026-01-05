"""
Baseline prompt for JSON IR generation.

Minimal prompting - tests raw LLM capability with required schema.
"""

from .schema import NODE_SCHEMA, TOOL_SCHEMA

BASELINE_SYSTEM = f"""You are a behavior tree planning agent. Generate behavior tree specifications for the given goal.

{NODE_SCHEMA}

## Available Devices
{{capability_model}}
"""

BASELINE_TOOL = TOOL_SCHEMA
