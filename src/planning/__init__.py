"""
Planning module for behavior tree generation.

Supports multiple reasoning strategies and output formats:

Reasoning strategies:
- none: Direct generation without explicit reasoning
- chain_of_thought: Single-turn reasoning before generation
- multi_turn: Multi-turn conversation for exploration
- reflection: Generate, critique, and refine

Output formats:
- json_ir: JSON intermediate representation (compiled to py_trees)
- python_code: Direct Python py_trees code (exec'd)
"""

from .base import Plan, PlanningResult
from .planner import Planner, create_planner

__all__ = [
    "Plan",
    "PlanningResult",
    "Planner",
    "create_planner",
]
