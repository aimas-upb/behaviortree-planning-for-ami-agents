"""
Prompt templates for behavior tree generation.

Supports multiple prompt strategies and output formats.

Prompt strategies:
- baseline: Minimal prompting
- detailed: Comprehensive documentation
- few_shot: Example-based learning
- icl: In-context learning with reasoning

Output formats:
- json_ir: JSON intermediate representation
- python_code: Direct Python py_trees code
"""

from .registry import get_prompt, list_strategies, get_strategy_descriptions

__all__ = [
    "get_prompt",
    "list_strategies",
    "get_strategy_descriptions",
]
