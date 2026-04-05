"""
Prompt registry for accessing prompts by strategy and format.
"""

import logging
from typing import Tuple

from . import code as code_prompts
from . import ir as ir_prompts

logger = logging.getLogger(__name__)


def get_prompt(strategy: str, output_format: str) -> Tuple[str, str]:
    """
    Get prompt template for the given strategy and output format.

    Args:
        strategy: Prompt strategy (baseline, detailed, few_shot, icl)
        output_format: Output format (json_ir, python_code, python_code_unconstrained)

    Returns:
        Tuple of (system_prompt, tool_description)
    """
    if output_format == "json_ir":
        prompts = ir_prompts.PROMPTS
    elif output_format == "python_code":
        prompts = code_prompts.PROMPTS
    elif output_format == "python_code_unconstrained":
        prompts = code_prompts.UNCONSTRAINED_PROMPTS
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    if strategy not in prompts:
        logger.warning(
            f"Strategy '{strategy}' not found for {output_format}, falling back to 'detailed'"
        )
        strategy = "detailed"

    return prompts[strategy]


def list_strategies() -> list[str]:
    """List available prompt strategies."""
    return list(ir_prompts.PROMPTS.keys())


def get_strategy_descriptions() -> dict[str, str]:
    """Get descriptions of available strategies."""
    return {
        "baseline": "Minimal prompting - tests raw LLM capability",
        "detailed": "Comprehensive node type documentation",
        "few_shot": "Example-based learning with diverse patterns",
        "icl": "In-context learning with reasoning traces",
    }
