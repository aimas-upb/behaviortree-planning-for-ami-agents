"""
Execution module for running behavior trees.

Supports multiple execution modes:
- ir_executor: Compile JSON IR to py_trees and execute
- code_executor: Execute generated Python code directly
- direct_agent: LLM agent with tool calls (no code generation)
"""

from .base import ExecutionResult
from .ir_executor import IRExecutor
from .code_executor import CodeExecutor
from .direct_agent import DirectAgentExecutor, DirectAgentResult

__all__ = [
    "ExecutionResult",
    "IRExecutor",
    "CodeExecutor",
    "DirectAgentExecutor",
    "DirectAgentResult",
]


def create_executor(output_format: str, max_ticks: int = 10):
    """Factory function to create an executor.

    Args:
        output_format: One of 'json_ir', 'python_code', or 'python_code_unconstrained'
        max_ticks: Maximum behavior tree ticks before timeout

    Returns:
        Appropriate executor for the output format
    """
    if output_format == "json_ir":
        return IRExecutor(max_ticks=max_ticks)
    elif output_format == "python_code":
        return CodeExecutor(max_ticks=max_ticks)
    elif output_format == "python_code_unconstrained":
        return CodeExecutor(max_ticks=max_ticks, unconstrained=True)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
