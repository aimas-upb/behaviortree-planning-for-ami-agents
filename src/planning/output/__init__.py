"""
Output generators for planning.

Formats:
- json_ir: JSON intermediate representation (compiled to py_trees)
- python_code: Direct Python py_trees code (constrained to template nodes)
- python_code_unconstrained: Python py_trees code with custom behaviors
"""

from .json_ir import JsonIRGenerator
from .python_code import PythonCodeGenerator, UnconstrainedPythonCodeGenerator

__all__ = [
    "JsonIRGenerator",
    "PythonCodeGenerator",
    "UnconstrainedPythonCodeGenerator",
]


def create_output_generator(format: str):
    """Factory function to create output generator."""
    if format == "json_ir":
        return JsonIRGenerator()
    elif format == "python_code":
        return PythonCodeGenerator()
    elif format == "python_code_unconstrained":
        return UnconstrainedPythonCodeGenerator()
    else:
        raise ValueError(f"Unknown output format: {format}")
