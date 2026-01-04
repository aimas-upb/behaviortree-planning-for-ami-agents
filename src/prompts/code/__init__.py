"""
Prompts for Python code generation.
"""

from .baseline import BASELINE_SYSTEM, BASELINE_TOOL
from .detailed import DETAILED_SYSTEM, DETAILED_TOOL
from .unconstrained import UNCONSTRAINED_SYSTEM, UNCONSTRAINED_TOOL

PROMPTS = {
    "baseline": (BASELINE_SYSTEM, BASELINE_TOOL),
    "detailed": (DETAILED_SYSTEM, DETAILED_TOOL),
    # few_shot and icl fall back to detailed for code generation
    "few_shot": (DETAILED_SYSTEM, DETAILED_TOOL),
    "icl": (DETAILED_SYSTEM, DETAILED_TOOL),
}

# Separate prompts for unconstrained mode (allows custom py_trees behaviors)
UNCONSTRAINED_PROMPTS = {
    "baseline": (UNCONSTRAINED_SYSTEM, UNCONSTRAINED_TOOL),
    "detailed": (UNCONSTRAINED_SYSTEM, UNCONSTRAINED_TOOL),
    "few_shot": (UNCONSTRAINED_SYSTEM, UNCONSTRAINED_TOOL),
    "icl": (UNCONSTRAINED_SYSTEM, UNCONSTRAINED_TOOL),
}
