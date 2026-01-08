"""
Prompts for JSON IR generation.
"""

from .baseline import BASELINE_SYSTEM, BASELINE_TOOL
from .detailed import DETAILED_SYSTEM, DETAILED_TOOL
from .few_shot import FEW_SHOT_SYSTEM, FEW_SHOT_TOOL
from .icl import ICL_SYSTEM, ICL_TOOL

PROMPTS = {
    "baseline": (BASELINE_SYSTEM, BASELINE_TOOL),
    "detailed": (DETAILED_SYSTEM, DETAILED_TOOL),
    "few_shot": (FEW_SHOT_SYSTEM, FEW_SHOT_TOOL),
    "icl": (ICL_SYSTEM, ICL_TOOL),
}
