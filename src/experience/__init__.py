"""
Experience-based planning module.

Provides experience accumulation and reuse across sequential test runs:
  1. Intent Extraction — parse user goals into structured intents
  2. Experience Matching — match intents to stored experiences
  3. Experience Adaptation — adapt matched experiences to new contexts
  4. BT Serialization — convert between py_trees and JSON-IR
  5. Engine — persistent storage for experiences
  6. Runner — orchestrate the full experience pipeline
  7. NeuroSymbolicRunner — neuro-symbolic planning without full discovery
"""

from .adaptation import ExperienceAdapter
from .bt_serialization import (
    combine_trees_parallel,
    extract_action_urls,
    extract_leaf_nodes_json_ir,
    extract_property_urls,
    py_tree_to_json_ir,
)
from .engine import ExperienceEngine, ExperienceEntry
from .intent import IntentExtractor, StructuredIntent
from .matching import ExperienceMatcher, MatchResult
from .neurosymbolic_runner import (
    FEPQwenNeuroSymbolicRunner,
    NeuroSymbolicRunner,
    NeuroSymbolicRunResult,
    QwenModifyCodegenNeuroSymbolicRunner,
    SparqlResolutionResult,
)

__all__ = [
    # Intent extraction
    "StructuredIntent",
    "IntentExtractor",
    # Engine
    "ExperienceEntry",
    "ExperienceEngine",
    # Matching
    "MatchResult",
    "ExperienceMatcher",
    # Adaptation
    "ExperienceAdapter",
    # BT serialization
    "py_tree_to_json_ir",
    "extract_leaf_nodes_json_ir",
    "extract_action_urls",
    "extract_property_urls",
    "combine_trees_parallel",
    # Neuro-symbolic runner
    "FEPQwenNeuroSymbolicRunner",
    "NeuroSymbolicRunner",
    "NeuroSymbolicRunResult",
    "QwenModifyCodegenNeuroSymbolicRunner",
    "SparqlResolutionResult",
]
