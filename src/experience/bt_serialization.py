"""
Behavior Tree JSON-IR Serialization Utilities.

Provides deterministic conversion between py_trees objects and a JSON
intermediate representation (JSON-IR). The JSON-IR format is compatible
with the IRExecutor and is used by the experience engine to store
behavior tree structures.

Uses an external visitor pattern — no modifications to the existing
``behavior_trees.affordance_nodes`` classes are required.
"""

import logging
from typing import Any

import py_trees
from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# py_trees -> JSON-IR
# ---------------------------------------------------------------------------


def py_tree_to_json_ir(node: py_trees.behaviour.Behaviour) -> dict:
    """
    Convert a py_trees behaviour tree to JSON-IR format.

    Handles composite nodes (Sequence, Selector, Parallel) and leaf nodes
    (ActionAffordanceNode, PropertyConditionNode, ComparisonPropertyConditionNode,
    PropertyAffordanceNode).

    Returns:
        A JSON-serialisable dict representing the tree.
    """
    # --- Composite nodes ---
    if isinstance(node, py_trees.composites.Sequence):
        return {
            "type": "sequence",
            "name": node.name,
            "children": [py_tree_to_json_ir(c) for c in node.children],
        }

    if isinstance(node, py_trees.composites.Selector):
        return {
            "type": "selector",
            "name": node.name,
            "children": [py_tree_to_json_ir(c) for c in node.children],
        }

    if isinstance(node, py_trees.composites.Parallel):
        # Determine policy string
        policy = node.policy
        if isinstance(policy, py_trees.common.ParallelPolicy.SuccessOnAll):
            policy_str = "success_on_all"
        elif isinstance(policy, py_trees.common.ParallelPolicy.SuccessOnOne):
            policy_str = "success_on_one"
        else:
            policy_str = str(policy)

        return {
            "type": "parallel",
            "name": node.name,
            "policy": policy_str,
            "children": [py_tree_to_json_ir(c) for c in node.children],
        }

    # --- Leaf nodes (check most-specific subclass first) ---
    if isinstance(node, ComparisonPropertyConditionNode):
        ir: dict[str, Any] = {
            "type": "condition",
            "name": node.name,
            "property_url": node.property_url,
            "expected_value": node.expected_value,
            "operator": node.operator.value,
        }
        if node.value_path:
            ir["value_path"] = node.value_path
        if node.negate:
            ir["negate"] = True
        return ir

    if isinstance(node, PropertyConditionNode):
        ir = {
            "type": "condition",
            "name": node.name,
            "property_url": node.property_url,
            "expected_value": node.expected_value,
        }
        if node.value_path:
            ir["value_path"] = node.value_path
        if node.negate:
            ir["negate"] = True
        return ir

    if isinstance(node, ActionAffordanceNode):
        ir = {
            "type": "action",
            "name": node.name,
            "action_url": node.action_url,
        }
        if node.parameters:
            ir["parameters"] = dict(node.parameters)
        if node.parameter_keys:
            ir["parameter_keys"] = dict(node.parameter_keys)
        if node.result_key:
            ir["result_key"] = node.result_key
        return ir

    if isinstance(node, PropertyAffordanceNode):
        ir = {
            "type": "property_read",
            "name": node.name,
            "property_url": node.property_url,
        }
        if node.result_key:
            ir["result_key"] = node.result_key
        if node.property_name:
            ir["property_name"] = node.property_name
        return ir

    # --- Generic composite fallback ---
    if isinstance(node, py_trees.composites.Composite):
        return {
            "type": "composite",
            "name": node.name,
            "children": [py_tree_to_json_ir(c) for c in node.children],
        }

    # --- Unknown leaf ---
    logger.warning(
        f"Unknown node type: {type(node).__name__}, name={node.name}"
    )
    return {
        "type": "unknown",
        "name": node.name,
        "class": type(node).__name__,
    }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_leaf_nodes_json_ir(
    node: py_trees.behaviour.Behaviour,
) -> list[dict]:
    """Return JSON-IR dicts for all *action* leaf nodes in the tree."""
    results: list[dict] = []
    _collect_action_leaves(node, results)
    return results


def _collect_action_leaves(
    node: py_trees.behaviour.Behaviour,
    out: list[dict],
) -> None:
    if isinstance(node, ActionAffordanceNode):
        out.append(py_tree_to_json_ir(node))
        return
    if isinstance(node, py_trees.composites.Composite):
        for child in node.children:
            _collect_action_leaves(child, out)


def extract_action_urls(json_ir: dict) -> list[str]:
    """Recursively extract all action URLs from a JSON-IR tree."""
    urls: list[str] = []
    _collect_action_urls(json_ir, urls)
    return urls


def _collect_action_urls(node: dict, out: list[str]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "action":
        url = node.get("action_url")
        if url:
            out.append(url)
    for child in node.get("children", []):
        _collect_action_urls(child, out)


def extract_property_urls(json_ir: dict) -> list[str]:
    """Recursively extract all property URLs from a JSON-IR tree."""
    urls: list[str] = []
    _collect_property_urls(json_ir, urls)
    return urls


def _collect_property_urls(node: dict, out: list[str]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") in ("condition", "property_read"):
        url = node.get("property_url")
        if url:
            out.append(url)
    for child in node.get("children", []):
        _collect_property_urls(child, out)


# ---------------------------------------------------------------------------
# Plan / tree combination
# ---------------------------------------------------------------------------


def combine_trees_parallel(
    trees: list[py_trees.behaviour.Behaviour],
) -> py_trees.behaviour.Behaviour:
    """
    Combine multiple py_trees roots into a single Parallel node
    with ``SuccessOnOne`` policy.

    If the list has exactly one tree, return it directly.
    If the list is empty, return a trivial SUCCESS behaviour.
    """
    if len(trees) == 0:
        return py_trees.behaviours.Success(name="EmptyPlan")

    if len(trees) == 1:
        return trees[0]

    return py_trees.composites.Parallel(
        name="CombinedPlan",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=trees,
    )
