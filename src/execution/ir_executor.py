"""
IR Executor - compiles JSON IR to py_trees and executes.
"""

import logging

import py_trees
from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    ComparisonOperator,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
)
from py_trees.common import Status

from ..planning import Plan
from .base import ExecutionResult

logger = logging.getLogger(__name__)


OPERATOR_MAP = {
    "==": ComparisonOperator.EQUAL,
    "!=": ComparisonOperator.NOT_EQUAL,
    ">": ComparisonOperator.GREATER_THAN,
    ">=": ComparisonOperator.GREATER_THAN_OR_EQUAL,
    "<": ComparisonOperator.LESS_THAN,
    "<=": ComparisonOperator.LESS_THAN_OR_EQUAL,
    "in": ComparisonOperator.IN,
    "not_in": ComparisonOperator.NOT_IN,
    "contains": ComparisonOperator.CONTAINS,
}


class IRExecutor:
    """
    Executor for JSON IR behavior trees.

    Compiles JSON specification to py_trees objects and executes.
    """

    def __init__(self, max_ticks: int = 10):
        self.max_ticks = max_ticks

    def execute(self, plan: Plan) -> ExecutionResult:
        """
        Execute a JSON IR plan.

        Args:
            plan: Plan with JSON IR content

        Returns:
            ExecutionResult with execution details
        """
        if plan.format != "json_ir":
            return ExecutionResult(
                success=False,
                error=f"Expected json_ir format, got {plan.format}",
            )

        if not isinstance(plan.content, dict):
            return ExecutionResult(
                success=False,
                error="Invalid tree specification",
            )

        if plan.explanation.startswith(
            "Generation failed"
        ) or plan.explanation.startswith("JSON parse error"):
            return ExecutionResult(
                success=False,
                error=f"Plan generation error: {plan.explanation}",
            )

        if not plan.content:
            # No behavior tree to execute
            logger.info("No behavior tree to execute;")
            return ExecutionResult(
                success=True,
                final_status="No behavior tree to execute.",
                ticks=0,
                tick_history=[],
            )

        try:
            # Compile
            logger.info("Compiling JSON IR to py_trees")
            tree = self._compile(plan.content)
            logger.info(f"Compiled tree: {tree.name}")

            # Execute
            logger.info("Executing behavior tree")
            result = self._execute_tree(tree)
            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
            )

    def _compile(self, spec: dict) -> py_trees.behaviour.Behaviour:
        """
        Compile a JSON specification to py_trees.

        Args:
            spec: JSON tree specification

        Returns:
            py_trees behavior
        """
        node_type = spec.get("type")
        name = spec.get("name", "unnamed")

        if node_type == "sequence":
            children = [
                self._compile(child) for child in spec.get("children", [])
            ]
            return py_trees.composites.Sequence(
                name=name, memory=True, children=children
            )

        elif node_type == "selector":
            children = [
                self._compile(child) for child in spec.get("children", [])
            ]
            return py_trees.composites.Selector(
                name=name, memory=False, children=children
            )

        elif node_type == "parallel":
            children = [
                self._compile(child) for child in spec.get("children", [])
            ]
            policy_name = spec.get("policy", "success_on_all")
            if policy_name == "success_on_one":
                policy = py_trees.common.ParallelPolicy.SuccessOnOne()
            else:
                policy = py_trees.common.ParallelPolicy.SuccessOnAll()
            return py_trees.composites.Parallel(
                name=name, policy=policy, children=children
            )

        elif node_type == "action":
            return ActionAffordanceNode(
                name=name,
                action_url=spec["action_url"],
                parameters=spec.get("parameters", {}),
                parameter_keys=spec.get("parameter_keys", {}),
                result_key=spec.get("result_key"),
            )

        elif node_type == "property_read":
            return PropertyAffordanceNode(
                name=name,
                property_url=spec["property_url"],
                result_key=spec.get("result_key"),
                property_name=spec.get("property_name"),
            )

        elif node_type == "condition":
            operator = spec.get("operator")
            if operator and operator != "==":
                return ComparisonPropertyConditionNode(
                    name=name,
                    property_url=spec["property_url"],
                    expected_value=spec["expected_value"],
                    operator=OPERATOR_MAP.get(
                        operator, ComparisonOperator.EQUAL
                    ),
                    value_path=spec.get("value_path"),
                )
            else:
                return PropertyConditionNode(
                    name=name,
                    property_url=spec["property_url"],
                    expected_value=spec["expected_value"],
                    value_path=spec.get("value_path"),
                )

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _execute_tree(
        self, tree: py_trees.behaviour.Behaviour
    ) -> ExecutionResult:
        """
        Execute a compiled behavior tree.

        Args:
            tree: Compiled py_trees behavior

        Returns:
            ExecutionResult with execution details
        """
        tree.setup_with_descendants()

        result = ExecutionResult(
            success=False,
            tree_name=tree.name,
            ticks=0,
            tick_history=[],
        )

        for tick in range(self.max_ticks):
            result.ticks = tick + 1
            tree.tick_once()

            status_name = tree.status.name
            result.tick_history.append(status_name)
            logger.debug(f"Tick {tick + 1}: {status_name}")

            if tree.status == Status.SUCCESS:
                result.final_status = "SUCCESS"
                result.success = True
                break
            elif tree.status == Status.FAILURE:
                result.final_status = "FAILURE"
                result.success = False
                break
        else:
            result.final_status = "RUNNING (max ticks reached)"

        tree.shutdown()
        logger.info(
            f"Execution complete: {result.final_status} after {result.ticks} ticks"
        )
        return result
