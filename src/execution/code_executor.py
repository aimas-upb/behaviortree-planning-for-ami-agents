"""
Code Executor - executes generated Python code directly.

Includes safety checks to prevent dangerous operations.
Supports both constrained (template nodes) and unconstrained (custom behaviors) modes.
"""

import ast
import re
import logging

import httpx
import py_trees
from py_trees.common import Status

from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator,
)

from ..planning import Plan
from .base import ExecutionResult

logger = logging.getLogger(__name__)


# Patterns that indicate potentially dangerous code (apply to both modes)
FORBIDDEN_PATTERNS = [
    r'\bos\.(remove|unlink|rmdir|rmtree|system|popen|exec|spawn)',
    r'\bshutil\.(rmtree|move|copy|copytree)',
    r'\bsubprocess\.',
    r'\b__import__\s*\(',
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'\bcompile\s*\(',
    r'\bopen\s*\([^)]*["\']w',  # write mode
    r'\brm\s+-rf',
    r'\bimport\s+subprocess',
    r'\bimport\s+shutil',
    r'\bimport\s+os\b',
    r'\bfrom\s+os\s+import',
    r'\bimport\s+sys\b',
    r'\bfrom\s+sys\s+import',
    r'\b__builtins__',
    r'\b__class__',
    r'\b__bases__',
    r'\b__subclasses__',
    r'\b__mro__',
    r'\b__globals__',
    r'\b__code__',
]

# Allowed imports for constrained mode
ALLOWED_IMPORTS_CONSTRAINED = {
    "py_trees",
}

# Allowed imports for unconstrained mode (http access via provided client)
ALLOWED_IMPORTS_UNCONSTRAINED = {
    "py_trees",
}


class CodeSafetyError(Exception):
    """Raised when generated code fails safety checks."""
    pass


class CodeExecutor:
    """
    Executor for Python code behavior trees.

    Includes safety checks before execution.
    Supports two modes:
    - Constrained (python_code): Uses predefined template nodes
    - Unconstrained (python_code_unconstrained): Allows custom behaviors with HTTP access
    """

    def __init__(self, max_ticks: int = 10, unconstrained: bool = False):
        self.max_ticks = max_ticks
        self.unconstrained = unconstrained
        self._http_client = None

    @property
    def http_client(self):
        """Lazy-init HTTP client for unconstrained mode."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def execute(self, plan: Plan) -> ExecutionResult:
        """
        Execute a Python code plan.

        Args:
            plan: Plan with Python code content

        Returns:
            ExecutionResult with execution details
        """
        valid_formats = ["python_code", "python_code_unconstrained"]
        if plan.format not in valid_formats:
            return ExecutionResult(
                success=False,
                error=f"Expected python_code or python_code_unconstrained format, got {plan.format}",
            )

        # Set unconstrained mode based on plan format
        is_unconstrained = plan.format == "python_code_unconstrained"

        code = plan.content
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult(
                success=False,
                error="Empty or invalid code",
            )

        try:
            # Safety check
            logger.info(f"Checking code safety (unconstrained={is_unconstrained})")
            self._check_safety(code, unconstrained=is_unconstrained)

            # Execute code
            logger.info("Executing Python code")
            tree = self._execute_code(code, unconstrained=is_unconstrained)

            # Execute tree
            logger.info("Executing behavior tree")
            result = self._execute_tree(tree)
            return result

        except CodeSafetyError as e:
            logger.error(f"Code safety check failed: {e}")
            return ExecutionResult(
                success=False,
                error=f"Safety check failed: {e}",
            )
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
            )
        finally:
            # Clean up HTTP client
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None

    def _check_safety(self, code: str, unconstrained: bool = False) -> None:
        """
        Check code for potentially dangerous patterns.

        Args:
            code: Python code string
            unconstrained: Whether to use unconstrained mode

        Raises:
            CodeSafetyError: If dangerous patterns are found
        """
        # Pattern matching (same for both modes)
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                raise CodeSafetyError(f"Forbidden pattern detected: {pattern}")

        # Choose allowed imports based on mode
        allowed_imports = ALLOWED_IMPORTS_UNCONSTRAINED if unconstrained else ALLOWED_IMPORTS_CONSTRAINED

        # AST-based import checking
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in allowed_imports:
                            raise CodeSafetyError(f"Forbidden import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in allowed_imports:
                            raise CodeSafetyError(f"Forbidden import from: {node.module}")
        except SyntaxError as e:
            raise CodeSafetyError(f"Invalid Python syntax: {e}")

    def _execute_code(self, code: str, unconstrained: bool = False) -> py_trees.behaviour.Behaviour:
        """
        Execute code and extract the behavior tree.

        Args:
            code: Python code string
            unconstrained: Whether to use unconstrained mode

        Returns:
            Constructed py_trees behavior

        Raises:
            CodeSafetyError: If tree is not defined or code fails
        """
        # Base builtins for both modes
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "True": True,
            "False": False,
            "None": None,
            "Exception": Exception,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "type": type,
            "super": super,
            # Math operations
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sum": sum,
            # Iteration helpers
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
        }

        # Prepare globals based on mode
        if unconstrained:
            # Unconstrained mode: full py_trees access + HTTP client
            # Add __build_class__ to allow class definitions
            unconstrained_builtins = dict(safe_builtins)
            unconstrained_builtins["__build_class__"] = __builtins__["__build_class__"] if isinstance(__builtins__, dict) else getattr(__builtins__, "__build_class__")
            unconstrained_builtins["__name__"] = "__main__"
            safe_globals = {
                "__builtins__": unconstrained_builtins,
                "py_trees": py_trees,
                "Status": Status,
                "http_client": self.http_client,
            }
            logger.info("Unconstrained mode: providing http_client and full py_trees access")
        else:
            # Constrained mode: template nodes only
            safe_globals = {
                "__builtins__": safe_builtins,
                "py_trees": py_trees,
                "ActionAffordanceNode": ActionAffordanceNode,
                "PropertyConditionNode": PropertyConditionNode,
                "ComparisonPropertyConditionNode": ComparisonPropertyConditionNode,
                "ComparisonOperator": ComparisonOperator,
            }

        local_namespace = {}

        try:
            exec(code, safe_globals, local_namespace)
        except Exception as e:
            raise CodeSafetyError(f"Code execution failed: {e}")

        # Extract tree
        if "tree" in local_namespace:
            tree = local_namespace["tree"]
            # Handle BehaviourTree wrapper - extract the root
            if isinstance(tree, py_trees.trees.BehaviourTree):
                tree = tree.root
            if not isinstance(tree, py_trees.behaviour.Behaviour):
                raise CodeSafetyError(
                    f"'tree' must be a py_trees.behaviour.Behaviour, got {type(tree)}"
                )
            return tree
        elif "build_tree" in local_namespace:
            try:
                tree = local_namespace["build_tree"]()
                # Handle BehaviourTree wrapper - extract the root
                if isinstance(tree, py_trees.trees.BehaviourTree):
                    tree = tree.root
                if not isinstance(tree, py_trees.behaviour.Behaviour):
                    raise CodeSafetyError(
                        f"build_tree() must return a Behaviour, got {type(tree)}"
                    )
                return tree
            except Exception as e:
                raise CodeSafetyError(f"build_tree() failed: {e}")
        else:
            raise CodeSafetyError("Code must define 'tree' variable or 'build_tree()' function")

    def _execute_tree(self, tree: py_trees.behaviour.Behaviour) -> ExecutionResult:
        """
        Execute a behavior tree.

        Args:
            tree: py_trees behavior

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
        logger.info(f"Execution complete: {result.final_status} after {result.ticks} ticks")
        return result
