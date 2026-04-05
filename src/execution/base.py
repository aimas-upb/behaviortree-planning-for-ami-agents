"""
Base classes for execution.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..planning import Plan


@dataclass
class ExecutionResult:
    """Result of executing a behavior tree."""

    success: bool
    tree_name: str = ""
    ticks: int = 0
    final_status: str = ""
    tick_history: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing."""
        return {
            "success": self.success,
            "tree_name": self.tree_name,
            "ticks": self.ticks,
            "final_status": self.final_status,
            "tick_history": self.tick_history,
            "error": self.error,
        }


class Executor(Protocol):
    """Protocol for executors."""

    def execute(self, plan: Plan) -> ExecutionResult:
        """Execute a plan and return the result."""
        ...
