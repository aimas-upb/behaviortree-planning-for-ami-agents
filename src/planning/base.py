"""
Base classes and dataclasses for planning.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol


@dataclass
class Plan:
    """
    Output of the planning phase.

    Contains either a JSON IR specification or Python code
    that constructs a behavior tree.
    """

    format: Literal["json_ir", "python_code"]
    content: str | dict  # JSON dict or Python code string
    explanation: str = ""
    reasoning_trace: list[str] = field(default_factory=list)
    detected_impossible: list[str] = field(
        default_factory=list
    )  # Sub-goals that cannot be achieved

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing."""
        return {
            "format": self.format,
            "content": self.content,  # Store actual content (JSON dict or Python code string)
            "explanation": self.explanation,
            "reasoning_trace": self.reasoning_trace,
            "has_reasoning": len(self.reasoning_trace) > 0,
            "detected_impossible": self.detected_impossible,
        }

    @property
    def is_json_ir(self) -> bool:
        return self.format == "json_ir"

    @property
    def is_python_code(self) -> bool:
        return self.format == "python_code"


@dataclass
class PlanningResult:
    """
    Complete result of the planning phase.

    Includes the plan and metadata about the planning process.
    """

    plan: Plan
    success: bool
    error: Optional[str] = None
    llm_calls: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing."""
        return {
            "plan": self.plan.to_dict(),
            "success": self.success,
            "error": self.error,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
        }


class ReasoningStrategy(Protocol):
    """Protocol for reasoning strategies."""

    def reason(
        self,
        goal: str,
        context: str,
        client,
        model: str,
    ) -> tuple[str, list[str]]:
        """
        Perform reasoning before plan generation.

        Args:
            goal: The user's goal
            context: Discovery context (affordances + state)
            client: OpenAI client
            model: Model name

        Returns:
            Tuple of (enhanced_context, reasoning_trace)
        """
        ...


class OutputGenerator(Protocol):
    """Protocol for output generators."""

    def generate(
        self,
        goal: str,
        context: str,
        client,
        model: str,
        prompt_strategy: str,
    ) -> Plan:
        """
        Generate a plan in the target format.

        Args:
            goal: The user's goal
            context: Context (possibly enhanced by reasoning)
            client: OpenAI client
            model: Model name
            prompt_strategy: Which prompt template to use

        Returns:
            Plan object with generated content
        """
        ...
