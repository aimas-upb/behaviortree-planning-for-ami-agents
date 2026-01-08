"""
Main Planner class that orchestrates reasoning and output generation.
"""

import logging
from typing import Optional

from openai import OpenAI

from .base import Plan, PlanningResult
from .reasoning import create_reasoning_strategy
from .output import create_output_generator
from ..config import PlanningConfig
from ..discovery import DiscoveryResult

logger = logging.getLogger(__name__)


class Planner:
    """
    Orchestrates the planning phase.

    Combines:
    1. Reasoning strategy (optional pre-generation thinking)
    2. Output generator (JSON IR or Python code)
    """

    def __init__(
        self,
        reasoning_strategy,
        output_generator,
        prompt_strategy: str = "detailed",
    ):
        self.reasoning_strategy = reasoning_strategy
        self.output_generator = output_generator
        self.prompt_strategy = prompt_strategy

    def plan(
        self,
        goal: str,
        discovery: DiscoveryResult,
        client: OpenAI,
        model: str,
    ) -> PlanningResult:
        """
        Generate a plan for the given goal.

        Args:
            goal: The user's goal
            discovery: Discovery result with affordances and state
            client: OpenAI client
            model: Model name

        Returns:
            PlanningResult with the generated plan
        """
        logger.info(f"Starting planning for goal: {goal}")
        llm_calls = 0

        try:
            # Get context from discovery
            context = discovery.to_prompt_context()

            # Phase 1: Reasoning (optional)
            enhanced_context, reasoning_trace = self.reasoning_strategy.reason(
                goal=goal,
                context=context,
                client=client,
                model=model,
            )

            # Count LLM calls from reasoning
            if reasoning_trace:
                # Rough estimate based on reasoning type
                llm_calls += len(reasoning_trace)

            # Phase 2: Output generation
            plan = self.output_generator.generate(
                goal=goal,
                context=enhanced_context,
                client=client,
                model=model,
                prompt_strategy=self.prompt_strategy,
            )
            llm_calls += 1

            # Attach reasoning trace to plan
            plan.reasoning_trace = reasoning_trace

            logger.info(
                f"Planning complete: {plan.format} output, "
                f"{len(reasoning_trace)} reasoning steps"
            )

            return PlanningResult(
                plan=plan,
                success=True,
                llm_calls=llm_calls,
            )

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return PlanningResult(
                plan=Plan(format="json_ir", content={}, explanation=str(e)),
                success=False,
                error=str(e),
                llm_calls=llm_calls,
            )


def create_planner(
    config: PlanningConfig,
) -> Planner:
    """
    Factory function to create a Planner from config.

    Args:
        config: Planning configuration

    Returns:
        Configured Planner
    """
    # Create reasoning strategy
    if config.reasoning.enabled:
        reasoning_strategy = create_reasoning_strategy(
            strategy=config.reasoning.strategy,
            max_turns=config.reasoning.max_turns,
        )
    else:
        reasoning_strategy = create_reasoning_strategy("none")

    # Create output generator
    output_generator = create_output_generator(config.output.format)

    return Planner(
        reasoning_strategy=reasoning_strategy,
        output_generator=output_generator,
        prompt_strategy=config.prompt_strategy,
    )
