"""
Discovery pipeline that combines affordance discovery and state gathering.
"""

from typing import Optional
import logging

from openai import OpenAI

from .base import DiscoveryResult, CapabilityModel, EnvironmentState
from .affordances import create_affordance_strategy
from .state import create_state_strategy
from ..config import DiscoveryConfig

logger = logging.getLogger(__name__)


class DiscoveryPipeline:
    """
    Discovery pipeline combining affordance discovery and state gathering.

    Orchestrates the two-phase discovery process:
    1. Discover affordances (what can be done)
    2. Gather state (current values)
    """

    def __init__(
        self,
        affordance_strategy,
        state_strategy,
    ):
        self.affordance_strategy = affordance_strategy
        self.state_strategy = state_strategy

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Run the discovery pipeline.

        Args:
            entry_point: Root workspace URI
            goal: Optional goal for guided discovery

        Returns:
            DiscoveryResult with affordances and state
        """
        logger.info(f"Starting discovery pipeline from {entry_point}")

        # Phase 1: Discover affordances
        affordances = self.affordance_strategy.discover(entry_point, goal)
        logger.info(
            f"Affordances discovered: {len(affordances.artifacts)} artifacts"
        )

        # Capture exploration trace if available (for agentic discovery)
        exploration_trace = []
        if hasattr(self.affordance_strategy, 'get_exploration_trace'):
            exploration_trace = self.affordance_strategy.get_exploration_trace()

        # Phase 2: Gather state
        state = self.state_strategy.gather(affordances, goal)
        logger.info(
            f"State gathered: {len(state.property_values)} property values"
        )

        # Capture state trace if available (for agentic state discovery)
        state_trace = []
        if hasattr(self.state_strategy, 'get_state_trace'):
            state_trace = self.state_strategy.get_state_trace()

        return DiscoveryResult(
            affordances=affordances,
            state=state,
            exploration_trace=exploration_trace,
            state_trace=state_trace,
        )


def create_discovery_pipeline(
    config: DiscoveryConfig,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
) -> DiscoveryPipeline:
    """
    Factory function to create a discovery pipeline from config.

    Args:
        config: Discovery configuration
        client: OpenAI client (required for agentic/relevant strategies)
        model: Model name for LLM-based strategies

    Returns:
        Configured DiscoveryPipeline
    """
    # Create affordance strategy
    affordance_strategy = create_affordance_strategy(
        strategy=config.affordances.strategy,
        client=client,
        model=model,
        max_workspaces=config.affordances.max_workspaces,
    )

    # Create state strategy
    state_strategy = create_state_strategy(
        strategy=config.state.strategy,
        client=client,
        model=model,
    )

    return DiscoveryPipeline(
        affordance_strategy=affordance_strategy,
        state_strategy=state_strategy,
    )
