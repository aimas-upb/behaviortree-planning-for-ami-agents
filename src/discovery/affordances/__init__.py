"""
Affordance discovery strategies.

Strategies:
- exhaustive: Explore all workspaces/artifacts upfront
- agentic: LLM-guided exploration based on goal
- relevant: Exhaustive discovery, then filter to goal-relevant
"""

from .exhaustive import ExhaustiveAffordanceDiscovery
from .agentic import AgenticAffordanceDiscovery
from .relevant import RelevantAffordanceDiscovery

__all__ = [
    "ExhaustiveAffordanceDiscovery",
    "AgenticAffordanceDiscovery",
    "RelevantAffordanceDiscovery",
]


def create_affordance_strategy(
    strategy: str,
    client=None,
    model: str = "gpt-4o",
    max_workspaces: int = 10,
):
    """Factory function to create affordance discovery strategy."""
    if strategy == "exhaustive":
        return ExhaustiveAffordanceDiscovery(max_workspaces=max_workspaces)
    elif strategy == "agentic":
        if client is None:
            raise ValueError("Agentic strategy requires an OpenAI client")
        return AgenticAffordanceDiscovery(client=client, model=model)
    elif strategy == "relevant":
        if client is None:
            raise ValueError("Relevant strategy requires an OpenAI client")
        return RelevantAffordanceDiscovery(
            client=client,
            model=model,
            max_workspaces=max_workspaces,
        )
    else:
        raise ValueError(f"Unknown affordance strategy: {strategy}")
