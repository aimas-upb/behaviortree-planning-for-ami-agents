"""
State gathering strategies.

Strategies:
- all: Read all property values from discovered artifacts
- relevant: Use LLM to filter to goal-relevant properties
- agentic: Use LLM to interactively explore and read properties
- none: No state gathering (current behavior)
"""

from .all import AllStateGathering
from .relevant import RelevantStateGathering
from .agentic import AgenticStateGathering
from .none import NoStateGathering

__all__ = [
    "AllStateGathering",
    "RelevantStateGathering",
    "AgenticStateGathering",
    "NoStateGathering",
]


def create_state_strategy(
    strategy: str,
    client=None,
    model: str = "gpt-4o",
):
    """Factory function to create state gathering strategy."""
    if strategy == "none":
        return NoStateGathering()
    elif strategy == "all":
        return AllStateGathering()
    elif strategy == "relevant":
        if client is None:
            raise ValueError("Relevant strategy requires an OpenAI client")
        return RelevantStateGathering(client=client, model=model)
    elif strategy == "agentic":
        if client is None:
            raise ValueError("Agentic strategy requires an OpenAI client")
        return AgenticStateGathering(client=client, model=model)
    else:
        raise ValueError(f"Unknown state strategy: {strategy}")
