"""
State gathering strategies.

Strategies:
- all: Read all property values from discovered artifacts
- relevant: Use LLM to filter to goal-relevant properties
- agentic: Use LLM to interactively explore and read properties
- none: No state gathering (current behavior)
"""

from .agentic import AgenticStateGathering
from .all import AllStateGathering
from .none import NoStateGathering
from .relevant import RelevantStateGathering

__all__ = [
    "AllStateGathering",
    "RelevantStateGathering",
    "AgenticStateGathering",
    "NoStateGathering",
]


def create_state_strategy(
    strategy: str,
    client=None,
    model_config=None,
):
    """Factory function to create state gathering strategy."""
    if strategy == "none":
        return NoStateGathering()
    elif strategy == "all":
        return AllStateGathering()
    elif strategy == "relevant":
        if client is None:
            raise ValueError("Relevant strategy requires an OpenAI client")
        return RelevantStateGathering(client=client, model_config=model_config)
    elif strategy == "agentic":
        if client is None:
            raise ValueError("Agentic strategy requires an OpenAI client")
        return AgenticStateGathering(client=client, model_config=model_config)
    else:
        raise ValueError(f"Unknown state strategy: {strategy}")
