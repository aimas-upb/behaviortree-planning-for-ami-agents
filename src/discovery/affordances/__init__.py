"""
Affordance discovery strategies.

Strategies:
- exhaustive: Explore all workspaces/artifacts upfront
- agentic: LLM-guided exploration based on goal
- relevant: Exhaustive discovery, then filter to goal-relevant
- agentic_query: LLM generates SPARQL queries to find relevant affordances
"""

from .agentic import AgenticAffordanceDiscovery
from .agentic_query import AgenticQueryAffordanceDiscovery
from .exhaustive import ExhaustiveAffordanceDiscovery
from .relevant import RelevantAffordanceDiscovery

__all__ = [
    "ExhaustiveAffordanceDiscovery",
    "AgenticAffordanceDiscovery",
    "RelevantAffordanceDiscovery",
    "AgenticQueryAffordanceDiscovery",
]


def create_affordance_strategy(
    strategy: str,
    client=None,
    model_config=None,
    max_workspaces: int = 10,
    semantic_query_prompt_path=None,
):
    """Factory function to create affordance discovery strategy."""
    if strategy == "exhaustive":
        return ExhaustiveAffordanceDiscovery(max_workspaces=max_workspaces)
    elif strategy == "agentic":
        if client is None:
            raise ValueError("Agentic strategy requires an OpenAI client")
        return AgenticAffordanceDiscovery(
            client=client, model_config=model_config
        )
    elif strategy == "relevant":
        if client is None:
            raise ValueError("Relevant strategy requires an OpenAI client")
        return RelevantAffordanceDiscovery(
            client=client,
            model_config=model_config,
            max_workspaces=max_workspaces,
        )
    elif strategy == "agentic_query":
        if client is None:
            raise ValueError("Agentic query strategy requires an OpenAI client")
        return AgenticQueryAffordanceDiscovery(
            client=client,
            model_config=model_config,
            semantic_query_prompt_path=semantic_query_prompt_path,
        )
    else:
        raise ValueError(f"Unknown affordance strategy: {strategy}")
