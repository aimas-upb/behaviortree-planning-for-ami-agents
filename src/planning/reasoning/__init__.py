"""
Reasoning strategies for planning.

Strategies:
- none: No reasoning, direct generation
- chain_of_thought: Single-turn CoT before generation
- multi_turn: Multi-turn exploration before generation
- reflection: Generate, critique, and refine
"""

from .chain_of_thought import ChainOfThoughtReasoning
from .multi_turn import MultiTurnReasoning
from .none import NoReasoning
from .reflection import ReflectionReasoning

__all__ = [
    "NoReasoning",
    "ChainOfThoughtReasoning",
    "MultiTurnReasoning",
    "ReflectionReasoning",
]


def create_reasoning_strategy(
    strategy: str,
    max_turns: int = 3,
):
    """Factory function to create reasoning strategy."""
    if strategy == "none" or not strategy:
        return NoReasoning()
    elif strategy == "chain_of_thought":
        return ChainOfThoughtReasoning()
    elif strategy == "multi_turn":
        return MultiTurnReasoning(max_turns=max_turns)
    elif strategy == "reflection":
        return ReflectionReasoning()
    else:
        raise ValueError(f"Unknown reasoning strategy: {strategy}")
