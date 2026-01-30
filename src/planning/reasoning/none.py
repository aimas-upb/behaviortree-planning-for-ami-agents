"""
No reasoning strategy.

Passes context through unchanged for direct generation.
"""

import logging

logger = logging.getLogger(__name__)


class NoReasoning:
    """
    No-op reasoning strategy.

    Returns context unchanged, no reasoning trace.
    """

    def reason(
        self,
        goal: str,
        context: str,
        client,
        model_config=None,
    ) -> tuple[str, list[str]]:
        """
        Pass through without reasoning.

        Args:
            goal: The user's goal
            context: Discovery context
            client: OpenAI client (unused)
            model_config: Model configuration (unused)

        Returns:
            Tuple of (unchanged context, empty trace)
        """
        logger.debug("No reasoning (strategy: none)")
        return context, []
