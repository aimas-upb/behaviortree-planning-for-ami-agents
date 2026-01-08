"""
No state gathering strategy.

Returns empty state - current behavior where state is read at execution time.
"""

from typing import Optional
import logging

from ..base import CapabilityModel, EnvironmentState

logger = logging.getLogger(__name__)


class NoStateGathering:
    """
    No-op state gathering strategy.

    Returns empty state. State will be read during execution.
    """

    def gather(
        self,
        affordances: CapabilityModel,
        goal: Optional[str] = None,
    ) -> EnvironmentState:
        """
        Return empty state.

        Args:
            affordances: Discovered capability model (ignored)
            goal: Goal (ignored)

        Returns:
            Empty EnvironmentState
        """
        logger.debug("No state gathering (strategy: none)")
        return EnvironmentState()
