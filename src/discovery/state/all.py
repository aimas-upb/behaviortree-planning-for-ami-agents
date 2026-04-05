"""
All state gathering strategy.

Reads all property values from discovered artifacts.
"""

import logging
from datetime import datetime
from typing import Optional

from ...hmas_client import GetPropertyError, get_property_by_uri
from ..base import CapabilityModel, EnvironmentState

logger = logging.getLogger(__name__)


class AllStateGathering:
    """
    Exhaustive state gathering strategy.

    Reads all property values from all discovered artifacts.
    """

    def gather(
        self,
        affordances: CapabilityModel,
        goal: Optional[str] = None,  # Ignored
    ) -> EnvironmentState:
        """
        Gather all property values.

        Args:
            affordances: Discovered capability model
            goal: Ignored in this strategy

        Returns:
            EnvironmentState with all property values
        """
        logger.info("Gathering all property values")

        state = EnvironmentState(timestamp=datetime.now().isoformat())
        property_uris = affordances.get_all_property_uris()

        for prop_uri in property_uris:
            try:
                value = get_property_by_uri(prop_uri)
                state.property_values[prop_uri] = value
                logger.debug(f"Read property {prop_uri}: {value}")
            except GetPropertyError as e:
                state.errors[prop_uri] = str(e)
                logger.warning(f"Failed to read property {prop_uri}: {e}")
            except Exception as e:
                state.errors[prop_uri] = str(e)
                logger.warning(
                    f"Unexpected error reading property {prop_uri}: {e}"
                )

        logger.info(
            f"State gathered: {len(state.property_values)} values, "
            f"{len(state.errors)} errors"
        )
        return state
