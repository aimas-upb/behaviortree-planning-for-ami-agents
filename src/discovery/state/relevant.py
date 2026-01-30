"""
Relevant state gathering strategy.

Uses LLM to identify goal-relevant properties, then reads only those.
"""

import json
from typing import Optional
import logging
from datetime import datetime

from openai import OpenAI

from ..base import CapabilityModel, EnvironmentState
from hmas_client import get_property_by_uri, GetPropertyError
from ...config import ModelConfig, get_model_kwargs

logger = logging.getLogger(__name__)


class RelevantStateGathering:
    """
    Relevant state gathering strategy.

    Uses LLM to filter to goal-relevant properties before reading.
    """

    def __init__(
        self,
        client: OpenAI,
        model_config: ModelConfig,
    ):
        self.client = client
        self.model_config = model_config
        self.model = model_config.name

    def gather(
        self,
        affordances: CapabilityModel,
        goal: Optional[str] = None,
    ) -> EnvironmentState:
        """
        Gather goal-relevant property values.

        Args:
            affordances: Discovered capability model
            goal: The goal to filter relevance

        Returns:
            EnvironmentState with relevant property values
        """
        if not goal:
            logger.info("No goal provided, gathering all state")
            return self._gather_all(affordances)

        logger.info(f"Gathering relevant state for goal: {goal}")

        # Build property list
        property_list = []
        for artifact in affordances.artifacts.values():
            for prop in artifact.properties:
                property_list.append({
                    "uri": prop.uri,
                    "name": prop.name,
                    "artifact": artifact.name,
                })

        if not property_list:
            return EnvironmentState()

        # Ask LLM which properties are relevant
        relevant_uris = self._filter_properties(property_list, goal)

        # Read only relevant properties
        state = EnvironmentState(timestamp=datetime.now().isoformat())

        for prop_uri in relevant_uris:
            try:
                value = get_property_by_uri(prop_uri)
                state.property_values[prop_uri] = value
                logger.debug(f"Read property {prop_uri}: {value}")
            except GetPropertyError as e:
                state.errors[prop_uri] = str(e)
                logger.warning(f"Failed to read property {prop_uri}: {e}")
            except Exception as e:
                state.errors[prop_uri] = str(e)
                logger.warning(f"Unexpected error reading property {prop_uri}: {e}")

        logger.info(
            f"State gathered: {len(state.property_values)} values "
            f"(filtered from {len(property_list)} total)"
        )
        return state

    def _filter_properties(
        self,
        property_list: list[dict],
        goal: str,
    ) -> list[str]:
        """Use LLM to filter to relevant properties."""
        filter_prompt = f"""Given this goal: "{goal}"

Which of these properties would be useful to read before planning? Return a JSON array of URIs.

Properties:
{json.dumps(property_list, indent=2)}

Return ONLY a JSON array of relevant URIs, nothing else."""

        try:
            api_kwargs = get_model_kwargs(self.model, model_config=self.model_config)
            api_kwargs["messages"] = [{"role": "user", "content": filter_prompt}]

            response = self.client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"Property filter failed ({e}), returning all properties")
            return [p["uri"] for p in property_list]

    def _gather_all(self, affordances: CapabilityModel) -> EnvironmentState:
        """Fallback to gather all properties."""
        from .all import AllStateGathering
        return AllStateGathering().gather(affordances)
