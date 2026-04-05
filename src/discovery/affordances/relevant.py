"""
Relevant affordance discovery strategy.

Performs exhaustive discovery, then uses LLM to filter to goal-relevant artifacts.
"""

import json
import logging
from typing import Optional

from openai import OpenAI

from ...config import ModelConfig, get_model_kwargs
from ..base import Artifact, CapabilityModel, Workspace
from .exhaustive import ExhaustiveAffordanceDiscovery

logger = logging.getLogger(__name__)


class RelevantAffordanceDiscovery:
    """
    Relevant filtering discovery strategy.

    Performs exhaustive discovery first, then uses LLM to filter
    the capability model to only goal-relevant artifacts.
    """

    def __init__(
        self,
        client: OpenAI,
        model_config: ModelConfig,
        max_workspaces: int = 10,
    ):
        self.client = client
        self.model_config = model_config
        self.model = model_config.name
        self.exhaustive = ExhaustiveAffordanceDiscovery(
            max_workspaces=max_workspaces
        )

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,
    ) -> CapabilityModel:
        """
        Discover affordances and filter to goal-relevant ones.

        Args:
            entry_point: Root workspace URI
            goal: The goal to filter relevance (if None, returns exhaustive)

        Returns:
            CapabilityModel with filtered affordances
        """
        # First, do exhaustive discovery
        full_model = self.exhaustive.discover(entry_point, goal)

        # If no goal, return full model
        if not goal:
            logger.info("No goal provided, returning full capability model")
            return full_model

        # Filter to relevant artifacts
        return self._filter_for_goal(full_model, goal)

    def _filter_for_goal(
        self,
        model: CapabilityModel,
        goal: str,
    ) -> CapabilityModel:
        """Use LLM to filter capability model to goal-relevant artifacts."""
        # Build list of all artifacts
        artifact_list = []
        for art_uri, art in model.artifacts.items():
            ws_name = art.workspace.split("/")[-1].replace("#workspace", "")
            artifact_list.append(
                {
                    "uri": art_uri,
                    "name": art.name,
                    "workspace": ws_name,
                    "actions": [a.name for a in art.actions],
                }
            )

        if not artifact_list:
            return model

        # Ask LLM which are relevant
        filter_prompt = f"""Given this goal: "{goal}"

Which of these devices are relevant? Return a JSON array of URIs.

Devices:
{json.dumps(artifact_list, indent=2)}

Return ONLY a JSON array of relevant URIs, nothing else."""

        try:
            api_kwargs = get_model_kwargs(
                self.model, model_config=self.model_config
            )
            api_kwargs["messages"] = [
                {"role": "user", "content": filter_prompt}
            ]

            response = self.client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            relevant_uris = set(json.loads(content))

        except Exception as e:
            logger.warning(f"Filter failed ({e}), using all capabilities")
            return model

        # Build filtered model
        filtered = CapabilityModel(entry_point=model.entry_point)

        for ws_uri, workspace in model.workspaces.items():
            filtered_arts = [
                u for u in workspace.artifact_uris if u in relevant_uris
            ]
            if filtered_arts:
                filtered.workspaces[ws_uri] = Workspace(
                    uri=ws_uri,
                    artifact_uris=filtered_arts,
                    semantic_type=workspace.semantic_type,
                )

        for art_uri, art in model.artifacts.items():
            if art_uri in relevant_uris:
                filtered.artifacts[art_uri] = art

        logger.info(
            f"Filtered: {len(model.artifacts)} -> {len(filtered.artifacts)} artifacts"
        )
        return filtered
