"""
Exhaustive affordance discovery strategy.

Explores all workspaces and artifacts upfront without any filtering.
"""

from typing import Optional
import logging

from ..base import CapabilityModel, Artifact, Affordance
from hmas_client import (
    list_workspaces,
    list_artifacts,
    list_properties,
    list_actions,
    get_artifact_name,
)

logger = logging.getLogger(__name__)


class ExhaustiveAffordanceDiscovery:
    """
    Exhaustive discovery strategy.

    Explores all workspaces up to max_workspaces limit and discovers
    all artifacts and their affordances.
    """

    def __init__(self, max_workspaces: int = 10):
        self.max_workspaces = max_workspaces

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,  # Ignored in exhaustive mode
    ) -> CapabilityModel:
        """
        Explore environment and build capability model.

        Args:
            entry_point: Root workspace URI
            goal: Ignored in exhaustive mode

        Returns:
            CapabilityModel with all discovered affordances
        """
        logger.info(f"Starting exhaustive discovery from {entry_point}")
        model = CapabilityModel(entry_point=entry_point)

        try:
            workspaces = list_workspaces(entry_point)[:self.max_workspaces]
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            return model

        for ws_uri in workspaces:
            ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
            logger.debug(f"Exploring workspace: {ws_name}")

            try:
                artifact_uris = list_artifacts(ws_uri)
                model.workspaces[ws_uri] = artifact_uris

                for art_uri in artifact_uris:
                    try:
                        artifact = self._discover_artifact(art_uri, ws_uri)
                        model.artifacts[art_uri] = artifact
                        logger.debug(
                            f"Discovered artifact: {artifact.name} "
                            f"({len(artifact.actions)} actions, {len(artifact.properties)} properties)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to inspect artifact {art_uri}: {e}")

            except Exception as e:
                logger.warning(f"Failed to list artifacts in {ws_name}: {e}")

        logger.info(
            f"Discovery complete: {len(model.artifacts)} artifacts, "
            f"{sum(len(a.actions) for a in model.artifacts.values())} actions"
        )
        return model

    def _discover_artifact(self, artifact_uri: str, workspace_uri: str) -> Artifact:
        """Discover a single artifact's affordances."""
        name = get_artifact_name(artifact_uri)

        actions = []
        for a in list_actions(artifact_uri):
            actions.append(Affordance(
                name=a["name"],
                uri=a["uri"],
                schema=a.get("input_schema", {})
            ))

        properties = []
        for p in list_properties(artifact_uri):
            properties.append(Affordance(
                name=p["name"],
                uri=p["uri"],
                schema=p.get("output_schema", {})
            ))

        return Artifact(
            name=name,
            uri=artifact_uri,
            workspace=workspace_uri,
            actions=actions,
            properties=properties
        )
