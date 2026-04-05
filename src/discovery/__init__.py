"""
Discovery module for environment exploration.

Handles two aspects:
1. Affordance discovery - What can be done (actions, properties)
2. State gathering - Current values of properties

Both support multiple strategies for ablation studies.
"""

from .base import (
    Affordance,
    Artifact,
    CapabilityModel,
    DiscoveryResult,
    EnvironmentState,
    Workspace,
)
from .pipeline import DiscoveryPipeline, create_discovery_pipeline

__all__ = [
    "DiscoveryResult",
    "EnvironmentState",
    "CapabilityModel",
    "Workspace",
    "Artifact",
    "Affordance",
    "create_discovery_pipeline",
    "DiscoveryPipeline",
]
