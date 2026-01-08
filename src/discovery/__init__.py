"""
Discovery module for environment exploration.

Handles two aspects:
1. Affordance discovery - What can be done (actions, properties)
2. State gathering - Current values of properties

Both support multiple strategies for ablation studies.
"""

from .base import (
    DiscoveryResult,
    EnvironmentState,
    CapabilityModel,
    Artifact,
    Affordance,
)
from .pipeline import create_discovery_pipeline, DiscoveryPipeline

__all__ = [
    "DiscoveryResult",
    "EnvironmentState",
    "CapabilityModel",
    "Artifact",
    "Affordance",
    "create_discovery_pipeline",
    "DiscoveryPipeline",
]
