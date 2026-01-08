"""
Modular Behavior Tree Planning Agent for HMAS environments.

This package provides a configurable pipeline for:
1. Discovery - Discover affordances and gather state from the environment
2. Planning - Generate behavior trees using various reasoning strategies
3. Execution - Execute behavior trees (JSON IR or Python code)

Configuration is YAML-driven for experiment reproducibility.
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
