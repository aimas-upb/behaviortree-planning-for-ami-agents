"""
Configuration models for experiment definition.

Uses Pydantic for validation and YAML for serialization.
"""

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class AffordanceConfig(BaseModel):
    """Configuration for affordance discovery."""

    strategy: Literal["exhaustive", "agentic", "relevant", "agentic_query"] = (
        "exhaustive"
    )
    max_workspaces: int = Field(
        default=10, description="Max workspaces to explore"
    )


class StateConfig(BaseModel):
    """Configuration for state gathering."""

    strategy: Literal["all", "relevant", "agentic", "none"] = "none"


class DiscoveryConfig(BaseModel):
    """Configuration for the discovery phase."""

    affordances: AffordanceConfig = Field(default_factory=AffordanceConfig)
    state: StateConfig = Field(default_factory=StateConfig)


class ReasoningConfig(BaseModel):
    """Configuration for reasoning strategies."""

    enabled: bool = False
    strategy: Literal["chain_of_thought", "multi_turn", "reflection"] = (
        "chain_of_thought"
    )
    max_turns: int = Field(
        default=3, description="Max turns for multi_turn strategy"
    )


class OutputConfig(BaseModel):
    """Configuration for plan output format.

    Formats:
    - json_ir: Declarative JSON intermediate representation
    - python_code: Python code using predefined template nodes
    - python_code_unconstrained: Python code with custom py_trees behaviors
    """

    format: Literal["json_ir", "python_code", "python_code_unconstrained"] = (
        "json_ir"
    )


class PlanningConfig(BaseModel):
    """Configuration for the planning phase."""

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompt_strategy: str = "detailed"


class ExecutionConfig(BaseModel):
    """Configuration for the execution phase."""

    max_ticks: int = Field(default=10, description="Max behavior tree ticks")


class ModelConfig(BaseModel):
    """Configuration for the LLM."""

    name: str = "gpt-4o"
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = Field(
        default=None, description="API key (or use env var)"
    )


class TracingConfig(BaseModel):
    """Configuration for tracing/logging."""

    enabled: bool = True
    output_dir: str = "traces/"
    verbose: bool = False


class ExperienceConfig(BaseModel):
    """Configuration for experience-based planning."""

    enabled: bool = False
    persistence_path: str = "experience_store.json"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    similarity_threshold: float = 0.85


class ExperimentMeta(BaseModel):
    """Experiment metadata."""

    name: str
    description: str = ""


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment: ExperimentMeta
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        return yaml.dump(
            self.model_dump(), default_flow_style=False, sort_keys=False
        )

    def save(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())


# Reasoning models that use reasoning_effort instead of temperature
REASONING_MODELS = frozenset(
    [
        "gpt-5-nano",
        "gpt-5-mini",
        # Add other reasoning models here as needed
    ]
)

# Default reasoning effort for reasoning models
DEFAULT_REASONING_EFFORT = "medium"


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model that uses reasoning_effort."""
    return model in REASONING_MODELS


def supports_temperature(model: str) -> bool:
    """Check if a model supports the temperature parameter."""
    return model not in REASONING_MODELS


def get_model_kwargs(
    model: str,
    temperature: float = 0.0,
    model_config: Optional["ModelConfig"] = None,
) -> dict:
    """
    Get appropriate kwargs for OpenAI API calls based on model type.

    Non-reasoning models (gpt-4o, gpt-4.1-mini): use temperature=0
    Reasoning models (gpt-5-mini, gpt-5-nano): use reasoning_effort from
    ModelConfig if set, otherwise DEFAULT_REASONING_EFFORT.
    """
    kwargs = {"model": model}
    if model in REASONING_MODELS:
        effort = DEFAULT_REASONING_EFFORT
        if (
            model_config is not None
            and model_config.reasoning_effort is not None
        ):
            effort = model_config.reasoning_effort
        kwargs["reasoning_effort"] = effort
    else:
        kwargs["temperature"] = temperature
    return kwargs


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


def create_default_config(name: str = "default") -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        experiment=ExperimentMeta(
            name=name, description="Default configuration"
        )
    )
