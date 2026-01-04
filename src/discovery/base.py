"""
Base classes and dataclasses for discovery.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Optional


@dataclass
class Affordance:
    """Represents an action or property affordance."""
    name: str
    uri: str
    schema: dict = field(default_factory=dict)


@dataclass
class Artifact:
    """Represents a discovered artifact with its affordances."""
    name: str
    uri: str
    workspace: str
    actions: list[Affordance] = field(default_factory=list)
    properties: list[Affordance] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "uri": self.uri,
            "workspace": self.workspace,
            "actions": [{"name": a.name, "uri": a.uri, "schema": a.schema} for a in self.actions],
            "properties": [{"name": p.name, "uri": p.uri, "schema": p.schema} for p in self.properties],
        }


def _format_param_constraints(param_schema: dict) -> str:
    """Format parameter constraints (type, min, max, enum) for display."""
    parts = []

    # Type
    param_type = param_schema.get("type", "any")
    parts.append(param_type)

    # Min/Max constraints
    if "minimum" in param_schema and "maximum" in param_schema:
        parts.append(f"range: {param_schema['minimum']}-{param_schema['maximum']}")
    elif "minimum" in param_schema:
        parts.append(f"min: {param_schema['minimum']}")
    elif "maximum" in param_schema:
        parts.append(f"max: {param_schema['maximum']}")

    # Enum values
    if "enum" in param_schema:
        enum_vals = param_schema["enum"]
        if len(enum_vals) <= 5:
            parts.append(f"values: {enum_vals}")
        else:
            parts.append(f"values: {enum_vals[:3]}... ({len(enum_vals)} options)")

    return ", ".join(parts)


def _format_action_schema(schema: dict) -> str:
    """Format action input schema with parameter types and constraints."""
    if not schema:
        return ""

    # Handle object schema with properties
    properties = schema.get("properties", {})
    if not properties:
        # Simple schema (not object type)
        if schema.get("type"):
            return f" (input: {_format_param_constraints(schema)})"
        return ""

    # Format each parameter with its constraints
    param_parts = []
    for param_name, param_schema in properties.items():
        constraint_str = _format_param_constraints(param_schema)
        param_parts.append(f"{param_name}: {constraint_str}")

    if param_parts:
        return f"\n      Parameters: {'; '.join(param_parts)}"
    return ""


def _format_property_type(schema: dict) -> str:
    """Format property output schema type."""
    if not schema:
        return ""

    prop_type = schema.get("type")
    if prop_type:
        extra = []
        if "enum" in schema:
            enum_vals = schema["enum"]
            if len(enum_vals) <= 5:
                extra.append(f"values: {enum_vals}")
            else:
                extra.append(f"{len(enum_vals)} possible values")

        if extra:
            return f" (type: {prop_type}, {', '.join(extra)})"
        return f" (type: {prop_type})"
    return ""


@dataclass
class CapabilityModel:
    """Complete model of discovered environment capabilities."""
    entry_point: str
    workspaces: dict[str, list[str]] = field(default_factory=dict)  # workspace_uri -> artifact_uris
    artifacts: dict[str, Artifact] = field(default_factory=dict)  # artifact_uri -> Artifact

    def to_summary(self) -> str:
        """Generate a concise summary for the LLM prompt."""
        lines = ["# Available Devices and Capabilities\n"]

        for ws_uri, artifact_uris in self.workspaces.items():
            ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
            lines.append(f"\n## {ws_name}")

            for art_uri in artifact_uris:
                art = self.artifacts.get(art_uri)
                if not art:
                    continue

                lines.append(f"\n### {art.name}")
                lines.append(f"URI: `{art_uri}`")

                if art.actions:
                    lines.append("**Actions:**")
                    for action in art.actions:
                        schema_info = _format_action_schema(action.schema)
                        lines.append(f"  - `{action.name}`: `{action.uri}`{schema_info}")

                if art.properties:
                    lines.append("**Properties:**")
                    for prop in art.properties:
                        type_info = _format_property_type(prop.schema)
                        lines.append(f"  - `{prop.name}`: `{prop.uri}`{type_info}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dict for tracing (summary stats)."""
        return {
            "entry_point": self.entry_point,
            "workspace_count": len(self.workspaces),
            "artifact_count": len(self.artifacts),
            "action_count": sum(len(a.actions) for a in self.artifacts.values()),
            "property_count": sum(len(a.properties) for a in self.artifacts.values()),
        }

    def to_full_dict(self) -> dict:
        """Convert to full dict with all discovered capabilities."""
        workspaces_data = {}
        for ws_uri, artifact_uris in self.workspaces.items():
            ws_name = ws_uri.split("/")[-1].replace("#workspace", "")
            artifacts_data = []
            for art_uri in artifact_uris:
                art = self.artifacts.get(art_uri)
                if art:
                    artifacts_data.append(art.to_dict())
            workspaces_data[ws_name] = artifacts_data

        return {
            "entry_point": self.entry_point,
            "workspaces": workspaces_data,
            "stats": self.to_dict(),
        }

    def get_all_property_uris(self) -> list[str]:
        """Get all property URIs from all artifacts."""
        uris = []
        for artifact in self.artifacts.values():
            for prop in artifact.properties:
                uris.append(prop.uri)
        return uris

    def get_artifact_by_name(self, name: str) -> Optional["Artifact"]:
        """Find artifact by name (case-insensitive)."""
        name_lower = name.lower()
        for artifact in self.artifacts.values():
            if artifact.name.lower() == name_lower:
                return artifact
        return None


@dataclass
class EnvironmentState:
    """Snapshot of property values from the environment."""
    property_values: dict[str, Any] = field(default_factory=dict)  # property_uri -> value
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: dict[str, str] = field(default_factory=dict)  # property_uri -> error message

    def get(self, property_uri: str) -> Any:
        """Get value for a property URI."""
        return self.property_values.get(property_uri)

    def to_summary(self) -> str:
        """Generate a summary for LLM prompt."""
        if not self.property_values:
            return "# Current State\n\nNo state information gathered."

        lines = ["# Current State\n"]
        for uri, value in self.property_values.items():
            # Extract readable name from URI
            name = uri.split("/")[-1] if "/" in uri else uri
            lines.append(f"- `{name}`: `{value}` ({uri})")

        if self.errors:
            lines.append("\n## Errors:")
            for uri, error in self.errors.items():
                lines.append(f"- `{uri}`: {error}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "property_values": self.property_values,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "property_count": len(self.property_values),
            "error_count": len(self.errors),
        }


@dataclass
class DiscoveryResult:
    """Complete discovery output combining affordances and state."""
    affordances: CapabilityModel
    state: EnvironmentState
    exploration_trace: list[dict] = field(default_factory=list)  # For agentic affordance discovery
    state_trace: list[dict] = field(default_factory=list)  # For agentic state discovery

    def to_prompt_context(self) -> str:
        """Format for injection into LLM prompt."""
        parts = [self.affordances.to_summary()]

        if self.state.property_values:
            parts.append("\n\n" + self.state.to_summary())

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing."""
        result = {
            "affordances": self.affordances.to_full_dict(),
            "state": self.state.to_dict(),
        }
        if self.exploration_trace:
            result["exploration_trace"] = self.exploration_trace
        if self.state_trace:
            result["state_trace"] = self.state_trace
        return result


class AffordanceDiscoveryStrategy(Protocol):
    """Protocol for affordance discovery strategies."""

    def discover(
        self,
        entry_point: str,
        goal: Optional[str] = None,
    ) -> CapabilityModel:
        """Discover affordances from the environment."""
        ...


class StateGatheringStrategy(Protocol):
    """Protocol for state gathering strategies."""

    def gather(
        self,
        affordances: CapabilityModel,
        goal: Optional[str] = None,
    ) -> EnvironmentState:
        """Gather state from the environment."""
        ...
