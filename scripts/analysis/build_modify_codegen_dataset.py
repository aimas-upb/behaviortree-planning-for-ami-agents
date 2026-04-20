#!/usr/bin/env python3
"""
Build a HomeBench fine-tuning dataset for modify-only BT code generation.

The generated examples mirror the runtime interface of
``NeuroSymbolicRunner._build_modify_tree``:
  - input: structured modify-only goal + discovery/state context
  - target: deterministic Python BT code following the modify-only template

The builder is fully offline. It uses:
  - raw HomeBench rows for source metadata
  - converted HomeBench rows for executable ground-truth affordances/tests
  - local HMAS TTL/state files for semantic types, schemas, and current values

Outputs:
  - ``all_examples.jsonl``
  - ``official/{train,valid,test}.jsonl``
  - ``home_disjoint/{train,valid,test}.jsonl``
  - ``manifest.json``
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional

import py_trees
from behavior_trees.affordance_nodes import (
    ActionAffordanceNode,
    ComparisonPropertyConditionNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
)
from rdflib import RDF, Graph, Namespace, URIRef

from scripts.common import resolve_repo_path

TD = Namespace("https://www.w3.org/2019/wot/td#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")
EX = Namespace("http://example.org/")

FORBIDDEN_PATTERNS = [
    r"\bos\.(remove|unlink|rmdir|rmtree|system|popen|exec|spawn)",
    r"\bshutil\.(rmtree|move|copy|copytree)",
    r"\bsubprocess\.",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r'\bopen\s*\([^)]*["\']w',
    r"\brm\s+-rf",
    r"\bimport\s+subprocess",
    r"\bimport\s+shutil",
    r"\bimport\s+os\b",
    r"\bfrom\s+os\s+import",
    r"\bimport\s+sys\b",
    r"\bfrom\s+sys\s+import",
    r"\b__builtins__",
    r"\b__class__",
    r"\b__bases__",
    r"\b__subclasses__",
    r"\b__mro__",
    r"\b__globals__",
    r"\b__code__",
]


def _load_discovery_base_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "src" / "discovery" / "base.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_modify_codegen_discovery_base", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load discovery base module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_discovery_base = _load_discovery_base_module()
Affordance = _discovery_base.Affordance
Artifact = _discovery_base.Artifact
CapabilityModel = _discovery_base.CapabilityModel
DiscoveryResult = _discovery_base.DiscoveryResult
EnvironmentState = _discovery_base.EnvironmentState

SOURCE_SPECS = (
    {
        "name": "train_data_part1",
        "official_split": "train",
        "raw_path": "data/homebench/raw/train_data_part1.jsonl",
        "converted_path": "data/homebench/converted/train_data_part1.json",
    },
    {
        "name": "train_data_part2",
        "official_split": "train",
        "raw_path": "data/homebench/raw/train_data_part2.jsonl",
        "converted_path": "data/homebench/converted/train_data_part2.json",
    },
    {
        "name": "valid_data",
        "official_split": "valid",
        "raw_path": "data/homebench/raw/valid_data.jsonl",
        "converted_path": "data/homebench/converted/valid_data.json",
    },
    {
        "name": "test_data",
        "official_split": "test",
        "raw_path": "data/homebench/raw/test_data.jsonl",
        "converted_path": "data/homebench/converted/test_data.json",
    },
)

DEVICE_ALIASES = {
    "AirConditioner": {"air conditioner", "ac"},
    "AirPurifiers": {"air purifier", "air purifiers"},
    "Aromatherapy": {"aromatherapy", "aromatherapy device"},
    "Blinds": {"blind", "blinds"},
    "Curtain": {"curtain", "curtains"},
    "Dehumidifier": {"dehumidifier", "dehumidifiers"},
    "Dehumidifiers": {"dehumidifier", "dehumidifiers"},
    "Fan": {"fan"},
    "GarageDoor": {"garage door", "door"},
    "Heating": {"heating", "heater", "heating device", "heating unit"},
    "Humidifier": {"humidifier"},
    "Light": {"light", "lights", "lamp"},
    "MediaPlayer": {"media player", "player", "media"},
    "Trash": {"trash", "bin"},
    "WaterHeater": {"water heater", "heater"},
}

PARAM_ALIASES = {
    "brightness": {"brightness"},
    "degree": {"degree", "degrees", "angle"},
    "fan_speed": {"fan speed", "speed"},
    "intensity": {"intensity"},
    "interval": {"interval", "minute", "minutes", "second", "seconds"},
    "mode": {"mode"},
    "speed": {"speed"},
    "swing": {"swing"},
    "temperature": {"temperature", "degrees", "degree"},
    "volume": {"volume"},
}

ACTION_STARTS = [
    "turn off",
    "turn on",
    "turn down",
    "turn up",
    "increase",
    "decrease",
    "lower",
    "raise",
    "reduce",
    "boost",
    "fine tune",
    "fine-tune",
    "enhance",
    "adjust",
    "change",
    "configure",
    "modify",
    "update",
    "switch",
    "set",
    "open",
    "close",
    "pause",
    "play",
    "stop",
]
CONJUNCTION_SPLIT_RE = re.compile(
    r"\s+(?:and|then|followed by|as well as)\s+(?=(?:"
    + "|".join(
        re.escape(v) for v in sorted(ACTION_STARTS, key=len, reverse=True)
    )
    + r")\b)",
    re.IGNORECASE,
)
BY_VALUE_RE = re.compile(
    r"\bby\s+(-?\d+(?:\.\d+)?)\s*(%|percent|percentage|degrees?|degree|minutes?|minute|seconds?|second)?\b",
    re.IGNORECASE,
)
INCREASE_RE = re.compile(
    r"\b(increase|raise|boost|turn up|enhance|fine tune|fine-tune)\b",
    re.IGNORECASE,
)
DECREASE_RE = re.compile(
    r"\b(decrease|lower|reduce|dim|turn down)\b",
    re.IGNORECASE,
)


@dataclass
class ClauseCandidate:
    index: int
    text: str
    normalized_text: str
    direction: Optional[str]
    delta_value: Optional[int | float]
    delta_unit: Optional[str]

    @property
    def is_relative(self) -> bool:
        return self.direction is not None and self.delta_value is not None


@dataclass
class PreparedOutput:
    output_index: int
    affordance: str
    property_url: str
    expected_value: int | float
    current_value: int | float
    parameter_name: str
    home_id: int
    action_meta: dict[str, Any]
    property_meta: dict[str, Any]
    workspace_type: str
    artifact_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a modify-only HomeBench codegen dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="data/homebench/modify_codegen",
        help="Directory where dataset files will be written.",
    )
    parser.add_argument(
        "--ttl-dir",
        default="data/homebench/hmas/home_description",
        help="Directory containing HMAS home TTL files.",
    )
    parser.add_argument(
        "--state-dir",
        default="data/homebench/hmas/home_description",
        help="Directory containing HMAS home state JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic home-disjoint splits.",
    )
    parser.add_argument(
        "--home-train-count",
        type=int,
        default=80,
        help="Number of homes for the home-disjoint training split.",
    )
    parser.add_argument(
        "--home-valid-count",
        type=int,
        default=10,
        help="Number of homes for the home-disjoint validation split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on processed rows per source file for debugging.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[spec["name"] for spec in SOURCE_SPECS],
        help="Subset of source spec names to process.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_home_id_from_test_id(test_id: str) -> int:
    match = re.match(r"home(\d+)_", test_id)
    if not match:
        raise ValueError(f"Could not parse home id from {test_id!r}")
    return int(match.group(1))


def camel_to_words(name: str) -> list[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|\d+", name)
    return [p.lower() for p in parts if p]


def camel_to_text(name: str) -> str:
    return " ".join(camel_to_words(name))


def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def slugify_identifier(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "value"
    if slug[0].isdigit():
        slug = f"v_{slug}"
    return slug.lower()


def canonical_number(value: int | float) -> int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def numeric_value(value: Any) -> Optional[int | float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return canonical_number(value)
    return None


def values_match(expected: int | float, actual: int | float) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        return abs(float(expected) - float(actual)) < 1e-9
    return expected == actual


def sha1_hexdigest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def semantic_type_with_prefix(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return f"ex:{name}"


def action_verb_matches(candidate_text: str, semantic_type: str) -> bool:
    """Check if the candidate text matches the semantic type."""
    lowered = candidate_text.lower()
    if semantic_type == "Light":
        return "light" in lowered or "lamp" in lowered
    return True


def room_phrase_for_type(workspace_type: str) -> str:
    return camel_to_text(workspace_type)


def device_aliases_for_type(artifact_type: str) -> set[str]:
    aliases = set(DEVICE_ALIASES.get(artifact_type, set()))
    base = camel_to_text(artifact_type)
    if base:
        aliases.add(base)
        if base.endswith("s"):
            aliases.add(base[:-1])
        else:
            aliases.add(f"{base}s")
    return {alias.lower() for alias in aliases}


def param_aliases_for_name(parameter_name: str) -> set[str]:
    aliases = set(PARAM_ALIASES.get(parameter_name, set()))
    aliases.add(parameter_name.replace("_", " ").lower())
    if "_" in parameter_name:
        aliases.add(parameter_name.split("_", 1)[-1].lower())
    return {alias.lower() for alias in aliases}


def split_into_clauses(text: str) -> list[str]:
    text = normalize_whitespace(text.strip())
    text = text.rstrip(".")
    if not text:
        return []

    parts = [text]
    for splitter in (r"\s*;\s*", r"\s*,\s*"):
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(
                piece.strip()
                for piece in re.split(splitter, part)
                if piece.strip()
            )
        parts = next_parts

    final_parts: list[str] = []
    for part in parts:
        final_parts.extend(
            piece.strip()
            for piece in CONJUNCTION_SPLIT_RE.split(part)
            if piece.strip()
        )

    return final_parts


def parse_relative_clause(text: str) -> ClauseCandidate:
    normalized = normalize_whitespace(text.lower())
    direction: Optional[str] = None
    if INCREASE_RE.search(normalized):
        direction = "increase"
    elif DECREASE_RE.search(normalized):
        direction = "decrease"

    delta_value: Optional[int | float] = None
    delta_unit: Optional[str] = None
    match = BY_VALUE_RE.search(normalized)
    if match and direction is not None:
        raw_value = float(match.group(1))
        delta_value = canonical_number(raw_value)
        unit = match.group(2)
        if unit:
            delta_unit = unit.lower()

    return ClauseCandidate(
        index=-1,
        text=text.strip().rstrip(".") + ".",
        normalized_text=normalized,
        direction=direction,
        delta_value=delta_value,
        delta_unit=delta_unit,
    )


def candidate_score(candidate: ClauseCandidate, output: PreparedOutput) -> int:
    if not candidate.is_relative:
        return -1

    score = 0
    room_text = room_phrase_for_type(output.workspace_type)
    if room_text and room_text in candidate.normalized_text:
        score += 4

    device_hits = [
        alias
        for alias in device_aliases_for_type(output.artifact_type)
        if alias in candidate.normalized_text
    ]
    if device_hits:
        score += 3

    param_hits = [
        alias
        for alias in param_aliases_for_name(output.parameter_name)
        if alias in candidate.normalized_text
    ]
    if param_hits:
        score += 2

    if action_verb_matches(candidate.normalized_text, output.artifact_type):
        score += 1

    return score


class LocalHomeGraphStore:
    """Offline HMAS metadata/state access backed by local TTL and JSON files."""

    def __init__(self, ttl_dir: Path, state_dir: Path):
        self.ttl_dir = ttl_dir
        self.state_dir = state_dir
        self._graph_cache: dict[int, Graph] = {}
        self._state_cache: dict[int, dict[str, Any]] = {}
        self._action_cache: dict[str, dict[str, Any]] = {}
        self._property_cache: dict[str, dict[str, Any]] = {}

    def _graph(self, home_id: int) -> Graph:
        if home_id not in self._graph_cache:
            path = self.ttl_dir / f"home_{home_id}.ttl"
            graph = Graph()
            graph.parse(path)
            self._graph_cache[home_id] = graph
        return self._graph_cache[home_id]

    def _state(self, home_id: int) -> dict[str, Any]:
        if home_id not in self._state_cache:
            path = self.state_dir / f"home_{home_id}_state.json"
            self._state_cache[home_id] = json.loads(path.read_text())
        return self._state_cache[home_id]

    @staticmethod
    def artifact_subject_uri_from_action_url(action_url: str) -> str:
        return action_url.rsplit("/", 1)[0] + "#artifact"

    @staticmethod
    def artifact_subject_uri_from_property_url(property_url: str) -> str:
        return property_url.split("/properties/", 1)[0] + "#artifact"

    @staticmethod
    def workspace_subject_uri_from_artifact_uri(artifact_uri: str) -> str:
        return artifact_uri.split("/artifacts/", 1)[0] + "#workspace"

    @staticmethod
    def _extract_semantic_type(graph: Graph, subject: URIRef) -> Optional[str]:
        prefix = str(EX)
        for node_type in graph.objects(subject, RDF.type):
            text = str(node_type)
            if text.startswith(prefix):
                return text[len(prefix) :]
        return None

    def _parse_schema(
        self, graph: Graph, schema_node: Optional[URIRef]
    ) -> dict[str, Any]:
        if not schema_node:
            return {}

        schema: dict[str, Any] = {}
        for node_type in graph.objects(schema_node, RDF.type):
            text = str(node_type)
            if "IntegerSchema" in text:
                schema["type"] = "integer"
            elif "StringSchema" in text:
                schema["type"] = "string"
            elif "NumberSchema" in text:
                schema["type"] = "number"
            elif "BooleanSchema" in text:
                schema["type"] = "boolean"
            elif "ObjectSchema" in text:
                schema["type"] = "object"
            elif "ArraySchema" in text:
                schema["type"] = "array"

        minimum = graph.value(schema_node, JSONSCHEMA.minimum)
        maximum = graph.value(schema_node, JSONSCHEMA.maximum)
        if minimum is not None:
            schema["minimum"] = canonical_number(float(minimum))
        if maximum is not None:
            schema["maximum"] = canonical_number(float(maximum))

        enum_values = [
            str(v) for v in graph.objects(schema_node, JSONSCHEMA.enum)
        ]
        if enum_values:
            schema["enum"] = enum_values

        properties: dict[str, Any] = {}
        for prop_node in graph.objects(schema_node, JSONSCHEMA.properties):
            prop_name = graph.value(prop_node, JSONSCHEMA.propertyName)
            if prop_name:
                properties[str(prop_name)] = self._parse_schema(
                    graph, prop_node
                )
        if properties:
            schema["properties"] = properties

        required_values = [
            str(v) for v in graph.objects(schema_node, JSONSCHEMA.required)
        ]
        if required_values:
            schema["required"] = required_values

        items_node = graph.value(schema_node, JSONSCHEMA.items)
        if items_node:
            schema["items"] = self._parse_schema(graph, items_node)

        return schema

    def _artifact_title(self, graph: Graph, artifact_ref: URIRef) -> str:
        title = graph.value(artifact_ref, TD.title)
        if title:
            return str(title)
        return str(artifact_ref).split("/")[-1].replace("#artifact", "")

    def _home_id_from_url(self, url: str) -> int:
        match = re.search(r"/workspaces/home(\d+)", url)
        if not match:
            raise ValueError(f"Could not parse home id from {url}")
        return int(match.group(1))

    def workspace_semantic_type(self, workspace_uri: str) -> str:
        home_id = self._home_id_from_url(workspace_uri)
        graph = self._graph(home_id)
        semantic_type = self._extract_semantic_type(
            graph, URIRef(workspace_uri)
        )
        if semantic_type is None:
            raise ValueError(f"No workspace semantic type for {workspace_uri}")
        return semantic_type

    def artifact_semantic_type(self, artifact_uri: str) -> str:
        home_id = self._home_id_from_url(artifact_uri)
        graph = self._graph(home_id)
        semantic_type = self._extract_semantic_type(graph, URIRef(artifact_uri))
        if semantic_type is None:
            raise ValueError(f"No artifact semantic type for {artifact_uri}")
        return semantic_type

    def list_actions_for_artifact(
        self, artifact_uri: str
    ) -> list[dict[str, Any]]:
        home_id = self._home_id_from_url(artifact_uri)
        graph = self._graph(home_id)
        artifact_ref = URIRef(artifact_uri)

        actions: list[dict[str, Any]] = []
        for affordance in graph.objects(artifact_ref, TD.hasActionAffordance):
            form = graph.value(affordance, TD.hasForm)
            target = graph.value(form, HCTL.hasTarget) if form else None
            input_schema = graph.value(affordance, TD.hasInputSchema)
            title = graph.value(affordance, TD.title)
            actions.append(
                {
                    "name": str(title) if title else "",
                    "uri": str(target) if target else "",
                    "schema": self._parse_schema(graph, input_schema),
                    "semantic_type": self._extract_semantic_type(
                        graph, affordance
                    ),
                }
            )
        return actions

    def list_properties_for_artifact(
        self, artifact_uri: str
    ) -> list[dict[str, Any]]:
        home_id = self._home_id_from_url(artifact_uri)
        graph = self._graph(home_id)
        artifact_ref = URIRef(artifact_uri)

        properties: list[dict[str, Any]] = []
        for affordance in graph.objects(artifact_ref, TD.hasPropertyAffordance):
            form = graph.value(affordance, TD.hasForm)
            target = graph.value(form, HCTL.hasTarget) if form else None
            output_schema = graph.value(affordance, TD.hasOutputSchema)
            title = graph.value(affordance, TD.title)
            properties.append(
                {
                    "name": str(title) if title else "",
                    "uri": str(target) if target else "",
                    "schema": self._parse_schema(graph, output_schema),
                    "semantic_type": self._extract_semantic_type(
                        graph, affordance
                    ),
                }
            )
        return properties

    def action_metadata(self, action_url: str) -> dict[str, Any]:
        if action_url in self._action_cache:
            return self._action_cache[action_url]

        home_id = self._home_id_from_url(action_url)
        graph = self._graph(home_id)
        target = URIRef(action_url)

        for form in graph.subjects(HCTL.hasTarget, target):
            for affordance in graph.subjects(TD.hasForm, form):
                title = graph.value(affordance, TD.title)
                input_schema = graph.value(affordance, TD.hasInputSchema)
                artifact_refs = list(
                    graph.subjects(TD.hasActionAffordance, affordance)
                )
                artifact_uri = (
                    str(artifact_refs[0])
                    if artifact_refs
                    else action_url.rsplit("/", 1)[0] + "#artifact"
                )
                workspace_uri = self.workspace_subject_uri_from_artifact_uri(
                    artifact_uri
                )
                metadata = {
                    "name": str(title) if title else "",
                    "uri": action_url,
                    "schema": self._parse_schema(graph, input_schema),
                    "semantic_type": self._extract_semantic_type(
                        graph, affordance
                    ),
                    "artifact_uri": artifact_uri,
                    "workspace_uri": workspace_uri,
                }
                self._action_cache[action_url] = metadata
                return metadata

        raise ValueError(f"No action affordance found for {action_url}")

    def property_metadata(self, property_url: str) -> dict[str, Any]:
        if property_url in self._property_cache:
            return self._property_cache[property_url]

        home_id = self._home_id_from_url(property_url)
        graph = self._graph(home_id)
        target = URIRef(property_url)

        for form in graph.subjects(HCTL.hasTarget, target):
            for affordance in graph.subjects(TD.hasForm, form):
                title = graph.value(affordance, TD.title)
                output_schema = graph.value(affordance, TD.hasOutputSchema)
                artifact_refs = list(
                    graph.subjects(TD.hasPropertyAffordance, affordance)
                )
                artifact_uri = (
                    str(artifact_refs[0])
                    if artifact_refs
                    else property_url.split("/properties/", 1)[0] + "#artifact"
                )
                metadata = {
                    "name": str(title) if title else "",
                    "uri": property_url,
                    "schema": self._parse_schema(graph, output_schema),
                    "semantic_type": self._extract_semantic_type(
                        graph, affordance
                    ),
                    "artifact_uri": artifact_uri,
                }
                self._property_cache[property_url] = metadata
                return metadata

        raise ValueError(f"No property affordance found for {property_url}")

    def current_property_value(self, property_url: str) -> Any:
        home_id = self._home_id_from_url(property_url)
        state = self._state(home_id)
        artifact_uri = self.artifact_subject_uri_from_property_url(property_url)
        prop_name = property_url.rsplit("/", 1)[-1]
        return state.get(artifact_uri, {}).get(prop_name)

    def build_discovery_result(
        self, home_id: int, action_urls: list[str]
    ) -> DiscoveryResult:
        entry_point = (
            f"http://localhost:8080/workspaces/home{home_id}#workspace"
        )
        capability_model = CapabilityModel(entry_point=entry_point)
        state = EnvironmentState(
            timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat()
        )

        for action_url in action_urls:
            action_meta = self.action_metadata(action_url)
            artifact_uri = action_meta["artifact_uri"]
            workspace_uri = action_meta["workspace_uri"]

            workspace = capability_model.get_or_create_workspace(
                workspace_uri,
                semantic_type=self.workspace_semantic_type(workspace_uri),
            )
            if artifact_uri not in workspace.artifact_uris:
                workspace.artifact_uris.append(artifact_uri)

            if artifact_uri in capability_model.artifacts:
                continue

            actions = [
                Affordance(
                    name=action["name"],
                    uri=action["uri"],
                    schema=action["schema"],
                    semantic_type=action["semantic_type"],
                )
                for action in self.list_actions_for_artifact(artifact_uri)
            ]
            properties = [
                Affordance(
                    name=prop["name"],
                    uri=prop["uri"],
                    schema=prop["schema"],
                    semantic_type=prop["semantic_type"],
                )
                for prop in self.list_properties_for_artifact(artifact_uri)
            ]
            capability_model.artifacts[artifact_uri] = Artifact(
                name=self._artifact_title(
                    self._graph(home_id), URIRef(artifact_uri)
                ),
                uri=artifact_uri,
                workspace=workspace_uri,
                actions=actions,
                properties=properties,
                semantic_type=self.artifact_semantic_type(artifact_uri),
            )

        for artifact in capability_model.artifacts.values():
            for prop in artifact.properties:
                current_value = self.current_property_value(prop.uri)
                if current_value is not None:
                    state.property_values[prop.uri] = current_value

        return DiscoveryResult(affordances=capability_model, state=state)


def validate_candidate_against_output(
    candidate: ClauseCandidate, output: PreparedOutput
) -> bool:
    if not candidate.is_relative:
        return False

    raw_new_value = (
        output.current_value + candidate.delta_value
        if candidate.direction == "increase"
        else output.current_value - candidate.delta_value
    )

    param_schema = (
        output.action_meta.get("schema", {})
        .get("properties", {})
        .get(output.parameter_name, {})
    )
    minimum = param_schema.get("minimum")
    maximum = param_schema.get("maximum")
    new_value = raw_new_value
    if minimum is not None:
        new_value = max(minimum, new_value)
    if maximum is not None:
        new_value = min(maximum, new_value)

    if param_schema.get("type") == "integer":
        new_value = int(round(float(new_value)))
    elif param_schema.get("type") == "number":
        new_value = float(new_value)

    return values_match(output.expected_value, canonical_number(new_value))


def make_intent_record(
    candidate: ClauseCandidate,
    output: PreparedOutput,
) -> dict[str, Any]:
    param_schema = (
        output.action_meta.get("schema", {})
        .get("properties", {})
        .get(output.parameter_name, {})
    )
    minimum = param_schema.get("minimum")
    maximum = param_schema.get("maximum")

    unclamped = (
        output.current_value + candidate.delta_value
        if candidate.direction == "increase"
        else output.current_value - candidate.delta_value
    )
    clamped = unclamped
    if minimum is not None:
        clamped = max(minimum, clamped)
    if maximum is not None:
        clamped = min(maximum, clamped)

    return {
        "text_intent": candidate.text,
        "text_intent_source": "aligned_clause",
        "clause_index": candidate.index,
        "delta_direction": candidate.direction,
        "delta_unit": candidate.delta_unit,
        "clamp_applied": not values_match(unclamped, clamped),
        "current_value": output.current_value,
        "expected_value": output.expected_value,
        "action": {
            "affordance_type": semantic_type_with_prefix(
                output.action_meta["semantic_type"]
            ),
            "verb": "modify",
            "parameter": output.parameter_name,
            "value": str(candidate.delta_value),
        },
        "target": {
            "artifact_type": semantic_type_with_prefix(output.artifact_type),
            "workspace_type": semantic_type_with_prefix(output.workspace_type),
        },
        "resolved": {
            "action_url": output.affordance,
            "property_url": output.property_url,
            "artifact_uri": output.action_meta["artifact_uri"],
            "workspace_uri": output.action_meta["workspace_uri"],
        },
        "action_schema": output.action_meta["schema"],
        "property_schema": output.property_meta["schema"],
        "source_output_index": output.output_index,
    }


def build_runtime_modify_goal(intents: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, intent in enumerate(intents, start=1):
        lines.append(f"**Intent {idx}**")
        lines.append(f"text_intent: {intent['text_intent']}")
        lines.append(
            f"action.affordance_type: {intent['action']['affordance_type']}"
        )
        lines.append(f"action.verb: {intent['action']['verb']}")
        lines.append(f"action.parameter: {intent['action']['parameter']}")
        lines.append(f"action.value: {intent['action']['value']}")
        lines.append(
            f"target.artifact_type: {intent['target']['artifact_type']}"
        )
        lines.append(
            f"target.workspace_type: {intent['target']['workspace_type']}"
        )
        lines.append(f"resolved.action_url: {intent['resolved']['action_url']}")
        lines.append(
            f"resolved.artifact_uri: {intent['resolved']['artifact_uri']}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_numeric(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def compute_expression(intent: dict[str, Any]) -> str:
    parameter = intent["action"]["parameter"]
    delta = float(intent["action"]["value"])
    delta = int(delta) if delta.is_integer() else delta
    operator = "+" if intent["delta_direction"] == "increase" else "-"
    expr = f"current {operator} {format_numeric(delta)}"

    param_schema = (
        intent["action_schema"].get("properties", {}).get(parameter, {})
    )
    minimum = param_schema.get("minimum")
    maximum = param_schema.get("maximum")

    if minimum is not None and maximum is not None:
        return (
            f"max({format_numeric(minimum)}, "
            f"min({format_numeric(maximum)}, {expr}))"
        )
    if minimum is not None:
        return f"max({format_numeric(minimum)}, {expr})"
    if maximum is not None:
        return f"min({format_numeric(maximum)}, {expr})"
    return expr


def synthesize_target_code(intents: list[dict[str, Any]]) -> str:
    code_lines: list[str] = []
    sequence_vars: list[str] = []

    for index, intent in enumerate(intents, start=1):
        workspace_type = intent["target"]["workspace_type"].split(":", 1)[-1]
        artifact_type = intent["target"]["artifact_type"].split(":", 1)[-1]
        parameter = intent["action"]["parameter"]

        key = slugify_identifier(
            f"{camel_to_text(workspace_type)}_{camel_to_text(artifact_type)}_{parameter}_value_{index}"
        )
        class_suffix = (
            f"{workspace_type}{artifact_type}{snake_to_camel(parameter)}{index}"
        )
        read_name = f"Read{snake_to_camel(parameter)}{index}"
        compute_name = f"Compute{class_suffix}"
        action_name = f"Set{snake_to_camel(parameter)}{index}"
        sequence_name = (
            f"Modify{workspace_type}{artifact_type}{snake_to_camel(parameter)}"
        )

        property_url = intent["resolved"]["property_url"]
        action_url = intent["resolved"]["action_url"]
        expression = compute_expression(intent)
        param_type = (
            intent["action_schema"]
            .get("properties", {})
            .get(parameter, {})
            .get("type")
        )

        code_lines.extend(
            [
                f"read_node_{index} = PropertyAffordanceNode(",
                f'    name="{read_name}",',
                f'    property_url="{property_url}",',
                f'    result_key="{key}"',
                ")",
                "",
                f"class {compute_name}(py_trees.behaviour.Behaviour):",
                "    def __init__(self):",
                f'        super().__init__(name="{compute_name}")',
                "        self.blackboard = self.attach_blackboard_client(",
                f'            name="{compute_name}"',
                "        )",
                "        self.blackboard.register_key(",
                f'            key="{key}",',
                "            access=py_trees.common.Access.WRITE,",
                "        )",
                "",
                "    def update(self):",
                f"        current = self.blackboard.{key}",
                "        if current is None:",
                "            return py_trees.common.Status.FAILURE",
                f"        new_value = {expression}",
            ]
        )
        if param_type == "integer":
            code_lines.append("        new_value = int(new_value)")
        elif param_type == "number":
            code_lines.append("        new_value = float(new_value)")
        code_lines.extend(
            [
                f"        self.blackboard.{key} = new_value",
                "        return py_trees.common.Status.SUCCESS",
                "",
                f"compute_node_{index} = {compute_name}()",
                "",
                f"action_node_{index} = ActionAffordanceNode(",
                f'    name="{action_name}",',
                f'    action_url="{action_url}",',
                f'    parameter_keys={{"{parameter}": "{key}"}}',
                ")",
                "",
                f"seq_{index} = py_trees.composites.Sequence(",
                f'    name="{sequence_name}",',
                "    memory=True,",
                (
                    f"    children=[read_node_{index}, compute_node_{index}, "
                    f"action_node_{index}]"
                ),
                ")",
                "",
            ]
        )
        sequence_vars.append(f"seq_{index}")

    if len(sequence_vars) == 1:
        code_lines.append(f"tree = {sequence_vars[0]}")
    else:
        joined = ", ".join(sequence_vars)
        code_lines.extend(
            [
                "tree = py_trees.composites.Parallel(",
                '    name="ModifyActions",',
                "    policy=py_trees.common.ParallelPolicy.SuccessOnOne(),",
                f"    children=[{joined}]",
                ")",
            ]
        )

    return "\n".join(code_lines).rstrip() + "\n"


def _check_code_safety(code: str) -> None:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            raise ValueError(f"Forbidden pattern detected: {pattern}")

    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name != "py_trees":
                    raise ValueError(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name != "py_trees":
                    raise ValueError(
                        f"Forbidden import from: {node.module}"
                    )


def _execute_tree_code(code: str) -> py_trees.behaviour.Behaviour:
    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "True": True,
        "False": False,
        "None": None,
        "Exception": Exception,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "type": type,
        "super": super,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sum": sum,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "__build_class__": (
            __builtins__["__build_class__"]
            if isinstance(__builtins__, dict)
            else getattr(__builtins__, "__build_class__")
        ),
        "__import__": __import__,
    }
    safe_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "py_trees": py_trees,
        "ActionAffordanceNode": ActionAffordanceNode,
        "PropertyAffordanceNode": PropertyAffordanceNode,
        "PropertyConditionNode": PropertyConditionNode,
        "ComparisonPropertyConditionNode": ComparisonPropertyConditionNode,
    }
    local_namespace: dict[str, Any] = {}

    try:
        exec(code, safe_globals, local_namespace)
    except Exception as exc:
        raise ValueError(f"Code execution failed: {exc}") from exc

    if "tree" not in local_namespace:
        raise ValueError("Code must define 'tree' variable")

    tree = local_namespace["tree"]
    if isinstance(tree, py_trees.trees.BehaviourTree):
        tree = tree.root
    if not isinstance(tree, py_trees.behaviour.Behaviour):
        raise ValueError(
            f"'tree' must be a py_trees.behaviour.Behaviour, got {type(tree)}"
        )
    return tree


def validate_target_code(code: str) -> dict[str, Any]:
    _check_code_safety(code)
    tree = _execute_tree_code(code)
    return {
        "syntax_ok": True,
        "tree_build_ok": True,
        "tree_root_name": tree.name,
        "tree_root_type": type(tree).__name__,
        "tree_ir": tree_to_ir_silent(tree),
    }


def tree_to_ir_silent(node: py_trees.behaviour.Behaviour) -> dict[str, Any]:
    if isinstance(node, py_trees.composites.Sequence):
        return {
            "type": "sequence",
            "name": node.name,
            "children": [tree_to_ir_silent(child) for child in node.children],
        }

    if isinstance(node, py_trees.composites.Selector):
        return {
            "type": "selector",
            "name": node.name,
            "children": [tree_to_ir_silent(child) for child in node.children],
        }

    if isinstance(node, py_trees.composites.Parallel):
        policy = node.policy
        if isinstance(policy, py_trees.common.ParallelPolicy.SuccessOnAll):
            policy_name = "success_on_all"
        elif isinstance(policy, py_trees.common.ParallelPolicy.SuccessOnOne):
            policy_name = "success_on_one"
        else:
            policy_name = str(policy)
        return {
            "type": "parallel",
            "name": node.name,
            "policy": policy_name,
            "children": [tree_to_ir_silent(child) for child in node.children],
        }

    if isinstance(node, ComparisonPropertyConditionNode):
        result = {
            "type": "condition",
            "name": node.name,
            "property_url": node.property_url,
            "expected_value": node.expected_value,
            "operator": node.operator.value,
        }
        if node.value_path:
            result["value_path"] = node.value_path
        if node.negate:
            result["negate"] = True
        return result

    if isinstance(node, PropertyConditionNode):
        result = {
            "type": "condition",
            "name": node.name,
            "property_url": node.property_url,
            "expected_value": node.expected_value,
        }
        if node.value_path:
            result["value_path"] = node.value_path
        if node.negate:
            result["negate"] = True
        return result

    if isinstance(node, ActionAffordanceNode):
        result = {
            "type": "action",
            "name": node.name,
            "action_url": node.action_url,
        }
        if node.parameters:
            result["parameters"] = dict(node.parameters)
        if node.parameter_keys:
            result["parameter_keys"] = dict(node.parameter_keys)
        if node.result_key:
            result["result_key"] = node.result_key
        return result

    if isinstance(node, PropertyAffordanceNode):
        result = {
            "type": "property_read",
            "name": node.name,
            "property_url": node.property_url,
        }
        if node.result_key:
            result["result_key"] = node.result_key
        if node.property_name:
            result["property_name"] = node.property_name
        return result

    if isinstance(node, py_trees.composites.Composite):
        return {
            "type": "composite",
            "name": node.name,
            "children": [tree_to_ir_silent(child) for child in node.children],
        }

    return {
        "type": "unknown",
        "name": node.name,
        "class": type(node).__name__,
    }


def build_exact_signature(
    runtime_modify_goal: str, runtime_context: str, target_code: str
) -> str:
    return sha1_hexdigest(
        {
            "runtime_modify_goal": runtime_modify_goal,
            "runtime_context": runtime_context,
            "target_code": target_code,
        }
    )


def assign_home_splits(
    seed: int, home_train_count: int, home_valid_count: int
) -> dict[int, str]:
    homes = list(range(100))
    rng = random.Random(seed)
    rng.shuffle(homes)

    if home_train_count + home_valid_count > len(homes):
        raise ValueError("Home split sizes exceed available homes")

    home_split: dict[int, str] = {}
    for home_id in homes[:home_train_count]:
        home_split[home_id] = "train"
    for home_id in homes[
        home_train_count : home_train_count + home_valid_count
    ]:
        home_split[home_id] = "valid"
    for home_id in homes[home_train_count + home_valid_count :]:
        home_split[home_id] = "test"
    return home_split


class DatasetWriters:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "official").mkdir(parents=True, exist_ok=True)
        (output_dir / "home_disjoint").mkdir(parents=True, exist_ok=True)

        self.handles = {
            "all": (output_dir / "all_examples.jsonl").open("w"),
            "official_train": (output_dir / "official" / "train.jsonl").open(
                "w"
            ),
            "official_valid": (output_dir / "official" / "valid.jsonl").open(
                "w"
            ),
            "official_test": (output_dir / "official" / "test.jsonl").open("w"),
            "home_train": (output_dir / "home_disjoint" / "train.jsonl").open(
                "w"
            ),
            "home_valid": (output_dir / "home_disjoint" / "valid.jsonl").open(
                "w"
            ),
            "home_test": (output_dir / "home_disjoint" / "test.jsonl").open(
                "w"
            ),
        }

    def _write_line(self, key: str, payload: dict[str, Any]) -> None:
        self.handles[key].write(json.dumps(payload, ensure_ascii=True) + "\n")

    def write(self, example: dict[str, Any]) -> None:
        self._write_line("all", example)
        self._write_line(f"official_{example['official_split']}", example)
        self._write_line(f"home_{example['home_disjoint_split']}", example)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def build_prepared_success_outputs(
    converted_outputs: list[dict[str, Any]],
    store: LocalHomeGraphStore,
) -> list[PreparedOutput]:
    prepared: list[PreparedOutput] = []
    for output_index, output in enumerate(converted_outputs):
        if output.get("execution") != "success":
            continue
        params = output.get("params") or {}
        if len(params) != 1:
            continue

        parameter_name = next(iter(params))
        property_url = (output.get("test") or {}).get("property")
        expected_value = numeric_value(
            (output.get("test") or {}).get("expected_value")
        )
        current_value = (
            numeric_value(store.current_property_value(property_url))
            if property_url
            else None
        )
        if (
            property_url is None
            or expected_value is None
            or current_value is None
        ):
            continue

        action_meta = store.action_metadata(output["affordance"])
        property_meta = store.property_metadata(property_url)
        workspace_type = store.workspace_semantic_type(
            action_meta["workspace_uri"]
        )
        artifact_type = store.artifact_semantic_type(
            action_meta["artifact_uri"]
        )
        parameter_schema = (
            action_meta.get("schema", {})
            .get("properties", {})
            .get(parameter_name, {})
        )
        if parameter_schema.get("type") not in {"integer", "number"}:
            continue

        prepared.append(
            PreparedOutput(
                output_index=output_index,
                affordance=output["affordance"],
                property_url=property_url,
                expected_value=expected_value,
                current_value=current_value,
                parameter_name=parameter_name,
                home_id=store._home_id_from_url(output["affordance"]),
                action_meta=action_meta,
                property_meta=property_meta,
                workspace_type=workspace_type,
                artifact_type=artifact_type,
            )
        )
    return prepared


def match_modify_intents(
    request_text: str,
    prepared_outputs: list[PreparedOutput],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_candidates = [
        parse_relative_clause(part) for part in split_into_clauses(request_text)
    ]
    relative_candidates: list[ClauseCandidate] = []
    for index, candidate in enumerate(raw_candidates):
        candidate.index = index
        if candidate.is_relative:
            relative_candidates.append(candidate)

    used_outputs: set[int] = set()
    intents: list[dict[str, Any]] = []

    for candidate in relative_candidates:
        matches: list[tuple[int, int, PreparedOutput]] = []
        for prepared in prepared_outputs:
            if prepared.output_index in used_outputs:
                continue
            score = candidate_score(candidate, prepared)
            if score < 5:
                continue
            if not validate_candidate_against_output(candidate, prepared):
                continue
            matches.append((score, prepared.output_index, prepared))

        if not matches:
            continue

        matches.sort(key=lambda item: (-item[0], item[1]))
        _, _, prepared = matches[0]
        used_outputs.add(prepared.output_index)
        intents.append(make_intent_record(candidate, prepared))

    intents.sort(key=lambda item: item["clause_index"])
    metadata = {
        "relative_candidate_count": len(relative_candidates),
        "matched_intent_count": len(intents),
        "matched_output_indices": [
            item["source_output_index"] for item in intents
        ],
    }
    return intents, metadata


def has_conflicting_targets(intents: list[dict[str, Any]]) -> bool:
    seen: dict[str, Any] = {}
    for intent in intents:
        property_url = intent["resolved"]["property_url"]
        expected_value = intent["expected_value"]
        if property_url in seen and seen[property_url] != expected_value:
            return True
        seen[property_url] = expected_value
    return False


def build_example(
    raw_entry: dict[str, Any],
    converted_entry: dict[str, Any],
    store: LocalHomeGraphStore,
    home_split_map: dict[int, str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    prepared_outputs = build_prepared_success_outputs(
        converted_entry["output"], store
    )
    if not prepared_outputs:
        return None, "no_numeric_success_outputs"

    intents, alignment = match_modify_intents(
        converted_entry["input"], prepared_outputs
    )
    if not intents:
        return None, "no_feasible_modify_intents"

    if has_conflicting_targets(intents):
        return None, "conflicting_property_targets"

    home_id = parse_home_id_from_test_id(raw_entry["id"])
    discovery = store.build_discovery_result(
        home_id, [intent["resolved"]["action_url"] for intent in intents]
    )
    runtime_modify_goal = build_runtime_modify_goal(intents)
    runtime_context = discovery.to_prompt_context()
    target_code = synthesize_target_code(intents)

    try:
        validation = validate_target_code(target_code)
    except Exception as exc:
        return None, f"validation_failed:{type(exc).__name__}:{exc}"

    exact_signature = build_exact_signature(
        runtime_modify_goal, runtime_context, target_code
    )

    example = {
        "id": raw_entry["id"],
        "source_file": raw_entry["source_file"],
        "source_row_type": raw_entry.get("type"),
        "official_split": raw_entry["official_split"],
        "home_disjoint_split": home_split_map[home_id],
        "home_id": home_id,
        "original_request": converted_entry["input"],
        "runtime_modify_goal": runtime_modify_goal,
        "runtime_context": runtime_context,
        "modify_intent_count": len(intents),
        "intents": intents,
        "target_code": target_code,
        "target_tree_ir": validation["tree_ir"],
        "validation_status": {
            "syntax_ok": validation["syntax_ok"],
            "tree_build_ok": validation["tree_build_ok"],
            "tree_root_name": validation["tree_root_name"],
            "tree_root_type": validation["tree_root_type"],
            "error": None,
        },
        "alignment": alignment,
        "label_source": "deterministic",
        "exact_signature": exact_signature,
    }
    return example, None


def iter_raw_rows(path: Path):
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    ttl_dir = resolve_repo_path(args.ttl_dir)
    state_dir = resolve_repo_path(args.state_dir)
    output_dir = resolve_repo_path(args.output_dir)
    selected_sources = set(args.sources)
    known_sources = {spec["name"] for spec in SOURCE_SPECS}
    unknown_sources = sorted(selected_sources - known_sources)
    if unknown_sources:
        raise ValueError(
            f"Unknown source names: {', '.join(unknown_sources)}"
        )

    home_split_map = assign_home_splits(
        seed=args.seed,
        home_train_count=args.home_train_count,
        home_valid_count=args.home_valid_count,
    )
    store = LocalHomeGraphStore(ttl_dir=ttl_dir, state_dir=state_dir)
    writers = DatasetWriters(output_dir)

    counters = {
        "rows_seen": 0,
        "examples_written": 0,
        "official_split": Counter(),
        "home_split": Counter(),
        "source_file": Counter(),
        "source_row_type": Counter(),
        "parameter": Counter(),
        "artifact_type": Counter(),
        "workspace_type": Counter(),
        "direction": Counter(),
        "intent_count": Counter(),
        "skip_reason": Counter(),
        "relative_candidate_count": Counter(),
    }

    try:
        for spec in SOURCE_SPECS:
            if spec["name"] not in selected_sources:
                continue

            raw_path = resolve_repo_path(spec["raw_path"])
            converted_path = resolve_repo_path(spec["converted_path"])
            converted_rows = json.loads(converted_path.read_text())

            for row_index, pair in enumerate(
                zip_longest(iter_raw_rows(raw_path), converted_rows)
            ):
                if args.limit is not None and row_index >= args.limit:
                    break

                raw_entry, converted_entry = pair
                if raw_entry is None or converted_entry is None:
                    raise ValueError(
                        f"Raw/converted length mismatch in {spec['name']}"
                    )

                if raw_entry["id"] != converted_entry["id"]:
                    raise ValueError(
                        "Raw/converted row mismatch at "
                        f"{spec['name']}[{row_index}]: "
                        f"{raw_entry['id']} != {converted_entry['id']}"
                    )

                counters["rows_seen"] += 1
                raw_entry = dict(raw_entry)
                raw_entry["official_split"] = spec["official_split"]
                raw_entry["source_file"] = spec["name"]

                example, skip_reason = build_example(
                    raw_entry=raw_entry,
                    converted_entry=converted_entry,
                    store=store,
                    home_split_map=home_split_map,
                )
                if example is None:
                    counters["skip_reason"][skip_reason] += 1
                    continue

                writers.write(example)
                counters["examples_written"] += 1
                counters["official_split"][example["official_split"]] += 1
                counters["home_split"][example["home_disjoint_split"]] += 1
                counters["source_file"][example["source_file"]] += 1
                counters["source_row_type"][example["source_row_type"]] += 1
                counters["intent_count"][example["modify_intent_count"]] += 1
                counters["relative_candidate_count"][
                    example["alignment"]["relative_candidate_count"]
                ] += 1

                for intent in example["intents"]:
                    counters["parameter"][intent["action"]["parameter"]] += 1
                    counters["direction"][intent["delta_direction"]] += 1
                    counters["artifact_type"][
                        intent["target"]["artifact_type"]
                    ] += 1
                    counters["workspace_type"][
                        intent["target"]["workspace_type"]
                    ] += 1
    finally:
        writers.close()

    home_lists: dict[str, list[int]] = defaultdict(list)
    for home_id, split in sorted(home_split_map.items()):
        home_lists[split].append(home_id)

    manifest = {
        "created_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "ttl_dir": str(ttl_dir),
        "state_dir": str(state_dir),
        "output_dir": str(output_dir),
        "sources": [
            spec for spec in SOURCE_SPECS if spec["name"] in selected_sources
        ],
        "home_disjoint_homes": dict(home_lists),
        "counts": {
            "rows_seen": counters["rows_seen"],
            "examples_written": counters["examples_written"],
            "official_split": dict(counters["official_split"]),
            "home_split": dict(counters["home_split"]),
            "source_file": dict(counters["source_file"]),
            "source_row_type": dict(counters["source_row_type"]),
            "parameter": dict(counters["parameter"]),
            "direction": dict(counters["direction"]),
            "artifact_type": dict(counters["artifact_type"]),
            "workspace_type": dict(counters["workspace_type"]),
            "intent_count": dict(counters["intent_count"]),
            "relative_candidate_count": dict(
                counters["relative_candidate_count"]
            ),
            "skip_reason": dict(counters["skip_reason"]),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
