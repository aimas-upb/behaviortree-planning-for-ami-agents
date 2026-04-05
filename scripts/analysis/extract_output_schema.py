"""
Utility to extract the output schema for a test entry's property affordance.

Usage examples:
  uv run python -m scripts.analysis.extract_output_schema --test-id home10_one_123
  uv run python -m scripts.analysis.extract_output_schema --test-id home10_one_123 --output-index 0
  uv run python -m scripts.analysis.extract_output_schema --property-url http://localhost:8080/workspaces/home10/living_room/artifacts/livingRoomAirPurifiers/properties/fan_speed
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import RDF, Graph, Namespace, URIRef

from scripts.common import PROJECT_ROOT

TD = Namespace("https://www.w3.org/2019/wot/td#")
JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract output schema for a test entry property."
    )
    parser.add_argument(
        "--test-file",
        default="data/homebench/converted/test_data.json",
        help="Path to the test data JSON file.",
    )
    parser.add_argument(
        "--test-id",
        help="Test entry id to look up (mutually exclusive with --property-url).",
    )
    parser.add_argument(
        "--property-url",
        help="Property URL to use directly (skips test file lookup).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate measurement unit statistics for all *_tests.json files in the benchmark directory.",
    )
    parser.add_argument(
        "--tests-dir",
        default="data/homebench/benchmarks/action_groups",
        help="Directory containing *_tests.json files (used with --stats).",
    )
    parser.add_argument(
        "--tests-glob",
        default="*_tests.json",
        help="Glob for test files inside tests-dir (used with --stats).",
    )
    return parser.parse_args()


def load_property_url_from_test(
    test_file: Path, test_id: str, output_index: int
) -> str:
    with test_file.open("r") as f:
        tests: List[Dict[str, Any]] = json.load(f)

    for entry in tests:
        if entry.get("id") != test_id:
            continue
        outputs = entry.get("output", [])
        if output_index >= len(outputs):
            raise IndexError(
                f"output-index {output_index} out of range for test {test_id}"
            )
        test_obj = outputs[output_index].get("test")
        if not test_obj or "property" not in test_obj:
            raise ValueError(
                f"No property test found at index {output_index} for test {test_id}"
            )
        return test_obj["property"]

    raise ValueError(f"Test id {test_id} not found in {test_file}")


def extract_home_id_from_url(property_url: str) -> str:
    match = re.search(r"/workspaces/home(\d+)", property_url)
    if not match:
        raise ValueError(f"Could not parse home id from {property_url}")
    return match.group(1)


def load_home_graph(
    home_id: str, description_dir: Path, graph_cache: Dict[str, Graph]
) -> Graph:
    if home_id in graph_cache:
        return graph_cache[home_id]
    ttl_path = description_dir / f"home_{home_id}.ttl"
    if not ttl_path.exists():
        raise FileNotFoundError(f"Home description not found: {ttl_path}")
    g = Graph()
    g.parse(ttl_path)
    graph_cache[home_id] = g
    return g


def load_output_schema(
    property_url: str,
    description_dir: Path,
    graph_cache: Optional[Dict[str, Graph]] = None,
    schema_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return schema information for a property affordance."""
    if schema_cache is not None and property_url in schema_cache:
        return schema_cache[property_url]

    home_id = extract_home_id_from_url(property_url)
    graph_cache = graph_cache or {}
    g = load_home_graph(home_id, description_dir, graph_cache)

    target = URIRef(property_url)

    schema_info: Dict[str, Any] = {}
    for form in g.subjects(predicate=HCTL.hasTarget, object=target):
        for affordance in g.subjects(predicate=TD.hasForm, object=form):
            schema_nodes = list(
                g.objects(subject=affordance, predicate=TD.hasOutputSchema)
            )
            if not schema_nodes:
                continue
            # Use the first matching schema node (should be unique per affordance).
            schema = schema_nodes[0]
            types = [
                str(t).split("#")[-1]
                for t in g.objects(subject=schema, predicate=RDF.type)
            ]
            if types:
                schema_info["types"] = types
            enums = [
                str(e)
                for e in g.objects(subject=schema, predicate=JSONSCHEMA.enum)
            ]
            if enums:
                schema_info["enum"] = enums
            minimums = [
                float(m)
                for m in g.objects(subject=schema, predicate=JSONSCHEMA.minimum)
            ]
            maximums = [
                float(m)
                for m in g.objects(subject=schema, predicate=JSONSCHEMA.maximum)
            ]
            if minimums:
                schema_info["minimum"] = min(minimums)
            if maximums:
                schema_info["maximum"] = max(maximums)
            if schema_cache is not None:
                schema_cache[property_url] = schema_info
            return schema_info

    raise ValueError(f"No output schema found for {property_url}")


def classify_schema(schema: Dict[str, Any]) -> str:
    """Map schema details to a coarse measurement unit label."""
    types = set(schema.get("types", []))
    if "BooleanSchema" in types:
        return "boolean"
    if "StringSchema" in types:
        if schema.get("enum"):
            return "enum"
        return "string"
    if "IntegerSchema" in types or "NumberSchema" in types:
        min_val = schema.get("minimum")
        max_val = schema.get("maximum")
        if (
            min_val is not None
            and max_val is not None
            and 0 <= min_val <= 1
            and max_val <= 100
        ):
            return "percentage"
        return "numeric"
    if "ArraySchema" in types:
        return "array"
    if "ObjectSchema" in types:
        return "object"
    return "unknown"


def iter_test_properties(test_entry: Dict[str, Any]):
    """Yield property URLs from a test entry outputs list."""
    for out in test_entry.get("output", []):
        test_obj = out.get("test") if isinstance(out, dict) else None
        if test_obj and "property" in test_obj:
            yield test_obj["property"]


def generate_stats(
    test_files: List[Path],
    description_dir: Path,
) -> Dict[str, Dict[str, int]]:
    """Return measurement unit stats per test file."""
    graph_cache: Dict[str, Graph] = {}
    schema_cache: Dict[str, Dict[str, Any]] = {}
    per_file_counts: Dict[str, Counter] = {}

    for path in test_files:
        counts: Counter = Counter()
        with path.open("r") as f:
            entries: List[Dict[str, Any]] = json.load(f)
        for entry in entries:
            for prop_url in iter_test_properties(entry):
                try:
                    schema = load_output_schema(
                        prop_url,
                        description_dir,
                        graph_cache=graph_cache,
                        schema_cache=schema_cache,
                    )
                    label = classify_schema(schema)
                    counts[label] += 1
                except Exception as exc:
                    print(
                        f"[warn] {path.name} {entry.get('id')}: {exc}",
                        file=sys.stderr,
                    )
                    counts["error"] += 1
        per_file_counts[path.name] = counts
    return {name: dict(counter) for name, counter in per_file_counts.items()}


def main() -> None:
    args = parse_args()
    description_dir = (
        PROJECT_ROOT / "data" / "homebench" / "hmas" / "home_description"
    )

    if args.stats:
        tests_dir = Path(args.tests_dir)
        if not tests_dir.is_absolute():
            tests_dir = PROJECT_ROOT / tests_dir
        test_files = sorted(tests_dir.glob(args.tests_glob))
        if not test_files:
            raise SystemExit(
                f"No test files matching {args.tests_glob} in {tests_dir}"
            )
        stats = generate_stats(test_files, description_dir)
        print(json.dumps(stats, indent=2))
        return

    if args.property_url:
        property_url = args.property_url
    elif args.test_id:
        test_file = Path(args.test_file)
        if not test_file.is_absolute():
            test_file = PROJECT_ROOT / test_file
        property_url = load_property_url_from_test(
            test_file, args.test_id, args.output_index
        )
    else:
        raise SystemExit("Provide either --property-url or --test-id.")

    schema = load_output_schema(property_url, description_dir)
    print(
        json.dumps(
            {"property": property_url, "output_schema": schema}, indent=2
        )
    )


if __name__ == "__main__":
    main()
