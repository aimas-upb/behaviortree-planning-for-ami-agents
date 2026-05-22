"""
This script samples 80 test cases from the converted modify-codegen test split at data/homebench/benchmarks/modify_codegen/home_disjoint/test.jsonl.

The sample is stratified by modify intent count:

10 tests with exactly 1 modify intent
30 tests with exactly 2 modify intents
40 tests with 3 or more modify intents

Usage:
    uv run python extract_samples_for_inference.py --output_path data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference.json
"""

import json
import random
import argparse


from pathlib import Path
from typing import Any, List, Dict

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common import resolve_repo_path

random.seed(42)

TESTS_DIR = resolve_repo_path(
    'data/homebench/benchmarks/modify_codegen/home_disjoint/test.jsonl'
)
HOMEBENCH_CONVERTED_PATH = resolve_repo_path(
    'data/homebench/converted/'
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract samples for inference.")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the extracted samples.",
    )
    return parser.parse_args()


def extract_samples(tests_path: Path, output_path: Path) -> List[Dict[str, Any]]:
    with open(tests_path, 'r') as f:
        tests = [json.loads(line) for line in f]

    # Group tests by modify intent count
    intent_count_groups: Dict[int, List[Dict[str, Any]]] = {}
    for test in tests:
        intent_count = test.get('modify_intent_count', 0)

        if intent_count >= 3:
            intent_count = 3  # Group all tests with >= 3 modify intents together

        if intent_count not in intent_count_groups:
            intent_count_groups[intent_count] = []
        intent_count_groups[intent_count].append(test)

    if 1 not in intent_count_groups or 2 not in intent_count_groups or 3 not in intent_count_groups:
        raise ValueError("Selected test file does not contain enough tests with the required modify intent counts.")

    # Sample tests from each group
    sampled_tests = []
    sampled_tests.extend(random.sample(intent_count_groups[1], min(10, len(intent_count_groups[1]))))
    sampled_tests.extend(random.sample(intent_count_groups[2], min(30, len(intent_count_groups[2]))))
    sampled_tests.extend(random.sample(intent_count_groups[3], min(40, len(intent_count_groups[3]))))

    return sampled_tests


def convert_to_benchmark_format(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []

    # Group tests by source file
    grouped_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for test in tests:
        source_file = test.get('source_file')

        if not source_file:
            raise ValueError(f"Test with id {test['id']} is missing 'source_file' field.")

        if source_file not in grouped_by_file:
            grouped_by_file[source_file] = []
        grouped_by_file[source_file].append(test)

    for source_file, tests in grouped_by_file.items():
        with open(HOMEBENCH_CONVERTED_PATH / f"{source_file}.json") as f:
            original_converted_tests = json.load(f)

        for test in tests:
            matching_test = next((t for t in original_converted_tests if t['id'] == test['id']), None)

            if not matching_test:
                raise ValueError(f"Test with id {test['id']} not found in original converted tests.")
            
            converted.append({
                **matching_test,
                "modify_intent_count": test.get("modify_intent_count", 0),
                "target_code": test.get("target_code", ""),
            })
        
    return converted


def main():
    args = parse_args()
    output_path = resolve_repo_path(Path(args.output_path))

    sampled_tests = extract_samples(TESTS_DIR, output_path)
    converted_tests = convert_to_benchmark_format(sampled_tests)

    with open(output_path, 'w') as f:
        json.dump(converted_tests, f, indent=2)


if __name__ == "__main__":
    main()
