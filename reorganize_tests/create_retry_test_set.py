"""Script to create a retry test set by extracting dining room and open/close tests from existing test files.
Usage: python reorganize_tests/create_retry_test_set.py
"""

import os
import json
from pathlib import Path

from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_FILE = REPO_ROOT / "data" / "homebench" / "converted" / "test_data.json"
BENCHMARKS_DIR = REPO_ROOT / "data" / "homebench" / "benchmarks" / "action_groups"
RETRIES_DIR = REPO_ROOT / "data" / "homebench" / "retries" / "action_groups"


def get_dining_room_and_open_close_tests(tests_dir: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    files = os.listdir(tests_dir)
    files = [f for f in files if f.endswith('_tests.json')]
    dining_room_tests = {}
    open_close_tests = {}

    for file in files:
        dining_room_tests[file] = {"count": 0, "tests": []}
        open_close_tests[file] = {"count": 0, "tests": []}

        with open(os.path.join(tests_dir, file), 'r') as f:
            tests = json.load(f)
        
        for test in tests:
            if "dining" in test.get("input", {}):
                dining_room_tests[file]["tests"].append(test)
                dining_room_tests[file]["count"] += 1

            if "open" in test.get("input", {}) or "close" in test.get("input", {}):
                open_close_tests[file]["tests"].append(test)
                open_close_tests[file]["count"] += 1

        print(f"File: {file} contains {dining_room_tests[file]['count']} dining room tests.")
        print(f"File: {file} contains {open_close_tests[file]['count']} open/close tests.")

    return dining_room_tests, open_close_tests


def save_dining_room_and_open_close_tests(
    dining_room_tests: Dict[str, Dict],
    open_close_tests: Dict[str, Dict]
) -> None:
    with open('reorganize_tests/dining_room_tests.json', 'w') as f:
        json.dump(dining_room_tests, f, indent=2)

    with open('reorganize_tests/open_close_tests.json', 'w') as f:
        json.dump(open_close_tests, f, indent=2)


def merge_dining_room_and_open_close_tests(
    dining_room_tests: Dict[str, Dict],
    open_close_tests: Dict[str, Dict]
) -> Dict[str, List[Dict]]:
    merged_tests = {}
    merged_tests_ids = set()

    for file, data in dining_room_tests.items():
        tests = data["tests"]

        for test in tests:
            if file not in merged_tests:
                merged_tests[file] = []

            merged_tests[file].append(test)
            merged_tests_ids.add(test["id"])

    for file, data in open_close_tests.items():
        tests = data["tests"]

        for test in tests:
            if test["id"] in merged_tests_ids:
                continue  # Skip duplicates

            if file not in merged_tests:
                merged_tests[file] = []

            merged_tests[file].append(test)
            merged_tests_ids.add(test["id"])

    print(f"Total unique tests in merged set: {len(merged_tests_ids)}")

    return merged_tests


def create_retry_tests(merged_tests: Dict[str, List[Dict]]) -> None:
    no_tests_to_retry = 0

    with open(GROUND_TRUTH_FILE) as f:
        ground_truth_tests = json.load(f)

    for file, tests in merged_tests.items():
        retry_tests = []
        feasible_retry_tests = []

        for test in tests:
            matching_gt_test = next((gt_test for gt_test in ground_truth_tests if gt_test["id"] == test["id"]), None)

            if matching_gt_test:
                if file == "single_unfeasible_action_tests.json" and matching_gt_test["output"][0]["execution"] != "error_input":
                    # Add this test to retry list as single_feasible_action
                    feasible_retry_tests.append(matching_gt_test)
                    no_tests_to_retry += 1
                    continue  # Skip this test as it doesn't match the expected unfeasible action

                retry_tests.append(matching_gt_test)
                no_tests_to_retry += 1
            else:
                print(f"Warning: No matching ground truth test found for test ID {test['id']}")

        if retry_tests:    
            # Save the retry tests back to the file
            with open(RETRIES_DIR / file, 'w') as f:
                json.dump(retry_tests, f, indent=2)

        if feasible_retry_tests:
            # Save the feasible retry tests back to the file
            with open(RETRIES_DIR / "single_feasible_action_tests.json", 'a') as f:
                json.dump(feasible_retry_tests, f, indent=2)

    print(f"Total tests to retry: {no_tests_to_retry}")


def main():
    dining_room_tests, open_close_tests = get_dining_room_and_open_close_tests(str(BENCHMARKS_DIR))

    save_dining_room_and_open_close_tests(dining_room_tests, open_close_tests)

    merged_tests = merge_dining_room_and_open_close_tests(dining_room_tests, open_close_tests)
    create_retry_tests(merged_tests)


if __name__ == "__main__":
    main()
