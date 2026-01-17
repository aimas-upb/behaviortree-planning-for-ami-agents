"""Compute action-level precision/recall/F1 per action type group for an experiment run.

Usage:
  python compute_action_f1_by_group.py /path/to/experiments/tests/test_run_dir \
      [--dataset-file datasets/HomeBench/converted/test_data.json]

The script expects:
- An experiment directory that contains a `results_*.json` file (array of test results).
- A dataset file with the test definitions (same format used by `data_preprocess/preprocess.py`)
  so test IDs can be mapped back to their action groups.

It aggregates counts of expected, matched, and extra actions for each action group
(`single_feasible_action`, `multi_feasible_action`, `multi_unfeasible_action`) and prints the
corresponding precision/recall/F1 scores.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List


class ActionGroup(str, Enum):
    SINGLE_FEASIBLE = "single_feasible_action"
    SINGLE_UNFEASIBLE = "single_unfeasible_action"
    MULTI_FEASIBLE = "multi_feasible_action"
    MULTI_UNFEASIBLE = "multi_mixed_action"


@dataclass
class ActionCounts:
    tests: int = 0
    expected: int = 0
    matched: int = 0
    extra: int = 0

    def precision(self) -> float:
        denom = self.matched + self.extra
        return self.matched / denom if denom else 0.0

    def recall(self) -> float:
        return self.matched / self.expected if self.expected else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def load_test_groups(dataset_path: str) -> Dict[str, ActionGroup]:
    """Load mapping from test_id to ActionGroup from the full dataset file."""

    def classify(entry: Dict[str, Any]) -> ActionGroup:
        test_id = entry.get("id", "")
        outputs = entry.get("output") or []
        if "one" in test_id:
            first = outputs[0] if outputs else {}
            return (
                ActionGroup.SINGLE_FEASIBLE
                if first.get("execution") != "error_input"
                else ActionGroup.SINGLE_UNFEASIBLE
            )
        for output in outputs:
            if output.get("execution") == "error_input":
                return ActionGroup.MULTI_UNFEASIBLE
        return ActionGroup.MULTI_FEASIBLE

    with open(dataset_path, "r") as f:
        data = json.load(f)

    mapping: Dict[str, ActionGroup] = {}
    for entry in data:
        group = classify(entry)
        test_id = entry.get("id")
        if test_id:
            mapping[test_id] = group
    return mapping


def find_latest_results_file(experiment_dir: str) -> str:
    """Pick the most recent results_*.json file inside the experiment directory."""
    candidates = [
        os.path.join(experiment_dir, f)
        for f in os.listdir(experiment_dir)
        if f.startswith("results_") and f.endswith(".json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found in {experiment_dir}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_results(results_path: str) -> List[Dict[str, Any]]:
    with open(results_path, "r") as f:
        return json.load(f)


def aggregate_by_group(
    results: List[Dict[str, Any]],
    group_map: Dict[str, ActionGroup],
    target_groups: Iterable[ActionGroup],
) -> Dict[str, Any]:
    """Aggregate expected/matched/extra action counts for each action group."""
    stats: Dict[ActionGroup, ActionCounts] = {g: ActionCounts() for g in target_groups}
    ignored_groups: Dict[str, str] = {}
    unknown_tests: List[str] = []

    for entry in results:
        test_id = entry.get("test_id")
        group = group_map.get(test_id)
        if not group:
            if test_id:
                unknown_tests.append(test_id)
            continue
        if group not in stats:
            if test_id:
                ignored_groups[test_id] = group.value
            continue

        expected_actions = entry.get("expected_actions", []) or []
        matched_actions = entry.get("matched_actions", []) or []
        extra_actions = entry.get("extra_actions", []) or []

        counts = stats[group]
        counts.tests += 1
        counts.expected += len(expected_actions)
        counts.matched += len(matched_actions)
        counts.extra += len(extra_actions)
    return {
        "stats": stats,
        "unknown_tests": sorted(set(unknown_tests)),
        "ignored_tests": ignored_groups,
    }


def build_report(
    stats: Dict[ActionGroup, ActionCounts],
    unknown_tests: List[str],
    ignored_tests: Dict[str, str],
) -> Dict[str, Any]:
    """Prepare a JSON-serializable report of group metrics and unmapped tests."""
    report: Dict[str, Any] = {}
    for group, counts in stats.items():
        report[group.value] = {
            "tests": counts.tests,
            "expected_actions": counts.expected,
            "matched_actions": counts.matched,
            "extra_actions": counts.extra,
            "action_precision": round(counts.precision(), 4),
            "action_recall": round(counts.recall(), 4),
            "action_f1": round(counts.f1(), 4),
        }

    if unknown_tests:
        report["unmapped_test_ids"] = unknown_tests
    if ignored_tests:
        report["ignored_test_ids"] = ignored_tests
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute action F1 per action group for an experiment run."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Path to the experiment output directory (contains results_*.json).",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=os.path.join("datasets", "HomeBench", "converted", "test_data.json"),
        help="Path to the dataset JSON file with all test definitions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_groups = (
        ActionGroup.SINGLE_FEASIBLE,
        ActionGroup.MULTI_FEASIBLE,
        ActionGroup.MULTI_UNFEASIBLE,
    )

    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset_file}. "
            "Pass --dataset-file to point to the correct test set."
        )

    group_map = load_test_groups(args.dataset_file)
    if not group_map:
        raise FileNotFoundError(
            f"No group mapping found in dataset {args.dataset_file}; "
            "check that the file contains the expected test definitions."
        )

    results_path = find_latest_results_file(args.experiment_dir)
    results = load_results(results_path)

    agg_result = aggregate_by_group(results, group_map, target_groups=target_groups)
    report = build_report(
        stats=agg_result["stats"],
        unknown_tests=agg_result["unknown_tests"],
        ignored_tests=agg_result["ignored_tests"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
