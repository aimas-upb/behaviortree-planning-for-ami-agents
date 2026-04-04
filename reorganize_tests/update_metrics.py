import os
import re
import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so repo packages can be imported when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_homebench import EvaluationMetrics, TestResult


def parse_results(path: str) -> list[TestResult]:
    with open(path, 'r') as f:
        results_data = json.load(f)
    
    results: list[TestResult] = []

    for result in results_data:
        results.append(TestResult(
            test_id=result.get("test_id", ""),
            success=result.get("success", ""),
            is_error_input_only=result.get("is_error_input_only", False),
            handled_correctly=result.get("handled_correctly", False),
            plan_generated=result.get("plan_generated", False),
            execution_success=result.get("execution_success", False),
            matched_actions=result.get("matched_actions", []),
            missing_actions=result.get("missing_actions", []),
            extra_actions=result.get("extra_actions", []),
            expected_actions=result.get("expected_actions", []),
            expected_params=result.get("expected_params", {}),
            actions_in_plan=result.get("actions_in_plan", []),
            params_in_plan=result.get("params_in_plan", {}),
            properties_matched=result.get("properties_matched", 0),
            properties_checked=result.get("properties_checked", 0),
            property_results=result.get("property_results", []),
            expected_impossible=result.get("expected_impossible", False),
            detected_impossible=result.get("detected_impossible", False),
            failure_type=result.get("failure_type", "other"),
            duration_seconds=result.get("duration", 0.0),
            error=result.get("error", ""),
            raw_result=result.get("trace", {}),  # Full experiment trace
        ))

    return results


def parse_evaluation_metrics(path: str) -> tuple[dict[str, Any], dict[str, Any], EvaluationMetrics]:
    with open(path, 'r') as f:
        metrics_data = json.load(f)
    
    metrics = metrics_data.get("metrics", {})
    config = metrics_data.get("config", {})
    args = metrics_data.get("filters", {})

    if not metrics:
        raise ValueError(f"No metrics found in the file: {path}")
    
    evaluation_metrics = EvaluationMetrics(
        total_tests=metrics.get("total_tests", 0),
        successful_tests=metrics.get("successful_tests", 0),
        quantifiable_tests=metrics.get("quantifiable_tests", 0),
        failed_tests=metrics.get("failed_tests", 0),
        plans_generated=metrics.get("plans_generated", 0),
        total_duration=metrics.get("total_duration", 0.0),
        total_expected_actions=metrics.get("total_expected_actions", 0),
        total_matched_actions=metrics.get("total_matched_actions", 0),
        total_missing_actions=metrics.get("total_missing_actions", 0),
        total_extra_actions=metrics.get("total_extra_actions", 0),
        total_properties_checked=metrics.get("total_properties_checked", 0),
        total_properties_matched=metrics.get("total_properties_matched", 0),
        total_expected_impossible=metrics.get("total_expected_impossible", 0),
        total_detected_impossible=metrics.get("total_detected_impossible", 0),
        failures_by_type=metrics.get("failures_by_type", {})
    )

    return config, args, evaluation_metrics


def get_wrong_tests() -> set[str]:
    dining_room_tests_path = os.path.join(os.path.dirname(__file__), 'dining_room_tests.json')
    open_close_tests_path = os.path.join(os.path.dirname(__file__), 'open_close_tests.json')

    with open(dining_room_tests_path, 'r') as f:
        dining_room_tests = json.load(f)

    with open(open_close_tests_path, 'r') as f:
        open_close_tests = json.load(f)

    wrong_tests_ids = set()
    for tests in dining_room_tests.values():
        for test in tests["tests"]:
            wrong_tests_ids.add(test["id"])

    for tests in open_close_tests.values():
        for test in tests["tests"]:
            wrong_tests_ids.add(test["id"])

    return wrong_tests_ids


def split_wrong_tests(results_path: str, metrics_path: str) -> None:
    wrong_tests = get_wrong_tests()
    results = parse_results(results_path)
    config, args, metrics = parse_evaluation_metrics(metrics_path)

    new_results: list[TestResult] = []

    for result in results:
        if result.test_id in wrong_tests:
            metrics.total_tests -= 1
            
            # Check if result was "True", "False", or "Quantifiable"
            if result.success == "True":
                metrics.successful_tests -= 1
            elif result.success == "Quantifiable":
                metrics.quantifiable_tests -= 1
            else:
                metrics.failed_tests -= 1

            # Adjust plans generated count
            if result.plan_generated:
                metrics.plans_generated -= 1

            # Adjust total time
            if result.duration_seconds != 0.0:
                metrics.total_duration -= result.duration_seconds

            # Adjust total expected/matched/missing/extra actions counts
            expected_count = len(result.expected_actions) if result.expected_actions else 0
            matched_count = len(result.matched_actions) if result.matched_actions else 0
            missing_count = len(result.missing_actions) if result.missing_actions else 0
            extra_count = len(result.extra_actions) if result.extra_actions else 0

            metrics.total_expected_actions -= expected_count
            metrics.total_matched_actions -= matched_count
            metrics.total_missing_actions -= missing_count
            metrics.total_extra_actions -= extra_count

            # Adjust total properties checked/matched counts
            metrics.total_properties_checked -= result.properties_checked
            metrics.total_properties_matched -= result.properties_checked

            # Check if the result was expected impossible and detected impossible
            metrics.total_expected_impossible -= result.expected_impossible

            if result.detected_impossible:
                metrics.total_detected_impossible -= len(result.detected_impossible)

            # Adjust failures by type
            failure_type = result.failure_type
            if failure_type and failure_type in metrics.failures_by_type:
                metrics.failures_by_type[failure_type] -= 1
                if metrics.failures_by_type[failure_type] == 0:
                    del metrics.failures_by_type[failure_type]
        else:
            new_results.append(result)

    # Save the updated results and metrics to the "results_good_<...>.json" and "metrics_good_<...>.json" files
    new_results_path = results_path.replace('results_', 'results_good_')
    new_metrics_path = metrics_path.replace('metrics_', 'metrics_good_')

    with open(new_results_path, 'w') as f:
        json.dump([{
            "test_id": r.test_id,
            "success": r.success,
            "is_error_input_only": r.is_error_input_only,
            "handled_correctly": r.handled_correctly,
            "plan_generated": r.plan_generated,
            "execution_success": r.execution_success,
            "matched_actions": r.matched_actions,
            "missing_actions": r.missing_actions,
            "extra_actions": r.extra_actions,
            "expected_actions": r.expected_actions,
            "expected_params": r.expected_params,
            "actions_in_plan": r.actions_in_plan,
            "params_in_plan": r.params_in_plan,
            "properties_matched": r.properties_matched,
            "properties_checked": r.properties_checked,
            "property_results": r.property_results,
            "expected_impossible": r.expected_impossible,
            "detected_impossible": r.detected_impossible,
            "failure_type": r.failure_type,
            "duration": r.duration_seconds,
            "error": r.error,
            "trace": r.raw_result,  # Full experiment trace
        } for r in new_results], f, indent=2, default=str)

    with open(new_metrics_path, 'w') as f:
        json.dump({
            "config": config,
            "filters": args,
            "metrics": metrics.to_dict(),
        }, f, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(description="Count specific types of tests in JSON files.")
    parser.add_argument('--tests_dir', type=str, help='Directory containing the results_<...>.json file.', required=True)

    args = parser.parse_args()
    tests_dir = args.tests_dir

    files = os.listdir(tests_dir)
    results_file = next((f for f in files if re.match(r'results_.*\.json', f)), None)
    metrics_file = next((f for f in files if re.match(r'metrics_.*\.json', f)), None)
    
    if not results_file or not metrics_file:
        raise FileNotFoundError("No results_<...>.json or metrics_<...>.json file found in the specified directory.")
    
    results_path = os.path.join(tests_dir, results_file)
    metrics_path = os.path.join(tests_dir, metrics_file)

    split_wrong_tests(results_path, metrics_path)

    # Generate new report
    output_report_path = tests_dir + '/new_report.html'

    subprocess.run(
        [sys.executable, "-m", "viewers.eval_viewer", str(tests_dir), "--output", output_report_path],
        check=True,
        cwd=PROJECT_ROOT,
    )
    
    if os.path.exists(output_report_path):
        print(f"New report generated at: {output_report_path}")
    else:
        print("Failed to generate the new report.")


if __name__ == "__main__":
    main()
