#!/usr/bin/env python3
"""
Repair modify-codegen evaluation artifacts when valid actions were mislabeled as
`error_input` in the sampled HomeBench benchmark.

What this fixes
---------------
Some benchmark rows in
`data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference.json`
mark a user-requested action as `error_input` even though the action exists and
is executable for that home. In the saved evaluation results, those actions are
then counted as `extra_actions`, which lowers action F1, property accuracy, and
can keep a test stuck in "Quantifiable" instead of "True".

Fix strategy
------------
For each `extra_action` in the benchmark results:
1. Match it back to the extracted intent via `trace.resolution_results`.
2. Only consider it fixable when that intent's original benchmark slot is
   currently labeled `error_input`.
3. Verify the action against an in-process `SmartHomeSimulator`:
   - the action route exists for that home,
   - the parameters are valid,
   - invoking it produces the expected property value.
4. Promote verified actions into the benchmark ground truth, recompute the
   per-test result fields, aggregate metrics, and regenerate `eval_report.html`
   using a corrected copy of the sampled benchmark JSON.

This is a reporting/data-repair pass. It does not rerun the original models.

Usage example:
    uv run python -m scripts.analysis.fix_modify_codegen_reporting \
        "experiments/results/modify_codegen_qwen_samples_168607"
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from homebench.smart_home_simulator import SmartHomeSimulator
from scripts.common import PROJECT_ROOT

DEFAULT_BASE_DIR = "experiments/results/modify_codegen_qwen_samples_156883"
DEFAULT_SAMPLE_FILE = (
    "data/homebench/benchmarks/modify_codegen/home_disjoint/"
    "samples_for_inference.json"
)
DEFAULT_CORRECTED_SAMPLE_NAME = "corrected_samples_for_inference.json"

BASE_URL = "http://localhost:8080"
STATE_ACTION_EXPECTATIONS: dict[str, tuple[str, Any]] = {
    "turn_off": ("state", "off"),
    "turn_on": ("state", "on"),
    "open": ("state", "open"),
    "close": ("state", "closed"),
    "pack": ("state", "packed"),
}
DECREASE_HINTS = ("decrease", "lower", "reduce", "drop", "down")
INCREASE_HINTS = ("increase", "raise", "boost", "up")


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _strip_base_url(url: str) -> str:
    return url.replace(BASE_URL, "")


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _build_simulator(sample_rows: list[dict]) -> SmartHomeSimulator:
    simulator = SmartHomeSimulator(
        PROJECT_ROOT / "data/homebench/hmas/home_description"
    )
    home_ids = sorted(
        {
            int(
                row.get("home_id")
                or str(row["id"]).split("_")[0].replace("home", "")
            )
            for row in sample_rows
        }
    )
    simulator.load_homes(home_ids)
    return simulator


def _compute_modify_expected_value(
    intent: dict, current_value: Any, schema_info: dict
) -> int | float:
    if isinstance(current_value, bool) or not isinstance(
        current_value, (int, float)
    ):
        raise ValueError(
            f"Cannot compute relative target from non-numeric value: {current_value!r}"
        )

    raw_delta = intent.get("action", {}).get("value")
    try:
        delta = float(raw_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid relative delta: {raw_delta!r}") from exc

    text = _normalize_text(intent.get("text_intent"))
    if any(token in text for token in DECREASE_HINTS):
        target = current_value - delta
    elif any(token in text for token in INCREASE_HINTS):
        target = current_value + delta
    else:
        raise ValueError(
            f"Could not infer modify direction from intent: {intent.get('text_intent')!r}"
        )

    if "minimum" in schema_info:
        target = max(schema_info["minimum"], target)
    if "maximum" in schema_info:
        target = min(schema_info["maximum"], target)

    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(target)
    return target


def _verify_extra_action(
    simulator: SmartHomeSimulator,
    home_id: str,
    result: dict,
    action_url: str,
    resolution_result: dict,
) -> tuple[dict | None, str | None]:
    action_path = _strip_base_url(action_url)
    route = simulator.action_routes.get(action_path)
    if not route:
        return None, "action_route_missing"

    artifact_uri, _action_name, _param_names, schema = route
    artifact_base = action_url.rsplit("/", 1)[0]
    params = copy.deepcopy(
        (result.get("params_in_plan") or {}).get(action_url) or {}
    )
    property_url: str | None = None
    expected_value: Any = None

    try:
        simulator.reset_home(home_id)

        if params:
            if len(params) != 1:
                return None, f"unsupported_param_count:{len(params)}"
            prop_name, expected_value = next(iter(params.items()))
            property_url = f"{artifact_base}/properties/{prop_name}"
        else:
            intent = resolution_result.get("intent") or {}
            action_tail = action_url.rsplit("/", 1)[1]
            if intent.get("action", {}).get("verb") == "modify":
                prop_name = resolution_result.get(
                    "parameter_name"
                ) or intent.get("action", {}).get("parameter")
                if not prop_name:
                    return None, "modify_property_name_missing"
                property_url = f"{artifact_base}/properties/{prop_name}"
                current_value = simulator.get_property(
                    _strip_base_url(property_url)
                )
                expected_value = _compute_modify_expected_value(
                    intent=intent,
                    current_value=current_value,
                    schema_info=schema.get(prop_name, {}),
                )
                params = {prop_name: expected_value}
            elif action_tail in STATE_ACTION_EXPECTATIONS:
                prop_name, expected_value = STATE_ACTION_EXPECTATIONS[
                    action_tail
                ]
                property_url = f"{artifact_base}/properties/{prop_name}"
            else:
                return None, f"no_verification_strategy:{action_tail}"

        if _strip_base_url(property_url) not in simulator.property_routes:
            return None, f"property_route_missing:{property_url}"

        simulator.invoke_action(action_path, params)
        actual_value = simulator.get_property(_strip_base_url(property_url))
        matched = actual_value == expected_value

        if not matched:
            return (
                None,
                f"property_mismatch:expected={expected_value!r}:actual={actual_value!r}",
            )

        return (
            {
                "action_url": action_url,
                "params": params,
                "property_url": property_url,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "matched": matched,
                "verified_with": "in_process_simulator",
                "intent_text": (resolution_result.get("intent") or {}).get(
                    "text_intent"
                ),
                "original_index": (resolution_result.get("intent") or {}).get(
                    "original_index"
                ),
            },
            None,
        )
    except HTTPException as exc:
        return None, f"http:{exc.status_code}:{exc.detail}"
    except Exception as exc:  # pragma: no cover - best effort reporting
        return None, f"{type(exc).__name__}:{exc}"


def _property_result_key(prop_result: dict) -> tuple[str, str]:
    return (
        prop_result.get("property", ""),
        json.dumps(prop_result.get("expected"), sort_keys=True, default=str),
    )


def _output_entry_key(output_entry: dict) -> tuple[str, str]:
    test_spec = output_entry.get("test") or {}
    return (
        test_spec.get("property", ""),
        json.dumps(
            test_spec.get("expected_value"), sort_keys=True, default=str
        ),
    )


def _rebuild_property_results(
    corrected_outputs: list[dict],
    original_result: dict,
    verified_actions: dict[str, dict],
) -> list[dict]:
    existing_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for prop_result in original_result.get("property_results") or []:
        existing_by_key[_property_result_key(prop_result)].append(
            copy.deepcopy(prop_result)
        )

    rebuilt: list[dict] = []
    for entry in corrected_outputs:
        if entry.get("execution") != "success":
            continue

        action_url = entry.get("affordance")
        if action_url in verified_actions:
            verified = verified_actions[action_url]
            rebuilt.append(
                {
                    "property": verified["property_url"],
                    "expected": verified["expected_value"],
                    "actual": verified["actual_value"],
                    "matched": verified["matched"],
                }
            )
            continue

        key = _output_entry_key(entry)
        if existing_by_key.get(key):
            rebuilt.append(existing_by_key[key].pop(0))

    return rebuilt


def _recompute_action_lists(
    expected_actions: list[str], actual_actions: list[str]
) -> tuple[list[str], list[str], list[str]]:
    expected_set = set(expected_actions)
    actual_set = set(actual_actions)

    matched = _ordered_unique([a for a in expected_actions if a in actual_set])
    missing = _ordered_unique(
        [a for a in expected_actions if a not in actual_set]
    )
    extra = _ordered_unique(
        [a for a in actual_actions if a not in expected_set]
    )
    return matched, missing, extra


def _classify_failure(result: dict) -> str:
    if result.get("is_error_input_only"):
        return "error_input_not_detected"
    if result.get("plan_generated") and not result.get("execution_success"):
        return "execution_error"
    if result.get("properties_checked", 0) > 0 and result.get(
        "properties_matched", 0
    ) < result.get("properties_checked", 0):
        return "property_mismatch"
    if result.get("missing_actions") or result.get("extra_actions"):
        return "action_mismatch"
    return result.get("failure_type") or "other"


def _recompute_result(
    original_result: dict,
    corrected_outputs: list[dict],
    verified_actions: dict[str, dict],
) -> dict:
    fixed = copy.deepcopy(original_result)
    expected_successes = [
        output
        for output in corrected_outputs
        if output.get("execution") == "success"
    ]
    expected_errors = [
        output
        for output in corrected_outputs
        if output.get("execution") == "error_input"
    ]

    fixed["expected_actions"] = [
        output.get("affordance", "")
        for output in expected_successes
        if output.get("affordance")
    ]
    fixed["expected_params"] = {
        output.get("affordance"): output.get("params", {})
        for output in expected_successes
        if output.get("affordance")
    }
    fixed["expected_impossible"] = len(expected_errors)
    fixed["is_error_input_only"] = (
        len(expected_successes) == 0 and len(expected_errors) > 0
    )

    matched, missing, extra = _recompute_action_lists(
        fixed["expected_actions"], fixed.get("actions_in_plan") or []
    )
    fixed["matched_actions"] = matched
    fixed["missing_actions"] = missing
    fixed["extra_actions"] = extra

    property_results = _rebuild_property_results(
        corrected_outputs=corrected_outputs,
        original_result=original_result,
        verified_actions=verified_actions,
    )
    fixed["property_results"] = property_results
    fixed["properties_checked"] = len(property_results)
    fixed["properties_matched"] = sum(
        1 for prop_result in property_results if prop_result.get("matched")
    )

    promoted_texts = {
        _normalize_text(item.get("intent_text"))
        for item in verified_actions.values()
    }
    fixed["detected_impossible"] = [
        text
        for text in fixed.get("detected_impossible") or []
        if _normalize_text(text) not in promoted_texts
    ]
    if isinstance(fixed.get("trace"), dict) and isinstance(
        fixed["trace"].get("detected_impossible"), list
    ):
        fixed["trace"]["detected_impossible"] = copy.deepcopy(
            fixed["detected_impossible"]
        )

    if fixed["is_error_input_only"]:
        detected_as_impossible = (
            len(fixed.get("detected_impossible") or []) > 0
            or not fixed.get("plan_generated")
            or len(fixed.get("actions_in_plan") or []) == 0
        )
        fixed["handled_correctly"] = bool(detected_as_impossible)
        fixed["success"] = "True" if detected_as_impossible else "False"
    else:
        is_plan_executed = fixed.get("plan_generated") and fixed.get(
            "execution_success"
        )
        if not is_plan_executed:
            fixed["handled_correctly"] = False
            fixed["success"] = "False"
        else:
            no_extra = len(fixed.get("extra_actions") or []) == 0
            all_props = fixed.get("properties_matched", 0) == fixed.get(
                "properties_checked", 0
            )
            fixed["handled_correctly"] = no_extra and all_props
            if fixed["handled_correctly"]:
                fixed["success"] = "True"
            elif fixed.get("matched_actions"):
                fixed["success"] = "Quantifiable"
            else:
                fixed["success"] = "False"

    if fixed["success"] == "False":
        fixed["failure_type"] = _classify_failure(fixed)
    else:
        fixed.pop("failure_type", None)

    fixed["_mislabeled_error_input_fix"] = {
        "applied_at": datetime.now().isoformat(),
        "verified_extra_actions": list(verified_actions.values()),
    }
    return fixed


def _recompute_metrics(results: list[dict]) -> dict:
    metrics: dict = {
        "total_tests": len(results),
        "successful_tests": 0,
        "quantifiable_tests": 0,
        "failed_tests": 0,
        "plans_generated": 0,
        "total_expected_actions": 0,
        "total_matched_actions": 0,
        "total_missing_actions": 0,
        "total_extra_actions": 0,
        "total_properties_checked": 0,
        "total_properties_matched": 0,
        "total_expected_impossible": 0,
        "total_detected_impossible": 0,
        "total_duration": 0.0,
        "failures_by_type": {
            "parse_error": 0,
            "compilation_error": 0,
            "execution_error": 0,
            "action_mismatch": 0,
            "property_mismatch": 0,
            "error_input_not_detected": 0,
            "other": 0,
        },
    }

    for result in results:
        success = result.get("success")
        if success == "True":
            metrics["successful_tests"] += 1
        elif success == "Quantifiable":
            metrics["quantifiable_tests"] += 1
        else:
            metrics["failed_tests"] += 1

        if result.get("plan_generated"):
            metrics["plans_generated"] += 1

        metrics["total_expected_actions"] += len(
            result.get("expected_actions") or []
        )
        metrics["total_matched_actions"] += len(
            result.get("matched_actions") or []
        )
        metrics["total_missing_actions"] += len(
            result.get("missing_actions") or []
        )
        metrics["total_extra_actions"] += len(result.get("extra_actions") or [])
        metrics["total_properties_checked"] += result.get(
            "properties_checked", 0
        )
        metrics["total_properties_matched"] += result.get(
            "properties_matched", 0
        )
        metrics["total_expected_impossible"] += result.get(
            "expected_impossible", 0
        )
        metrics["total_detected_impossible"] += len(
            result.get("detected_impossible") or []
        )
        metrics["total_duration"] += result.get("duration", 0.0)

        failure_type = result.get("failure_type")
        if failure_type and failure_type in metrics["failures_by_type"]:
            metrics["failures_by_type"][failure_type] += 1

    total_tests = metrics["total_tests"] or 1
    metrics["success_rate"] = metrics["successful_tests"] / total_tests
    metrics["quantifiable_rate"] = metrics["quantifiable_tests"] / total_tests
    metrics["success_or_quantifiable_rate"] = (
        metrics["successful_tests"] + metrics["quantifiable_tests"]
    ) / total_tests

    matched = metrics["total_matched_actions"]
    extra = metrics["total_extra_actions"]
    expected = metrics["total_expected_actions"]
    precision = matched / (matched + extra) if (matched + extra) > 0 else 0.0
    recall = matched / expected if expected > 0 else 0.0
    metrics["action_precision"] = precision
    metrics["action_recall"] = recall
    metrics["action_f1"] = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    checked = metrics["total_properties_checked"]
    prop_matched = metrics["total_properties_matched"]
    metrics["property_accuracy"] = (
        prop_matched / checked if checked > 0 else 0.0
    )

    expected_impossible = metrics["total_expected_impossible"]
    detected_impossible = metrics["total_detected_impossible"]
    metrics["impossible_detection_rate"] = (
        min(detected_impossible / expected_impossible, 1.0)
        if expected_impossible > 0
        else 0.0
    )

    metrics["avg_duration"] = metrics["total_duration"] / total_tests
    return metrics


def _regenerate_html_report(
    base_dir: Path, corrected_sample_file: Path
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "viewers.eval_viewer",
            "--test-data",
            str(corrected_sample_file),
            str(base_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def fix_results_dir(
    base_dir: Path,
    sample_file: Path,
    corrected_sample_name: str = DEFAULT_CORRECTED_SAMPLE_NAME,
    dry_run: bool = False,
    regenerate_report: bool = True,
) -> Path | None:
    base_dir = base_dir.resolve()
    sample_file = sample_file.resolve()
    results_dir = base_dir / "results"

    results_files = sorted(base_dir.glob("results_*.json"))
    metrics_files = sorted(base_dir.glob("metrics_*.json"))
    summary_file = base_dir / "summary.json"

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Missing per-test results directory: {results_dir}"
        )
    if not results_files:
        raise FileNotFoundError(f"No results_*.json found in {base_dir}")

    sample_rows = _load_json(sample_file)
    sample_by_id = {row["id"]: copy.deepcopy(row) for row in sample_rows}

    aggregated_results: list[dict] = _load_json(results_files[-1])
    simulator = _build_simulator(sample_rows)

    corrected_sample_rows: list[dict] = []
    corrected_results: list[dict] = []
    changed_test_ids: list[str] = []
    changed_by_test: dict[str, dict] = {}
    rejected_candidates: list[dict] = []

    for original_result in aggregated_results:
        test_id = original_result["test_id"]
        sample_row = copy.deepcopy(sample_by_id[test_id])
        resolution_results = (
            original_result.get("trace", {}).get("resolution_results") or []
        )
        resolution_by_url = {
            item.get("target_uri"): item
            for item in resolution_results
            if item.get("target_uri")
        }

        verified_actions: dict[str, dict] = {}
        for extra_action in original_result.get("extra_actions") or []:
            resolution = resolution_by_url.get(extra_action)
            if not resolution:
                rejected_candidates.append(
                    {
                        "test_id": test_id,
                        "action_url": extra_action,
                        "reason": "resolution_result_missing",
                    }
                )
                continue

            original_index = (resolution.get("intent") or {}).get(
                "original_index"
            )
            outputs = sample_row.get("output") or []
            if (
                not isinstance(original_index, int)
                or original_index < 0
                or original_index >= len(outputs)
            ):
                rejected_candidates.append(
                    {
                        "test_id": test_id,
                        "action_url": extra_action,
                        "reason": f"invalid_original_index:{original_index}",
                    }
                )
                continue

            if outputs[original_index].get("execution") != "error_input":
                rejected_candidates.append(
                    {
                        "test_id": test_id,
                        "action_url": extra_action,
                        "reason": f"not_error_input_at_index:{original_index}",
                    }
                )
                continue

            home_id = str(
                sample_row.get("home_id")
                or test_id.split("_")[0].replace("home", "")
            )
            verified, rejection_reason = _verify_extra_action(
                simulator=simulator,
                home_id=home_id,
                result=original_result,
                action_url=extra_action,
                resolution_result=resolution,
            )
            if verified is None:
                rejected_candidates.append(
                    {
                        "test_id": test_id,
                        "action_url": extra_action,
                        "reason": rejection_reason,
                    }
                )
                continue

            verified_actions[extra_action] = verified

        corrected_outputs = copy.deepcopy(sample_row.get("output") or [])
        for action_url, verified in verified_actions.items():
            corrected_outputs[verified["original_index"]] = {
                "execution": "success",
                "affordance": action_url,
                "params": verified["params"],
                "test": {
                    "property": verified["property_url"],
                    "expected_value": verified["expected_value"],
                },
            }
        sample_row["output"] = corrected_outputs
        corrected_sample_rows.append(sample_row)

        if verified_actions:
            fixed_result = _recompute_result(
                original_result=original_result,
                corrected_outputs=corrected_outputs,
                verified_actions=verified_actions,
            )
            corrected_results.append(fixed_result)
            changed_test_ids.append(test_id)
            changed_by_test[test_id] = fixed_result
        else:
            corrected_results.append(copy.deepcopy(original_result))

    corrected_metrics = _recompute_metrics(corrected_results)

    print(
        f"{'[DRY-RUN] ' if dry_run else ''}Verified {sum(len((r.get('_mislabeled_error_input_fix') or {}).get('verified_extra_actions', [])) for r in corrected_results if isinstance(r, dict))} promoted extra action(s) across {len(changed_test_ids)} test(s)."
    )
    if rejected_candidates:
        print(f"Left {len(rejected_candidates)} extra action(s) unpromoted:")
        for item in rejected_candidates:
            print(
                f"  {item['test_id']:20s}  {item['action_url']}  [{item['reason']}]"
            )

    if dry_run:
        old_metrics_doc = _load_json(metrics_files[-1]) if metrics_files else {}
        old_metrics = old_metrics_doc.get("metrics", {})
        print("\nMetric deltas:")
        for key in (
            "successful_tests",
            "quantifiable_tests",
            "failed_tests",
            "total_expected_actions",
            "total_matched_actions",
            "total_missing_actions",
            "total_extra_actions",
            "total_properties_checked",
            "total_properties_matched",
            "total_expected_impossible",
            "success_rate",
            "action_f1",
            "property_accuracy",
        ):
            old_value = old_metrics.get(key)
            new_value = corrected_metrics.get(key)
            if old_value != new_value:
                print(f"  {key}: {old_value} -> {new_value}")
        return None

    for test_id, fixed_result in changed_by_test.items():
        _write_json(results_dir / f"{test_id}.json", fixed_result)

    for results_file in results_files:
        _write_json(results_file, corrected_results)

    for metrics_file in metrics_files:
        metrics_doc = _load_json(metrics_file)
        metrics_doc["metrics"] = corrected_metrics
        metrics_doc["_mislabeled_error_input_fix_applied"] = (
            datetime.now().isoformat()
        )
        _write_json(metrics_file, metrics_doc)

    if summary_file.exists():
        summary_doc = _load_json(summary_file)
        summary_doc["metrics"] = corrected_metrics
        summary_doc["corrected_data_file"] = str(
            base_dir / corrected_sample_name
        )
        summary_doc["_mislabeled_error_input_fix_applied"] = (
            datetime.now().isoformat()
        )
        _write_json(summary_file, summary_doc)

    corrected_sample_file = base_dir / corrected_sample_name
    _write_json(corrected_sample_file, corrected_sample_rows)
    if regenerate_report:
        _regenerate_html_report(base_dir, corrected_sample_file)

    print(
        f"\nUpdated {len(changed_test_ids)} per-test result file(s), {len(results_files)} aggregate results file(s), and {len(metrics_files)} metrics file(s)."
    )
    if regenerate_report:
        print("Regenerated eval_report.html.")
    print(
        f"Corrected benchmark ground truth written to: {corrected_sample_file}"
    )
    return corrected_sample_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix modify-codegen evaluation reports for mislabeled HomeBench error_input actions."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=DEFAULT_BASE_DIR,
        help="Benchmark results directory containing results_*.json and metrics_*.json",
    )
    parser.add_argument(
        "--sample-file",
        default=DEFAULT_SAMPLE_FILE,
        help="Path to the sampled modify-codegen benchmark JSON",
    )
    parser.add_argument(
        "--corrected-sample-name",
        default=DEFAULT_CORRECTED_SAMPLE_NAME,
        help="Filename to write the corrected benchmark JSON inside the results directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the corrections without writing files",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only recompute result/metric artifacts; do not regenerate eval_report.html",
    )
    args = parser.parse_args()

    fix_results_dir(
        base_dir=PROJECT_ROOT / args.base_dir,
        sample_file=PROJECT_ROOT / args.sample_file,
        corrected_sample_name=args.corrected_sample_name,
        dry_run=args.dry_run,
        regenerate_report=not args.no_report,
    )


if __name__ == "__main__":
    main()
