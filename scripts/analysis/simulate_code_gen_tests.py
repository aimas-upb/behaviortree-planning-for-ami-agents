#!/usr/bin/env python3
"""
Validate modify-codegen benchmark examples against the smart home simulator.

This script executes the generated Python BT code for each benchmark example
and verifies only the modify-target properties. For multi-action originals, the
scope is restricted via each intent's ``source_output_index`` instead of
checking every successful original action.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import PROJECT_ROOT, resolve_repo_path
from scripts.experiments.run_homebench import TestCase
from src.execution.code_executor import CodeExecutor
from src.planning.base import Plan

DEFAULT_TEST_CASES_PATH = resolve_repo_path(
    "data/homebench/benchmarks/modify_codegen/all_examples.jsonl"
)
CONVERTED_DATA_PATH = resolve_repo_path("data/homebench/converted")
SIMULATOR_DATA_PATH = resolve_repo_path(
    "data/homebench/hmas/home_description"
)
SIMULATOR_URL = "http://localhost:8080"
SIMULATOR_HEALTH_URL = f"{SIMULATOR_URL}/health"
SIMULATOR_RESET_URL = f"{SIMULATOR_URL}/reset"


@dataclass(frozen=True)
class StateExpectation:
    property_url: str
    expected_value: Any
    source_output_index: Optional[int] = None


@dataclass
class BenchmarkCase:
    raw_case: dict[str, Any]
    original_test: Optional[TestCase] = None
    expectations: list[StateExpectation] = field(default_factory=list)
    metadata_error: Optional[str] = None

    @property
    def test_id(self) -> str:
        return str(self.raw_case.get("id", "<unknown>"))

    @property
    def home_id(self) -> str:
        raw_home_id = self.raw_case.get("home_id")
        if raw_home_id is not None:
            text = str(raw_home_id)
            return text if text.startswith("home") else f"home{text}"

        match = re.match(r"(home\d+)_", self.test_id)
        if not match:
            raise ValueError(f"Could not infer home id for {self.test_id}")
        return match.group(1)

    @property
    def target_code(self) -> str:
        return str(self.raw_case.get("target_code", ""))


@dataclass
class ValidationResult:
    test_id: str
    home_id: str
    execution_success: bool = False
    execution_ticks: int = 0
    execution_status: str = ""
    execution_error: Optional[str] = None
    metadata_error: Optional[str] = None
    property_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def state_success(self) -> bool:
        return bool(self.property_results) and all(
            result["matched"] for result in self.property_results
        )

    @property
    def success(self) -> bool:
        return (
            self.metadata_error is None
            and self.execution_success
            and self.state_success
        )


_converted_cache: dict[str, list[dict[str, Any]]] = {}


def configure_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "httpx",
        "httpcore",
        "src.execution.code_executor",
        "behavior_trees.affordance_nodes",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def resolve_test_cases_path(path_str: str | Path) -> Path:
    path = resolve_repo_path(path_str)
    if path.is_dir():
        path = path / "all_examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")
    return path


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def extract_original_test(test_id: str, source_file: str) -> TestCase:
    if source_file not in _converted_cache:
        path = CONVERTED_DATA_PATH / f"{source_file}.json"
        if not path.exists():
            raise FileNotFoundError(f"Converted test file not found: {path}")
        _converted_cache[source_file] = json.loads(path.read_text())

    test = next(
        (row for row in _converted_cache[source_file] if row["id"] == test_id),
        None,
    )
    if test is None:
        raise ValueError(
            f"Original test case {test_id!r} not found in {source_file}.json"
        )

    return TestCase.from_dict(test)


def build_modify_expectations(
    code_gen_test: dict[str, Any], original_test: TestCase
) -> list[StateExpectation]:
    intents = code_gen_test.get("intents") or []
    expectations_by_property: dict[str, StateExpectation] = {}

    # source_output_index points into the original converted output list, which
    # lets us validate only the modify-aligned actions from the source test.
    for intent in intents:
        action = intent.get("action") or {}
        if action.get("verb") != "modify":
            continue

        resolved = intent.get("resolved") or {}
        property_url = resolved.get("property_url")
        expected_value = intent.get("expected_value")
        source_output_index = intent.get("source_output_index")

        if source_output_index is not None:
            if not isinstance(source_output_index, int):
                raise ValueError(
                    f"{code_gen_test['id']}: invalid source_output_index "
                    f"{source_output_index!r}"
                )

            try:
                original_output = original_test.expected_outputs[
                    source_output_index
                ]
            except IndexError as exc:
                raise ValueError(
                    f"{code_gen_test['id']}: source_output_index "
                    f"{source_output_index} is out of range"
                ) from exc

            if original_output.get("execution") != "success":
                raise ValueError(
                    f"{code_gen_test['id']}: source output {source_output_index} "
                    "is not a successful action"
                )

            original_test_spec = original_output.get("test") or {}
            original_property_url = original_test_spec.get("property")
            original_expected_value = original_test_spec.get("expected_value")

            if property_url != original_property_url:
                raise ValueError(
                    f"{code_gen_test['id']}: property mismatch for source output "
                    f"{source_output_index}: benchmark={property_url!r}, "
                    f"original={original_property_url!r}"
                )

            if expected_value != original_expected_value:
                raise ValueError(
                    f"{code_gen_test['id']}: expected value mismatch for source "
                    f"output {source_output_index}: benchmark={expected_value!r}, "
                    f"original={original_expected_value!r}"
                )

        if property_url is None:
            raise ValueError(
                f"{code_gen_test['id']}: missing property_url for modify intent"
            )

        existing = expectations_by_property.get(property_url)
        if existing is not None and existing.expected_value != expected_value:
            raise ValueError(
                f"{code_gen_test['id']}: conflicting expected values for "
                f"{property_url}: {existing.expected_value!r} vs {expected_value!r}"
            )

        expectations_by_property[property_url] = StateExpectation(
            property_url=property_url,
            expected_value=expected_value,
            source_output_index=source_output_index,
        )

    expectations = list(expectations_by_property.values())
    if not expectations:
        raise ValueError(
            f"{code_gen_test['id']}: no modify expectations could be derived"
        )
    return expectations


def load_code_gen_tests(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    resolved_path = resolve_test_cases_path(path)

    for item in load_records(resolved_path):
        case = BenchmarkCase(raw_case=item)
        try:
            source_file = str(item["source_file"])
            case.original_test = extract_original_test(item["id"], source_file)
            case.expectations = build_modify_expectations(
                item, case.original_test
            )
            if not case.target_code.strip():
                raise ValueError(f"{case.test_id}: target_code is empty")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            case.metadata_error = str(exc)
        cases.append(case)

    return cases


def filter_cases(
    cases: list[BenchmarkCase],
    limit: Optional[int] = None,
    test_ids: Optional[set[str]] = None,
    homes: Optional[set[str]] = None,
) -> list[BenchmarkCase]:
    filtered = cases

    if test_ids:
        filtered = [case for case in filtered if case.test_id in test_ids]

    if homes:
        filtered = [case for case in filtered if case.home_id in homes]

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


def query_simulator_state(
    client: httpx.Client, property_uri: str
) -> Any:
    response = client.get(property_uri)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def reset_home(client: httpx.Client, home_id: str) -> None:
    response = client.post(
        SIMULATOR_RESET_URL,
        json={"home": home_id.replace("home", "")},
    )
    response.raise_for_status()


def is_simulator_healthy(client: httpx.Client) -> bool:
    try:
        response = client.get(SIMULATOR_HEALTH_URL)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False
    return payload.get("status") == "healthy"


def start_simulator(home_ids: list[str]) -> subprocess.Popen[Any]:
    command = [
        sys.executable,
        "-m",
        "homebench.smart_home_simulator",
        "--data-dir",
        str(SIMULATOR_DATA_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]

    unique_home_ids = sorted({home_id.replace("home", "") for home_id in home_ids})
    if unique_home_ids:
        command.extend(["--home", ",".join(unique_home_ids)])

    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_simulator_running(
    client: httpx.Client,
    home_ids: list[str],
    startup_timeout_seconds: float,
    no_start_simulator: bool,
) -> Optional[subprocess.Popen[Any]]:
    if is_simulator_healthy(client):
        print(
            "Reusing existing simulator on http://localhost:8080. "
            "The required homes must already be loaded."
        )
        return None

    if no_start_simulator:
        raise RuntimeError(
            "Simulator is not reachable at http://localhost:8080 and "
            "--no-start-simulator was set."
        )

    process = start_simulator(home_ids)
    deadline = time.time() + startup_timeout_seconds

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Simulator process exited before becoming healthy. "
                "Make sure port 8080 is available and dependencies are installed."
            )
        if is_simulator_healthy(client):
            return process
        time.sleep(0.5)

    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError(
        "Timed out waiting for the simulator to become healthy."
    )


def stop_simulator(process: Optional[subprocess.Popen[Any]]) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def validate_case(
    case: BenchmarkCase,
    code_executor: CodeExecutor,
    client: httpx.Client,
) -> ValidationResult:
    result = ValidationResult(
        test_id=case.test_id,
        home_id=case.home_id,
        metadata_error=case.metadata_error,
    )

    if case.metadata_error is not None:
        return result

    try:
        reset_home(client, case.home_id)
    except Exception as exc:
        result.execution_error = (
            f"Failed to reset {case.home_id} before execution: {exc}"
        )
        return result

    execution = code_executor.execute(
        Plan(format="python_code", content=case.target_code)
    )
    result.execution_success = execution.success
    result.execution_ticks = execution.ticks
    result.execution_status = execution.final_status
    result.execution_error = execution.error

    for expectation in case.expectations:
        property_result: dict[str, Any] = {
            "property_url": expectation.property_url,
            "expected_value": expectation.expected_value,
            "source_output_index": expectation.source_output_index,
            "actual_value": None,
            "matched": False,
            "error": None,
        }

        try:
            actual_value = query_simulator_state(client, expectation.property_url)
            property_result["actual_value"] = actual_value
            property_result["matched"] = actual_value == expectation.expected_value
        except Exception as exc:
            property_result["error"] = str(exc)

        result.property_results.append(property_result)

    return result


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    state_matches = sum(1 for item in result.property_results if item["matched"])
    state_total = len(result.property_results)
    status = "PASS" if result.success else "FAIL"

    execution_summary = (
        f"execution={result.execution_status or 'NOT_RUN'}"
        f", ticks={result.execution_ticks}"
    )
    if result.execution_error:
        execution_summary += f", error={result.execution_error}"

    print(
        f"[{status}] {result.test_id} ({result.home_id}) "
        f"{execution_summary}, state={state_matches}/{state_total}"
    )

    if result.metadata_error is not None:
        print(f"  metadata_error: {result.metadata_error}")
        return

    for property_result in result.property_results:
        if property_result["matched"] and not verbose:
            continue

        message = (
            f"  property={property_result['property_url']} "
            f"expected={property_result['expected_value']!r} "
            f"actual={property_result['actual_value']!r}"
        )
        if property_result["source_output_index"] is not None:
            message += (
                f" source_output_index={property_result['source_output_index']}"
            )
        if property_result["error"]:
            message += f" error={property_result['error']}"
        print(message)


def print_summary(results: list[ValidationResult]) -> None:
    passed = sum(result.success for result in results)
    failed = len(results) - passed
    metadata_failures = sum(result.metadata_error is not None for result in results)
    execution_failures = sum(
        result.metadata_error is None and not result.execution_success
        for result in results
    )
    state_failures = sum(
        result.metadata_error is None
        and result.execution_success
        and not result.state_success
        for result in results
    )
    total_properties = sum(len(result.property_results) for result in results)
    matched_properties = sum(
        sum(1 for item in result.property_results if item["matched"])
        for result in results
    )

    print(
        "\nSummary: "
        f"passed={passed}, failed={failed}, "
        f"metadata_failures={metadata_failures}, "
        f"execution_failures={execution_failures}, "
        f"state_failures={state_failures}, "
        f"properties={matched_properties}/{total_properties}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute modify-codegen benchmark cases and verify that only the "
            "modify-aligned simulator states match the expected values."
        )
    )
    parser.add_argument(
        "--test-cases",
        default=str(DEFAULT_TEST_CASES_PATH),
        help=(
            "Path to a benchmark file or directory. Directories default to "
            "all_examples.jsonl."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N benchmark cases after filtering.",
    )
    parser.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="Restrict validation to one or more specific benchmark ids.",
    )
    parser.add_argument(
        "--home",
        action="append",
        default=[],
        help="Restrict validation to one or more homes (e.g. 86 or home86).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=10,
        help="Maximum number of BT ticks per benchmark example.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a newly started simulator to become healthy.",
    )
    parser.add_argument(
        "--no-start-simulator",
        action="store_true",
        help="Require an already running simulator on localhost:8080.",
    )
    parser.add_argument(
        "--keep-simulator",
        action="store_true",
        help="Do not stop the simulator if this script started it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-property details for passing cases too.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    cases = load_code_gen_tests(args.test_cases)

    homes = {
        value if str(value).startswith("home") else f"home{value}"
        for value in args.home
    } or None
    test_ids = set(args.test_id) or None

    cases = filter_cases(
        cases,
        limit=args.limit,
        test_ids=test_ids,
        homes=homes,
    )

    if not cases:
        raise ValueError("No benchmark cases matched the requested filters.")

    client = httpx.Client(timeout=30.0)
    code_executor = CodeExecutor(max_ticks=args.max_ticks)
    simulator_process: Optional[subprocess.Popen[Any]] = None

    try:
        simulator_process = ensure_simulator_running(
            client=client,
            home_ids=[case.home_id for case in cases],
            startup_timeout_seconds=args.startup_timeout,
            no_start_simulator=args.no_start_simulator,
        )

        results: list[ValidationResult] = []
        for case in cases:
            result = validate_case(case, code_executor, client)
            results.append(result)
            print_result(result, verbose=args.verbose)

        print_summary(results)
        return 0 if all(result.success for result in results) else 1
    finally:
        client.close()
        if simulator_process is not None and not args.keep_simulator:
            stop_simulator(simulator_process)


if __name__ == "__main__":
    raise SystemExit(main())
