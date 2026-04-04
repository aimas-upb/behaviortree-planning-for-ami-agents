"""
Fix misreported results caused by the mixed JSON-IR + code-plan concatenation bug.

Root cause
----------
When the experience pipeline produces both a matched JSON-IR plan (from an
adapted experience) and an unmatched code plan (from full discovery+planning),
the runner's Step-5 concatenation drops the code plan because it cannot mix
the two formats.  The code plan was already executed in Step 4 (to extract its
JSON-IR), which correctly updated the simulator state.  But `combined_plan_ir`
only contains the matched JSON-IR leaf, so `actions_in_plan` is incomplete and
the action-comparison step produces spurious "Missing actions" / "Extra actions"
entries — causing tests that fully succeeded to be marked as False or Quantifiable.

Fix strategy (reporting-only, no re-runs)
-----------------------------------------
For each affected result (matched_plans AND code_plans both non-empty in the
trace), reconstruct the true `actions_in_plan` as:

    matched_plan_action_urls  ∪  unmatched_code_plan_action_urls

Re-derive matched/missing/extra and the success verdict using the same logic as
the original evaluator.  The property_results list (live simulator verification)
is preserved unchanged — it is the authoritative ground truth for what actually
happened in the simulator.

Only results whose verdict changes are written back; a dry-run mode is provided.
The original files are backed up before any write.
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.common import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_action_urls(node: dict) -> list[str]:
    """Recursively collect action_url values from a JSON-IR tree."""
    if not isinstance(node, dict):
        return []
    if node.get("type") == "action":
        url = node.get("action_url")
        return [url] if url else []
    urls: list[str] = []
    for child in node.get("children", []):
        urls.extend(_extract_action_urls(child))
    return urls


def _corrected_actions_in_plan(trace: dict) -> list[str] | None:
    """
    Return the corrected actions_in_plan for an affected test, or None if the
    test is not affected (i.e. does not have both matched_plans and code_plans).
    """
    matched_plans = trace.get("matched_plans") or []
    unmatched_plans = trace.get("unmatched_plans") or []
    code_plans = [u for u in unmatched_plans if u.get("_code_plan")]

    if not matched_plans or not code_plans:
        return None  # not the affected scenario

    # Matched plan action URLs (each adapted plan is a single action leaf)
    matched_urls: list[str] = []
    for mp in matched_plans:
        url = mp.get("action_url")
        if url:
            matched_urls.append(url)

    # Unmatched code plan action URLs (from the JSON-IR extracted during step 4)
    unmatched_urls: list[str] = []
    for cp in code_plans:
        json_ir = cp.get("_json_ir")
        if json_ir:
            unmatched_urls.extend(_extract_action_urls(json_ir))

    # Deduplicate while preserving rough order (matched first)
    seen: set[str] = set()
    combined: list[str] = []
    for url in matched_urls + unmatched_urls:
        if url not in seen:
            seen.add(url)
            combined.append(url)

    return combined


def _recompute_result(result: dict) -> dict | None:
    """
    Recompute a single result dict.  Returns a corrected copy if the verdict
    changes, or None if no change is needed.
    """
    trace = result.get("trace") or {}
    corrected_actions = _corrected_actions_in_plan(trace)
    if corrected_actions is None:
        return None  # not affected

    expected_set = set(result.get("expected_actions") or [])
    corrected_set = set(corrected_actions)

    corrected_matched = sorted(expected_set & corrected_set)
    corrected_missing = sorted(expected_set - corrected_set)
    corrected_extra   = sorted(corrected_set - expected_set)

    props_matched = result.get("properties_matched", 0)
    props_checked = result.get("properties_checked", 0)
    all_props_ok  = (props_matched == props_checked)
    exec_ok       = result.get("execution_success", False)
    is_error_only = result.get("is_error_input_only", False)

    # Replicate the same success-determination logic as the evaluator
    if is_error_only:
        # Error-only tests are not affected by this bug (they have no expected
        # actions and no matched plans for success goals); skip.
        return None

    if not exec_ok:
        # Execution truly failed; nothing to fix
        return None

    no_extra = len(corrected_extra) == 0
    has_matched = len(corrected_matched) > 0

    if no_extra and all_props_ok:
        new_success = "True"
        new_handled = True
        new_failure_type = None
    elif has_matched:
        new_success = "Quantifiable"
        new_handled = False
        new_failure_type = None
    else:
        new_success = "False"
        new_handled = False
        # Classify failure: if there are extra/missing actions it's still action_mismatch
        new_failure_type = result.get("failure_type")

    old_success = result.get("success")
    if new_success == old_success:
        # Even if actions changed internally, verdict is the same — still update
        # the action fields for consistency, but only if they actually differ.
        if (
            sorted(result.get("actions_in_plan") or []) == sorted(corrected_actions)
            and sorted(result.get("matched_actions") or []) == corrected_matched
            and sorted(result.get("missing_actions") or []) == corrected_missing
            and sorted(result.get("extra_actions")   or []) == corrected_extra
        ):
            return None  # nothing changed at all

    fixed = copy.deepcopy(result)
    fixed["actions_in_plan"]  = corrected_actions
    fixed["matched_actions"]  = corrected_matched
    fixed["missing_actions"]  = corrected_missing
    fixed["extra_actions"]    = corrected_extra
    fixed["success"]          = new_success
    fixed["handled_correctly"] = new_handled
    if new_failure_type is None:
        fixed.pop("failure_type", None)
    else:
        fixed["failure_type"] = new_failure_type
    fixed["_reporting_fix_applied"] = True
    return fixed


def _recompute_metrics(results: list[dict]) -> dict:
    """Recompute the EvaluationMetrics aggregate from a corrected results list."""
    m: dict = {
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

    for r in results:
        s = r.get("success")
        if s == "True":
            m["successful_tests"] += 1
        elif s == "Quantifiable":
            m["quantifiable_tests"] += 1
        else:
            m["failed_tests"] += 1

        if r.get("plan_generated"):
            m["plans_generated"] += 1

        m["total_expected_actions"]  += len(r.get("expected_actions")  or [])
        m["total_matched_actions"]   += len(r.get("matched_actions")   or [])
        m["total_missing_actions"]   += len(r.get("missing_actions")   or [])
        m["total_extra_actions"]     += len(r.get("extra_actions")     or [])
        m["total_properties_checked"] += r.get("properties_checked", 0)
        m["total_properties_matched"] += r.get("properties_matched", 0)
        m["total_expected_impossible"] += r.get("expected_impossible", 0)
        m["total_detected_impossible"] += len(r.get("detected_impossible") or [])
        m["total_duration"]           += r.get("duration", 0.0)

        ft = r.get("failure_type")
        if ft and ft in m["failures_by_type"]:
            m["failures_by_type"][ft] += 1

    # Computed rates (mirror EvaluationMetrics properties)
    total = m["total_tests"] or 1
    m["success_rate"]              = m["successful_tests"] / total
    m["quantifiable_rate"]         = m["quantifiable_tests"] / total
    m["success_or_quantifiable_rate"] = (
        m["successful_tests"] + m["quantifiable_tests"]
    ) / total

    ma  = m["total_matched_actions"]
    ex  = m["total_extra_actions"]
    exp = m["total_expected_actions"]
    precision = ma / (ma + ex)  if (ma + ex) > 0 else 0.0
    recall    = ma / exp         if exp > 0 else 0.0
    m["action_precision"] = precision
    m["action_recall"]    = recall
    m["action_f1"] = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    pc = m["total_properties_checked"]
    pm = m["total_properties_matched"]
    m["property_accuracy"] = pm / pc if pc > 0 else 0.0

    ei = m["total_expected_impossible"]
    di = m["total_detected_impossible"]
    m["impossible_detection_rate"] = min(di / ei, 1.0) if ei > 0 else 0.0

    m["avg_duration"] = m["total_duration"] / total

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _regenerate_html_report(results_dir: Path, test_data: str) -> None:
    """Regenerate eval_report.html via the canonical viewer module."""
    report_file = results_dir / "eval_report.html"
    print(f"\nRegenerating {report_file.name} …")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "viewers.eval_viewer",
                "--test-data", test_data,
                str(results_dir),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        print(f"  Report updated: {report_file}")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: viewers.eval_viewer failed (exit {e.returncode}); report not updated")


def fix_results_dir(results_dir: Path, dry_run: bool = False, test_data: str = "data/homebench/converted/test_data.json") -> None:
    # Exclude backup files (contain ".bak_" in the stem)
    results_files  = sorted(f for f in results_dir.glob("results_*.json") if ".bak_" not in f.name)
    metrics_files  = sorted(f for f in results_dir.glob("metrics_*.json") if ".bak_" not in f.name)

    if not results_files:
        print(f"No results_*.json found in {results_dir}")
        return

    for results_file in results_files:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Processing {results_file.name} …")

        with open(results_file) as f:
            results: list[dict] = json.load(f)

        changed_ids: list[str] = []
        corrected_results: list[dict] = []

        for r in results:
            fixed = _recompute_result(r)
            if fixed is not None:
                old_s = r.get("success")
                new_s = fixed.get("success")
                changed_ids.append(r["test_id"])
                corrected_results.append(fixed)
                verdict_tag = (
                    f"{old_s} -> {new_s}" if old_s != new_s
                    else f"{old_s} (actions corrected)"
                )
                print(f"  CHANGED  {r['test_id']:35s}  {verdict_tag}")
            else:
                corrected_results.append(r)

        print(f"  {len(changed_ids)} result(s) updated out of {len(results)}")

        if not dry_run and changed_ids:
            # Back up original
            backup = results_file.with_suffix(
                f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            shutil.copy2(results_file, backup)
            print(f"  Backed up original to {backup.name}")

            with open(results_file, "w") as f:
                json.dump(corrected_results, f, indent=2, default=str)
            print(f"  Written corrected results to {results_file.name}")

    # Recompute metrics from the final corrected results list
    if not dry_run and results_files:
        # Use the last results file (assumes one per run)
        with open(results_files[-1]) as f:
            final_results = json.load(f)

        corrected_metrics = _recompute_metrics(final_results)

        for metrics_file in metrics_files:
            print(f"\nUpdating {metrics_file.name} …")
            with open(metrics_file) as f:
                metrics_doc: dict = json.load(f)

            backup = metrics_file.with_suffix(
                f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            shutil.copy2(metrics_file, backup)
            print(f"  Backed up original to {backup.name}")

            old_m = metrics_doc.get("metrics", {})
            metrics_doc["metrics"] = corrected_metrics
            metrics_doc["_reporting_fix_applied"] = datetime.now().isoformat()

            with open(metrics_file, "w") as f:
                json.dump(metrics_doc, f, indent=2, default=str)

            print(f"  Metric changes:")
            for key in (
                "successful_tests", "quantifiable_tests", "failed_tests",
                "total_matched_actions", "total_missing_actions", "total_extra_actions",
                "success_rate", "action_precision", "action_recall", "action_f1",
            ):
                old_v = old_m.get(key, "–")
                new_v = corrected_metrics.get(key, "–")
                if old_v != new_v:
                    if isinstance(new_v, float):
                        print(f"    {key}: {old_v:.4f} -> {new_v:.4f}"
                              if isinstance(old_v, float)
                              else f"    {key}: {old_v} -> {new_v:.4f}")
                    else:
                        print(f"    {key}: {old_v} -> {new_v}")

        if (results_dir / "eval_report.html").exists() or changed_ids:
            _regenerate_html_report(results_dir, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix misreported results from the mixed JSON-IR/code-plan bug."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="experiments/results/experience_reuse_structured_gpt-5-mini",
        help="Directory containing results_*.json and metrics_*.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files",
    )
    parser.add_argument(
        "--test-data",
        default="data/homebench/converted/test_data.json",
        help="Path to original test data JSON (passed to viewers.eval_viewer)",
    )
    args = parser.parse_args()

    fix_results_dir(Path(args.results_dir), dry_run=args.dry_run, test_data=args.test_data)
