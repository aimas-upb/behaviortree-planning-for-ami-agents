#!/usr/bin/env python3
"""
HomeBench Evaluation Viewer.

Generates a comprehensive interactive HTML report for HomeBench evaluation results.
Shows full traces, ground truth, generated plans, and detailed failure analysis.

Usage:
    uv run python -m viewers.eval_viewer experiments/results/homebench_*/
    uv run python -m viewers.eval_viewer --output report.html experiments/results/homebench_test/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from scripts.common import resolve_repo_path


def _html_escape(text) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _format_uri(uri: str) -> str:
    """Format a URI for display, extracting the key parts."""
    if not uri:
        return ""
    parts = uri.split("/")
    try:
        if "artifacts" in parts:
            art_idx = parts.index("artifacts") + 1
            remaining = "/".join(parts[art_idx:])
            ws_idx = parts.index("workspaces") + 1
            return f"{parts[ws_idx]}/{remaining}"
    except (ValueError, IndexError):
        pass
    return uri.split("/")[-1] if "/" in uri else uri


def _format_params(params: dict) -> str:
    """Format parameters for display."""
    if not params:
        return "(none)"
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in params.items())


def _format_json(obj, indent: int = 2) -> str:
    """Format JSON for display."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except:
        return str(obj)


def generate_css() -> str:
    """Generate all CSS styles."""
    return """
    <style>
        * { box-sizing: border-box; }
        body {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { color: #38bdf8; text-align: center; margin-bottom: 5px; }
        h2 { color: #94a3b8; font-size: 16px; margin: 20px 0 10px; }
        .subtitle { text-align: center; color: #64748b; margin-bottom: 30px; }

        /* Info box for explanations */
        .info-box {
            background: #1e3a5f;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 25px;
            font-size: 14px;
        }
        .info-box h3 { color: #60a5fa; margin: 0 0 10px; font-size: 15px; }
        .info-box p { margin: 5px 0; color: #cbd5e1; }
        .info-box code { background: #0f172a; padding: 2px 6px; border-radius: 3px; }

        /* Config summary */
        .config-summary {
            display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;
            padding: 15px; background: #1e293b; border-radius: 8px; margin-bottom: 20px; font-size: 14px;
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px; margin-bottom: 25px;
        }
        .metric-card {
            background: #1e293b; border-radius: 10px; padding: 16px; text-align: center;
            border: 2px solid #334155;
        }
        .metric-card.success { border-color: #22c55e; }
        .metric-card.warning { border-color: #eab308; }
        .metric-card.error { border-color: #ef4444; }
        .metric-value { font-size: 28px; font-weight: bold; color: #f8fafc; }
        .metric-label { font-size: 13px; color: #94a3b8; margin-top: 4px; }
        .metric-detail { font-size: 11px; color: #64748b; margin-top: 3px; }

        /* Filters */
        .filters { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
        .filter-btn {
            padding: 8px 16px; border: 2px solid #334155; background: #1e293b;
            color: #e2e8f0; border-radius: 6px; cursor: pointer; font-size: 14px; transition: all 0.2s;
        }
        .filter-btn:hover { border-color: #38bdf8; }
        .filter-btn.active { border-color: #38bdf8; background: #0c4a6e; }
        .filter-count { margin-left: 5px; opacity: 0.7; }
        .search-input {
            padding: 8px 16px; border: 2px solid #334155; background: #1e293b;
            color: #e2e8f0; border-radius: 6px; font-size: 14px; width: 300px;
        }
        .search-input:focus { outline: none; border-color: #38bdf8; }

        /* Test cards */
        .test-card {
            background: #1e293b; border-radius: 8px; margin-bottom: 12px;
            border: 2px solid #334155; overflow: hidden;
        }
        .test-card.success { border-left: 4px solid #22c55e; }
        .test-card.quantifiable { border-left: 4px solid #f59e0b; }
        .test-card.error { border-left: 4px solid #ef4444; }
        .test-header {
            display: flex; align-items: center; padding: 12px 16px; cursor: pointer; gap: 12px;
        }
        .test-header:hover { background: #334155; }
        .test-status { font-weight: bold; font-size: 12px; padding: 4px 10px; border-radius: 4px; }
        .test-status.success { background: #166534; color: #86efac; }
        .test-status.quantifiable { background: #92400e; color: #fbbf24; }
        .test-status.error { background: #7f1d1d; color: #fca5a5; }
        .test-id { font-family: monospace; color: #94a3b8; flex-grow: 1; }
        .test-expand-icon { color: #64748b; transition: transform 0.2s; font-size: 12px; }
        .test-card.expanded .test-expand-icon { transform: rotate(180deg); }

        .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 500; }
        .badge-impossible { background: #7c3aed; color: #e9d5ff; }
        .badge-normal { background: #334155; color: #94a3b8; }
        .trace-link {
            font-size: 12px; padding: 4px 10px; border-radius: 4px;
            background: #0c4a6e; color: #38bdf8; text-decoration: none;
            margin-left: 8px; transition: background 0.2s;
        }
        .trace-link:hover { background: #075985; }

        .test-goal {
            padding: 8px 16px 12px; color: #f1f5f9; font-size: 15px;
            border-bottom: 1px solid #334155; font-style: italic;
        }
        .test-details { display: none; padding: 0; background: #0f172a; }
        .test-card.expanded .test-details { display: block; }

        /* Sections within test details */
        .detail-section {
            border-bottom: 1px solid #1e293b; padding: 16px;
        }
        .detail-section:last-child { border-bottom: none; }
        .section-header {
            display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
            cursor: pointer;
        }
        .section-title {
            font-weight: 600; font-size: 14px; color: #94a3b8;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .section-icon { color: #64748b; font-size: 10px; transition: transform 0.2s; }
        .detail-section.collapsed .section-content { display: none; }
        .detail-section.collapsed .section-icon { transform: rotate(-90deg); }

        /* Ground truth */
        .ground-truth {
            background: #1a2744; border: 1px solid #3b82f6; border-radius: 6px; padding: 12px;
        }
        .gt-label { color: #60a5fa; font-size: 12px; font-weight: 600; margin-bottom: 6px; }
        .gt-action {
            background: #0f172a; padding: 8px 12px; border-radius: 4px; margin: 6px 0;
            font-family: monospace; font-size: 13px;
        }
        .gt-action .action-name { color: #22c55e; }
        .gt-action .params { color: #fbbf24; margin-left: 10px; }
        .gt-test { color: #94a3b8; font-size: 12px; margin-top: 4px; }
        .gt-test .prop { color: #60a5fa; }
        .gt-test .expected { color: #22c55e; }

        /* Impossible request explanation */
        .impossible-box {
            background: #2d1f4e; border: 1px solid #7c3aed; border-radius: 6px; padding: 12px;
        }
        .impossible-box h4 { color: #a78bfa; margin: 0 0 8px; font-size: 14px; }
        .impossible-box p { color: #c4b5fd; margin: 4px 0; font-size: 13px; }

        /* Comparison boxes */
        .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        @media (max-width: 900px) { .comparison-grid { grid-template-columns: 1fr; } }

        .comparison-box {
            background: #1e293b; border-radius: 6px; padding: 12px;
            border: 1px solid #334155;
        }
        .comparison-box.expected { border-color: #3b82f6; }
        .comparison-box.actual { border-color: #f59e0b; }
        .comparison-box h4 {
            margin: 0 0 10px; font-size: 13px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .comparison-box.expected h4 { color: #60a5fa; }
        .comparison-box.actual h4 { color: #fbbf24; }

        /* Action lists */
        .action-list { list-style: none; padding: 0; margin: 0; }
        .action-list li {
            padding: 6px 10px; background: #0f172a; border-radius: 4px; margin: 4px 0;
            font-family: monospace; font-size: 12px; display: flex; flex-wrap: wrap; gap: 8px;
        }
        .action-url { color: #e2e8f0; word-break: break-all; }
        .action-params { color: #fbbf24; }
        .action-status { margin-left: auto; font-weight: bold; }
        .action-status.matched { color: #22c55e; }
        .action-status.missing { color: #ef4444; }
        .action-status.extra { color: #f59e0b; }

        /* Code blocks */
        pre.code-block {
            background: #0d1117; padding: 12px; border-radius: 6px;
            overflow-x: auto; font-size: 12px; line-height: 1.4;
            max-height: 400px; overflow-y: auto;
        }

        /* Execution result */
        .exec-result {
            display: flex; flex-wrap: wrap; gap: 15px; padding: 12px;
            background: #1e293b; border-radius: 6px;
        }
        .exec-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
        .exec-label { color: #94a3b8; }
        .exec-value { font-weight: 600; }
        .exec-value.success { color: #22c55e; }
        .exec-value.failure { color: #ef4444; }

        /* Property verification */
        .prop-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .prop-table th {
            text-align: left; padding: 8px 10px; background: #1e293b; color: #94a3b8; font-weight: 500;
        }
        .prop-table td { padding: 8px 10px; border-bottom: 1px solid #334155; }
        .prop-table tr.match { background: rgba(34, 197, 94, 0.1); }
        .prop-table tr.mismatch { background: rgba(239, 68, 68, 0.1); }
        .match-icon { font-size: 14px; }
        .match-icon.yes { color: #22c55e; }
        .match-icon.no { color: #ef4444; }

        /* Failure analysis */
        .failure-analysis {
            background: #2d1f1f; border: 1px solid #ef4444; border-radius: 6px; padding: 12px;
        }
        .failure-analysis h4 { color: #fca5a5; margin: 0 0 10px; font-size: 14px; }
        
        /* Partial success analysis */
        .partial-analysis {
            background: #2d2a1f; border: 1px solid #f59e0b; border-radius: 6px; padding: 12px;
        }
        .partial-analysis h4 { color: #fbbf24; margin: 0 0 10px; font-size: 14px; }
        
        .failure-reason { color: #fecaca; font-size: 13px; margin: 4px 0; }
        .failure-reason strong { color: #f87171; }
        .partial-analysis .failure-reason { color: #fef3c7; }
        .partial-analysis .failure-reason strong { color: #fbbf24; }

        /* Failure type badges */
        .failure-type-badge {
            font-size: 10px; padding: 2px 8px; border-radius: 10px;
            font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .failure-type-badge.parse_error { background: #7c3aed; color: #e9d5ff; }
        .failure-type-badge.compilation_error { background: #dc2626; color: #fecaca; }
        .failure-type-badge.execution_error { background: #ea580c; color: #fed7aa; }
        .failure-type-badge.action_mismatch { background: #ca8a04; color: #fef3c7; }
        .failure-type-badge.property_mismatch { background: #0891b2; color: #cffafe; }
        .failure-type-badge.error_input_not_detected { background: #db2777; color: #fbcfe8; }
        .failure-type-badge.other { background: #4b5563; color: #e5e7eb; }

        /* Hidden */
        .test-card.hidden { display: none; }
    </style>
    """


def generate_info_box() -> str:
    """Generate the explanation box for terminology."""
    return """
    <div class="info-box">
        <h3>Understanding This Report</h3>
        <p><strong>Ground Truth:</strong> The expected correct behavior from the HomeBench dataset - which actions should be called with what parameters, and what property values should result.</p>
        <p><strong>Impossible Request:</strong> Test cases where the user asks for something that <em>cannot</em> be done with the available devices. For example, asking to "set brightness to 90" on a light that only has on/off capability (no brightness control). The system should recognize these and <em>not</em> attempt them - success means correctly identifying the request as impossible.</p>
        <p><strong>Result Types:</strong></p>
        <ul style="margin: 5px 0 5px 20px; color: #cbd5e1;">
            <li><strong style="color: #22c55e;">Pass (Success)</strong> - All expected actions matched, all properties verified correctly</li>
            <li><strong style="color: #fbbf24;">Partial (Quantifiable)</strong> - Plan executed but with some missing/extra actions or property mismatches</li>
            <li><strong style="color: #ef4444;">Fail</strong> - No plan generated, execution failed, or impossible request not detected</li>
        </ul>
        <p><strong>Action Matching:</strong> <code>Matched</code> = correctly identified action, <code>Missing</code> = expected action not in plan, <code>Extra</code> = unexpected action added to plan.</p>
        <p><strong>Property Verification:</strong> After execution, we check if device properties match expected values (e.g., temperature is actually 21 after setting it).</p>
    </div>
    """


def generate_metrics_card(metrics: dict) -> str:
    """Generate the metrics summary card."""
    success_rate = metrics.get("success_rate", 0) * 100
    quantifiable_rate = metrics.get("quantifiable_rate", 0) * 100
    success_or_quantifiable_rate = metrics.get("success_or_quantifiable_rate", 0) * 100
    action_precision = metrics.get("action_precision", 0) * 100
    action_recall = metrics.get("action_recall", 0) * 100
    action_f1 = metrics.get("action_f1", 0) * 100
    property_accuracy = metrics.get("property_accuracy", 0) * 100
    impossible_detection_rate = metrics.get("impossible_detection_rate", 0) * 100

    success_class = "success" if success_rate >= 50 else "warning" if success_rate >= 25 else "error"
    quantifiable_class = "warning"
    combined_class = "success" if success_or_quantifiable_rate >= 50 else "warning" if success_or_quantifiable_rate >= 25 else "error"

    # Failure type breakdown
    failures_by_type = metrics.get("failures_by_type", {})
    failed_tests = metrics.get("failed_tests", 0)
    failure_breakdown_html = ""
    if failed_tests > 0 and failures_by_type:
        breakdown_items = []
        for ftype, count in failures_by_type.items():
            if count > 0:
                pct = count / failed_tests * 100
                breakdown_items.append(f"{ftype}: {count} ({pct:.0f}%)")
        if breakdown_items:
            failure_breakdown_html = f"""
            <div class="metric-card error">
                <div class="metric-value">{failed_tests}</div>
                <div class="metric-label">Failed Tests</div>
                <div class="metric-detail">{', '.join(breakdown_items[:3])}</div>
            </div>
            """

    return f"""
    <div class="metrics-grid">
        <div class="metric-card {success_class}">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Full Success</div>
            <div class="metric-detail">{metrics.get('successful_tests', 0)}/{metrics.get('total_tests', 0)} tests</div>
        </div>
        <div class="metric-card {quantifiable_class}">
            <div class="metric-value">{quantifiable_rate:.1f}%</div>
            <div class="metric-label">Quantifiable</div>
            <div class="metric-detail">{metrics.get('quantifiable_tests', 0)} partial success</div>
        </div>
        <div class="metric-card {combined_class}">
            <div class="metric-value">{success_or_quantifiable_rate:.1f}%</div>
            <div class="metric-label">Success + Quantifiable</div>
            <div class="metric-detail">{metrics.get('successful_tests', 0) + metrics.get('quantifiable_tests', 0)}/{metrics.get('total_tests', 0)} tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{action_f1:.1f}%</div>
            <div class="metric-label">Action F1</div>
            <div class="metric-detail">P:{action_precision:.0f}% R:{action_recall:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{property_accuracy:.1f}%</div>
            <div class="metric-label">Property Match</div>
            <div class="metric-detail">{metrics.get('total_properties_matched', 0)}/{metrics.get('total_properties_checked', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{impossible_detection_rate:.1f}%</div>
            <div class="metric-label">Impossible Detected</div>
            <div class="metric-detail">{metrics.get('total_detected_impossible', 0)}/{metrics.get('total_expected_impossible', 0)} sub-goals</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('avg_duration', 0):.1f}s</div>
            <div class="metric-label">Avg Duration</div>
            <div class="metric-detail">Total: {metrics.get('total_duration', 0):.0f}s</div>
        </div>
        {failure_breakdown_html}
    </div>
    """


def generate_ground_truth_section(test_data: dict, is_error_input: bool) -> str:
    """Generate the ground truth section."""
    if is_error_input:
        return f"""
        <div class="detail-section">
            <div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="section-icon">&#9660;</span>
                <span class="section-title">Ground Truth (Expected Behavior)</span>
            </div>
            <div class="section-content">
                <div class="impossible-box">
                    <h4>This is an Impossible Request</h4>
                    <p>The user is asking for something that <strong>cannot be done</strong> with the available devices.</p>
                    <p>The system should recognize this and <strong>not attempt</strong> any action, or fail gracefully.</p>
                    <p><strong>Success criteria:</strong> No successful execution of an action, or recognizing the request is invalid.</p>
                </div>
            </div>
        </div>
        """

    outputs = test_data.get("output", [])
    success_outputs = [o for o in outputs if o.get("execution") == "success"]

    if not success_outputs:
        return ""

    actions_html = ""
    for out in success_outputs:
        affordance = out.get("affordance", "")
        params = out.get("params", {})
        test = out.get("test", {})

        action_name = _format_uri(affordance)
        params_str = _format_params(params) if params else ""

        test_html = ""
        if test:
            prop = _format_uri(test.get("property", ""))
            expected = test.get("expected_value")
            test_html = f'<div class="gt-test">Verify: <span class="prop">{_html_escape(prop)}</span> = <span class="expected">{_html_escape(str(expected))}</span></div>'

        actions_html += f"""
        <div class="gt-action">
            <span class="action-name">{_html_escape(action_name)}</span>
            <span class="params">{_html_escape(params_str)}</span>
            {test_html}
        </div>
        """

    return f"""
    <div class="detail-section">
        <div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="section-icon">&#9660;</span>
            <span class="section-title">Ground Truth (Expected Behavior)</span>
        </div>
        <div class="section-content">
            <div class="ground-truth">
                <div class="gt-label">Expected Actions & Results</div>
                {actions_html}
            </div>
        </div>
    </div>
    """


def generate_action_comparison_section(result: dict) -> str:
    """Generate the action comparison section."""
    matched = result.get("matched_actions", [])
    missing = result.get("missing_actions", [])
    extra = result.get("extra_actions", [])
    expected_params = result.get("expected_params", {})
    actual_params = result.get("params_in_plan", {})

    # Reconstruct expected and actual from matched/missing/extra if not provided
    expected = result.get("expected_actions", [])
    if not expected:
        expected = list(set(matched + missing))

    actual = result.get("actions_in_plan", [])
    if not actual:
        actual = list(set(matched + extra))

    # Build expected actions list
    expected_html = "<ul class='action-list'>"
    if expected:
        for action in expected:
            params = expected_params.get(action, {})
            status = "matched" if action in matched else "missing"
            status_text = "MATCHED" if action in matched else "MISSING"
            expected_html += f"""
            <li>
                <span class="action-url">{_html_escape(_format_uri(action))}</span>
                <span class="action-params">{_html_escape(_format_params(params))}</span>
                <span class="action-status {status}">{status_text}</span>
            </li>
            """
    else:
        expected_html += "<li style='color: #64748b; font-style: italic;'>No actions expected</li>"
    expected_html += "</ul>"

    # Build actual actions list
    actual_html = "<ul class='action-list'>"
    if actual:
        for action in actual:
            params = actual_params.get(action, {})
            if action in matched:
                status, status_text = "matched", "MATCHED"
            elif action in extra:
                status, status_text = "extra", "EXTRA"
            else:
                status, status_text = "", ""
            actual_html += f"""
            <li>
                <span class="action-url">{_html_escape(_format_uri(action))}</span>
                <span class="action-params">{_html_escape(_format_params(params))}</span>
                <span class="action-status {status}">{status_text}</span>
            </li>
            """
    else:
        actual_html += "<li style='color: #64748b; font-style: italic;'>No actions generated</li>"
    actual_html += "</ul>"

    return f"""
    <div class="detail-section">
        <div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="section-icon">&#9660;</span>
            <span class="section-title">Action Comparison</span>
        </div>
        <div class="section-content">
            <div class="comparison-grid">
                <div class="comparison-box expected">
                    <h4>Expected Actions (Ground Truth)</h4>
                    {expected_html}
                </div>
                <div class="comparison-box actual">
                    <h4>Generated Actions (From Plan)</h4>
                    {actual_html}
                </div>
            </div>
        </div>
    </div>
    """


def generate_execution_section(result: dict) -> str:
    """Generate the execution results section."""
    trace = result.get("trace", {}) or {}
    execution = trace.get("execution", {}) or {}

    plan_gen = result.get("plan_generated", False)
    exec_success = result.get("execution_success", False)
    duration = result.get("duration", 0)

    ticks = execution.get("ticks", 0) if execution else 0
    tree_name = execution.get("tree_name", "N/A") if execution else "N/A"
    final_status = execution.get("final_status", "N/A") if execution else "N/A"

    # Property verification
    prop_results = result.get("property_results", [])
    props_html = ""
    if prop_results:
        props_html = """
        <h2 style="margin-top: 15px;">Property Verification</h2>
        <table class="prop-table">
            <tr><th>Property</th><th>Expected</th><th>Actual</th><th></th></tr>
        """
        for prop in prop_results:
            match = prop.get("matched", False)
            match_class = "match" if match else "mismatch"
            match_icon = '<span class="match-icon yes">&#10003;</span>' if match else '<span class="match-icon no">&#10007;</span>'
            props_html += f"""
            <tr class="{match_class}">
                <td>{_html_escape(_format_uri(prop.get('property', '')))}</td>
                <td><code>{_html_escape(str(prop.get('expected', '')))}</code></td>
                <td><code>{_html_escape(str(prop.get('actual', '')))}</code></td>
                <td>{match_icon}</td>
            </tr>
            """
        props_html += "</table>"

    return f"""
    <div class="detail-section">
        <div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="section-icon">&#9660;</span>
            <span class="section-title">Execution Results</span>
        </div>
        <div class="section-content">
            <div class="exec-result">
                <div class="exec-item">
                    <span class="exec-label">Plan Generated:</span>
                    <span class="exec-value {'success' if plan_gen else 'failure'}">{'Yes' if plan_gen else 'No'}</span>
                </div>
                <div class="exec-item">
                    <span class="exec-label">Execution:</span>
                    <span class="exec-value {'success' if exec_success else 'failure'}">{'Success' if exec_success else 'Failed'}</span>
                </div>
                <div class="exec-item">
                    <span class="exec-label">Ticks:</span>
                    <span class="exec-value">{ticks}</span>
                </div>
                <div class="exec-item">
                    <span class="exec-label">Final Status:</span>
                    <span class="exec-value">{_html_escape(final_status)}</span>
                </div>
                <div class="exec-item">
                    <span class="exec-label">Duration:</span>
                    <span class="exec-value">{duration:.2f}s</span>
                </div>
            </div>
            {props_html}
        </div>
    </div>
    """


def generate_failure_analysis(result: dict, test_data: dict) -> str:
    """Generate failure/partial success analysis section."""
    success = result.get("success", "False")
    if success == "True":
        return ""

    is_error_input = result.get("is_error_input_only", False)
    failure_type = result.get("failure_type", "")
    is_quantifiable = success == "Quantifiable"
    reasons = []

    # For quantifiable results, show what was partially successful
    if is_quantifiable:
        header_text = "Why is this a partial success?"
        analysis_class = "partial-analysis"
    else:
        header_text = "Why did this test fail?"
        analysis_class = "failure-analysis"

    # Add failure type as first reason if available (only for actual failures)
    if failure_type and not is_quantifiable:
        type_labels = {
            "parse_error": "Parse Error - LLM output couldn't be parsed as JSON",
            "compilation_error": "Compilation Error - Invalid node type or missing fields",
            "execution_error": "Execution Error - Behavior tree execution failed",
            "action_mismatch": "Action Mismatch - Wrong actions in the generated plan",
            "property_mismatch": "Property Mismatch - Actions correct but final state wrong",
            "error_input_not_detected": "Error Input Not Detected - System executed an impossible goal",
            "other": "Other - Unknown failure type",
        }
        type_label = type_labels.get(failure_type, failure_type)
        reasons.append(f'<strong>Failure Type:</strong> <span style="color: #f59e0b;">{_html_escape(type_label)}</span>')

    if is_error_input:
        # For impossible requests, failure means we incorrectly tried to execute
        if result.get("execution_success"):
            reasons.append("System <strong>incorrectly executed</strong> an action for an impossible request")
        if result.get("actions_in_plan"):
            reasons.append(f"Generated actions when none were possible: {len(result.get('actions_in_plan', []))} actions")
    else:
        # Normal test failure/partial analysis
        if not result.get("plan_generated"):
            reasons.append("Failed to generate a plan")

        missing = result.get("missing_actions", [])
        if missing:
            reasons.append(f"<strong>Missing actions:</strong> {len(missing)} expected action(s) not in generated plan")
            for action in missing[:3]:
                reasons.append(f"&nbsp;&nbsp;- {_html_escape(_format_uri(action))}")

        extra = result.get("extra_actions", [])
        if extra:
            reasons.append(f"<strong>Extra actions:</strong> {len(extra)} unexpected action(s) in plan")

        # Check param mismatches
        expected_params = result.get("expected_params", {})
        actual_params = result.get("params_in_plan", {})
        for action in result.get("matched_actions", []):
            exp = expected_params.get(action, {})
            act = actual_params.get(action, {})
            if exp != act:
                reasons.append(f"<strong>Parameter mismatch</strong> for {_html_escape(_format_uri(action))}")
                reasons.append(f"&nbsp;&nbsp;Expected: {_html_escape(_format_params(exp))}")
                reasons.append(f"&nbsp;&nbsp;Actual: {_html_escape(_format_params(act))}")

        if not result.get("execution_success") and result.get("plan_generated"):
            reasons.append("Plan was generated but <strong>execution failed</strong>")

        prop_results = result.get("property_results", [])
        failed_props = [p for p in prop_results if not p.get("matched")]
        if failed_props:
            reasons.append(f"<strong>Property verification failed:</strong> {len(failed_props)} property mismatch(es)")

        if result.get("error"):
            reasons.append(f"<strong>Error:</strong> {_html_escape(result.get('error'))}")

    if not reasons:
        if is_quantifiable:
            reasons.append("Partial success - some actions or properties matched but not all")
        else:
            reasons.append("Unknown failure reason")

    reasons_html = "\n".join(f'<div class="failure-reason">{r}</div>' for r in reasons)

    # Use different styling for quantifiable vs failed
    section_title = "Partial Success Analysis" if is_quantifiable else "Failure Analysis"
    box_class = "partial-analysis" if is_quantifiable else "failure-analysis"

    return f"""
    <div class="detail-section">
        <div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="section-icon">&#9660;</span>
            <span class="section-title">{section_title}</span>
        </div>
        <div class="section-content">
            <div class="{box_class}">
                <h4>{header_text}</h4>
                {reasons_html}
            </div>
        </div>
    </div>
    """


def generate_test_card(result: dict, test_data: Optional[dict] = None, has_trace_html: bool = False) -> str:
    """Generate a comprehensive card for a single test result."""
    test_id = result.get("test_id", "unknown")
    success = result.get("success", "False")  # Can be "True", "False", or "Quantifiable"
    is_error_input = result.get("is_error_input_only", False)
    failure_type = result.get("failure_type", "")

    # Determine status class and text based on success value
    if success == "True":
        status_class = "success"
        status_text = "PASS"
    elif success == "Quantifiable":
        status_class = "quantifiable"
        status_text = "PARTIAL"
    else:
        status_class = "error"
        status_text = "FAIL"

    if is_error_input:
        badge = '<span class="badge badge-impossible">IMPOSSIBLE REQUEST</span>'
    else:
        badge = '<span class="badge badge-normal">NORMAL</span>'

    # Add failure type badge for failed tests
    failure_badge = ""
    if success == "False" and failure_type:
        failure_badge = f'<span class="failure-type-badge {_html_escape(failure_type)}">{_html_escape(failure_type.replace("_", " "))}</span>'

    # Add link to individual trace if available
    trace_link = ""
    if has_trace_html:
        trace_link = f'<a href="traces/{_html_escape(test_id)}.html" class="trace-link" target="_blank" onclick="event.stopPropagation()">View Full Trace ↗</a>'

    input_goal = test_data.get("input", "N/A") if test_data else "N/A"

    # Generate sections (detailed traces are in individual trace files)
    ground_truth = generate_ground_truth_section(test_data or {}, is_error_input)
    action_comparison = generate_action_comparison_section(result)
    execution_section = generate_execution_section(result)
    failure_analysis = generate_failure_analysis(result, test_data or {}) if success != "True" else ""

    return f"""
    <div class="test-card {status_class}" data-success="{_html_escape(str(success))}" data-error-input="{str(is_error_input).lower()}" data-failure-type="{_html_escape(failure_type)}">
        <div class="test-header" onclick="this.parentElement.classList.toggle('expanded')">
            <div class="test-status {status_class}">{status_text}</div>
            <div class="test-id">{_html_escape(test_id)}</div>
            {badge}
            {failure_badge}
            {trace_link}
            <div class="test-expand-icon">&#9660;</div>
        </div>
        <div class="test-goal">"{_html_escape(input_goal)}"</div>
        <div class="test-details">
            {failure_analysis}
            {ground_truth}
            {action_comparison}
            {execution_section}
        </div>
    </div>
    """


def generate_html_report(
    metrics: dict,
    results: list[dict],
    config: dict,
    test_data_map: dict,
    output_path: str,
):
    """Generate the complete HTML report."""
    output_dir = Path(output_path).parent
    traces_dir = output_dir / "traces"

    # Generate test cards
    test_cards = []
    for result in results:
        test_id = result.get("test_id", "")
        test_data = test_data_map.get(test_id, {})
        # Check if individual trace HTML exists
        has_trace_html = (traces_dir / f"{test_id}.html").exists()
        test_cards.append(generate_test_card(result, test_data, has_trace_html))

    tests_html = "\n".join(test_cards)

    # Counts
    total = len(results)
    passed = sum(1 for r in results if r.get("success") == "True")
    quantifiable = sum(1 for r in results if r.get("success") == "Quantifiable")
    failed = sum(1 for r in results if r.get("success") == "False")
    error_input_count = sum(1 for r in results if r.get("is_error_input_only"))

    # Config summary
    config_html = ""
    if config:
        model = config.get("model", {}).get("name", "N/A")
        discovery = config.get("discovery", {})
        planning = config.get("planning", {})
        aff_strategy = discovery.get("affordances", {}).get("strategy", "N/A")
        state_strategy = discovery.get("state", {}).get("strategy", "N/A")
        reasoning = planning.get("reasoning", {})
        reasoning_str = reasoning.get("strategy", "none") if reasoning.get("enabled") else "none"
        output_format = planning.get("output", {}).get("format", "N/A")

        config_html = f"""
        <div class="config-summary">
            <span><strong>Model:</strong> {_html_escape(model)}</span>
            <span><strong>Discovery:</strong> {_html_escape(aff_strategy)} / {_html_escape(state_strategy)}</span>
            <span><strong>Reasoning:</strong> {_html_escape(reasoning_str)}</span>
            <span><strong>Output:</strong> {_html_escape(output_format)}</span>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HomeBench Evaluation Report</title>
    {generate_css()}
</head>
<body>
    <div class="container">
        <h1>HomeBench Evaluation Report</h1>
        <p class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        {config_html}
        {generate_info_box()}
        {generate_metrics_card(metrics)}

        <div class="filters">
            <button class="filter-btn active" onclick="filterTests('all')">
                All<span class="filter-count">({total})</span>
            </button>
            <button class="filter-btn" onclick="filterTests('passed')">
                Passed<span class="filter-count">({passed})</span>
            </button>
            <button class="filter-btn" onclick="filterTests('quantifiable')">
                Quantifiable<span class="filter-count">({quantifiable})</span>
            </button>
            <button class="filter-btn" onclick="filterTests('failed')">
                Failed<span class="filter-count">({failed})</span>
            </button>
            <button class="filter-btn" onclick="filterTests('impossible')">
                Impossible<span class="filter-count">({error_input_count})</span>
            </button>
            <input type="text" class="search-input" placeholder="Search by test ID or goal..." oninput="searchTests(this.value)">
            <button class="filter-btn" onclick="expandAll()" style="margin-left: auto;">Expand All</button>
            <button class="filter-btn" onclick="collapseAll()">Collapse All</button>
        </div>

        <div class="tests-container">
            {tests_html}
        </div>
    </div>

    <script>
        function filterTests(filter) {{
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            document.querySelectorAll('.test-card').forEach(card => {{
                const success = card.dataset.success;
                const errorInput = card.dataset.errorInput === 'true';

                let show = true;
                if (filter === 'passed') show = success === 'True';
                else if (filter === 'quantifiable') show = success === 'Quantifiable';
                else if (filter === 'failed') show = success === 'False';
                else if (filter === 'impossible') show = errorInput;

                card.classList.toggle('hidden', !show);
            }});
        }}

        function searchTests(query) {{
            const lowerQuery = query.toLowerCase();
            document.querySelectorAll('.test-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                card.classList.toggle('hidden', !text.includes(lowerQuery));
            }});
            if (query) {{
                document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            }} else {{
                document.querySelector('.filter-btn').classList.add('active');
            }}
        }}

        function expandAll() {{
            document.querySelectorAll('.test-card').forEach(card => {{
                if (!card.classList.contains('hidden')) {{
                    card.classList.add('expanded');
                }}
            }});
        }}

        function collapseAll() {{
            document.querySelectorAll('.test-card').forEach(card => {{
                card.classList.remove('expanded');
            }});
        }}

        // Auto-expand failed tests
        document.querySelectorAll('.test-card.error').forEach((card, idx) => {{
            if (idx < 5) card.classList.add('expanded');
        }});
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report generated: {output_path}")


def load_evaluation_data(eval_dir: Path) -> tuple[dict, list[dict], dict]:
    """Load metrics and results from evaluation directory."""
    metrics = {}
    results = []
    config = {}

    metrics_files = sorted(eval_dir.glob("metrics_*.json"), reverse=True)
    results_files = sorted(eval_dir.glob("results_*.json"), reverse=True)

    if metrics_files:
        with open(metrics_files[0]) as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            config = data.get("config", {})

    if results_files:
        with open(results_files[0]) as f:
            results = json.load(f)

    return metrics, results, config


def load_test_data(data_path: str) -> dict:
    """Load original test data to get input goals and ground truth."""
    test_map = {}
    try:
        with open(resolve_repo_path(data_path)) as f:
            data = json.load(f)
            for item in data:
                test_map[item["id"]] = item
    except Exception as e:
        print(f"Warning: Could not load test data: {e}")
    return test_map


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive HTML report for HomeBench evaluation",
    )
    parser.add_argument("eval_dirs", nargs="+", help="Evaluation result directories")
    parser.add_argument("--output", "-o", help="Output HTML file")
    parser.add_argument(
        "--test-data",
        default="data/homebench/converted/test_data.json",
        help="Path to original test data JSON"
    )

    args = parser.parse_args()
    test_data_map = load_test_data(args.test_data)

    for eval_dir in args.eval_dirs:
        eval_path = Path(eval_dir)
        if not eval_path.exists() or not eval_path.is_dir():
            print(f"Error: {eval_dir} not found or not a directory")
            continue

        metrics, results, config = load_evaluation_data(eval_path)

        if not results:
            print(f"No results found in {eval_dir}")
            continue

        output_file = args.output if args.output else str(eval_path / "eval_report.html")
        generate_html_report(metrics, results, config, test_data_map, output_file)


if __name__ == "__main__":
    main()
