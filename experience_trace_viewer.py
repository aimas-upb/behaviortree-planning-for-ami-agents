#!/usr/bin/env python3
"""
Experience-Based Experiment Trace Viewer.

Custom HTML renderer for experience pipeline traces that extends the
standard trace_viewer with sections for:
  - Intent extraction results
  - Experience matching (similarity scores, slot keys)
  - Matched experience adaptation traces
  - SPARQL query traces for unmatched discovery
  - Standard planning/execution sections

Usage:
    uv run python experience_trace_viewer.py experiments/results/experience_*/traces/*.json
    uv run python experience_trace_viewer.py --latest
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Reuse helpers from the standard trace_viewer
from trace_viewer import (
    _html_escape,
    _generate_html_card,
    format_duration,
    _generate_discovery_html,
    _generate_exploration_trace_html,
    _generate_state_trace_html,
    _generate_planning_context_html,
    _generate_reasoning_html,
    _generate_plan_html,
    _generate_execution_html,
    _generate_errors_html,
    console,
)


# =========================================================================
# Experience-specific HTML section generators
# =========================================================================


def _generate_experience_overview_html(data: dict) -> str:
    """Generate overview section with experience-specific info."""
    config = data.get("config", {})
    success = data.get("success", False)
    status_class = "success" if success else "error"

    rows = [
        ("Status", f'<span class="{status_class}">{"SUCCESS" if success else "FAILED"}</span>'),
        ("Goal", _html_escape(data.get("goal", "N/A"))),
        ("Config", _html_escape(data.get("config_name", config.get("experiment", {}).get("name", "N/A")))),
        ("Model", _html_escape(config.get("model", {}).get("name", "N/A"))),
        ("Duration", format_duration(data.get("duration_seconds", 0))),
        ("Affordance Strategy", _html_escape(config.get("discovery", {}).get("affordances", {}).get("strategy", "N/A"))),
        ("State Strategy", _html_escape(config.get("discovery", {}).get("state", {}).get("strategy", "N/A"))),
    ]

    reasoning = config.get("planning", {}).get("reasoning", {})
    if reasoning.get("enabled"):
        rows.append(("Reasoning", _html_escape(reasoning.get("strategy", "N/A"))))
    else:
        rows.append(("Reasoning", "disabled"))

    rows.append(("Output Format", _html_escape(config.get("planning", {}).get("output", {}).get("format", "N/A"))))

    # Experience-specific timing
    matched_time = data.get("matched_plan_time_seconds", 0)
    unmatched_time = data.get("unmatched_plan_time_seconds", 0)
    if matched_time or unmatched_time:
        rows.append(("Matched Planning Time", format_duration(matched_time)))
        rows.append(("Unmatched Planning Time", format_duration(unmatched_time)))

    table_rows = "\n".join([f'<tr><td class="label">{k}</td><td>{v}</td></tr>' for k, v in rows])
    return _generate_html_card("Experiment Overview", f"<table>{table_rows}</table>", "#4a9eff")


def _generate_intent_extraction_html(data: dict) -> str:
    """Generate intent extraction section showing parsed intents."""
    intents = data.get("intents", [])
    if not intents:
        return ""

    html = "<div class='intent-list'>"
    for i, intent in enumerate(intents):
        if not isinstance(intent, dict):
            continue

        text = _html_escape(intent.get("text_intent", "N/A"))
        action = intent.get("action", {})
        target = intent.get("target", {})
        aff_type = _html_escape(action.get("affordance_type", "N/A"))
        verb = _html_escape(action.get("verb", "N/A"))
        art_type = _html_escape(target.get("artifact_type", "N/A"))
        ws_type = _html_escape(target.get("workspace_type", "N/A"))
        idx = intent.get("original_index", i)

        html += f"""
        <div class='intent-card'>
            <div class='intent-header'>
                <span class='intent-index'>#{idx}</span>
                <span class='intent-text'>"{text}"</span>
            </div>
            <div class='intent-details'>
                <span class='intent-tag action-tag'>{aff_type}</span>
                <span class='intent-tag verb-tag'>{verb}</span>
                <span class='intent-tag artifact-tag'>{art_type}</span>
                <span class='intent-tag workspace-tag'>{ws_type}</span>
            </div>
            <div class='intent-slot'>
                Slot key: ({aff_type}, {art_type}, {ws_type})
            </div>
        </div>
        """

    html += "</div>"

    return _generate_html_card(
        f"Intent Extraction ({len(intents)} intents)",
        html,
        "#8b5cf6",  # purple
    )


def _generate_experience_matching_html(data: dict) -> str:
    """Generate experience matching section with similarity scores."""
    match_results = data.get("match_results", [])
    if not match_results:
        return ""

    matched_count = sum(1 for m in match_results if m.get("matched"))
    unmatched_count = sum(1 for m in match_results if not m.get("matched"))
    infeasible_count = sum(1 for m in match_results if m.get("is_infeasible"))

    # Summary bar
    html = f"""
    <div class='match-summary'>
        <span class='match-stat matched'>{matched_count} matched</span>
        <span class='match-stat unmatched'>{unmatched_count} unmatched</span>
        <span class='match-stat infeasible'>{infeasible_count} infeasible</span>
    </div>
    """

    # Individual match results
    html += "<div class='match-list'>"
    for mr in match_results:
        if not isinstance(mr, dict):
            continue

        intent = mr.get("intent", {})
        text = _html_escape(intent.get("text_intent", "N/A"))
        matched = mr.get("matched", False)
        sim_score = mr.get("similarity_score", 0.0)
        is_inf = mr.get("is_infeasible", False)
        exp_id = mr.get("experience_id")

        if matched:
            if is_inf:
                status_class = "infeasible"
                status_text = "INFEASIBLE"
                border = "#ef4444"
            else:
                status_class = "matched"
                status_text = "MATCHED"
                border = "#22c55e"
        else:
            status_class = "unmatched"
            status_text = "UNMATCHED"
            border = "#f59e0b"

        # Similarity bar
        sim_pct = sim_score * 100
        sim_bar = f"""
        <div class='similarity-bar-container'>
            <div class='similarity-bar' style='width: {sim_pct:.0f}%;
                 background: {"#22c55e" if sim_score >= 0.85 else "#f59e0b" if sim_score >= 0.5 else "#ef4444"}'></div>
            <span class='similarity-value'>{sim_score:.3f}</span>
        </div>
        """ if matched else ""

        exp_info = ""
        if exp_id:
            short_id = exp_id[:8] if len(exp_id) > 8 else exp_id
            exp_info = f"<div class='match-exp-id'>Experience: {_html_escape(short_id)}...</div>"

        html += f"""
        <div class='match-card' style='border-left: 3px solid {border};'>
            <div class='match-header'>
                <span class='match-status {status_class}'>{status_text}</span>
                <span class='match-text'>"{text}"</span>
            </div>
            {sim_bar}
            {exp_info}
        </div>
        """

    html += "</div>"

    return _generate_html_card(
        f"Experience Matching ({matched_count}/{len(match_results)} matched)",
        html,
        "#06b6d4",  # cyan
    )


def _generate_adaptation_traces_html(data: dict) -> str:
    """Generate adaptation trace section for matched intents."""
    traces = data.get("matched_plan_traces", [])
    if not traces:
        return ""

    html = ""
    for i, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue

        intent = trace.get("intent", {})
        text = _html_escape(intent.get("text_intent", "N/A"))
        exp_id = _html_escape(trace.get("experience_id", "N/A"))
        error = trace.get("error")

        # URL resolution info
        url_res = trace.get("url_resolution", {})
        url_info = ""
        if url_res and isinstance(url_res, dict):
            action_url = _html_escape(url_res.get("action_url", "N/A"))
            action_name = _html_escape(url_res.get("action_name", "N/A"))
            artifact_name = _html_escape(url_res.get("artifact_name", "N/A"))
            url_info = f"""
            <div class='adaptation-phase'>
                <div class='phase-title'>Phase 1: URL Resolution</div>
                <table>
                    <tr><td class='label'>Action</td><td><span class='action'>{action_name}</span></td></tr>
                    <tr><td class='label'>Artifact</td><td><span class='artifact'>{artifact_name}</span></td></tr>
                    <tr><td class='label'>URL</td><td class='dim'>{action_url}</td></tr>
                </table>
            </div>
            """

        # Parameter adaptation info
        param_adapt = trace.get("param_adaptation", {})
        param_info = ""
        if param_adapt and isinstance(param_adapt, dict):
            response = param_adapt.get("response", {})
            params = response.get("parameters", {}) if response else {}
            explanation = response.get("explanation", "") if response else ""
            param_error = param_adapt.get("error")

            if params:
                params_json = json.dumps(params, indent=2)
                param_info = f"""
                <div class='adaptation-phase'>
                    <div class='phase-title'>Phase 2: Parameter Adaptation</div>
                    <pre class='code json'>{_html_escape(params_json)}</pre>
                """
                if explanation:
                    param_info += f"<div class='explanation'>{_html_escape(explanation)}</div>"
                param_info += "</div>"
            elif param_error:
                param_info = f"""
                <div class='adaptation-phase'>
                    <div class='phase-title'>Phase 2: Parameter Adaptation</div>
                    <span class='error'>{_html_escape(param_error)}</span>
                </div>
                """

        # Error message
        error_html = ""
        if error:
            error_html = f"<div class='adaptation-error error'>{_html_escape(error)}</div>"

        html += f"""
        <div class='adaptation-card'>
            <div class='adaptation-header'>
                <span class='step-num'>Adaptation {i + 1}:</span>
                "{text}"
                <span class='dim'>(experience: {exp_id[:8]}...)</span>
            </div>
            {url_info}
            {param_info}
            {error_html}
        </div>
        """

    return _generate_html_card(
        f"Experience Adaptation ({len(traces)} adaptations)",
        html,
        "#f97316",  # orange
    )


def _generate_sparql_trace_html(data: dict) -> str:
    """
    Generate SPARQL query trace section from unmatched discovery.

    Extracts SPARQL query data from the exploration_trace in the
    discovery section (which comes from agentic_query strategy).
    """
    discovery = data.get("discovery", {})
    exploration_trace = discovery.get("exploration_trace", [])

    if not exploration_trace:
        return ""

    # Filter to SPARQL-related entries
    query_gen_entries = [
        e for e in exploration_trace
        if isinstance(e, dict) and e.get("phase") == "query_generation"
    ]
    query_exec_entries = [
        e for e in exploration_trace
        if isinstance(e, dict) and e.get("phase") == "query_execution"
    ]

    if not query_gen_entries and not query_exec_entries:
        return ""

    html = ""

    # Query generation phase
    for gen in query_gen_entries:
        queries = gen.get("queries", [])
        goal = _html_escape(gen.get("goal", "N/A"))

        html += f"<div class='sparql-section'>"
        html += f"<div class='sparql-goal'><strong>Goal:</strong> {goal}</div>"

        if queries:
            for q in queries:
                cmd = _html_escape(q.get("command", "N/A"))
                query = q.get("query", "")
                html += f"""
                <div class='sparql-query-block'>
                    <div class='sparql-command'>{cmd}</div>
                    <pre class='code sparql'>{_html_escape(query)}</pre>
                </div>
                """

        gen_error = gen.get("error")
        if gen_error:
            html += f"<div class='error'>Generation error: {_html_escape(gen_error)}</div>"

        html += "</div>"

    # Query execution results
    if query_exec_entries:
        html += "<hr>"
        html += "<div class='sparql-section'>"
        html += "<div class='phase-title'>Query Execution Results</div>"

        for exec_entry in query_exec_entries:
            cmd = _html_escape(exec_entry.get("command", "N/A"))
            query = exec_entry.get("query", "")
            result = exec_entry.get("result", {})

            bindings_count = result.get("bindings_count", 0)
            bindings = result.get("bindings", [])
            message = result.get("message", "")
            exec_error = result.get("error") or exec_entry.get("error")

            status = "success" if bindings_count > 0 else "error"

            html += f"""
            <div class='sparql-exec-block'>
                <div class='sparql-command'>{cmd}</div>
                <pre class='code sparql'>{_html_escape(query)}</pre>
                <div class='sparql-result'>
                    <span class='{status}'>Bindings: {bindings_count}</span>
            """

            if bindings:
                # Show bindings in a compact format
                for binding in bindings[:5]:  # limit to first 5
                    binding_parts = []
                    for var_name, var_val in binding.items():
                        if isinstance(var_val, dict):
                            val = var_val.get("value", str(var_val))
                        else:
                            val = str(var_val)
                        # Shorten URIs
                        if isinstance(val, str) and "/" in val and len(val) > 60:
                            val = "..." + val.split("/")[-1] if "/" in val else val
                        binding_parts.append(
                            f"<span class='binding-var'>{_html_escape(var_name)}</span>="
                            f"<span class='binding-val'>{_html_escape(str(val))}</span>"
                        )
                    html += f"<div class='sparql-binding'>{', '.join(binding_parts)}</div>"
                if len(bindings) > 5:
                    html += f"<div class='dim'>... and {len(bindings) - 5} more</div>"

            if message:
                html += f"<div class='dim'>{_html_escape(message)}</div>"

            if exec_error:
                html += f"<div class='error'>{_html_escape(exec_error)}</div>"

            html += "</div></div>"

        html += "</div>"

    total_queries = sum(len(g.get("queries", [])) for g in query_gen_entries)
    return _generate_html_card(
        f"SPARQL Query Discovery ({total_queries} queries, {len(query_exec_entries)} executions)",
        html,
        "#10b981",  # emerald
    )


def _generate_non_sparql_exploration_trace_html(data: dict) -> str:
    """
    Render the standard exploration trace ONLY for non-SPARQL entries.

    Filters out entries with ``phase`` in (query_generation, query_execution)
    since those are handled by the dedicated SPARQL section.
    """
    discovery = data.get("discovery", {})
    exploration_trace = discovery.get("exploration_trace", [])

    if not exploration_trace:
        return ""

    # Check if there are any non-SPARQL entries
    sparql_phases = {"query_generation", "query_execution"}
    non_sparql = [
        e for e in exploration_trace
        if isinstance(e, dict) and e.get("phase") not in sparql_phases
        and e.get("function") is not None
    ]

    if not non_sparql:
        return ""

    # Delegate to standard exploration trace renderer with filtered data
    filtered_data = dict(data)
    filtered_discovery = dict(discovery)
    filtered_discovery["exploration_trace"] = non_sparql
    filtered_data["discovery"] = filtered_discovery
    return _generate_exploration_trace_html(filtered_data)


def _generate_infeasible_commands_html(data: dict) -> str:
    """Generate section for infeasible commands detected during discovery."""
    discovery = data.get("discovery", {})
    affordances = discovery.get("affordances", {})
    infeasible = affordances.get("infeasible_commands", [])

    if not infeasible:
        return ""

    html = "<div class='infeasible-list'>"
    for entry in infeasible:
        cmd = _html_escape(entry.get("command", "unknown"))
        reason = _html_escape(entry.get("reason", "unknown"))
        html += f"""
        <div class='infeasible-item'>
            <span class='error'>{cmd}</span>
            <span class='dim'> — {reason}</span>
        </div>
        """
    html += "</div>"

    return _generate_html_card(
        f"Possibly Infeasible Commands ({len(infeasible)})",
        html,
        "#ef4444",
    )


# =========================================================================
# CSS styles specific to the experience viewer
# =========================================================================

EXPERIENCE_CSS = """
/* Intent extraction styles */
.intent-list { margin: 10px 0; }
.intent-card {
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    border-left: 3px solid #8b5cf6;
}
.intent-header { margin-bottom: 8px; }
.intent-index {
    color: #8b5cf6;
    font-weight: bold;
    margin-right: 8px;
}
.intent-text { color: #e0e0e0; }
.intent-details { margin: 6px 0; }
.intent-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
}
.action-tag { background: rgba(34,197,94,0.2); color: #22c55e; }
.verb-tag { background: rgba(168,85,247,0.2); color: #a855f7; }
.artifact-tag { background: rgba(234,179,8,0.2); color: #eab308; }
.workspace-tag { background: rgba(6,182,212,0.2); color: #06b6d4; }
.intent-slot { color: #666; font-size: 12px; margin-top: 4px; }

/* Experience matching styles */
.match-summary {
    display: flex;
    gap: 16px;
    margin: 10px 0;
    padding: 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
}
.match-stat {
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
}
.match-stat.matched { background: rgba(34,197,94,0.2); color: #22c55e; }
.match-stat.unmatched { background: rgba(245,158,11,0.2); color: #f59e0b; }
.match-stat.infeasible { background: rgba(239,68,68,0.2); color: #ef4444; }

.match-list { margin: 10px 0; }
.match-card {
    background: rgba(0,0,0,0.2);
    border-radius: 0 6px 6px 0;
    padding: 10px 12px;
    margin: 8px 0;
}
.match-header { display: flex; align-items: center; gap: 10px; }
.match-status {
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 12px;
}
.match-status.matched { background: rgba(34,197,94,0.2); color: #22c55e; }
.match-status.unmatched { background: rgba(245,158,11,0.2); color: #f59e0b; }
.match-status.infeasible { background: rgba(239,68,68,0.2); color: #ef4444; }
.match-text { color: #ccc; }
.match-exp-id { color: #666; font-size: 12px; margin-top: 4px; }

/* Similarity bar */
.similarity-bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
}
.similarity-bar {
    height: 6px;
    border-radius: 3px;
    transition: width 0.3s ease;
}
.similarity-value {
    color: #888;
    font-size: 12px;
    font-weight: bold;
    min-width: 40px;
}

/* Adaptation trace styles */
.adaptation-card {
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
    padding: 12px;
    margin: 10px 0;
    border-left: 3px solid #f97316;
}
.adaptation-header { margin-bottom: 10px; }
.adaptation-phase {
    margin: 8px 0;
    padding: 8px;
    background: rgba(0,0,0,0.15);
    border-radius: 4px;
}
.phase-title {
    font-weight: bold;
    color: #8b5cf6;
    margin-bottom: 6px;
    font-size: 13px;
}
.adaptation-error { margin-top: 8px; }

/* SPARQL trace styles */
.sparql-section { margin: 10px 0; }
.sparql-goal { margin-bottom: 10px; }
.sparql-query-block {
    margin: 10px 0;
    padding: 10px;
    background: rgba(0,0,0,0.15);
    border-radius: 4px;
}
.sparql-command {
    color: #06b6d4;
    font-weight: bold;
    margin-bottom: 6px;
}
.sparql-exec-block {
    margin: 10px 0;
    padding: 10px;
    background: rgba(0,0,0,0.15);
    border-radius: 4px;
}
.sparql-result { margin-top: 8px; }
.sparql-binding {
    margin: 4px 0 4px 16px;
    font-size: 13px;
}
.binding-var { color: #8b5cf6; }
.binding-val { color: #eab308; }

/* Infeasible commands */
.infeasible-list { margin: 10px 0; }
.infeasible-item {
    margin: 6px 0;
    padding: 6px 10px;
    background: rgba(239,68,68,0.1);
    border-radius: 4px;
}

/* Code block for SPARQL */
.code.sparql {
    background: #0d1117;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
"""


# =========================================================================
# Main export function
# =========================================================================


def export_experience_html(data: dict, output_path: str):
    """
    Export an experience pipeline trace to an HTML file.

    Renders experience-specific sections (intents, matching, adaptation,
    SPARQL queries) alongside the standard sections (discovery, planning,
    execution).
    """
    sections = [
        # Overview
        _generate_experience_overview_html(data),
        # Errors
        _generate_errors_html(data),
        # Step 1: Intent Extraction
        _generate_intent_extraction_html(data),
        # Step 2: Experience Matching
        _generate_experience_matching_html(data),
        # Step 3: Adaptation traces (for matched intents)
        _generate_adaptation_traces_html(data),
        # Step 4 (unmatched): SPARQL query discovery
        _generate_sparql_trace_html(data),
        # Standard discovery details (artifacts, state)
        _generate_discovery_html(data),
        # Standard exploration trace (filtered: only non-SPARQL entries)
        _generate_non_sparql_exploration_trace_html(data),
        # State trace
        _generate_state_trace_html(data),
        # Infeasible commands from discovery
        _generate_infeasible_commands_html(data),
        # Planning context
        _generate_planning_context_html(data),
        # Reasoning
        _generate_reasoning_html(data),
        # Generated plan
        _generate_plan_html(data),
        # Execution
        _generate_execution_html(data),
    ]

    # Filter out empty sections
    sections = [s for s in sections if s and s.strip()]

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experience Trace - {_html_escape(data.get('config_name', 'Unknown'))}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            background-color: #1a1a2e;
            color: #e0e0e0;
            padding: 20px;
            font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #8b5cf6;
            text-align: center;
            border-bottom: 2px solid #8b5cf6;
            padding-bottom: 10px;
        }}
        .card {{
            border: 2px solid;
            border-radius: 8px;
            margin: 20px 0;
            background: #16213e;
        }}
        .card-title {{
            font-weight: bold;
            font-size: 16px;
            padding: 12px 16px;
            border-bottom: 1px solid #333;
            background: rgba(0,0,0,0.2);
            border-radius: 6px 6px 0 0;
        }}
        .card-content {{
            padding: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #333;
        }}
        td.label {{
            color: #06b6d4;
            font-weight: bold;
            width: 200px;
        }}
        .success {{ color: #22c55e; font-weight: bold; }}
        .error {{ color: #ef4444; font-weight: bold; }}
        .highlight {{ color: #eab308; }}
        .dim {{ color: #888; }}
        .workspace {{ color: #06b6d4; }}
        .artifact {{ color: #eab308; }}
        .action {{ color: #22c55e; }}
        .property {{ color: #4a9eff; }}
        .constraint {{ color: #a855f7; }}
        .tree {{ margin: 10px 0; }}
        .tree-node {{ margin: 4px 0; }}
        .indent {{ margin-left: 20px; }}
        .indent2 {{ margin-left: 40px; }}
        .indent3 {{ margin-left: 60px; }}
        .trace-steps {{ margin: 10px 0; }}
        .trace-step {{
            margin: 12px 0;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
        }}
        .trace-result {{
            margin-left: 20px;
            color: #888;
            font-size: 13px;
        }}
        .step-num {{ color: #06b6d4; font-weight: bold; }}
        .code {{
            background: #0d1117;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
        }}
        .explanation {{
            font-style: italic;
            color: #888;
            margin-bottom: 16px;
        }}
        .stats {{ color: #666; font-size: 12px; margin-top: 10px; }}
        .reasoning-block {{
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        hr {{ border: none; border-top: 1px solid #333; margin: 16px 0; }}
        /* Collapsible card styles */
        .collapsible .card-title {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible .card-title:hover {{
            opacity: 0.8;
        }}
        .collapsible .collapse-icon {{
            display: inline-block;
            transition: transform 0.2s ease;
            margin-right: 8px;
        }}
        .collapsible.collapsed .collapse-icon {{
            transform: rotate(-90deg);
        }}
        .collapsible.collapsed .card-content {{
            display: none;
        }}
        .planning-context {{
            background: #0d1117;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        /* Experience-specific styles */
        {EXPERIENCE_CSS}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experience Pipeline Trace Viewer</h1>
        {"".join(sections)}
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(full_html)

    console.print(f"[green]Exported experience trace to {output_path}")


# =========================================================================
# CLI
# =========================================================================


def get_latest_trace(results_dir: str = "experiments/results") -> Optional[Path]:
    """Get the most recent trace JSON file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    json_files = list(results_path.glob("**/traces/*.json"))
    if not json_files:
        json_files = list(results_path.glob("**/*.json"))
        json_files = [f for f in json_files if "metrics" not in f.name and "results" not in f.name]

    if not json_files:
        return None

    return max(json_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="View experience pipeline traces in HTML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python experience_trace_viewer.py experiments/results/experience_*/traces/*.json
  uv run python experience_trace_viewer.py --latest
  uv run python experience_trace_viewer.py trace.json --html output.html
        """,
    )
    parser.add_argument("files", nargs="*", help="Trace JSON files")
    parser.add_argument("--latest", action="store_true", help="View the most recent trace")
    parser.add_argument("--html", metavar="FILE", help="Output HTML path (default: same dir as input)")
    parser.add_argument("--results-dir", default="experiments/results", help="Results directory")

    args = parser.parse_args()

    files = []

    if args.latest:
        latest = get_latest_trace(args.results_dir)
        if latest:
            files.append(latest)
        else:
            console.print(f"[red]No traces found in {args.results_dir}/")
            return 1

    files.extend(Path(f) for f in args.files)

    if not files:
        console.print("[yellow]Usage: experience_trace_viewer.py [--latest] [--html FILE] [trace_files...]")
        console.print("\nRun with --help for more options.")
        return 1

    for filepath in files:
        if not filepath.exists():
            console.print(f"[red]File not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        console.print(f"[dim]Loading: {filepath}")

        if args.html:
            output_path = args.html
        else:
            output_path = str(filepath.with_suffix(".html"))

        export_experience_html(data, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
