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

    # Support both nested ExperimentConfig shape and flat runner config shape.
    # Nested: config["model"] == {"name": "gpt-4o", ...}
    # Flat:   config["model"] == "gpt-4o"
    model_raw = config.get("model", {})
    if isinstance(model_raw, dict):
        model_name = model_raw.get("name", "N/A")
    else:
        model_name = str(model_raw) if model_raw else "N/A"

    discovery = config.get("discovery", {})
    if isinstance(discovery, dict):
        affordances = discovery.get("affordances", {})
        aff_strategy = affordances.get("strategy", "N/A") if isinstance(affordances, dict) else str(affordances)
        state = discovery.get("state", {})
        state_strategy = state.get("strategy", "N/A") if isinstance(state, dict) else str(state)
    else:
        aff_strategy = "N/A"
        state_strategy = "N/A"

    planning = config.get("planning", {})
    if isinstance(planning, dict):
        reasoning = planning.get("reasoning", {})
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        output = planning.get("output", {})
        output_format = output.get("format", "N/A") if isinstance(output, dict) else str(output)
    else:
        reasoning = {}
        output_format = "N/A"

    rows = [
        ("Status", f'<span class="{status_class}">{"SUCCESS" if success else "FAILED"}</span>'),
        ("Goal", _html_escape(data.get("goal", "N/A"))),
        ("Config", _html_escape(data.get("config_name", config.get("config_name", config.get("experiment", {}).get("name", "N/A")) if isinstance(config, dict) else "N/A"))),
        ("Model", _html_escape(model_name)),
        ("Execution Backend", _html_escape(str(data.get("execution_backend", "N/A")))),
        ("Duration", format_duration(data.get("duration_seconds", 0))),
        ("Affordance Strategy", _html_escape(aff_strategy)),
        ("State Strategy", _html_escape(state_strategy)),
    ]

    if reasoning.get("enabled"):
        rows.append(("Reasoning", _html_escape(reasoning.get("strategy", "N/A"))))
    else:
        rows.append(("Reasoning", "disabled"))

    rows.append(("Output Format", _html_escape(output_format)))

    # Experience-specific timing
    matched_time = data.get("matched_plan_time_seconds", 0)
    unmatched_time = data.get("unmatched_plan_time_seconds", 0)
    if matched_time or unmatched_time:
        rows.append(("Matched Planning Time", format_duration(matched_time)))
        rows.append(("Unmatched Planning Time", format_duration(unmatched_time)))

    table_rows = "\n".join([f'<tr><td class="label">{k}</td><td>{v}</td></tr>' for k, v in rows])
    return _generate_html_card("Experiment Overview", f"<table>{table_rows}</table>", "#4a9eff")


def _build_intent_route_map(data: dict) -> dict:
    """
    Build a mapping from text_intent → route label for neuro-symbolic traces.

    Route labels: "SET", "MODIFY", "IMPOSSIBLE"

    Returns an empty dict for non-neuro-symbolic traces (no resolution_results).
    """
    resolution_results = data.get("resolution_results", [])
    impossible_details = data.get("impossible_details") or []

    # Only build a route map for NS traces (must have resolution_results or impossible_details)
    if not resolution_results and not impossible_details:
        return {}

    route_map: dict[str, str] = {}

    # Experience-impossible intents (skipped before SPARQL)
    for imp in impossible_details:
        if not isinstance(imp, dict):
            continue
        text = imp.get("intent", {}).get("text_intent", "")
        if text:
            route_map[text] = "IMPOSSIBLE"

    for res in resolution_results:
        if not isinstance(res, dict):
            continue
        intent_dict = res.get("intent", {})
        text = intent_dict.get("text_intent", "")
        success = res.get("success", False)
        if not success:
            route_map[text] = "IMPOSSIBLE"
        else:
            verb = intent_dict.get("action", {}).get("verb", "set")
            route_map[text] = "SET" if verb == "set" else "MODIFY"

    return route_map


def _generate_intent_extraction_html(data: dict) -> str:
    """Generate intent extraction section showing parsed intents."""
    intents = data.get("intents", [])
    if not intents:
        return ""

    # Build routing map (only populated for neuro-symbolic traces)
    route_map = _build_intent_route_map(data)

    _ROUTE_BADGE = {
        "SET":       ("<span class='intent-route-badge route-set'>SET</span>", "#22c55e"),
        "MODIFY":    ("<span class='intent-route-badge route-modify'>MODIFY</span>", "#a855f7"),
        "IMPOSSIBLE":("<span class='intent-route-badge route-impossible'>IMPOSSIBLE</span>", "#ef4444"),
    }

    html = "<div class='intent-list'>"
    for i, intent in enumerate(intents):
        if not isinstance(intent, dict):
            continue

        text = _html_escape(intent.get("text_intent", "N/A"))
        action = intent.get("action", {})
        target = intent.get("target", {})
        aff_type = _html_escape(action.get("affordance_type", "N/A"))
        verb = _html_escape(action.get("verb", "N/A"))
        parameter = action.get("parameter")
        value = action.get("value")
        art_type = _html_escape(target.get("artifact_type", "N/A"))
        ws_type = _html_escape(target.get("workspace_type", "N/A"))
        idx = intent.get("original_index", i)

        # Routing badge (only for neuro-symbolic traces)
        raw_text = intent.get("text_intent", "")
        route = route_map.get(raw_text)
        if route:
            badge_html, border_color = _ROUTE_BADGE[route]
        else:
            badge_html, border_color = "", "#8b5cf6"

        # Parameter/value row — only shown when at least one is present
        param_html = ""
        if parameter is not None or value is not None:
            param_display = _html_escape(str(parameter)) if parameter is not None else "<span class='dim'>none</span>"
            value_display = _html_escape(str(value)) if value is not None else "<span class='dim'>none</span>"
            param_html = f"""
            <div class='intent-param-row'>
                <span class='intent-param-label'>parameter:</span>
                <span class='intent-param-name'>{param_display}</span>
                <span class='intent-param-label'>value:</span>
                <span class='intent-param-value'>{value_display}</span>
            </div>
            """

        html += f"""
        <div class='intent-card' style='border-left-color: {border_color};'>
            <div class='intent-header'>
                <span class='intent-index'>#{idx}</span>
                <span class='intent-text'>"{text}"</span>
                {badge_html}
            </div>
            <div class='intent-details'>
                <span class='intent-tag action-tag'>{aff_type}</span>
                <span class='intent-tag verb-tag'>{verb}</span>
                <span class='intent-tag artifact-tag'>{art_type}</span>
                <span class='intent-tag workspace-tag'>{ws_type}</span>
            </div>
            {param_html}
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


def _generate_ns_routing_html(data: dict) -> str:
    """
    Generate a neuro-symbolic routing summary card.

    Only rendered when the trace contains neuro-symbolic routing data
    (i.e. has resolution_results and at least one of set_count / modify_count /
    impossible_count).

    Shows the SPARQL query used for each intent (collapsible toggle).
    """
    resolution_results = data.get("resolution_results", [])
    impossible_details = data.get("impossible_details") or []

    # Render if there's anything to show (SPARQL results or experience-impossible intents)
    if not resolution_results and not impossible_details:
        return ""

    set_count = data.get("set_count", 0)
    modify_count = data.get("modify_count", 0)
    impossible_count = data.get("impossible_count", 0)
    modify_plan_time = data.get("modify_plan_time_seconds", 0.0)

    # Summary bar
    html = f"""
    <div class='ns-routing-summary'>
        <span class='ns-route-stat route-set-stat'>{set_count} SET</span>
        <span class='ns-route-stat route-modify-stat'>{modify_count} MODIFY</span>
        <span class='ns-route-stat route-impossible-stat'>{impossible_count} IMPOSSIBLE</span>
    </div>
    """

    # Experience-impossible intents (skipped before SPARQL) — shown first
    for imp in impossible_details:
        if not isinstance(imp, dict):
            continue
        intent_dict = imp.get("intent", {})
        text = _html_escape(intent_dict.get("text_intent", "N/A"))
        reason = imp.get("reason", "sparql_no_result")
        matched_exp = imp.get("matched_experience")

        if reason == "experience_match" and matched_exp:
            src_test = _html_escape(matched_exp.get("source_test_id", "?"))
            src_home = _html_escape(matched_exp.get("home_id", "?"))
            exp_text = _html_escape(matched_exp.get("text_intent", "?"))
            created = _html_escape((matched_exp.get("created_at") or "")[:10])
            match_detail = f"""
            <div class='ns-impossible-match'>
                <span class='dim'>matched experience:</span>
                <span class='ns-imp-exp-text'>"{exp_text}"</span>
                <span class='exp-badge home-badge'>home: {src_home}</span>
                <span class='exp-badge src-badge'>from: {src_test}</span>
                {f"<span class='exp-badge dim'>{created}</span>" if created else ""}
            </div>
            """
            reason_label = "experience match"
        else:
            match_detail = ""
            reason_label = "SPARQL no result"

        html += f"""
        <div class='ns-resolution-card' style='border-left-color: #ef4444;'>
            <div class='ns-resolution-header'>
                <span class='ns-intent-text'>{text}</span>
                <span class='intent-route-badge route-impossible'>IMPOSSIBLE</span>
                <span class='ns-resolution-sparql-status dim'>via {_html_escape(reason_label)}</span>
            </div>
            {match_detail}
        </div>
        """

    # Per-SPARQL-resolution detail: one block per active intent
    for res_idx, res in enumerate(resolution_results):
        if not isinstance(res, dict):
            continue
        intent_dict = res.get("intent", {})
        action = intent_dict.get("action", {})
        target = intent_dict.get("target", {})
        text = _html_escape(intent_dict.get("text_intent", "N/A"))
        success = res.get("success", False)
        verb = action.get("verb", "set")
        target_uri = res.get("target_uri") or ""
        parameter_name = res.get("parameter_name") or ""
        parameter_schema = res.get("parameter_schema_type") or ""
        error = res.get("error") or ""
        query = res.get("query") or ""
        bindings_count = res.get("bindings_count", 0)

        # Route badge + SPARQL status
        if not success:
            route_badge = "<span class='intent-route-badge route-impossible'>IMPOSSIBLE</span>"
            sparql_status = "<span class='ns-sparql-fail'>✗ no result</span>"
            if error:
                sparql_status += f" <span class='dim'>({_html_escape(error[:80])})</span>"
            border_color = "#ef4444"
        else:
            if verb == "set":
                route_badge = "<span class='intent-route-badge route-set'>SET</span>"
                border_color = "#22c55e"
            else:
                route_badge = "<span class='intent-route-badge route-modify'>MODIFY</span>"
                border_color = "#a855f7"
            sparql_status = f"<span class='ns-sparql-ok'>✓ {bindings_count} binding(s)</span>"

        # Target display: resolved URI when SPARQL succeeded, queried intent types when it failed
        if target_uri:
            short_uri = target_uri.split("/")[-1] if "/" in target_uri else target_uri
            uri_html = f"<span class='dim' title='{_html_escape(target_uri)}'>{_html_escape(short_uri)}</span>"
        else:
            # Show what the SPARQL was looking for (from the intent)
            artifact_type = _html_escape(target.get("artifact_type") or action.get("affordance_type") or "?")
            workspace_type = _html_escape(target.get("workspace_type") or "?")
            uri_html = (
                f"<span class='dim ns-queried-types'>"
                f"looking for: <span class='intent-tag artifact-tag'>{artifact_type}</span> "
                f"in <span class='intent-tag workspace-tag'>{workspace_type}</span>"
                f"</span>"
            )

        # Parameter display: resolved name+schema when SPARQL succeeded, queried parameter when it failed
        if parameter_name:
            schema_local = parameter_schema.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if parameter_schema else ""
            param_html = f"<span class='intent-param-name'>{_html_escape(parameter_name)}</span>"
            if schema_local:
                param_html += f" <span class='dim'>({_html_escape(schema_local)})</span>"
        elif action.get("parameter"):
            # Show the intent-level parameter name (not yet resolved by SPARQL)
            queried_param = _html_escape(action["parameter"])
            queried_val = _html_escape(str(action.get("value") or ""))
            param_html = (
                f"<span class='dim'>looking for: </span>"
                f"<span class='intent-param-name'>{queried_param}</span>"
            )
            if queried_val:
                param_html += f" = <span class='intent-param-value'>{queried_val}</span>"
        else:
            param_html = "<span class='dim'>—</span>"

        # SPARQL query + bindings (collapsible)
        toggle_id = f"sparql-query-{res_idx}"
        bindings = res.get("bindings", [])
        if query:
            # Render bindings table (all rows — SPARQL results are compact)
            bindings_html = ""
            if bindings:
                # Collect all variable names across all bindings
                all_vars: list[str] = []
                for b in bindings:
                    for v in b:
                        if v not in all_vars:
                            all_vars.append(v)
                header_cells = "".join(
                    f"<th>{_html_escape(v)}</th>" for v in all_vars
                )
                rows_html = ""
                for b in bindings:
                    cells = ""
                    for v in all_vars:
                        raw_val = b.get(v, {})
                        val = raw_val.get("value", "") if isinstance(raw_val, dict) else str(raw_val)
                        # Shorten long URIs: keep only the last path segment
                        display = val.split("/")[-1] if "/" in val and len(val) > 50 else val
                        cells += f"<td title='{_html_escape(val)}'>{_html_escape(display)}</td>"
                    rows_html += f"<tr>{cells}</tr>"
                bindings_html = f"""
                <div class='ns-bindings-label'>Results ({len(bindings)} row{'s' if len(bindings) != 1 else ''})</div>
                <table class='ns-bindings-table'>
                    <tr>{header_cells}</tr>
                    {rows_html}
                </table>
                """
            elif success:
                bindings_html = "<span class='dim'>No bindings returned</span>"

            query_block = f"""
            <div class='ns-sparql-toggle' onclick="
                var el=document.getElementById('{toggle_id}');
                el.style.display = el.style.display==='none' ? 'block' : 'none';
                this.textContent = el.style.display==='none' ? '▶ Show SPARQL query' : '▼ Hide SPARQL query';
            ">▶ Show SPARQL query</div>
            <div id='{toggle_id}' style='display:none;'>
                <pre class='code sparql ns-sparql-block'>{_html_escape(query)}</pre>
                {bindings_html}
            </div>
            """
        else:
            query_block = "<span class='dim'>No query recorded</span>"

        html += f"""
        <div class='ns-resolution-card' style='border-left-color: {border_color};'>
            <div class='ns-resolution-header'>
                <span class='ns-intent-text'>{text}</span>
                {route_badge}
                <span class='ns-resolution-sparql-status'>{sparql_status}</span>
            </div>
            <div class='ns-resolution-meta'>
                <span class='dim'>{'resolved target' if success else 'target'}:</span> {uri_html}
                &nbsp;&nbsp;
                <span class='dim'>{'resolved parameter' if success else 'parameter'}:</span> {param_html}
            </div>
            {query_block}
        </div>
        """

    # Timing info for modify planning
    if modify_count > 0 and modify_plan_time > 0:
        html += f"<div class='ns-timing'>Modify LLM planning: {format_duration(modify_plan_time)}</div>"

    return _generate_html_card(
        f"Neuro-Symbolic Routing ({set_count} set · {modify_count} modify · {impossible_count} impossible)",
        html,
        "#22c55e",  # green
    )


def _generate_experience_plan_html(data: dict) -> str:
    """
    Generate plan section HTML with neuro-symbolic fallbacks.

    Standard traces use data["planning"]["plan"]. Neuro-symbolic traces,
    especially SET-only routes, may only populate JSON-IR fields such as
    combined_plan_ir/set_actions_tree_ir without a standard planning payload.
    """
    planning = data.get("planning", {})
    plan = planning.get("plan", {}) if isinstance(planning, dict) else {}

    # Standard planning path
    if isinstance(plan, dict) and plan.get("content"):
        return _generate_plan_html(data)

    # Neuro-symbolic fallback path
    combined_ir = data.get("combined_plan_ir")
    set_ir = data.get("set_actions_tree_ir")
    modify_ir = data.get("modify_actions_tree_ir")
    modify_trace = data.get("modify_plan_trace") or {}
    modify_planning = modify_trace.get("planning") if isinstance(modify_trace, dict) else None
    modify_plan = modify_planning.get("plan") if isinstance(modify_planning, dict) else None

    format_type = ""
    content = None
    explanation = ""
    llm_calls = 0
    source = ""

    if combined_ir:
        format_type = "json_ir"
        content = combined_ir
        source = "combined_plan_ir"
    elif set_ir:
        format_type = "json_ir"
        content = set_ir
        source = "set_actions_tree_ir"
    elif modify_ir:
        format_type = "json_ir"
        content = modify_ir
        source = "modify_actions_tree_ir"
    elif isinstance(modify_plan, dict) and modify_plan.get("content") is not None:
        format_type = modify_plan.get("format", "unknown")
        content = modify_plan.get("content")
        explanation = modify_plan.get("explanation", "")
        llm_calls = modify_planning.get("llm_calls", 0) if isinstance(modify_planning, dict) else 0
        source = "modify_plan_trace.planning.plan"
    else:
        return _generate_plan_html(data)

    html_content = ""
    if source:
        html_content += f"<p class='stats'>Source: {_html_escape(source)}</p>"
    if explanation:
        html_content += f"<p class='explanation'>{_html_escape(explanation)}</p><hr>"

    if format_type == "json_ir":
        json_str = json.dumps(content, indent=2)
        html_content += f"<pre class='code json'>{_html_escape(json_str)}</pre>"
    elif format_type == "python_code":
        html_content += f"<pre class='code python'>{_html_escape(str(content))}</pre>"
    else:
        html_content += f"<pre>{_html_escape(str(content))}</pre>"

    html_content += f"<p class='stats'>LLM Calls: {llm_calls}</p>"
    return _generate_html_card(f"Generated Plan ({format_type or 'unknown'})", html_content, "#a855f7")


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
            exp_entry = mr.get("experience")
            if exp_entry and isinstance(exp_entry, dict):
                exp_text = _html_escape(exp_entry.get("text_intent", "N/A"))
                exp_aff = _html_escape(exp_entry.get("affordance_type", "N/A"))
                exp_verb = _html_escape(exp_entry.get("verb", "N/A"))
                exp_art = _html_escape(exp_entry.get("artifact_type", "N/A"))
                exp_ws = _html_escape(exp_entry.get("workspace_type", "N/A"))
                exp_home = _html_escape(exp_entry.get("home_id", ""))
                exp_src = _html_escape(exp_entry.get("source_test_id", ""))
                exp_created = _html_escape(exp_entry.get("created_at", ""))
                bt_leaf = exp_entry.get("bt_leaf_json_ir")

                home_badge = (
                    f"<span class='exp-badge home-badge'>home: {exp_home}</span>"
                    if exp_home else ""
                )
                src_badge = (
                    f"<span class='exp-badge src-badge'>from: {exp_src}</span>"
                    if exp_src else ""
                )
                created_badge = (
                    f"<span class='exp-badge dim'>{exp_created[:10]}</span>"
                    if exp_created else ""
                )

                bt_leaf_html = ""
                if bt_leaf and isinstance(bt_leaf, dict) and bt_leaf:
                    bt_json = json.dumps(bt_leaf, indent=2)
                    bt_leaf_html = f"""
                    <div class='exp-bt-leaf'>
                        <div class='exp-detail-label'>BT Leaf (JSON-IR)</div>
                        <pre class='code json exp-leaf-code'>{_html_escape(bt_json)}</pre>
                    </div>
                    """

                exp_info = f"""
                <div class='match-experience-detail'>
                    <div class='exp-detail-label'>Matched experience <span class='dim'>({short_id}...)</span></div>
                    <div class='exp-intent-text'>"{exp_text}"</div>
                    <div class='exp-tags'>
                        <span class='intent-tag action-tag'>{exp_aff}</span>
                        <span class='intent-tag verb-tag'>{exp_verb}</span>
                        <span class='intent-tag artifact-tag'>{exp_art}</span>
                        <span class='intent-tag workspace-tag'>{exp_ws}</span>
                    </div>
                    <div class='exp-meta'>{home_badge}{src_badge}{created_badge}</div>
                    {bt_leaf_html}
                </div>
                """
            else:
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
.intent-param-row { display: flex; align-items: center; gap: 6px; margin-top: 4px; font-size: 12px; }
.intent-param-label { color: #888; }
.intent-param-name { background: rgba(251,146,60,0.2); color: #fb923c; border-radius: 3px; padding: 1px 5px; font-family: monospace; }
.intent-param-value { background: rgba(52,211,153,0.2); color: #34d399; border-radius: 3px; padding: 1px 5px; font-family: monospace; }

/* Routing badges on intent cards */
.intent-route-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.05em;
    margin-left: 8px;
    vertical-align: middle;
}
.route-set       { background: rgba(34,197,94,0.25);  color: #22c55e; }
.route-modify    { background: rgba(168,85,247,0.25); color: #a855f7; }
.route-impossible{ background: rgba(239,68,68,0.25);  color: #ef4444; }

/* Neuro-symbolic routing summary card */
.ns-routing-summary {
    display: flex;
    gap: 12px;
    margin: 10px 0 16px;
    padding: 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
}
.ns-route-stat {
    padding: 4px 14px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
}
.route-set-stat       { background: rgba(34,197,94,0.2);  color: #22c55e; }
.route-modify-stat    { background: rgba(168,85,247,0.2); color: #a855f7; }
.route-impossible-stat{ background: rgba(239,68,68,0.2);  color: #ef4444; }

.ns-resolution-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 6px;
}
.ns-resolution-table th {
    color: #888;
    font-weight: bold;
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #333;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.ns-resolution-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #222;
    vertical-align: middle;
}
.ns-intent-cell { color: #e0e0e0; max-width: 340px; }
.ns-sparql-ok   { color: #22c55e; font-weight: bold; }
.ns-sparql-fail { color: #ef4444; font-weight: bold; }
.ns-timing { color: #888; font-size: 12px; margin-top: 10px; }

/* Per-intent resolution cards (replacing flat table rows) */
.ns-resolution-card {
    background: rgba(0,0,0,0.2);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0;
    border-left: 3px solid #22c55e;
}
.ns-resolution-header {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}
.ns-intent-text { color: #e0e0e0; flex: 1; min-width: 0; }
.ns-resolution-sparql-status { font-size: 12px; }
.ns-resolution-meta {
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
}
.ns-sparql-toggle {
    font-size: 12px;
    color: #06b6d4;
    cursor: pointer;
    user-select: none;
    margin-top: 4px;
    display: inline-block;
}
.ns-sparql-toggle:hover { opacity: 0.8; }
.ns-sparql-block {
    margin-top: 6px;
    font-size: 11px;
    max-height: 300px;
    overflow-y: auto;
}
.ns-queried-types { font-size: 12px; }
.ns-queried-types .intent-tag { font-size: 11px; padding: 1px 5px; }
.ns-impossible-match {
    margin-top: 6px;
    font-size: 12px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}
.ns-imp-exp-text {
    color: #fca5a5;
    font-style: italic;
}
.ns-bindings-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #888;
    font-weight: bold;
    margin: 8px 0 4px;
}
.ns-bindings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin-top: 2px;
}
.ns-bindings-table th {
    color: #06b6d4;
    font-weight: bold;
    text-align: left;
    padding: 4px 8px;
    border-bottom: 1px solid #333;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.ns-bindings-table td {
    padding: 4px 8px;
    border-bottom: 1px solid #1a1a1a;
    color: #eab308;
    font-family: monospace;
    word-break: break-all;
}

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

/* Matched experience detail (inside match card) */
.match-experience-detail {
    margin-top: 10px;
    padding: 10px;
    background: rgba(0,0,0,0.25);
    border-radius: 4px;
    border-left: 2px solid #06b6d4;
}
.exp-detail-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #06b6d4;
    margin-bottom: 6px;
    font-weight: bold;
}
.exp-intent-text {
    color: #e0e0e0;
    margin-bottom: 6px;
    font-style: italic;
}
.exp-tags { margin-bottom: 6px; }
.exp-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
}
.exp-badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 11px;
}
.home-badge { background: rgba(6,182,212,0.15); color: #06b6d4; }
.src-badge  { background: rgba(168,85,247,0.15); color: #a855f7; }
.exp-bt-leaf { margin-top: 8px; }
.exp-leaf-code {
    font-size: 11px;
    max-height: 160px;
    overflow-y: auto;
    margin-top: 4px;
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
        # Step 1b: Neuro-symbolic routing summary (only for NS traces)
        _generate_ns_routing_html(data),
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
        _generate_experience_plan_html(data),
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
