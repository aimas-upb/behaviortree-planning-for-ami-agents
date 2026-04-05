#!/usr/bin/env python3
"""
Enhanced Experiment Trace Viewer.

Interactive terminal viewer for experiment results with:
- Expandable sections for LLM calls, affordances, reasoning
- Syntax-highlighted JSON and code
- HTML export for sharing

Usage:
    uv run python -m viewers.trace_viewer experiments/results/ablation_*/cot_*.json
    uv run python -m viewers.trace_viewer --latest
    uv run python -m viewers.trace_viewer --html output.html experiments/results/*.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


def truncate_uri(uri: str, max_len: int = 60) -> str:
    """Truncate URI for display, keeping important parts."""
    if len(uri) <= max_len:
        return uri
    # Keep the last part (artifact/action name)
    parts = uri.split("/")
    if len(parts) > 3:
        return f".../{'/'.join(parts[-3:])}"
    return uri[:max_len] + "..."


def _extract_item_name(item: Any, fallback: str = "unknown") -> str:
    """Extract display name from dict/string item payloads."""
    if isinstance(item, dict):
        name = item.get("name")
        if name:
            return str(name)
        uri = item.get("uri")
        if isinstance(uri, str) and uri:
            return uri.split("/")[-1].replace("#artifact", "")
        return fallback
    if isinstance(item, str):
        return item.split("/")[-1].replace("#artifact", "")
    return fallback


def _extract_item_uri(item: Any) -> str:
    """Extract URI from dict/string item payloads."""
    if isinstance(item, dict):
        uri = item.get("uri")
        return str(uri) if uri else ""
    if isinstance(item, str):
        return item
    return ""


def _extract_workspace_artifacts(workspace_payload: Any) -> list[Any]:
    """Extract artifacts list from workspace payload.

    Supports both legacy shape:
        - workspace_name -> [artifact, ...]
    and newer shape:
        - workspace_name -> {"artifacts": [...], "semantic_type": ...}
    """
    if isinstance(workspace_payload, list):
        return workspace_payload
    if isinstance(workspace_payload, dict):
        artifacts = workspace_payload.get("artifacts", [])
        return artifacts if isinstance(artifacts, list) else []
    return []


def create_overview_panel(data: dict) -> Panel:
    """Create overview panel with experiment metadata."""
    config = data.get("config", {})

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value")

    # Result status
    success = data.get("success", False)
    status_style = "bold green" if success else "bold red"
    status_text = "SUCCESS" if success else "FAILED"
    table.add_row("Status", Text(status_text, style=status_style))

    # Goal
    table.add_row("Goal", data.get("goal", "N/A"))

    # Config name
    table.add_row(
        "Config",
        data.get(
            "config_name", config.get("experiment", {}).get("name", "N/A")
        ),
    )

    # Model
    table.add_row("Model", config.get("model", {}).get("name", "N/A"))

    # Duration
    duration = data.get("duration_seconds", 0)
    table.add_row("Duration", format_duration(duration))

    # Strategies
    discovery_cfg = config.get("discovery", {})
    planning_cfg = config.get("planning", {})

    table.add_row(
        "Affordance Strategy",
        discovery_cfg.get("affordances", {}).get("strategy", "N/A"),
    )
    table.add_row(
        "State Strategy", discovery_cfg.get("state", {}).get("strategy", "N/A")
    )

    reasoning = planning_cfg.get("reasoning", {})
    if reasoning.get("enabled"):
        table.add_row("Reasoning", f"{reasoning.get('strategy', 'N/A')}")
    else:
        table.add_row("Reasoning", "disabled")

    table.add_row(
        "Output Format", planning_cfg.get("output", {}).get("format", "N/A")
    )
    table.add_row("Prompt Strategy", planning_cfg.get("prompt_strategy", "N/A"))

    return Panel(table, title="[bold]Experiment Overview", border_style="blue")


def create_exploration_trace_panel(data: dict) -> Optional[Panel]:
    """Create panel showing agentic discovery exploration trace."""
    discovery = data.get("discovery", {})
    exploration_trace = discovery.get("exploration_trace", [])

    if not exploration_trace:
        return None

    tree = Tree("[bold]LLM Exploration Steps")

    for i, step in enumerate(exploration_trace):
        fn_name = step.get("function", "unknown")
        args = step.get("arguments", {})
        result = step.get("result", {})
        iteration = step.get("iteration", i + 1)

        # Format the step
        if fn_name == "explore_workspace":
            ws_uri = args.get("workspace_uri", "")
            ws_name = (
                ws_uri.split("/")[-1].replace("#workspace", "")
                if ws_uri
                else "unknown"
            )
            step_branch = tree.add(
                f"[cyan]Step {iteration}:[/cyan] explore_workspace([yellow]{ws_name}[/yellow])"
            )

            if result:
                if result.get("sub_workspaces"):
                    subs = [
                        _extract_item_name(s, "?")
                        for s in result["sub_workspaces"]
                    ]
                    step_branch.add(f"[dim]Sub-workspaces:[/dim] {subs}")
                if result.get("artifacts"):
                    arts = [
                        _extract_item_name(a, "?") for a in result["artifacts"]
                    ]
                    step_branch.add(f"[dim]Artifacts:[/dim] {arts}")

        elif fn_name == "inspect_artifact":
            art_uri = args.get("artifact_uri", "")
            art_name = (
                art_uri.split("/")[-1].replace("#artifact", "")
                if art_uri
                else "unknown"
            )
            step_branch = tree.add(
                f"[cyan]Step {iteration}:[/cyan] inspect_artifact([yellow]{art_name}[/yellow])"
            )

            if result:
                if result.get("actions"):
                    actions = [
                        _extract_item_name(a, "?") for a in result["actions"]
                    ]
                    step_branch.add(
                        f"[dim]Actions:[/dim] [green]{actions}[/green]"
                    )
                if result.get("properties"):
                    props = [
                        _extract_item_name(p, "?") for p in result["properties"]
                    ]
                    step_branch.add(
                        f"[dim]Properties:[/dim] [blue]{props}[/blue]"
                    )

        elif fn_name == "done_exploring":
            reason = args.get("reason", result.get("reason", "done"))
            tree.add(
                f'[cyan]Step {iteration}:[/cyan] done_exploring([green]"{reason}"[/green])'
            )

        else:
            tree.add(f"[cyan]Step {iteration}:[/cyan] {fn_name}({args})")

    return Panel(
        tree,
        title=f"[bold]Agentic Discovery Trace ({len(exploration_trace)} tool calls)",
        border_style="blue",
    )


def create_state_trace_panel(data: dict) -> Optional[Panel]:
    """Create panel showing agentic state gathering trace."""
    discovery = data.get("discovery", {})
    state_trace = discovery.get("state_trace", [])

    if not state_trace:
        return None

    tree = Tree("[bold]LLM State Gathering Steps")

    for i, step in enumerate(state_trace):
        fn_name = step.get("function", "unknown")
        args = step.get("arguments", {})
        result = step.get("result", {})
        iteration = step.get("iteration", i + 1)

        if fn_name == "read_property":
            prop_uri = args.get("property_uri", "")
            reason = args.get("reason", "")
            # Extract artifact and property name from URI
            # URI format: .../artifacts/{artifact_name}/properties/{property_name}
            parts = prop_uri.split("/") if prop_uri else []
            prop_name = parts[-1] if parts else "unknown"
            artifact_name = "unknown"
            if "artifacts" in parts:
                art_idx = parts.index("artifacts")
                if art_idx + 1 < len(parts):
                    artifact_name = parts[art_idx + 1]
            display_name = f"{artifact_name}.{prop_name}"

            if result.get("success"):
                value = result.get("value")
                step_branch = tree.add(
                    f"[cyan]Step {iteration}:[/cyan] read_property([yellow]{display_name}[/yellow]) "
                    f"-> [green]{value}[/green]"
                )
            else:
                error = result.get("error", "unknown error")
                step_branch = tree.add(
                    f"[cyan]Step {iteration}:[/cyan] read_property([yellow]{display_name}[/yellow]) "
                    f"-> [red]ERROR: {error}[/red]"
                )

            if reason:
                step_branch.add(f"[dim]Reason:[/dim] {reason}")

        elif fn_name == "done_gathering":
            summary = args.get("summary", result.get("summary", "done"))
            tree.add(
                f'[cyan]Step {iteration}:[/cyan] done_gathering([green]"{summary}"[/green])'
            )

        else:
            tree.add(f"[cyan]Step {iteration}:[/cyan] {fn_name}({args})")

    return Panel(
        tree,
        title=f"[bold]Agentic State Trace ({len(state_trace)} reads)",
        border_style="cyan",
    )


def create_discovery_panel(data: dict, expanded: bool = False) -> Panel:
    """Create discovery panel showing affordances and state."""
    discovery = data.get("discovery", {})
    affordances = discovery.get("affordances", {})
    state = discovery.get("state", {})

    # Stats summary
    stats = affordances.get("stats", {})
    summary = Text()
    summary.append(f"Workspaces: ", style="bold")
    summary.append(f"{stats.get('workspace_count', 0)}  ")
    summary.append(f"Artifacts: ", style="bold")
    summary.append(f"{stats.get('artifact_count', 0)}  ")
    summary.append(f"Actions: ", style="bold")
    summary.append(f"{stats.get('action_count', 0)}  ")
    summary.append(f"Properties: ", style="bold")
    summary.append(f"{stats.get('property_count', 0)}")

    # State summary
    state_count = state.get("property_count", 0)
    error_count = state.get("error_count", 0)
    summary.append(f"\nState Values: ", style="bold")
    summary.append(f"{state_count}")
    if error_count:
        summary.append(f" ({error_count} errors)", style="red")

    if not expanded:
        return Panel(
            summary,
            title="[bold]Discovery Phase",
            subtitle="[dim]Use --expand to see details",
            border_style="green",
        )

    # Expanded view with artifact tree
    tree = Tree("[bold]Discovered Devices")
    workspaces = affordances.get("workspaces", {})

    for ws_name, workspace_payload in workspaces.items():
        artifacts = _extract_workspace_artifacts(workspace_payload)
        ws_branch = tree.add(f"[cyan]{ws_name}/")
        for artifact in artifacts:
            art_name = _extract_item_name(artifact)
            artifact_data = artifact if isinstance(artifact, dict) else {}
            art_branch = ws_branch.add(f"[yellow]{art_name}")

            # Actions with constraints
            actions = artifact_data.get("actions", [])
            if actions:
                act_branch = art_branch.add("[dim]Actions:")
                for action in actions:
                    action_data = action if isinstance(action, dict) else {}
                    schema = action_data.get("schema", {})
                    props = schema.get("properties", {})
                    if props:
                        params = []
                        for pname, pschema in props.items():
                            ptype = pschema.get("type", "any")
                            constraint = ""
                            if "minimum" in pschema and "maximum" in pschema:
                                constraint = f" [magenta](range: {pschema['minimum']}-{pschema['maximum']})"
                            elif "enum" in pschema:
                                constraint = (
                                    f" [magenta](values: {pschema['enum']})"
                                )
                            params.append(f"{pname}: {ptype}{constraint}")
                        param_str = ", ".join(params)
                        act_branch.add(
                            f"[green]{_extract_item_name(action)}[/green]({param_str})"
                        )
                    else:
                        act_branch.add(
                            f"[green]{_extract_item_name(action)}[/green]()"
                        )

            # Properties
            properties = artifact_data.get("properties", [])
            if properties:
                prop_branch = art_branch.add("[dim]Properties:")
                for prop in properties:
                    prop_data = prop if isinstance(prop, dict) else {}
                    schema = prop_data.get("schema", {})
                    ptype = schema.get("type", "")
                    if "enum" in schema:
                        ptype += f" [magenta](values: {schema['enum']})"
                    prop_branch.add(
                        f"[blue]{_extract_item_name(prop)}[/blue]: {ptype}"
                    )

    # Infeasible commands
    infeasible = affordances.get("infeasible_commands", [])
    if infeasible:
        infeasible_tree = Tree("[bold]Possibly Infeasible Commands")
        for entry in infeasible:
            cmd = entry.get("command", "unknown")
            reason = entry.get("reason", "unknown")
            infeasible_tree.add(f"[red]{cmd}[/red]: [dim]{reason}[/dim]")
        content = Group(
            summary, Rule(style="dim"), tree, Rule(style="dim"), infeasible_tree
        )
    else:
        content = Group(summary, Rule(style="dim"), tree)
    return Panel(content, title="[bold]Discovery Phase", border_style="green")


def create_state_panel(data: dict) -> Optional[Panel]:
    """Create panel showing current state values."""
    state = data.get("discovery", {}).get("state", {})
    property_values = state.get("property_values", {})

    if not property_values:
        return None

    # Group by workspace/artifact
    grouped = {}
    for uri, value in property_values.items():
        parts = uri.split("/")
        # Extract workspace and artifact from URI
        try:
            ws_idx = parts.index("workspaces") + 1
            if ws_idx < len(parts):
                ws_name = parts[ws_idx].split("#")[0]
                art_idx = (
                    parts.index("artifacts") + 1 if "artifacts" in parts else -1
                )
                if art_idx > 0 and art_idx < len(parts):
                    art_name = parts[art_idx]
                    prop_name = parts[-1]
                    key = f"{ws_name}/{art_name}"
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append((prop_name, value))
        except (ValueError, IndexError):
            pass

    # Create tree
    tree = Tree("[bold]Current State")
    for location, props in sorted(grouped.items()):
        branch = tree.add(f"[cyan]{location}")
        for prop_name, value in props:
            # Format value based on type
            if isinstance(value, bool):
                val_str = "[green]true" if value else "[red]false"
            elif isinstance(value, str):
                val_str = f'[yellow]"{value}"'
            elif isinstance(value, (list, tuple)):
                val_str = f"[magenta]{value}"
            else:
                val_str = str(value)
            branch.add(f"{prop_name}: {val_str}")

    return Panel(tree, title="[bold]Environment State", border_style="cyan")


def create_reasoning_panel(data: dict) -> Optional[Panel]:
    """Create panel showing reasoning trace."""
    planning = data.get("planning", {})
    plan = planning.get("plan", {})
    reasoning_trace = plan.get("reasoning_trace", [])

    if not reasoning_trace:
        return None

    # Render reasoning as markdown
    content_parts = []
    for i, trace in enumerate(reasoning_trace):
        if i > 0:
            content_parts.append(Rule(style="dim"))
        content_parts.append(Markdown(trace))

    return Panel(
        Group(*content_parts),
        title="[bold]Reasoning Trace (LLM Analysis)",
        border_style="yellow",
    )


def create_plan_panel(data: dict) -> Panel:
    """Create panel showing generated plan."""
    planning = data.get("planning", {})
    plan = planning.get("plan", {})

    format_type = plan.get("format", "unknown")
    content = plan.get("content", {})
    explanation = plan.get("explanation", "")

    parts = []

    # Explanation
    if explanation:
        parts.append(Text(explanation, style="italic"))
        parts.append(Rule(style="dim"))

    # Plan content
    if format_type == "json_ir":
        json_str = json.dumps(content, indent=2)
        parts.append(
            Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        )
    elif format_type == "python_code":
        parts.append(
            Syntax(
                str(content),
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        )
    else:
        parts.append(Text(str(content)))

    # LLM call stats
    llm_calls = planning.get("llm_calls", 0)
    tokens = planning.get("total_tokens", 0)
    stats = Text(f"\nLLM Calls: {llm_calls}", style="dim")
    if tokens:
        stats.append(f"  Tokens: {tokens}")
    parts.append(stats)

    return Panel(
        Group(*parts),
        title=f"[bold]Generated Plan ({format_type})",
        border_style="magenta",
    )


def create_execution_panel(data: dict) -> Panel:
    """Create panel showing execution results."""
    execution = data.get("execution", {})

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    success = execution.get("success", False)
    status_style = "bold green" if success else "bold red"
    table.add_row(
        "Result", Text("SUCCESS" if success else "FAILED", style=status_style)
    )

    table.add_row("Tree Name", execution.get("tree_name", "N/A"))
    table.add_row("Ticks", str(execution.get("ticks", 0)))
    table.add_row("Final Status", execution.get("final_status", "N/A"))

    tick_history = execution.get("tick_history", [])
    if tick_history:
        history_text = " -> ".join(tick_history)
        table.add_row("Tick History", history_text)

    error = execution.get("error")
    if error:
        table.add_row("Error", Text(error, style="red"))

    return Panel(
        table,
        title="[bold]Execution Results",
        border_style="red" if not success else "green",
    )


def create_errors_panel(data: dict) -> Optional[Panel]:
    """Create panel showing all errors from the experiment."""
    errors = []

    # Top-level error
    if data.get("error"):
        errors.append(("Experiment", data["error"]))

    # Discovery errors (state gathering failures)
    discovery = data.get("discovery", {})
    state = discovery.get("state", {})
    state_errors = state.get("errors", {})
    for uri, error in state_errors.items():
        # Shorten URI for display
        short_uri = uri.split("/")[-3:] if "/" in uri else [uri]
        errors.append((f"State: {'/'.join(short_uri)}", error))

    # Planning error
    planning = data.get("planning", {})
    if planning.get("error"):
        errors.append(("Planning", planning["error"]))

    # Execution error
    execution = data.get("execution", {})
    if execution.get("error"):
        errors.append(("Execution", execution["error"]))

    if not errors:
        return None

    # Build error table
    table = Table(show_header=True, box=box.SIMPLE, expand=True)
    table.add_column("Phase", style="bold yellow", width=30)
    table.add_column("Error", style="red")

    for phase, error in errors:
        table.add_row(phase, Text(str(error), overflow="fold"))

    return Panel(
        table,
        title=f"[bold red]Errors ({len(errors)} found)",
        border_style="red",
    )


def render_experiment(data: dict, expanded: bool = False):
    """Render a complete experiment result."""
    console.print()
    console.print(Rule("[bold blue]Experiment Trace Viewer", style="blue"))
    console.print()

    # Overview
    console.print(create_overview_panel(data))
    console.print()

    # Errors (if any) - show prominently at the top
    errors_panel = create_errors_panel(data)
    if errors_panel:
        console.print(errors_panel)
        console.print()

    # Discovery
    console.print(create_discovery_panel(data, expanded=expanded))
    console.print()

    # Exploration trace (for agentic discovery)
    exploration_panel = create_exploration_trace_panel(data)
    if exploration_panel:
        console.print(exploration_panel)
        console.print()

    # State trace (for agentic state discovery)
    state_trace_panel = create_state_trace_panel(data)
    if state_trace_panel:
        console.print(state_trace_panel)
        console.print()

    # State (if available)
    state_panel = create_state_panel(data)
    if state_panel and expanded:
        console.print(state_panel)
        console.print()

    # Reasoning (if available)
    reasoning_panel = create_reasoning_panel(data)
    if reasoning_panel:
        console.print(reasoning_panel)
        console.print()

    # Generated Plan
    console.print(create_plan_panel(data))
    console.print()

    # Execution
    console.print(create_execution_panel(data))
    console.print()


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _generate_html_card(
    title: str,
    content: str,
    border_color: str = "#4a9eff",
    collapsible: bool = False,
    collapsed: bool = False,
) -> str:
    """Generate an HTML card with proper styling. Optionally collapsible."""
    if collapsible:
        state = "collapsed" if collapsed else ""
        return f"""
    <div class="card collapsible {state}" style="border-color: {border_color};">
        <div class="card-title" style="color: {border_color};" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="collapse-icon">▼</span> {_html_escape(title)}
        </div>
        <div class="card-content">{content}</div>
    </div>
    """
    return f"""
    <div class="card" style="border-color: {border_color};">
        <div class="card-title" style="color: {border_color};">{_html_escape(title)}</div>
        <div class="card-content">{content}</div>
    </div>
    """


def _format_action_schema_html(schema: dict) -> str:
    """Format action schema for HTML display."""
    if not schema:
        return ""
    props = schema.get("properties", {})
    if not props:
        return ""

    parts = []
    for name, prop_schema in props.items():
        ptype = prop_schema.get("type", "any")
        constraint = ""
        if "minimum" in prop_schema and "maximum" in prop_schema:
            constraint = f" <span class='constraint'>(range: {prop_schema['minimum']}-{prop_schema['maximum']})</span>"
        elif "enum" in prop_schema:
            constraint = f" <span class='constraint'>(values: {prop_schema['enum']})</span>"
        parts.append(f"{name}: {ptype}{constraint}")
    return " - Parameters: " + ", ".join(parts)


def _format_property_type_html(schema: dict) -> str:
    """Format property type for HTML display."""
    if not schema:
        return ""
    ptype = schema.get("type", "")
    if "enum" in schema:
        return f" ({ptype}, values: {schema['enum']})"
    return f" ({ptype})" if ptype else ""


def _reconstruct_planning_context(data: dict) -> str:
    """Reconstruct what the model received as capability_model + state context."""
    discovery = data.get("discovery", {})
    affordances = discovery.get("affordances", {})
    state = discovery.get("state", {})

    lines = ["# Available Devices and Capabilities\n"]

    # Build from workspaces
    workspaces = affordances.get("workspaces", {})
    for ws_name, workspace_payload in workspaces.items():
        artifacts = _extract_workspace_artifacts(workspace_payload)
        lines.append(f"\n## {ws_name}")

        for artifact in artifacts:
            art_name = _extract_item_name(artifact)
            art_uri = _extract_item_uri(artifact)
            artifact_data = artifact if isinstance(artifact, dict) else {}
            lines.append(f"\n### {art_name}")
            lines.append(f"URI: `{art_uri}`")

            actions = artifact_data.get("actions", [])
            if actions:
                lines.append("**Actions:**")
                for action in actions:
                    action_data = action if isinstance(action, dict) else {}
                    schema = action_data.get("schema", {})
                    schema_info = ""
                    props = schema.get("properties", {})
                    if props:
                        params = []
                        for pname, pschema in props.items():
                            ptype = pschema.get("type", "any")
                            constraint = ""
                            if "minimum" in pschema and "maximum" in pschema:
                                constraint = f" (range: {pschema['minimum']}-{pschema['maximum']})"
                            elif "enum" in pschema:
                                constraint = f" (values: {pschema['enum']})"
                            params.append(f"{pname}: {ptype}{constraint}")
                        schema_info = " - Parameters: " + ", ".join(params)
                    lines.append(
                        f"  - `{_extract_item_name(action)}`: `{_extract_item_uri(action)}`{schema_info}"
                    )

            properties = artifact_data.get("properties", [])
            if properties:
                lines.append("**Properties:**")
                for prop in properties:
                    prop_data = prop if isinstance(prop, dict) else {}
                    schema = prop_data.get("schema", {})
                    type_info = ""
                    ptype = schema.get("type", "")
                    if "enum" in schema:
                        type_info = f" ({ptype}, values: {schema['enum']})"
                    elif ptype:
                        type_info = f" ({ptype})"
                    lines.append(
                        f"  - `{_extract_item_name(prop)}`: `{_extract_item_uri(prop)}`{type_info}"
                    )

    # Add infeasible commands section if available
    infeasible = affordances.get("infeasible_commands", [])
    if infeasible:
        lines.append("\n\n# Possibly Infeasible Commands\n")
        for entry in infeasible:
            lines.append(
                f"- **{entry.get('command', 'unknown')}**: {entry.get('reason', 'unknown')}"
            )

    # Add state section if available
    property_values = state.get("property_values", {})
    if property_values:
        lines.append("\n\n# Current State\n")
        for uri, value in property_values.items():
            name = uri.split("/")[-1] if "/" in uri else uri
            lines.append(f"- `{name}`: `{value}` ({uri})")

        errors = state.get("errors", {})
        if errors:
            lines.append("\n## Errors:")
            for uri, error in errors.items():
                lines.append(f"- `{uri}`: {error}")

    return "\n".join(lines)


def _generate_planning_context_html(data: dict) -> str:
    """Generate planning context section HTML."""
    context = _reconstruct_planning_context(data)

    if (
        not context
        or context.strip() == "# Available Devices and Capabilities\n"
    ):
        return ""

    # Convert the context to HTML-friendly format
    html_content = (
        f"<pre class='planning-context'>{_html_escape(context)}</pre>"
    )

    return _generate_html_card(
        "Planning Context (What the Model Sees)",
        html_content,
        "#f59e0b",  # amber/orange color
        collapsible=True,
        collapsed=True,  # Start collapsed by default
    )


def _generate_overview_html(data: dict) -> str:
    """Generate overview section HTML."""
    config = data.get("config", {})
    success = data.get("success", False)
    status_class = "success" if success else "error"

    rows = [
        (
            "Status",
            f'<span class="{status_class}">{"SUCCESS" if success else "FAILED"}</span>',
        ),
        ("Goal", _html_escape(data.get("goal", "N/A"))),
        (
            "Config",
            _html_escape(
                data.get(
                    "config_name",
                    config.get("experiment", {}).get("name", "N/A"),
                )
            ),
        ),
        ("Model", _html_escape(config.get("model", {}).get("name", "N/A"))),
        ("Duration", format_duration(data.get("duration_seconds", 0))),
        (
            "Affordance Strategy",
            _html_escape(
                config.get("discovery", {})
                .get("affordances", {})
                .get("strategy", "N/A")
            ),
        ),
        (
            "State Strategy",
            _html_escape(
                config.get("discovery", {})
                .get("state", {})
                .get("strategy", "N/A")
            ),
        ),
    ]

    reasoning = config.get("planning", {}).get("reasoning", {})
    if reasoning.get("enabled"):
        rows.append(
            ("Reasoning", _html_escape(reasoning.get("strategy", "N/A")))
        )
    else:
        rows.append(("Reasoning", "disabled"))

    rows.append(
        (
            "Output Format",
            _html_escape(
                config.get("planning", {})
                .get("output", {})
                .get("format", "N/A")
            ),
        )
    )

    table_rows = "\n".join(
        [f'<tr><td class="label">{k}</td><td>{v}</td></tr>' for k, v in rows]
    )
    return _generate_html_card(
        "Experiment Overview", f"<table>{table_rows}</table>", "#4a9eff"
    )


def _generate_discovery_html(data: dict) -> str:
    """Generate discovery section HTML."""
    discovery = data.get("discovery", {})
    affordances = discovery.get("affordances", {})
    state = discovery.get("state", {})
    stats = affordances.get("stats", {})

    summary = f"""
    <p><strong>Workspaces:</strong> {stats.get('workspace_count', 0)} |
    <strong>Artifacts:</strong> {stats.get('artifact_count', 0)} |
    <strong>Actions:</strong> {stats.get('action_count', 0)} |
    <strong>Properties:</strong> {stats.get('property_count', 0)}</p>
    <p><strong>State Values:</strong> {state.get('property_count', 0)}</p>
    """

    # Build device tree
    workspaces = affordances.get("workspaces", {})
    tree_html = "<div class='tree'>"
    for ws_name, workspace_payload in workspaces.items():
        artifacts = _extract_workspace_artifacts(workspace_payload)
        tree_html += f"<div class='tree-node'><span class='workspace'>{_html_escape(ws_name)}/</span>"
        for artifact in artifacts:
            art_name = _extract_item_name(artifact)
            artifact_data = artifact if isinstance(artifact, dict) else {}
            tree_html += f"<div class='tree-node indent'><span class='artifact'>{_html_escape(art_name)}</span>"

            actions = artifact_data.get("actions", [])
            if actions:
                tree_html += "<div class='tree-node indent2'><span class='dim'>Actions:</span>"
                for action in actions:
                    action_data = action if isinstance(action, dict) else {}
                    schema = action_data.get("schema", {})
                    props = schema.get("properties", {})
                    params = []
                    for pname, pschema in props.items():
                        constraint = ""
                        if "minimum" in pschema and "maximum" in pschema:
                            constraint = f" <span class='constraint'>(range: {pschema['minimum']}-{pschema['maximum']})</span>"
                        elif "enum" in pschema:
                            constraint = f" <span class='constraint'>(values: {pschema['enum']})</span>"
                        params.append(
                            f"{pname}: {pschema.get('type', 'any')}{constraint}"
                        )
                    param_str = ", ".join(params) if params else ""
                    tree_html += f"<div class='tree-node indent3'><span class='action'>{_html_escape(_extract_item_name(action))}</span>({param_str})</div>"
                tree_html += "</div>"

            properties = artifact_data.get("properties", [])
            if properties:
                tree_html += "<div class='tree-node indent2'><span class='dim'>Properties:</span>"
                for prop in properties:
                    prop_data = prop if isinstance(prop, dict) else {}
                    schema = prop_data.get("schema", {})
                    ptype = schema.get("type", "")
                    tree_html += f"<div class='tree-node indent3'><span class='property'>{_html_escape(_extract_item_name(prop))}</span>: {ptype}</div>"
                tree_html += "</div>"

            tree_html += "</div>"
        tree_html += "</div>"
    tree_html += "</div>"

    # Infeasible commands
    infeasible = affordances.get("infeasible_commands", [])
    infeasible_html = ""
    if infeasible:
        infeasible_html = "<hr><div class='tree'><strong>Possibly Infeasible Commands:</strong>"
        for entry in infeasible:
            cmd = _html_escape(entry.get("command", "unknown"))
            reason = _html_escape(entry.get("reason", "unknown"))
            infeasible_html += f"<div class='tree-node indent'><span class='error'>{cmd}</span>: <span class='dim'>{reason}</span></div>"
        infeasible_html += "</div>"

    return _generate_html_card(
        "Discovery Phase", summary + tree_html + infeasible_html, "#22c55e"
    )


def _generate_exploration_trace_html(data: dict) -> str:
    """Generate exploration trace HTML."""
    discovery = data.get("discovery", {})
    exploration_trace = discovery.get("exploration_trace", [])

    if not exploration_trace:
        return ""

    steps_html = "<div class='trace-steps'>"
    for step in exploration_trace:
        fn_name = step.get("function", "unknown")
        args = step.get("arguments", {})
        result = step.get("result", {})
        iteration = step.get("iteration", 0)

        if fn_name == "explore_workspace":
            ws_uri = args.get("workspace_uri", "")
            ws_name = (
                ws_uri.split("/")[-1].replace("#workspace", "")
                if ws_uri
                else "unknown"
            )
            steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> explore_workspace(<span class='highlight'>{_html_escape(ws_name)}</span>)"
            if result.get("sub_workspaces"):
                subs = [
                    _extract_item_name(s, "?") for s in result["sub_workspaces"]
                ]
                steps_html += (
                    f"<div class='trace-result'>Sub-workspaces: {subs}</div>"
                )
            if result.get("artifacts"):
                arts = [_extract_item_name(a, "?") for a in result["artifacts"]]
                steps_html += (
                    f"<div class='trace-result'>Artifacts: {arts}</div>"
                )
            steps_html += "</div>"

        elif fn_name == "inspect_artifact":
            art_uri = args.get("artifact_uri", "")
            art_name = (
                art_uri.split("/")[-1].replace("#artifact", "")
                if art_uri
                else "unknown"
            )
            steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> inspect_artifact(<span class='highlight'>{_html_escape(art_name)}</span>)"
            if result.get("actions"):
                actions = [
                    _extract_item_name(a, "?") for a in result["actions"]
                ]
                steps_html += f"<div class='trace-result'>Actions: <span class='action'>{actions}</span></div>"
            if result.get("properties"):
                props = [
                    _extract_item_name(p, "?") for p in result["properties"]
                ]
                steps_html += f"<div class='trace-result'>Properties: <span class='property'>{props}</span></div>"
            steps_html += "</div>"

        elif fn_name == "done_exploring":
            reason = args.get("reason", result.get("reason", "done"))
            steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> done_exploring(<span class='success'>\"{_html_escape(reason)}\"</span>)</div>"

    steps_html += "</div>"
    return _generate_html_card(
        f"Agentic Discovery Trace ({len(exploration_trace)} tool calls)",
        steps_html,
        "#4a9eff",
    )


def _generate_state_trace_html(data: dict) -> str:
    """Generate state trace HTML."""
    discovery = data.get("discovery", {})
    state_trace = discovery.get("state_trace", [])

    if not state_trace:
        return ""

    steps_html = "<div class='trace-steps'>"
    for step in state_trace:
        fn_name = step.get("function", "unknown")
        args = step.get("arguments", {})
        result = step.get("result", {})
        iteration = step.get("iteration", 0)

        if fn_name == "read_property":
            prop_uri = args.get("property_uri", "")
            reason = args.get("reason", "")
            # Extract artifact and property name from URI
            parts = prop_uri.split("/") if prop_uri else []
            prop_name = parts[-1] if parts else "unknown"
            artifact_name = "unknown"
            if "artifacts" in parts:
                art_idx = parts.index("artifacts")
                if art_idx + 1 < len(parts):
                    artifact_name = parts[art_idx + 1]
            display_name = f"{artifact_name}.{prop_name}"

            if result.get("success"):
                value = result.get("value")
                steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> read_property(<span class='highlight'>{_html_escape(display_name)}</span>) -> <span class='success'>{value}</span>"
            else:
                error = result.get("error", "unknown error")
                steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> read_property(<span class='highlight'>{_html_escape(display_name)}</span>) -> <span class='error'>ERROR: {_html_escape(error)}</span>"

            if reason:
                steps_html += f"<div class='trace-result'>Reason: {_html_escape(reason)}</div>"
            steps_html += "</div>"

        elif fn_name == "done_gathering":
            summary = args.get("summary", result.get("summary", "done"))
            steps_html += f"<div class='trace-step'><span class='step-num'>Step {iteration}:</span> done_gathering(<span class='success'>\"{_html_escape(summary)}\"</span>)</div>"

    steps_html += "</div>"
    return _generate_html_card(
        f"Agentic State Trace ({len(state_trace)} reads)", steps_html, "#06b6d4"
    )


def _generate_reasoning_html(data: dict) -> str:
    """Generate reasoning trace HTML."""
    planning = data.get("planning", {})
    plan = planning.get("plan", {})
    reasoning_trace = plan.get("reasoning_trace", [])

    if not reasoning_trace:
        return ""

    content = ""
    for trace in reasoning_trace:
        # Convert markdown-like formatting to HTML
        formatted = _html_escape(trace)
        formatted = formatted.replace("\n", "<br>")
        content += f"<div class='reasoning-block'>{formatted}</div>"

    return _generate_html_card(
        "Reasoning Trace (LLM Analysis)", content, "#eab308"
    )


def _generate_plan_html(data: dict) -> str:
    """Generate plan HTML."""
    planning = data.get("planning", {})
    plan = planning.get("plan", {})
    format_type = plan.get("format", "unknown")
    content = plan.get("content", {})
    explanation = plan.get("explanation", "")

    html_content = ""
    if explanation:
        html_content += (
            f"<p class='explanation'>{_html_escape(explanation)}</p><hr>"
        )

    if format_type == "json_ir":
        json_str = json.dumps(content, indent=2)
        html_content += f"<pre class='code json'>{_html_escape(json_str)}</pre>"
    elif format_type == "python_code":
        html_content += (
            f"<pre class='code python'>{_html_escape(str(content))}</pre>"
        )
    else:
        html_content += f"<pre>{_html_escape(str(content))}</pre>"

    llm_calls = planning.get("llm_calls", 0)
    html_content += f"<p class='stats'>LLM Calls: {llm_calls}</p>"

    return _generate_html_card(
        f"Generated Plan ({format_type})", html_content, "#a855f7"
    )


def _generate_execution_html(data: dict) -> str:
    """Generate execution results HTML."""
    execution = data.get("execution", {})
    success = execution.get("success", False)
    status_class = "success" if success else "error"

    rows = [
        (
            "Result",
            f'<span class="{status_class}">{"SUCCESS" if success else "FAILED"}</span>',
        ),
        ("Tree Name", _html_escape(execution.get("tree_name", "N/A"))),
        ("Ticks", str(execution.get("ticks", 0))),
        ("Final Status", _html_escape(execution.get("final_status", "N/A"))),
    ]

    tick_history = execution.get("tick_history", [])
    if tick_history:
        rows.append(("Tick History", " -> ".join(tick_history)))

    error = execution.get("error")
    if error:
        rows.append(
            ("Error", f'<span class="error">{_html_escape(error)}</span>')
        )

    table_rows = "\n".join(
        [f'<tr><td class="label">{k}</td><td>{v}</td></tr>' for k, v in rows]
    )
    border_color = "#22c55e" if success else "#ef4444"
    return _generate_html_card(
        "Execution Results", f"<table>{table_rows}</table>", border_color
    )


def _generate_errors_html(data: dict) -> str:
    """Generate errors panel HTML."""
    errors = []

    if data.get("error"):
        errors.append(("Experiment", data["error"]))

    discovery = data.get("discovery", {})
    state = discovery.get("state", {})
    state_errors = state.get("errors", {})
    for uri, error in state_errors.items():
        short_uri = "/".join(uri.split("/")[-3:]) if "/" in uri else uri
        errors.append((f"State: {short_uri}", error))

    planning = data.get("planning", {})
    if planning.get("error"):
        errors.append(("Planning", planning["error"]))

    execution = data.get("execution", {})
    if execution.get("error"):
        errors.append(("Execution", execution["error"]))

    if not errors:
        return ""

    table_rows = "\n".join(
        [
            f'<tr><td class="label">{_html_escape(phase)}</td><td class="error">{_html_escape(err)}</td></tr>'
            for phase, err in errors
        ]
    )
    return _generate_html_card(
        f"Errors ({len(errors)} found)",
        f"<table>{table_rows}</table>",
        "#ef4444",
    )


def export_html(data: dict, output_path: str):
    """Export experiment to a clean HTML file with proper styling."""

    sections = [
        _generate_overview_html(data),
        _generate_errors_html(data),
        _generate_discovery_html(data),
        _generate_exploration_trace_html(data),
        _generate_state_trace_html(data),
        _generate_planning_context_html(data),  # Collapsible planning context
        _generate_reasoning_html(data),
        _generate_plan_html(data),
        _generate_execution_html(data),
    ]

    # Filter out empty sections
    sections = [s for s in sections if s.strip()]

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Trace - {_html_escape(data.get('config_name', 'Unknown'))}</title>
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
            color: #4a9eff;
            text-align: center;
            border-bottom: 2px solid #4a9eff;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Trace Viewer</h1>
        {"".join(sections)}
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(full_html)

    console.print(f"[green]Exported to {output_path}")


def get_latest_experiment(
    results_dir: str = "experiments/results",
) -> Optional[Path]:
    """Get the most recent experiment result file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    # Find all JSON files
    json_files = list(results_path.glob("**/*.json"))
    json_files = [f for f in json_files if f.name != "summary.json"]

    if not json_files:
        return None

    return max(json_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="View experiment traces in human-readable format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m viewers.trace_viewer experiments/results/ablation_*/cot_*.json
  uv run python -m viewers.trace_viewer --latest
  uv run python -m viewers.trace_viewer --expand experiments/results/ablation_*/cot_*.json
  uv run python -m viewers.trace_viewer --html output.html experiments/results/ablation_*/cot_*.json
        """,
    )
    parser.add_argument("files", nargs="*", help="Experiment result JSON files")
    parser.add_argument(
        "--latest", action="store_true", help="View the most recent experiment"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Show expanded details (all affordances, state)",
    )
    parser.add_argument("--html", metavar="FILE", help="Export to HTML file")
    parser.add_argument(
        "--results-dir", default="experiments/results", help="Results directory"
    )

    args = parser.parse_args()

    files = []

    if args.latest:
        latest = get_latest_experiment(args.results_dir)
        if latest:
            files.append(latest)
        else:
            console.print(f"[red]No experiments found in {args.results_dir}/")
            return 1

    files.extend(Path(f) for f in args.files)

    if not files:
        console.print(
            "[yellow]Usage: python -m viewers.trace_viewer [--latest] [--expand] [--html FILE] [experiment_files...]"
        )
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
            export_html(data, args.html)
        else:
            render_experiment(data, expanded=args.expand)

    return 0


if __name__ == "__main__":
    sys.exit(main())
