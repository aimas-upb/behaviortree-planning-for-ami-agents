#!/usr/bin/env python3
"""
Generate HTML reports for ablation experiment results.

Usage:
    uv run python -m scripts.analysis.generate_html_report experiments/results/homebench_ablation_*/
    uv run python -m scripts.analysis.generate_html_report --latest
"""

import json
import html
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional


def escape(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text else ""


def format_json(obj, indent: int = 2) -> str:
    """Format JSON for display."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except:
        return str(obj)


def get_status_class(success: bool) -> str:
    """Get CSS class for status."""
    return "success" if success else "error"


def get_status_text(success: bool) -> str:
    """Get status text."""
    return "SUCCESS" if success else "FAILED"


CSS_STYLES = """
* { box-sizing: border-box; }
body {
    background-color: #1a1a2e;
    color: #e0e0e0;
    padding: 20px;
    font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 14px;
    line-height: 1.5;
}
.container {
    max-width: 1400px;
    margin: 0 auto;
}
h1 {
    color: #4a9eff;
    text-align: center;
    border-bottom: 2px solid #4a9eff;
    padding-bottom: 10px;
}
h2 {
    color: #4a9eff;
    margin-top: 30px;
}
.card {
    border: 2px solid;
    border-radius: 8px;
    margin: 20px 0;
    background: #16213e;
}
.card-title {
    font-weight: bold;
    font-size: 16px;
    padding: 12px 16px;
    border-bottom: 1px solid #333;
    background: rgba(0,0,0,0.2);
    border-radius: 6px 6px 0 0;
}
.card-content {
    padding: 16px;
}
table {
    width: 100%;
    border-collapse: collapse;
}
td, th {
    padding: 8px 12px;
    border-bottom: 1px solid #333;
    text-align: left;
}
th {
    color: #4a9eff;
    background: rgba(0,0,0,0.3);
}
td.label {
    color: #06b6d4;
    font-weight: bold;
    width: 200px;
}
.success { color: #22c55e; font-weight: bold; }
.error { color: #ef4444; font-weight: bold; }
.warning { color: #f59e0b; }
.highlight { color: #eab308; }
.dim { color: #888; }
.workspace { color: #06b6d4; }
.artifact { color: #eab308; }
.action { color: #22c55e; }
.property { color: #4a9eff; }
.code {
    background: #0d1117;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.collapsible .card-title {
    cursor: pointer;
    user-select: none;
}
.collapsible .card-title:hover {
    opacity: 0.8;
}
.collapsible .collapse-icon {
    display: inline-block;
    transition: transform 0.2s ease;
    margin-right: 8px;
}
.collapsible.collapsed .collapse-icon {
    transform: rotate(-90deg);
}
.collapsible.collapsed .card-content {
    display: none;
}
.trace-step {
    margin: 12px 0;
    padding: 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 4px;
    border-left: 3px solid #4a9eff;
}
.trace-step .step-num {
    color: #06b6d4;
    font-weight: bold;
}
.trace-step .function {
    color: #22c55e;
}
.trace-result {
    margin-left: 20px;
    color: #888;
    font-size: 13px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 16px 0;
}
.metric-box {
    background: rgba(0,0,0,0.3);
    padding: 16px;
    border-radius: 8px;
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #4a9eff;
}
.metric-label {
    color: #888;
    font-size: 12px;
    margin-top: 4px;
}
.nav-links {
    display: flex;
    gap: 16px;
    margin: 20px 0;
    flex-wrap: wrap;
}
.nav-links a {
    color: #4a9eff;
    text-decoration: none;
    padding: 8px 16px;
    background: rgba(74, 158, 255, 0.1);
    border-radius: 4px;
    border: 1px solid #4a9eff;
}
.nav-links a:hover {
    background: rgba(74, 158, 255, 0.2);
}
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
}
.result-card {
    background: #16213e;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 16px;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #4a9eff;
}
.result-card.success-card {
    border-left: 4px solid #22c55e;
}
.result-card.error-card {
    border-left: 4px solid #ef4444;
}
.result-card a {
    color: #4a9eff;
    text-decoration: none;
}
.result-card a:hover {
    text-decoration: underline;
}
.result-title {
    font-weight: bold;
    margin-bottom: 8px;
}
.result-meta {
    color: #888;
    font-size: 12px;
}
.reasoning-block {
    background: rgba(0,0,0,0.2);
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
    border-left: 3px solid #a855f7;
}
.filter-bar {
    display: flex;
    gap: 16px;
    margin: 20px 0;
    flex-wrap: wrap;
    align-items: center;
}
.filter-bar label {
    color: #888;
}
.filter-bar select {
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    padding: 8px 12px;
    border-radius: 4px;
}
"""


def generate_individual_html(result: dict, output_path: Path) -> None:
    """Generate HTML for a single experiment result."""
    config = result.get('config', {})
    discovery = result.get('discovery', {})
    planning = result.get('planning', {})
    execution = result.get('execution', {})
    evaluation = result.get('evaluation', {})

    success = result.get('success', False)
    goal = result.get('goal', 'N/A')
    config_name = result.get('config_name', 'unknown')
    prompt_id = result.get('prompt_id', 'unknown')
    duration = result.get('duration_seconds', 0)

    # Extract config details
    model_name = config.get('model', {}).get('name', 'N/A')
    affordance_strategy = config.get('discovery', {}).get('affordances', {}).get('strategy', 'N/A')
    state_strategy = config.get('discovery', {}).get('state', {}).get('strategy', 'N/A')
    reasoning_config = config.get('planning', {}).get('reasoning', {})
    reasoning_enabled = reasoning_config.get('enabled', False)
    reasoning_strategy = reasoning_config.get('strategy', 'N/A') if reasoning_enabled else 'disabled'
    output_format = config.get('planning', {}).get('output', {}).get('format', 'N/A')

    # Discovery stats
    affordances = discovery.get('affordances', {})
    stats = affordances.get('stats', {})
    workspaces = affordances.get('workspaces', {})
    state_values = discovery.get('state', {}).get('property_values', {})
    exploration_trace = discovery.get('exploration_trace', [])

    # Planning
    plan = planning.get('plan', {})
    plan_content = plan.get('content', {})
    explanation = plan.get('explanation', '')
    reasoning_trace = plan.get('reasoning_trace', [])

    # Execution
    exec_success = execution.get('success', False)
    tree_name = execution.get('tree_name', 'N/A')
    ticks = execution.get('ticks', 0)
    final_status = execution.get('final_status', 'N/A')
    tick_history = execution.get('tick_history', [])
    exec_error = execution.get('error', '')

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment: {escape(prompt_id)} - {escape(config_name)}</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <h1>Experiment Trace Viewer</h1>

        <div class="nav-links">
            <a href="index.html">← Back to Index</a>
        </div>

        <!-- Overview Card -->
        <div class="card" style="border-color: #4a9eff;">
            <div class="card-title" style="color: #4a9eff;">Experiment Overview</div>
            <div class="card-content">
                <table>
                    <tr><td class="label">Status</td><td><span class="{get_status_class(success)}">{get_status_text(success)}</span></td></tr>
                    <tr><td class="label">Prompt ID</td><td>{escape(prompt_id)}</td></tr>
                    <tr><td class="label">Config</td><td>{escape(config_name)}</td></tr>
                    <tr><td class="label">Goal</td><td>{escape(goal)}</td></tr>
                    <tr><td class="label">Model</td><td>{escape(model_name)}</td></tr>
                    <tr><td class="label">Duration</td><td>{duration:.2f}s</td></tr>
                    <tr><td class="label">Affordance Strategy</td><td>{escape(affordance_strategy)}</td></tr>
                    <tr><td class="label">State Strategy</td><td>{escape(state_strategy)}</td></tr>
                    <tr><td class="label">Reasoning</td><td>{escape(reasoning_strategy)}</td></tr>
                    <tr><td class="label">Output Format</td><td>{escape(output_format)}</td></tr>
                </table>
                {f'<div class="error" style="margin-top: 16px;">Error: {escape(result.get("error", ""))}</div>' if result.get('error') else ''}
            </div>
        </div>

        <!-- Evaluation Metrics -->
        <div class="card" style="border-color: #06b6d4;">
            <div class="card-title" style="color: #06b6d4;">Evaluation Metrics</div>
            <div class="card-content">
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-value">{evaluation.get('correct', 0)}/{evaluation.get('total_expected', 0)}</div>
                        <div class="metric-label">Correct Actions</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{evaluation.get('precision', 0)*100:.1f}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{evaluation.get('recall', 0)*100:.1f}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{evaluation.get('total_executed', 0)}</div>
                        <div class="metric-label">Executed Actions</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Discovery Phase -->
        <div class="card collapsible" style="border-color: #22c55e;">
            <div class="card-title" style="color: #22c55e;" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="collapse-icon">▼</span> Discovery Phase
            </div>
            <div class="card-content">
                <p>
                    <strong>Workspaces:</strong> {stats.get('workspace_count', 0)} |
                    <strong>Artifacts:</strong> {stats.get('artifact_count', 0)} |
                    <strong>Actions:</strong> {stats.get('action_count', 0)} |
                    <strong>Properties:</strong> {stats.get('property_count', 0)}
                </p>
                <p><strong>State Values Collected:</strong> {len(state_values)}</p>

                {"<h4>Exploration Trace</h4>" if exploration_trace else ""}
                {"".join(f'''
                <div class="trace-step">
                    <span class="step-num">Step {step.get('iteration', i+1)}:</span>
                    <span class="function">{escape(step.get('function', ''))}</span>
                    (<code>{escape(str(step.get('arguments', {})))[:100]}</code>)
                    <div class="trace-result">{escape(str(step.get('result', {})))[:200]}...</div>
                </div>
                ''' for i, step in enumerate(exploration_trace))}

                <h4>Discovered Devices</h4>
                <div class="code">{escape(format_json(workspaces))}</div>

                {f'<h4>State Values</h4><div class="code">{escape(format_json(state_values))}</div>' if state_values else ''}
            </div>
        </div>

        <!-- Planning Phase -->
        <div class="card collapsible" style="border-color: #a855f7;">
            <div class="card-title" style="color: #a855f7;" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="collapse-icon">▼</span> Planning Phase ({escape(output_format)})
            </div>
            <div class="card-content">
                {f'<div class="explanation"><strong>Explanation:</strong> {escape(explanation)}</div>' if explanation else ''}

                {f'''
                <h4>Reasoning Trace</h4>
                {"".join(f'<div class="reasoning-block">{escape(trace)}</div>' for trace in reasoning_trace)}
                ''' if reasoning_trace else ''}

                <h4>Generated Plan</h4>
                <div class="code">{escape(format_json(plan_content))}</div>

                <p class="dim">LLM Calls: {planning.get('llm_calls', 0)} | Tokens: {planning.get('total_tokens', 0)}</p>
            </div>
        </div>

        <!-- Execution Phase -->
        <div class="card" style="border-color: {'#22c55e' if exec_success else '#ef4444'};">
            <div class="card-title" style="color: {'#22c55e' if exec_success else '#ef4444'};">Execution Results</div>
            <div class="card-content">
                <table>
                    <tr><td class="label">Result</td><td><span class="{get_status_class(exec_success)}">{get_status_text(exec_success)}</span></td></tr>
                    <tr><td class="label">Tree Name</td><td>{escape(tree_name) or 'N/A'}</td></tr>
                    <tr><td class="label">Ticks</td><td>{ticks}</td></tr>
                    <tr><td class="label">Final Status</td><td>{escape(final_status) or 'N/A'}</td></tr>
                </table>
                {f'<div class="error" style="margin-top: 16px;">Error: {escape(exec_error)}</div>' if exec_error else ''}

                {f'''
                <h4>Tick History</h4>
                <div class="code">{escape(format_json(tick_history))}</div>
                ''' if tick_history else ''}
            </div>
        </div>

        <!-- Expected vs Executed -->
        <div class="card collapsible collapsed" style="border-color: #f59e0b;">
            <div class="card-title" style="color: #f59e0b;" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="collapse-icon">▼</span> Expected vs Executed Actions
            </div>
            <div class="card-content">
                <h4>Expected Actions</h4>
                <div class="code">{escape(format_json(evaluation.get('expected_actions', [])))}</div>

                <h4>Executed Actions</h4>
                <div class="code">{escape(format_json(evaluation.get('executed_actions', [])))}</div>
            </div>
        </div>

        <!-- Full Result JSON -->
        <div class="card collapsible collapsed" style="border-color: #666;">
            <div class="card-title" style="color: #666;" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="collapse-icon">▼</span> Full Result JSON
            </div>
            <div class="card-content">
                <div class="code">{escape(format_json(result))}</div>
            </div>
        </div>

    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_index_html(results: list[dict], output_dir: Path, charts_dir: Optional[Path] = None) -> None:
    """Generate index HTML with overview of all results."""

    # Group results by config and prompt
    by_config = defaultdict(list)
    by_prompt = defaultdict(list)
    for r in results:
        by_config[r.get('config_name', 'unknown')].append(r)
        by_prompt[r.get('prompt_id', 'unknown')].append(r)

    # Calculate statistics
    total = len(results)
    total_success = sum(1 for r in results if r.get('success'))
    configs = list(by_config.keys())
    prompts = list(by_prompt.keys())

    # Config stats
    config_stats = []
    for config in sorted(configs):
        config_results = by_config[config]
        successes = sum(1 for r in config_results if r.get('success'))
        config_stats.append({
            'name': config,
            'successes': successes,
            'total': len(config_results),
            'rate': successes / len(config_results) * 100 if config_results else 0,
        })

    # Sort by success rate descending
    config_stats.sort(key=lambda x: x['rate'], reverse=True)

    # Generate result cards HTML
    result_cards = []
    for r in sorted(results, key=lambda x: (x.get('prompt_id', ''), x.get('config_name', ''))):
        prompt_id = r.get('prompt_id', 'unknown')
        config_name = r.get('config_name', 'unknown')
        success = r.get('success', False)
        goal = r.get('goal', '')[:80] + '...' if len(r.get('goal', '')) > 80 else r.get('goal', '')
        duration = r.get('duration_seconds', 0)
        eval_data = r.get('evaluation', {})

        filename = f"{prompt_id}_{config_name}.html"
        card_class = "success-card" if success else "error-card"

        result_cards.append(f"""
        <div class="result-card {card_class}" data-config="{escape(config_name)}" data-prompt="{escape(prompt_id)}" data-success="{str(success).lower()}">
            <div class="result-title">
                <a href="{filename}">{escape(prompt_id)}</a>
            </div>
            <div>Config: <strong>{escape(config_name)}</strong></div>
            <div>Status: <span class="{get_status_class(success)}">{get_status_text(success)}</span></div>
            <div class="result-meta">
                Actions: {eval_data.get('correct', 0)}/{eval_data.get('total_expected', 0)} |
                Duration: {duration:.1f}s
            </div>
            <div class="result-meta" style="margin-top: 8px;">{escape(goal)}</div>
        </div>
        """)

    # Charts section
    charts_html = ""
    if charts_dir and charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            charts_html = """
            <h2>Charts</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            """
            for chart_file in sorted(chart_files):
                rel_path = f"charts/{chart_file.name}"
                charts_html += f"""
                <div style="background: #16213e; padding: 16px; border-radius: 8px;">
                    <img src="{rel_path}" style="max-width: 100%; border-radius: 4px;" alt="{chart_file.stem}">
                </div>
                """
            charts_html += "</div>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ablation Experiment Results</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <h1>Ablation Experiment Results</h1>

        <!-- Summary Metrics -->
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Experiments</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: #22c55e;">{total_success}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{total_success/total*100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len(configs)}</div>
                <div class="metric-label">Configurations</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{len(prompts)}</div>
                <div class="metric-label">Prompts Tested</div>
            </div>
        </div>

        <!-- Config Stats Table -->
        <div class="card" style="border-color: #4a9eff;">
            <div class="card-title" style="color: #4a9eff;">Configuration Performance</div>
            <div class="card-content">
                <table>
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>Success Rate</th>
                            <th>Successes</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td>{escape(s["name"])}</td>
                            <td><span class="{"success" if s["rate"] >= 50 else "warning" if s["rate"] >= 25 else "error"}">{s["rate"]:.1f}%</span></td>
                            <td>{s["successes"]}</td>
                            <td>{s["total"]}</td>
                        </tr>
                        ''' for s in config_stats)}
                    </tbody>
                </table>
            </div>
        </div>

        {charts_html}

        <!-- Filter Bar -->
        <h2>Individual Results</h2>
        <div class="filter-bar">
            <label>Config:</label>
            <select id="configFilter" onchange="filterResults()">
                <option value="">All Configs</option>
                {"".join(f'<option value="{escape(c)}">{escape(c)}</option>' for c in sorted(configs))}
            </select>

            <label>Prompt:</label>
            <select id="promptFilter" onchange="filterResults()">
                <option value="">All Prompts</option>
                {"".join(f'<option value="{escape(p)}">{escape(p)}</option>' for p in sorted(prompts))}
            </select>

            <label>Status:</label>
            <select id="statusFilter" onchange="filterResults()">
                <option value="">All</option>
                <option value="true">Success</option>
                <option value="false">Failed</option>
            </select>
        </div>

        <!-- Result Cards -->
        <div class="result-grid" id="resultsGrid">
            {"".join(result_cards)}
        </div>

    </div>

    <script>
    function filterResults() {{
        const configFilter = document.getElementById('configFilter').value;
        const promptFilter = document.getElementById('promptFilter').value;
        const statusFilter = document.getElementById('statusFilter').value;

        const cards = document.querySelectorAll('.result-card');
        cards.forEach(card => {{
            const config = card.dataset.config;
            const prompt = card.dataset.prompt;
            const success = card.dataset.success;

            const configMatch = !configFilter || config === configFilter;
            const promptMatch = !promptFilter || prompt === promptFilter;
            const statusMatch = !statusFilter || success === statusFilter;

            card.style.display = (configMatch && promptMatch && statusMatch) ? 'block' : 'none';
        }});
    }}
    </script>
</body>
</html>
"""

    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)


def load_results(results_dir: Path) -> list[dict]:
    """Load all experiment results from a directory."""
    results = []
    for f in results_dir.glob("*.json"):
        if f.name == "summary.json":
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
                data['filename'] = f.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


def get_latest_results_dir(base_dir: str = "experiments/results") -> Path:
    """Find the most recent results directory."""
    base = Path(base_dir)
    dirs = sorted(base.glob("homebench_ablation_*"))
    if not dirs:
        raise ValueError(f"No results found in {base_dir}")
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports for ablation results")
    parser.add_argument("results_dir", nargs="?", help="Results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results")
    parser.add_argument("--output", "-o", default=None, help="Output directory for HTML files")

    args = parser.parse_args()

    if args.latest or not args.results_dir:
        results_dir = get_latest_results_dir()
    else:
        results_dir = Path(args.results_dir)

    print(f"Generating HTML reports from: {results_dir}")

    # Load results
    results = load_results(results_dir)
    if not results:
        print("No results found!")
        return 1

    print(f"Loaded {len(results)} experiment results")

    # Output directory
    output_dir = Path(args.output) if args.output else results_dir / "html"
    output_dir.mkdir(exist_ok=True)

    # Check for charts
    charts_dir = results_dir / "charts"

    # Generate individual HTML files
    print("Generating individual HTML reports...")
    for result in results:
        prompt_id = result.get('prompt_id', 'unknown')
        config_name = result.get('config_name', 'unknown')
        filename = f"{prompt_id}_{config_name}.html"
        output_path = output_dir / filename
        generate_individual_html(result, output_path)

    # Generate index
    print("Generating index.html...")
    generate_index_html(results, output_dir, charts_dir)

    print(f"\nHTML reports saved to: {output_dir}")
    print(f"Open: {output_dir / 'index.html'}")

    return 0


if __name__ == "__main__":
    exit(main())
