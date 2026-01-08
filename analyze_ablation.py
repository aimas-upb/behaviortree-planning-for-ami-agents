#!/usr/bin/env python3
"""
Analyze ablation experiment results and generate charts.

Usage:
    python analyze_ablation.py experiments/results/homebench_ablation_*/
    python analyze_ablation.py --latest
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


def analyze_results(results: list[dict]) -> pd.DataFrame:
    """Convert results to a DataFrame for analysis."""
    rows = []
    for r in results:
        config = r.get('config_name', 'unknown')
        prompt_id = r.get('prompt_id', r.get('filename', 'unknown'))

        # Determine task complexity
        if '_one_' in prompt_id:
            complexity = 'single'
        elif '_multi_' in prompt_id:
            complexity = 'multi'
        else:
            complexity = 'unknown'

        # Get evaluation metrics
        eval_data = r.get('evaluation', {})

        rows.append({
            'config': config,
            'prompt_id': prompt_id,
            'complexity': complexity,
            'success': r.get('success', False),
            'error': r.get('error'),
            'correct_actions': eval_data.get('correct', 0),
            'expected_actions': eval_data.get('total_expected', 0),
            'executed_actions': eval_data.get('total_executed', 0),
            'precision': eval_data.get('precision', 0),
            'recall': eval_data.get('recall', 0),
            'duration': r.get('duration_seconds', 0),
        })

    return pd.DataFrame(rows)


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT ANALYSIS")
    print("=" * 70)

    print(f"\nTotal experiments: {len(df)}")
    print(f"Configs tested: {df['config'].nunique()}")
    print(f"Prompts tested: {df['prompt_id'].nunique()}")

    # Success rate by config
    print("\n" + "-" * 50)
    print("SUCCESS RATE BY CONFIG")
    print("-" * 50)

    config_stats = df.groupby('config').agg({
        'success': ['sum', 'count', 'mean'],
        'correct_actions': 'sum',
        'expected_actions': 'sum',
        'precision': 'mean',
        'recall': 'mean',
        'duration': 'mean',
    }).round(3)

    config_stats.columns = ['successes', 'total', 'success_rate',
                            'correct_actions', 'expected_actions',
                            'avg_precision', 'avg_recall', 'avg_duration']
    config_stats['action_accuracy'] = (config_stats['correct_actions'] /
                                        config_stats['expected_actions']).round(3)

    print(config_stats.to_string())

    # Success rate by complexity
    print("\n" + "-" * 50)
    print("SUCCESS RATE BY TASK COMPLEXITY")
    print("-" * 50)

    complexity_stats = df.groupby(['config', 'complexity']).agg({
        'success': ['sum', 'count', 'mean'],
    }).round(3)
    complexity_stats.columns = ['successes', 'total', 'success_rate']
    print(complexity_stats.to_string())

    # Error analysis
    print("\n" + "-" * 50)
    print("ERROR ANALYSIS")
    print("-" * 50)

    errors = df[df['error'].notna()]['error'].value_counts().head(10)
    if len(errors) > 0:
        print("\nTop errors:")
        for err, count in errors.items():
            err_short = str(err)[:80] + "..." if len(str(err)) > 80 else str(err)
            print(f"  {count}x: {err_short}")
    else:
        print("No errors recorded.")

    return config_stats


def create_charts(df: pd.DataFrame, output_dir: Path):
    """Generate visualization charts."""
    output_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
              '#00BCD4', '#FFEB3B', '#795548', '#607D8B']

    configs = df['config'].unique()

    # 1. Success Rate Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = df.groupby('config')['success'].mean() * 100
    bars = ax.bar(range(len(success_rates)), success_rates.values,
                  color=colors[:len(configs)])
    ax.set_xticks(range(len(success_rates)))
    ax.set_xticklabels(success_rates.index, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Configuration')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, success_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'success_rate.png'}")

    # 2. Success Rate by Complexity (Grouped Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    complexity_data = df.groupby(['config', 'complexity'])['success'].mean() * 100
    complexity_df = complexity_data.unstack(fill_value=0)

    x = np.arange(len(complexity_df.index))
    width = 0.35

    if 'single' in complexity_df.columns:
        bars1 = ax.bar(x - width/2, complexity_df['single'], width,
                       label='Single Action', color='#4CAF50')
    if 'multi' in complexity_df.columns:
        bars2 = ax.bar(x + width/2, complexity_df['multi'], width,
                       label='Multi Action', color='#2196F3')

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Config and Task Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(complexity_df.index, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_complexity.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'success_by_complexity.png'}")

    # 3. Action Accuracy (Precision/Recall)
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = df.groupby('config').agg({
        'precision': 'mean',
        'recall': 'mean',
    }) * 100

    x = np.arange(len(metrics.index))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics['precision'], width,
                   label='Precision', color='#E91E63')
    bars2 = ax.bar(x + width/2, metrics['recall'], width,
                   label='Recall', color='#9C27B0')

    ax.set_ylabel('Score (%)')
    ax.set_title('Precision and Recall by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.index, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'precision_recall.png'}")

    # 4. Execution Time Distribution
    if df['duration'].sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        config_durations = [df[df['config'] == c]['duration'].values
                           for c in configs]
        bp = ax.boxplot(config_durations, labels=configs, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors[:len(configs)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Duration (seconds)')
        ax.set_title('Execution Time Distribution by Configuration')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_dir / 'duration_boxplot.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'duration_boxplot.png'}")

    # 5. Heatmap of Success by Prompt and Config
    if len(configs) > 1 and len(df['prompt_id'].unique()) <= 50:
        fig, ax = plt.subplots(figsize=(12, 8))

        pivot = df.pivot_table(index='prompt_id', columns='config',
                               values='success', aggfunc='first')

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([p[:25] + '...' if len(p) > 25 else p for p in pivot.index],
                          fontsize=7)
        ax.set_title('Success Heatmap (Green=Success, Red=Failure)')

        plt.colorbar(im, ax=ax, label='Success')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_heatmap.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir / 'success_heatmap.png'}")

    # 6. Summary Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Success rate pie
    ax = axes[0, 0]
    success_counts = df.groupby('config')['success'].sum()
    fail_counts = df.groupby('config')['success'].count() - success_counts

    success_rates = df.groupby('config')['success'].mean() * 100
    ax.bar(range(len(success_rates)), success_rates.values, color=colors[:len(configs)])
    ax.set_xticks(range(len(success_rates)))
    ax.set_xticklabels(success_rates.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.set_ylim(0, 100)

    # Complexity breakdown
    ax = axes[0, 1]
    complexity_counts = df.groupby('complexity').size()
    ax.pie(complexity_counts.values, labels=complexity_counts.index,
           autopct='%1.1f%%', colors=['#4CAF50', '#2196F3', '#FF9800'])
    ax.set_title('Task Complexity Distribution')

    # Precision/Recall
    ax = axes[1, 0]
    metrics = df.groupby('config').agg({'precision': 'mean', 'recall': 'mean'}) * 100
    x = np.arange(len(metrics.index))
    width = 0.35
    ax.bar(x - width/2, metrics['precision'], width, label='Precision', color='#E91E63')
    ax.bar(x + width/2, metrics['recall'], width, label='Recall', color='#9C27B0')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('Precision & Recall')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Summary stats text
    ax = axes[1, 1]
    ax.axis('off')

    total = len(df)
    successes = df['success'].sum()
    summary_text = f"""
SUMMARY STATISTICS

Total Experiments: {total}
Overall Success Rate: {successes/total*100:.1f}%

Correct Actions: {df['correct_actions'].sum()}/{df['expected_actions'].sum()}
Action Accuracy: {df['correct_actions'].sum()/max(df['expected_actions'].sum(),1)*100:.1f}%

Best Config: {success_rates.idxmax()} ({success_rates.max():.1f}%)
Worst Config: {success_rates.idxmin()} ({success_rates.min():.1f}%)

Avg Duration: {df['duration'].mean():.2f}s
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Summary')

    plt.tight_layout()
    plt.savefig(output_dir / 'dashboard.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'dashboard.png'}")


def get_latest_results_dir(base_dir: str = "experiments/results") -> Path:
    """Find the most recent results directory."""
    base = Path(base_dir)
    dirs = sorted(base.glob("homebench_ablation_*"))
    if not dirs:
        raise ValueError(f"No results found in {base_dir}")
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation experiment results")
    parser.add_argument("results_dir", nargs="?", help="Results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results")
    parser.add_argument("--output", "-o", default=None, help="Output directory for charts")

    args = parser.parse_args()

    if args.latest or not args.results_dir:
        results_dir = get_latest_results_dir()
    else:
        results_dir = Path(args.results_dir)

    print(f"Analyzing results from: {results_dir}")

    # Load results
    results = load_results(results_dir)
    if not results:
        print("No results found!")
        return 1

    print(f"Loaded {len(results)} experiment results")

    # Analyze
    df = analyze_results(results)
    config_stats = print_summary_stats(df)

    # Generate charts
    output_dir = Path(args.output) if args.output else results_dir / "charts"
    create_charts(df, output_dir)

    print(f"\nCharts saved to: {output_dir}")
    print("\nTo view the dashboard:")
    print(f"  xdg-open {output_dir / 'dashboard.png'}")

    return 0


if __name__ == "__main__":
    exit(main())
