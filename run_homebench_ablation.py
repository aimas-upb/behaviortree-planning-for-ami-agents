"""
HomeBench Ablation Experiment Runner.

Runs ablation experiments using prompts from the HomeBench dataset
with expected outputs for proper evaluation.
"""

import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    ExperimentConfig, ExperimentMeta, DiscoveryConfig, PlanningConfig,
    ExecutionConfig, ModelConfig, TracingConfig, AffordanceConfig,
    StateConfig, ReasoningConfig, OutputConfig
)
from src.runner import run_experiment


def load_homebench_prompts(
    dataset_path: str = "datasets/HomeBench/converted/test_data.json",
    home_id: Optional[int] = None,
    num_samples: int = 10,
    seed: int = 42,
    filter_success_only: bool = True,
) -> list[dict]:
    """
    Load prompts from HomeBench dataset.

    Args:
        dataset_path: Path to the converted dataset JSON (or benchmark JSON)
        home_id: If specified, only load prompts for this home
        num_samples: Number of prompts to sample
        seed: Random seed for reproducibility
        filter_success_only: Only include prompts with at least one successful action

    Returns:
        List of prompt dictionaries with 'id', 'input', 'output', 'home_id'
    """
    with open(dataset_path) as f:
        raw_data = json.load(f)

    # Handle benchmark format (has 'samples' key) vs raw format (list)
    if isinstance(raw_data, dict) and 'samples' in raw_data:
        data = raw_data['samples']
    else:
        data = raw_data

    # Filter by home_id if specified
    if home_id is not None:
        home_prefix = f"home{home_id}_"
        data = [d for d in data if d['id'].startswith(home_prefix)]

    # Filter to only include prompts with successful actions
    if filter_success_only:
        data = [
            d for d in data
            if any(o.get('execution') == 'success' for o in d['output'])
        ]

    # Sample
    random.seed(seed)
    if len(data) > num_samples:
        data = random.sample(data, num_samples)

    # Extract home_id from each prompt
    for item in data:
        item['home_id'] = int(item['id'].split('_')[0].replace('home', ''))

    return data


def create_config(
    name: str,
    affordance_strategy: str = "exhaustive",
    state_strategy: str = "none",
    reasoning_enabled: bool = False,
    reasoning_strategy: str = "chain_of_thought",
    output_format: str = "json_ir",
    prompt_strategy: str = "detailed",
    model: str = "gpt-4o",
) -> ExperimentConfig:
    """Create an experiment config programmatically."""
    return ExperimentConfig(
        experiment=ExperimentMeta(name=name, description=f"HomeBench Ablation: {name}"),
        discovery=DiscoveryConfig(
            affordances=AffordanceConfig(strategy=affordance_strategy),
            state=StateConfig(strategy=state_strategy),
        ),
        planning=PlanningConfig(
            reasoning=ReasoningConfig(
                enabled=reasoning_enabled,
                strategy=reasoning_strategy,
                max_turns=3,
            ),
            output=OutputConfig(format=output_format),
            prompt_strategy=prompt_strategy,
        ),
        execution=ExecutionConfig(max_ticks=10),
        model=ModelConfig(name=model, temperature=0.0),
        tracing=TracingConfig(enabled=True, output_dir="traces/", verbose=False),
    )


# Ablation configurations to test (all 13 configs)
ABLATION_CONFIGS = {
    # === JSON IR Output Configs (7 configs) ===

    # Baseline configurations
    "baseline": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "none",
    },
    "baseline_with_state": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Reasoning variants
    "cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "reflection": {
        "reasoning_enabled": True,
        "reasoning_strategy": "reflection",
        "output_format": "json_ir",
        "state_strategy": "all",
    },
    "multi_turn": {
        "reasoning_enabled": True,
        "reasoning_strategy": "multi_turn",
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Agentic variants
    "agentic": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },
    "agentic_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },

    # === Fully Agentic Configs (2 configs) ===
    "fully_agentic": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },
    "fully_agentic_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },

    # === Python Code Configs (4 configs) ===
    "python_direct": {
        "reasoning_enabled": False,
        "output_format": "python_code",
        "state_strategy": "none",
    },
    "python_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code",
        "state_strategy": "all",
    },
    "python_unconstrained": {
        "reasoning_enabled": False,
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },
    "python_unconstrained_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },
}


def evaluate_result(result: dict, expected_output: list[dict]) -> dict:
    """
    Evaluate experiment result against expected output.

    Returns dict with:
        - expected_actions: list of expected successful actions
        - executed_actions: list of actions that were executed
        - correct: number of correct actions
        - total_expected: total expected actions
        - precision: correct / executed
        - recall: correct / expected
    """
    # Extract expected successful actions
    expected_actions = []
    for out in expected_output:
        if out.get('execution') == 'success':
            expected_actions.append({
                'affordance': out['affordance'],
                'params': out.get('params', {}),
                'test': out.get('test', {}),
            })

    # Extract executed actions from result
    executed_actions = []
    if result.get('execution') and result['execution'].get('action_history'):
        for action in result['execution']['action_history']:
            executed_actions.append({
                'affordance': action.get('affordance', ''),
                'params': action.get('params', {}),
            })

    # Calculate metrics
    correct = 0
    for expected in expected_actions:
        for executed in executed_actions:
            # Check if affordance matches (URL comparison)
            if expected['affordance'] == executed['affordance']:
                # Check if params match
                if expected['params'] == executed['params']:
                    correct += 1
                    break

    total_expected = len(expected_actions)
    total_executed = len(executed_actions)

    return {
        'expected_actions': expected_actions,
        'executed_actions': executed_actions,
        'correct': correct,
        'total_expected': total_expected,
        'total_executed': total_executed,
        'precision': correct / total_executed if total_executed > 0 else 0.0,
        'recall': correct / total_expected if total_expected > 0 else 0.0,
    }


def run_homebench_ablation(
    configs_to_run: Optional[list[str]] = None,
    home_id: Optional[int] = None,  # None = use all homes in dataset
    num_prompts: int = 50,  # Default to full benchmark
    model: str = "gpt-4o",
    dataset_path: str = "datasets/HomeBench/converted/benchmark_50.json",  # Use curated benchmark
    seed: int = 42,
):
    """Run ablation experiments on HomeBench prompts."""

    # Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)

    # Load prompts
    print(f"Loading prompts from {dataset_path}...")
    prompts = load_homebench_prompts(
        dataset_path=dataset_path,
        home_id=home_id,
        num_samples=num_prompts,
        seed=seed,
    )
    print(f"Loaded {len(prompts)} prompts")

    configs_to_run = configs_to_run or list(ABLATION_CONFIGS.keys())

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/homebench_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"HOMEBENCH ABLATION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Home ID: {home_id if home_id else 'all'}")
    print(f"Prompts: {len(prompts)}")
    print(f"Configurations: {configs_to_run}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")

    # Show prompts
    print("\nPrompts to test:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. [{p['id']}] {p['input'][:60]}...")

    all_results = []

    for prompt_data in prompts:
        prompt_id = prompt_data['id']
        goal = prompt_data['input']
        expected = prompt_data['output']
        prompt_home_id = prompt_data['home_id']
        entry_point = f"http://localhost:8080/workspaces/home{prompt_home_id}#workspace"

        print(f"\n{'=' * 70}")
        print(f"PROMPT: {prompt_id}")
        print(f"Goal: {goal[:100]}...")
        print(f"Entry: {entry_point}")
        print(f"{'=' * 70}")

        for config_name in configs_to_run:
            config_params = ABLATION_CONFIGS[config_name]

            print(f"\n  {'-' * 50}")
            print(f"  CONFIG: {config_name}")
            print(f"  {'-' * 50}")

            try:
                config = create_config(
                    name=config_name,
                    model=model,
                    **config_params,
                )

                result = run_experiment(
                    config=config,
                    goal=goal,
                    entry_point=entry_point,
                    client=client,
                )

                # Evaluate against expected output
                evaluation = evaluate_result(result, expected)

                # Add metadata
                result['prompt_id'] = prompt_id
                result['config_name'] = config_name
                result['goal'] = goal
                result['expected_output'] = expected
                result['evaluation'] = evaluation

                # Print summary
                status = "✓" if result.get('success') else "✗"
                print(f"  {status} Success: {result.get('success')}")
                print(f"    Correct: {evaluation['correct']}/{evaluation['total_expected']}")
                print(f"    Precision: {evaluation['precision']:.2f}, Recall: {evaluation['recall']:.2f}")

                all_results.append(result)

                # Save individual result
                safe_prompt_id = prompt_id.replace('/', '_')
                result_file = output_dir / f"{safe_prompt_id}_{config_name}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'prompt_id': prompt_id,
                    'config_name': config_name,
                    'goal': goal,
                    'success': False,
                    'error': str(e),
                })

    # Generate summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Aggregate by config
    config_stats = {}
    for config_name in configs_to_run:
        config_results = [r for r in all_results if r.get('config_name') == config_name]
        successes = sum(1 for r in config_results if r.get('success'))

        # Average evaluation metrics
        evaluations = [r.get('evaluation', {}) for r in config_results if r.get('evaluation')]
        avg_precision = sum(e.get('precision', 0) for e in evaluations) / len(evaluations) if evaluations else 0
        avg_recall = sum(e.get('recall', 0) for e in evaluations) / len(evaluations) if evaluations else 0
        total_correct = sum(e.get('correct', 0) for e in evaluations)
        total_expected = sum(e.get('total_expected', 0) for e in evaluations)

        config_stats[config_name] = {
            'successes': successes,
            'total': len(config_results),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
        }

        print(f"\n{config_name}:")
        print(f"  Success rate: {successes}/{len(config_results)}")
        print(f"  Avg Precision: {avg_precision:.2f}")
        print(f"  Avg Recall: {avg_recall:.2f}")
        print(f"  Actions correct: {total_correct}/{total_expected}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'home_id': home_id,
        'num_prompts': len(prompts),
        'configs': configs_to_run,
        'model': model,
        'config_stats': config_stats,
        'prompts': [{'id': p['id'], 'input': p['input'][:100]} for p in prompts],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return all_results


def list_sample_prompts(
    dataset_path: str = "datasets/HomeBench/converted/test_data.json",
    home_id: Optional[int] = None,
    num_samples: int = 20,
):
    """List sample prompts from the dataset."""
    prompts = load_homebench_prompts(
        dataset_path=dataset_path,
        home_id=home_id,
        num_samples=num_samples,
        filter_success_only=True,
    )

    print(f"Sample prompts (home={home_id if home_id else 'all'}):\n")
    for p in prompts:
        num_actions = len([o for o in p['output'] if o.get('execution') == 'success'])
        print(f"[{p['id']}] ({num_actions} actions)")
        print(f"  {p['input'][:100]}...")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HomeBench ablation experiments")
    parser.add_argument("--configs", nargs="+", help="Configs to run (default: all)")
    parser.add_argument("--home", type=int, default=None, help="Home ID filter (default: all homes)")
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts (default: 50)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")
    parser.add_argument("--list-prompts", action="store_true", help="List sample prompts")
    parser.add_argument("--dataset", default="datasets/HomeBench/converted/benchmark_50.json",
                       help="Path to dataset (default: curated 50-sample benchmark)")

    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:\n")
        for name, params in ABLATION_CONFIGS.items():
            print(f"  {name}:")
            for k, v in params.items():
                print(f"    {k}: {v}")
        sys.exit(0)

    if args.list_prompts:
        list_sample_prompts(
            dataset_path=args.dataset,
            home_id=args.home,
            num_samples=20,
        )
        sys.exit(0)

    run_homebench_ablation(
        configs_to_run=args.configs,
        home_id=args.home,
        num_prompts=args.num_prompts,
        model=args.model,
        dataset_path=args.dataset,
        seed=args.seed,
    )
