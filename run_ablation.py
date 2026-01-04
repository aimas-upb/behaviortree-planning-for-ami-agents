"""
Ablation experiment runner.

Runs multiple configurations and saves traces for comparison.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig, load_config, ExperimentMeta, DiscoveryConfig, PlanningConfig, ExecutionConfig, ModelConfig, TracingConfig, AffordanceConfig, StateConfig, ReasoningConfig, OutputConfig
from src.runner import run_experiment


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
        experiment=ExperimentMeta(name=name, description=f"Ablation: {name}"),
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


# Define ablation configurations
ABLATION_CONFIGS = {
    # Baseline: no reasoning, JSON IR
    "baseline_json_ir": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "none",
    },

    # With state gathering
    "baseline_with_state": {
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Chain of thought reasoning
    "cot_reasoning": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Reflection reasoning
    "reflection_reasoning": {
        "reasoning_enabled": True,
        "reasoning_strategy": "reflection",
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Multi-turn reasoning
    "multi_turn_reasoning": {
        "reasoning_enabled": True,
        "reasoning_strategy": "multi_turn",
        "output_format": "json_ir",
        "state_strategy": "all",
    },

    # Python code generation (no reasoning)
    "python_code_direct": {
        "reasoning_enabled": False,
        "output_format": "python_code",
        "state_strategy": "none",
    },

    # Python code with CoT
    "python_code_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code",
        "state_strategy": "all",
    },

    # Agentic discovery (LLM-guided exploration)
    "agentic_discovery": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },

    # Agentic discovery with CoT reasoning
    "agentic_discovery_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "relevant",
    },

    # Fully agentic (both affordance and state discovery)
    "fully_agentic": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": False,
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },

    # Fully agentic with CoT reasoning
    "fully_agentic_cot": {
        "affordance_strategy": "agentic",
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "json_ir",
        "state_strategy": "agentic",
    },

    # Unconstrained Python code (custom py_trees behaviors)
    "python_unconstrained": {
        "reasoning_enabled": False,
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },

    # Unconstrained Python code with CoT
    "python_unconstrained_cot": {
        "reasoning_enabled": True,
        "reasoning_strategy": "chain_of_thought",
        "output_format": "python_code_unconstrained",
        "state_strategy": "all",
    },
}


# Test goals
TEST_GOALS = [
    "Turn on the bathroom light",
    # "Turn on all lights in the living room",
    # "Set the master bedroom AC to 22 degrees",
]


def run_ablation(
    configs_to_run: list[str] | None = None,
    goals: list[str] | None = None,
    home_id: int = 96,
    model: str = "gpt-4o",
):
    """Run ablation experiments."""

    # Setup
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = OpenAI(api_key=api_key)
    entry_point = f"http://localhost:8080/workspaces/home{home_id}#workspace"

    configs_to_run = configs_to_run or list(ABLATION_CONFIGS.keys())
    goals = goals or TEST_GOALS

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"ABLATION EXPERIMENT")
    print(f"=" * 70)
    print(f"Entry point: {entry_point}")
    print(f"Configurations: {configs_to_run}")
    print(f"Goals: {goals}")
    print(f"Output: {output_dir}")
    print(f"=" * 70)

    all_results = []

    for goal in goals:
        print(f"\n{'=' * 70}")
        print(f"GOAL: {goal}")
        print(f"{'=' * 70}")

        for config_name in configs_to_run:
            config_params = ABLATION_CONFIGS[config_name]

            print(f"\n{'-' * 50}")
            print(f"CONFIG: {config_name}")
            print(f"  Reasoning: {config_params.get('reasoning_enabled', False)} ({config_params.get('reasoning_strategy', 'none')})")
            print(f"  Output: {config_params.get('output_format', 'json_ir')}")
            print(f"  State: {config_params.get('state_strategy', 'none')}")
            print(f"{'-' * 50}")

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

                # Add metadata
                result["config_name"] = config_name
                result["goal"] = goal

                # Print summary
                print(f"\n  Result: {'SUCCESS' if result['success'] else 'FAILED'}")
                if result.get("execution"):
                    exec_result = result["execution"]
                    print(f"  Ticks: {exec_result.get('ticks', 'N/A')}")
                    print(f"  Final status: {exec_result.get('final_status', 'N/A')}")

                if result.get("planning"):
                    plan = result["planning"].get("plan", {})
                    print(f"  Reasoning steps: {len(plan.get('reasoning_trace', []))}")
                    if plan.get("explanation"):
                        print(f"  Explanation: {plan['explanation'][:100]}...")

                if result.get("error"):
                    print(f"  Error: {result['error']}")

                all_results.append(result)

                # Save individual result
                result_file = output_dir / f"{config_name}_{goal[:30].replace(' ', '_')}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "config_name": config_name,
                    "goal": goal,
                    "success": False,
                    "error": str(e),
                })

    # Save summary
    summary_file = output_dir / "summary.json"
    summary = {
        "timestamp": timestamp,
        "entry_point": entry_point,
        "goals": goals,
        "configs": configs_to_run,
        "results": [
            {
                "config": r.get("config_name"),
                "goal": r.get("goal"),
                "success": r.get("success", False),
                "error": r.get("error"),
                "ticks": r.get("execution", {}).get("ticks") if r.get("execution") else None,
            }
            for r in all_results
        ],
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for r in all_results:
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r.get('config_name', 'unknown')}: {r.get('goal', 'unknown')[:40]}")

    print(f"\nResults saved to: {output_dir}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--configs", nargs="+", help="Configs to run (default: all)")
    parser.add_argument("--goals", nargs="+", help="Goals to test")
    parser.add_argument("--home", type=int, default=96, help="Home ID")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--list", action="store_true", help="List available configs")

    args = parser.parse_args()

    if args.list:
        print("Available configurations:")
        for name, params in ABLATION_CONFIGS.items():
            print(f"  {name}:")
            for k, v in params.items():
                print(f"    {k}: {v}")
        sys.exit(0)

    run_ablation(
        configs_to_run=args.configs,
        goals=args.goals,
        home_id=args.home,
        model=args.model,
    )
