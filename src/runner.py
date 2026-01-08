"""
Experiment runner for behavior tree planning.

Orchestrates the complete pipeline:
1. Load configuration
2. Discovery phase (affordances + state)
3. Planning phase (reasoning + generation)
4. Execution phase
5. Tracing and results
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .config import ExperimentConfig, load_config, create_default_config
from .discovery import create_discovery_pipeline, DiscoveryResult
from .planning import create_planner, PlanningResult
from .execution import create_executor, ExecutionResult

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(
    config: ExperimentConfig,
    goal: str,
    entry_point: str,
    client: OpenAI,
) -> dict:
    """
    Run a single experiment with the given configuration.

    Args:
        config: Experiment configuration
        goal: User's goal
        entry_point: Environment entry point URI
        client: OpenAI client

    Returns:
        Dictionary with all experiment results
    """
    start_time = datetime.now()
    model = config.model.name

    results = {
        "config": config.model_dump(),
        "goal": goal,
        "entry_point": entry_point,
        "start_time": start_time.isoformat(),
        "discovery": None,
        "planning": None,
        "execution": None,
        "success": False,
        "error": None,
    }

    try:
        # Phase 1: Discovery
        logger.info("=" * 60)
        logger.info("PHASE 1: Discovery")
        logger.info("=" * 60)

        discovery_pipeline = create_discovery_pipeline(
            config=config.discovery,
            client=client,
            model=model,
        )
        discovery_result = discovery_pipeline.discover(entry_point, goal)
        results["discovery"] = discovery_result.to_dict()

        logger.info(
            f"Discovery complete: {len(discovery_result.affordances.artifacts)} artifacts, "
            f"{len(discovery_result.state.property_values)} property values"
        )

        # Phase 2: Planning
        logger.info("=" * 60)
        logger.info("PHASE 2: Planning")
        logger.info("=" * 60)

        planner = create_planner(config.planning)
        planning_result = planner.plan(
            goal=goal,
            discovery=discovery_result,
            client=client,
            model=model,
        )
        results["planning"] = planning_result.to_dict()

        if not planning_result.success:
            results["error"] = f"Planning failed: {planning_result.error}"
            logger.error(results["error"])
            return results

        logger.info(
            f"Planning complete: {planning_result.plan.format} output, "
            f"{len(planning_result.plan.reasoning_trace)} reasoning steps"
        )

        # Log plan details
        if planning_result.plan.is_json_ir:
            logger.info(f"Tree spec:\n{json.dumps(planning_result.plan.content, indent=2)}")
        else:
            logger.info(f"Generated code:\n{planning_result.plan.content}")

        # Phase 3: Execution
        logger.info("=" * 60)
        logger.info("PHASE 3: Execution")
        logger.info("=" * 60)

        executor = create_executor(
            output_format=config.planning.output.format,
            max_ticks=config.execution.max_ticks,
        )
        execution_result = executor.execute(planning_result.plan)
        results["execution"] = execution_result.to_dict()

        if execution_result.success:
            logger.info(
                f"Execution SUCCESS: {execution_result.tree_name} "
                f"completed in {execution_result.ticks} ticks"
            )
            results["success"] = True
        else:
            logger.warning(
                f"Execution FAILED: {execution_result.error or execution_result.final_status}"
            )
            results["error"] = execution_result.error

    except Exception as e:
        logger.exception("Experiment failed with exception")
        results["error"] = str(e)

    # Record end time
    end_time = datetime.now()
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = (end_time - start_time).total_seconds()

    return results


def save_results(results: dict, output_dir: str, name: str) -> str:
    """Save experiment results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Behavior Tree Planning Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  uv run python -m src.runner --config experiments/configs/baseline.yaml --goal "Turn on bathroom light"

  # Run with default config
  uv run python -m src.runner --goal "Turn on bathroom light" --home 0

  # Run with custom options
  uv run python -m src.runner \\
      --goal "Turn on all lights" \\
      --discovery-affordances agentic \\
      --discovery-state all \\
      --planning-reasoning chain_of_thought \\
      --planning-output json_ir

  # Save results
  uv run python -m src.runner --config baseline.yaml --goal "..." --output results/
        """
    )

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Goal and environment
    parser.add_argument("--goal", type=str, help="Goal to accomplish")
    parser.add_argument("--home", type=int, default=0, help="Home ID (0-99)")
    parser.add_argument("--entry", type=str, help="Entry point URI (overrides --home)")

    # Model settings
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--base-url", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")

    # Discovery overrides
    parser.add_argument(
        "--discovery-affordances",
        choices=["exhaustive", "agentic", "relevant"],
        help="Affordance discovery strategy"
    )
    parser.add_argument(
        "--discovery-state",
        choices=["all", "relevant", "none"],
        help="State gathering strategy"
    )

    # Planning overrides
    parser.add_argument(
        "--planning-reasoning",
        choices=["none", "chain_of_thought", "multi_turn", "reflection"],
        help="Reasoning strategy"
    )
    parser.add_argument(
        "--planning-output",
        choices=["json_ir", "python_code", "python_code_unconstrained"],
        help="Output format"
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=["baseline", "detailed", "few_shot", "icl"],
        help="Prompt strategy"
    )

    # Output
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load or create config
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_default_config("cli_run")
        logger.info("Using default configuration")

    # Apply CLI overrides
    if args.model:
        config.model.name = args.model
    if args.base_url:
        config.model.base_url = args.base_url
    if args.api_key:
        config.model.api_key = args.api_key

    if args.discovery_affordances:
        config.discovery.affordances.strategy = args.discovery_affordances
    if args.discovery_state:
        config.discovery.state.strategy = args.discovery_state

    if args.planning_reasoning:
        if args.planning_reasoning == "none":
            config.planning.reasoning.enabled = False
        else:
            config.planning.reasoning.enabled = True
            config.planning.reasoning.strategy = args.planning_reasoning
    if args.planning_output:
        config.planning.output.format = args.planning_output
    if args.prompt_strategy:
        config.planning.prompt_strategy = args.prompt_strategy

    # Determine entry point
    if args.entry:
        entry_point = args.entry
    else:
        entry_point = f"http://localhost:8080/workspaces/home{args.home}#workspace"

    # Get goal
    if not args.goal:
        print("Error: --goal is required")
        parser.print_help()
        sys.exit(1)
    goal = args.goal

    # Setup API client
    api_key = args.api_key or config.model.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or config.model.base_url

    if not api_key and not base_url:
        print("Error: Set OPENAI_API_KEY in .env or provide --api-key")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=base_url,
    )

    # Print config summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Goal: {goal}")
    logger.info(f"Entry point: {entry_point}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Discovery - Affordances: {config.discovery.affordances.strategy}")
    logger.info(f"Discovery - State: {config.discovery.state.strategy}")
    logger.info(f"Planning - Reasoning: {config.planning.reasoning.strategy if config.planning.reasoning.enabled else 'none'}")
    logger.info(f"Planning - Output: {config.planning.output.format}")
    logger.info(f"Planning - Prompt: {config.planning.prompt_strategy}")
    logger.info("=" * 60)

    # Run experiment
    results = run_experiment(
        config=config,
        goal=goal,
        entry_point=entry_point,
        client=client,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Success: {results['success']}")
    if results.get("error"):
        logger.info(f"Error: {results['error']}")
    logger.info(f"Duration: {results.get('duration_seconds', 0):.2f}s")

    # Save results
    if args.output:
        filepath = save_results(
            results,
            args.output,
            config.experiment.name,
        )
        logger.info(f"Results saved to: {filepath}")

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
