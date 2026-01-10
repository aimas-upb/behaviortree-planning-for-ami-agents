#!/bin/bash

set -e

EXPERIMENT_CONFIGS='[
    {"strategy": "fully_agentic_detailed", "discovery": "agentic", "planning_output": "json_ir", "prompt_strategy": "detailed"},
    {"strategy": "fully_agentic_few_shot", "discovery": "agentic", "planning_output": "json_ir", "prompt_strategy": "few_shot"},
    {"strategy": "fully_agentic_icl", "discovery": "agentic", "planning_output": "json_ir", "prompt_strategy": "icl"},
    {"strategy": "fully_agentic_python_code_detailed", "discovery": "agentic", "planning_output": "python_code", "prompt_strategy": "detailed"},
    {"strategy": "fully_agentic_python_code_icl", "discovery": "agentic", "planning_output": "python_code", "prompt_strategy": "icl"},
    {"strategy": "fully_agentic_python_code_few_shot", "discovery": "agentic", "planning_output": "python_code", "prompt_strategy": "few_shot"}
]'

for config in $(echo "${EXPERIMENT_CONFIGS}" | jq -c '.[]'); do
    STRATEGY=$(echo $config | jq -r '.strategy')
    DISCOVERY=$(echo $config | jq -r '.discovery')
    PLANNING_OUTPUT=$(echo $config | jq -r '.planning_output')
    PROMPT_STRATEGY=$(echo $config | jq -r '.prompt_strategy')

    # Start simulator in the background
    uv run python homebench/smart_home_simulator.py --data-dir datasets/HomeBench/hmas_format/home_description &
    SIMULATOR_PID=$!

    # Wait for the simulator to start
    sleep 20

    # Run experiment
    uv run python run_homebench.py \
        --discovery-state ${DISCOVERY} \
        --discovery-affordances ${DISCOVERY} \
        --prompt-strategy ${PROMPT_STRATEGY} \
        --planning-output ${PLANNING_OUTPUT} \
        --output experiments/tests/${STRATEGY} \
        --limit 15 \
        --model gpt-4.1 \
        --generate-report \
        --generate-traces || true

    # Kill the simulator by PID
    kill $SIMULATOR_PID 2>/dev/null || true
    wait $SIMULATOR_PID 2>/dev/null || true
done
