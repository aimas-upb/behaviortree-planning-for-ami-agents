#!/bin/bash

set -e

EXPERIMENT_CONFIGS='[
    {"strategy": "agentic_relevant_json_ir_detailed", "discovery_state": "agentic", "discovery_affordances": "relevant", "planning_output": "json_ir", "prompt_strategy": "detailed", "planning_reasoning": "chain_of_thought"},
    {"strategy": "agentic_relevant_json_ir_icl", "discovery_state": "agentic", "discovery_affordances": "relevant", "planning_output": "json_ir", "prompt_strategy": "icl", "planning_reasoning": "chain_of_thought"},
    {"strategy": "agentic_relevant_python_code_detailed", "discovery_state": "agentic", "discovery_affordances": "relevant", "planning_output": "python_code", "prompt_strategy": "detailed", "planning_reasoning": "chain_of_thought"},
    {"strategy": "agentic_relevant_python_code_icl", "discovery_state": "agentic", "discovery_affordances": "relevant", "planning_output": "python_code", "prompt_strategy": "icl", "planning_reasoning": "chain_of_thought"}
]'

for config in $(echo "${EXPERIMENT_CONFIGS}" | jq -c '.[]'); do
    STRATEGY=$(echo $config | jq -r '.strategy')
    DISCOVERY_STATE=$(echo $config | jq -r '.discovery_state')
    DISCOVERY_AFFORDANCES=$(echo $config | jq -r '.discovery_affordances')
    PLANNING_OUTPUT=$(echo $config | jq -r '.planning_output')
    PROMPT_STRATEGY=$(echo $config | jq -r '.prompt_strategy')
    PLANNING_REASONING=$(echo $config | jq -r '.planning_reasoning')

    # Start simulator in the background
    uv run python homebench/smart_home_simulator.py --data-dir datasets/HomeBench/hmas_format/home_description &
    SIMULATOR_PID=$!

    # Wait for the simulator to start
    sleep 20

    # Run experiment
    uv run python run_homebench.py \
        --discovery-state ${DISCOVERY_STATE} \
        --discovery-affordances ${DISCOVERY_AFFORDANCES} \
        --planning-reasoning ${PLANNING_REASONING} \
        --prompt-strategy ${PROMPT_STRATEGY} \
        --planning-output ${PLANNING_OUTPUT} \
        --output experiments/tests/${STRATEGY} \
        --limit 20 \
        --model gpt-4o \
        --generate-report \
        --generate-traces || true

    # Kill the simulator by PID
    kill $SIMULATOR_PID 2>/dev/null || true
    wait $SIMULATOR_PID 2>/dev/null || true
done
