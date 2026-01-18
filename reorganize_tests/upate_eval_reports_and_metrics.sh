#!/bin/bash
# Usage: From the root directory of the project, run: bash reorganize_tests/upate_eval_reports_and_metrics.sh

set -e

EXPERIMENTS_DIR="/mnt/c/Users/rvulp/OneDrive/Desktop/tests/"

# Loop through each experiment directory
for dir in "$EXPERIMENTS_DIR"*/ ; do
    echo "Processing directory: $dir"

    # Run the update_metrics.py script
    python reorganize_tests/update_metrics.py --tests_dir "$dir"

    # Run the compute_action_f1_by_group.py script
    python reorganize_tests/compute_action_f1_by_group.py --experiment_dir "$dir"
done

echo "All experiments processed."
