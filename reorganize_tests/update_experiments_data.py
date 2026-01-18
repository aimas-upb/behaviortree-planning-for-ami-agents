"""Script to update experiment test data by aligning with ground truth tests and reorganizing test groups.
Usage: python reorganize_tests/update_experiments_data.py
"""

import os
import json

with open("datasets/HomeBench/converted/test_data.json") as f:
    ground_truth_tests = json.load(f)

test_groups_files = os.listdir("old_experiments_data/")
test_groups_files = [f for f in test_groups_files if f.endswith("_tests.json")]

new_feasible_retry_tests = []
total_number_of_tests_per_group = {}

for test_group_file in test_groups_files:
    with open(os.path.join("old_experiments_data", test_group_file)) as f:
        test_group_tests = json.load(f)

    fixed_test = []

    for test in test_group_tests:
        # Find the matching ground truth test
        matching_gt_test = next((gt_test for gt_test in ground_truth_tests if gt_test["id"] == test["id"]), None)

        if matching_gt_test:
            if test_group_file == "single_unfeasible_action_tests.json" and matching_gt_test["output"][0]["execution"] != "error_input":
                new_feasible_retry_tests.append(matching_gt_test)
                continue  # Skip this test as it doesn't match the expected unfeasible action

            fixed_test.append(matching_gt_test)
        else:
            print(f"Warning: No matching ground truth test found for test ID {test['id']}")

    # Save the fixed tests back to the file
    with open(os.path.join("experiments_data", test_group_file), 'w') as f:
        json.dump(fixed_test, f, indent=2)

    total_number_of_tests_per_group[test_group_file] = len(fixed_test)

if new_feasible_retry_tests:
    # Merge with existing single_feasible_action_tests.json if it exists
    feasible_file_path = os.path.join("experiments_data", "single_feasible_action_tests.json")
    print(f"Total new feasible retry tests: {len(new_feasible_retry_tests)}")

    with open(feasible_file_path, 'r') as f:
        existing_feasible_tests = json.load(f)
    
    existing_feasible_tests.extend(new_feasible_retry_tests)

    with open(feasible_file_path, 'w') as f:
        json.dump(existing_feasible_tests, f, indent=2)

    total_number_of_tests_per_group["single_feasible_action_tests.json"] += len(new_feasible_retry_tests)

print("Total number of tests per group:")
for group, count in total_number_of_tests_per_group.items():
    print(f"{group}: {count}")
