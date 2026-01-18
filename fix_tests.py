import os
import json


with open("datasets/HomeBench/converted/test_data.json") as f:
    ground_truth_tests = json.load(f)

test_groups_files = os.listdir("experiments_data/")
test_groups_files = [f for f in test_groups_files if f.endswith("_tests.json")]

for test_group_file in test_groups_files:
    with open(os.path.join("experiments_data", test_group_file)) as f:
        test_group_tests = json.load(f)

    fixed_test = []

    for test in test_group_tests:
        # Find the matching ground truth test
        matching_gt_test = next((gt_test for gt_test in ground_truth_tests if gt_test["id"] == test["id"]), None)

        if matching_gt_test:
            if test_group_file == "single_unfeasible_action_tests.json" and matching_gt_test["output"][0]["execution"] != "error_input":
                continue  # Skip this test as it doesn't match the expected unfeasible action

            fixed_test.append(matching_gt_test)
        else:
            print(f"Warning: No matching ground truth test found for test ID {test['id']}")

    # Save the fixed tests back to the file
    with open(os.path.join("experiments_data", test_group_file), 'w') as f:
        json.dump(fixed_test, f, indent=2)
