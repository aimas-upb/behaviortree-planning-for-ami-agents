"""Pre-process test data by splitting into categories and sampling entries.

Run with: uv run python data_split/preprocess.py --test_data_dir path/to/test_data.json
"""

import os
import json
import random
import argparse

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

random.seed(42)


class TestOutput(BaseModel):
    execution: str
    affordance: Optional[str] = None
    params: Optional[dict[str, str | int]] = None
    test: Optional[dict[str, str | int]] = None


class TestEntry(BaseModel):
    id: str
    input: str
    output: List[TestOutput]


class TestStatistics(BaseModel):
    devices: List[str]
    device_affordance_counts: dict[str, int] # counts of device/affordance pairs
    correct_inputs_to_entries_mapping: Optional[dict[int, int]] = None # only for multi feasible actions; maps how many entries are with 2 correct input, 3 correct inputs, etc.
    error_inputs_to_entries_mapping: Optional[dict[int, int]] = None # only for multi unfeasible actions; maps how many entries are with 0 error inputs, 1 error input, etc.


class TestType(str, Enum):
    SINGLE_FEASIBLE_ACTION = "single_feasible_action"
    SINGLE_UNFEASIBLE_ACTION = "single_unfeasible_action"
    MULTI_FEASIBLE_ACTION = "multi_feasible_action"
    MULTI_UNFEASIBLE_ACTION = "multi_unfeasible_action"


class DataService:
    """Service used to pre-process the test data."""

    def __init__(self, test_data_dir: str) -> None:
        self.test_data_dir = test_data_dir

    def _load_test_file(self) -> List[TestEntry]:
        """Load test entries from a JSON file."""
        with open(self.test_data_dir, 'r') as f:
            data = json.load(f)
        return [TestEntry(**entry) for entry in data]

    def _save_test_file(self, file_path: str, test_entries: List[TestEntry]) -> None:
        """Save test entries to a JSON file."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, 'w') as f:
            json.dump([entry.model_dump(exclude_none=True) for entry in test_entries], f, indent=2)

    def _classify_test_entry(self, entry: TestEntry) -> TestType:
        """Classify test entry based on its outputs."""
        if "one" in entry.id:
            return TestType.SINGLE_FEASIBLE_ACTION if entry.output[0].execution != "error_input" else TestType.SINGLE_UNFEASIBLE_ACTION
        else:
            for output in entry.output:
                if output.execution == "error_input":
                    return TestType.MULTI_UNFEASIBLE_ACTION
            return TestType.MULTI_FEASIBLE_ACTION

    def _split_entries_by_type(self, entries: List[TestEntry]) -> dict[TestType, List[TestEntry]]:
        """Split test entries by their classified type."""
        split_entries = {
            TestType.SINGLE_FEASIBLE_ACTION: [],
            TestType.SINGLE_UNFEASIBLE_ACTION: [],
            TestType.MULTI_FEASIBLE_ACTION: [],
            TestType.MULTI_UNFEASIBLE_ACTION: [],
        }
        for entry in entries:
            entry_type = self._classify_test_entry(entry)
            split_entries[entry_type].append(entry)
        return split_entries
    
    def _sample_entries(self, entries: List[TestEntry], sample_size: int = 100) -> List[TestEntry]:
        """Randomly sample a specified number of test entries."""
        if len(entries) <= sample_size:
            return entries
        return random.sample(entries, sample_size)
    
    def _compute_default_statistics(self, entries: List[TestEntry]) -> TestStatistics:
        """Compute default statistics for all test types."""
        device_affordance_counter: dict[str, int] = {}
        all_devices = set()
    
        for entry in entries:
            for output in entry.output:
                if output.execution == "error_input":
                    continue

                device, affordance = output.affordance.split('/')[-2:]
                all_devices.add(device)
                key = f"{device}/{affordance}"
                device_affordance_counter[key] = device_affordance_counter.get(key, 0) + 1

        return TestStatistics(
            devices=list(all_devices),
            device_affordance_counts=device_affordance_counter,
        )

    def _save_statistics(self, test_type: TestType, entries: List[TestEntry]) -> None:
        """Display statistics of the split test entries."""
        print(f"{test_type.value}: {len(entries)} entries")

        statistic = self._compute_default_statistics(entries)
        if test_type in TestType.SINGLE_FEASIBLE_ACTION:
            with open(f"experiments_data/{test_type.value}_statistics.json", 'w') as f:
                json.dump(statistic.model_dump(exclude_none=True), f, indent=2)
        elif test_type in TestType.MULTI_FEASIBLE_ACTION:
            # Compute the entries to correct inputs mapping
            correct_inputs_mapping: dict[int, int] = {}
            for entry in entries:
                correct_count = len(entry.output)
                correct_inputs_mapping[correct_count] = correct_inputs_mapping.get(correct_count, 0) + 1
                statistic.correct_inputs_to_entries_mapping = correct_inputs_mapping
                with open(f"experiments_data/{test_type.value}_statistics.json", 'w') as f:
                    json.dump(statistic.model_dump(exclude_none=True), f, indent=2)
        elif test_type in TestType.MULTI_UNFEASIBLE_ACTION:
            # Compute the entries to error inputs mapping
            error_inputs_mapping: dict[int, int] = {}
            for entry in entries:
                error_count = sum(1 for output in entry.output if output.execution == "error_input")
                error_inputs_mapping[error_count] = error_inputs_mapping.get(error_count, 0) + 1

            statistic.error_inputs_to_entries_mapping = error_inputs_mapping
            with open(f"experiments_data/{test_type.value}_statistics.json", 'w') as f:
                json.dump(statistic.model_dump(exclude_none=True), f, indent=2)

    def pre_process_data(self) -> None:
        """Select 100 random test entries from each type and save to output directory."""
        test_entries = self._load_test_file()
        entries_by_type = self._split_entries_by_type(test_entries)

        for test_type, entries in entries_by_type.items():
            sampled_entries = self._sample_entries(entries)
            output_file_path = f"experiments_data/{test_type.value}_tests.json"
            self._save_test_file(output_file_path, sampled_entries)
            self._save_statistics(test_type, sampled_entries)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Pre-process test data by splitting and sampling.")
    argparser.add_argument("--test_data_dir", type=str, required=True, help="Path to the test data JSON file.")
    args = argparser.parse_args()

    data_service = DataService(test_data_dir=args.test_data_dir)
    data_service.pre_process_data()
