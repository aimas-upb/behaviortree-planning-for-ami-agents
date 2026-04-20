from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.export_modify_codegen_sft_dataset import (
    build_conversational_record,
    build_standard_record,
    export_dataset,
)


def make_example(example_id: str) -> dict:
    return {
        "id": example_id,
        "home_id": 7,
        "source_file": "train_data_part1",
        "source_row_type": "normal",
        "official_split": "train",
        "home_disjoint_split": "train",
        "modify_intent_count": 1,
        "label_source": "deterministic",
        "exact_signature": "abc123",
        "runtime_modify_goal": (
            "**Intent 1**\n"
            "text_intent: increase the brightness of the light in the foyer by 5.\n"
            "action.verb: modify"
        ),
        "runtime_context": (
            "# Available Devices and Capabilities\n\n"
            "## foyer\n\n"
            "- setBrightness\n"
        ),
        "target_code": "tree = seq_1\n",
    }


def test_build_conversational_record_compact() -> None:
    record = build_conversational_record(make_example("home7_one_1"), "compact")

    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert "Return only Python code" in record["messages"][0]["content"]
    assert "Modify goals:" in record["messages"][1]["content"]
    assert (
        record["messages"][2]["content"] == "tree = seq_1\n"
    )
    assert record["target_format"] == "python_code"


def test_build_standard_record_contains_assistant_prefix() -> None:
    record = build_standard_record(make_example("home7_one_2"), "compact")

    assert "<SYSTEM>" in record["prompt"]
    assert "<USER>" in record["prompt"]
    assert record["prompt"].endswith("<ASSISTANT>\n")
    assert record["completion"] == "tree = seq_1\n"


def test_export_dataset_maps_valid_to_validation(tmp_path) -> None:
    benchmark_dir = tmp_path / "modify_codegen"
    split_dir = benchmark_dir / "home_disjoint"
    split_dir.mkdir(parents=True)

    train_example = make_example("home7_one_train")
    valid_example = {**make_example("home7_one_valid"), "home_disjoint_split": "valid"}
    test_example = {**make_example("home7_one_test"), "home_disjoint_split": "test"}

    for name, example in (
        ("train", train_example),
        ("valid", valid_example),
        ("test", test_example),
    ):
        (split_dir / f"{name}.jsonl").write_text(json.dumps(example) + "\n")

    args = argparse.Namespace(
        benchmark_dir=str(benchmark_dir),
        split_family="home_disjoint",
        dataset_format="conversational",
        prompt_style="compact",
        output_root=str(tmp_path / "exports"),
        limit=None,
    )

    manifests = export_dataset(args)

    output_dir = (
        tmp_path / "exports" / "conversational" / "compact"
    )
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert manifests["conversational"]["counts"]["validation"] == 1
