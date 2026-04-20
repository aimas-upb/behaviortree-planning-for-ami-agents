#!/usr/bin/env python3
"""
Export modify_codegen benchmark splits into SFTTrainer-ready JSONL datasets.

This script reads the existing benchmark rows and renders them into one of the
formats accepted by Hugging Face SFT fine-tuning pipelines:

- conversational: ``{"messages": [...]}``
- standard: ``{"prompt": "...", "completion": "..."}``

The default export targets the ``home_disjoint`` split family in conversational
format, which is a good fit for instruct-tuned coder models such as
Qwen2.5-Coder-3B-Instruct.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import resolve_repo_path

INPUT_SPLITS = ("train", "valid", "test")
OUTPUT_SPLIT_NAMES = {"train": "train", "valid": "validation", "test": "test"}

COMPACT_SYSTEM_PROMPT = """You are a behavior tree planning assistant.
Generate Python code that constructs a py_trees behavior tree for modify-only smart-home intents.

Rules:
- Return only Python code. Do not add markdown fences or explanation.
- Do not use any import statements.
- Use only the already-available names: py_trees, ActionAffordanceNode, PropertyAffordanceNode, PropertyConditionNode, ComparisonPropertyConditionNode, ComparisonOperator.
- For each modify intent, create a Sequence that:
  1. Reads the current property value with PropertyAffordanceNode.
  2. Computes the new value in an inline py_trees.behaviour.Behaviour subclass.
  3. Applies the new value with ActionAffordanceNode using parameter_keys.
- Use exact property and action URLs from the provided capability model.
- Respect the numeric bounds from the provided schemas and clamp when needed.
- Use flat blackboard keys.
- Define a top-level variable named `tree`.
- If there are multiple modify intents, combine their sequences in a Parallel with SuccessOnOne policy."""

DIRECT_RESPONSE_SUFFIX = (
    "\n\nReturn only the Python code with no markdown fences or explanation."
)


def load_modify_only_system_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "prompts"
        / "code"
        / "detailed_structured_modify_only.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_modify_codegen_prompt_template", prompt_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt module from {prompt_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.MODIFY_ONLY_SYSTEM)


MODIFY_ONLY_SYSTEM_PROMPT = load_modify_only_system_prompt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export modify_codegen benchmark rows into SFT JSONL."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/homebench/benchmarks/modify_codegen",
        help="Path to the modify_codegen benchmark root directory.",
    )
    parser.add_argument(
        "--split-family",
        choices=("home_disjoint", "official"),
        default="home_disjoint",
        help="Which benchmark split family to export.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("conversational", "standard", "both"),
        default="conversational",
        help="Output schema to write for SFT.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("compact", "planner_exact"),
        default="compact",
        help="Prompt rendering style to use.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root directory for exported datasets. Defaults to "
            "<benchmark-dir>/sft/<split-family>."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on records per split, useful for smoke tests.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_output_root(
    benchmark_dir: Path, split_family: str, explicit_root: str | None
) -> Path:
    if explicit_root:
        return resolve_repo_path(explicit_root)
    return benchmark_dir / "sft" / split_family


def build_system_prompt(example: dict[str, Any], prompt_style: str) -> str:
    if prompt_style == "compact":
        return COMPACT_SYSTEM_PROMPT
    if prompt_style == "planner_exact":
        return (
            MODIFY_ONLY_SYSTEM_PROMPT.format(
                capability_model=example["runtime_context"].strip()
            ).strip()
            + DIRECT_RESPONSE_SUFFIX
        )
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def build_user_prompt(example: dict[str, Any], prompt_style: str) -> str:
    goal = str(example["runtime_modify_goal"]).strip()
    context = str(example["runtime_context"]).strip()

    if prompt_style == "planner_exact":
        return goal

    return "\n\n".join(
        [
            "Modify goals:",
            goal,
            "Available devices, capabilities, and current state:",
            context,
        ]
    )


def base_record_metadata(
    example: dict[str, Any], prompt_style: str
) -> dict[str, Any]:
    return {
        "id": example["id"],
        "home_id": example["home_id"],
        "source_file": example.get("source_file"),
        "source_row_type": example.get("source_row_type"),
        "official_split": example.get("official_split"),
        "home_disjoint_split": example.get("home_disjoint_split"),
        "modify_intent_count": example.get("modify_intent_count"),
        "label_source": example.get("label_source"),
        "exact_signature": example.get("exact_signature"),
        "target_format": "python_code",
        "prompt_style": prompt_style,
    }


def build_conversational_record(
    example: dict[str, Any], prompt_style: str
) -> dict[str, Any]:
    system_prompt = build_system_prompt(example, prompt_style)
    user_prompt = build_user_prompt(example, prompt_style)
    record = base_record_metadata(example, prompt_style)
    record["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["target_code"]},
    ]
    return record


def build_standard_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<SYSTEM>\n"
        f"{system_prompt}\n"
        "</SYSTEM>\n\n"
        "<USER>\n"
        f"{user_prompt}\n"
        "</USER>\n\n"
        "<ASSISTANT>\n"
    )


def build_standard_record(
    example: dict[str, Any], prompt_style: str
) -> dict[str, Any]:
    system_prompt = build_system_prompt(example, prompt_style)
    user_prompt = build_user_prompt(example, prompt_style)
    record = base_record_metadata(example, prompt_style)
    record["prompt"] = build_standard_prompt(system_prompt, user_prompt)
    record["completion"] = example["target_code"]
    return record


def build_record(
    example: dict[str, Any], dataset_format: str, prompt_style: str
) -> dict[str, Any]:
    if dataset_format == "conversational":
        return build_conversational_record(example, prompt_style)
    if dataset_format == "standard":
        return build_standard_record(example, prompt_style)
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def estimate_lengths(
    record: dict[str, Any], dataset_format: str
) -> tuple[int, int]:
    if dataset_format == "conversational":
        prompt_chars = sum(
            len(message["content"])
            for message in record["messages"]
            if message["role"] != "assistant"
        )
        completion_chars = sum(
            len(message["content"])
            for message in record["messages"]
            if message["role"] == "assistant"
        )
        return prompt_chars, completion_chars

    return len(record["prompt"]), len(record["completion"])


def export_one_format(
    benchmark_dir: Path,
    split_family: str,
    dataset_format: str,
    prompt_style: str,
    output_root: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    input_dir = benchmark_dir / split_family
    if not input_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {input_dir}")

    format_dir = output_root / dataset_format / prompt_style
    format_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    prompt_chars: Counter[str] = Counter()
    completion_chars: Counter[str] = Counter()
    max_prompt_chars: Counter[str] = Counter()
    max_completion_chars: Counter[str] = Counter()

    for input_split in INPUT_SPLITS:
        output_split = OUTPUT_SPLIT_NAMES[input_split]
        input_path = input_dir / f"{input_split}.jsonl"
        output_path = format_dir / f"{output_split}.jsonl"

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split file: {input_path}")

        with output_path.open("w") as handle:
            for row_index, example in enumerate(iter_jsonl(input_path)):
                if limit is not None and row_index >= limit:
                    break

                record = build_record(
                    example=example,
                    dataset_format=dataset_format,
                    prompt_style=prompt_style,
                )
                record["split"] = output_split
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

                counts[output_split] += 1
                prompt_len, completion_len = estimate_lengths(
                    record, dataset_format
                )
                prompt_chars[output_split] += prompt_len
                completion_chars[output_split] += completion_len
                max_prompt_chars[output_split] = max(
                    max_prompt_chars[output_split], prompt_len
                )
                max_completion_chars[output_split] = max(
                    max_completion_chars[output_split], completion_len
                )

    split_stats: dict[str, Any] = {}
    for split_name in OUTPUT_SPLIT_NAMES.values():
        count = counts[split_name]
        split_stats[split_name] = {
            "count": count,
            "avg_prompt_chars": (
                round(prompt_chars[split_name] / count, 2) if count else 0.0
            ),
            "avg_completion_chars": (
                round(completion_chars[split_name] / count, 2)
                if count
                else 0.0
            ),
            "max_prompt_chars": max_prompt_chars[split_name],
            "max_completion_chars": max_completion_chars[split_name],
        }

    manifest = {
        "created_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "benchmark_dir": str(benchmark_dir),
        "split_family": split_family,
        "dataset_format": dataset_format,
        "prompt_style": prompt_style,
        "output_dir": str(format_dir),
        "split_name_map": OUTPUT_SPLIT_NAMES,
        "counts": {
            "total": sum(counts.values()),
            **{split: counts[split] for split in OUTPUT_SPLIT_NAMES.values()},
        },
        "split_stats": split_stats,
    }
    (format_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_dir = resolve_repo_path(args.benchmark_dir)
    output_root = resolve_output_root(
        benchmark_dir=benchmark_dir,
        split_family=args.split_family,
        explicit_root=args.output_root,
    )

    dataset_formats = (
        ("conversational", "standard")
        if args.dataset_format == "both"
        else (args.dataset_format,)
    )
    manifests: dict[str, Any] = {}
    for dataset_format in dataset_formats:
        manifests[dataset_format] = export_one_format(
            benchmark_dir=benchmark_dir,
            split_family=args.split_family,
            dataset_format=dataset_format,
            prompt_style=args.prompt_style,
            output_root=output_root,
            limit=args.limit,
        )
    return manifests


def main() -> None:
    args = parse_args()
    export_dataset(args)


if __name__ == "__main__":
    main()
