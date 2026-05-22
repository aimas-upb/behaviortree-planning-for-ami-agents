#!/usr/bin/env python3
"""
Fine-tune a causal LM on conversational SFT data with TRL + LoRA.

This script is intended for the HomeBench modify_codegen dataset exported as
JSONL records containing a `messages` field. The training configuration lives in
YAML so FEP cluster runs stay reproducible.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import resolve_repo_path
from src.experience.qwen_local_codegen import _load_codegen_tokenizer

SUPPORTED_PACKING_ATTENTION_IMPLEMENTATIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRL SFT fine-tuning with a LoRA adapter."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file relative to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for training.output_dir from the config file.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional override for training.run_name from the config file.",
    )
    parser.add_argument(
        "--logging-dir",
        help="Optional override for training.logging_dir from the config file.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--report-to",
        help="Comma-separated reporting backends, for example tensorboard,wandb.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional cap on training rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        help="Optional cap on validation rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        help="Optional cap on test rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip training and only run validation/test evaluation against "
            "the adapter previously saved at <output_dir>/best_adapter. Use "
            "this to recover metrics from a run whose post-training "
            "evaluation crashed."
        ),
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return config


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid `{name}` section in config.")
    return value


def normalize_report_to(value: Any) -> list[str] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or "none"
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or "none"
    raise ValueError(
        "`report_to` must be either a string or a list of strings."
    )


def resolve_path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_local_files_only(runtime_config: dict[str, Any]) -> bool:
    if "local_files_only" in runtime_config:
        return coerce_bool(runtime_config.get("local_files_only"))
    return coerce_bool(os.environ.get("HF_HUB_OFFLINE"))


def resolve_hf_token() -> str | None:
    return normalize_optional_string(os.environ.get("HF_TOKEN"))


def resolve_attn_implementation(model_config: dict[str, Any]) -> str | None:
    return normalize_optional_string(model_config.get("attn_implementation"))


def resolve_effective_packing(
    training_config: dict[str, Any], attn_implementation: str | None
) -> tuple[bool, str | None]:
    packing_requested = coerce_bool(training_config.get("packing", True))
    if not packing_requested:
        return False, None
    if attn_implementation in SUPPORTED_PACKING_ATTENTION_IMPLEMENTATIONS:
        return True, None
    return (
        False,
        "Disabling packing because `attn_implementation` is not set to a "
        "flash-attention backend known to safely support packed samples.",
    )


def resolve_effective_padding_free(
    training_config: dict[str, Any], attn_implementation: str | None
) -> tuple[bool, str | None]:
    padding_free_requested = coerce_bool(
        training_config.get("padding_free", False)
    )
    if not padding_free_requested:
        return False, None
    if attn_implementation in SUPPORTED_PACKING_ATTENTION_IMPLEMENTATIONS:
        return True, None
    return (
        False,
        "Disabling padding-free mode because `attn_implementation` is not set "
        "to a supported flash-attention backend.",
    )


def apply_chat_template_config(
    tokenizer: Any,
    model_config: dict[str, Any],
    pretrained_load_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Optionally copy a chat template onto ``tokenizer``.

    Supports two YAML knobs under ``model``:

    - ``chat_template``: an inline jinja string assigned verbatim.
    - ``chat_template_source``: another HF model name; its tokenizer's
      ``chat_template`` is loaded (honouring ``local_files_only`` / ``token``
      from ``pretrained_load_kwargs``) and copied across.

    The inline ``chat_template`` takes precedence when both are set.
    The function is a no-op when neither is set, leaving any preexisting
    template on ``tokenizer`` untouched.
    """

    info: dict[str, Any] = {"applied": False}

    inline_template = model_config.get("chat_template")
    if isinstance(inline_template, str) and inline_template.strip():
        tokenizer.chat_template = inline_template
        info.update({"applied": True, "source": "inline"})
        return info

    source = normalize_optional_string(model_config.get("chat_template_source"))
    if not source:
        return info

    source_tokenizer = AutoTokenizer.from_pretrained(
        source,
        use_fast=True,
        **pretrained_load_kwargs,
    )
    source_template = getattr(source_tokenizer, "chat_template", None)
    if not source_template:
        raise ValueError(
            f"`model.chat_template_source` ({source!r}) does not expose a "
            "tokenizer.chat_template; pick a tokenizer that ships one."
        )
    tokenizer.chat_template = source_template
    info.update({"applied": True, "source": source})
    return info


def apply_special_token_config(model: Any, tokenizer: Any) -> dict[str, Any]:
    updated_tokens: dict[str, Any] = {}
    config_targets = [
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
    ]
    for token_attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if not hasattr(tokenizer, token_attr):
            continue
        tokenizer_value = getattr(tokenizer, token_attr)
        for config_target in config_targets:
            if config_target is None or not hasattr(config_target, token_attr):
                continue
            if getattr(config_target, token_attr) != tokenizer_value:
                setattr(config_target, token_attr, tokenizer_value)
                updated_tokens[token_attr] = tokenizer_value
    return updated_tokens


def estimate_total_training_steps(trainer: SFTTrainer) -> int:
    configured_max_steps = int(getattr(trainer.args, "max_steps", -1))
    if configured_max_steps > 0:
        return configured_max_steps

    train_dataset = getattr(trainer, "train_dataset", None)
    if train_dataset is None:
        return 0

    try:
        train_dataset_size = len(train_dataset)
    except TypeError:
        return 0

    if train_dataset_size <= 0:
        return 0

    world_size = max(1, int(getattr(trainer.args, "world_size", 1)))
    per_device_batch_size = max(
        1, int(trainer.args.per_device_train_batch_size)
    )
    gradient_accumulation_steps = max(
        1, int(trainer.args.gradient_accumulation_steps)
    )

    micro_batches_per_epoch = math.ceil(
        train_dataset_size / (per_device_batch_size * world_size)
    )
    update_steps_per_epoch = max(
        1, math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    )
    return max(
        1,
        math.ceil(
            float(trainer.args.num_train_epochs) * update_steps_per_epoch
        ),
    )


def resolve_effective_warmup_steps(
    training_config: dict[str, Any], trainer: SFTTrainer
) -> int:
    configured_warmup_steps = training_config.get("warmup_steps")
    if configured_warmup_steps is not None:
        return max(0, int(configured_warmup_steps))

    warmup_ratio = float(training_config.get("warmup_ratio", 0.0))
    if warmup_ratio <= 0.0:
        return 0

    total_training_steps = estimate_total_training_steps(trainer)
    if total_training_steps <= 0:
        return 0
    return max(1, math.ceil(total_training_steps * warmup_ratio))


def remap_output_subdir(
    configured_output_dir: Path,
    configured_subdir: Path,
    overridden_output_dir: Path,
) -> Path | None:
    try:
        relative_subdir = configured_subdir.relative_to(configured_output_dir)
    except ValueError:
        return None
    return overridden_output_dir / relative_subdir


def resolve_logging_dir(
    training_config: dict[str, Any],
    output_dir: Path,
    output_dir_overridden: bool,
    logging_dir_override: str | None,
) -> Path:
    if logging_dir_override:
        return resolve_path(logging_dir_override)

    configured_logging_dir = training_config.get("logging_dir")
    if not configured_logging_dir:
        return output_dir / "logs"

    resolved_logging_dir = resolve_path(str(configured_logging_dir))
    if not output_dir_overridden:
        return resolved_logging_dir

    configured_output_dir = resolve_path(str(training_config["output_dir"]))
    remapped_logging_dir = remap_output_subdir(
        configured_output_dir=configured_output_dir,
        configured_subdir=resolved_logging_dir,
        overridden_output_dir=output_dir,
    )
    if remapped_logging_dir is not None:
        return remapped_logging_dir
    return output_dir / resolved_logging_dir.name


def keep_messages_only(dataset: Dataset) -> Dataset:
    if "messages" not in dataset.column_names:
        raise ValueError("Expected a `messages` column in the SFT dataset.")
    extra_columns = [
        column for column in dataset.column_names if column != "messages"
    ]
    if extra_columns:
        return dataset.remove_columns(extra_columns)
    return dataset


def maybe_select(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    limit = min(max_samples, len(dataset))
    return dataset.select(range(limit))


def tokenize_sft_example_for_eval(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize a single SFT conversation for evaluation.

    This mirrors what TRL's :meth:`SFTTrainer._prepare_dataset` produces for an
    eval split (``input_ids`` / ``attention_mask`` / ``labels`` with ``labels``
    equal to ``input_ids`` for full-conversation language modelling), but lives
    at module level so :meth:`Dataset.map` workers can pickle the closure
    without accidentally dragging in the trainer (and its already on-GPU
    model). Calling ``trainer._prepare_dataset`` after the trainer has been
    constructed otherwise hits ``Cannot re-initialize CUDA in forked
    subprocess`` because HF Trainer's ``__init__`` already moves the model to
    GPU via ``_move_model_to_device``.
    """

    rendered = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(
        rendered,
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),
    }


def tokenize_eval_dataset(
    dataset: Dataset,
    *,
    tokenizer: Any,
    max_length: int,
    num_proc: int,
    desc: str,
) -> Dataset:
    if "input_ids" in dataset.column_names:
        return dataset
    return dataset.map(
        tokenize_sft_example_for_eval,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=list(dataset.column_names),
        num_proc=max(1, int(num_proc)),
        desc=desc,
    )


def supports_argument(target: Any, name: str) -> bool:
    try:
        return name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def safe_perplexity(loss_value: float) -> float | None:
    try:
        return math.exp(float(loss_value))
    except OverflowError:
        return None


def add_perplexity(
    metrics: dict[str, Any], loss_key: str, perplexity_key: str
) -> dict[str, Any]:
    if loss_key in metrics and metrics[loss_key] is not None:
        metrics[perplexity_key] = safe_perplexity(metrics[loss_key])
    return metrics


def get_parameter_stats(model: Any) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count

    ratio = 0.0
    if total_parameters:
        ratio = (trainable_parameters / total_parameters) * 100.0

    return {
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_pct": round(ratio, 6),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)

    model_config = require_section(config, "model")
    data_config = require_section(config, "data")
    training_config = require_section(config, "training")
    lora_config = require_section(config, "lora")
    runtime_config = config.get("runtime", {})
    if not isinstance(runtime_config, dict):
        raise ValueError("`runtime` must be a mapping when provided.")

    model_name_or_path = str(model_config["name_or_path"])
    train_file = resolve_path(str(data_config["train_file"]))
    validation_file = resolve_path(str(data_config["validation_file"]))
    test_file = None
    if data_config.get("test_file"):
        test_file = resolve_path(str(data_config["test_file"]))

    output_dir_overridden = args.output_dir is not None
    output_dir = resolve_path(
        str(args.output_dir or training_config["output_dir"])
    )
    logging_dir = resolve_logging_dir(
        training_config=training_config,
        output_dir=output_dir,
        output_dir_overridden=output_dir_overridden,
        logging_dir_override=args.logging_dir,
    )
    run_name = str(
        args.run_name or training_config.get("run_name") or output_dir.name
    )
    report_to = normalize_report_to(args.report_to)
    if report_to is None:
        report_to = normalize_report_to(
            training_config.get("report_to", ["tensorboard"])
        )
    resume_from_checkpoint = args.resume_from_checkpoint or runtime_config.get(
        "resume_from_checkpoint"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    best_adapter_dir = output_dir / "best_adapter"
    if args.eval_only and not best_adapter_dir.exists():
        raise FileNotFoundError(
            f"--eval-only requested but no adapter directory was found at "
            f"{best_adapter_dir}. Run training first or drop --eval-only."
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training job.")

    set_seed(int(training_config.get("seed", 42)))

    data_files: dict[str, str] = {
        "train": str(train_file),
        "validation": str(validation_file),
    }
    if test_file is not None:
        data_files["test"] = str(test_file)

    dataset = load_dataset("json", data_files=data_files)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict from load_dataset(...).")

    dataset["train"] = maybe_select(
        keep_messages_only(dataset["train"]), args.max_train_samples
    )
    dataset["validation"] = maybe_select(
        keep_messages_only(dataset["validation"]), args.max_eval_samples
    )
    if "test" in dataset:
        dataset["test"] = maybe_select(
            keep_messages_only(dataset["test"]), args.max_test_samples
        )

    attn_implementation = resolve_attn_implementation(model_config)
    local_files_only = resolve_local_files_only(runtime_config)
    hf_token = resolve_hf_token()

    pretrained_load_kwargs: dict[str, Any] = {}
    if local_files_only:
        pretrained_load_kwargs["local_files_only"] = True
    if hf_token:
        pretrained_load_kwargs["token"] = hf_token

    tokenizer = _load_codegen_tokenizer(
        AutoTokenizer,
        PreTrainedTokenizerFast,
        model_name_or_path,
        local_files_only,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    chat_template_info = apply_chat_template_config(
        tokenizer=tokenizer,
        model_config=model_config,
        pretrained_load_kwargs=pretrained_load_kwargs,
    )
    if chat_template_info.get("applied"):
        print(
            f"[config] Applied chat_template from {chat_template_info['source']!r}"
            f" onto tokenizer for {model_name_or_path}."
        )

    use_bf16 = (
        bool(training_config.get("bf16", True))
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = False
    if not use_bf16:
        use_fp16 = bool(training_config.get("fp16", True))

    if bool(training_config.get("tf32", True)) and hasattr(
        torch, "set_float32_matmul_precision"
    ):
        torch.set_float32_matmul_precision("high")

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_load_kwargs: dict[str, Any] = {
        **pretrained_load_kwargs,
        "low_cpu_mem_usage": True,
        "dtype": torch_dtype,
    }
    if attn_implementation:
        model_load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_load_kwargs,
    )
    aligned_special_tokens = apply_special_token_config(model, tokenizer)
    if bool(training_config.get("gradient_checkpointing", True)):
        model.config.use_cache = False

    target_modules = lora_config.get("target_modules", "all-linear")
    if isinstance(target_modules, list):
        target_modules = [str(item) for item in target_modules]
    elif target_modules is not None:
        target_modules = str(target_modules)

    modules_to_save = lora_config.get("modules_to_save") or None
    if isinstance(modules_to_save, list):
        modules_to_save = [str(item) for item in modules_to_save]

    if args.eval_only:
        print(
            f"[eval-only] Attaching saved LoRA adapter from {best_adapter_dir}."
        )
        model = PeftModel.from_pretrained(
            model,
            str(best_adapter_dir),
            is_trainable=False,
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_config.get("r", 64)),
            lora_alpha=int(lora_config.get("alpha", 128)),
            lora_dropout=float(lora_config.get("dropout", 0.05)),
            target_modules=target_modules,
            bias=str(lora_config.get("bias", "none")),
            use_rslora=bool(lora_config.get("use_rslora", True)),
            modules_to_save=modules_to_save,
        )

    packing, packing_note = resolve_effective_packing(
        training_config, attn_implementation
    )
    if packing_note:
        print(f"[config] {packing_note}")

    padding_free, padding_free_note = resolve_effective_padding_free(
        training_config, attn_implementation
    )
    if padding_free_note:
        print(f"[config] {padding_free_note}")

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "num_train_epochs": float(training_config.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(
            training_config.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            training_config.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training_config.get("gradient_accumulation_steps", 1)
        ),
        "learning_rate": float(training_config.get("learning_rate", 1e-4)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "lr_scheduler_type": str(
            training_config.get("lr_scheduler_type", "cosine")
        ),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 2)),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(
            training_config.get("metric_for_best_model", "eval_loss")
        ),
        "greater_is_better": bool(
            training_config.get("greater_is_better", False)
        ),
        "report_to": report_to,
        "gradient_checkpointing": bool(
            training_config.get("gradient_checkpointing", True)
        ),
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),
        "remove_unused_columns": False,
        "seed": int(training_config.get("seed", 42)),
        "save_safetensors": bool(training_config.get("save_safetensors", True)),
        "dataloader_num_workers": int(
            training_config.get("dataloader_num_workers", 4)
        ),
        "packing": packing,
        "assistant_only_loss": bool(
            training_config.get("assistant_only_loss", False)
        ),
        "dataset_num_proc": int(training_config.get("dataset_num_proc", 4)),
        "optim": str(training_config.get("optim", "adamw_torch")),
        "logging_first_step": True,
        "eos_token": tokenizer.eos_token,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "bf16": use_bf16,
        "fp16": use_fp16,
    }

    eval_strategy_key = (
        "eval_strategy"
        if supports_argument(SFTConfig, "eval_strategy")
        else "evaluation_strategy"
    )
    sft_kwargs[eval_strategy_key] = "steps"

    max_length_key = (
        "max_length"
        if supports_argument(SFTConfig, "max_length")
        else "max_seq_length"
    )
    sft_kwargs[max_length_key] = int(
        training_config.get("max_seq_length", 3072)
    )

    if supports_argument(SFTConfig, "gradient_checkpointing_kwargs"):
        gradient_checkpointing_kwargs = training_config.get(
            "gradient_checkpointing_kwargs"
        )
        if isinstance(gradient_checkpointing_kwargs, dict):
            sft_kwargs["gradient_checkpointing_kwargs"] = (
                gradient_checkpointing_kwargs
            )

    if supports_argument(SFTConfig, "ddp_find_unused_parameters"):
        sft_kwargs["ddp_find_unused_parameters"] = bool(
            training_config.get("ddp_find_unused_parameters", False)
        )

    if supports_argument(SFTConfig, "tf32"):
        sft_kwargs["tf32"] = bool(training_config.get("tf32", True))

    if supports_argument(SFTConfig, "include_num_input_tokens_seen"):
        sft_kwargs["include_num_input_tokens_seen"] = True

    if supports_argument(SFTConfig, "padding_free"):
        sft_kwargs["padding_free"] = padding_free

    supported_sft_args = set(inspect.signature(SFTConfig).parameters)
    filtered_sft_kwargs = {
        key: value
        for key, value in sft_kwargs.items()
        if key in supported_sft_args
    }
    sft_config = SFTConfig(**filtered_sft_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
    }
    if supports_argument(SFTTrainer, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # Tokenize the test split BEFORE constructing the SFTTrainer. HF Trainer's
    # __init__ calls _move_model_to_device, so any later call into TRL's
    # _prepare_dataset captures a closure that pickles GPU tensors and crashes
    # multiprocess workers with "Cannot re-initialize CUDA in forked
    # subprocess". Doing this here keeps the closure (top-level fn + tokenizer
    # + ints) entirely CPU/picklable.
    prepared_test_dataset: Dataset | None = None
    if "test" in dataset:
        prepared_test_dataset = tokenize_eval_dataset(
            dataset["test"],
            tokenizer=tokenizer,
            max_length=int(training_config.get("max_seq_length", 3072)),
            num_proc=int(training_config.get("dataset_num_proc", 4)),
            desc="Tokenizing test dataset",
        )

    trainer = SFTTrainer(**trainer_kwargs)
    parameter_stats = get_parameter_stats(trainer.model)
    effective_warmup_steps = resolve_effective_warmup_steps(
        training_config, trainer
    )
    trainer.args.warmup_steps = effective_warmup_steps
    if hasattr(trainer.args, "warmup_ratio"):
        trainer.args.warmup_ratio = 0.0

    if trainer.is_world_process_zero():
        resolved_config = {
            "config_path": config_path,
            "model": {
                "name_or_path": model_name_or_path,
                "attn_implementation": attn_implementation,
                "aligned_special_tokens": aligned_special_tokens,
                "chat_template": chat_template_info,
            },
            "data": {
                "train_file": train_file,
                "validation_file": validation_file,
                "test_file": test_file,
                "train_samples": len(dataset["train"]),
                "validation_samples": len(dataset["validation"]),
                "test_samples": (
                    len(dataset["test"]) if "test" in dataset else 0
                ),
            },
            "training": {
                **training_config,
                "output_dir": output_dir,
                "logging_dir": logging_dir,
                "run_name": run_name,
                "report_to": report_to,
                "bf16": use_bf16,
                "fp16": use_fp16,
                "packing": packing,
                "padding_free": padding_free,
                "warmup_steps": effective_warmup_steps,
            },
            "lora": {
                **lora_config,
                "target_modules": target_modules,
                "modules_to_save": modules_to_save,
            },
            "runtime": {
                "resume_from_checkpoint": resume_from_checkpoint,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
                "max_test_samples": args.max_test_samples,
                "local_files_only": local_files_only,
                "hf_token_present": hf_token is not None,
            },
            "parameter_stats": parameter_stats,
        }
        resolved_config_filename = (
            "resolved_config.eval_only.json"
            if args.eval_only
            else "resolved_config.json"
        )
        save_json(output_dir / resolved_config_filename, resolved_config)

    if args.eval_only:
        if trainer.is_world_process_zero():
            print(
                f"[eval-only] Skipping trainer.train(); reusing the adapter "
                f"at {best_adapter_dir}."
            )
        train_metrics_path = output_dir / "train_results.json"
        if train_metrics_path.exists():
            with train_metrics_path.open("r", encoding="utf-8") as handle:
                train_metrics = json.load(handle)
        else:
            train_metrics = {}
    else:
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
        trainer.save_state()

        trainer.save_model(best_adapter_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(best_adapter_dir)

        train_metrics = add_perplexity(
            dict(train_result.metrics), "train_loss", "train_perplexity"
        )
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    validation_metrics = add_perplexity(
        dict(validation_metrics),
        "validation_loss",
        "validation_perplexity",
    )
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    summary: dict[str, Any] = {}
    summary_path = output_dir / "run_summary.json"
    if args.eval_only and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary.update(
        {
            "model_name_or_path": summary.get(
                "model_name_or_path", model_name_or_path
            ),
            "best_metric": summary.get(
                "best_metric", trainer.state.best_metric
            ),
            "best_model_checkpoint": summary.get(
                "best_model_checkpoint", trainer.state.best_model_checkpoint
            ),
            "best_adapter_dir": best_adapter_dir,
            "parameter_stats": summary.get("parameter_stats", parameter_stats),
            "train_metrics": train_metrics or summary.get("train_metrics", {}),
            "validation_metrics": validation_metrics,
            "test_metrics": summary.get("test_metrics", {}),
        }
    )
    if trainer.is_world_process_zero():
        save_json(summary_path, summary)

    test_metrics: dict[str, Any] = {}
    if prepared_test_dataset is not None:
        test_metrics = trainer.evaluate(
            eval_dataset=prepared_test_dataset,
            metric_key_prefix="test",
        )
        test_metrics = add_perplexity(
            dict(test_metrics), "test_loss", "test_perplexity"
        )
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        summary["test_metrics"] = test_metrics

    if trainer.is_world_process_zero():
        save_json(output_dir / "run_summary.json", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
