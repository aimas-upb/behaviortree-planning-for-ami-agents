"""
Local Qwen modify-codegen inference utilities.

Loads the base Qwen model plus an optional LoRA adapter and serves generation
requests directly in-process.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CHAT_TURN_BOUNDARY_TOKENS = (
    "<end_of_turn>",
    "<start_of_turn>",
    "<|im_end|>",
    "<|im_start|>",
    "<|EOT|>",
    "<｜end▁of▁sentence｜>",
)

WHITESPACE_ROUND_TRIP_PROBE = "a b\ndef f():\n    return 1"


def _resolve_runtime_path(path: str | None) -> str | None:
    """Resolve local filesystem paths while leaving Hub repo ids untouched."""
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    if candidate.exists():
        return str(candidate.resolve())

    return path


def _load_transformers_stack():
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )

    return (
        torch,
        PeftConfig,
        PeftModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )


def _resolve_torch_dtype(torch_module, name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _tokenizer_preserves_whitespace(tokenizer) -> bool:
    encoded = tokenizer(
        WHITESPACE_ROUND_TRIP_PROBE,
        add_special_tokens=False,
    )
    decoded = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return decoded == WHITESPACE_ROUND_TRIP_PROBE


def _special_token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _load_tokenizer_config_file(
    tokenizer_source: str,
    local_files_only: bool,
) -> Path:
    source_path = Path(tokenizer_source)
    if source_path.is_dir():
        return source_path / "tokenizer_config.json"

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=tokenizer_source,
            filename="tokenizer_config.json",
            local_files_only=local_files_only,
        )
    )


def _load_tokenizer_json_file(
    tokenizer_source: str,
    local_files_only: bool,
) -> Path:
    source_path = Path(tokenizer_source)
    if source_path.is_dir():
        return source_path / "tokenizer.json"

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=tokenizer_source,
            filename="tokenizer.json",
            local_files_only=local_files_only,
        )
    )


def _load_fast_tokenizer_from_tokenizer_json(
    tokenizer_cls,
    tokenizer_source: str,
    local_files_only: bool,
):
    tokenizer_json_file = _load_tokenizer_json_file(
        tokenizer_source, local_files_only
    )
    tokenizer_config_file = _load_tokenizer_config_file(
        tokenizer_source, local_files_only
    )

    with tokenizer_config_file.open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)

    tokenizer_kwargs: dict[str, Any] = {}
    for token_name in (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
    ):
        token = _special_token_content(tokenizer_config.get(token_name))
        if token is not None:
            tokenizer_kwargs[token_name] = token

    tokenizer = tokenizer_cls(
        tokenizer_file=str(tokenizer_json_file),
        **tokenizer_kwargs,
    )
    if "clean_up_tokenization_spaces" in tokenizer_config:
        tokenizer.clean_up_tokenization_spaces = bool(
            tokenizer_config["clean_up_tokenization_spaces"]
        )
    chat_template = tokenizer_config.get("chat_template")
    if isinstance(chat_template, str):
        tokenizer.chat_template = chat_template
    return tokenizer


def _load_codegen_tokenizer(
    auto_tokenizer_cls,
    fast_tokenizer_cls,
    tokenizer_source: str,
    local_files_only: bool,
):
    tokenizer = auto_tokenizer_cls.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if _tokenizer_preserves_whitespace(tokenizer):
        return tokenizer

    LOGGER.warning(
        "Tokenizer at %s does not preserve whitespace; loading tokenizer.json "
        "directly for code-generation-safe encoding/decoding.",
        tokenizer_source,
    )
    fallback_tokenizer = _load_fast_tokenizer_from_tokenizer_json(
        fast_tokenizer_cls,
        tokenizer_source,
        local_files_only,
    )
    if not _tokenizer_preserves_whitespace(fallback_tokenizer):
        raise RuntimeError(
            f"Tokenizer at {tokenizer_source} does not preserve code "
            "whitespace, and tokenizer.json fallback did not fix it."
        )
    return fallback_tokenizer


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _single_token_id(tokenizer, token: str) -> int | None:
    token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)

    if isinstance(token_id, int):
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id >= 0 and token_id != unk_token_id:
            return token_id

    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) == 1:
            encoded_id = int(encoded[0])
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            if encoded_id >= 0 and encoded_id != unk_token_id:
                return encoded_id

    return None


def _resolve_stop_token_ids(tokenizer) -> int | list[int] | None:
    stop_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_token_ids.append(eos_token_id)

    for token in CHAT_TURN_BOUNDARY_TOKENS:
        token_id = _single_token_id(tokenizer, token)
        if token_id is not None:
            stop_token_ids.append(token_id)

    stop_token_ids = _unique_ints(stop_token_ids)
    if not stop_token_ids:
        return None
    if len(stop_token_ids) == 1:
        return stop_token_ids[0]
    return stop_token_ids


def _stop_token_id_set(stop_token_ids: int | list[int] | None) -> set[int]:
    if stop_token_ids is None:
        return set()
    if isinstance(stop_token_ids, int):
        return {stop_token_ids}
    return set(stop_token_ids)


def _truncate_at_chat_turn_boundary(text: str) -> str:
    positions = [
        text.find(token)
        for token in CHAT_TURN_BOUNDARY_TOKENS
        if text.find(token) != -1
    ]
    if not positions:
        return text
    return text[: min(positions)].rstrip()


@dataclass(frozen=True)
class QwenLocalGenerationSettings:
    """Runtime settings for loading the local Qwen modify-codegen model."""

    base_model_name_or_path: str | None
    adapter_path: str | None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    local_files_only: bool = True

    def cache_key(self) -> tuple[str | None, str | None, str, str, bool]:
        return (
            _resolve_runtime_path(self.base_model_name_or_path),
            _resolve_runtime_path(self.adapter_path),
            self.device_map,
            self.torch_dtype,
            self.local_files_only,
        )


@dataclass
class QwenGenerationResponse:
    """Minimal generation response for the direct modify-codegen backend."""

    text: str
    model_name: str
    usage: dict[str, int]
    finish_reason: str


class QwenLocalModifyCodegenModel:
    """
    In-process Qwen modify-codegen runner with simple per-process caching.

    The sampled benchmark runs sequentially inside a single worker process, so
    loading the model once and reusing it avoids repeated multi-GB reloads.
    """

    _CACHE: dict[
        tuple[str | None, str | None, str, str, bool],
        "QwenLocalModifyCodegenModel",
    ] = {}

    @classmethod
    def from_settings(
        cls, settings: QwenLocalGenerationSettings
    ) -> "QwenLocalModifyCodegenModel":
        cache_key = settings.cache_key()
        cached = cls._CACHE.get(cache_key)
        if cached is not None:
            return cached

        instance = cls(settings=settings)
        cls._CACHE[cache_key] = instance
        return instance

    def __init__(self, settings: QwenLocalGenerationSettings):
        self.settings = QwenLocalGenerationSettings(
            base_model_name_or_path=_resolve_runtime_path(
                settings.base_model_name_or_path
            ),
            adapter_path=_resolve_runtime_path(settings.adapter_path),
            device_map=settings.device_map,
            torch_dtype=settings.torch_dtype,
            local_files_only=settings.local_files_only,
        )

        (
            self._torch,
            peft_config_cls,
            peft_model_cls,
            auto_model_cls,
            auto_tokenizer_cls,
            fast_tokenizer_cls,
        ) = _load_transformers_stack()
        self._peft_model_cls = peft_model_cls

        resolved_base_model = self.settings.base_model_name_or_path
        if self.settings.adapter_path:
            peft_config = peft_config_cls.from_pretrained(
                self.settings.adapter_path,
                local_files_only=self.settings.local_files_only,
            )
            if not resolved_base_model:
                resolved_base_model = peft_config.base_model_name_or_path

        if not resolved_base_model:
            raise ValueError(
                "A base model path is required when no adapter path is provided."
            )

        self.base_model_name_or_path = resolved_base_model

        tokenizer_source = (
            self.settings.adapter_path
            if self.settings.adapter_path
            and Path(self.settings.adapter_path).is_dir()
            and (
                Path(self.settings.adapter_path) / "tokenizer_config.json"
            ).exists()
            else self.base_model_name_or_path
        )

        LOGGER.info(
            "Loading local tokenizer from %s",
            tokenizer_source,
        )
        self.tokenizer = _load_codegen_tokenizer(
            auto_tokenizer_cls,
            fast_tokenizer_cls,
            tokenizer_source,
            self.settings.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                f"Tokenizer at {tokenizer_source} has no `chat_template`; "
                "modify-codegen inference relies on `apply_chat_template`. "
                "For StarCoder2, ensure the training job ran "
                "`tokenizer.save_pretrained(best_adapter_dir)` so the cloned "
                "ChatML template ships alongside the adapter."
            )

        LOGGER.info(
            "Loading local base model from %s",
            self.base_model_name_or_path,
        )
        self.model = auto_model_cls.from_pretrained(
            self.base_model_name_or_path,
            torch_dtype=_resolve_torch_dtype(
                self._torch, self.settings.torch_dtype
            ),
            device_map=self.settings.device_map,
            local_files_only=self.settings.local_files_only,
        )

        if self.settings.adapter_path:
            LOGGER.info(
                "Applying LoRA adapter from %s",
                self.settings.adapter_path,
            )
            self.model = self._peft_model_cls.from_pretrained(
                self.model,
                self.settings.adapter_path,
                local_files_only=self.settings.local_files_only,
            )

        self.model.eval()
        self.model_name = (
            self.settings.adapter_path or self.base_model_name_or_path
        )

    @property
    def model_device(self):
        return next(self.model.parameters()).device

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> QwenGenerationResponse:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(self.model_device) for key, value in encoded.items()
        }

        do_sample = temperature > 0.0
        stop_token_ids = _resolve_stop_token_ids(self.tokenizer)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": stop_token_ids,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with self._torch.inference_mode():
            outputs = self.model.generate(**encoded, **generation_kwargs)

        input_tokens = int(encoded["input_ids"].shape[-1])
        generated_ids = outputs[0][input_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        generated_token_ids = [int(token_id) for token_id in generated_ids]
        stopped_on_token = bool(
            generated_token_ids
            and generated_token_ids[-1] in _stop_token_id_set(stop_token_ids)
        )
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = _truncate_at_chat_turn_boundary(text)
        finish_reason = (
            "stop"
            if stopped_on_token
            else "length" if completion_tokens >= max_new_tokens else "stop"
        )

        return QwenGenerationResponse(
            text=text,
            model_name=self.model_name,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens,
            },
            finish_reason=finish_reason,
        )
