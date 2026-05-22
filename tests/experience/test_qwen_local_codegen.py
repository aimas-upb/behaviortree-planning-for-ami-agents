from src.experience.qwen_local_codegen import (
    _resolve_stop_token_ids,
    _stop_token_id_set,
    _tokenizer_preserves_whitespace,
    _truncate_at_chat_turn_boundary,
)


class FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<end_of_turn>": 2,
            "<start_of_turn>": 3,
        }.get(token, self.unk_token_id)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {
            "<|im_end|>": [4],
            "<|im_start|>": [5],
        }.get(token, [self.unk_token_id])


def test_resolve_stop_token_ids_includes_chat_turn_boundaries():
    assert _resolve_stop_token_ids(FakeTokenizer()) == [1, 2, 3, 4, 5]


def test_resolve_stop_token_ids_deduplicates_ids():
    class DuplicateTokenizer(FakeTokenizer):
        eos_token_id = 2

    assert _resolve_stop_token_ids(DuplicateTokenizer()) == [2, 3, 4, 5]


def test_stop_token_id_set_accepts_scalar_list_and_none():
    assert _stop_token_id_set(7) == {7}
    assert _stop_token_id_set([7, 8]) == {7, 8}
    assert _stop_token_id_set(None) == set()


def test_tokenizer_preserves_whitespace_detects_lossy_tokenizers():
    class PreservingTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False):
            del add_special_tokens
            return {"input_ids": text}

        def decode(self, input_ids, **kwargs):
            del kwargs
            return input_ids

    class LossyTokenizer(PreservingTokenizer):
        def decode(self, input_ids, **kwargs):
            del kwargs
            return input_ids.replace(" ", "").replace("\n", "")

    assert _tokenizer_preserves_whitespace(PreservingTokenizer())
    assert not _tokenizer_preserves_whitespace(LossyTokenizer())


def test_truncate_at_chat_turn_boundary_removes_repeated_assistant_turn():
    text = "tree = seq_2<end_of_turn>\n<start_of_turn>model\nrepeated"

    assert _truncate_at_chat_turn_boundary(text) == "tree = seq_2"


def test_truncate_at_chat_turn_boundary_removes_deepseek_eot():
    text = "tree = seq_1<|EOT|>\n"

    assert _truncate_at_chat_turn_boundary(text) == "tree = seq_1"
