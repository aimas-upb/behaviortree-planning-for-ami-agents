from src.experience.modify_codegen_prompting import (
    build_modify_codegen_messages,
    sanitize_direct_codegen_response,
)


def test_build_modify_codegen_messages_planner_exact_matches_sft_shape():
    messages = build_modify_codegen_messages(
        runtime_modify_goal="**Intent 1**\ntext_intent: increase brightness",
        runtime_context="# Available Devices\n- foyerLight",
        prompt_style="planner_exact",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "# Available Devices" in messages[0]["content"]
    assert "Return only the Python code" in messages[0]["content"]
    assert messages[1]["content"] == "**Intent 1**\ntext_intent: increase brightness"


def test_build_modify_codegen_messages_compact_includes_context_in_user():
    messages = build_modify_codegen_messages(
        runtime_modify_goal="increase brightness by 5",
        runtime_context="# context",
        prompt_style="compact",
    )

    assert "Modify goals:" in messages[1]["content"]
    assert "Available devices, capabilities, and current state:" in messages[1]["content"]
    assert "# context" in messages[1]["content"]


def test_sanitize_direct_codegen_response_strips_prefixes_and_fences():
    raw = "assistant\n```python\n\n# IMPOSSIBLE: unsupported\n\ntree = seq_1\n```"

    assert (
        sanitize_direct_codegen_response(raw)
        == "# IMPOSSIBLE: unsupported\n\ntree = seq_1"
    )
