"""
Shared prompt helpers for modify-codegen training and inference.

The fine-tuned Qwen path should render prompts exactly like the SFT export,
so the dataset exporter and the direct inference runner both depend on this
module.
"""

from __future__ import annotations

import re

from ..prompts.code.detailed_structured_modify_only import MODIFY_ONLY_SYSTEM

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


def build_modify_codegen_system_prompt(
    runtime_context: str, prompt_style: str
) -> str:
    """Render the system prompt for the requested modify-codegen prompt style."""
    context = str(runtime_context).strip()

    if prompt_style == "compact":
        return COMPACT_SYSTEM_PROMPT

    if prompt_style == "planner_exact":
        return (
            MODIFY_ONLY_SYSTEM.format(capability_model=context).strip()
            + DIRECT_RESPONSE_SUFFIX
        )

    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def build_modify_codegen_user_prompt(
    runtime_modify_goal: str,
    runtime_context: str,
    prompt_style: str,
) -> str:
    """Render the user prompt for the requested modify-codegen prompt style."""
    goal = str(runtime_modify_goal).strip()
    context = str(runtime_context).strip()

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


def build_modify_codegen_messages(
    runtime_modify_goal: str,
    runtime_context: str,
    prompt_style: str = "planner_exact",
) -> list[dict[str, str]]:
    """Build direct-chat messages for modify-only code generation."""
    return [
        {
            "role": "system",
            "content": build_modify_codegen_system_prompt(
                runtime_context, prompt_style
            ),
        },
        {
            "role": "user",
            "content": build_modify_codegen_user_prompt(
                runtime_modify_goal, runtime_context, prompt_style
            ),
        },
    ]


def sanitize_direct_codegen_response(text: str) -> str:
    """Strip wrapper text so direct model output can be exec'd as Python."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = re.sub(
        r"^assistant(?:\s*:)?\s+",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )

    fence_match = re.fullmatch(
        r"```(?:python)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    if cleaned.startswith("```python"):
        cleaned = cleaned[9:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()
