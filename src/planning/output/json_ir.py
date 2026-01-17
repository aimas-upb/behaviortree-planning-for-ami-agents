"""
JSON IR output generator.

Generates JSON intermediate representation that is compiled to py_trees.
"""

import json
import logging

from openai import OpenAI

from ..base import Plan
from ...config import get_model_kwargs

logger = logging.getLogger(__name__)


# Shared JSON schema for behavior tree nodes
TREE_PARAMETER_SCHEMA = {
    "type": "object",
    "minProperties": 1,
    "properties": {
        "name": {"type": "string"},
        "type": {
            "type": "string",
            "enum": ["sequence", "selector", "parallel", "action", "condition"],
        },
        "children": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "object"},
            "description": "Child nodes (required for sequence/selector/parallel).",
        },
        "policy": {
            "type": "string",
            "enum": ["success_on_all", "success_on_one"],
            "description": "Policy for parallel nodes.",
        },
        "action_url": {
            "type": "string",
            "description": "HTTP POST URL to invoke for action nodes.",
        },
        "parameters": {
            "type": "object",
            "description": "Optional parameters for action nodes.",
        },
        "property_url": {
            "type": "string",
            "description": "HTTP GET URL to check for condition nodes.",
        },
        "expected_value": {
            "description": "Expected value for condition nodes.",
        },
        "operator": {
            "type": "string",
            "enum": ["==", "!=", ">", "<", ">=", "<="],
            "description": "Optional comparison operator for condition nodes.",
        },
        "value_path": {
            "type": "string",
            "description": "Optional JSON path for nested condition values.",
        },
    },
    "required": ["name", "type"],
    "oneOf": [
        {
            "title": "Sequence",
            "properties": {"type": {"const": "sequence"}},
            "required": ["children"],
        },
        {
            "title": "Selector",
            "properties": {"type": {"const": "selector"}},
            "required": ["children"],
        },
        {
            "title": "Parallel",
            "properties": {"type": {"const": "parallel"}},
            "required": ["children"],
        },
        {
            "title": "Action",
            "properties": {"type": {"const": "action"}},
            "required": ["action_url"],
        },
        {
            "title": "Condition",
            "properties": {"type": {"const": "condition"}},
            "required": ["property_url", "expected_value"],
        },
    ],
    "description": "Behavior tree specification in JSON format; DO NOT leave this empty if the goal is possible to achieve.",
}


# Tool definition for generating behavior trees
GENERATE_BT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_behavior_tree",
        "description": "Generate a behavior tree specification.",
        "parameters": {
            "type": "object",
            "properties": {
                "tree": TREE_PARAMETER_SCHEMA,
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the plan",
                },
                "impossible": {
                    "type": "boolean",
                    "description": "If the goal is impossible to achieve, mark this as true",
                }
            },
            "required": ["tree", "explanation", "impossible"],
        },
    },
}


class JsonIRGenerator:
    """
    JSON IR output generator.

    Generates JSON behavior tree specification using LLM tool calling.
    """

    def generate(
        self,
        goal: str,
        context: str,
        client: OpenAI,
        model: str,
        prompt_strategy: str,
    ) -> Plan:
        """
        Generate JSON IR behavior tree.

        Args:
            goal: The user's goal
            context: Context (possibly enhanced by reasoning)
            client: OpenAI client
            model: Model name
            prompt_strategy: Which prompt template to use

        Returns:
            Plan with JSON IR content
        """
        from ...prompts import get_prompt

        logger.info(f"Generating JSON IR with prompt strategy: {prompt_strategy}")

        # Get prompt template
        system_prompt, tool_description = get_prompt(prompt_strategy, "json_ir")

        # Format system prompt with context
        formatted_system = system_prompt.format(capability_model=context)

        # Update tool description
        tool = dict(GENERATE_BT_TOOL)
        tool["function"]["description"] = tool_description

        messages = [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": goal},
        ]

        max_attempts = 3
        last_explanation = ""
        validation_errors: list[str] = []

        for attempt in range(max_attempts):
            api_kwargs = get_model_kwargs(model)
            api_kwargs.update({
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "generate_behavior_tree"}},
            })
            response = client.chat.completions.create(**api_kwargs)

            message = response.choices[0].message

            assistant_message = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_message["tool_calls"] = message.tool_calls
            messages.append(assistant_message)

            if not message.tool_calls:
                logger.warning("No tool call in response")
                return Plan(
                    format="json_ir",
                    content={},
                    explanation=f"Generation failed: {message.content}",
                )

            tool_call = message.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return Plan(
                    format="json_ir",
                    content={},
                    explanation=f"JSON parse error: {e}",
                )

            tree_spec = args.get("tree", {})
            last_explanation = args.get("explanation", "")
            impossible = args.get("impossible", False)

            if impossible:
                return Plan(
                    format="json_ir",
                    content={},
                    explanation=last_explanation,
                )

            validation_errors = self._validate_tree(tree_spec)
            if not validation_errors:
                logger.info(f"Generated JSON IR with {self._count_nodes(tree_spec)} nodes")
                return Plan(
                    format="json_ir",
                    content=tree_spec,
                    explanation=last_explanation,
                )

            logger.warning(
                "Invalid tree spec (attempt %d/%d): %s",
                attempt + 1,
                max_attempts,
                "; ".join(validation_errors),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        "The previous behavior tree was invalid:\n- "
                        + "\n- ".join(validation_errors)
                        + "\nPlease call generate_behavior_tree again with a corrected, non-empty tree."
                    ),
                }
            )

        return Plan(
            format="json_ir",
            content={},
            explanation="Generation failed: " + "; ".join(validation_errors),
        )

    def _count_nodes(self, spec: dict) -> int:
        """Count nodes in a tree spec."""
        if not spec:
            return 0
        count = 1
        for child in spec.get("children", []):
            count += self._count_nodes(child)
        return count

    def _validate_tree(self, spec: dict, path: str = "tree") -> list[str]:
        """Validate that the tree spec is non-empty and structurally sound."""
        errors: list[str] = []

        if not spec:
            errors.append(f"{path}: tree is empty")
            return errors

        if not isinstance(spec, dict):
            errors.append(f"{path}: expected object, got {type(spec).__name__}")
            return errors

        name = spec.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"{path}: missing 'name'")

        node_type = spec.get("type")
        valid_types = {"sequence", "selector", "parallel", "action", "condition"}
        if node_type not in valid_types:
            errors.append(f"{path}: missing or invalid 'type'")

        # Composite nodes
        if node_type in {"sequence", "selector", "parallel"}:
            children = spec.get("children")
            if not isinstance(children, list) or not children:
                errors.append(f"{path}: composite nodes require non-empty 'children'")
            else:
                for idx, child in enumerate(children):
                    errors.extend(
                        self._validate_tree(child, path=f"{path}.children[{idx}]")
                    )
        # Action nodes
        elif node_type == "action":
            action_url = spec.get("action_url")
            if not action_url or not isinstance(action_url, str):
                errors.append(f"{path}: action nodes require 'action_url'")
        # Condition nodes
        elif node_type == "condition":
            property_url = spec.get("property_url")
            if not property_url or not isinstance(property_url, str):
                errors.append(f"{path}: condition nodes require 'property_url'")
            if "expected_value" not in spec:
                errors.append(f"{path}: condition nodes require 'expected_value'")

        return errors
