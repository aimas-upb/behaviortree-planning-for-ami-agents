"""
JSON IR output generator.

Generates JSON intermediate representation that is compiled to py_trees.
"""

import json
import logging

from openai import OpenAI

from ..base import Plan

logger = logging.getLogger(__name__)


# Tool definition for generating behavior trees
GENERATE_BT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_behavior_tree",
        "description": "Generate a behavior tree specification.",
        "parameters": {
            "type": "object",
            "properties": {
                "tree": {
                    "type": "object",
                    "description": "Behavior tree specification in JSON format; DO NOT leave this empty",
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the plan",
                },
            },
            "required": ["tree", "explanation"],
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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "generate_behavior_tree"}},
            temperature=0.0,
        )

        message = response.choices[0].message

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
        explanation = args.get("explanation", "")

        logger.info(f"Generated JSON IR with {self._count_nodes(tree_spec)} nodes")

        return Plan(
            format="json_ir",
            content=tree_spec,
            explanation=explanation,
        )

    def _count_nodes(self, spec: dict) -> int:
        """Count nodes in a tree spec."""
        if not spec:
            return 0
        count = 1
        for child in spec.get("children", []):
            count += self._count_nodes(child)
        return count
