"""
Python code output generator.

Generates Python py_trees code that is exec'd directly.
"""

import logging

from ..base import Plan

logger = logging.getLogger(__name__)


# Tool definition for generating Python code
GENERATE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_behavior_tree_code",
        "description": "Generate Python code that constructs a py_trees behavior tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code that defines a 'tree' variable containing the behavior tree",
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the code",
                },
            },
            "required": ["code", "explanation"],
        },
    },
}


class PythonCodeGenerator:
    """
    Python code output generator.

    Generates Python code that constructs py_trees behavior trees.
    """

    def generate(
        self,
        goal: str,
        context: str,
        client,
        model: str,
        prompt_strategy: str,
    ) -> Plan:
        """
        Generate Python py_trees code.

        Args:
            goal: The user's goal
            context: Context (possibly enhanced by reasoning)
            client: OpenAI client
            model: Model name
            prompt_strategy: Which prompt template to use

        Returns:
            Plan with Python code content
        """
        from ...prompts import get_prompt

        logger.info(f"Generating Python code with prompt strategy: {prompt_strategy}")

        # Get prompt template
        system_prompt, tool_description = get_prompt(prompt_strategy, "python_code")

        # Format system prompt with context
        formatted_system = system_prompt.format(capability_model=context)

        # Update tool description
        tool = dict(GENERATE_CODE_TOOL)
        tool["function"]["description"] = tool_description

        messages = [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": goal},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "generate_behavior_tree_code"}},
            temperature=0.0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            logger.warning("No tool call in response")
            return Plan(
                format="python_code",
                content="",
                explanation=f"Generation failed: {message.content}",
            )

        tool_call = message.tool_calls[0]
        try:
            import json
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return Plan(
                format="python_code",
                content="",
                explanation=f"Parse error: {e}",
            )

        code = args.get("code", "")
        explanation = args.get("explanation", "")

        # Clean up code if wrapped in markdown
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        logger.info(f"Generated Python code ({len(code)} chars)")

        return Plan(
            format="python_code",
            content=code,
            explanation=explanation,
        )


class UnconstrainedPythonCodeGenerator:
    """
    Unconstrained Python code output generator.

    Generates Python code that can define custom py_trees behaviors
    with direct HTTP access to the environment.
    """

    def generate(
        self,
        goal: str,
        context: str,
        client,
        model: str,
        prompt_strategy: str,
    ) -> Plan:
        """
        Generate unconstrained Python py_trees code.

        Args:
            goal: The user's goal
            context: Context (possibly enhanced by reasoning)
            client: OpenAI client
            model: Model name
            prompt_strategy: Which prompt template to use

        Returns:
            Plan with Python code content (unconstrained format)
        """
        from ...prompts import get_prompt

        logger.info(f"Generating unconstrained Python code with prompt strategy: {prompt_strategy}")

        # Get unconstrained prompt template
        system_prompt, tool_description = get_prompt(prompt_strategy, "python_code_unconstrained")

        # Format system prompt with context
        formatted_system = system_prompt.format(capability_model=context)

        # Update tool description
        tool = dict(GENERATE_CODE_TOOL)
        tool["function"]["description"] = tool_description

        messages = [
            {"role": "system", "content": formatted_system},
            {"role": "user", "content": goal},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "generate_behavior_tree_code"}},
            temperature=0.0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            logger.warning("No tool call in response")
            return Plan(
                format="python_code_unconstrained",
                content="",
                explanation=f"Generation failed: {message.content}",
            )

        tool_call = message.tool_calls[0]
        try:
            import json
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return Plan(
                format="python_code_unconstrained",
                content="",
                explanation=f"Parse error: {e}",
            )

        code = args.get("code", "")
        explanation = args.get("explanation", "")

        # Clean up code if wrapped in markdown
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        logger.info(f"Generated unconstrained Python code ({len(code)} chars)")

        return Plan(
            format="python_code_unconstrained",
            content=code,
            explanation=explanation,
        )
