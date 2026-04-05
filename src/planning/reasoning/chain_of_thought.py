"""
Chain of Thought reasoning strategy.

Single-turn reasoning before generation.
"""

import logging

from ...config import ModelConfig, get_model_kwargs

logger = logging.getLogger(__name__)


COT_SYSTEM_PROMPT = """You are a planning assistant that thinks step-by-step before generating behavior trees.

Your task is to analyze the goal and available capabilities, then provide a structured analysis that will help with behavior tree generation.

## Analysis Steps

1. **Goal Decomposition**: Break down the goal into atomic sub-goals
2. **Device Identification**: Identify which devices/artifacts are needed
3. **Action Mapping**: Map sub-goals to specific actions
4. **Constraint Validation**: Check if requested values are within device constraints. Actions show parameter constraints like "range: MIN-MAX" or "values: [a, b, c]". If the goal requests a value outside valid range, note the closest valid value to use instead.
5. **Dependency Analysis**: Determine order and dependencies between actions
6. **Pattern Selection**: Choose appropriate BT patterns (sequence, selector, parallel)
7. **Edge Cases**: Consider what could go wrong and how to handle it

**IMPORTANT**: Always check parameter constraints! If an action shows "range: 30-100" for temperature, and the goal asks for 22 degrees, you MUST note that 30 is the minimum valid value and should be used instead.

Provide your analysis in a structured format that will be used for the next step of plan generation."""


COT_USER_PROMPT = """## Available Capabilities

{context}

## Goal

{goal}

Please analyze this goal step-by-step following the analysis framework. Be thorough but concise."""


class ChainOfThoughtReasoning:
    """
    Chain of Thought reasoning strategy.

    Performs a single reasoning turn to analyze the goal and context
    before plan generation.
    """

    def reason(
        self,
        goal: str,
        context: str,
        client,
        model_config: ModelConfig,
    ) -> tuple[str, list[str]]:
        """
        Perform chain-of-thought reasoning.

        Args:
            goal: The user's goal
            context: Discovery context
            client: OpenAI client
            model_config: Model configuration

        Returns:
            Tuple of (enhanced context with reasoning, reasoning trace)
        """
        logger.info("Starting chain-of-thought reasoning")

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": COT_USER_PROMPT.format(context=context, goal=goal),
            },
        ]

        api_kwargs = get_model_kwargs(
            model_config.name, model_config=model_config
        )
        api_kwargs["messages"] = messages

        response = client.chat.completions.create(**api_kwargs)

        reasoning = response.choices[0].message.content
        logger.info("Chain-of-thought reasoning complete")

        # Enhance context with reasoning
        enhanced_context = f"""{context}

## Planning Analysis

{reasoning}"""

        return enhanced_context, [reasoning]
