"""
Reflection reasoning strategy.

Generate, critique, and refine approach.
"""

import logging

from ...config import get_model_kwargs

logger = logging.getLogger(__name__)


INITIAL_ANALYSIS_PROMPT = """You are a planning assistant. Analyze the following goal and available capabilities to create an initial plan.

## Available Capabilities

{context}

## Goal

{goal}

Provide an initial analysis including:
1. What devices are needed
2. What actions need to be taken
3. What order they should happen in
4. What behavior tree structure would work best

Be specific about which URIs to use."""


CRITIQUE_PROMPT = """You are a critical reviewer of behavior tree plans. Review the following initial analysis and identify potential issues.

## Original Goal

{goal}

## Available Capabilities

{context}

## Initial Analysis

{initial_analysis}

Critically evaluate this plan:
1. Are there any missing steps?
2. Are there any edge cases not handled?
3. Is the proposed structure optimal?
4. Are there any incorrect assumptions?
5. Could anything be done in parallel that isn't?
6. Are conditions/guards needed that weren't mentioned?

Be specific and constructive in your critique."""


REFINEMENT_PROMPT = """You are refining a behavior tree plan based on critique.

## Original Goal

{goal}

## Available Capabilities

{context}

## Initial Analysis

{initial_analysis}

## Critique

{critique}

Based on the critique, provide a refined analysis that addresses the identified issues. Be specific about:
1. The exact structure of the behavior tree
2. Which URIs to use for each action/condition
3. How edge cases will be handled
4. Why this refined approach is better"""


class ReflectionReasoning:
    """
    Reflection reasoning strategy.

    Three-phase approach:
    1. Initial analysis
    2. Critique
    3. Refined analysis
    """

    def reason(
        self,
        goal: str,
        context: str,
        client,
        model: str,
    ) -> tuple[str, list[str]]:
        """
        Perform reflection reasoning.

        Args:
            goal: The user's goal
            context: Discovery context
            client: OpenAI client
            model: Model name

        Returns:
            Tuple of (enhanced context with reasoning, reasoning trace)
        """
        logger.info("Starting reflection reasoning")
        reasoning_trace = []

        # Phase 1: Initial analysis
        logger.debug("Phase 1: Initial analysis")
        api_kwargs_1 = get_model_kwargs(model)
        api_kwargs_1["messages"] = [{
            "role": "user",
            "content": INITIAL_ANALYSIS_PROMPT.format(context=context, goal=goal),
        }]

        initial_response = client.chat.completions.create(**api_kwargs_1)
        initial_analysis = initial_response.choices[0].message.content
        reasoning_trace.append(f"## Initial Analysis\n\n{initial_analysis}")

        # Phase 2: Critique
        logger.debug("Phase 2: Critique")
        api_kwargs_2 = get_model_kwargs(model)
        api_kwargs_2["messages"] = [{
            "role": "user",
            "content": CRITIQUE_PROMPT.format(
                goal=goal,
                context=context,
                initial_analysis=initial_analysis,
            ),
        }]

        critique_response = client.chat.completions.create(**api_kwargs_2)
        critique = critique_response.choices[0].message.content
        reasoning_trace.append(f"## Critique\n\n{critique}")

        # Phase 3: Refinement
        logger.debug("Phase 3: Refinement")
        api_kwargs_3 = get_model_kwargs(model)
        api_kwargs_3["messages"] = [{
            "role": "user",
            "content": REFINEMENT_PROMPT.format(
                goal=goal,
                context=context,
                initial_analysis=initial_analysis,
                critique=critique,
            ),
        }]

        refinement_response = client.chat.completions.create(**api_kwargs_3)
        refined_analysis = refinement_response.choices[0].message.content
        reasoning_trace.append(f"## Refined Analysis\n\n{refined_analysis}")

        logger.info("Reflection reasoning complete (3 phases)")

        # Enhanced context with refined analysis
        enhanced_context = f"""{context}

## Planning Analysis (Reflection)

### Initial Analysis
{initial_analysis}

### Critique
{critique}

### Refined Plan
{refined_analysis}"""

        return enhanced_context, reasoning_trace
