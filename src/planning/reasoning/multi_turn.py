"""
Multi-turn reasoning strategy.

Multi-turn conversation for deeper exploration before generation.
"""

import json
import logging

from ...config import ModelConfig, get_model_kwargs

logger = logging.getLogger(__name__)


MULTI_TURN_SYSTEM_PROMPT = """You are a planning assistant that explores and reasons through problems step-by-step.

You have tools to help you think through the problem:

1. analyze_goal - Break down the goal into components
2. identify_dependencies - Analyze dependencies between actions
3. consider_edge_cases - Think about what could go wrong
4. propose_structure - Propose a high-level BT structure
5. done_reasoning - Signal you're ready to proceed with generation

Use these tools to thoroughly analyze the problem before generating a behavior tree.
Think carefully about the order of operations, parallelism opportunities, and error handling."""


MULTI_TURN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_goal",
            "description": "Break down the goal into sub-goals and identify required devices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sub-goals",
                    },
                    "required_devices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of devices/artifacts needed",
                    },
                },
                "required": ["sub_goals", "required_devices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_dependencies",
            "description": "Analyze dependencies between actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequential": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Actions that must happen in order",
                    },
                    "parallel": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Actions that can happen simultaneously",
                    },
                    "conditional": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Actions that depend on conditions",
                    },
                },
                "required": ["sequential", "parallel", "conditional"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consider_edge_cases",
            "description": "Think about what could go wrong and how to handle it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edge_cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "case": {"type": "string"},
                                "handling": {"type": "string"},
                            },
                        },
                        "description": "Edge cases and their handling",
                    },
                },
                "required": ["edge_cases"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_structure",
            "description": "Propose a high-level behavior tree structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_type": {
                        "type": "string",
                        "enum": ["sequence", "selector", "parallel"],
                        "description": "Type of root node",
                    },
                    "structure_description": {
                        "type": "string",
                        "description": "Description of the proposed structure",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this structure is appropriate",
                    },
                },
                "required": ["root_type", "structure_description", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_reasoning",
            "description": "Signal that you're done reasoning and ready to generate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of your analysis",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


class MultiTurnReasoning:
    """
    Multi-turn reasoning strategy.

    Uses multiple turns of tool-calling to explore the problem space
    before plan generation.
    """

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns

    def reason(
        self,
        goal: str,
        context: str,
        client,
        model_config: ModelConfig,
    ) -> tuple[str, list[str]]:
        """
        Perform multi-turn reasoning.

        Args:
            goal: The user's goal
            context: Discovery context
            client: OpenAI client
            model_config: Model configuration

        Returns:
            Tuple of (enhanced context with reasoning, reasoning trace)
        """
        logger.info(f"Starting multi-turn reasoning (max {self.max_turns} turns)")

        messages = [
            {"role": "system", "content": MULTI_TURN_SYSTEM_PROMPT},
            {"role": "user", "content": f"## Context\n\n{context}\n\n## Goal\n\n{goal}\n\nAnalyze this goal step by step using the available tools."},
        ]

        reasoning_trace = []
        analysis_results = []

        for turn in range(self.max_turns):
            api_kwargs = get_model_kwargs(model_config.name, model_config=model_config)
            api_kwargs.update({
                "messages": messages,
                "tools": MULTI_TURN_TOOLS,
                "tool_choice": "auto",
            })
            response = client.chat.completions.create(**api_kwargs)

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                # Model responded without tool call
                if message.content:
                    reasoning_trace.append(f"Turn {turn + 1}: {message.content}")
                break

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Record reasoning
                reasoning_trace.append(f"Turn {turn + 1} - {fn_name}: {json.dumps(fn_args, indent=2)}")
                analysis_results.append({
                    "tool": fn_name,
                    "result": fn_args,
                })

                if fn_name == "done_reasoning":
                    logger.info(f"Multi-turn reasoning complete after {turn + 1} turns")
                    break

                # Echo back the tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"status": "recorded", "data": fn_args}),
                })

            # Check if done_reasoning was called
            if any(tc.function.name == "done_reasoning" for tc in (message.tool_calls or [])):
                break

        # Build enhanced context
        analysis_text = "\n\n".join([
            f"### {r['tool']}\n{json.dumps(r['result'], indent=2)}"
            for r in analysis_results
        ])

        enhanced_context = f"""{context}

## Multi-Turn Analysis

{analysis_text}"""

        logger.info(f"Multi-turn reasoning produced {len(reasoning_trace)} trace entries")
        return enhanced_context, reasoning_trace
