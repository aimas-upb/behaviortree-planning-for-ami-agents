"""
Step 1: Intent Extraction.

Parses a user's natural language goal into structured intents
using an LLM grounded in the homeont ontology.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config import ModelConfig, get_model_kwargs

logger = logging.getLogger(__name__)


@dataclass
class StructuredIntent:
    """A single atomic intent extracted from a user goal."""

    text_intent: str
    affordance_type: str  # e.g. "SetBrightnessCommand"
    verb: str  # "set" or "modify"
    artifact_type: str  # e.g. "Light"
    workspace_type: str  # e.g. "Bathroom"
    original_index: int = 0

    def slot_key(self) -> tuple:
        """Slot-matching key: (affordance_type, artifact_type, workspace_type)."""
        return (self.affordance_type, self.artifact_type, self.workspace_type)

    def to_dict(self) -> dict:
        return {
            "text_intent": self.text_intent,
            "action": {
                "affordance_type": self.affordance_type,
                "verb": self.verb,
            },
            "target": {
                "artifact_type": self.artifact_type,
                "workspace_type": self.workspace_type,
            },
            "original_index": self.original_index,
        }

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "StructuredIntent":
        action = data.get("action", {})
        target = data.get("target", {})
        return cls(
            text_intent=data.get("text_intent", ""),
            affordance_type=action.get("affordance_type", ""),
            verb=action.get("verb", "set"),
            artifact_type=target.get("artifact_type", ""),
            workspace_type=target.get("workspace_type", ""),
            original_index=index,
        )


# ---------------------------------------------------------------------------
# LLM tool definition for intent extraction
# ---------------------------------------------------------------------------

EXTRACT_INTENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_intents",
        "description": "Extract structured intents from a user goal about smart home devices.",
        "parameters": {
            "type": "object",
            "properties": {
                "intents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text_intent": {
                                "type": "string",
                                "description": (
                                    "The natural language subsequence from the user request "
                                    "that corresponds to this single atomic intent."
                                ),
                            },
                            "action": {
                                "type": "object",
                                "properties": {
                                    "affordance_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology command class name, e.g. TurnOnCommand, "
                                            "SetBrightnessCommand. Must be a subclass of "
                                            "ActionAffordance from the ontology."
                                        ),
                                    },
                                    "verb": {
                                        "type": "string",
                                        "enum": ["set", "modify"],
                                        "description": (
                                            "'set' for absolute value actions or parameterless "
                                            "actions (turn on/off, open/close). "
                                            "'modify' for relative changes (increase/decrease by)."
                                        ),
                                    },
                                },
                                "required": ["affordance_type", "verb"],
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                    "artifact_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology device class name, e.g. Light, Fan, "
                                            "AirConditioner. Must be a subclass of Artifact "
                                            "from the ontology."
                                        ),
                                    },
                                    "workspace_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology workspace class name, e.g. Bathroom, "
                                            "Kitchen, LivingRoom. Must be a subclass of "
                                            "Workspace from the ontology."
                                        ),
                                    },
                                },
                                "required": ["artifact_type", "workspace_type"],
                            },
                        },
                        "required": ["text_intent", "action", "target"],
                    },
                },
            },
            "required": ["intents"],
        },
    },
}


# ---------------------------------------------------------------------------
# System prompt for intent extraction
# ---------------------------------------------------------------------------

INTENT_EXTRACTION_SYSTEM = """You are a smart home intent parser. Given a user's natural language request about smart home devices, decompose it into individual **atomic intents**.

Each atomic intent represents exactly ONE action on ONE device in ONE room/workspace.

## Ontology Reference

The following ontology defines the valid workspace types, device (artifact) types, and command (action affordance) types available in the smart home domain. 
Use the **class local names** (e.g. ``TurnOnCommand``, ``Light``, ``Bathroom``) and the ``rdfs:comment`` descriptions to determine the best match for each intent.

```turtle
{ontology}
```

## Parsing Rules

1. Identify each distinct action the user wants to perform. A single sentence may contain multiple intents (e.g. "turn on the light and set brightness to 80" has two intents). Multiple sentences or clauses may also describe separate intents.
2. For each intent, determine:
   - **text_intent**: the natural language fragment from the original request that describes this atomic intent.
   - **action.affordance_type**: the ontology command class name that best matches the requested action (e.g. ``TurnOnCommand``, ``SetBrightnessCommand``).
   - **action.verb**: ``"set"`` if the action sets an attribute to a specific value or is parameterless (turn on/off, open/close, play, pause, stop, pack); ``"modify"`` if the action describes a relative change (increase by, decrease by, raise, lower, reduce, boost).
   - **target.artifact_type**: the ontology device class name (e.g. ``Light``, ``AirConditioner``, ``MediaPlayer``).
   - **target.workspace_type**: the ontology workspace class name (e.g. ``Bathroom``, ``Kitchen``, ``LivingRoom``).
3. Use EXACT class names from the ontology (case-sensitive).
4. If the user mentions a device or room that does not appear in the ontology, use the closest matching ontology class.
5. Produce intents in the order they appear in the user's request.
"""


class IntentExtractor:
    """Extracts structured intents from natural language goals using LLM + ontology."""

    def __init__(self, ontology_text: str):
        self.ontology_text = ontology_text

    def extract(
        self,
        goal: str,
        client,
        model_config: ModelConfig,
    ) -> tuple[list[StructuredIntent], dict]:
        """
        Extract structured intents from a user goal.

        Returns:
            (list of StructuredIntent, trace dict with LLM messages/response)
        """
        logger.info(f"Extracting intents from goal: {goal!r}")

        system_prompt = INTENT_EXTRACTION_SYSTEM.format(ontology=self.ontology_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        api_kwargs = get_model_kwargs(model_config.name, model_config=model_config)
        api_kwargs.update({
            "messages": messages,
            "tools": [EXTRACT_INTENTS_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "extract_intents"},
            },
        })

        trace = {
            "phase": "intent_extraction",
            "goal": goal,
            "system_prompt_length": len(system_prompt),
            "response": None,
            "error": None,
        }

        try:
            response = client.chat.completions.create(**api_kwargs)
            message = response.choices[0].message

            if not message.tool_calls:
                logger.warning("No tool call in intent extraction response")
                trace["error"] = f"No tool call: {message.content}"
                return [], trace

            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            trace["response"] = args

            raw_intents = args.get("intents", [])
            intents = []
            for i, raw in enumerate(raw_intents):
                intent = StructuredIntent.from_dict(raw, index=i)
                intents.append(intent)

            logger.info(f"Extracted {len(intents)} intents from goal")
            for intent in intents:
                logger.info(
                    f"  [{intent.original_index}] {intent.text_intent!r} "
                    f"-> {intent.affordance_type} on {intent.artifact_type} "
                    f"in {intent.workspace_type} (verb={intent.verb})"
                )

            return intents, trace

        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            trace["error"] = str(e)
            return [], trace
