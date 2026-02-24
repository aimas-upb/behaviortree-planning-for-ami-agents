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
    affordance_type: str  # e.g. "ex:SetBrightnessCommand"
    verb: str  # "set" or "modify"
    artifact_type: str  # e.g. "ex:Light"
    workspace_type: str  # e.g. "ex:Bathroom"
    original_index: int = 0
    # Neuro-symbolic fields (may be None for intents without parameter/value)
    parameter: Optional[str] = None   # e.g. "brightness", "mode"; None if parameterless
    value: Optional[str] = None       # e.g. "63", "heat", "auto"; None if not extracted

    def slot_key(self) -> tuple:
        """Slot-matching key: (affordance_type, artifact_type, workspace_type)."""
        return (self.affordance_type, self.artifact_type, self.workspace_type)

    def is_fully_structured(self) -> bool:
        """Return True if all neuro-symbolic fields are present (or parameter is None for parameterless commands)."""
        # affordance_type, artifact_type, workspace_type must always be set
        if not self.affordance_type or not self.artifact_type or not self.workspace_type:
            return False
        # value must be set (even parameterless actions still need the verb resolved)
        # For parameterless commands (parameter is None), value is not required
        return True

    def to_dict(self) -> dict:
        return {
            "text_intent": self.text_intent,
            "action": {
                "affordance_type": self.affordance_type,
                "verb": self.verb,
                "parameter": self.parameter,
                "value": self.value,
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
            parameter=action.get("parameter"),
            value=action.get("value"),
        )


# ---------------------------------------------------------------------------
# LLM tool definition for intent extraction
# ---------------------------------------------------------------------------

EXTRACT_INTENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_intents",
        "description": "Extract structured atomic intents from a user goal about smart home devices.",
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
                                    "The natural language fragment from the original request "
                                    "that describes this single atomic intent."
                                ),
                            },
                            "action": {
                                "type": "object",
                                "properties": {
                                    "affordance_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology command class in namespace:localname format, "
                                            "e.g. ex:TurnOnCommand, ex:SetBrightnessCommand. "
                                            "Must be a subclass of "
                                            "ActionAffordance from the ontology."
                                        ),
                                    },
                                    "parameter": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "The name of the parameter that the affordance_type "
                                            "requires as payload (e.g. 'brightness', 'mode', "
                                            "'temperature'). Set to null if the command is "
                                            "parameterless (e.g. TurnOnCommand, CloseCommand). "
                                            "Refer to the rdfs:comment of the ActionAffordance "
                                            "subclass in the ontology."
                                        ),
                                    },
                                    "value": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "The value to set or the amount by which to modify "
                                            "the parameter. Extract only the numeric or enum "
                                            "value as a string, NOT measurement units "
                                            "(e.g. '63' not '63%', '24' not '24 degrees'). "
                                            "Set to null if no value is mentioned or if the "
                                            "command is parameterless."
                                        ),
                                    },
                                    "verb": {
                                        "type": "string",
                                        "enum": ["set", "modify"],
                                        "description": (
                                            "'set' if the action sets an attribute to a specific "
                                            "value or is parameterless (turn on/off, open/close, "
                                            "play, pause, stop, pack). "
                                            "'modify' if the action describes a relative change "
                                            "(increase by, decrease by, raise, lower, reduce, boost)."
                                        ),
                                    },
                                },
                                "required": ["affordance_type", "parameter", "value", "verb"],
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                    "artifact_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology device class in namespace:localname format, "
                                            "e.g. ex:Light, ex:Fan, "
                                            "AirConditioner. Must be a subclass of Artifact "
                                            "from the ontology."
                                        ),
                                    },
                                    "workspace_type": {
                                        "type": "string",
                                        "description": (
                                            "Ontology workspace class in namespace:localname format, "
                                            "e.g. ex:Bathroom, ex:Kitchen, ex:LivingRoom. "
                                            "Must be a subclass of "
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

An **atomic intent** is an intent that is fully independent of any other intent — it can be acted upon without knowing the outcome or state of any other intent.

## Atomicity Rules

- A request phrasing like "Increase the brightness of the lights in the master bedroom by 63%, set the air conditioner in the guest bedroom to auto swing mode, switch the air conditioner in the study room to heat mode" results in **three** atomic intents, because each sub-request targets a different device/room and is fully independent.
- A phrasing like "Adjust the air conditioner in the living room to 24 degrees **whenever** the fan in the kitchen is set to low speed" results in a **single** atomic intent (conditional/compound phrasing — cannot split without losing the conditioning relationship).
- Coordinate conjunctions ("and", "also", "then") between actions on different devices or rooms almost always signal separate atomic intents.
- Conditional, causal, or temporal subordination ("whenever", "if", "once", "after", "until") means the whole clause is a single compound intent.

## Ontology Reference

The following ontology defines the valid workspace types, device (artifact) types, and command (action affordance) types available in the smart home domain.
Use **namespace:localname identifiers** (e.g. ``ex:TurnOnCommand``, ``ex:Light``, ``ex:Bathroom``) and the ``rdfs:comment`` descriptions to determine the best match for each intent.

```turtle
{ontology}
```

## Parsing Rules

1. Identify each distinct, independent action the user wants to perform. Multiple coordinated actions on different devices/rooms are separate atomic intents.
2. For each atomic intent, extract ALL of the following fields:
   - **text_intent**: the natural language fragment from the original request that describes this atomic intent.
    - **action.affordance_type**: the ontology command class in namespace:localname format that best matches the requested action (e.g. ``ex:TurnOnCommand``, ``ex:SetBrightnessCommand``).
   - **action.parameter**: the name of the parameter that the ``affordance_type`` requires as payload (e.g. ``brightness``, ``mode``). Determine this from the ``rdfs:comment`` of the ``ActionAffordance`` subclass in the ontology. Set to ``null`` if the command is parameterless (e.g. ``TurnOnCommand``, ``CloseCommand``).
   - **action.value**: the value to set the parameter to, or the amount by which to modify it. Extract only the numeric or enum value as a string, **not** the measurement units (e.g. ``"63"`` not ``"63%"``, ``"24"`` not ``"24 degrees"``). Set to ``null`` if no value is mentioned or if the command is parameterless.
   - **action.verb**: ``"set"`` if the action sets an attribute to a specific value or is parameterless (turn on/off, open/close, play, pause, stop, pack); ``"modify"`` if the action describes a relative change (increase by, decrease by, raise, lower, reduce, boost).
    - **target.artifact_type**: the ontology device class in namespace:localname format (e.g. ``ex:Light``, ``ex:AirConditioner``, ``ex:MediaPlayer``).
    - **target.workspace_type**: the ontology workspace class in namespace:localname format (e.g. ``ex:Bathroom``, ``ex:Kitchen``, ``ex:LivingRoom``).
3. Use EXACT namespace:localname identifiers from the ontology prefixes and class names (case-sensitive).
4. Do not output bare local names for these fields; always include the namespace prefix.
5. If the user mentions a device or room that does not appear in the ontology, use the closest matching ontology class.
6. Produce intents in the order they appear in the user's request.
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
                    f"in {intent.workspace_type} (verb={intent.verb}, "
                    f"parameter={intent.parameter!r}, value={intent.value!r})"
                )

            return intents, trace

        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            trace["error"] = str(e)
            return [], trace
