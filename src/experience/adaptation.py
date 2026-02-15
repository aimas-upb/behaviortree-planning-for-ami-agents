"""
Step 3: Matched Experience Adaptation.

Adapts a stored BT leaf-node experience to a new context by:
  1. Deterministic URL resolution — finds the target device in the
     current home's CapabilityModel.
  2. LLM-based parameter adaptation — adjusts parameters to match the
     new request and current environment state.
"""

import json
import logging
from typing import Optional

from ..config import ModelConfig, get_model_kwargs
from ..discovery.base import Artifact, CapabilityModel, DiscoveryResult, EnvironmentState
from .engine import ExperienceEntry
from .intent import StructuredIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON schema for parameter adaptation output
# ---------------------------------------------------------------------------

ADAPTATION_OUTPUT_SCHEMA = {
    "name": "adapt_action_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "parameters": {
                "type": "array",
                "description": (
                    "List of adapted parameter updates for the action. "
                    "Each item contains parameter name, value, and explanation."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "parameter_name": {
                            "type": "string",
                            "description": "Parameter name accepted by action schema.",
                        },
                        "parameter_value": {
                            "type": ["string", "number", "boolean", "null"],
                            "description": "Adapted parameter value.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation for this parameter update.",
                        },
                    },
                    "required": ["parameter_name", "parameter_value", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["parameters"],
        "additionalProperties": False,
    },
}


ADAPTATION_SYSTEM = """You are a parameter adapter for smart home actions.

You are given:
1. A previously successful action affordance (with its URL and parameter schema).
2. A new user request (natural language intent) that maps to the SAME type of action.
3. The current state of the target device (property values).

Your task: determine the correct **parameters** for the action to satisfy the new request.

## Rules

- Use the parameter schema to determine valid parameter names and value ranges.
- For absolute requests (e.g. "set brightness to 80"): use the stated value directly.
- For relative requests (e.g. "increase brightness by 20%"):
  - Read the current property value from the environment state.
  - Compute the new value (e.g. current + 20).
  - Clamp to the valid range from the schema.
- If the request has no parameters (e.g. "turn on"), return an empty parameters list [].
- Only include parameters that the action schema accepts.

## Output format (mandatory)

- Return ONLY a JSON object (no markdown, no prose outside JSON).
- The JSON must match this shape exactly:
    {{
        "parameters": [
            {{
                "parameter_name": "<parameter_name>",
                "parameter_value": <parameter_value>,
                "explanation": "<brief explanation for this parameter>"
            }}
        ]
    }}
- Put each adapted parameter in its own object in "parameters".
- Do not wrap parameter updates in tool-call syntax.

## Action Affordance

URL: {action_url}
Schema:
```json
{action_schema}
```

## Current Device State

{device_state}
"""


class ExperienceAdapter:
    """
    Adapts stored experience BT leaf nodes to new contexts.

    Phase 1 — URL resolution (deterministic):
      Match the intent's semantic types (artifact_type, workspace_type)
      against the current home's CapabilityModel to find the correct
      device instance and action URL.

    Phase 2 — Parameter adaptation (LLM-based):
      Call an LLM to adapt parameters using the intent text and the
      current environment state.
    """

    def adapt(
        self,
        experience: ExperienceEntry,
        intent: StructuredIntent,
        discovery_result: DiscoveryResult,
        client,
        model_config: ModelConfig,
    ) -> tuple[Optional[dict], dict]:
        """
        Adapt a stored experience to the current home context.

        Args:
            experience: The matched ExperienceEntry from the engine.
            intent: The target StructuredIntent from the current request.
            discovery_result: DiscoveryResult for the current home
                (must include affordances and state).
            client: OpenAI client.
            model_config: Model configuration.

        Returns:
            (adapted_json_ir, trace_dict)
            adapted_json_ir is a JSON-IR dict for the adapted action node,
            or None if adaptation failed.
        """
        trace: dict = {
            "phase": "experience_adaptation",
            "intent": intent.to_dict(),
            "experience_id": experience.id,
            "url_resolution": None,
            "param_adaptation": None,
            "error": None,
        }

        capability = discovery_result.affordances

        # Phase 1: URL resolution
        resolved = self._resolve_url(intent, capability, discovery_result.state)
        trace["url_resolution"] = resolved

        if resolved is None:
            msg = (
                f"Could not resolve device for intent "
                f"({intent.artifact_type} in {intent.workspace_type})"
            )
            logger.warning(msg)
            trace["error"] = msg
            return None, trace

        action_url = resolved["action_url"]
        action_schema = resolved["action_schema"]
        device_state = resolved["device_state"]

        # Phase 2: LLM parameter adaptation
        try:
            params, param_trace = self._adapt_parameters(
                action_url=action_url,
                action_schema=action_schema,
                device_state=device_state,
                intent=intent,
                client=client,
                model_config=model_config,
            )
            trace["param_adaptation"] = param_trace
        except Exception as e:
            logger.error(f"Parameter adaptation failed: {e}")
            trace["error"] = str(e)
            return None, trace

        # Build adapted JSON-IR leaf node
        adapted_ir: dict = {
            "type": "action",
            "name": f"Adapted_{intent.affordance_type}",
            "action_url": action_url,
            "semantic_type": intent.affordance_type,
        }
        if params:
            adapted_ir["parameters"] = params

        logger.info(
            f"Adapted experience for intent {intent.text_intent!r}: "
            f"url={action_url}, params={params}"
        )

        return adapted_ir, trace

    # ------------------------------------------------------------------
    # Phase 1: URL resolution
    # ------------------------------------------------------------------

    def _resolve_url(
        self,
        intent: StructuredIntent,
        capability: CapabilityModel,
        env_state: EnvironmentState,
    ) -> Optional[dict]:
        """
        Find the target device and action URL in the current home.

        Matches by ontology semantic types (case-insensitive):
          - workspace semantic_type == intent.workspace_type
          - artifact semantic_type == intent.artifact_type
          - action semantic_type == intent.affordance_type

        Returns dict with action_url, action_schema, device_state info,
        or None if no match.
        """
        ws_type_lower = intent.workspace_type.lower()
        art_type_lower = intent.artifact_type.lower()
        aff_type_lower = intent.affordance_type.lower()

        for ws_uri, workspace in capability.workspaces.items():
            # Match workspace by semantic type
            ws_sem = (workspace.semantic_type or "").lower()
            if ws_sem != ws_type_lower:
                continue

            for art_uri in workspace.artifact_uris:
                artifact = capability.artifacts.get(art_uri)
                if artifact is None:
                    continue

                # Match artifact by semantic type
                art_sem = (artifact.semantic_type or "").lower()
                if art_sem != art_type_lower:
                    continue

                # Match action by semantic type
                for action in artifact.actions:
                    action_sem = (action.semantic_type or "").lower()
                    if action_sem != aff_type_lower:
                        continue

                    state_lines = self._build_device_state(
                        artifact, env_state
                    )

                    return {
                        "action_url": action.uri,
                        "action_schema": action.schema,
                        "action_name": action.name,
                        "artifact_name": artifact.name,
                        "artifact_uri": art_uri,
                        "workspace_uri": ws_uri,
                        "device_state": state_lines,
                    }

        return None

    @staticmethod
    def _build_device_state(
        artifact: Artifact,
        env_state: EnvironmentState,
    ) -> str:
        """Build a text summary of the device's current property values."""
        lines: list[str] = []
        for prop in artifact.properties:
            value = env_state.get(prop.uri)
            if value is not None:
                lines.append(f"- {prop.name}: {value}")
            else:
                error = env_state.errors.get(prop.uri)
                if error:
                    lines.append(f"- {prop.name}: (error reading: {error})")
                else:
                    lines.append(f"- {prop.name}: (not available)")
        if not lines:
            return "No properties available for this device."
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 2: LLM parameter adaptation
    # ------------------------------------------------------------------

    def _adapt_parameters(
        self,
        action_url: str,
        action_schema: dict,
        device_state: str,
        intent: StructuredIntent,
        client,
        model_config: ModelConfig,
    ) -> tuple[dict, dict]:
        """
        Use an LLM call to adapt parameters for the resolved action.

        Returns (parameters_dict, trace_dict).
        """
        system_prompt = ADAPTATION_SYSTEM.format(
            action_url=action_url,
            action_schema=json.dumps(action_schema, indent=2),
            device_state=device_state,
        )

        user_message = (
            f"User request: {intent.text_intent}\n\n"
            f"Determine the correct parameters for the action."
        )

        logger.debug(
            f"Adaptation prompt:\n"
            f"  action_url: {action_url}\n"
            f"  action_schema: {json.dumps(action_schema, indent=2)}\n"
            f"  device_state: {device_state}\n"
            f"  user_message: {user_message}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        api_kwargs = get_model_kwargs(model_config.name, model_config=model_config)
        api_kwargs.update({
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": ADAPTATION_OUTPUT_SCHEMA,
            },
        })

        trace = {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": None,
            "error": None,
        }

        response = client.chat.completions.create(**api_kwargs)
        message = response.choices[0].message

        raw_content = message.content or ""
        try:
            args = json.loads(raw_content)
        except json.JSONDecodeError:
            trace["error"] = f"Invalid JSON adaptation response: {raw_content}"
            logger.warning("Invalid JSON in adaptation response")
            return {}, trace

        trace["response"] = args

        parameters_list = args.get("parameters", [])
        if not isinstance(parameters_list, list):
            trace["error"] = f"Invalid parameters payload: {parameters_list}"
            logger.warning("Adaptation response has non-list parameters")
            return {}, trace

        params: dict = {}
        for entry in parameters_list:
            if not isinstance(entry, dict):
                continue
            parameter_name = entry.get("parameter_name")
            if not isinstance(parameter_name, str) or not parameter_name.strip():
                continue
            params[parameter_name] = entry.get("parameter_value")

        trace["normalized_parameters"] = params

        return params, trace
