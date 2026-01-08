"""
Tracing module for agent activity logging.

Captures all agent interactions for later visualization and analysis.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from enum import Enum


class TraceEventType(Enum):
    """Types of trace events."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    CAPABILITY_MODEL_START = "capability_model_start"
    CAPABILITY_MODEL_END = "capability_model_end"
    CAPABILITY_MODEL_SUMMARY = "capability_model_summary"  # Full model for analysis
    WORKSPACE_DISCOVERED = "workspace_discovered"
    ARTIFACT_DISCOVERED = "artifact_discovered"
    # Discovery conversation events (for agentic mode)
    DISCOVERY_LLM_TURN = "discovery_llm_turn"  # LLM response with tool calls
    DISCOVERY_TOOL_CALL = "discovery_tool_call"  # Individual tool call
    DISCOVERY_TOOL_RESULT = "discovery_tool_result"  # Tool result
    # BT generation events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_PROMPT_CONTENT = "llm_prompt_content"  # Full prompt for analysis
    BT_SPEC_GENERATED = "bt_spec_generated"
    BT_COMPILED = "bt_compiled"
    BT_TICK = "bt_tick"
    BT_EXECUTION_END = "bt_execution_end"
    ERROR = "error"


@dataclass
class TraceEvent:
    """A single trace event."""
    event_type: str
    timestamp: str
    elapsed_ms: float
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentTrace:
    """Complete trace of an agent run."""
    trace_id: str
    goal: str
    model: str
    entry_point: str
    start_time: str
    end_time: Optional[str] = None
    success: Optional[bool] = None
    events: list[TraceEvent] = field(default_factory=list)
    ablation_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "model": self.model,
            "entry_point": self.entry_point,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "success": self.success,
            "ablation_config": self.ablation_config,
            "events": [e.to_dict() for e in self.events],
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        event_counts = {}
        for e in self.events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        total_time = 0
        if self.events:
            total_time = sum(e.elapsed_ms for e in self.events)

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "total_time_ms": total_time,
        }


class Tracer:
    """
    Tracer for capturing agent activity.

    Usage:
        tracer = Tracer(enabled=True)
        tracer.start_trace("Turn on light", model="gpt-4o", entry_point="...")

        tracer.log(TraceEventType.LLM_REQUEST, {"prompt": "..."})
        tracer.log(TraceEventType.LLM_RESPONSE, {"response": "..."})

        tracer.end_trace(success=True)
        tracer.save("traces/")
    """

    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.current_trace: Optional[AgentTrace] = None
        self._start_time: float = 0
        self._last_event_time: float = 0

    def start_trace(
        self,
        goal: str,
        model: str,
        entry_point: str,
        ablation_config: Optional[dict] = None,
    ) -> None:
        """Start a new trace."""
        if not self.enabled:
            return

        self._start_time = time.time()
        self._last_event_time = self._start_time

        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.current_trace = AgentTrace(
            trace_id=trace_id,
            goal=goal,
            model=model,
            entry_point=entry_point,
            start_time=datetime.now().isoformat(),
            ablation_config=ablation_config or {},
        )

        self.log(TraceEventType.AGENT_START, {
            "goal": goal,
            "model": model,
        })

    def log(self, event_type: TraceEventType, data: Optional[dict] = None) -> None:
        """Log a trace event."""
        if not self.enabled or not self.current_trace:
            return

        now = time.time()
        elapsed = (now - self._last_event_time) * 1000  # ms
        self._last_event_time = now

        event = TraceEvent(
            event_type=event_type.value,
            timestamp=datetime.now().isoformat(),
            elapsed_ms=round(elapsed, 2),
            data=data or {},
        )

        self.current_trace.events.append(event)

        if self.verbose:
            print(f"  [TRACE] {event_type.value}: {elapsed:.1f}ms")

    def end_trace(self, success: bool, result: Optional[str] = None) -> None:
        """End the current trace."""
        if not self.enabled or not self.current_trace:
            return

        self.log(TraceEventType.AGENT_END, {
            "success": success,
            "result": result,
        })

        self.current_trace.end_time = datetime.now().isoformat()
        self.current_trace.success = success

    def save(self, output_dir: str = "traces") -> Optional[str]:
        """Save trace to JSON file."""
        if not self.enabled or not self.current_trace:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"trace_{self.current_trace.trace_id}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(self.current_trace.to_dict(), f, indent=2)

        return str(filepath)

    def get_trace(self) -> Optional[dict]:
        """Get current trace as dict."""
        if not self.current_trace:
            return None
        return self.current_trace.to_dict()


# Global tracer instance (can be replaced)
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get or create global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(enabled=False)
    return _global_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set global tracer."""
    global _global_tracer
    _global_tracer = tracer
