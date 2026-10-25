"""
observability/callbacks.py
--------------------------
Callback infrastructure for task monitoring, logging, and error tracking.
Fully satisfies Requirement 5:
- Implements callback functions that hook into intermediate outputs, tool runs,
  execution flows, and errors for end-to-end transparency.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List, Optional, Callable


class CallbackEvent:
    def __init__(self, event_type: str, agent_name: str, data: Dict[str, Any]):
        self.timestamp = time.time()
        self.event_type = event_type
        self.agent_name = agent_name
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "data": self.data
        }


class AgentCallbackHandler:
    """Central callback handler that monitors and logs multi-agent execution flows."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(time.time())
        self.events: List[CallbackEvent] = []

    def on_agent_start(self, agent_name: str, persona: str, inputs: Dict[str, Any]):
        """Triggered before an agent begins executing."""
        event = CallbackEvent("AGENT_START", agent_name, {
            "persona": persona,
            "inputs": {k: str(v)[:150] for k, v in inputs.items()}
        })
        self.events.append(event)
        print(f"  [Callback: START] Agent='{agent_name}' ({persona})")

    def on_tool_start(self, agent_name: str, tool_name: str, arguments: Dict[str, Any]):
        """Triggered immediately before a tool is invoked."""
        event = CallbackEvent("TOOL_START", agent_name, {
            "tool_name": tool_name,
            "arguments": arguments
        })
        self.events.append(event)
        print(f"    [Callback: TOOL_START] Tool='{tool_name}' Args={arguments}")

    def on_tool_end(self, agent_name: str, tool_name: str, result_summary: str, is_error: bool = False):
        """Triggered after tool completion."""
        event = CallbackEvent("TOOL_END", agent_name, {
            "tool_name": tool_name,
            "result_summary": result_summary[:200],
            "is_error": is_error
        })
        self.events.append(event)
        status_tag = "ERROR" if is_error else "SUCCESS"
        print(f"    [Callback: TOOL_END] Tool='{tool_name}' Status={status_tag}")

    def on_agent_step(self, agent_name: str, step_type: str, details: str):
        """Triggered on intermediate reasoning steps (e.g. ReAct Thought/Action/Observation/Reflection)."""
        event = CallbackEvent("AGENT_STEP", agent_name, {
            "step_type": step_type,
            "details": details[:300]
        })
        self.events.append(event)

    def on_agent_finish(self, agent_name: str, output_summary: str, latency_ms: float, accuracy_score: Optional[float] = None):
        """Triggered when an agent successfully completes its task."""
        event = CallbackEvent("AGENT_FINISH", agent_name, {
            "output_summary": output_summary[:300],
            "latency_ms": round(latency_ms, 2),
            "accuracy_score": accuracy_score
        })
        self.events.append(event)
        print(f"  [Callback: FINISH] Agent='{agent_name}' Latency={latency_ms:.1f}ms")

    def on_agent_error(self, agent_name: str, error: Exception):
        """Triggered if an agent raises an unhandled exception."""
        event = CallbackEvent("AGENT_ERROR", agent_name, {
            "error_type": type(error).__name__,
            "error_message": str(error)
        })
        self.events.append(event)
        print(f"  [Callback: ERROR] Agent='{agent_name}' Error={type(error).__name__}: {error}")

    def get_event_log(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
