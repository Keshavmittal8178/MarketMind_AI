"""
blackboard.py
--------------
Implements the Blackboard communication protocol for MarketMind AI.

WHY BLACKBOARD (and not Peer-to-Peer / MCP / Broadcast)?
- We have 6 agents that all need to read/write a common, evolving picture
  of "what do we currently know about company X".
- Peer-to-Peer would need up to N*(N-1) point-to-point channels between
  6 agents (30 possible directed links) -- unnecessary complexity.
- Broadcast / MCP-style tool-calling adds protocol overhead for agents
  that mostly need to READ a shared state, not call remote tools.
- A single shared Blackboard object lets every agent post its findings
  under its own namespaced key, and read any other agent's findings,
  with the Manager/Orchestrator Agent controlling turn-taking.

Every write is timestamped and versioned so the full history of how the
"belief state" evolved during one run can be inspected -- this is what
we show in the observability dashboard.
"""

from __future__ import annotations
import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BlackboardEntry:
    agent: str
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)


class Blackboard:
    """A shared, in-memory market-data board for one analysis run."""

    def __init__(self, run_id: str | None = None, ticker: str | None = None):
        self.run_id: str = run_id or str(uuid.uuid4())[:8]
        self.ticker: str = ticker or "UNKNOWN"
        self.created_at: float = time.time()
        self._store: Dict[str, Any] = {}
        self._history: List[BlackboardEntry] = []
        self._lock = threading.Lock()

    # ---- core read/write API -------------------------------------------------
    def write(self, agent: str, key: str, value: Any) -> None:
        """An agent posts a finding onto the board. Thread-safe, since worker
        agents that don't depend on each other now run concurrently."""
        with self._lock:
            self._store[key] = value
            self._history.append(BlackboardEntry(agent=agent, key=key, value=value))

    def read(self, key: str, default: Any = None) -> Any:
        """Any agent can read any other agent's posted finding."""
        return self._store.get(key, default)

    def read_all(self) -> Dict[str, Any]:
        return dict(self._store)

    def history(self) -> List[Dict[str, Any]]:
        return [
            {"agent": e.agent, "key": e.key, "value": e.value, "timestamp": e.timestamp}
            for e in self._history
        ]

    # ---- persistence (useful for the dashboard / debugging) ------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "ticker": self.ticker,
                "created_at": self.created_at,
                "state": self._store,
                "history": self.history(),
            },
            default=str,
            indent=2,
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
