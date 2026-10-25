"""
logger.py
---------
Observability layer for MarketMind AI.

Guideline requirement: "logs/traces per run, linkage to personas,
at least 40 runs". This module gives every agent invocation a row in a
local SQLite database (logs/observability.db) with:
    run_id, ticker, agent_name, persona, started_at, latency_ms,
    accuracy_score (LLM-as-judge, 0-1), output_summary, status

The Streamlit dashboard (dashboard/app.py) reads straight from this DB,
so there is a single source of truth for "what happened during each run".
"""

from __future__ import annotations
import sqlite3
import time
import os
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "observability.db")
DB_PATH = os.path.abspath(DB_PATH)


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ticker TEXT,
            agent_name TEXT NOT NULL,
            persona TEXT,
            started_at REAL,
            latency_ms REAL,
            accuracy_score REAL,
            status TEXT,
            output_summary TEXT
        )
        """
    )
    conn.commit()
    return conn


class AgentTimer:
    """Context manager: times one agent's execution and writes a log row.

    Usage:
        with AgentTimer(run_id, ticker, "DataScoutAgent", "Research Analyst") as t:
            result = do_work()
            t.set_output(str(result)[:300])
            t.set_accuracy(0.92)   # optional, e.g. from an LLM-as-judge call
    """

    def __init__(self, run_id: str, ticker: str, agent_name: str, persona: str):
        self.run_id = run_id
        self.ticker = ticker
        self.agent_name = agent_name
        self.persona = persona
        self._t0: Optional[float] = None
        self._output_summary = ""
        self._accuracy: Optional[float] = None
        self._status = "success"

    def set_output(self, text: str) -> None:
        self._output_summary = text

    def set_accuracy(self, score: float) -> None:
        self._accuracy = score

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self._t0) * 1000
        if exc_type is not None:
            self._status = "error"
            self._output_summary = f"{exc_type.__name__}: {exc_val}"
        conn = _connect()
        conn.execute(
            """INSERT INTO agent_runs
               (run_id, ticker, agent_name, persona, started_at, latency_ms,
                accuracy_score, status, output_summary)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id,
                self.ticker,
                self.agent_name,
                self.persona,
                self._t0,
                latency_ms,
                self._accuracy,
                self._status,
                self._output_summary,
            ),
        )
        conn.commit()
        conn.close()
        # do not suppress exceptions -- let the caller see them
        return False


def fetch_all_runs():
    conn = _connect()
    cur = conn.execute(
        "SELECT run_id, ticker, agent_name, persona, started_at, latency_ms, "
        "accuracy_score, status, output_summary FROM agent_runs ORDER BY started_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    cols = ["run_id", "ticker", "agent_name", "persona", "started_at", "latency_ms",
            "accuracy_score", "status", "output_summary"]
    return [dict(zip(cols, r)) for r in rows]


def count_runs() -> int:
    """Number of distinct end-to-end pipeline runs logged so far."""
    conn = _connect()
    cur = conn.execute("SELECT COUNT(DISTINCT run_id) FROM agent_runs")
    n = cur.fetchone()[0]
    conn.close()
    return n
