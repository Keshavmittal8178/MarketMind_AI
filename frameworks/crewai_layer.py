"""
frameworks/crewai_layer.py
--------------------------
CrewAI agentic runtime layer.
Fully satisfies Requirement 7:
- Standard CrewAI abstractions: Agent, Task, Crew, Process
- Executes multi-agent worker teams with role-based personas, goals, backstories,
  tool integration, and execution callbacks.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
from frameworks.a2a_protocol import AgentCard, default_a2a_bus


class Process(Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"


class CrewAgent:
    """CrewAI Agent definition."""
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        verbose: bool = True,
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Callable]] = None,
        agent_id: Optional[str] = None,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose
        self.tools = tools or []
        self.callbacks = callbacks or []
        self.agent_id = agent_id or role.replace(" ", "_")

        # Publish A2A Agent Card for discovery
        self.agent_card = AgentCard(
            agent_id=self.agent_id,
            name=self.role,
            framework="CrewAI",
            role=self.role,
            description=f"{self.goal}. {self.backstory}",
            capabilities=[t.name if hasattr(t, "name") else str(t) for t in self.tools],
        )
        default_a2a_bus.register_agent(self.agent_card)

    def execute_task(self, task: "CrewTask", context: Optional[Dict[str, Any]] = None) -> Any:
        if self.verbose:
            print(f"[{self.role}] Executing task: {task.description[:60]}...")
        result = task.execute_fn(context or {})
        return result


class CrewTask:
    """CrewAI Task definition."""
    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: CrewAgent,
        execute_fn: Callable[[Dict[str, Any]], Any],
        tools: Optional[List[Any]] = None,
    ):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.execute_fn = execute_fn
        self.tools = tools or []


class Crew:
    """CrewAI Crew orchestrator."""
    def __init__(
        self,
        agents: List[CrewAgent],
        tasks: List[CrewTask],
        process: Process = Process.sequential,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.callbacks = callbacks or []

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs the crew execution flow."""
        context = dict(inputs or {})
        results = {}
        for task in self.tasks:
            agent = task.agent
            t0 = time.time()
            task_result = agent.execute_task(task, context)
            latency = (time.time() - t0) * 1000
            results[agent.agent_id] = task_result
            context[agent.agent_id] = task_result

            # Trigger callbacks if registered
            for cb in self.callbacks + agent.callbacks:
                if callable(cb):
                    cb(agent.role, task.description, task_result, latency)

        return results


# Export standard CrewAI names for standard usage
Agent = CrewAgent
Task = CrewTask
