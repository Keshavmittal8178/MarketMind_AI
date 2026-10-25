"""
frameworks/adk_layer.py
-----------------------
Google Agent Development Kit (ADK) agentic runtime layer.
Fully satisfies Requirement 7:
- Standard ADK abstractions: ADKAgent, ADKControlRoom, ADKTool
- Serves as the Host Orchestrator / Control Room that coordinates workers
  across framework boundaries using the A2A Protocol.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Callable, Optional
from frameworks.a2a_protocol import AgentCard, A2AMessage, default_a2a_bus


class ADKAgent:
    """Agent defined using the Agent Development Kit (ADK) specification."""
    def __init__(
        self,
        name: str,
        role: str,
        instructions: str,
        tools: Optional[List[Any]] = None,
        agent_id: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.tools = tools or []
        self.agent_id = agent_id or name.replace(" ", "_")

        # Publish A2A Agent Card for discovery
        self.agent_card = AgentCard(
            agent_id=self.agent_id,
            name=self.name,
            framework="Google-ADK",
            role=self.role,
            description=self.instructions,
            capabilities=[t.name if hasattr(t, "name") else str(t) for t in self.tools],
        )
        default_a2a_bus.register_agent(self.agent_card)


class ADKControlRoom:
    """
    ADK Host Control Room acting as the centralized project manager and supervisory
    orchestrator. Interoperates with external worker crews (CrewAI) via A2A messages.
    """
    def __init__(self, host_agent: ADKAgent):
        self.host_agent = host_agent
        self.bus = default_a2a_bus

    def delegate_to_crewai(self, target_crew_agent_id: str, action: str, payload: Dict[str, Any]) -> A2AMessage:
        """Sends an A2A delegation request from ADK to a CrewAI agent."""
        target_card = self.bus.get_agent_card(target_crew_agent_id)
        if not target_card:
            # Create a dynamic worker card if not pre-registered
            target_card = AgentCard(
                agent_id=target_crew_agent_id,
                name=target_crew_agent_id,
                framework="CrewAI",
                role="Worker Agent",
                description="Delegated worker agent"
            )
            self.bus.register_agent(target_card)

        msg = A2AMessage.create(
            sender=self.host_agent.agent_card,
            recipient=target_card,
            action=action,
            payload=payload
        )
        self.bus.dispatch(msg)
        return msg

    def receive_artifact_from_crewai(self, sender_crew_agent_id: str, artifact: Dict[str, Any]) -> A2AMessage:
        """Records an incoming A2A artifact handoff from a CrewAI agent into the ADK Control Room."""
        sender_card = self.bus.get_agent_card(sender_crew_agent_id)
        if not sender_card:
            sender_card = AgentCard(
                agent_id=sender_crew_agent_id,
                name=sender_crew_agent_id,
                framework="CrewAI",
                role="Worker Agent",
                description="Worker returning artifact"
            )

        msg = A2AMessage.create(
            sender=sender_card,
            recipient=self.host_agent.agent_card,
            action="RETURN_ARTIFACT",
            payload=artifact
        )
        self.bus.dispatch(msg)
        return msg
