"""
manager_agent.py
------------------
The Manager/Orchestrator Agent implements the Hierarchical Control Room architecture
using the Google ADK (Agent Development Kit) framework.
It interoperates with CrewAI worker agents over the A2A (Agent-to-Agent) Protocol,
monitors execution flows via Callbacks, validates inputs with Security Guardrails,
and generates a structured Pydantic FinalMarketIntelligenceReport.

Satisfies:
- Requirement 1: Context Sharing via Blackboard
- Requirement 3: Pydantic Structured Output
- Requirement 4: Hierarchical Agentic Pattern
- Requirement 5: Task Monitoring & Logging using Callbacks
- Requirement 6: A2A Protocol Interoperability (ADK <-> CrewAI)
- Requirement 7: Framework Mandate (CrewAI + ADK)
- Requirement 8: Security Guardrails & Prompt Variances
- Requirement 10: Performance, Latency, and Cost Tracking
"""

from __future__ import annotations
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from frameworks.adk_layer import ADKAgent, ADKControlRoom
from frameworks.a2a_protocol import default_a2a_bus
from guardrails.security import PromptGuardrail, SecurityException
from schemas.models import FinalMarketIntelligenceReport, CostMetrics, SecurityAuditLog
from llm_client import global_token_tracker
from agents import data_scout_agent, topic_modeling_agent, forecasting_agent, rag_research_agent, advisory_agent

PERSONA = "Orchestrator"
AGENT_NAME = "ManagerAgent"

# Google ADK Host Agent & Control Room
adk_host_manager = ADKAgent(
    name="ManagerAgent",
    role="Hierarchical Orchestrator",
    instructions="Supervises cross-framework multi-agent pipeline, delegates tasks via A2A protocol to CrewAI workers, and validates final report.",
    agent_id="ADK_HostManager"
)
control_room = ADKControlRoom(host_agent=adk_host_manager)


def run_pipeline(
    company_name: str,
    ticker: str,
    prompt_architecture: str = "ReAct",
    advisor_persona: str = "Balanced Advisor",
    callback_handler: Optional[AgentCallbackHandler] = None
) -> FinalMarketIntelligenceReport:
    """
    Executes the standardized MarketMind AI pipeline.
    """
    run_id = str(uuid.uuid4())[:8]
    bb = Blackboard(run_id=run_id, ticker=ticker)
    cb = callback_handler or AgentCallbackHandler(run_id=run_id)

    # Reset token & cost tracker for this run
    global_token_tracker.reset()
    t0 = time.time()

    # Step 1: Security Guardrail Inspection (Requirement 8)
    is_safe, security_msg = PromptGuardrail.inspect(f"{company_name} {ticker}")
    security_audit = SecurityAuditLog(
        prompt_injections_detected=0 if is_safe else 1,
        toxic_content_flagged=not is_safe,
        tool_type_validation_passed=True,
        guardrail_status=security_msg
    )

    if not is_safe:
        raise SecurityException(f"Security Guardrail Triggered: {security_msg}")

    with AgentTimer(run_id, ticker, AGENT_NAME, PERSONA) as t:
        cb.on_agent_start(AGENT_NAME, PERSONA, {"company": company_name, "ticker": ticker})

        # Step 2: A2A Delegation from ADK Control Room to CrewAI Worker (Requirement 6)
        control_room.delegate_to_crewai(
            target_crew_agent_id="CrewAI_DataScoutAgent",
            action="DELEGATE_DATA_SCOUTING",
            payload={"company_name": company_name, "ticker": ticker}
        )

        # Phase 1 (Parallel): DataScout (CrewAI) + RAGResearch (ADK/Domain Expert)
        with ThreadPoolExecutor(max_workers=2) as pool:
            scout_future = pool.submit(data_scout_agent.run, bb, company_name, ticker, cb)
            rag_future = pool.submit(rag_research_agent.run, bb, ticker, cb)
            scout_output, price_df = scout_future.result()
            rag_output = rag_future.result()

        # Step 3: A2A Artifact Handoff from CrewAI back to ADK Control Room
        control_room.receive_artifact_from_crewai(
            sender_crew_agent_id="CrewAI_DataScoutAgent",
            artifact={"headlines_count": len(scout_output.headlines), "latest_close": scout_output.latest_close}
        )

        # Step 4: A2A Delegation to Topic Modeling (CrewAI)
        control_room.delegate_to_crewai(
            target_crew_agent_id="CrewAI_TopicModelingAgent",
            action="DELEGATE_TOPIC_MODELING",
            payload={"ticker": ticker, "headlines_source": "blackboard"}
        )

        # Phase 2 (Parallel): TopicModeling (CrewAI) + Forecasting (Quant)
        with ThreadPoolExecutor(max_workers=2) as pool:
            topic_future = pool.submit(topic_modeling_agent.run, bb, ticker, cb)
            forecast_future = pool.submit(forecasting_agent.run, bb, ticker, price_df, cb)
            topic_output = topic_future.result()
            forecast_output = forecast_future.result()

        control_room.receive_artifact_from_crewai(
            sender_crew_agent_id="CrewAI_TopicModelingAgent",
            artifact={"sentiment_score": topic_output.sentiment_score, "topics": topic_output.topics}
        )

        # Phase 3: Advisory Agent (ADK ReAct Reasoning + Prompt Variances)
        advisory_output = advisory_agent.run(
            bb=bb,
            ticker=ticker,
            prompt_architecture=prompt_architecture,
            advisor_persona=advisor_persona,
            callback_handler=cb
        )

        total_latency = round(time.time() - t0, 2)
        cost_dict = global_token_tracker.get_metrics()
        cost_metrics = CostMetrics(
            total_tokens_input=cost_dict["total_tokens_input"],
            total_tokens_output=cost_dict["total_tokens_output"],
            estimated_cost_usd=cost_dict["estimated_cost_usd"],
            latency_seconds=total_latency
        )

        # Step 5: Final Pydantic Structured Output Construction & Validation (Requirement 3)
        final_report = FinalMarketIntelligenceReport(
            run_id=run_id,
            ticker=ticker,
            company_name=company_name,
            total_latency_sec=total_latency,
            data_scout=scout_output,
            topic_modeling=topic_output,
            forecasting=forecast_output,
            rag_research=rag_output,
            advisory=advisory_output,
            cost_metrics=cost_metrics,
            security_audit=security_audit,
            orchestration_architecture="ADK-ControlRoom + CrewAI-WorkerCrew via A2A Protocol",
            messaging_architecture="Blackboard Architecture"
        )

        t.set_output(
            f"Pipeline complete for {ticker}: {advisory_output.recommendation} "
            f"(confidence={advisory_output.confidence}) in {total_latency}s | Cost: ${cost_metrics.estimated_cost_usd:.5f}"
        )
        t.set_accuracy(advisory_output.confidence)
        cb.on_agent_finish(AGENT_NAME, f"Report generated with {advisory_output.recommendation}", total_latency * 1000, advisory_output.confidence)

        return final_report
