"""
services/api_gateway.py
------------------------
MarketMind AI: Enterprise Networked Microservices & A2A HTTP Gateway.
Satisfies Phase 2 Milestone (Distributed Networked A2A Protocol & REST/SSE Microservices).

Endpoints:
- GET  /.well-known/agent-card.json : Autonomous A2A Discovery.
- POST /api/v1/a2a/dispatch         : Cross-network A2AMessage dispatching.
- POST /api/v1/pipeline/run         : Full autonomous multi-agent pipeline over HTTP.
- GET  /api/v1/trading/portfolio    : Live portfolio equity, cash, and PnL.
- GET  /api/v1/trading/orders       : Human-in-the-Loop (HITL) pending orders queue.
- POST /api/v1/trading/approve/{id} : HITL trade confirmation & execution.
- POST /api/v1/trading/reject/{id}  : HITL trade cancellation.
- GET  /metrics                     : Prometheus metrics exporter for cluster observability.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from dataclasses import asdict
from frameworks.a2a_protocol import AgentCard, A2AMessage, default_a2a_bus
from agents.manager_agent import run_pipeline
from trading.broker_execution import BrokerExecutionEngine
from blackboard.redis_blackboard import RedisBlackboard

app = FastAPI(
    title="MarketMind AI - Distributed Microservice Gateway",
    description="Enterprise Networked Agent-to-Agent (A2A) and Algorithmic Trading API",
    version="2.0.0"
)

broker = BrokerExecutionEngine.get_instance()
a2a_bus = default_a2a_bus


class PipelineRequest(BaseModel):
    company_name: str = Field(..., example="Apple Inc")
    ticker: str = Field(..., example="AAPL")
    prompt_architecture: str = Field("ReAct", example="ReAct")
    advisor_persona: str = Field("Balanced Advisor", example="Balanced Advisor")
    auto_create_order: bool = Field(True, description="Automatically queue an order ticket in HITL")


class OrderActionRequest(BaseModel):
    custom_quantity: Optional[int] = None
    reason: Optional[str] = "Approved via API Gateway"


@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "MarketMind_AI_Gateway",
        "timestamp": time.time(),
        "phase": "Phase 2 Enterprise (100% Milestone)"
    }


@app.get("/.well-known/agent-card.json")
def a2a_discovery():
    """Satisfies standard A2A protocol discovery across networks."""
    cards = [
        AgentCard(
            agent_id="ADK_ManagerAgent",
            name="ManagerAgent",
            framework="Google-ADK",
            role="Control Room Orchestrator",
            description="Coordinates multi-wave parallel worker execution and Blackboard governance.",
            capabilities=["task_scheduling", "blackboard_governance", "report_synthesis"]
        ),
        AgentCard(
            agent_id="Crew_DataScoutAgent",
            name="DataScoutAgent",
            framework="CrewAI",
            role="Market Data & News Scout",
            description="Fetches live prices and financial news feeds via MCP tools.",
            capabilities=["mcp_market_data", "mcp_news_search"]
        ),
        AgentCard(
            agent_id="ADK_ForecastingAgent",
            name="ForecastingAgent",
            framework="Google-ADK",
            role="Quantitative Analyst",
            description="Executes statistical ARIMA and LSTM hybrid non-linear price projections.",
            capabilities=["arima_forecasting", "lstm_hybrid_projection"]
        ),
        AgentCard(
            agent_id="Crew_TopicModelingAgent",
            name="TopicModelingAgent",
            framework="CrewAI",
            role="Trend & Sentiment Analyst",
            description="Performs NMF topic clustering and sentiment polarity scoring on news.",
            capabilities=["nmf_topic_extraction", "vader_sentiment"]
        ),
        AgentCard(
            agent_id="ADK_RAGResearchAgent",
            name="RAGResearchAgent",
            framework="Google-ADK",
            role="Domain SEC Filing Expert",
            description="Retrieves grounded fundamental facts from corporate filings.",
            capabilities=["vector_retrieval", "10k_grounding"]
        ),
        AgentCard(
            agent_id="ADK_AdvisoryAgent",
            name="AdvisoryAgent",
            framework="Google-ADK",
            role="Senior Investment Advisor",
            description="Runs ReAct reasoning loop with self-reflection and HITL order generation.",
            capabilities=["react_reasoning", "reflection_loop", "hitl_order_generation"]
        )
    ]
    return {"a2a_version": "1.0", "registered_agents": [asdict(c) for c in cards]}


@app.post("/api/v1/a2a/dispatch")
def a2a_dispatch(msg: Dict[str, Any]):
    """Receives an A2AMessage over HTTP REST and posts it to the central A2A bus."""
    a2a_msg = A2AMessage(
        message_id=msg.get("message_id", "msg-001"),
        sender_agent_id=msg.get("sender_agent_id", "external_client"),
        sender_framework=msg.get("sender_framework", "REST_API"),
        recipient_agent_id=msg.get("recipient_agent_id", "ADK_ManagerAgent"),
        recipient_framework=msg.get("recipient_framework", "Google-ADK"),
        action=msg.get("action", "DELEGATE_TASK"),
        payload=msg.get("payload", {})
    )
    res = a2a_bus.dispatch(a2a_msg)
    return {"status": "DISPATCHED", "result": res, "timestamp": time.time()}


@app.post("/api/v1/pipeline/run")
def trigger_pipeline(req: PipelineRequest):
    """Executes the full multi-agent pipeline and optionally registers a HITL order ticket."""
    report = run_pipeline(
        company_name=req.company_name,
        ticker=req.ticker,
        prompt_architecture=req.prompt_architecture,
        advisor_persona=req.advisor_persona
    )

    order_ticket = None
    if req.auto_create_order and report.advisory.recommendation in ("BUY", "SELL"):
        order_ticket = broker.create_order_ticket_from_advisory(
            symbol=req.ticker,
            company_name=req.company_name,
            call=report.advisory.recommendation,
            current_price=report.quant.current_price,
            confidence=report.advisory.confidence,
            rationale=report.advisory.narrative
        )

    return {
        "report": report.model_dump(),
        "queued_order": order_ticket.model_dump() if order_ticket else None
    }


@app.get("/api/v1/trading/portfolio")
def get_portfolio():
    summary = broker.get_account_summary()
    positions = broker.get_positions()
    return {
        "account": summary.model_dump(),
        "positions": [p.model_dump() for p in positions]
    }


@app.get("/api/v1/trading/orders")
def get_pending_orders():
    orders = broker.get_pending_orders()
    return {"pending_orders": [o.model_dump() for o in orders], "total_pending": len(orders)}


@app.post("/api/v1/trading/approve/{order_id}")
def approve_order(order_id: str, req: OrderActionRequest):
    ok, msg = broker.approve_order(order_id, custom_qty=req.custom_quantity)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "SUCCESS", "message": msg}


@app.post("/api/v1/trading/reject/{order_id}")
def reject_order(order_id: str, req: OrderActionRequest):
    ok, msg = broker.reject_order(order_id, reason=req.reason or "Rejected by API Operator")
    return {"status": "REJECTED", "message": msg}


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible plain text metrics for Kubernetes & Grafana."""
    summary = broker.get_account_summary()
    lines = [
        "# HELP marketmind_portfolio_value Current total portfolio valuation in USD",
        "# TYPE marketmind_portfolio_value gauge",
        f"marketmind_portfolio_value {summary.portfolio_value}",
        "# HELP marketmind_cash_balance Available cash balance in USD",
        "# TYPE marketmind_cash_balance gauge",
        f"marketmind_cash_balance {summary.cash_balance}",
        "# HELP marketmind_pending_approvals Count of pending HITL trade orders",
        "# TYPE marketmind_pending_approvals gauge",
        f"marketmind_pending_approvals {summary.pending_approvals}",
        "# HELP marketmind_total_positions Number of active stock positions",
        "# TYPE marketmind_total_positions gauge",
        f"marketmind_total_positions {summary.total_positions}"
    ]
    return "\n".join(lines) + "\n"
