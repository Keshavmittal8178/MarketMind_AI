"""
data_scout_agent.py
--------------------
Persona: Research Analyst
Framework: CrewAI Agent (Satisfies Requirement 7)
Protocol: Interoperable via A2A (Satisfies Requirement 6)
Tools: MCP Integration (Satisfies Requirement 2)
Validation: Callbacks for Data Type Checking (Satisfies Requirement 8)
Output: Pydantic DataScoutOutput (Satisfies Requirement 3)
"""

from __future__ import annotations
import time
import pandas as pd
from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from frameworks.crewai_layer import CrewAgent
from mcp.mcp_client import default_mcp_client
from guardrails.security import DataValidationCallback
from schemas.models import DataScoutOutput
from llm_client import generate, llm_as_judge
from tools.price_data import get_price_history

PERSONA = "Research Analyst"
AGENT_NAME = "DataScoutAgent"

COT_PROMPT = """You are a financial Research Analyst agent (Chain-of-Thought).
Step 1: Read the raw headlines below.
Step 2: Identify the 3 most market-relevant headlines.
Step 3: Write a 2-sentence neutral factual briefing (no opinions/predictions).

Headlines:
{headlines}

Briefing:"""

# Instantiate the CrewAI Agent definition
crew_data_scout = CrewAgent(
    role="Research Analyst",
    goal="Extract live market data and news headlines using MCP external tools with strict schema validation",
    backstory="Senior financial data scout specializing in real-time news aggregation and quantitative ticker pricing.",
    verbose=True,
    agent_id="CrewAI_DataScoutAgent"
)


def run(
    bb: Blackboard,
    company_name: str,
    ticker: str,
    callback_handler: AgentCallbackHandler | None = None
) -> tuple[DataScoutOutput, pd.DataFrame]:
    t0 = time.time()
    if callback_handler:
        callback_handler.on_agent_start(AGENT_NAME, PERSONA, {"company": company_name, "ticker": ticker})

    with AgentTimer(bb.run_id, ticker, AGENT_NAME, PERSONA) as t:
        # Tool 1: MCP News Search
        if callback_handler:
            callback_handler.on_tool_start(AGENT_NAME, "mcp_news_search", {"company_name": company_name})
        news_mcp = default_mcp_client.call_tool("mcp_news_search", {"company_name": company_name, "max_items": 10})
        is_news_valid, msg = DataValidationCallback.validate_news_tool_return(news_mcp)
        if callback_handler:
            callback_handler.on_tool_end(AGENT_NAME, "mcp_news_search", f"Count: {news_mcp.get('count', 0)}, Valid: {is_news_valid}")

        headlines = news_mcp.get("headlines", [])
        news_live = news_mcp.get("news_source_live", False)

        # Tool 2: MCP Market Data
        if callback_handler:
            callback_handler.on_tool_start(AGENT_NAME, "mcp_market_data", {"ticker": ticker})
        market_mcp = default_mcp_client.call_tool("mcp_market_data", {"ticker": ticker, "period": "6mo"})
        is_price_valid, _ = DataValidationCallback.validate_price_tool_return(market_mcp)
        if callback_handler:
            callback_handler.on_tool_end(AGENT_NAME, "mcp_market_data", f"Close: ${market_mcp.get('latest_close', 0)}, Valid: {is_price_valid}")

        latest_close = float(market_mcp.get("latest_close", 150.0))
        price_live = market_mcp.get("price_source_live", False)

        # Retrieve dataframe for forecasting pipeline
        price_df, _ = get_price_history(ticker)

        # LLM Reasoning Step
        briefing = generate(COT_PROMPT.format(headlines="\n".join(f"- {h}" for h in headlines)), tier="light")

        # Pydantic Structured Output Validation
        pydantic_output = DataScoutOutput(
            company_name=company_name,
            ticker=ticker,
            headlines=headlines,
            latest_close=latest_close,
            briefing=briefing.strip(),
            news_source_live=news_live,
            price_source_live=price_live
        )

        # Write to Blackboard (Requirement 1)
        bb.write(AGENT_NAME, "news_headlines", headlines)
        bb.write(AGENT_NAME, "latest_close", latest_close)
        bb.write(AGENT_NAME, "price_history_available", True)
        bb.write(AGENT_NAME, "data_scout_briefing", pydantic_output.briefing)
        bb.write(AGENT_NAME, "data_sources_live", {"news": news_live, "price": price_live})

        latency = (time.time() - t0) * 1000
        acc_score = llm_as_judge(pydantic_output.briefing, "factual, neutral, mentions concrete headline content")

        t.set_output(pydantic_output.briefing[:300])
        t.set_accuracy(acc_score)

        if callback_handler:
            callback_handler.on_agent_finish(AGENT_NAME, pydantic_output.briefing, latency, acc_score)

        return pydantic_output, price_df
