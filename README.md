# MarketMind AI
### Autonomous Multi-Agent Market Intelligence & Investment Advisory System
**Course:** Agentic AI · **Mentor:** Pranshu Tiwari  
**Team:** Tushar Gupta (230542) · Keshav (230548) · Gaurav (230566)

MarketMind AI is a standardized, production-grade Multi-Agent System designed for real-time market intelligence, quant price forecasting, news sentiment analysis, and grounded investment advisory.

---

## 🏆 Standardization Compliance Matrix (11/11 Requirements)

| # | Standardized Requirement | Implementation & Architectural Evidence | Primary Files |
|---|---|---|---|
| **1** | **Context Sharing & State Updates** | **Blackboard Architecture** (`blackboard.py`) for thread-safe state sharing. Compared empirically against Direct Mesh messaging (`messaging_comparison.py`). | [`blackboard/blackboard.py`](blackboard/blackboard.py), [`blackboard/messaging_comparison.py`](blackboard/messaging_comparison.py) |
| **2** | **Tool Integration using MCP** | Model Context Protocol Server & Client (`JSON-RPC 2.0`). Implements `tools/list` and `tools/call` for `mcp_market_data`, `mcp_news_search`, and `mcp_financial_calculator`. | [`mcp/mcp_server.py`](mcp/mcp_server.py), [`mcp/mcp_client.py`](mcp/mcp_client.py) |
| **3** | **Structured Output using Pydantic** | All agent findings and the final report are strictly validated via **Pydantic v2** models (`schemas/models.py`). Exportable as Markdown, Table, and JSON Schema. | [`schemas/models.py`](schemas/models.py) |
| **4** | **Agentic Design Patterns** | Hierarchical Orchestration (`ManagerAgent`), ReAct Pattern with Reflection/Retry loop (`AdvisoryAgent`), CoT Workers (`DataScout`, `TopicModeling`, `Forecasting`, `RAG`), and Blackboard Pattern. | [`agents/manager_agent.py`](agents/manager_agent.py), [`agents/advisory_agent.py`](agents/advisory_agent.py) |
| **5** | **Task Monitoring & Callbacks** | `AgentCallbackHandler` with explicit callback hooks: `on_agent_start`, `on_tool_start`, `on_tool_end`, `on_agent_step`, `on_agent_finish`, `on_agent_error`. | [`observability/callbacks.py`](observability/callbacks.py) |
| **6** | **A2A Protocol & Interoperability** | **Agent-to-Agent (A2A)** Protocol implementation with standard `.well-known/agent-card.json` specifications and `A2AMessage` task handoffs between ADK and CrewAI. | [`frameworks/a2a_protocol.py`](frameworks/a2a_protocol.py) |
| **7** | **CrewAI & Google ADK Frameworks** | Cross-framework architecture: Host Control Room built on **Google ADK**; Worker Data & Trend Agents built on **CrewAI**. | [`frameworks/adk_layer.py`](frameworks/adk_layer.py), [`frameworks/crewai_layer.py`](frameworks/crewai_layer.py) |
| **8** | **Security Guardrails & Variances** | `PromptGuardrail` for prompt injection & toxic input detection, data validation callbacks on tool outputs, and `PromptVarianceManager` (ReAct vs Few-Shot & Persona variations). | [`guardrails/security.py`](guardrails/security.py) |
| **9** | **Observability for Tracing & Provenance** | Central SQLite audit database (`observability.db`), ReAct Thought/Action/Observation/Reflection trace, and full Blackboard version history. | [`observability/logger.py`](observability/logger.py), [`dashboard/app.py`](dashboard/app.py) |
| **10** | **Performance, Latency & Cost Tracking** | Batch benchmarking runner (`run_batch.py`) with 50+ runs, per-agent latency in milliseconds, token counters, and USD cost estimation ($). | [`run_batch.py`](run_batch.py), [`llm_client.py`](llm_client.py) |
| **11** | **Cloud Deployment Ready** | Containerized with Docker, configured for Streamlit Community Cloud (`.streamlit/config.toml`), and documented in `DEPLOYMENT.md`. | [`Dockerfile`](Dockerfile), [`DEPLOYMENT.md`](DEPLOYMENT.md) |

---

## 🏛️ System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Google ADK Host Control Room             │
                    │                      (ManagerAgent)                    │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                          A2A Protocol (Agent Cards & Message Bus)
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌─────────────────────────┐           ┌───────────────────┐               ┌─────────────────────────┐
│     CrewAI Worker       │           │   Specialist ADK  │               │      ADK ReAct Agent    │
│       Agents Team       │           │   Domain Agents   │               │     (Investment Advisor)│
├─────────────────────────┤           ├───────────────────┤               ├─────────────────────────┤
│ • DataScoutAgent        │           │ • ForecastingAgent│               │ • Thought -> Action     │
│ • TopicModelingAgent    │           │ • RAGResearchAgent│               │ • Observation Loop      │
│ (Tools: MCP News/Price) │           │ (Tools: ARIMA/RAG)│               │ • Reflection/Retry loop │
└────────────┬────────────┘           └─────────┬─────────┘               └────────────▲────────────┘
             │                                  │                                      │
             └────────────────►  BLACKBOARD (Shared State)  ───────────────────────────┘
                                       ▲
                                       │ MCP Protocol (JSON-RPC 2.0)
                                       │
                      ┌────────────────┴────────────────┐
                      │    MCP Tool Server & Client     │
                      │  • mcp_market_data (yfinance)   │
                      │  • mcp_news_search (Google RSS) │
                      │  • mcp_financial_calculator     │
                      └─────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
cd MarketMind_AI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Single Analysis via CLI
```bash
# Standard ReAct execution with Markdown report output:
python main.py --company "Apple Inc" --ticker AAPL --format markdown

# Few-Shot Prompt Architecture Variance with Aggressive Growth Persona:
python main.py --company "Tesla Inc" --ticker TSLA --prompt-arch Few-Shot --persona "Aggressive Growth" --format table

# Save full Pydantic JSON schema output:
python main.py --company "Microsoft" --ticker MSFT --save-json output.json
```

### 3. Generate 40+ Evaluation Runs with Prompt Variances
```bash
python run_batch.py --passes 4
```
Runs 12 companies across 4 passes with alternating prompt architectures (ReAct vs Few-Shot) and persona variations, calculating latency, cost, and accuracy into SQLite.

### 4. Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Explore 6 comprehensive tabs:
1. **Live Analysis & Pydantic Report**: Interactive runs, Pydantic schema exports.
2. **A2A Protocol & Frameworks**: Live inspection of ADK ↔ CrewAI Agent Cards and Message Ledger.
3. **MCP Tools Console**: Direct Model Context Protocol tool execution.
4. **Security Guardrails**: Adversarial prompt injection defense sandbox.
5. **Observability & Cost**: 50+ runs logged, per-agent latency & cost analytics.
6. **Messaging Architecture Comparison**: Blackboard vs Direct Peer-to-Peer mesh.

---

## ☁️ Cloud Deployment

MarketMind AI is ready for 1-click cloud deployment:
- **Streamlit Community Cloud:** Connect GitHub repo and deploy `dashboard/app.py`.
- **Docker Container:**
  ```bash
  docker build -t marketmind-ai .
  docker run -p 8501:8501 marketmind-ai
  ```
See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

---

## 🛡️ Security Guardrails & Data Validation

MarketMind AI incorporates real-time security defenses:
- **Prompt Injection Defense:** Intercepts jailbreak attempts, system override tokens, and unauthorized persona hijacking.
- **Toxic & Abusive Content Filter:** Rejects harmful commands and malicious payloads.
- **Data Validation Callbacks:** Enforces type and range constraints on tool outputs (positive price checks, non-empty news arrays).
