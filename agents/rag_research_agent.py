"""
rag_research_agent.py
-----------------------
Persona: Domain Expert
Prompt Strategy: Chain-of-Thought (CoT), grounded via RAG.
Tool: TF-IDF VectorStore (rag/knowledge_base.py).
Output: Pydantic RAGResearchOutput (Satisfies Requirement 3)
"""

from __future__ import annotations
import time
from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from tools.price_data import get_company_profile
from rag.knowledge_base import VectorStore
from schemas.models import RAGResearchOutput, GroundedQAPair
from llm_client import generate, llm_as_judge

PERSONA = "Domain Expert"
AGENT_NAME = "RAGResearchAgent"

QUESTIONS = [
    "What sector and industry does the company operate in?",
    "What is the company's current valuation (market cap, P/E ratio)?",
    "What does the company's business model involve?",
]

RAG_PROMPT = """You are a Domain Expert agent. Answer ONLY using the retrieved context below
(Chain-of-Thought: read context -> extract relevant facts -> answer concisely).
Do not invent facts that are not present in the context.

Question: {question}
Retrieved context:
{context}

Grounded answer:"""


def run(
    bb: Blackboard,
    ticker: str,
    callback_handler: AgentCallbackHandler | None = None
) -> RAGResearchOutput:
    t0 = time.time()
    if callback_handler:
        callback_handler.on_agent_start(AGENT_NAME, PERSONA, {"ticker": ticker, "questions_count": len(QUESTIONS)})

    with AgentTimer(bb.run_id, ticker, AGENT_NAME, PERSONA) as t:
        profile, profile_live = get_company_profile(ticker)
        store = VectorStore.from_company_profile(profile)

        qa_pairs: list[GroundedQAPair] = []
        for q in QUESTIONS:
            if callback_handler:
                callback_handler.on_tool_start(AGENT_NAME, "tfidf_rag_retriever", {"query": q})
            context_chunks = store.retrieve(q, top_k=2)
            if callback_handler:
                callback_handler.on_tool_end(AGENT_NAME, "tfidf_rag_retriever", f"Chunks: {len(context_chunks)}")

            context = "\n".join(context_chunks) if context_chunks else "(no relevant context retrieved)"
            answer = generate(RAG_PROMPT.format(question=q, context=context), tier="light")
            qa_pairs.append(GroundedQAPair(question=q, context_used=context_chunks, answer=answer.strip()))

        pydantic_output = RAGResearchOutput(
            profile_source_live=profile_live,
            company_profile=profile,
            grounded_qa=qa_pairs
        )

        bb.write(AGENT_NAME, "company_profile", profile)
        bb.write(AGENT_NAME, "grounded_qa", [qa.model_dump() for qa in qa_pairs])

        combined_answers = " ".join(p.answer for p in qa_pairs)
        latency = (time.time() - t0) * 1000
        acc_score = llm_as_judge(combined_answers, "answers are grounded in the retrieved context, not invented")

        t.set_output(combined_answers[:300])
        t.set_accuracy(acc_score)

        if callback_handler:
            callback_handler.on_agent_finish(AGENT_NAME, combined_answers[:300], latency, acc_score)

        return pydantic_output
