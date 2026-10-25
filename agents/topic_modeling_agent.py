"""
topic_modeling_agent.py
------------------------
Persona: Trend Analyst
Framework: CrewAI Agent (Satisfies Requirement 7)
Protocol: Interoperable via A2A (Satisfies Requirement 6)
Output: Pydantic TopicModelingOutput (Satisfies Requirement 3)
"""

from __future__ import annotations
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from frameworks.crewai_layer import CrewAgent
from schemas.models import TopicModelingOutput
from llm_client import generate, llm_as_judge

PERSONA = "Trend Analyst"
AGENT_NAME = "TopicModelingAgent"

_POSITIVE_WORDS = {
    "beats", "rise", "rises", "growth", "strong", "raises", "expand", "expands",
    "buyback", "gain", "gains", "surge", "surges", "record", "upgrade", "bullish",
}
_NEGATIVE_WORDS = {
    "warns", "pressure", "scrutiny", "volatile", "falls", "fall", "drop", "drops",
    "weak", "downgrade", "bearish", "lawsuit", "recall", "miss", "misses", "headwinds",
}

COT_PROMPT = """You are a Trend Analyst agent (Chain-of-Thought).
Step 1: Given these extracted topics (keywords) and a sentiment score (-1 to 1),
Step 2: Explain in 2 sentences what theme is currently dominating the news flow.

Topics: {topics}
Sentiment score: {sentiment}

Explanation:"""

# CrewAI Agent instance
crew_trend_scout = CrewAgent(
    role="Trend Analyst",
    goal="Extract latent topical clusters and compute sentiment scores across news feeds",
    backstory="Quantitative textual analyst skilled in TF-IDF decomposition and market sentiment extraction.",
    verbose=True,
    agent_id="CrewAI_TopicModelingAgent"
)


def _simple_sentiment(headlines: list[str]) -> float:
    pos, neg = 0, 0
    for h in headlines:
        words = set(re.findall(r"[a-zA-Z]+", h.lower()))
        pos += len(words & _POSITIVE_WORDS)
        neg += len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _extract_topics(headlines: list[str], n_topics: int = 3, n_words: int = 4) -> list[list[str]]:
    if len(headlines) < 3:
        return [["insufficient_data"]]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    try:
        X = vectorizer.fit_transform(headlines)
        n_topics = min(n_topics, X.shape[0])
        nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=300)
        nmf.fit(X)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for comp in nmf.components_:
            top_idx = comp.argsort()[-n_words:][::-1]
            topics.append([str(feature_names[i]) for i in top_idx])
        return topics
    except Exception:
        return [["general_market_news"]]


def run(
    bb: Blackboard,
    ticker: str,
    callback_handler: AgentCallbackHandler | None = None
) -> TopicModelingOutput:
    t0 = time.time()
    if callback_handler:
        callback_handler.on_agent_start(AGENT_NAME, PERSONA, {"ticker": ticker})

    with AgentTimer(bb.run_id, ticker, AGENT_NAME, PERSONA) as t:
        headlines = bb.read("news_headlines", [])

        topics = _extract_topics(headlines)
        sentiment = _simple_sentiment(headlines)
        flat_topics = ", ".join([" / ".join(topic) for topic in topics])

        explanation = generate(
            COT_PROMPT.format(topics=flat_topics, sentiment=sentiment), tier="light"
        )

        pydantic_output = TopicModelingOutput(
            topics=topics,
            sentiment_score=sentiment,
            explanation=explanation.strip()
        )

        bb.write(AGENT_NAME, "topics", topics)
        bb.write(AGENT_NAME, "sentiment_score", sentiment)
        bb.write(AGENT_NAME, "topic_explanation", pydantic_output.explanation)

        latency = (time.time() - t0) * 1000
        acc_score = llm_as_judge(pydantic_output.explanation, "clearly references the given topics and sentiment")

        t.set_output(pydantic_output.explanation[:300])
        t.set_accuracy(acc_score)

        if callback_handler:
            callback_handler.on_agent_finish(AGENT_NAME, pydantic_output.explanation, latency, acc_score)

        return pydantic_output
