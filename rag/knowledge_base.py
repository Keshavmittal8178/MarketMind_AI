"""
knowledge_base.py
------------------
A minimal, dependency-light Retrieval-Augmented Generation (RAG) store.

Documents = chunked company profile / business-summary text (pulled from
yfinance's company info -- the closest free equivalent to a filing/
annual-report excerpt, with an offline placeholder if unavailable).

Retrieval = TF-IDF cosine similarity (scikit-learn). We use TF-IDF
instead of a neural embedding model (e.g. sentence-transformers)
because it needs no model download / GPU / external embedding API --
which keeps RAG fully local, fast, and 100% explainable for a viva
("here is exactly which terms matched"), while still being genuine
retrieval-augmented generation: retrieve top-k relevant chunks, then
ground the LLM's answer in them.
"""

from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunk_text(text: str, chunk_size: int = 220) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


class VectorStore:
    def __init__(self, documents: list[str]):
        self.chunks = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks) if self.chunks else None

    @classmethod
    def from_company_profile(cls, profile: dict) -> "VectorStore":
        base_text = (
            f"{profile.get('name')} operates in the {profile.get('sector')} sector, "
            f"specifically the {profile.get('industry')} industry. "
            f"Market capitalisation: {profile.get('market_cap')}. "
            f"Trailing P/E ratio: {profile.get('pe_ratio')}. "
            f"{profile.get('summary', '')}"
        )
        return cls(_chunk_text(base_text))

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]
