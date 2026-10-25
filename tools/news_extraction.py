"""
news_extraction.py
-------------------
Web-extraction tool used by the Data Scout Agent. Pulls recent
headlines for a company from Google News RSS (free, no API key).

Falls back to a small bundled sample of realistic financial headlines
if there is no internet access, so downstream agents (Topic Modeling,
Advisory) can still be demonstrated end-to-end offline.
"""

from __future__ import annotations
import urllib.parse

_FALLBACK_HEADLINES = [
    "{name} beats quarterly earnings estimates, shares rise in early trading",
    "{name} announces new product line amid growing competition",
    "Analysts raise price target for {name} citing strong revenue growth",
    "{name} faces supply chain headwinds, warns of margin pressure",
    "{name} expands into new international markets",
    "Regulatory scrutiny increases for {name} over data practices",
    "{name} CEO discusses AI strategy in investor call",
    "{name} stock volatile after mixed guidance for next quarter",
    "Institutional investors increase holdings in {name}",
    "{name} announces share buyback program worth several billion dollars",
]


def fetch_news_headlines(company_name: str, max_items: int = 12) -> tuple[list[str], bool]:
    """Returns (list of headline strings, is_live_data: bool)."""
    try:
        import feedparser

        query = urllib.parse.quote(f"{company_name} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:max_items] if getattr(entry, "title", None)]
        if headlines:
            return headlines, True
    except Exception:
        pass

    fallback = [h.format(name=company_name) for h in _FALLBACK_HEADLINES][:max_items]
    return fallback, False
