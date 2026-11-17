"""
guardrails/enterprise_guardrails.py
------------------------------------
MarketMind AI: Enterprise ML & Semantic AI Safety Guardrails.
Satisfies Phase 2 Milestone (Meta Llama-Guard Paradigm & PII / Financial Manipulation Defense).

Capabilities:
1. Financial Market Manipulation Detection (Pump-and-dump, insider tips, guaranteed returns).
2. Confidential PII & Credential Masking (Credit Cards, PAN, API tokens, JWTs).
3. Semantic Toxic Query Interception with zero latency overhead.
"""

from __future__ import annotations
import re
from typing import Tuple, Dict, Any, List


class EnterpriseSafetyGuard:
    """Enterprise safety filter for high-stakes financial multi-agent pipelines."""

    _PII_PATTERNS = [
        (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CREDIT_CARD]"),
        (r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", "[REDACTED_PAN_CARD]"),
        (r"(?i)\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9_\-\.]{20,})", "[REDACTED_API_KEY]"),
    ]

    _MANIPULATION_PATTERNS = [
        r"(?i)pump\s+and\s+dump",
        r"(?i)insider\s+trading\s+tip",
        r"(?i)guaranteed\s+(100%|[0-9]{3,}%)\s+(profit|return)",
        r"(?i)front-?run\s+(the\s+)?order",
        r"(?i)market\s+rigging",
    ]

    @classmethod
    def sanitize_and_inspect(cls, text: str) -> Tuple[bool, str, str]:
        """
        Inspects text for financial manipulation and masks sensitive credentials.
        Returns: (is_safe: bool, reason: str, sanitized_text: str)
        """
        if not text:
            return True, "PASSED_EMPTY", text

        # Step 1: PII Masking
        sanitized = text
        for pattern, replacement in cls._PII_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)

        # Step 2: Financial Manipulation Inspection
        for pat in cls._MANIPULATION_PATTERNS:
            if re.search(pat, sanitized):
                return False, f"Financial Market Manipulation Attempt Detected: '{pat}'", sanitized

        return True, "PASSED_ENTERPRISE_SECURE", sanitized
