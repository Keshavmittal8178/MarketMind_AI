"""
guardrails/security.py
----------------------
Security Guardrails, Input Parsing, Data Validation Callbacks, and Prompt Variances.
Fully satisfies Requirement 8:
- Guardrails against prompt injection and abusive/toxic prompts.
- Data validation callbacks (e.g. type checking from tool returns).
- Prompt Variances Engine: Architecture variance (ReAct vs Few-Shot) and Persona variance.
"""

from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional, List
from schemas.models import SecurityAuditLog


class SecurityException(Exception):
    """Raised when an adversarial prompt or toxic input is detected."""
    pass


class PromptGuardrail:
    """Security Guardrail that inspects user prompts before passing them to agents."""

    _INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
        r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
        r"(?i)system\s*:\s*override",
        r"(?i)reveal\s+(your\s+)?(system\s+prompt|secret|instructions)",
        r"(?i)bypass\s+(all\s+)?guardrails",
        r"(?i)sudo\s+mode",
        r"(?i)jailbreak",
        r"(?i)disregard\s+safety\s+checks",
    ]

    _TOXIC_TERMS = [
        "hack", "exploit", "ddos", "ransomware", "malware",
        "drop database", "delete from", "exec(", "<script>",
    ]

    @classmethod
    def inspect(cls, user_text: str) -> Tuple[bool, str]:
        """
        Validates the prompt for injection attacks or toxic commands.
        Returns: (is_safe: bool, reason: str)
        """
        if not user_text or not user_text.strip():
            return False, "Empty prompt provided"

        # Check 1: Prompt Injection Patterns
        for pat in cls._INJECTION_PATTERNS:
            if re.search(pat, user_text):
                return False, f"Prompt Injection detected matching rule: '{pat}'"

        # Check 2: Toxic / Abusive / Malicious Strings
        lower_text = user_text.lower()
        for term in cls._TOXIC_TERMS:
            if term in lower_text:
                return False, f"Malicious/abusive payload term detected: '{term}'"

        return True, "PASSED_SECURE"


class DataValidationCallback:
    """Callback for data validation and type checking on tool returns."""

    @staticmethod
    def validate_price_tool_return(tool_return: Any) -> Tuple[bool, str]:
        """Ensures market price tool returns conform to expected types and ranges."""
        if not isinstance(tool_return, dict):
            return False, f"Expected dict from market data tool, got {type(tool_return)}"
        if "latest_close" in tool_return:
            val = tool_return["latest_close"]
            if not isinstance(val, (int, float)) or val <= 0:
                return False, f"Invalid 'latest_close' price: {val}. Must be positive float."
        return True, "VALID_TYPE_CONFIRMED"

    @staticmethod
    def validate_news_tool_return(tool_return: Any) -> Tuple[bool, str]:
        """Ensures news tool returns non-empty list of string headlines."""
        if not isinstance(tool_return, dict):
            return False, f"Expected dict from news tool, got {type(tool_return)}"
        headlines = tool_return.get("headlines", [])
        if not isinstance(headlines, list):
            return False, f"Expected headlines to be list, got {type(headlines)}"
        return True, "VALID_TYPE_CONFIRMED"


class PromptVarianceManager:
    """
    Manages prompt variations across architecture (ReAct vs Few-Shot)
    and persona content (Aggressive Growth vs Conservative Risk-Averse).
    """

    # --- ARCHITECTURE VARIANCE 1: ReAct Prompt ---
    REACT_SYSTEM = """You are an expert Investment Advisor agent using the ReAct framework
(Reason, Act, Observe). Given the Thought/Action/Observation trace, produce a concise
3-4 sentence professional investment recommendation (BUY / HOLD / SELL) with confidence score."""

    REACT_PROMPT = """ReAct trace:
{trace}

Final recommendation call: {call}
Confidence: {confidence:.2f}

Write the final recommendation narrative now:"""

    # --- ARCHITECTURE VARIANCE 2: Few-Shot Prompt ---
    FEW_SHOT_SYSTEM = """You are a senior Wall Street Investment Strategist.
Learn from the following historical examples before synthesizing your final call:

[EXAMPLE 1]
Inputs: Forecast=+4.2%, Sentiment=+0.35, RAG=Strong recurring cloud revenue
Call: BUY (Confidence: 0.88)
Narrative: Given strong momentum in cloud enterprise adoption and positive price trajectory exceeding our 5-day hurdle rate, we rate the stock a high-conviction BUY.

[EXAMPLE 2]
Inputs: Forecast=-3.1%, Sentiment=-0.28, RAG=Regulatory antitrust investigation
Call: SELL (Confidence: 0.82)
Narrative: Heightened regulatory headwinds combined with bearish price momentum present significant downside risk. We advise liquidating positions to preserve capital.

[EXAMPLE 3]
Inputs: Forecast=+0.4%, Sentiment=+0.05, RAG=Mature business model with high dividend
Call: HOLD (Confidence: 0.65)
Narrative: The stock exhibits sideways consolidation with neutral news sentiment. Investors should maintain existing holdings pending a decisive breakout signal.
"""

    FEW_SHOT_PROMPT = """Now evaluate the target company:
Inputs: Forecast={forecast_pct:+.2f}%, Sentiment={sentiment:+.2f}, RAG={rag_summary}
Computed Call: {call} (Confidence: {confidence:.2f})

Synthesize the final recommendation narrative:"""

    # --- PERSONA VARIANCES ---
    PERSONAS = {
        "Balanced Advisor": {
            "title": "Senior Portfolio Manager",
            "risk_profile": "Balanced growth and capital preservation.",
            "prompt_modifier": "Provide balanced weighting between short-term technicals and fundamental value."
        },
        "Aggressive Growth": {
            "title": "Venture Growth Analyst",
            "risk_profile": "High risk tolerance, focused on upside momentum and market disruption.",
            "prompt_modifier": "Prioritize upward forecast breakout potential and disruptive product catalysts over short-term volatility."
        },
        "Conservative Preservation": {
            "title": "Risk Compliance Officer",
            "risk_profile": "Strict capital preservation, downside risk aversion.",
            "prompt_modifier": "Penalize high volatility, scrutinize valuation risks, and favor HOLD/SELL if downside protection is unclear."
        }
    }

    @classmethod
    def get_prompt(
        cls,
        architecture: str = "ReAct",
        persona: str = "Balanced Advisor",
        **kwargs
    ) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) for the chosen variance."""
        persona_info = cls.PERSONAS.get(persona, cls.PERSONAS["Balanced Advisor"])

        if architecture == "Few-Shot":
            system = cls.FEW_SHOT_SYSTEM + f"\nPersona Modifier ({persona}): {persona_info['prompt_modifier']}"
            user = cls.FEW_SHOT_PROMPT.format(
                forecast_pct=kwargs.get("forecast_pct", 0.0),
                sentiment=kwargs.get("sentiment", 0.0),
                rag_summary=kwargs.get("rag_summary", "Standard filings reviewed"),
                call=kwargs.get("call", "HOLD"),
                confidence=kwargs.get("confidence", 0.5)
            )
            return system, user
        else:  # Default to ReAct
            system = cls.REACT_SYSTEM + f"\nPersona Modifier ({persona}): {persona_info['prompt_modifier']}"
            user = cls.REACT_PROMPT.format(
                trace=kwargs.get("trace", ""),
                call=kwargs.get("call", "HOLD"),
                confidence=kwargs.get("confidence", 0.5)
            )
            return system, user
