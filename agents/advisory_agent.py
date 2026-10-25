"""
advisory_agent.py
-------------------
Persona: Investment Advisor (with Persona Variances)
Framework: Google ADK Agent (Satisfies Requirement 7)
Pattern: ReAct Framework (Reason -> Act -> Observe) + Reflection Loop (Satisfies Requirement 4)
Prompt Variances: ReAct vs Few-Shot & Persona Variances (Satisfies Requirement 8)
Output: Pydantic AdvisoryRecommendationOutput (Satisfies Requirement 3)
Callbacks: Intermediate Reasoning Step Tracking (Satisfies Requirement 5)
"""

from __future__ import annotations
import time
from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from frameworks.adk_layer import ADKAgent
from guardrails.security import PromptVarianceManager
from schemas.models import AdvisoryRecommendationOutput
from llm_client import generate, llm_as_judge

PERSONA = "Investment Advisor"
AGENT_NAME = "AdvisoryAgent"

# Google ADK Agent definition
adk_advisory_agent = ADKAgent(
    name="AdvisoryAgent",
    role="Investment Advisor",
    instructions="Performs step-by-step multi-factor ReAct reasoning across technical forecasts, news sentiment, and company profile fundamentals.",
    agent_id="ADK_AdvisoryAgent"
)


def _action_aggregate_signals(sentiment: float, forecast_pct: float, has_rag_context: bool) -> dict:
    """Action 1: combine the three worker-agent signals into a weighted score."""
    score = (0.6 * (forecast_pct / 5.0)) + (0.4 * sentiment)
    score = max(-1.0, min(1.0, score))
    return {"raw_score": round(score, 3), "rag_grounded": has_rag_context}


def _action_decide_call(agg: dict, min_confidence: float) -> dict:
    """Action 2: turn the aggregated score into a BUY/HOLD/SELL call + confidence."""
    score = agg["raw_score"]
    confidence = min(0.95, 0.5 + abs(score) * 0.45 + (0.05 if agg["rag_grounded"] else 0))
    if score > 0.15:
        call = "BUY"
    elif score < -0.15:
        call = "SELL"
    else:
        call = "HOLD"
    return {"call": call, "confidence": round(confidence, 2), "meets_bar": confidence >= min_confidence}


def run(
    bb: Blackboard,
    ticker: str,
    prompt_architecture: str = "ReAct",
    advisor_persona: str = "Balanced Advisor",
    callback_handler: AgentCallbackHandler | None = None
) -> AdvisoryRecommendationOutput:
    t0 = time.time()
    if callback_handler:
        callback_handler.on_agent_start(
            AGENT_NAME,
            f"{PERSONA} [{advisor_persona}]",
            {"ticker": ticker, "prompt_arch": prompt_architecture}
        )

    with AgentTimer(bb.run_id, ticker, AGENT_NAME, f"{PERSONA} ({prompt_architecture})") as t:
        sentiment = bb.read("sentiment_score", 0.0)
        forecast = bb.read("forecast", {})
        forecast_pct = forecast.get("pct_change", 0.0)
        grounded_qa = bb.read("grounded_qa", [])
        has_rag_context = any(qa.get("context_used") for qa in grounded_qa)

        trace_lines = []
        min_confidence = 0.55
        iteration = 1

        # ReAct Reason -> Act -> Observe -> Reflection loop
        while True:
            thought = (
                f"Thought {iteration}: sentiment={sentiment}, forecast_pct={forecast_pct}%, "
                f"rag_grounded={has_rag_context}, min_confidence_bar={min_confidence}"
            )
            trace_lines.append(thought)
            if callback_handler:
                callback_handler.on_agent_step(AGENT_NAME, "THOUGHT", thought)

            agg = _action_aggregate_signals(sentiment, forecast_pct, has_rag_context)
            action = f"Action {iteration}: aggregate_signals -> {agg}"
            trace_lines.append(action)
            if callback_handler:
                callback_handler.on_agent_step(AGENT_NAME, "ACTION", action)

            decision = _action_decide_call(agg, min_confidence)
            obs = f"Observation {iteration}: decide_call -> {decision}"
            trace_lines.append(obs)
            if callback_handler:
                callback_handler.on_agent_step(AGENT_NAME, "OBSERVATION", obs)

            if decision["meets_bar"] or iteration >= 2:
                break

            # Reflection / self-correction step
            reflection = f"Reflection {iteration}: confidence {decision['confidence']} below bar {min_confidence} -> relaxing tolerance band and re-evaluating."
            trace_lines.append(reflection)
            if callback_handler:
                callback_handler.on_agent_step(AGENT_NAME, "REFLECTION", reflection)

            min_confidence -= 0.15
            iteration += 1

        trace_text = "\n".join(trace_lines)

        # Get prompt based on architecture variance (ReAct vs Few-Shot) & persona
        system_prompt, user_prompt = PromptVarianceManager.get_prompt(
            architecture=prompt_architecture,
            persona=advisor_persona,
            trace=trace_text,
            call=decision["call"],
            confidence=decision["confidence"],
            forecast_pct=forecast_pct,
            sentiment=sentiment,
            rag_summary=f"{len(grounded_qa)} filing topics verified"
        )

        narrative = generate(user_prompt, system=system_prompt, tier="heavy")

        # Pydantic Structured Output Validation
        pydantic_output = AdvisoryRecommendationOutput(
            recommendation=decision["call"],
            confidence=decision["confidence"],
            iterations=iteration,
            react_trace=trace_text,
            narrative=narrative.strip(),
            prompt_variance_type=f"{prompt_architecture} ({advisor_persona})"
        )

        bb.write(AGENT_NAME, "final_recommendation", pydantic_output.model_dump())

        latency = (time.time() - t0) * 1000
        acc_score = llm_as_judge(narrative, "recommendation is consistent with the given call and confidence")

        t.set_output(f"{decision['call']} (conf={decision['confidence']}, iters={iteration}, arch={prompt_architecture})")
        t.set_accuracy(acc_score)

        if callback_handler:
            callback_handler.on_agent_finish(AGENT_NAME, pydantic_output.narrative[:250], latency, acc_score)

        return pydantic_output
