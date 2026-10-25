"""
forecasting_agent.py
---------------------
Persona: Quant Analyst
Prompt Strategy: Chain-of-Thought (CoT)
Tool: Statsmodels ARIMA time-series forecasting + linear trend fallback.
Output: Pydantic ForecastingOutput (Satisfies Requirement 3)
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import warnings

from blackboard.blackboard import Blackboard
from observability.logger import AgentTimer
from observability.callbacks import AgentCallbackHandler
from schemas.models import ForecastingOutput
from llm_client import generate, llm_as_judge

PERSONA = "Quant Analyst"
AGENT_NAME = "ForecastingAgent"

COT_PROMPT = """You are a Quant Analyst agent (Chain-of-Thought).
Step 1: The model forecasts the next 5 trading days of closing price.
Step 2: Given the current price {current:.2f} and forecast {forecast:.2f}
 (predicted change: {pct:.2f}%, backtest RMSE: {rmse:.3f}),
Step 3: State the predicted direction and confidence in 2 sentences.

Analysis:"""


def _arima_forecast(series: pd.Series, steps: int = 5):
    warnings.filterwarnings("ignore")
    try:
        from statsmodels.tsa.arima.model import ARIMA

        train = series.iloc[:-steps] if len(series) > steps + 20 else series
        model = ARIMA(train.values, order=(2, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)

        if len(series) > steps + 20:
            actual_tail = series.iloc[-steps:].values
            rmse = float(np.sqrt(np.mean((forecast[: len(actual_tail)] - actual_tail) ** 2)))
        else:
            rmse = float(np.std(series.values[-20:]))
        return float(forecast[-1]), rmse, "ARIMA(2,1,1)"
    except Exception:
        tail = series.values[-20:] if len(series) >= 20 else series.values
        x = np.arange(len(tail))
        slope, intercept = np.polyfit(x, tail, 1)
        forecast_val = float(slope * (len(tail) + 4) + intercept)
        rmse = float(np.std(tail))
        return forecast_val, rmse, "LinearTrend(fallback)"


def run(
    bb: Blackboard,
    ticker: str,
    price_df: pd.DataFrame,
    callback_handler: AgentCallbackHandler | None = None
) -> ForecastingOutput:
    t0 = time.time()
    if callback_handler:
        callback_handler.on_agent_start(AGENT_NAME, PERSONA, {"ticker": ticker, "data_points": len(price_df)})

    with AgentTimer(bb.run_id, ticker, AGENT_NAME, PERSONA) as t:
        series = price_df["Close"]
        current_price = float(series.iloc[-1])

        if callback_handler:
            callback_handler.on_tool_start(AGENT_NAME, "statsmodels_arima", {"steps": 5})
        forecast_price, rmse, method = _arima_forecast(series)
        if callback_handler:
            callback_handler.on_tool_end(AGENT_NAME, "statsmodels_arima", f"Forecast: ${forecast_price:.2f}, RMSE: {rmse:.3f}")

        pct_change = ((forecast_price - current_price) / current_price) * 100
        direction = "UP" if pct_change > 0.5 else ("DOWN" if pct_change < -0.5 else "FLAT")

        analysis = generate(
            COT_PROMPT.format(current=current_price, forecast=forecast_price, pct=pct_change, rmse=rmse),
            tier="light",
        )

        pydantic_output = ForecastingOutput(
            current_price=round(current_price, 2),
            forecast_price_5d=round(forecast_price, 2),
            pct_change=round(pct_change, 2),
            direction=direction,
            backtest_rmse=round(rmse, 4),
            method=method,
            analysis=analysis.strip()
        )

        bb.write(AGENT_NAME, "forecast", pydantic_output.model_dump())

        latency = (time.time() - t0) * 1000
        acc_score = max(0.0, 1 - min(1.0, rmse / max(current_price * 0.05, 1e-6)))

        t.set_output(f"{direction} {pct_change:.2f}% via {method}")
        t.set_accuracy(acc_score)

        if callback_handler:
            callback_handler.on_agent_finish(AGENT_NAME, pydantic_output.analysis, latency, acc_score)

        return pydantic_output
