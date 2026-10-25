"""
run_batch.py
------------
Runs the standardized pipeline across multiple companies and passes,
inducing explicit PROMPT VARIANCES (ReAct architecture vs Few-Shot architecture,
and Persona content variances: Balanced vs Aggressive Growth vs Conservative Preservation).
Fully satisfies Requirement 10:
- Multi-run end-to-end evaluation
- Cost and running time estimates
- Systematic prompt variance benchmarking
"""

from __future__ import annotations
import argparse
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.manager_agent import run_pipeline
from observability.logger import count_runs

COMPANIES = [
    ("Apple Inc", "AAPL"),
    ("Microsoft Corporation", "MSFT"),
    ("Tesla Inc", "TSLA"),
    ("Amazon.com Inc", "AMZN"),
    ("NVIDIA Corporation", "NVDA"),
    ("Alphabet Inc", "GOOGL"),
    ("Meta Platforms Inc", "META"),
    ("Netflix Inc", "NFLX"),
    ("JPMorgan Chase & Co", "JPM"),
    ("Reliance Industries", "RELIANCE.NS"),
    ("Tata Consultancy Services", "TCS.NS"),
    ("Infosys Limited", "INFY.NS"),
]

PROMPT_VARIANCES = [
    ("ReAct", "Balanced Advisor"),
    ("Few-Shot", "Aggressive Growth"),
    ("ReAct", "Conservative Preservation"),
    ("Few-Shot", "Balanced Advisor"),
]


def main():
    parser = argparse.ArgumentParser(description="Multi-run batch evaluation with prompt variances and cost estimation.")
    parser.add_argument("--passes", type=int, default=2, help="Number of passes across companies")
    args = parser.parse_args()

    total_runs = len(COMPANIES) * args.passes
    print(f"🚀 Starting Standardized Batch Evaluation: {len(COMPANIES)} companies x {args.passes} passes = {total_runs} runs.")
    print("🔬 Inducing Prompt Variances: ReAct vs Few-Shot + Persona Variances.\n")

    done = 0
    total_cost_usd = 0.0

    for p in range(args.passes):
        arch, persona = PROMPT_VARIANCES[p % len(PROMPT_VARIANCES)]
        print(f"\n=== PASS {p + 1}/{args.passes} [Prompt Variance: Architecture={arch}, Persona='{persona}'] ===")

        for company, ticker in COMPANIES:
            done += 1
            print(f"[{done}/{total_runs}] {company} ({ticker}) | Arch={arch} | Persona={persona}")
            try:
                report = run_pipeline(
                    company_name=company,
                    ticker=ticker,
                    prompt_architecture=arch,
                    advisor_persona=persona
                )
                cost = report.cost_metrics.estimated_cost_usd
                total_cost_usd += cost
                print(f"    -> {report.advisory.recommendation} (Conf: {report.advisory.confidence * 100:.0f}%) | "
                      f"Latency: {report.total_latency_sec:.2f}s | Cost: ${cost:.6f}")
            except Exception as e:
                print(f"    -> FAILED: {e}")

            time.sleep(0.3)

    print("\n" + "=" * 60)
    print(f"✅ Batch Evaluation Complete!")
    print(f"Total Runs in DB: {count_runs()}")
    print(f"Total Batch Cost: ${total_cost_usd:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
