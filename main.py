"""
main.py
-------
CLI entry point for MarketMind AI.
Standardized Multi-Agent Intelligence System adhering to all 11 project requirements:
- Google ADK + CrewAI runtime
- A2A Protocol interoperability
- MCP Tool execution
- Pydantic structured output (JSON / Markdown / Table)
- Security Guardrails & Prompt Variances
- Latency & Cost Tracking
"""

from __future__ import annotations
import argparse
import json
import sys

from agents.manager_agent import run_pipeline

# Ensure utf-8 encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="MarketMind AI — Standardized Multi-Agent Intelligence Pipeline")
    parser.add_argument("--company", required=True, help="Company name, e.g. 'Apple Inc'")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--prompt-arch", default="ReAct", choices=["ReAct", "Few-Shot"], help="Prompt Architecture Variance")
    parser.add_argument("--persona", default="Balanced Advisor", choices=["Balanced Advisor", "Aggressive Growth", "Conservative Preservation"], help="Advisor Persona Variance")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json", "table"], help="Output presentation format")
    parser.add_argument("--save-json", default=None, help="Optional path to save full output JSON")
    args = parser.parse_args()

    print(f"\n🔎 [MarketMind AI] Initializing Multi-Agent Pipeline for {args.company} ({args.ticker})...")
    print(f"⚙️ Frameworks: Google-ADK (Host) <-> CrewAI (Workers) via A2A Protocol")
    print(f"🛡️ Guardrails: Active | Prompt Arch: {args.prompt_arch} | Persona: {args.persona}\n")

    report = run_pipeline(
        company_name=args.company,
        ticker=args.ticker,
        prompt_architecture=args.prompt_arch,
        advisor_persona=args.persona
    )

    if args.format == "markdown":
        print(report.to_markdown())
    elif args.format == "json":
        print(report.model_dump_json(indent=2))
    elif args.format == "table":
        table_rows = report.to_summary_table()
        print(f"\n{'METRIC':<25} | {'VALUE'}")
        print("-" * 50)
        for r in table_rows:
            print(f"{r['Metric']:<25} | {r['Value']}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\n✅ Structured Pydantic report saved to {args.save_json}")


if __name__ == "__main__":
    sys.exit(main())
