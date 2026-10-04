"""
Phase 3 — LLM synthesis.

Takes everything Phases 1-2 have gathered (company snapshot, computed
metrics, a filing excerpt, recent news) and asks Claude to synthesise
it into a structured screening summary: a short narrative, risk flags,
and key themes — grounded strictly in the supplied text, not general
knowledge about the company.

Requires ANTHROPIC_API_KEY to be set (loaded from .env via python-dotenv).
"""

import json
import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a private equity / investment banking analyst assistant \
helping with first-pass due diligence screening.

You will be given: a company snapshot, computed financial metrics, an \
excerpt from the company's most recent SEC filing, and recent news \
headlines. Your job is to synthesise this into a structured screening \
summary.

CRITICAL GROUNDING RULES:
- Base every claim ONLY on the text provided to you. Do not use outside \
knowledge about the company, even if you recognise it.
- If the filing excerpt or news don't mention something, do not invent it.
- Flag numeric figures only if they appear in the provided metrics or text \
— never fabricate a number.
- If the provided information is too sparse to support a claim, say so \
rather than filling the gap with assumption.

Respond with ONLY a JSON object (no markdown fences, no preamble), matching \
exactly this schema:
{
  "summary": "2-3 sentence narrative overview grounded in the provided data",
  "risk_flags": ["short risk flag 1", "short risk flag 2", ...],
  "key_themes": ["short theme 1", "short theme 2", ...]
}

Use 2-5 risk flags and 2-4 key themes. Keep each flag/theme under 20 words."""


def _format_metrics(metrics: dict) -> str:
    """Turns the Phase 2 metrics dict into a short readable block for the prompt."""
    trend = metrics.get("trend_table")
    leverage = metrics.get("leverage", {})
    multiples = metrics.get("multiples", {})

    lines = []
    if trend is not None and not trend.empty:
        latest_year = trend.columns[0]
        latest = trend[latest_year]
        lines.append(
            f"Latest year metrics — Revenue growth: {latest.get('revenue_growth_pct')}%, "
            f"Gross margin: {latest.get('gross_margin_pct')}%, "
            f"EBITDA margin: {latest.get('ebitda_margin_pct')}%, "
            f"Net margin: {latest.get('net_margin_pct')}%"
        )
    lines.append(
        f"Leverage — Net Debt/EBITDA: {leverage.get('net_debt_to_ebitda')}x, "
        f"Debt/Equity: {leverage.get('debt_to_equity')}x"
    )
    lines.append(
        f"Valuation — EV/EBITDA: {multiples.get('ev_to_ebitda')}x, "
        f"P/E (proxy): {multiples.get('trailing_pe_proxy')}x"
    )
    return "\n".join(lines)


def _format_news(news_articles: list[dict], max_items: int = 8) -> str:
    """Turns the Phase 1 news list into a short headline block for the prompt."""
    if not news_articles:
        return "No recent news articles were found."
    lines = [f"- [{a.get('seendate', '?')}] {a.get('title', '(no title)')}" for a in news_articles[:max_items]]
    return "\n".join(lines)


def build_context(snapshot: dict, metrics: dict, filing_excerpt: str, news_articles: list[dict]) -> str:
    """Assembles the full grounding context sent to Claude as the user message."""
    return f"""COMPANY SNAPSHOT
Name: {snapshot.get('name')}
Sector: {snapshot.get('sector')} | Industry: {snapshot.get('industry')}
Market cap: {snapshot.get('market_cap')}

COMPUTED METRICS
{_format_metrics(metrics)}

RECENT NEWS HEADLINES
{_format_news(news_articles)}

FILING EXCERPT (most recent 10-K/10-Q)
{filing_excerpt[:8000] if filing_excerpt else "No filing excerpt available."}
"""


def _strip_code_fences(text: str) -> str:
    """Claude sometimes wraps JSON in ```json fences despite instructions not to — strip them if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def synthesize_screening_summary(
    snapshot: dict, metrics: dict, filing_excerpt: str, news_articles: list[dict]
) -> dict:
    """
    Calls Claude to synthesise the gathered data into a structured summary.
    Returns {"summary": str, "risk_flags": [...], "key_themes": [...]}.
    On any failure (API error, malformed JSON), returns a dict with an
    "error" key instead of raising — a screening tool should degrade
    gracefully rather than crash the whole pipeline over one LLM call.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY not set — check your .env file."}

    context = build_context(snapshot, metrics, filing_excerpt, news_articles)

    try:
        client = Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        raw_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        cleaned = _strip_code_fences(raw_text)
        parsed = json.loads(cleaned)

        # Basic schema sanity check — don't trust the model blindly.
        if not all(k in parsed for k in ("summary", "risk_flags", "key_themes")):
            return {"error": f"LLM response missing expected fields. Raw: {raw_text[:300]}"}

        return parsed

    except json.JSONDecodeError:
        return {"error": f"Could not parse LLM response as JSON. Raw: {raw_text[:300]}"}
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}


if __name__ == "__main__":
    # Manual test: python -m src.llm.synthesize DE
    import sys

    sys.path.insert(0, ".")
    from src.ingestion.financials import get_company_snapshot, get_financial_statements
    from src.ingestion.filings import get_latest_filings, fetch_filing_text
    from src.ingestion.news import get_recent_news
    from src.metrics.compute import get_full_metrics

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DE"

    snap = get_company_snapshot(ticker)
    stmts = get_financial_statements(ticker)
    metrics = get_full_metrics(snap, stmts)
    filings = get_latest_filings(ticker)
    excerpt = fetch_filing_text(filings[0]["primary_doc_url"], max_chars=8000) if filings else ""
    news = get_recent_news(snap.get("name") or ticker)

    result = synthesize_screening_summary(snap, metrics, excerpt, news)
    print(json.dumps(result, indent=2))
