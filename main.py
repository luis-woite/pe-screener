"""
Entry point for the screening pipeline.

Status:
  Phase 1 (data ingestion) -> DONE, see src/ingestion
  Phase 2 (metrics)        -> DONE, see src/metrics
  Phase 3 (LLM synthesis)  -> DONE, see src/llm
  Phase 4 (memo output)    -> DONE, see src/memo
  Phase 6 (ML signal)      -> DONE, see src/ml (optional — requires a trained model)
"""

import sys
from config import TEST_COMPANIES, OUTPUT_DIR
from src.ingestion.financials import get_company_snapshot, get_financial_statements
from src.ingestion.filings import get_latest_filings, fetch_filing_text
from src.ingestion.news import get_recent_news
from src.metrics.compute import get_full_metrics
from src.llm.synthesize import synthesize_screening_summary
from src.memo.generate import generate_memo
from src.ml.predict import predict_margin_trajectory


def _fmt(value, suffix: str = "") -> str:
    """Formats a metric for display, turning missing/NaN values into 'N/A'."""
    if value is None:
        return "N/A"
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return "N/A"
    except TypeError:
        pass
    return f"{value}{suffix}"


def run_pipeline(ticker: str) -> None:
    print(f"\n--- Running screening pipeline for {ticker} ---")

    # Stage 1: Data ingestion (Phase 1)
    print("[1/4] Ingesting financials, filings, news...")

    snapshot = get_company_snapshot(ticker)
    company_name = snapshot.get("name") or ticker
    print(f"   Company: {company_name} | Sector: {snapshot.get('sector')} | Market cap: {snapshot.get('market_cap')}")

    statements = get_financial_statements(ticker)
    income = statements.get("income_statement")
    years_pulled = len(income.columns) if income is not None and not income.empty else 0
    print(f"   Pulled {years_pulled} year(s) of annual financials")

    filing_excerpt = ""
    try:
        filings = get_latest_filings(ticker)
        print(f"   Found {len(filings)} recent filing(s) (10-K/10-Q)")
        if filings:
            filing_excerpt = fetch_filing_text(filings[0]["primary_doc_url"], max_chars=8000)
    except ValueError as e:
        filings = []
        print(f"   Filings lookup skipped: {e}")
    except Exception as e:
        print(f"   Could not fetch filing text: {e}")

    try:
        news = get_recent_news(company_name)
        print(f"   Found {len(news)} recent news article(s)")
    except Exception as e:
        news = []
        print(f"   News lookup failed: {e}")

    # Stage 2: Metric computation (Phase 2)
    print("[2/4] Computing screening metrics...")
    metrics = get_full_metrics(snapshot, statements)

    latest_year = metrics["trend_table"].columns[0] if not metrics["trend_table"].empty else None
    if latest_year is not None:
        latest = metrics["trend_table"][latest_year]
        print(
            f"   Latest year — Revenue growth: {_fmt(latest['revenue_growth_pct'], '%')} | "
            f"Gross margin: {_fmt(latest['gross_margin_pct'], '%')} | "
            f"EBITDA margin: {_fmt(latest['ebitda_margin_pct'], '%')} | "
            f"Net margin: {_fmt(latest['net_margin_pct'], '%')}"
        )
    else:
        print("   No income statement data available to compute margins/growth")

    lev = metrics["leverage"]
    print(f"   Net Debt/EBITDA: {_fmt(lev['net_debt_to_ebitda'], 'x')} | Debt/Equity: {_fmt(lev['debt_to_equity'], 'x')}")

    mult = metrics["multiples"]
    print(f"   EV/EBITDA: {_fmt(mult['ev_to_ebitda'], 'x')} | P/E (proxy): {_fmt(mult['trailing_pe_proxy'], 'x')}")

    # Stage 3: LLM synthesis (Phase 3)
    print("[3/4] Synthesising risk flags & narrative...")
    synthesis = synthesize_screening_summary(snapshot, metrics, filing_excerpt, news)

    if "error" in synthesis:
        print(f"   LLM synthesis failed: {synthesis['error']}")
    else:
        print(f"   Summary: {synthesis['summary']}")
        print(f"   Risk flags:")
        for flag in synthesis.get("risk_flags", []):
            print(f"     - {flag}")
        print(f"   Key themes:")
        for theme in synthesis.get("key_themes", []):
            print(f"     - {theme}")

    # Stage 4: Memo generation (Phase 4, + optional Phase 6 model signal)
    print("[4/4] Generating memo...")
    model_signal = predict_margin_trajectory(snapshot, metrics)
    if "error" in model_signal:
        print(f"   Model signal unavailable: {model_signal['error']}")
    else:
        prob = model_signal["probability_expansion"]
        direction = "expansion" if prob >= 0.5 else "contraction"
        confidence = prob if prob >= 0.5 else 1 - prob
        print(f"   Model signal ({model_signal['model_name']}): {confidence:.0%} probability of margin {direction}")

    try:
        memo_path = generate_memo(ticker, snapshot, metrics, synthesis, output_dir=OUTPUT_DIR, model_signal=model_signal)
        print(f"   Memo saved to: {memo_path}")
    except Exception as e:
        print(f"   Memo generation failed: {e}")

    print(f"--- Done for {ticker} ---\n")


if __name__ == "__main__":
    # Run on one ticker if passed as an argument, otherwise loop over all test companies.
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1].upper())
    else:
        for t in TEST_COMPANIES:
            run_pipeline(t)
