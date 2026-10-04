"""
Phase 1 — Financials ingestion via yfinance.

Pulls a company's snapshot info and core financial statements so the
metrics layer (Phase 2) has something to compute on.
"""

import yfinance as yf


def get_company_snapshot(ticker: str) -> dict:
    """
    Company-level info: name, sector, market cap, etc. Used for the memo header.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "employees": info.get("fullTimeEmployees"),
        "currency": info.get("currency"),
        "business_summary": info.get("longBusinessSummary"),
    }


def get_financial_statements(ticker: str) -> dict:
    """
    Core annual statements as pandas DataFrames (years as columns,
    line items as rows — yfinance's default shape).
    """
    t = yf.Ticker(ticker)
    return {
        "income_statement": t.financials,
        "balance_sheet": t.balance_sheet,
        "cash_flow": t.cashflow,
    }


def get_quarterly_financials(ticker: str) -> dict:
    """Same as above, but quarterly — useful for recent-trend checks."""
    t = yf.Ticker(ticker)
    return {
        "income_statement": t.quarterly_financials,
        "balance_sheet": t.quarterly_balance_sheet,
        "cash_flow": t.quarterly_cashflow,
    }


if __name__ == "__main__":
    # Manual test: python -m src.ingestion.financials DE
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DE"

    snap = get_company_snapshot(ticker)
    print("--- Snapshot ---")
    for k, v in snap.items():
        if k == "business_summary" and v:
            v = v[:150] + "..."
        print(f"{k}: {v}")

    stmts = get_financial_statements(ticker)
    print("\n--- Income statement (head) ---")
    print(stmts["income_statement"].head() if stmts["income_statement"] is not None else "No data returned")
