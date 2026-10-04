"""
Phase 2 — Metric computation.

Takes the raw financial statement DataFrames from Phase 1 (yfinance's
shape: years as columns, line items as rows) and computes the standard
screening metrics an analyst would actually want: margins, growth,
leverage, and basic valuation multiples.

yfinance's exact row labels can shift slightly between companies/versions,
so every lookup goes through _get_row(), which tries a list of known
aliases and falls back to a case-insensitive partial match. Any metric
that can't be computed (missing data) is returned as None rather than
raising — a screening tool should degrade gracefully, not crash on one
missing line item.
"""

from typing import Optional

import pandas as pd


def _get_row(df: pd.DataFrame, aliases: list[str]) -> Optional[pd.Series]:
    """
    Finds a row in a financial statement DataFrame by trying a list of
    known label aliases (exact match first, then case-insensitive
    substring match). Returns None if nothing matches.
    """
    if df is None or df.empty:
        return None

    for alias in aliases:
        if alias in df.index:
            return df.loc[alias]

    lower_index = {str(idx).lower(): idx for idx in df.index}
    for alias in aliases:
        for lower_label, original_label in lower_index.items():
            if alias.lower() in lower_label:
                return df.loc[original_label]

    return None


def _safe_div(numerator, denominator):
    """Division that returns None instead of raising on bad inputs."""
    try:
        if numerator is None or denominator is None or denominator == 0:
            return None
        if pd.isna(numerator) or pd.isna(denominator):
            return None
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


def _pct(numerator, denominator) -> Optional[float]:
    """Returns numerator/denominator as a rounded percentage, or None."""
    ratio = _safe_div(numerator, denominator)
    return round(ratio * 100, 1) if ratio is not None else None


def compute_margin_and_growth_trend(income_statement: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame (years as columns) with revenue, YoY growth,
    gross/operating/net margins for every year available in the
    income statement.
    """
    revenue = _get_row(income_statement, ["Total Revenue", "Revenue"])
    gross_profit = _get_row(income_statement, ["Gross Profit"])
    operating_income = _get_row(income_statement, ["Operating Income", "EBIT"])
    net_income = _get_row(income_statement, ["Net Income", "Net Income Common Stockholders"])
    ebitda = _get_row(income_statement, ["EBITDA", "Normalized EBITDA"])

    years = income_statement.columns if income_statement is not None else []
    rows = {}

    for i, year in enumerate(years):
        rev = revenue.get(year) if revenue is not None else None
        prev_rev = revenue.iloc[i + 1] if revenue is not None and i + 1 < len(revenue) else None

        growth = None
        if rev is not None and prev_rev is not None:
            growth_ratio = _safe_div(rev - prev_rev, prev_rev)
            if growth_ratio is not None:
                growth = round(growth_ratio * 100, 1)

        rows[year] = {
            "revenue": rev,
            "revenue_growth_pct": growth,
            "gross_margin_pct": _pct(gross_profit.get(year), rev) if gross_profit is not None else None,
            "operating_margin_pct": _pct(operating_income.get(year), rev) if operating_income is not None else None,
            "ebitda_margin_pct": _pct(ebitda.get(year), rev) if ebitda is not None else None,
            "net_margin_pct": _pct(net_income.get(year), rev) if net_income is not None else None,
        }

    return pd.DataFrame(rows)


def compute_leverage(balance_sheet: pd.DataFrame, income_statement: pd.DataFrame) -> dict:
    """
    Leverage ratios for the most recent year available: Total Debt /
    EBITDA, Net Debt / EBITDA, and Debt / Equity.
    """
    if balance_sheet is None or balance_sheet.empty:
        return {"total_debt": None, "net_debt": None, "debt_to_equity": None, "net_debt_to_ebitda": None}

    latest_year = balance_sheet.columns[0]

    total_debt_row = _get_row(balance_sheet, ["Total Debt"])
    cash_row = _get_row(balance_sheet, ["Cash And Cash Equivalents", "Cash Financial"])
    equity_row = _get_row(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    ebitda_row = _get_row(income_statement, ["EBITDA", "Normalized EBITDA"]) if income_statement is not None else None

    total_debt = total_debt_row.get(latest_year) if total_debt_row is not None else None
    cash = cash_row.get(latest_year) if cash_row is not None else None
    equity = equity_row.get(latest_year) if equity_row is not None else None
    ebitda = ebitda_row.get(latest_year) if ebitda_row is not None else None

    net_debt = (total_debt - cash) if (total_debt is not None and cash is not None) else None

    d_to_e = _safe_div(total_debt, equity)
    nd_to_ebitda = _safe_div(net_debt, ebitda)

    return {
        "total_debt": total_debt,
        "net_debt": net_debt,
        "debt_to_equity": round(d_to_e, 2) if d_to_e is not None else None,
        "net_debt_to_ebitda": round(nd_to_ebitda, 2) if nd_to_ebitda is not None else None,
    }


def compute_valuation_multiples(snapshot: dict, income_statement: pd.DataFrame, leverage: dict) -> dict:
    """
    Basic valuation multiples using market cap (from the Phase 1
    snapshot) plus net debt (to get enterprise value) and the most
    recent year's EBITDA / net income.
    """
    market_cap = snapshot.get("market_cap")
    net_debt = leverage.get("net_debt")
    enterprise_value = (market_cap + net_debt) if (market_cap is not None and net_debt is not None) else None

    if income_statement is None or income_statement.empty:
        return {"enterprise_value": enterprise_value, "ev_to_ebitda": None, "trailing_pe_proxy": None}

    latest_year = income_statement.columns[0]
    ebitda_row = _get_row(income_statement, ["EBITDA", "Normalized EBITDA"])
    net_income_row = _get_row(income_statement, ["Net Income", "Net Income Common Stockholders"])

    ebitda = ebitda_row.get(latest_year) if ebitda_row is not None else None
    net_income = net_income_row.get(latest_year) if net_income_row is not None else None

    ev_ebitda = _safe_div(enterprise_value, ebitda)
    pe_proxy = _safe_div(market_cap, net_income)

    return {
        "enterprise_value": enterprise_value,
        "ev_to_ebitda": round(ev_ebitda, 1) if ev_ebitda is not None else None,
        # Proxy only — a true trailing P/E needs trailing-twelve-month EPS, not fiscal-year net income.
        "trailing_pe_proxy": round(pe_proxy, 1) if pe_proxy is not None else None,
    }


def get_full_metrics(snapshot: dict, statements: dict) -> dict:
    """
    One-call wrapper: takes the Phase 1 snapshot + statements dict and
    returns everything Phase 2 computes, ready to hand to Phase 3/4.
    """
    income_statement = statements.get("income_statement")
    balance_sheet = statements.get("balance_sheet")

    trend = compute_margin_and_growth_trend(income_statement)
    leverage = compute_leverage(balance_sheet, income_statement)
    multiples = compute_valuation_multiples(snapshot, income_statement, leverage)

    return {
        "trend_table": trend,
        "leverage": leverage,
        "multiples": multiples,
    }


if __name__ == "__main__":
    # Manual test: python -m src.metrics.compute DE
    import sys

    sys.path.insert(0, ".")
    from src.ingestion.financials import get_company_snapshot, get_financial_statements

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DE"
    snap = get_company_snapshot(ticker)
    stmts = get_financial_statements(ticker)
    metrics = get_full_metrics(snap, stmts)

    print(f"--- Margin & growth trend for {ticker} ---")
    print(metrics["trend_table"])
    print(f"\n--- Leverage ---")
    print(metrics["leverage"])
    print(f"\n--- Valuation multiples ---")
    print(metrics["multiples"])
