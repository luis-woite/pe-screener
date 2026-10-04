"""
Phase 6 — Training dataset construction.

Builds a company-year feature/label table for the margin-trajectory
classifier: for each (ticker, year) pair, computes the same features
Phase 2 uses for screening, and labels whether EBITDA margin expanded
or contracted in the following year.

Pulling several years of financials for ~70 companies means a lot of
network calls, and yfinance can rate-limit under sustained load — so
this writes progress to a CSV after every ticker and skips tickers
already present on a rerun. Safe to stop (Ctrl+C) and resume later.
"""

import os
import time

import pandas as pd

from src.ingestion.financials import get_company_snapshot, get_financial_statements
from src.metrics.compute import compute_margin_and_growth_trend, compute_leverage, compute_valuation_multiples
from src.ml.universe import TRAINING_UNIVERSE

DATASET_PATH = "data/training_dataset.csv"


def _build_rows_for_ticker(ticker: str) -> list[dict]:
    """
    Returns one feature+label row per usable year for this ticker. A
    year is usable if it has both its own features and a following
    year's EBITDA margin to compute the label from.
    """
    snapshot = get_company_snapshot(ticker)
    statements = get_financial_statements(ticker)
    income_statement = statements.get("income_statement")
    balance_sheet = statements.get("balance_sheet")

    if income_statement is None or income_statement.empty:
        return []

    trend = compute_margin_and_growth_trend(income_statement)
    years = list(trend.columns)  # most recent first (index 0 = most recent)

    rows = []
    # i=0 is the most recent year with no "following year" yet available,
    # so every row we build starts from i=1 onward.
    for i in range(1, len(years)):
        year = years[i]
        following_year = years[i - 1]  # one year more recent than `year`

        ebitda_margin = trend[year].get("ebitda_margin_pct")
        following_ebitda_margin = trend[following_year].get("ebitda_margin_pct")

        if pd.isna(ebitda_margin) or pd.isna(following_ebitda_margin):
            continue  # can't label this row without both margins

        label = 1 if following_ebitda_margin > ebitda_margin else 0

        # Slice statements to only columns <= `year` so leverage/multiples
        # are computed from what would have been known at the time —
        # this avoids leaking future-year data into the features.
        income_upto = income_statement[[c for c in income_statement.columns if c <= year]]
        balance_upto = (
            balance_sheet[[c for c in balance_sheet.columns if c <= year]]
            if balance_sheet is not None and not balance_sheet.empty
            else pd.DataFrame()
        )
        leverage = compute_leverage(balance_upto, income_upto)
        multiples = compute_valuation_multiples(snapshot, income_upto, leverage)

        rows.append(
            {
                "ticker": ticker,
                "year": str(year.date()) if hasattr(year, "date") else str(year),
                "sector": snapshot.get("sector"),
                "revenue_growth_pct": trend[year].get("revenue_growth_pct"),
                "gross_margin_pct": trend[year].get("gross_margin_pct"),
                "operating_margin_pct": trend[year].get("operating_margin_pct"),
                "ebitda_margin_pct": ebitda_margin,
                "net_margin_pct": trend[year].get("net_margin_pct"),
                "net_debt_to_ebitda": leverage.get("net_debt_to_ebitda"),
                "debt_to_equity": leverage.get("debt_to_equity"),
                "ev_to_ebitda": multiples.get("ev_to_ebitda"),
                "label_margin_expanded": label,
            }
        )

    return rows


def build_dataset(output_path: str = DATASET_PATH, pause_seconds: float = 1.0) -> pd.DataFrame:
    """
    Builds (or resumes building) the training dataset across the full
    TRAINING_UNIVERSE, writing progress after every ticker.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    already_done = set()
    all_rows = []
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if not existing.empty:
            already_done = set(existing["ticker"].unique())
            all_rows = existing.to_dict("records")
            print(f"Resuming — {len(already_done)} ticker(s) already in dataset, skipping them.")

    remaining = [t for t in TRAINING_UNIVERSE if t not in already_done]
    print(f"Building dataset for {len(remaining)} remaining ticker(s) (of {len(TRAINING_UNIVERSE)} total universe)...")

    for idx, ticker in enumerate(remaining, 1):
        try:
            rows = _build_rows_for_ticker(ticker)
            all_rows.extend(rows)
            print(f"  [{idx}/{len(remaining)}] {ticker}: {len(rows)} row(s)")
        except Exception as e:
            print(f"  [{idx}/{len(remaining)}] {ticker}: FAILED ({e})")

        # Save after every ticker so an interrupt or crash doesn't lose progress.
        pd.DataFrame(all_rows).to_csv(output_path, index=False)
        time.sleep(pause_seconds)

    df = pd.DataFrame(all_rows)
    print(f"\nDataset complete: {len(df)} total row(s) saved to {output_path}")
    return df


if __name__ == "__main__":
    # Run: python -m src.ml.build_dataset
    # Takes a while (~70 tickers x ~1s pause + network time each) — safe to
    # stop with Ctrl+C and rerun later; it picks up where it left off.
    build_dataset()
