"""
Central config for the screening tool.
Keep anything you might want to tweak between runs in here, not buried in scripts.
"""

# --- Test companies for building & validating the pipeline (Phase 0 decision) ---
# Deliberately spans different financial profiles so each pipeline stage gets
# stress-tested against more than one "shape" of company:
#   DE   - Deere & Co:      industrials, capital-intensive, leverage-heavy
#   KKR  - KKR & Co:        alternative asset manager (thematically close to PE)
#   CVNA - Carvana:         consumer/auto retail, high-leverage turnaround story
#   DDOG - Datadog:         growth/SaaS, different metric profile (growth > margin)
#   KR   - Kroger:          consumer staples, stable "control" case
TEST_COMPANIES = ["DE", "KKR", "CVNA", "DDOG", "KR"]

# --- Lookback window for news / filings ingestion (Phase 1) ---
NEWS_LOOKBACK_DAYS = 30

# --- Output location for generated memos ---
OUTPUT_DIR = "output"
