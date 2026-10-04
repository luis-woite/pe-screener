"""
Phase 6 — Training universe.

A deliberately diverse set of established, US-listed companies across
sectors, used to build the margin-trajectory training dataset. Picked
for long, clean public filing/financial histories (large/mid caps),
not for any investment view — this list exists purely to give the
model varied financial profiles to learn from.
"""

TRAINING_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "ORCL", "CSCO", "ADBE", "CRM", "IBM", "INTC", "TXN", "QCOM",
    # Industrials
    "DE", "CAT", "HON", "GE", "MMM", "UPS", "LMT", "RTX", "BA", "UNP",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG",
    # Consumer Staples
    "KO", "PEP", "PG", "WMT", "COST", "CL", "KMB", "MDLZ",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW",
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "ABT", "TMO",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Materials
    "LIN", "APD", "ECL",
    # Utilities
    "NEE", "DUK", "SO",
    # Communication Services
    "VZ", "T", "CMCSA", "DIS",
    # Real Estate
    "PLD", "AMT", "SPG",
]
