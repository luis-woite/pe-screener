"""
Phase 6 — Inference.

Loads the trained margin-trajectory model and scores a single
company's current-year features — this is what feeds the optional
"Model Signal" line in the screening memo.
"""

import os

import joblib
import pandas as pd

from src.ml.train_model import MODEL_PATH


def predict_margin_trajectory(snapshot: dict, metrics: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Returns {"probability_expansion": float, "model_name": str} using the
    most recent year's features, or {"error": str} if the model isn't
    trained yet or a feature can't be built.
    """
    if not os.path.exists(model_path):
        return {"error": f"No trained model found at {model_path}. Run 'python -m src.ml.train_model' first."}

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    trend = metrics.get("trend_table")
    if trend is None or trend.empty:
        return {"error": "No trend data available to build model features."}

    latest = trend[trend.columns[0]]
    leverage = metrics.get("leverage", {})
    multiples = metrics.get("multiples", {})

    row = {
        "revenue_growth_pct": latest.get("revenue_growth_pct"),
        "gross_margin_pct": latest.get("gross_margin_pct"),
        "operating_margin_pct": latest.get("operating_margin_pct"),
        "ebitda_margin_pct": latest.get("ebitda_margin_pct"),
        "net_margin_pct": latest.get("net_margin_pct"),
        "net_debt_to_ebitda": leverage.get("net_debt_to_ebitda"),
        "debt_to_equity": leverage.get("debt_to_equity"),
        "ev_to_ebitda": multiples.get("ev_to_ebitda"),
        "sector": snapshot.get("sector"),
    }

    try:
        X = pd.DataFrame([row])
        proba = pipeline.predict_proba(X)[0]
        prob_expansion = float(proba[list(pipeline.classes_).index(1)])
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    return {
        "probability_expansion": round(prob_expansion, 3),
        "model_name": bundle.get("model_name", "Unknown"),
    }


if __name__ == "__main__":
    # Manual test: python -m src.ml.predict DE
    import sys

    sys.path.insert(0, ".")
    from src.ingestion.financials import get_company_snapshot, get_financial_statements
    from src.metrics.compute import get_full_metrics

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DE"
    snap = get_company_snapshot(ticker)
    stmts = get_financial_statements(ticker)
    metrics = get_full_metrics(snap, stmts)

    result = predict_margin_trajectory(snap, metrics)
    print(result)
