"""
Phase 1 — News ingestion via GDELT (free, no API key required).

Pulls recent news articles mentioning the company so the LLM synthesis
layer (Phase 3) has current context beyond what's in the filings.
"""

import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def get_recent_news(company_name: str, max_articles: int = 10) -> list[dict]:
    """
    Returns recent articles: [{"title": ..., "url": ..., "domain": ..., "seendate": ...}]

    Use the full company name (e.g. "Deere & Company"), not just the
    ticker — GDELT searches article text, not stock symbols.
    """
    params = {
        "query": f'"{company_name}"',
        "mode": "ArtList",
        "maxrecords": max_articles,
        "format": "json",
        "sort": "DateDesc",
    }
    resp = requests.get(GDELT_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title"),
            "url": a.get("url"),
            "domain": a.get("domain"),
            "seendate": a.get("seendate"),
        }
        for a in articles
    ]


if __name__ == "__main__":
    # Manual test: python -m src.ingestion.news "Deere & Company"
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Deere & Company"
    articles = get_recent_news(name)
    print(f"Recent news for '{name}':")
    for a in articles:
        print(f"  [{a['seendate']}] {a['title']} ({a['domain']})")
