"""
Phase 1 — SEC EDGAR filings ingestion.

Looks up a company's CIK, finds its most recent 10-K/10-Q filings, and
pulls a readable text excerpt for the LLM synthesis layer (Phase 3).

SEC requires every request to carry a descriptive User-Agent identifying
who's making it (see https://www.sec.gov/os/webmaster-faq#developers).
Set SEC_CONTACT_EMAIL in your .env file — never hardcode it here, since
this file is safe to publish publicly (e.g. on GitHub) but your email
shouldn't be.
"""

import os
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

_contact_email = os.environ.get("SEC_CONTACT_EMAIL", "your_email@example.com")
HEADERS = {"User-Agent": f"pe_screener_project {_contact_email}"}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_CIK_CACHE: dict = {}


def get_cik(ticker: str) -> Optional[str]:
    """Looks up a company's 10-digit zero-padded CIK from its ticker."""
    global _CIK_CACHE
    if not _CIK_CACHE:
        resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _CIK_CACHE = {entry["ticker"]: str(entry["cik_str"]).zfill(10) for entry in data.values()}

    return _CIK_CACHE.get(ticker.upper())


def get_latest_filings(ticker: str, form_types=("10-K", "10-Q"), limit: int = 2) -> list[dict]:
    """
    Returns metadata for the most recent filings of the given form types:
    [{"form": "10-K", "filed": "2026-02-10", "accession": "...", "primary_doc_url": "..."}]
    """
    cik = get_cik(ticker)
    if not cik:
        raise ValueError(f"Could not find a CIK for ticker '{ticker}' — check it's a US-listed company.")

    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    results = []
    for i, form in enumerate(recent["form"]):
        if form in form_types:
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            results.append(
                {
                    "form": form,
                    "filed": recent["filingDate"][i],
                    "accession": recent["accessionNumber"][i],
                    "primary_doc_url": url,
                }
            )
            if len(results) >= limit:
                break
    return results


def fetch_filing_text(url: str, max_chars: int = 20000) -> str:
    """
    Downloads a filing document and strips it to plain text, truncated
    to max_chars (filings can be huge; Phase 3 only needs an excerpt).
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()  # strip JS/CSS so it can't leak into the extracted text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:max_chars]


if __name__ == "__main__":
    # Manual test: python -m src.ingestion.filings DE
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DE"

    filings = get_latest_filings(ticker)
    print(f"Latest filings for {ticker}:")
    for f in filings:
        print(f"  {f['form']} filed {f['filed']} -> {f['primary_doc_url']}")

    if filings:
        print("\n--- Excerpt of most recent filing ---")
        print(fetch_filing_text(filings[0]["primary_doc_url"], max_chars=1000))
