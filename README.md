# PE Screener — Automated Preliminary Due Diligence Tool

Pulls a company's financials, computes standard screening metrics, and uses an LLM
to synthesise recent filings/news into a structured first-pass screening memo —
with an optional trained classifier predicting next-year margin direction.

## Status: learning project, not production-grade

This is a working personal project built to learn applied LLM integration and
basic ML, not a polished product. It's functional end-to-end, but has known
limitations worth being upfront about:

- The ML classifier (`src/ml/`) is trained on a small dataset (~190 rows from
  ~70 companies × ~3 years each) — promising but not statistically precise;
  the 95% confidence interval on its test accuracy is wide.
- Evaluation uses a single time-based train/test split, not k-fold cross-validation.
- The split holds out by year, not by company — a stricter evaluation would also
  hold out entire companies the model has never seen in training.
- Financial-sector companies (banks, broker-dealers) mostly drop out of the
  training data, since they don't report a conventional EBITDA figure — the
  model is implicitly trained mostly on non-financial sectors.
- The LLM synthesis (`src/llm/`) is prompt-based grounding, not a trained or
  fine-tuned model.

None of this is hidden in the code — see the docstrings in `src/ml/train_model.py`
for the same caveats, in more detail.

## Project structure

```
pe_screener/
├── main.py              # Orchestrator — runs the full pipeline end-to-end
├── config.py             # Test companies, lookback window, output settings
├── requirements.txt
├── .env.example           # Copy to .env and fill in your API keys
├── .gitignore
└── src/
    ├── ingestion/        # Phase 1 — pulling financials, filings, news
    ├── metrics/          # Phase 2 — margins, growth, leverage, multiples
    ├── llm/              # Phase 3 — LLM synthesis (risk flags, narrative)
    └── memo/             # Phase 4 — assembling the final output memo
```

## Setup (do this on your own machine — see note below)

1. **Clone/copy this folder**, then open a terminal in it.

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API keys:**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and paste in:
   - `ANTHROPIC_API_KEY` — from https://console.anthropic.com/settings/keys (needed from Phase 3 onward)
   - `SEC_CONTACT_EMAIL` — your real email; SEC requires this on every EDGAR request (Phase 1)
   - `NEWSAPI_KEY` — optional, from https://newsapi.org/register (only if you use NewsAPI instead of GDELT in Phase 1)

5. **Test the scaffold runs:**
   ```bash
   python main.py DE
   ```
   You should see four `TODO` stage placeholders print for Deere & Co. If that
   works, your environment is ready for Phase 1.

### Why run this locally and not in this chat?

Yahoo Finance, SEC EDGAR, and news APIs aren't reachable from this sandbox's
network, so live data-pulling has to happen on your machine. I'll still write
and sanity-check the logic for each phase here — you just run it locally to
hit the real APIs.

## Test companies (config.py)

Chosen to stress-test the pipeline against different financial profiles:

| Ticker | Company | Why it's useful as a test case |
|---|---|---|
| DE | Deere & Co | Industrials, capital-intensive, leverage-heavy |
| KKR | KKR & Co | Alternative asset manager — thematically close to PE |
| CVNA | Carvana | Consumer/auto retail, high-leverage turnaround story |
| DDOG | Datadog | Growth/SaaS — metrics skew growth over margin |
| KR | Kroger | Consumer staples, stable "control" case |

Swap these in `config.py` any time — nothing else depends on this exact list.

## Phase 6 (optional) — ML margin-trajectory signal

Beyond the core pipeline, there's an optional trained classifier that
predicts whether a company's EBITDA margin is more likely to expand or
contract next year, based on its current margin/growth/leverage profile.
This is genuine trained ML (scikit-learn, time-based train/test split),
not another LLM call — see `src/ml/`.

To enable it:

```bash
# 1. Build the training dataset (pulls financials for ~70 companies —
#    takes a while, and is safe to Ctrl+C and rerun; it resumes automatically)
python -m src.ml.build_dataset

# 2. Train the model (quick — prints accuracy, confusion matrix, and
#    feature importances for both a Logistic Regression and a Random
#    Forest candidate, then saves whichever did better)
python -m src.ml.train_model
```

Once `data/margin_model.joblib` exists, every future `python main.py <TICKER>`
run automatically includes a "Model Signal" section in the memo — no
further setup needed. If the model file isn't present yet, the pipeline
just skips that section gracefully.

## Roadmap

See the project plan doc for the full phase-by-phase build plan (Phases 1–5).
Phase 6 (above) is a self-contained addition on top of the core 4-phase pipeline.
