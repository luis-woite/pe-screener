# PE Screener

A tool that takes a ticker, pulls the company's financials, filings, and recent news, and spits out a first-pass screening memo. 

It also has an optional ML model bolted on that predicts whether margins are likely to expand or contract next year.

Purpose of this is to show one can build a tool that compresses hours of analyst work into minutes. It's an aid to independent judgment - not replacing it!

Ofc this model can be tweaked for screening e.g. comps or precedent transactions candidates. It's a super simplified version here. 

Drop me a message on LinkedIn if you'd want my more advanced models built into your firm's workflow!

## Where this actually stands

I built this to learn applied LLM integration and get my hands dirty with real ML, not to ship a product. It works end to end, but I want to be upfront about its limits rather than let the README oversell it:

- The ML model is trained on a pretty small dataset — around 190 rows from ~70 companies over 3 years each. The signal is real (it beats the naive baseline), but with a test set that size, the confidence interval on accuracy is wide. I wouldn't call it precise.
- I evaluate with one time-based train/test split (train on older years, test on the most recent), not k-fold cross-validation. Good enough for v1, not rigorous.
- The holdout is by year, not by company, so the model may have seen a company's 2019 data in training and its 2023 data in testing. A cleaner setup would hold out whole companies.
- Banks and broker-dealers mostly disappear from the training set because they don't report EBITDA the way industrial or consumer companies do. So the model is really trained on non-financials.
- The LLM part (risk flags, summary) is prompt-based grounding against retrieved text - not a fine-tuned model.

None of this is buried - the same caveats are in the docstrings in `src/ml/train_model.py` if you want the long version.

## Structure
pe_screener/
├── main.py # runs the whole pipeline
├── config.py # test tickers, output paths etc.
├── requirements.txt
├── .env.example # copy to .env, fill in your keys
└── src/
├── ingestion/ # pulling financials, filings, news
├── metrics/ # margins, growth, leverage, multiples
├── llm/ # the Claude-powered synthesis step
├── memo/ # builds the final .docx
└── ml/ # the margin-trajectory classifier


## Running it

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env`:
- `ANTHROPIC_API_KEY` — from console.anthropic.com/settings/keys
- `SEC_CONTACT_EMAIL` — your real email; SEC requires this on every EDGAR request
- `NEWSAPI_KEY` — optional, only if you swap GDELT for NewsAPI

Then just:
```bash
python main.py DE
```

### Why this has to run locally
The dev environment I built this in can't reach Yahoo Finance, SEC EDGAR, or GDELT — so the live data pulls only work on your own machine. I wrote and tested the logic in a sandbox; you're the one actually hitting the real APIs.

## Test companies

I picked these specifically to stress-test the pipeline against different financial shapes, not because of any investment view:

| Ticker | Company | Why |
|---|---|---|
| DE | Deere & Co | Industrials, leverage-heavy |
| KKR | KKR & Co | Alt asset manager, close to the PE world |
| CVNA | Carvana | High-leverage consumer turnaround |
| DDOG | Datadog | Growth/SaaS — different metric profile entirely |
| KR | Kroger | Boring, stable — a good control case |

Swap these out in `config.py` whenever.

## The ML piece (optional)

On top of the core pipeline there's a trained classifier predicting whether a company's EBITDA margin expands or contracts next year, based on its current margin/growth/leverage profile. This is actual trained ML — scikit-learn, time-based split — not another LLM call dressed up.

```bash
# builds the training set across ~70 companies — slow, but resumable if you Ctrl+C
python -m src.ml.build_dataset

# trains Logistic Regression and Random Forest, keeps whichever wins,
# prints accuracy / confusion matrix / feature importances
python -m src.ml.train_model
```

Once `data/margin_model.joblib` exists, every memo automatically includes a model signal. If it doesn't exist yet, the pipeline just skips that section — no crash.