"""
Phase 6 — Model training.

Trains a classifier to predict whether a company's EBITDA margin will
expand or contract in the following year, using the same features the
screening tool already computes (Phase 2).

Evaluated with a TIME-BASED split — train on older years, test on the
most recent year — rather than a random split. A random split would
let the model implicitly learn from future-dated rows when predicting
older ones, which is a lookahead bias a real forecasting model should
never be evaluated under. With a training universe this size, a
cleaner evaluation would also hold out entire companies the model has
never seen in training (not just later years of companies it has seen
some of); that's a natural next improvement, noted here rather than
hidden.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.build_dataset import DATASET_PATH

NUMERIC_FEATURES = [
    "revenue_growth_pct",
    "gross_margin_pct",
    "operating_margin_pct",
    "ebitda_margin_pct",
    "net_margin_pct",
    "net_debt_to_ebitda",
    "debt_to_equity",
    "ev_to_ebitda",
]
CATEGORICAL_FEATURES = ["sector"]
LABEL_COL = "label_margin_expanded"
MODEL_PATH = "data/margin_model.joblib"


def _time_based_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Sorts by year and holds out the most recent test_frac as the test set."""
    df = df.copy()
    df["year_parsed"] = pd.to_datetime(df["year"], errors="coerce")
    df = df.dropna(subset=["year_parsed"]).sort_values("year_parsed")

    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _build_pipeline(model) -> Pipeline:
    """Wraps a classifier with imputation/scaling/encoding so it can take raw feature rows directly."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(dataset_path: str = DATASET_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[LABEL_COL])
    print(f"Loaded {len(df)} labeled row(s) from {dataset_path}")

    train_df, test_df = _time_based_split(df)
    print(
        f"Time-based split: {len(train_df)} train row(s) / {len(test_df)} test row(s) "
        f"(test = most recent rows by year)"
    )

    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError("Not enough data to split into train/test — build a larger dataset first.")

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[LABEL_COL]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[LABEL_COL]

    candidates = {
        "Logistic Regression": _build_pipeline(LogisticRegression(max_iter=1000)),
        "Random Forest": _build_pipeline(RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (pipeline, acc)
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.1%}")
        print(classification_report(y_test, preds, target_names=["Contracted", "Expanded"], zero_division=0))
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(confusion_matrix(y_test, preds))

    # Report the naive baseline too — "always predict the majority class" —
    # so the model's accuracy is judged against doing nothing clever, not in isolation.
    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    baseline_label = "Expanded" if majority_class == 1 else "Contracted"
    print(f"\nBaseline (always predict '{baseline_label}'): {baseline_acc:.1%}")

    best_name = max(results, key=lambda k: results[k][1])
    best_pipeline, best_acc = results[best_name]
    print(f"\nBest model: {best_name} ({best_acc:.1%} accuracy) — saving to {model_path}")

    import os

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "model_name": best_name, "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES},
        model_path,
    )

    if best_name == "Random Forest":
        rf = best_pipeline.named_steps["model"]
        onehot = best_pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        feature_names = NUMERIC_FEATURES + list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))
        importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
        print("\nTop feature importances:")
        for feat, imp in importances[:8]:
            print(f"  {feat}: {imp:.3f}")

    return best_pipeline, results


if __name__ == "__main__":
    # Run: python -m src.ml.train_model (after build_dataset.py has produced data)
    train_and_evaluate()
