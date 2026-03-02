"""
train.py
--------
Trains multiple regression models on the Olist delivery-time dataset,
evaluates them, selects the best, and persists a ready-to-serve pipeline.

Metric Selection — Justification
=================================
We report four metrics.  The business cares about different aspects:

1. MAE (Mean Absolute Error)  ← PRIMARY metric
   --------------------------------------------------
   Interpretation: on average, predictions are ± N days off.
   Why it wins here:
   * Delivery estimates are shown to customers in plain language
     ("your order arrives in ~8 days").  A ±1-day error is very
     different from ±5 days in customer experience, and MAE captures
     this linearly — over- and under-estimates are penalised equally.
   * Robust to the moderate outliers that remain after the 99th-pct cap.
   * Directly maps to the SLA question: "How far off are we, on average?"

2. RMSE (Root Mean Squared Error)
   --------------------------------------------------
   Why we still include it:
   * Heavy penalty for large errors — a model that is usually close but
     occasionally catastrophically wrong will score well on MAE but
     poorly on RMSE.  Bad surprises (promising 5 days, delivering in 30)
     destroy customer trust.
   * Useful for comparing models when tail errors matter.

3. Within-2-days accuracy (%)  ← BUSINESS KPI
   --------------------------------------------------
   "What fraction of estimates fall within ±2 days of the true delivery?"
   * Olist's complaints data shows that reviews < 3 stars spike when the
     customer-perceived delay exceeds 2 days.
   * This threshold metric is directly actionable for ops teams.

4. R²  (coefficient of determination)
   --------------------------------------------------
   * Provides a normalised [0, 1] score useful for communicating "how
     much of the variance in delivery time does our model explain?"
   * Less actionable than MAE/Within-2d, but useful for benchmarking.

Metrics we intentionally omit
------------------------------
* MAPE: undefined / noisy when delivery_time_days is small (near zero).
* Median AE: hides tail behaviour; not helpful for the worst-case story.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure sibling modules in src/ are importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from split import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    get_Xy,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def get_candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_pred: np.ndarray, split_name: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    within_2d = float(np.mean(np.abs(y_true - y_pred) <= 2) * 100)
    within_1d = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100)
    label = f"[{split_name}] " if split_name else ""
    print(
        f"  {label}MAE={mae:.2f}d  RMSE={rmse:.2f}d  "
        f"R²={r2:.3f}  within-2d={within_2d:.1f}%  within-1d={within_1d:.1f}%"
    )
    return dict(MAE=mae, RMSE=rmse, R2=r2, within_2d=within_2d, within_1d=within_1d)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_compare(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Train all candidate models, evaluate on val set, return results table
    and fitted pipelines keyed by model name.
    """
    preprocessor = build_preprocessor()
    X_train, y_train = get_Xy(train_df)
    X_val, y_val = get_Xy(val_df)

    rows = []
    pipelines = {}

    for name, model in get_candidate_models().items():
        print(f"\nTraining: {name}")
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
        pipe.fit(X_train, y_train)

        train_metrics = evaluate(y_train, pipe.predict(X_train), "train")
        val_metrics = evaluate(y_val, pipe.predict(X_val), "val")

        rows.append({"model": name, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pipelines[name] = pipe

    results = pd.DataFrame(rows).sort_values("val_MAE")
    return results, pipelines


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names post-preprocessing."""
    pre = pipeline.named_steps["preprocessor"]
    cat_names = list(
        pre.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return NUMERIC_FEATURES + cat_names


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(features_path: str = "outputs/features.parquet", out_dir: str = "models"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    train_df, val_df, test_df = temporal_split(df)

    # Save splits for downstream scripts
    splits_dir = Path("outputs")
    splits_dir.mkdir(exist_ok=True)
    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    val_df.to_parquet(splits_dir / "val.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)

    results, pipelines = train_and_compare(train_df, val_df)

    print("\n" + "=" * 60)
    print("Validation results (sorted by MAE):")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    print(f"\nBest model on validation: {best_name}")

    # Final evaluation on hold-out test set
    X_test, y_test = get_Xy(test_df)
    best_pipe = pipelines[best_name]
    y_pred_test = best_pipe.predict(X_test)

    print("\n[FINAL] Hold-out test performance:")
    test_metrics = evaluate(y_test, y_pred_test, "test")

    # Persist best pipeline
    model_path = out_dir / "best_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\nSaved pipeline → {model_path}")

    # Persist metadata
    meta = {
        "best_model": best_name,
        "feature_names": get_feature_names(best_pipe),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "test_metrics": test_metrics,
        "val_results": results.to_dict(orient="records"),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata → {meta_path}")

    return best_pipe, results, test_df


if __name__ == "__main__":
    import sys
    features_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/features.parquet"
    main(features_path)