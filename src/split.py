"""
split.py
--------
Implements a temporal train / validation / test split for the Olist
delivery-time prediction task, and computes seller- and category-level
target-encoding features *using only training data* to prevent leakage.

Splitting Strategy
==================
Why time-based and not random?

  * Orders have a natural time ordering. A model trained on random 80 %
    of all orders would "see the future" — it would learn patterns from
    orders that arrived *after* the ones it's being evaluated on.
  * In production the model will always predict for new, incoming orders,
    so test performance should reflect a strict forward-looking scenario.

Split ratios (chronological):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Train 70 %  │  Validation 15 %  │  Test (hold-out) 15 %            │
  └──────────────────────────────────────────────────────────────────────┘

  * Train: used to fit all models and compute seller / category statistics.
  * Validation: used for hyper-parameter search and early model selection.
    Never touches test data.
  * Test: used once, at the very end, to report final performance numbers.
    Simulates deployment on genuinely unseen future orders.

Leakage-safe seller & category statistics
==========================================
Features like "seller_avg_delivery" can be powerful predictors but must
be computed from the *training window only*, then joined onto validation
and test sets.  Unseen sellers / categories fall back to the global mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Feature lists (shared with train.py and app.py) ─────────────────────────

NUMERIC_FEATURES = [
    "distance_km",
    "product_weight_kg",
    "product_volume_cm3",
    "price",
    "freight_value",
    "freight_ratio",
    "purchase_hour",
    "n_items",
    "seller_avg_delivery",
    "seller_std_delivery",
    "seller_avg_processing",
    "seller_std_processing",
    "seller_avg_transit",
    "seller_std_transit",
    "cat_avg_processing",
    "cat_avg_transit",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_quarter",
    "is_weekend",
    "product_category_name_english",
]

TARGET = "delivery_time_days"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_seller_stats(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    global_avg_processing: float,
    global_avg_transit: float,
    global_avg_delivery: float,
) -> list[pd.DataFrame]:
    """Compute seller-level stats from *train*, join onto all DataFrames."""
    s_proc = (
        train.groupby("seller_id")["processing_days"]
        .agg(seller_avg_processing="mean", seller_std_processing="std")
        .reset_index()
    )
    s_tran = (
        train.groupby("seller_id")["transit_days"]
        .agg(seller_avg_transit="mean", seller_std_transit="std")
        .reset_index()
    )
    s_del = (
        train.groupby("seller_id")["delivery_time_days"]
        .agg(seller_avg_delivery="mean", seller_std_delivery="std")
        .reset_index()
    )

    out = []
    for df in [train, *others]:
        df = (
            df.merge(s_proc, on="seller_id", how="left")
            .merge(s_tran, on="seller_id", how="left")
            .merge(s_del, on="seller_id", how="left")
        )
        defaults = {
            "seller_avg_processing": global_avg_processing,
            "seller_std_processing": 0.0,
            "seller_avg_transit": global_avg_transit,
            "seller_std_transit": 0.0,
            "seller_avg_delivery": global_avg_delivery,
            "seller_std_delivery": 0.0,
        }
        for col, fill in defaults.items():
            df[col] = df[col].fillna(fill)
        out.append(df)
    return out


def _add_category_stats(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    global_avg_processing: float,
    global_avg_transit: float,
) -> list[pd.DataFrame]:
    """Compute category-level stats from *train*, join onto all DataFrames."""
    c_proc = (
        train.groupby("product_category_name_english")["processing_days"]
        .mean()
        .rename("cat_avg_processing")
        .reset_index()
    )
    c_tran = (
        train.groupby("product_category_name_english")["transit_days"]
        .mean()
        .rename("cat_avg_transit")
        .reset_index()
    )

    out = []
    for df in [train, *others]:
        df = (
            df.merge(c_proc, on="product_category_name_english", how="left")
            .merge(c_tran, on="product_category_name_english", how="left")
        )
        df["cat_avg_processing"] = df["cat_avg_processing"].fillna(global_avg_processing)
        df["cat_avg_transit"] = df["cat_avg_transit"].fillna(global_avg_transit)
        out.append(df)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split *df* into train / validation / test.

    Parameters
    ----------
    df         : feature DataFrame from feature_engineering.py
    train_frac : proportion of orders in training set
    val_frac   : proportion of orders in validation set
                 (test receives the remainder)

    Returns
    -------
    train_df, val_df, test_df  — each enriched with seller & category stats
    """
    if "order_purchase_timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'order_purchase_timestamp'")

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    raw_train = df.iloc[:train_end].copy()
    raw_val = df.iloc[train_end:val_end].copy()
    raw_test = df.iloc[val_end:].copy()

    print(
        f"Split sizes  —  train: {len(raw_train):,} | "
        f"val: {len(raw_val):,} | test: {len(raw_test):,}"
    )
    print(
        f"Train period : {raw_train['order_purchase_timestamp'].min().date()} "
        f"→ {raw_train['order_purchase_timestamp'].max().date()}"
    )
    print(
        f"Val period   : {raw_val['order_purchase_timestamp'].min().date()} "
        f"→ {raw_val['order_purchase_timestamp'].max().date()}"
    )
    print(
        f"Test period  : {raw_test['order_purchase_timestamp'].min().date()} "
        f"→ {raw_test['order_purchase_timestamp'].max().date()}"
    )

    # Compute global fallback stats from training data only
    g_proc = raw_train["processing_days"].mean()
    g_tran = raw_train["transit_days"].mean()
    g_del = raw_train["delivery_time_days"].mean()

    # Fill missing distance with *training* median
    dist_median = raw_train["distance_km"].median()
    for df_ in [raw_train, raw_val, raw_test]:
        df_["distance_km"] = df_["distance_km"].fillna(dist_median)

    # Add seller & category stats (leakage-safe: computed from train only)
    train_df, val_df, test_df = _add_seller_stats(
        raw_train, raw_val, raw_test,
        global_avg_processing=g_proc,
        global_avg_transit=g_tran,
        global_avg_delivery=g_del,
    )
    train_df, val_df, test_df = _add_category_stats(
        train_df, val_df, test_df,
        global_avg_processing=g_proc,
        global_avg_transit=g_tran,
    )

    # Drop rows where any required feature is still null
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(split)
        split.dropna(subset=all_features, inplace=True)
        dropped = before - len(split)
        if dropped:
            print(f"  Dropped {dropped} rows with missing features from {name}")

    return train_df, val_df, test_df


def get_Xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a split DataFrame."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/features.parquet"
    df = pd.read_parquet(features_path)
    train, val, test = temporal_split(df)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    print("Saved train / val / test parquets to outputs/")