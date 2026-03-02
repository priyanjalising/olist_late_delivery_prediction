"""
feature_engineering.py
-----------------------
Loads raw Olist CSVs, merges datasets, engineers features, and returns a
clean order-level DataFrame ready for splitting and modelling.

All features produced here are knowable at the moment an order is placed,
ensuring zero target leakage.
"""

from pathlib import Path
from math import radians, sin, cos, sqrt, asin

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) pairs in kilometres."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def _vectorised_haversine(df: pd.DataFrame) -> pd.Series:
    """Apply haversine row-wise only where all four coords are present."""
    mask = (
        df["seller_lat"].notna()
        & df["seller_lng"].notna()
        & df["cust_lat"].notna()
        & df["cust_lng"].notna()
    )
    result = pd.Series(np.nan, index=df.index)
    result[mask] = df[mask].apply(
        lambda r: haversine(r.seller_lat, r.seller_lng, r.cust_lat, r.cust_lng),
        axis=1,
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all Olist CSV files from *data_dir*."""
    data_dir = Path(data_dir)
    filenames = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for key, fname in filenames.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")
        dfs[key] = pd.read_csv(path)
    print(f"Loaded {len(dfs)} datasets from {data_dir}")
    return dfs


def _parse_dates(dfs: dict[str, pd.DataFrame]) -> None:
    """Parse date columns in-place."""
    order_date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in order_date_cols:
        dfs["orders"][col] = pd.to_datetime(dfs["orders"][col])
    dfs["items"]["shipping_limit_date"] = pd.to_datetime(dfs["items"]["shipping_limit_date"])
    for col in ["review_creation_date", "review_answer_timestamp"]:
        dfs["reviews"][col] = pd.to_datetime(dfs["reviews"][col])


def _build_targets(orders: pd.DataFrame) -> pd.DataFrame:
    """Derive delivery time targets; keep only fully-delivered orders."""
    df = orders.copy()
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400

    df["processing_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86_400

    df["transit_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86_400

    # Clip obvious data-quality noise
    df["processing_days"] = df["processing_days"].clip(0, 30)
    df["transit_days"] = df["transit_days"].clip(0, 60)

    # Keep only rows where the delivery actually completed
    df = df[df["delivery_time_days"].notna()].copy()

    # Remove extreme outliers (>99th percentile – anomalous logistics events)
    upper = df["delivery_time_days"].quantile(0.99)
    df = df[df["delivery_time_days"] <= upper].copy()
    print(f"Delivered orders after cleaning: {len(df):,}")
    return df


def _build_geolocation_lookup(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Average lat/lng per zip prefix."""
    geo = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    return geo


def build_features(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return a clean, order-level feature DataFrame.

    Parameters
    ----------
    dfs : output of load_raw_data()

    Returns
    -------
    DataFrame with one row per order, features + targets.
    """
    _parse_dates(dfs)

    delivered = _build_targets(dfs["orders"])
    geo_lookup = _build_geolocation_lookup(dfs["geolocation"])

    # ── Merge items ──────────────────────────────────────────────────────────
    oi = delivered[["order_id", "order_purchase_timestamp"]].merge(
        dfs["items"][["order_id", "product_id", "seller_id", "price", "freight_value"]],
        on="order_id",
        how="left",
    )

    # ── Merge product attributes ─────────────────────────────────────────────
    oi = oi.merge(
        dfs["products"][
            ["product_id", "product_category_name",
             "product_weight_g", "product_length_cm",
             "product_height_cm", "product_width_cm"]
        ],
        on="product_id",
        how="left",
    )

    # ── Seller & customer zip codes ──────────────────────────────────────────
    oi = oi.merge(
        dfs["sellers"][["seller_id", "seller_zip_code_prefix"]],
        on="seller_id",
        how="left",
    )
    cust_zip = (
        delivered[["order_id", "customer_id"]]
        .merge(dfs["customers"][["customer_id", "customer_zip_code_prefix"]], on="customer_id", how="left")
    )
    oi = oi.merge(cust_zip[["order_id", "customer_zip_code_prefix"]], on="order_id", how="left")

    # ── Coordinates ──────────────────────────────────────────────────────────
    oi = (
        oi.merge(geo_lookup, left_on="seller_zip_code_prefix", right_on="zip_code_prefix", how="left")
        .rename(columns={"lat": "seller_lat", "lng": "seller_lng"})
        .drop(columns="zip_code_prefix")
    )
    oi = (
        oi.merge(geo_lookup, left_on="customer_zip_code_prefix", right_on="zip_code_prefix", how="left")
        .rename(columns={"lat": "cust_lat", "lng": "cust_lng"})
        .drop(columns="zip_code_prefix")
    )

    oi["distance_km"] = _vectorised_haversine(oi)

    # ── Temporal features (knowable at order time) ───────────────────────────
    ts = oi["order_purchase_timestamp"]
    oi["purchase_month"] = ts.dt.month
    oi["purchase_day_of_week"] = ts.dt.dayofweek   # Mon=0
    oi["purchase_hour"] = ts.dt.hour
    oi["is_weekend"] = (oi["purchase_day_of_week"] >= 5).astype(int)
    oi["purchase_quarter"] = ts.dt.quarter

    # ── Product features ─────────────────────────────────────────────────────
    oi["product_volume_cm3"] = (
        oi["product_length_cm"] * oi["product_height_cm"] * oi["product_width_cm"]
    ).fillna(0)
    oi["product_weight_kg"] = oi["product_weight_g"] / 1000
    # Freight as a share of price — proxy for shipping complexity
    oi["freight_ratio"] = oi["freight_value"] / (oi["price"] + 1e-5)

    # ── Category translation ─────────────────────────────────────────────────
    oi = oi.merge(dfs["category_translation"], on="product_category_name", how="left")
    oi["product_category_name_english"] = oi["product_category_name_english"].fillna("unknown")

    # ── Aggregate to order level ─────────────────────────────────────────────
    order_features = oi.groupby("order_id").agg(
        distance_km=("distance_km", "mean"),
        product_weight_kg=("product_weight_kg", "sum"),
        product_volume_cm3=("product_volume_cm3", "sum"),
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        freight_ratio=("freight_ratio", "mean"),
        purchase_month=("purchase_month", "first"),
        purchase_day_of_week=("purchase_day_of_week", "first"),
        purchase_hour=("purchase_hour", "first"),
        is_weekend=("is_weekend", "first"),
        purchase_quarter=("purchase_quarter", "first"),
        n_items=("product_id", "count"),
        product_category_name_english=(
            "product_category_name_english",
            lambda x: x.mode()[0] if not x.mode().empty else "unknown",
        ),
        seller_id=("seller_id", "first"),
    ).reset_index()

    # ── Merge targets ────────────────────────────────────────────────────────
    order_features = order_features.merge(
        delivered[["order_id", "delivery_time_days", "processing_days",
                   "transit_days", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
    )

    print(f"Final feature matrix: {order_features.shape}")
    return order_features


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    dfs = load_raw_data(data_dir)
    features = build_features(dfs)
    out = Path("outputs/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    print(f"Saved → {out}")