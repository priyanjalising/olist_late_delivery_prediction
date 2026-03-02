"""
app.py
------
FastAPI service for the Olist delivery-time predictor.

Endpoints
---------
POST /predict          — Predict delivery time for a single order
POST /predict/batch    — Predict for multiple orders
GET  /health           — Liveness check
GET  /model/info       — Model metadata and feature list

Input payload example
---------------------
{
  "distance_km": 850.3,
  "product_weight_kg": 1.2,
  "product_volume_cm3": 4500,
  "price": 129.90,
  "freight_value": 18.50,
  "freight_ratio": 0.142,
  "purchase_hour": 14,
  "n_items": 2,
  "seller_avg_delivery": 11.4,
  "seller_std_delivery": 2.1,
  "seller_avg_processing": 3.2,
  "seller_std_processing": 0.8,
  "seller_avg_transit": 8.2,
  "seller_std_transit": 1.5,
  "cat_avg_processing": 3.5,
  "cat_avg_transit": 7.9,
  "purchase_month": 11,
  "purchase_day_of_week": 1,
  "purchase_quarter": 4,
  "is_weekend": 0,
  "product_category_name_english": "electronics"
}

Response
--------
{
  "predicted_delivery_days": 12.4,
  "confidence_note": "Established seller — prediction reliability: HIGH",
  "top_features": [
    {"feature": "seller_avg_delivery", "shap_value": 2.31, "direction": "increases"},
    ...
  ]
}
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Artefact paths ────────────────────────────────────────────────────────────
PIPELINE_PATH = Path("models/best_pipeline.pkl")
META_PATH = Path("models/metadata.json")

# ── Load once at start-up ─────────────────────────────────────────────────────

def _load_pipeline():
    if not PIPELINE_PATH.exists():
        raise RuntimeError(f"Model not found at {PIPELINE_PATH}. Run train.py first.")
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)


def _load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)


_pipeline = _load_pipeline()
_meta = _load_meta()

# Build SHAP explainer once (uses the underlying regressor after preprocessing)
_preprocessor = _pipeline.named_steps["preprocessor"]
_regressor = _pipeline.named_steps["regressor"]

# TreeExplainer is fast for RF / XGBoost; fallback to KernelExplainer otherwise
try:
    _explainer = shap.TreeExplainer(_regressor)
    _explainer_type = "tree"
except Exception:
    _explainer = None
    _explainer_type = "none"

FEATURE_NAMES: list[str] = _meta.get("feature_names", [])

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Olist Delivery Time Predictor",
    description=(
        "Predicts customer delivery time (days) for Olist marketplace orders "
        "and explains the top contributing factors using SHAP values."
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class OrderFeatures(BaseModel):
    """Features knowable at order-placement time."""
    # Numeric
    distance_km: float = Field(..., ge=0, description="Seller-to-customer haversine distance in km")
    product_weight_kg: float = Field(..., ge=0)
    product_volume_cm3: float = Field(..., ge=0)
    price: float = Field(..., ge=0)
    freight_value: float = Field(..., ge=0)
    freight_ratio: float = Field(..., ge=0)
    purchase_hour: int = Field(..., ge=0, le=23)
    n_items: int = Field(default=1, ge=1)

    # Seller historical stats (use global averages if seller is new)
    seller_avg_delivery: float = Field(default=12.5)
    seller_std_delivery: float = Field(default=0.0)
    seller_avg_processing: float = Field(default=3.5)
    seller_std_processing: float = Field(default=0.0)
    seller_avg_transit: float = Field(default=9.0)
    seller_std_transit: float = Field(default=0.0)

    # Category-level stats (use global averages if category unknown)
    cat_avg_processing: float = Field(default=3.5)
    cat_avg_transit: float = Field(default=9.0)

    # Categorical
    purchase_month: int = Field(..., ge=1, le=12)
    purchase_day_of_week: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")
    purchase_quarter: int = Field(..., ge=1, le=4)
    is_weekend: int = Field(..., ge=0, le=1)
    product_category_name_english: str = Field(default="unknown")

    @field_validator("is_weekend", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        return int(bool(v))


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str   # "increases" or "decreases"


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
    confidence_note: str
    top_features: list[FeatureContribution]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = _meta.get("numeric_features", [])
CATEGORICAL_FEATURES = _meta.get("categorical_features", [])


def _order_to_df(order: OrderFeatures) -> pd.DataFrame:
    return pd.DataFrame([order.model_dump()])


def _predict_single(df: pd.DataFrame) -> tuple[float, list[dict]]:
    """Return (prediction, shap_top_features)."""
    pred = float(_pipeline.predict(df)[0])
    pred = max(pred, 1.0)   # delivery must be at least 1 day

    top_features: list[dict] = []

    if _explainer_type == "tree" and FEATURE_NAMES:
        X_transformed = _preprocessor.transform(df)
        shap_values = _explainer.shap_values(X_transformed)[0]  # 1-D array

        contrib = sorted(
            zip(FEATURE_NAMES, shap_values),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:5]

        for feat, sv in contrib:
            top_features.append(
                FeatureContribution(
                    feature=feat,
                    shap_value=round(float(sv), 3),
                    direction="increases" if sv > 0 else "decreases",
                ).model_dump()
            )

    return pred, top_features


def _confidence_note(order: OrderFeatures) -> str:
    meta_avg = 12.5  # approximate global average
    is_new_seller = abs(order.seller_avg_delivery - meta_avg) < 0.01
    if is_new_seller:
        return "New/unseen seller — prediction uses global averages; reliability: LOW"
    if order.distance_km > 2000:
        return "Long-distance shipment (>2000km) — wider error margin expected; reliability: MEDIUM"
    if order.purchase_month in (11, 12, 1):
        return "Holiday season order — possible delay; reliability: MEDIUM"
    return "Established seller, typical route — reliability: HIGH"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": _meta.get("best_model", "unknown")}


@app.get("/model/info")
def model_info():
    return JSONResponse(content=_meta)


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderFeatures):
    try:
        df = _order_to_df(order)
        pred, top_features = _predict_single(df)
        note = _confidence_note(order)
        return PredictionResponse(
            predicted_delivery_days=round(pred, 1),
            confidence_note=note,
            top_features=top_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(orders: list[OrderFeatures]):
    if len(orders) > 500:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 500")
    results = []
    for order in orders:
        df = _order_to_df(order)
        pred, top_features = _predict_single(df)
        results.append(
            PredictionResponse(
                predicted_delivery_days=round(pred, 1),
                confidence_note=_confidence_note(order),
                top_features=top_features,
            )
        )
    return BatchPredictionResponse(predictions=results)


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)