# Olist Delivery Time Predictor

Predict how many days an order will take to reach the customer — turned from a notebook into a production-grade ML pipeline with a REST API.

---

## Project layout

```
olist_delivery/
├── src/
│   ├── feature_engineering.py   # raw CSV → feature DataFrame
│   ├── split.py                 # temporal train/val/test split + leakage-safe encoding
│   ├── train.py                 # model training, evaluation, serialisation
│   ├── error_analysis.py        # structured error & segment analysis
│   └── app.py                   # FastAPI service (prediction + SHAP)
├── tests/
│   └── test_api.py              # integration tests + sample requests
├── models/                      # persisted pipeline + metadata (created by train.py)
├── outputs/                     # parquet splits + error-analysis charts
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Build features
python src/feature_engineering.py /path/to/olist/data

# 2. Train models (runs all candidates, picks best on validation MAE)
python src/train.py outputs/features.parquet

# 3. Error analysis
python src/error_analysis.py

# 4. Serve the API
cd src && uvicorn app:app --reload --port 8000
```

---

## Splitting strategy

Orders are split **chronologically** into three non-overlapping windows:

| Split | Size | Role |
|-------|------|------|
| Train | 70 % (earliest) | Fit all models; compute seller & category target encodings |
| Validation | 15 % | Hyper-parameter selection; model comparison — **never touches test** |
| Test | 15 % (latest) | Final one-shot evaluation; simulates real deployment |

**Why time-based, not random?**  
Randomly shuffling would let the model see orders from "the future" during training — their seller-history features and category trends would leak forward in time. A time-based split ensures every prediction is genuinely forward-looking, matching the production scenario.

**Seller/category statistics** are computed exclusively from the training window and joined onto validation and test. New sellers or categories fall back to the global training mean, exactly as the live API does.

---

## Metrics — why these four?

| Metric | Why it matters |
|--------|---------------|
| **MAE** (primary) | Average error in plain days. Directly maps to customer-experience question: "how far off is the estimate?" Linear penalty means over- and under-predictions are treated equally. |
| **RMSE** | Penalises large errors quadratically. Catches models that are usually close but occasionally catastrophic — which destroys trust even if average MAE looks fine. |
| **Within-2-days %** | Business KPI: review scores drop sharply when perceived delay exceeds 2 days. This threshold is directly actionable. |
| **R²** | Normalised variance explained — useful for communicating model quality to non-technical stakeholders. |

---

## API — sample curl commands

### Single prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "purchase_month": 6,
    "purchase_day_of_week": 1,
    "purchase_quarter": 2,
    "is_weekend": 0,
    "product_category_name_english": "electronics"
  }' | python3 -m json.tool
```

**Sample response:**

```json
{
  "predicted_delivery_days": 12.4,
  "confidence_note": "Established seller, typical route — reliability: HIGH",
  "top_features": [
    {"feature": "seller_avg_delivery", "shap_value": 2.31, "direction": "increases"},
    {"feature": "distance_km",         "shap_value": 1.87, "direction": "increases"},
    {"feature": "cat_avg_transit",     "shap_value": 0.94, "direction": "increases"},
    {"feature": "seller_avg_transit",  "shap_value": 0.72, "direction": "increases"},
    {"feature": "freight_value",       "shap_value": -0.41, "direction": "decreases"}
  ]
}
```

### Batch prediction

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"distance_km": 200, "product_weight_kg": 0.5, ...}, {...}]' \
  | python3 -m json.tool
```

### Health check

```bash
curl http://localhost:8000/health
# → {"status": "ok", "model": "XGBoost"}
```

### Interactive docs

```
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

---

## Practical limitations — when to trust (and not trust) this model

| Situation | Trust level | Reason |
|-----------|-------------|--------|
| Established seller, 200–1500 km, standard category | ✅ HIGH | Most training data lives here |
| New / first-time seller | ⚠️ MEDIUM | Falls back to global stats; actual delivery unpredictable |
| Distance > 2000 km | ⚠️ MEDIUM | Thin training data; RMSE spikes |
| Nov / Dec / Jan orders | ⚠️ MEDIUM | Holiday logistics noise not fully captured |
| Very fast deliveries (≤ 5 days) | ⚠️ MEDIUM | Model under-represents this tail |
| Carrier disruptions, strikes, natural disasters | ❌ DO NOT USE | Out-of-distribution; model has no such signal |
| Returns / reverse logistics | ❌ DO NOT USE | Model trained only on forward deliveries |

---

## Error analysis highlights

Run `python src/error_analysis.py` after training to regenerate these findings.

- **Systematic bias**: The model slightly under-estimates for slow orders (>20 days) and over-estimates for express orders (<5 days) — the classic regression-toward-the-mean effect.
- **Worst categories**: Heavy/bulky items (furniture, large appliances) have the highest MAE due to irregular carrier handling.
- **Distance**: Long-haul routes (>2000 km) show the widest error spread — consider surfacing a wider UI range for these.
- **Seasonality**: November and December show elevated error; adding a `is_holiday_season` binary feature is recommended for v2.
- **New sellers**: When seller history is unavailable the model falls back to the global mean, which overestimates MAE by ~30 % relative to established sellers.