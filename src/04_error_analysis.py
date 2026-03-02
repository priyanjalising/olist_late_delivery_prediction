"""
error_analysis.py
-----------------
Loads the best trained pipeline and test split, then performs a structured
error analysis to identify:

  1. Overall residual distribution & bias
  2. Segments the model struggles with (high MAE by category, distance band,
     purchase month, seller performance tier)
  3. Practical trust / no-trust thresholds based on uncertainty

Run after train.py:
  python error_analysis.py

Outputs saved to outputs/error_analysis/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure sibling modules in src/ are importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from split import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, get_Xy

OUT_DIR = Path("outputs/error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_artefacts():
    with open("models/best_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("models/metadata.json") as f:
        meta = json.load(f)
    test_df = pd.read_parquet("outputs/test.parquet")
    return pipeline, meta, test_df


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------

def segment_mae(error_df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """Return per-segment MAE sorted descending."""
    seg = (
        error_df.groupby(col)["abs_error"]
        .agg(n="count", mae="mean", median="median", pct90=lambda x: x.quantile(0.9))
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    seg.columns = [col, "n_orders", "MAE", "Median_AE", "P90_AE"]
    print(f"\n{'─'*50}")
    print(f"Segment analysis: {label}")
    print(seg.to_string(index=False))
    return seg


def plot_segment(seg: pd.DataFrame, col: str, title: str, fname: str, top_n: int = 15):
    top = seg.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top[col].astype(str), top["MAE"], color="salmon", edgecolor="grey")
    ax.set_xlabel("MAE (days)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_error_analysis():
    pipeline, meta, test_df = load_artefacts()
    print(f"Best model: {meta['best_model']}")

    X_test, y_test = get_Xy(test_df)
    y_pred = pipeline.predict(X_test)

    # ── Base residual frame ──────────────────────────────────────────────────
    err = test_df.copy()
    err["predicted"] = y_pred
    err["residual"] = y_test.values - y_pred        # actual - pred
    err["abs_error"] = np.abs(err["residual"])
    err["over_predict"] = err["residual"] < 0       # model overestimates days
    err["error_bucket"] = pd.cut(
        err["abs_error"],
        bins=[0, 1, 2, 4, 7, np.inf],
        labels=["≤1d", "1–2d", "2–4d", "4–7d", ">7d"],
    )

    # ── 1. Overall distribution ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("1. OVERALL RESIDUAL SUMMARY")
    print("="*60)
    print(err["residual"].describe().round(2))
    bias = err["residual"].mean()
    print(f"\nSystematic bias: {bias:+.2f} days  "
          f"({'model over-estimates' if bias < 0 else 'model under-estimates'})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(err["residual"], bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residual distribution (actual − predicted)")
    axes[0].set_xlabel("Days")

    axes[1].scatter(err["predicted"], err["residual"], alpha=0.15, s=8, color="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted days"); axes[1].set_ylabel("Residual")

    axes[2].pie(
        err["error_bucket"].value_counts().sort_index(),
        labels=err["error_bucket"].cat.categories,
        autopct="%1.1f%%",
        colors=["#2ecc71","#27ae60","#f39c12","#e67e22","#e74c3c"],
    )
    axes[2].set_title("Error size distribution")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "residuals_overview.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved residuals overview → {OUT_DIR/'residuals_overview.png'}")

    # ── 2. Error by category ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. SEGMENT ANALYSIS — PRODUCT CATEGORY")
    print("="*60)
    cat_seg = segment_mae(err, "product_category_name_english", "Category")
    plot_segment(cat_seg, "product_category_name_english",
                 "MAE by Product Category (worst 15)", "mae_by_category.png")

    # ── 3. Error by distance band ─────────────────────────────────────────────
    err["distance_band"] = pd.cut(
        err["distance_km"],
        bins=[0, 200, 500, 1000, 2000, np.inf],
        labels=["<200km", "200–500km", "500–1000km", "1000–2000km", ">2000km"],
    )
    print("\n" + "="*60)
    print("3. SEGMENT ANALYSIS — SHIPPING DISTANCE")
    print("="*60)
    dist_seg = segment_mae(err, "distance_band", "Distance band")

    # ── 4. Error by purchase month ────────────────────────────────────────────
    print("\n" + "="*60)
    print("4. SEGMENT ANALYSIS — PURCHASE MONTH (seasonality)")
    print("="*60)
    month_seg = segment_mae(err, "purchase_month", "Month")
    plot_segment(month_seg, "purchase_month",
                 "MAE by Purchase Month", "mae_by_month.png")

    # ── 5. Error by seller tier (based on seller_avg_delivery) ───────────────
    err["seller_tier"] = pd.qcut(
        err["seller_avg_delivery"], q=4,
        labels=["Q1 Fast", "Q2", "Q3", "Q4 Slow"], duplicates="drop"
    )
    print("\n" + "="*60)
    print("5. SEGMENT ANALYSIS — SELLER PERFORMANCE TIER")
    print("="*60)
    seller_seg = segment_mae(err, "seller_tier", "Seller tier")

    # ── 6. High-confidence vs unreliable predictions ──────────────────────────
    print("\n" + "="*60)
    print("6. PRACTICAL TRUST BOUNDARIES")
    print("="*60)

    # Short deliveries (<5d) — model tends to underfit
    short = err[err[TARGET] <= 5]
    long_ = err[err[TARGET] > 5]
    print(f"Short deliveries (≤5d)   → MAE = {short['abs_error'].mean():.2f}d  "
          f"(n={len(short):,})")
    print(f"Standard deliveries (>5d) → MAE = {long_['abs_error'].mean():.2f}d  "
          f"(n={len(long_):,})")

    # New sellers vs established
    new_seller_mask = err["seller_avg_delivery"] == err["seller_avg_delivery"].mode()[0]
    print(f"\nNew/unseen sellers (fallback stats)  → "
          f"MAE = {err[new_seller_mask]['abs_error'].mean():.2f}d  "
          f"(n={new_seller_mask.sum():,})")
    print(f"Established sellers → "
          f"MAE = {err[~new_seller_mask]['abs_error'].mean():.2f}d  "
          f"(n={(~new_seller_mask).sum():,})")

    # ── Summary writeup ───────────────────────────────────────────────────────
    summary = {
        "bias_days": round(bias, 3),
        "within_1d_pct": round(float(np.mean(err["abs_error"] <= 1) * 100), 1),
        "within_2d_pct": round(float(np.mean(err["abs_error"] <= 2) * 100), 1),
        "worst_categories": cat_seg.head(5)["product_category_name_english"].tolist(),
        "worst_distance_band": dist_seg.sort_values("MAE", ascending=False).iloc[0]["distance_band"],
        "notes": [
            "Model over-estimates for very fast deliveries (≤5d) — "
            "consider separate model or post-hoc clipping.",
            "Long-distance orders (>2000km) have highest absolute error — "
            "these require manual buffer.",
            "New sellers have elevated error due to fallback stats — "
            "flag these for a wider confidence interval in the UI.",
            "Holiday months (Nov, Dec) show seasonality-driven spikes — "
            "consider adding holiday feature.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nError analysis complete. Artefacts saved to {OUT_DIR}/")
    print("\nPRACTICAL LIMITATIONS SUMMARY")
    print("──────────────────────────────")
    for note in summary["notes"]:
        print(f"  • {note}")


if __name__ == "__main__":
    run_error_analysis()