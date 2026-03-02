# %% [markdown]
# # Delivery Window Prediction for Olist Marketplace
# 
# **Goal**: Predict a delivery window (e.g., 10th–90th percentile) instead of a single point estimate.  
# **Why**: Delivery times are inherently noisy; a range sets better expectations and reduces customer complaints.
# 
# **Approach**:
# - Use quantile regression to predict lower (0.1) and upper (0.9) quantiles.
# - Evaluate coverage (percentage of true values within the predicted interval) and interval width.
# - Compare with a simple baseline (e.g., global percentiles).

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
# ! pip install lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

from pathlib import Path
warnings.filterwarnings('ignore')

from math import radians, sin, cos, sqrt, asin

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

# Load datasets
print("Loading datasets...")
DATA_DIR = Path("C:/Users/priya/Downloads/olist/data")

print("Loading datasets...")
orders = pd.read_csv(DATA_DIR / 'olist_orders_dataset.csv')
items = pd.read_csv(DATA_DIR / 'olist_order_items_dataset.csv')
products = pd.read_csv(DATA_DIR / 'olist_products_dataset.csv')
sellers = pd.read_csv(DATA_DIR / 'olist_sellers_dataset.csv')
customers = pd.read_csv(DATA_DIR / 'olist_customers_dataset.csv')
geolocation = pd.read_csv(DATA_DIR / 'olist_geolocation_dataset.csv')
payments = pd.read_csv(DATA_DIR / 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(DATA_DIR / 'olist_order_reviews_dataset.csv')
category_translation = pd.read_csv(DATA_DIR / 'product_category_name_translation.csv')
print("Datasets loaded successfully!")

print(f"Orders: {len(orders)}")
print(f"Items: {len(items)}")
print(f"Products: {len(products)}")
print(f"Sellers: {len(sellers)}")
print(f"Customers: {len(customers)}")
print(f"Geolocation: {len(geolocation)}")

# %% [markdown]
# ## 2. Data Preparation and Target Definition

# %%
# Convert date columns to datetime
date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
             'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    if col in orders.columns:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')

# Compute total delivery time in days (target)
orders['delivery_time_days'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / (24*3600)

# Compute seller processing time (from approval to carrier delivery)
orders['processing_days'] = (orders['order_delivered_carrier_date'] - orders['order_approved_at']).dt.total_seconds() / (24*3600)

# Compute transit time (from carrier to customer)
orders['transit_days'] = (orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']).dt.total_seconds() / (24*3600)

# Clip unrealistic values (data quality)
orders['processing_days'] = orders['processing_days'].clip(lower=0, upper=30)
orders['transit_days'] = orders['transit_days'].clip(lower=0, upper=60)

# Keep only delivered orders (where delivery_time_days is not null)
delivered_orders = orders[orders['delivery_time_days'].notna()].copy()
print(f"Delivered orders: {len(delivered_orders)}")

# Remove extreme outliers (>99th percentile)
upper_limit = delivered_orders['delivery_time_days'].quantile(0.99)
delivered_orders = delivered_orders[delivered_orders['delivery_time_days'] <= upper_limit]
print(f"After removing extreme outliers (>99th percentile): {len(delivered_orders)}")

# %% [markdown]
# ## 3. Build Order‑Level Feature Table
# (Same as before, but we'll keep all features and impute missing values.)

# %%
# Merge items to get products and sellers per order
order_items = delivered_orders[['order_id', 'order_purchase_timestamp']].merge(
    items[['order_id', 'product_id', 'seller_id', 'price', 'freight_value']],
    on='order_id', how='left'
)

# Merge product info
order_items = order_items.merge(
    products[['product_id', 'product_category_name', 'product_weight_g',
              'product_length_cm', 'product_height_cm', 'product_width_cm']],
    on='product_id', how='left'
)

# Merge seller location
order_items = order_items.merge(
    sellers[['seller_id', 'seller_zip_code_prefix', 'seller_state']],
    on='seller_id', how='left'
)

# Merge customer location (via customers)
customer_geo = customers[['customer_id', 'customer_zip_code_prefix', 'customer_state']]
orders_with_customer = delivered_orders[['order_id', 'customer_id']].merge(
    customer_geo, on='customer_id', how='left')
order_items = order_items.merge(
    orders_with_customer[['order_id', 'customer_zip_code_prefix', 'customer_state']],
    on='order_id', how='left'
)

# Add geolocation coordinates for seller and customer zip prefixes
geolocation_unique = geolocation.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()
geolocation_unique.columns = ['zip_code_prefix', 'lat', 'lng']

# Merge seller coordinates
order_items = order_items.merge(
    geolocation_unique,
    left_on='seller_zip_code_prefix',
    right_on='zip_code_prefix',
    how='left'
).rename(columns={'lat': 'seller_lat', 'lng': 'seller_lng'}).drop('zip_code_prefix', axis=1)

# Merge customer coordinates
order_items = order_items.merge(
    geolocation_unique,
    left_on='customer_zip_code_prefix',
    right_on='zip_code_prefix',
    how='left'
).rename(columns={'lat': 'cust_lat', 'lng': 'cust_lng'}).drop('zip_code_prefix', axis=1)

# Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

order_items['distance_km'] = order_items.apply(
    lambda row: haversine(row['seller_lat'], row['seller_lng'],
                          row['cust_lat'], row['cust_lng']),
    axis=1
)

# Temporal features
order_items['purchase_month'] = order_items['order_purchase_timestamp'].dt.month
order_items['purchase_day_of_week'] = order_items['order_purchase_timestamp'].dt.dayofweek
order_items['purchase_hour'] = order_items['order_purchase_timestamp'].dt.hour
order_items['is_weekend'] = (order_items['purchase_day_of_week'] >= 5).astype(int)

# Product features
order_items['product_volume_cm3'] = (order_items['product_length_cm'] *
                                      order_items['product_height_cm'] *
                                      order_items['product_width_cm']).fillna(0)
order_items['product_weight_kg'] = order_items['product_weight_g'] / 1000
order_items['freight_ratio'] = order_items['freight_value'] / (order_items['price'] + 1e-5)

# Category translation
order_items = order_items.merge(category_translation, on='product_category_name', how='left')
order_items['product_category_name_english'].fillna('unknown', inplace=True)

# Aggregate to order level
order_features = order_items.groupby('order_id').agg({
    'distance_km': 'mean',
    'product_weight_kg': 'sum',
    'product_volume_cm3': 'sum',
    'price': 'sum',
    'freight_value': 'sum',
    'freight_ratio': 'mean',
    'purchase_month': 'first',
    'purchase_day_of_week': 'first',
    'purchase_hour': 'first',
    'is_weekend': 'first',
    'product_category_name_english': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
    'seller_id': 'first',
    'seller_zip_code_prefix': 'first',
    'customer_zip_code_prefix': 'first',
    'seller_state': 'first',
    'customer_state': 'first'
}).reset_index()

# Merge back target variables (delivery_time_days, processing_days, transit_days)
order_features = order_features.merge(
    delivered_orders[['order_id', 'delivery_time_days', 'processing_days', 'transit_days']],
    on='order_id', how='inner'
)

print(f"Final feature set shape: {order_features.shape}")

# %% [markdown]
# ## 4. Time‑Based Train/Test Split

# %%
# Get purchase timestamp for sorting
order_time = orders[['order_id', 'order_purchase_timestamp']].drop_duplicates()
order_features = order_features.merge(order_time, on='order_id', how='left')
order_features = order_features.sort_values('order_purchase_timestamp')

split_idx = int(len(order_features) * 0.8)
train_df = order_features.iloc[:split_idx].copy()
test_df = order_features.iloc[split_idx:].copy()

print(f"Train period: {train_df['order_purchase_timestamp'].min()} to {train_df['order_purchase_timestamp'].max()}")
print(f"Test period: {test_df['order_purchase_timestamp'].min()} to {test_df['order_purchase_timestamp'].max()}")
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# %% [markdown]
# ## 5. Feature Engineering (using training data only)
# Same as before, with careful imputation.

# %%
# Global averages for filling missing values
global_avg_delivery = train_df['delivery_time_days'].mean()
global_avg_processing = train_df['processing_days'].mean()
global_avg_transit = train_df['transit_days'].mean()

# 5.1 Seller historical stats
seller_processing_stats = train_df.groupby('seller_id')['processing_days'].agg(['mean', 'std']).reset_index()
seller_processing_stats.columns = ['seller_id', 'seller_avg_processing', 'seller_std_processing']

seller_transit_stats = train_df.groupby('seller_id')['transit_days'].agg(['mean', 'std']).reset_index()
seller_transit_stats.columns = ['seller_id', 'seller_avg_transit', 'seller_std_transit']

seller_total_stats = train_df.groupby('seller_id')['delivery_time_days'].agg(['mean', 'std']).reset_index()
seller_total_stats.columns = ['seller_id', 'seller_avg_delivery', 'seller_std_delivery']

train_df = train_df.merge(seller_processing_stats, on='seller_id', how='left')
train_df = train_df.merge(seller_transit_stats, on='seller_id', how='left')
train_df = train_df.merge(seller_total_stats, on='seller_id', how='left')

test_df = test_df.merge(seller_processing_stats, on='seller_id', how='left')
test_df = test_df.merge(seller_transit_stats, on='seller_id', how='left')
test_df = test_df.merge(seller_total_stats, on='seller_id', how='left')

for df in [train_df, test_df]:
    df['seller_avg_processing'].fillna(global_avg_processing, inplace=True)
    df['seller_std_processing'].fillna(0, inplace=True)
    df['seller_avg_transit'].fillna(global_avg_transit, inplace=True)
    df['seller_std_transit'].fillna(0, inplace=True)
    df['seller_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['seller_std_delivery'].fillna(0, inplace=True)

# 5.2 Category-level averages
cat_processing_avg = train_df.groupby('product_category_name_english')['processing_days'].mean().reset_index()
cat_processing_avg.columns = ['product_category_name_english', 'cat_avg_processing']
train_df = train_df.merge(cat_processing_avg, on='product_category_name_english', how='left')
test_df = test_df.merge(cat_processing_avg, on='product_category_name_english', how='left')
for df in [train_df, test_df]:
    df['cat_avg_processing'].fillna(global_avg_processing, inplace=True)

cat_transit_avg = train_df.groupby('product_category_name_english')['transit_days'].mean().reset_index()
cat_transit_avg.columns = ['product_category_name_english', 'cat_avg_transit']
train_df = train_df.merge(cat_transit_avg, on='product_category_name_english', how='left')
test_df = test_df.merge(cat_transit_avg, on='product_category_name_english', how='left')
for df in [train_df, test_df]:
    df['cat_avg_transit'].fillna(global_avg_transit, inplace=True)

# 5.3 Seller total orders (experience proxy)
seller_order_counts = train_df.groupby('seller_id').size().reset_index(name='seller_total_orders')
train_df = train_df.merge(seller_order_counts, on='seller_id', how='left')
test_df = test_df.merge(seller_order_counts, on='seller_id', how='left')
for df in [train_df, test_df]:
    df['seller_total_orders'].fillna(0, inplace=True)

# 5.4 Seller tenure
seller_first_order = train_df.groupby('seller_id')['order_purchase_timestamp'].min().reset_index()
seller_first_order.columns = ['seller_id', 'seller_first_order_date']
train_df = train_df.merge(seller_first_order, on='seller_id', how='left')
test_df = test_df.merge(seller_first_order, on='seller_id', how='left')
for df in [train_df, test_df]:
    df['seller_tenure_days'] = (df['order_purchase_timestamp'] - df['seller_first_order_date']).dt.days
    df['seller_tenure_days'].fillna(0, inplace=True)
    df.drop('seller_first_order_date', axis=1, inplace=True)

# 5.5 Route feature
train_df['route'] = train_df['seller_zip_code_prefix'].astype(str) + '_' + train_df['customer_zip_code_prefix'].astype(str)
test_df['route'] = test_df['seller_zip_code_prefix'].astype(str) + '_' + test_df['customer_zip_code_prefix'].astype(str)

route_stats = train_df.groupby('route')['delivery_time_days'].agg(['mean', 'std']).reset_index()
route_stats.columns = ['route', 'route_avg_delivery', 'route_std_delivery']

train_df = train_df.merge(route_stats, on='route', how='left')
test_df = test_df.merge(route_stats, on='route', how='left')
for df in [train_df, test_df]:
    df['route_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['route_std_delivery'].fillna(train_df['route_std_delivery'].median(), inplace=True)

# 5.6 Rolling seller averages (no leakage)
train_df = train_df.sort_values('order_purchase_timestamp')
train_df['seller_roll_avg_delivery'] = train_df.groupby('seller_id')['delivery_time_days'] \
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).mean())
train_df['seller_roll_std_delivery'] = train_df.groupby('seller_id')['delivery_time_days'] \
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).std())
train_df['seller_roll_avg_delivery'].fillna(global_avg_delivery, inplace=True)
train_df['seller_roll_std_delivery'].fillna(train_df['seller_roll_std_delivery'].median(), inplace=True)

seller_roll_stats = train_df.groupby('seller_id')[['seller_roll_avg_delivery', 'seller_roll_std_delivery']].last().reset_index()
test_df = test_df.merge(seller_roll_stats, on='seller_id', how='left')
test_df['seller_roll_avg_delivery'].fillna(global_avg_delivery, inplace=True)
test_df['seller_roll_std_delivery'].fillna(train_df['seller_roll_std_delivery'].median(), inplace=True)

# 5.7 State pair feature
train_df['state_pair'] = train_df['seller_state'] + '_' + train_df['customer_state']
test_df['state_pair'] = test_df['seller_state'] + '_' + test_df['customer_state']

state_stats = train_df.groupby('state_pair')['delivery_time_days'].agg(['mean', 'std']).reset_index()
state_stats.columns = ['state_pair', 'state_pair_avg_delivery', 'state_pair_std_delivery']

train_df = train_df.merge(state_stats, on='state_pair', how='left')
test_df = test_df.merge(state_stats, on='state_pair', how='left')
for df in [train_df, test_df]:
    df['state_pair_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['state_pair_std_delivery'].fillna(train_df['state_pair_std_delivery'].median(), inplace=True)

# 5.8 Order complexity
order_complexity = train_df.groupby('order_id').size().reset_index(name='num_items_in_order')
train_df = train_df.merge(order_complexity, on='order_id', how='left')
test_df = test_df.merge(order_complexity, on='order_id', how='left')
for df in [train_df, test_df]:
    df['num_items_in_order'].fillna(1, inplace=True)

# 5.9 Fill distance_km
train_df['distance_km'].fillna(train_df['distance_km'].median(), inplace=True)
test_df['distance_km'].fillna(train_df['distance_km'].median(), inplace=True)

print(f"Train shape after feature engineering: {train_df.shape}")
print(f"Test shape after feature engineering: {test_df.shape}")

# %% [markdown]
# ## 6. Prepare Feature Matrix

# %%
numeric_features = [
    'distance_km', 'product_weight_kg', 'product_volume_cm3', 'price', 'freight_value',
    'freight_ratio', 'purchase_hour',
    'seller_avg_delivery', 'seller_std_delivery',
    'seller_avg_processing', 'seller_std_processing',
    'seller_avg_transit', 'seller_std_transit',
    'cat_avg_processing', 'cat_avg_transit',
    'seller_total_orders', 'seller_tenure_days',
    'route_avg_delivery', 'route_std_delivery',
    'seller_roll_avg_delivery', 'seller_roll_std_delivery',
    'state_pair_avg_delivery', 'state_pair_std_delivery',
    'num_items_in_order'
]

categorical_features = [
    'purchase_month', 'purchase_day_of_week', 'is_weekend',
    'product_category_name_english'
]

X_train = train_df[numeric_features + categorical_features]
X_test = test_df[numeric_features + categorical_features]

# Targets (original days, not log)
y_train = train_df['delivery_time_days']
y_test = test_df['delivery_time_days']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# %% [markdown]
# ## 7. Preprocessing Pipeline

# %%
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %% [markdown]
# ## 8. Quantile Regression Models
# We'll use `GradientBoostingRegressor` with quantile loss.  
# Quantiles: 0.1 (lower bound), 0.5 (median), 0.9 (upper bound).

# %%
quantiles = [0.1, 0.5, 0.9]
models = {}

# for q in quantiles:
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', GradientBoostingRegressor(
#             loss='quantile', alpha=q,
#             n_estimators=100, max_depth=5, random_state=42
#         ))
#     ])
#     model.fit(X_train, y_train)
#     models[q] = model
#     print(f"Trained model for quantile {q}")

# # %% [markdown]
# # ## 9. Predictions on Test Set

# # %%
# y_pred_lower = models[0.1].predict(X_test)
# y_pred_median = models[0.5].predict(X_test)
# y_pred_upper = models[0.9].predict(X_test)

# # Ensure lower <= median <= upper (if not, sort them)
# y_pred_lower, y_pred_median, y_pred_upper = np.sort([y_pred_lower, y_pred_median, y_pred_upper], axis=0)

from lightgbm import LGBMRegressor

# Log transform target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Preprocessor (same as before)
preprocessor = ColumnTransformer(...)

quantiles = [0.1, 0.5, 0.9]
models = {}

for q in quantiles:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        ))
    ])
    model.fit(X_train, y_train_log)
    models[q] = model

# Predict on log scale
y_pred_lower_log = models[0.1].predict(X_test)
y_pred_median_log = models[0.5].predict(X_test)
y_pred_upper_log = models[0.9].predict(X_test)

# Back-transform to original days
y_pred_lower = np.expm1(y_pred_lower_log)
y_pred_median = np.expm1(y_pred_median_log)
y_pred_upper = np.expm1(y_pred_upper_log)

# Ensure monotonicity (if needed)
y_pred_lower, y_pred_median, y_pred_upper = np.sort([y_pred_lower, y_pred_median, y_pred_upper], axis=0)
# %% [markdown]
# ## 10. Evaluation Metrics

# %%
# 10.1 Coverage: percentage of true values within predicted interval
inside_interval = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
coverage = np.mean(inside_interval) * 100

# 10.2 Average interval width
interval_width = np.mean(y_pred_upper - y_pred_lower)

# 10.3 Pinball loss for each quantile (lower is better)
pinball_10 = mean_pinball_loss(y_test, y_pred_lower, alpha=0.1)
pinball_50 = mean_pinball_loss(y_test, y_pred_median, alpha=0.5)  # same as MAE/2
pinball_90 = mean_pinball_loss(y_test, y_pred_upper, alpha=0.9)

# 10.4 Median absolute error
mae_median = mean_absolute_error(y_test, y_pred_median)

print("\n=== Delivery Window Performance ===")
print(f"Coverage (10th–90th percentile): {coverage:.1f}%")
print(f"Average interval width: {interval_width:.2f} days")
print(f"Pinball loss (0.1): {pinball_10:.3f}")
print(f"Pinball loss (0.5): {pinball_50:.3f}")
print(f"Pinball loss (0.9): {pinball_90:.3f}")
print(f"Median MAE: {mae_median:.2f} days")

# %% [markdown]
# ## 11. Comparison with Simple Baseline
# Baseline: use global quantiles from training set.

# %%
global_lower = np.percentile(y_train, 10)
global_upper = np.percentile(y_train, 90)
global_interval_width = global_upper - global_lower
global_coverage = np.mean((y_test >= global_lower) & (y_test <= global_upper)) * 100

print("\n=== Baseline (Global Percentiles) ===")
print(f"Global 10th percentile: {global_lower:.2f} days")
print(f"Global 90th percentile: {global_upper:.2f} days")
print(f"Coverage on test set: {global_coverage:.1f}%")
print(f"Interval width: {global_interval_width:.2f} days")

# %% [markdown]
# ## 12. Visualization

# %%
# Plot a sample of predictions vs actuals
sample_idx = np.random.choice(len(y_test), size=200, replace=False)
sample_idx.sort()
x_plot = np.arange(len(sample_idx))

plt.figure(figsize=(12, 6))
plt.fill_between(x_plot, y_pred_lower[sample_idx], y_pred_upper[sample_idx],
                 alpha=0.3, color='blue', label='Predicted 10th–90th interval')
plt.plot(x_plot, y_test.values[sample_idx], 'o', markersize=3, color='red', label='Actual')
plt.plot(x_plot, y_pred_median[sample_idx], '-', color='blue', label='Predicted median')
plt.xlabel('Sample order (sorted)')
plt.ylabel('Delivery time (days)')
plt.title('Delivery Window Predictions (Random Sample of Test Set)')
plt.legend()
plt.tight_layout()
plt.show()

# Coverage plot: sort by actual and show intervals
sorted_idx = np.argsort(y_test)
sorted_y = y_test.iloc[sorted_idx].values
sorted_lower = y_pred_lower[sorted_idx]
sorted_upper = y_pred_upper[sorted_idx]
sorted_median = y_pred_median[sorted_idx]

plt.figure(figsize=(12, 6))
plt.fill_between(np.arange(len(sorted_y)), sorted_lower, sorted_upper,
                 alpha=0.3, color='blue', label='Predicted 10th–90th interval')
plt.plot(np.arange(len(sorted_y)), sorted_y, 'o', markersize=2, color='red', label='Actual')
plt.plot(np.arange(len(sorted_y)), sorted_median, '-', color='blue', label='Predicted median')
plt.xlabel('Test samples (sorted by actual delivery time)')
plt.ylabel('Delivery time (days)')
plt.title('Delivery Window Predictions (Test Set Sorted by Actual)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Interpretation and Business Use
# 
# - The model achieves **coverage of X%** (ideally close to 80%) with an average interval width of **Y days**.
# - Compared to the global baseline, the model provides **narrower intervals** while maintaining similar or better coverage.
# - For a new order, we can display: *"Estimated delivery: between L and U days"*.
# - This sets realistic expectations and reduces the risk of over‑promising.
# 
# **Next Steps**:
# - Monitor coverage over time; retrain periodically.
# - If coverage is too low, adjust quantiles (e.g., use 5th–95th).
# - Consider adding more features (carrier, road distance) to improve accuracy.

# %% [markdown]
# ## 14. Save Models

# %%
import joblib
for q, model in models.items():
    joblib.dump(model, f'delivery_window_model_q{int(q*100)}.pkl')
print("Models saved.")