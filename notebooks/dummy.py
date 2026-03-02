# %% [markdown]
# # Delivery Time Estimator for Olist Marketplace (Corrected)
# 
# **Goal**: Predict the number of days between order placement and delivery to set accurate expectations for customers.
# 
# **Why**: Reducing delivery-related complaints (which account for ~45% of bad reviews) by providing realistic estimates.
# 
# **Key Improvements in this version**:
# - **No data loss**: Missing values are imputed, not dropped, preserving all rows.
# - **Deduplicated features**: Each feature appears only once in the correct category.
# - **Proper imputation**: After merging any new feature derived from training, NaNs are filled with global statistics.
# - **Time-based split** preserved.
# - **Log‑transformed target** with back‑transformation for evaluation.
# 
# **Data Sources**:
# - `orders_dataset`, `items_dataset`, `products_dataset`, `sellers_dataset`, `customers_dataset`, `geolocation_dataset`, `category_translation`.

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from math import radians, sin, cos, sqrt, asin

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

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
# We'll create one row per order by aggregating items.

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
# ## 5. Advanced Feature Engineering (using training data only)

# %%
# Global averages for filling missing values
global_avg_delivery = train_df['delivery_time_days'].mean()
global_avg_processing = train_df['processing_days'].mean()
global_avg_transit = train_df['transit_days'].mean()

# 5.1 Seller historical stats (processing, transit, total)
seller_processing_stats = train_df.groupby('seller_id')['processing_days'].agg(['mean', 'std']).reset_index()
seller_processing_stats.columns = ['seller_id', 'seller_avg_processing', 'seller_std_processing']

seller_transit_stats = train_df.groupby('seller_id')['transit_days'].agg(['mean', 'std']).reset_index()
seller_transit_stats.columns = ['seller_id', 'seller_avg_transit', 'seller_std_transit']

seller_total_stats = train_df.groupby('seller_id')['delivery_time_days'].agg(['mean', 'std']).reset_index()
seller_total_stats.columns = ['seller_id', 'seller_avg_delivery', 'seller_std_delivery']

# Merge into train and test
train_df = train_df.merge(seller_processing_stats, on='seller_id', how='left')
train_df = train_df.merge(seller_transit_stats, on='seller_id', how='left')
train_df = train_df.merge(seller_total_stats, on='seller_id', how='left')

test_df = test_df.merge(seller_processing_stats, on='seller_id', how='left')
test_df = test_df.merge(seller_transit_stats, on='seller_id', how='left')
test_df = test_df.merge(seller_total_stats, on='seller_id', how='left')

# Fill missing for new sellers in test
for df in [train_df, test_df]:
    df['seller_avg_processing'].fillna(global_avg_processing, inplace=True)
    df['seller_std_processing'].fillna(0, inplace=True)
    df['seller_avg_transit'].fillna(global_avg_transit, inplace=True)
    df['seller_std_transit'].fillna(0, inplace=True)
    df['seller_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['seller_std_delivery'].fillna(0, inplace=True)

# 5.2 Category-level processing average
cat_processing_avg = train_df.groupby('product_category_name_english')['processing_days'].mean().reset_index()
cat_processing_avg.columns = ['product_category_name_english', 'cat_avg_processing']
train_df = train_df.merge(cat_processing_avg, on='product_category_name_english', how='left')
test_df = test_df.merge(cat_processing_avg, on='product_category_name_english', how='left')
for df in [train_df, test_df]:
    df['cat_avg_processing'].fillna(global_avg_processing, inplace=True)

# 5.3 Category-level transit average
cat_transit_avg = train_df.groupby('product_category_name_english')['transit_days'].mean().reset_index()
cat_transit_avg.columns = ['product_category_name_english', 'cat_avg_transit']
train_df = train_df.merge(cat_transit_avg, on='product_category_name_english', how='left')
test_df = test_df.merge(cat_transit_avg, on='product_category_name_english', how='left')
for df in [train_df, test_df]:
    df['cat_avg_transit'].fillna(global_avg_transit, inplace=True)

# 5.4 Seller total orders (experience proxy)
seller_order_counts = train_df.groupby('seller_id').size().reset_index(name='seller_total_orders')
train_df = train_df.merge(seller_order_counts, on='seller_id', how='left')
test_df = test_df.merge(seller_order_counts, on='seller_id', how='left')
for df in [train_df, test_df]:
    df['seller_total_orders'].fillna(0, inplace=True)

# 5.5 Seller tenure (days since first order)
seller_first_order = train_df.groupby('seller_id')['order_purchase_timestamp'].min().reset_index()
seller_first_order.columns = ['seller_id', 'seller_first_order_date']
train_df = train_df.merge(seller_first_order, on='seller_id', how='left')
test_df = test_df.merge(seller_first_order, on='seller_id', how='left')
for df in [train_df, test_df]:
    df['seller_tenure_days'] = (df['order_purchase_timestamp'] - df['seller_first_order_date']).dt.days
    df['seller_tenure_days'].fillna(0, inplace=True)
    df.drop('seller_first_order_date', axis=1, inplace=True)

# 5.6 Route feature (seller_zip + customer_zip)
train_df['route'] = train_df['seller_zip_code_prefix'].astype(str) + '_' + train_df['customer_zip_code_prefix'].astype(str)
test_df['route'] = test_df['seller_zip_code_prefix'].astype(str) + '_' + test_df['customer_zip_code_prefix'].astype(str)

route_stats = train_df.groupby('route')['delivery_time_days'].agg(['mean', 'std']).reset_index()
route_stats.columns = ['route', 'route_avg_delivery', 'route_std_delivery']

train_df = train_df.merge(route_stats, on='route', how='left')
test_df = test_df.merge(route_stats, on='route', how='left')
for df in [train_df, test_df]:
    df['route_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['route_std_delivery'].fillna(train_df['route_std_delivery'].median(), inplace=True)

# 5.7 Rolling seller averages (no leakage)
train_df = train_df.sort_values('order_purchase_timestamp')
train_df['seller_roll_avg_delivery'] = train_df.groupby('seller_id')['delivery_time_days'] \
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).mean())
train_df['seller_roll_std_delivery'] = train_df.groupby('seller_id')['delivery_time_days'] \
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).std())
train_df['seller_roll_avg_delivery'].fillna(global_avg_delivery, inplace=True)
train_df['seller_roll_std_delivery'].fillna(train_df['seller_roll_std_delivery'].median(), inplace=True)

# For test, use the last known rolling stats from train (or global)
seller_roll_stats = train_df.groupby('seller_id')[['seller_roll_avg_delivery', 'seller_roll_std_delivery']].last().reset_index()
test_df = test_df.merge(seller_roll_stats, on='seller_id', how='left')
test_df['seller_roll_avg_delivery'].fillna(global_avg_delivery, inplace=True)
test_df['seller_roll_std_delivery'].fillna(train_df['seller_roll_std_delivery'].median(), inplace=True)

# 5.8 State pair feature
train_df['state_pair'] = train_df['seller_state'] + '_' + train_df['customer_state']
test_df['state_pair'] = test_df['seller_state'] + '_' + test_df['customer_state']

state_stats = train_df.groupby('state_pair')['delivery_time_days'].agg(['mean', 'std']).reset_index()
state_stats.columns = ['state_pair', 'state_pair_avg_delivery', 'state_pair_std_delivery']

train_df = train_df.merge(state_stats, on='state_pair', how='left')
test_df = test_df.merge(state_stats, on='state_pair', how='left')
for df in [train_df, test_df]:
    df['state_pair_avg_delivery'].fillna(global_avg_delivery, inplace=True)
    df['state_pair_std_delivery'].fillna(train_df['state_pair_std_delivery'].median(), inplace=True)

# 5.9 Order complexity (number of items)
order_complexity = train_df.groupby('order_id').size().reset_index(name='num_items_in_order')
train_df = train_df.merge(order_complexity, on='order_id', how='left')
test_df = test_df.merge(order_complexity, on='order_id', how='left')
for df in [train_df, test_df]:
    df['num_items_in_order'].fillna(1, inplace=True)

# 5.10 Fill distance_km missing values (should be few)
train_df['distance_km'].fillna(train_df['distance_km'].median(), inplace=True)
test_df['distance_km'].fillna(train_df['distance_km'].median(), inplace=True)

# Finally, drop any remaining rows with missing target (none) – but we keep all rows.
# We will rely on the pipeline's imputers for any leftover NaNs.
print(f"Train shape after feature engineering: {train_df.shape}")
print(f"Test shape after feature engineering: {test_df.shape}")

# %% [markdown]
# ## 6. Prepare Feature Matrix (with log target)

# %%
# Define feature columns (unique, no duplicates)
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

# Log transform target
train_df['delivery_time_log'] = np.log1p(train_df['delivery_time_days'])
test_df['delivery_time_log'] = np.log1p(test_df['delivery_time_days'])

X_train = train_df[numeric_features + categorical_features]
y_train = train_df['delivery_time_log']
X_test = test_df[numeric_features + categorical_features]
y_test = test_df['delivery_time_log']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# %% [markdown]
# ## 7. Preprocessing Pipelines

# %%
# Numeric pipeline: impute median and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing with 'missing' and one-hot encode
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
# ## 8. Model Training and Comparison

# %%
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)

    # Convert back to original days
    y_pred_days = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)

    mae = mean_absolute_error(y_test_days, y_pred_days)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred_days))
    r2 = r2_score(y_test_days, y_pred_days)
    within_2_days = np.mean(np.abs(y_test_days - y_pred_days) <= 2) * 100

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Within 2 days (%)': within_2_days
    })

    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, Within 2d: {within_2_days:.1f}%")

# %% [markdown]
# ## 9. Model Comparison Table

# %%
results_df = pd.DataFrame(results).sort_values('MAE')
print("\n=== Model Performance Comparison ===")
print(results_df.to_string(index=False))

# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE', 'RMSE', 'Within 2 days (%)']
for i, metric in enumerate(metrics):
    sorted_df = results_df.sort_values(metric, ascending=(metric != 'Within 2 days (%)'))
    sorted_df.plot(x='Model', y=metric, kind='bar', ax=axes[i], legend=False, color='skyblue')
    axes[i].set_title(metric)
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Feature Importance (from best tree‑based model)

# %%
best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']
print(f"Best model: {best_model_name}")

# Get feature names after preprocessing
preprocessor.fit(X_train)
cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

if best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', models[best_model_name])])
    best_pipeline.fit(X_train, y_train)

    importances = best_pipeline.named_steps['regressor'].feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})\
                 .sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp.head(20)['feature'], feat_imp.head(20)['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("Best model is linear; feature importance not directly available.")

# %% [markdown]
# ## 11. Residual Analysis

# %%
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', models[best_model_name])])
best_pipeline.fit(X_train, y_train)
y_pred_log = best_pipeline.predict(X_test)
y_pred_days = np.expm1(y_pred_log)
y_test_days = np.expm1(y_test)

residuals = y_test_days - y_pred_days

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_days, residuals, alpha=0.3)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted (days)')
axes[0].set_ylabel('Residual (days)')
axes[0].set_title('Residual Plot')

axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_xlabel('Residual (days)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Save Model

# %%
joblib.dump(best_pipeline, 'delivery_time_estimator_corrected.pkl')
print("Model saved as 'delivery_time_estimator_corrected.pkl'")