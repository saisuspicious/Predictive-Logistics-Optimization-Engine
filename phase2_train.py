import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

print("🔄 Loading the processed sample dataset...")
df = pd.read_csv("processed_logistics_data.csv")

# 1. We removed 'day_of_week' to completely eliminate the data type string bug!
features = ['trip_distance', 'pickup_hour', 'is_weekend', 'passenger_count']
target = 'duration_min'

# Drop any rows containing missing target or feature variables
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 2. Split into Training (80%) and Validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"⚡ Training LightGBM model on {len(X_train)} records...")

# 3. Configure the LightGBM Regressor
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

# 4. Train the model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=15)]
)

# 5. Evaluate predictions
predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, predictions)
print(f"\n📊 Model Performance Assessment:")
print(f"👉 Mean Absolute Error: {mae:.2f} minutes")

# 6. Save the trained file
model_filename = "delivery_time_model.pkl"
joblib.dump(model, model_filename)
print(f"✅ Phase 2 complete! Machine learning model saved as: {model_filename}")