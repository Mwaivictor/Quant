"""
Show Feature Vector X_t Structure
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from arbitrex.feature_engine.config import FeatureEngineConfig
from arbitrex.feature_engine.pipeline import FeaturePipeline

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=200, freq='h', tz='UTC')
df = pd.DataFrame({
    'timestamp_utc': dates,
    'symbol': 'EURUSD',
    'timeframe': '1H',
    'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
    'high': 100 + np.cumsum(np.random.randn(200) * 0.1) + 0.05,
    'low': 100 + np.cumsum(np.random.randn(200) * 0.1) - 0.05,
    'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
    'volume': np.random.uniform(1000, 2000, 200),
    'spread': np.random.uniform(0.0001, 0.0003, 200),
    'log_return_1': np.random.randn(200) * 0.001,
    'valid_bar': True
})

# Compute features
config = FeatureEngineConfig()
pipeline = FeaturePipeline(config)

feature_df, metadata = pipeline.compute_features(
    df,
    symbol='EURUSD',
    timeframe='1H',
    normalize=True
)

# Get feature vector at specific timestamp t
timestamp_t = dates[150]  # Choose timestamp at index 150
vector_t = pipeline.freeze_feature_vector(
    feature_df, 
    timestamp_t, 
    symbol='EURUSD', 
    timeframe='1H',
    ml_only=True
)

print("=" * 80)
print("FEATURE VECTOR X_t STRUCTURE")
print("=" * 80)
print()

print(f"Timestamp t: {timestamp_t}")
print(f"Symbol: {vector_t.symbol}")
print(f"Timeframe: {vector_t.timeframe}")
print()

print(f"Feature Vector X_t Shape: ({len(vector_t.feature_values)},)")
print(f"Config Version: {vector_t.feature_version}")
print(f"ML Ready: {vector_t.is_ml_ready}")
print()

print("-" * 80)
print("FEATURE VECTOR X_t COMPONENTS:")
print("-" * 80)
print()

# Display each feature with its value
for i, (name, value) in enumerate(zip(vector_t.feature_names, vector_t.feature_values)):
    category = ""
    if "return" in name or "momentum" in name:
        category = "[A: Returns/Momentum]"
    elif "vol" in name or "atr" in name:
        category = "[B: Volatility]"
    elif "ma" in name or "distance" in name:
        category = "[C: Trend]"
    elif "efficiency" in name or "range" in name:
        category = "[D: Efficiency]"
    elif "regime" in name or "stress" in name:
        category = "[E: Regime]"
    elif "spread" in name:
        category = "[F: Execution]"
    
    print(f"X_t[{i:2d}] = {value:+8.4f}  |  {name:30s} {category}")

print()
print("-" * 80)
print("FEATURE VECTOR X_t AS NUMPY ARRAY:")
print("-" * 80)
print()
print(f"X_t = {vector_t.feature_values}")
print()

print("-" * 80)
print("STATISTICAL PROPERTIES:")
print("-" * 80)
print()
print(f"Mean:     {np.mean(vector_t.feature_values):+8.4f}")
print(f"Std Dev:  {np.std(vector_t.feature_values):8.4f}")
print(f"Min:      {np.min(vector_t.feature_values):+8.4f}")
print(f"Max:      {np.max(vector_t.feature_values):+8.4f}")
print(f"Range:    {np.ptp(vector_t.feature_values):8.4f}")
print()

print("-" * 80)
print("USAGE IN ML MODEL:")
print("-" * 80)
print()
print("""
# Model prediction with X_t
y_pred = model.predict(X_t.reshape(1, -1))  # Add batch dimension

# Or batch prediction with multiple timestamps
X_batch = np.vstack([X_t1, X_t2, X_t3, ...])  # Shape: (n_samples, n_features)
y_batch = model.predict(X_batch)
""")

print("=" * 80)
print(f"✓ Feature vector X_t contains {len(vector_t.feature_values)} normalized features")
print(f"✓ All features are z-score normalized (mean≈0, std≈1)")
print(f"✓ Vector is immutable and versioned with hash: {vector_t.feature_version[:12]}...")
print("=" * 80)
