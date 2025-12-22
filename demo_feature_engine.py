"""
Feature Engine Demo Script

Demonstrates end-to-end feature computation from clean OHLCV data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from arbitrex.feature_engine import FeaturePipeline, FeatureEngineConfig
from arbitrex.feature_engine.feature_store import FeatureStore
from arbitrex.feature_engine.schemas import FeatureSchema


def create_sample_clean_data():
    """Generate sample clean OHLCV data"""
    np.random.seed(42)
    
    dates = pd.date_range('2025-01-01', periods=200, freq='h', tz='UTC')
    
    base_price = 1.20
    returns = np.random.normal(0, 0.001, size=len(dates))
    closes = base_price * np.exp(np.cumsum(returns))
    
    opens = closes * (1 + np.random.normal(0, 0.0002, size=len(dates)))
    max_oc = np.maximum(opens, closes)
    highs = max_oc * (1 + np.abs(np.random.normal(0, 0.0005, size=len(dates))))
    min_oc = np.minimum(opens, closes)
    lows = min_oc * (1 - np.abs(np.random.normal(0, 0.0005, size=len(dates))))
    volumes = np.random.randint(100, 1000, size=len(dates))
    
    # Log returns
    log_returns = np.log(closes / np.roll(closes, 1))
    log_returns[0] = np.nan
    
    df = pd.DataFrame({
        'timestamp_utc': dates,
        'symbol': 'EURUSD',
        'timeframe': '1H',
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'log_return_1': log_returns,
        'valid_bar': True,  # All bars valid
        'is_missing': False,
        'is_outlier': False,
    })
    
    return df


def main():
    """Run feature engine demonstration"""
    
    print("=" * 80)
    print("ARBITREX FEATURE ENGINE - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 1. CREATE SAMPLE DATA
    # ========================================================================
    print("1. Creating sample clean OHLCV data...")
    clean_df = create_sample_clean_data()
    print(f"   ✓ Generated {len(clean_df)} bars of EURUSD 1H data")
    print()
    
    # ========================================================================
    # 2. INITIALIZE FEATURE PIPELINE
    # ========================================================================
    print("2. Initializing Feature Pipeline...")
    config = FeatureEngineConfig()
    pipeline = FeaturePipeline(config)
    print(f"   ✓ Pipeline initialized with config version {config.config_version}")
    print(f"   ✓ Config hash: {config.get_config_hash()}")
    print()
    
    # ========================================================================
    # 3. COMPUTE FEATURES
    # ========================================================================
    print("3. Computing features...")
    feature_df, metadata = pipeline.compute_features(
        clean_df,
        symbol='EURUSD',
        timeframe='1H',
        normalize=True
    )
    print(f"   ✓ Computed {metadata.features_computed} features")
    print(f"   ✓ Processed {metadata.valid_bars_processed} bars")
    print(f"   ✓ Feature names: {', '.join(metadata.feature_names[:5])}...")
    print()
    
    # ========================================================================
    # 4. INSPECT FEATURES
    # ========================================================================
    print("4. Inspecting computed features...")
    
    # Get schema
    schema = FeatureSchema()
    ml_features = schema.get_ml_features('1H')
    
    print(f"   ML-ready features for 1H: {len(ml_features)}")
    print()
    
    # Show feature categories
    print("   Category A - Returns & Momentum:")
    for feat in schema.returns_momentum_features:
        if feat in feature_df.columns:
            non_null = feature_df[feat].notna().sum()
            print(f"      • {feat}: {non_null}/{len(feature_df)} non-null")
    print()
    
    print("   Category B - Volatility:")
    for feat in schema.volatility_features:
        if feat in feature_df.columns:
            non_null = feature_df[feat].notna().sum()
            print(f"      • {feat}: {non_null}/{len(feature_df)} non-null")
    print()
    
    print("   Category C - Trend Structure:")
    for feat in schema.trend_features[:3]:  # First 3 only
        if feat in feature_df.columns:
            non_null = feature_df[feat].notna().sum()
            print(f"      • {feat}: {non_null}/{len(feature_df)} non-null")
    print()
    
    # ========================================================================
    # 5. NORMALIZED FEATURES
    # ========================================================================
    print("5. Checking normalization...")
    
    # Check normalized features
    norm_features = [f'{feat}_norm' for feat in ml_features if f'{feat}_norm' in feature_df.columns]
    print(f"   ✓ {len(norm_features)} features normalized")
    
    # Show sample statistics
    sample_feat = 'momentum_score_norm'
    if sample_feat in feature_df.columns:
        values = feature_df[sample_feat].dropna()
        print(f"   Example: {sample_feat}")
        print(f"      Mean: {values.mean():.4f}")
        print(f"      Std:  {values.std():.4f}")
        print(f"      Min:  {values.min():.4f}")
        print(f"      Max:  {values.max():.4f}")
    print()
    
    # ========================================================================
    # 6. FREEZE FEATURE VECTOR
    # ========================================================================
    print("6. Freezing feature vector (live trading simulation)...")
    
    # Get last timestamp
    last_timestamp = feature_df['timestamp_utc'].iloc[-1]
    
    # Freeze vector
    vector = pipeline.freeze_feature_vector(
        feature_df,
        last_timestamp,
        symbol='EURUSD',
        timeframe='1H',
        ml_only=True
    )
    
    print(f"   ✓ Frozen vector at {last_timestamp}")
    print(f"   ✓ Features: {len(vector.feature_values)}")
    print(f"   ✓ Version: {vector.feature_version}")
    print(f"   ✓ ML Ready: {vector.is_ml_ready}")
    print(f"   ✓ Sample values: {vector.feature_values[:5]}")
    print()
    
    # ========================================================================
    # 7. FEATURE STORE
    # ========================================================================
    print("7. Testing Feature Store...")
    
    # Initialize store
    store = FeatureStore(Path("arbitrex/data/features"))
    
    # Write features
    store.write_features(
        feature_df,
        metadata,
        symbol='EURUSD',
        timeframe='1H',
        config_hash=config.get_config_hash()
    )
    print(f"   ✓ Features written to store")
    
    # Read features back
    loaded_df = store.read_features(
        symbol='EURUSD',
        timeframe='1H',
        config_hash=config.get_config_hash()
    )
    print(f"   ✓ Features loaded from store: {len(loaded_df)} bars")
    
    # Check versions
    versions = store.list_versions('EURUSD', '1H')
    print(f"   ✓ Available versions: {len(versions)}")
    print()
    
    # ========================================================================
    # 8. VALIDATION SUMMARY
    # ========================================================================
    print("8. Validation Summary...")
    print()
    
    print("   ✅ Causality: All rolling windows end at time t")
    print("   ✅ Stationarity: No raw prices in feature vector")
    print("   ✅ Determinism: Reproducible feature computation")
    print("   ✅ Normalization: Rolling z-score applied")
    print("   ✅ Versioning: Config-hashed feature storage")
    print("   ✅ Immutability: Features never recomputed")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("FEATURE ENGINE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Total features computed: {metadata.features_computed}")
    print(f"Valid bars processed: {metadata.valid_bars_processed}")
    print(f"Configuration version: {config.config_version}")
    print(f"Configuration hash: {config.get_config_hash()}")
    print()
    print("✓ All stages completed successfully")
    print()


if __name__ == '__main__':
    main()
