"""
Integration Test: Raw Layer → Clean Layer

Demonstrates the complete data pipeline working in synergy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arbitrex.clean_data.integration import RawToCleanBridge
from arbitrex.clean_data.schemas import CleanOHLCVSchema


def create_synthetic_raw_data(symbol: str, timeframe: str, num_bars: int = 100) -> pd.DataFrame:
    """
    Create synthetic raw data mimicking raw layer output.
    
    This simulates what the raw layer would produce with:
    - timestamp_utc column
    - OHLCV data
    - Some missing bars (gaps)
    - Some outliers
    """
    print(f"Creating synthetic raw data: {symbol} {timeframe} ({num_bars} bars)")
    
    # Generate timestamps (1 hour apart)
    start_time = datetime(2025, 12, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_bars)]
    
    # Generate price data (random walk)
    np.random.seed(42)
    base_price = 1.2000
    returns = np.random.normal(0, 0.0005, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV bars
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Generate OHLC around close
        spread = close * 0.0002  # 2 bps spread
        open_price = close + np.random.normal(0, spread)
        high_price = max(open_price, close) + abs(np.random.normal(0, spread))
        low_price = min(open_price, close) - abs(np.random.normal(0, spread))
        volume = np.random.randint(100, 2000)
        
        # Add some intentional issues for clean layer to detect
        
        # Issue 1: Missing bar (gap)
        if i in [20, 45, 70]:
            continue  # Skip this bar (creates gap)
        
        # Issue 2: Outlier (extreme price jump)
        if i == 50:
            close = close * 1.15  # 15% jump (should be flagged)
        
        # Issue 3: OHLC inconsistency
        if i == 80:
            low_price = high_price + 0.001  # Low > High (invalid)
        
        data.append({
            "timestamp_utc": ts,
            "timestamp_broker": int(ts.timestamp()),
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close,
            "volume": volume,
        })
    
    df = pd.DataFrame(data)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    
    print(f"  Generated {len(df)} bars (with {num_bars - len(df)} intentional gaps)")
    
    return df


def test_integration():
    """
    Test complete raw → clean pipeline integration.
    """
    print("="*80)
    print("RAW-CLEAN LAYER INTEGRATION TEST")
    print("="*80)
    
    # Create synthetic raw data
    symbol = "EURUSD"
    timeframe = "1H"
    
    raw_df = create_synthetic_raw_data(symbol, timeframe, num_bars=100)
    
    print(f"\nRaw data summary:")
    print(f"  Bars: {len(raw_df)}")
    print(f"  Date range: {raw_df['timestamp_utc'].min()} → {raw_df['timestamp_utc'].max()}")
    print(f"  Price range: {raw_df['close'].min():.4f} → {raw_df['close'].max():.4f}")
    
    # Initialize clean pipeline (via integration bridge)
    print("\n" + "-"*80)
    print("PROCESSING THROUGH CLEAN PIPELINE")
    print("-"*80)
    
    # Note: We'll process directly through pipeline since we have in-memory data
    from arbitrex.clean_data import CleanDataPipeline
    
    pipeline = CleanDataPipeline()
    
    cleaned_df, metadata = pipeline.process_symbol(
        raw_df=raw_df,
        symbol=symbol,
        timeframe=timeframe,
        source_id="synthetic_test_data"
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("CLEAN DATA RESULTS")
    print("="*80)
    
    if cleaned_df is None:
        print("✗ Processing failed!")
        print(f"Errors: {metadata.errors}")
        return False
    
    # Schema validation
    try:
        CleanOHLCVSchema.validate_dataframe(cleaned_df)
        print("✓ Schema validation passed")
    except ValueError as e:
        print(f"✗ Schema validation failed: {e}")
        return False
    
    # Statistics
    print(f"\nProcessing Statistics:")
    print(f"  Total bars processed:  {metadata.total_bars_processed}")
    print(f"  Valid bars:            {metadata.valid_bars} ({metadata.valid_bars/metadata.total_bars_processed*100:.1f}%)")
    print(f"  Missing bars:          {metadata.missing_bars}")
    print(f"  Outlier bars:          {metadata.outlier_bars}")
    print(f"  Invalid bars:          {metadata.invalid_bars}")
    
    # Detailed breakdown
    print(f"\nDetailed Breakdown:")
    print(f"  is_missing=True:       {cleaned_df['is_missing'].sum()}")
    print(f"  is_outlier=True:       {cleaned_df['is_outlier'].sum()}")
    print(f"  valid_bar=True:        {cleaned_df['valid_bar'].sum()}")
    
    # Verify expected issues were caught
    print(f"\n" + "-"*80)
    print("VALIDATION CHECKS")
    print("-"*80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Missing bars detected
    checks_total += 1
    if cleaned_df['is_missing'].sum() >= 3:  # We created 3 gaps
        print("✓ Check 1: Missing bars detected (3 gaps)")
        checks_passed += 1
    else:
        print(f"✗ Check 1: Expected 3 missing bars, found {cleaned_df['is_missing'].sum()}")
    
    # Check 2: Outliers detected
    checks_total += 1
    if cleaned_df['is_outlier'].sum() >= 1:  # We created 1 outlier + 1 OHLC issue
        print(f"✓ Check 2: Outliers detected ({cleaned_df['is_outlier'].sum()} bars)")
        checks_passed += 1
    else:
        print("✗ Check 2: No outliers detected (expected at least 1)")
    
    # Check 3: Returns computed safely
    checks_total += 1
    first_return = cleaned_df.iloc[0]['log_return_1']
    if pd.isna(first_return):
        print("✓ Check 3: First bar has NULL return (correct)")
        checks_passed += 1
    else:
        print("✗ Check 3: First bar has non-NULL return (incorrect)")
    
    # Check 4: No returns across missing bars
    checks_total += 1
    missing_indices = cleaned_df[cleaned_df['is_missing']].index
    if len(missing_indices) > 0:
        # Check bars after missing bars
        after_missing = [i+1 for i in missing_indices if i+1 < len(cleaned_df)]
        returns_after_missing = cleaned_df.loc[after_missing, 'log_return_1'].notna()
        
        if not returns_after_missing.any():
            print("✓ Check 4: No returns computed after missing bars (correct)")
            checks_passed += 1
        else:
            print("✗ Check 4: Returns found after missing bars (should be NULL)")
    else:
        print("⚠ Check 4: No missing bars to check")
    
    # Check 5: Valid bars have all data
    checks_total += 1
    valid_bars = cleaned_df[cleaned_df['valid_bar']]
    if len(valid_bars) > 0:
        all_have_prices = valid_bars['close'].notna().all()
        if all_have_prices:
            print(f"✓ Check 5: All valid bars have prices ({len(valid_bars)} bars)")
            checks_passed += 1
        else:
            print("✗ Check 5: Some valid bars have NULL prices")
    else:
        print("⚠ Check 5: No valid bars")
    
    # Check 6: Timestamps monotonic increasing
    checks_total += 1
    if cleaned_df['timestamp_utc'].is_monotonic_increasing:
        print("✓ Check 6: Timestamps monotonic increasing")
        checks_passed += 1
    else:
        print("✗ Check 6: Timestamps not monotonic")
    
    # Summary
    print(f"\n" + "="*80)
    print(f"INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Checks passed: {checks_passed}/{checks_total}")
    
    if checks_passed == checks_total:
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nRaw-Clean integration working correctly!")
        return True
    else:
        print(f"⚠ {checks_total - checks_passed} checks failed")
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
