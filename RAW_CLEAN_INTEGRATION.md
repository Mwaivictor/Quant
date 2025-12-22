# Raw-Clean Layer Integration Guide

## Overview

The Arbitrex data pipeline consists of two complementary layers working in synergy:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW DATA LAYER                              │
│  • MT5 ingestion (OHLCV bars + ticks)                           │
│  • Immutable storage with dual timestamps                        │
│  • Real-time streaming + batch processing                        │
│  • Comprehensive metadata and health monitoring                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ CSV files (UTC + broker timestamps)
                         │ arbitrex/data/raw/ohlcv/fx/
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CLEAN DATA LAYER                             │
│  • UTC time alignment to canonical grids                         │
│  • Missing bar detection (never forward-fill)                    │
│  • Outlier flagging (never correction)                           │
│  • Safe return calculation                                       │
│  • Strict validation gate (valid_bar)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Validated CSV files (schema-enforced)
                         │ arbitrex/data/clean/ohlcv/fx/
                         │
                         ▼
                  FEATURE ENGINEERING
                  MODEL TRAINING
                  BACKTESTING
```

---

## Integration Points

### 1. **Data Flow**

**Raw Layer Output:**
```
arbitrex/data/raw/ohlcv/fx/
├── EURUSD/
│   ├── 1H/
│   │   ├── 2025-12-20.csv
│   │   ├── 2025-12-21.csv
│   │   └── 2025-12-22.csv
│   └── 4H/
└── GBPUSD/
    └── 1H/
```

**Clean Layer Output:**
```
arbitrex/data/clean/ohlcv/fx/
├── EURUSD/
│   ├── 1H/
│   │   ├── EURUSD_1H_20251222_120000_clean.csv
│   │   └── EURUSD_1H_20251222_120000_metadata.json
│   └── 4H/
└── GBPUSD/
    └── 1H/
```

### 2. **Schema Compatibility**

**Raw Layer CSV Format:**
```csv
timestamp_utc,timestamp_broker,open,high,low,close,volume
2025-12-22T00:00:00Z,1640127600,1.2000,1.2010,1.1990,1.2005,1500
```

**Clean Layer CSV Format:**
```csv
timestamp_utc,symbol,timeframe,open,high,low,close,volume,log_return_1,spread_estimate,is_missing,is_outlier,valid_bar,source_id,schema_version
2025-12-22T00:00:00Z,EURUSD,1H,1.2000,1.2010,1.1990,1.2005,1500,0.00166,0.0008,False,False,True,raw_cycle_123,1.0.0
```

### 3. **Metadata Linkage**

Clean layer metadata references raw layer source:

```json
{
  "raw_source_path": "raw_cycle_20251222_103000",
  "raw_source_timestamp": "2025-12-22T10:30:00Z",
  "source_id": "2025-12-22T075357Z_EURUSD_1H"
}
```

This enables full audit trail: Clean bar → Raw cycle → MT5 ingestion

---

## Usage Patterns

### Pattern 1: Direct Integration (Recommended)

Use the `RawToCleanBridge` for seamless processing:

```python
from arbitrex.clean_data import RawToCleanBridge

# Initialize bridge
bridge = RawToCleanBridge()

# Process single symbol
cleaned_df, metadata = bridge.process_symbol(
    symbol="EURUSD",
    timeframe="1H"
)

# Automatically reads from: arbitrex/data/raw/ohlcv/fx/EURUSD/1H/
# Automatically writes to:  arbitrex/data/clean/ohlcv/fx/EURUSD/1H/
```

### Pattern 2: Batch Processing

Process entire trading universe:

```python
from arbitrex.clean_data import RawToCleanBridge

bridge = RawToCleanBridge()

# Process all FX pairs
results = bridge.process_universe(
    timeframe="1H",
    asset_classes=["FX"]
)

# Generate report
report = bridge.get_processing_report(results)
print(f"Success rate: {report['summary']['success_rate']}%")
```

### Pattern 3: CLI Pipeline Runner

Command-line orchestration:

```bash
# Process specific symbols
python -m arbitrex.scripts.run_data_pipeline \
    --timeframe 1H \
    --symbols EURUSD GBPUSD USDJPY

# Process entire FX asset class
python -m arbitrex.scripts.run_data_pipeline \
    --timeframe 4H \
    --asset-class FX \
    --report report.json

# Process all available data
python -m arbitrex.scripts.run_data_pipeline \
    --timeframe 1H \
    --all
```

### Pattern 4: Scheduled Automation

Cron job for regular processing:

```bash
# Process every hour after raw ingestion
0 * * * * cd /path/to/arbitrex && python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --all --report /logs/clean_$(date +\%Y\%m\%d_\%H).json
```

---

## Complete Workflow Examples

### Example 1: End-to-End Daily Processing

```python
"""
Daily data pipeline: Ingest raw → Clean → Feature engineering
"""

from arbitrex.raw_layer.runner import ingest_historical_once
from arbitrex.clean_data import RawToCleanBridge
from arbitrex.raw_layer.config import TRADING_UNIVERSE
import pandas as pd

# Step 1: Ingest fresh raw data (if needed)
print("Step 1: Ingesting raw data...")
# (Assuming raw layer runner has already run or runs separately)

# Step 2: Process through clean pipeline
print("Step 2: Cleaning data...")
bridge = RawToCleanBridge()

results = bridge.process_universe(
    timeframe="1H",
    asset_classes=["FX"]
)

# Step 3: Generate quality report
report = bridge.get_processing_report(results)

print(f"\nProcessing Summary:")
print(f"  Total symbols: {report['summary']['total_symbols']}")
print(f"  Success rate: {report['summary']['success_rate']}%")
print(f"  Valid bars: {report['aggregated_statistics']['total_valid_bars']:,}")
print(f"  Validation rate: {report['aggregated_statistics']['overall_validation_rate']:.2f}%")

# Step 4: Use cleaned data for analysis
for symbol, (df, meta) in results.items():
    if df is not None:
        # Filter to valid bars only
        valid_df = df[df["valid_bar"] == True].copy()
        
        print(f"\n{symbol}: {len(valid_df)} valid bars ready for analysis")
        
        # Now safe to compute features, train models, etc.
        # valid_df["sma_20"] = valid_df["close"].rolling(20).mean()
```

### Example 2: Real-Time Processing Pipeline

```python
"""
Real-time pipeline: Stream raw ticks → Aggregate bars → Clean → Analyze
"""

from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool
from arbitrex.clean_data import RawToCleanBridge
from datetime import datetime, timedelta
import time

# Initialize components
pool = MT5ConnectionPool()
bridge = RawToCleanBridge()

symbols = ["EURUSD", "GBPUSD"]

# Start tick streaming (runs in background)
pool.start_tick_collector(symbols=symbols)

# Periodic cleaning (every hour on the hour)
while True:
    now = datetime.utcnow()
    
    # Wait until top of hour
    if now.minute == 0:
        print(f"\n[{now}] Starting hourly clean processing...")
        
        # Process latest raw data through clean pipeline
        for symbol in symbols:
            try:
                # Read last 24 hours of raw data
                start_date = now - timedelta(hours=24)
                
                cleaned_df, metadata = bridge.process_symbol(
                    symbol=symbol,
                    timeframe="1H",
                    start_date=start_date,
                    end_date=now
                )
                
                if cleaned_df is not None:
                    # Get most recent valid bar
                    latest_valid = cleaned_df[
                        cleaned_df["valid_bar"] == True
                    ].iloc[-1]
                    
                    print(f"{symbol}: Latest valid close = {latest_valid['close']}")
                    
                    # Push to downstream systems
                    # publish_to_feature_engine(symbol, latest_valid)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        # Sleep until next hour
        time.sleep(3600)
    else:
        time.sleep(60)  # Check every minute
```

### Example 3: Backtest Data Preparation

```python
"""
Prepare clean historical data for backtesting
"""

from arbitrex.clean_data import RawToCleanBridge
from datetime import datetime
import pandas as pd

# Initialize bridge
bridge = RawToCleanBridge()

# Process historical data for backtesting period
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
timeframe = "1H"
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

backtest_data = {}

for symbol in symbols:
    print(f"Preparing {symbol} backtest data...")
    
    cleaned_df, metadata = bridge.process_symbol(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if cleaned_df is not None:
        # Filter to valid bars only
        valid_df = cleaned_df[cleaned_df["valid_bar"] == True].copy()
        
        # Validate no lookahead bias
        assert valid_df["timestamp_utc"].is_monotonic_increasing
        assert valid_df["log_return_1"].iloc[0] is pd.NA  # First bar has no return
        
        # Store for backtesting
        backtest_data[symbol] = valid_df
        
        print(f"  ✓ {symbol}: {len(valid_df)} valid bars")
        print(f"    Date range: {valid_df['timestamp_utc'].min()} → {valid_df['timestamp_utc'].max()}")
        print(f"    Missing bars: {metadata.missing_bars}")
        print(f"    Outliers: {metadata.outlier_bars}")
    else:
        print(f"  ✗ {symbol}: Failed to process")

# Now ready for backtesting with clean, validated data
print(f"\nBacktest data prepared for {len(backtest_data)} symbols")
```

---

## Configuration Coordination

### Shared Configuration

Both layers can reference the same trading universe:

```python
# arbitrex/raw_layer/config.py
TRADING_UNIVERSE = {
    "FX": ["EURUSD", "GBPUSD", "USDJPY", ...],
    "Metals": ["XAUUSD", "XAGUSD", ...],
}

# Used by both:
from arbitrex.raw_layer.config import TRADING_UNIVERSE

# Raw layer: Determines which symbols to ingest
# Clean layer: Determines which symbols to process
```

### Timeframe Alignment

Both layers use same timeframe identifiers:

```python
DEFAULT_TIMEFRAMES = ["1H", "4H", "1D", "1M"]
```

Ensures raw layer outputs match clean layer inputs.

---

## Quality Control Integration

### Validation Pipeline

```python
from arbitrex.clean_data import RawToCleanBridge
from arbitrex.clean_data.schemas import CleanOHLCVSchema

bridge = RawToCleanBridge()

# Process
cleaned_df, metadata = bridge.process_symbol("EURUSD", "1H")

# Multi-level validation
if cleaned_df is not None:
    # 1. Schema validation
    CleanOHLCVSchema.validate_dataframe(cleaned_df)
    
    # 2. Quality thresholds
    validation_rate = metadata.valid_bars / metadata.total_bars_processed
    assert validation_rate >= 0.90, "Validation rate too low"
    
    # 3. Consistency checks
    assert not cleaned_df["timestamp_utc"].duplicated().any()
    assert cleaned_df["timestamp_utc"].is_monotonic_increasing
    
    # 4. No lookahead bias
    assert cleaned_df.iloc[0]["log_return_1"] is pd.NA
    
    print("✓ All validation checks passed")
```

### Monitoring Integration

Both layers can share health monitoring:

```python
from arbitrex.raw_layer.health import init_health_monitor
from arbitrex.clean_data import RawToCleanBridge

# Raw layer health monitor
raw_health = init_health_monitor()

# Clean layer processing
bridge = RawToCleanBridge()
results = bridge.process_universe(timeframe="1H", asset_classes=["FX"])

# Combined health report
report = bridge.get_processing_report(results)

# Alert if quality degrades
if report['aggregated_statistics']['overall_validation_rate'] < 90.0:
    print("⚠ WARNING: Clean data validation rate below threshold")
    # Send alert to monitoring system
```

---

## Troubleshooting Integration Issues

### Issue: "No raw data found"

**Cause:** Raw layer hasn't ingested data yet or paths misconfigured

**Solution:**
```python
# Check raw data availability
from pathlib import Path
raw_dir = Path("arbitrex/data/raw/ohlcv/fx")
print(f"Raw data exists: {raw_dir.exists()}")
print(f"Symbols available: {list(raw_dir.iterdir())}")

# Manually specify paths
bridge = RawToCleanBridge(
    raw_base_dir="path/to/raw/data",
    clean_base_dir="path/to/clean/output"
)
```

### Issue: "Timestamp column not found"

**Cause:** Raw layer output format changed or missing timestamp_utc column

**Solution:**
Ensure raw layer writes both `timestamp_utc` and `timestamp_broker` columns (should be default after TIME_NORMALIZATION implementation).

### Issue: "All bars marked invalid"

**Cause:** Clean layer thresholds too strict for available data quality

**Solution:**
```python
from arbitrex.clean_data.config import CleanDataConfig, OutlierThresholds

# Relax thresholds
config = CleanDataConfig(
    outlier_thresholds=OutlierThresholds(
        price_jump_std_multiplier=10.0,  # More tolerant
        max_abs_log_return=0.20,         # Allow larger moves
    )
)

bridge = RawToCleanBridge(config=config)
```

---

## Performance Considerations

### Batch vs. Real-Time

**Batch Processing (Recommended for Historical):**
- Process large date ranges efficiently
- Parallelizable across symbols
- Suitable for backtesting data preparation

**Real-Time Processing:**
- Process incremental updates only
- Lower latency for live trading
- Smaller memory footprint

### Optimization Tips

1. **Process in parallel** (across symbols, not implemented yet):
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(bridge.process_symbol, symbol, "1H")
        for symbol in symbols
    ]
    results = [f.result() for f in futures]
```

2. **Use date ranges** to limit data volume:
```python
from datetime import datetime, timedelta

# Only process last 7 days
start_date = datetime.utcnow() - timedelta(days=7)
cleaned_df, meta = bridge.process_symbol(
    symbol="EURUSD",
    timeframe="1H",
    start_date=start_date
)
```

3. **Cache processed data** to avoid re-cleaning:
Check if clean data already exists before processing.

---

## Summary

The Raw-Clean integration provides:

✅ **Seamless data flow** from ingestion to validation  
✅ **Shared configuration** (symbols, timeframes)  
✅ **Complete audit trail** (metadata linkage)  
✅ **Flexible processing** (batch, real-time, scheduled)  
✅ **Quality assurance** (validation at every step)  
✅ **Production-ready** (CLI tools, logging, monitoring)

### Quick Start

```bash
# 1. Ingest raw data (if not already running)
python -m arbitrex.raw_layer.runner --symbols EURUSD GBPUSD --timeframes 1H

# 2. Process through clean pipeline
python -m arbitrex.scripts.run_data_pipeline --timeframe 1H --symbols EURUSD GBPUSD

# 3. Verify output
ls arbitrex/data/clean/ohlcv/fx/EURUSD/1H/
```

**Your data is now ready for feature engineering, model training, and backtesting! 🚀**
