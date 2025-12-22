# Arbitrex Clean Data Layer

**Version:** 1.0.0  
**Purpose:** Transform raw OHLCV bars into analysis-safe bars without introducing lookahead bias, smoothing, or hidden assumptions.

---

## Philosophy

> **"The Clean Data Layer is not here to make data usable — it is here to make data trustworthy or rejected."**

### Core Principles

1. **Deterministic:** Same input → Same output, always
2. **Auditable:** Every decision logged and versioned
3. **Failure-Intolerant:** Prefer no data over bad data
4. **Flag, Don't Fix:** Detection only, never silent correction
5. **Explicit Everything:** No hidden assumptions or magic numbers

---

## What This Layer Does

✅ **Consumes** raw OHLCV bars from the Raw Data Layer  
✅ **Aligns** timestamps to canonical UTC grids  
✅ **Detects** missing bars and gaps (never forward-fills)  
✅ **Flags** statistical outliers (never removes)  
✅ **Computes** log returns (only between valid bars)  
✅ **Validates** each bar with strict rules  
✅ **Produces** `fx_ohlcv_clean` for downstream systems

---

## What This Layer Does NOT Do

❌ Generate features  
❌ Train models  
❌ Execute trades  
❌ Infer signals  
❌ Modify raw OHLC values  
❌ Forward-fill missing data  
❌ Smooth or interpolate  
❌ Make trading assumptions

---

## Module Structure

```
clean_data/
│
├── __init__.py                  # Package exports
├── config.py                    # Thresholds, schedules, rules
├── schemas.py                   # fx_ohlcv_clean schema + metadata
├── time_alignment.py            # UTC normalization & grid alignment
├── missing_bar_detection.py     # Gap detection & flagging
├── outlier_detection.py         # Statistical + OHLC sanity checks
├── return_calculation.py        # Log return computation
├── spread_estimation.py         # Optional bid-ask spread proxy
├── validator.py                 # valid_bar logic (mandatory gate)
├── pipeline.py                  # Orchestration & execution
└── README.md                    # This file
```

### Single Responsibility Guarantee

Each module has **exactly one responsibility**:
- `time_alignment.py`: Time only
- `missing_bar_detection.py`: Gaps only
- `outlier_detection.py`: Anomalies only
- `return_calculation.py`: Returns only
- `spread_estimation.py`: Spreads only
- `validator.py`: Validation only
- `pipeline.py`: Orchestration only

---

## Input Contract (FROM Raw Layer)

### Expected Input

Raw OHLCV bars with:
```python
{
    "timestamp": int,              # Unix timestamp (broker or UTC)
    "symbol": str,                 # e.g., "EURUSD"
    "timeframe": str,              # "1H" / "4H" / "1D"
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "broker_utc_offset_hours": int,  # Optional: for conversion
    # ... other raw metadata
}
```

### Input Guarantees

- Raw data is **immutable** (never modified by Clean Layer)
- Raw data is **untrusted** (may contain errors, gaps, outliers)
- Raw data is **never overwritten**

---

## Output Contract (fx_ohlcv_clean)

### Schema Definition

```python
@dataclass
class CleanOHLCVSchema:
    # Primary key
    timestamp_utc: datetime     # Canonical bar close time (UTC)
    symbol: str                 # FX pair
    timeframe: str              # 1H / 4H / 1D / 1M
    
    # Raw OHLCV (NEVER MODIFIED)
    open: Optional[float]       # NULL if missing
    high: Optional[float]       # NULL if missing
    low: Optional[float]        # NULL if missing
    close: Optional[float]      # NULL if missing
    volume: Optional[float]     # NULL if missing
    
    # Derived quantities (minimal)
    log_return_1: Optional[float]     # log(close_t / close_{t-1})
    spread_estimate: Optional[float]  # Optional spread proxy
    
    # Quality flags (EXPLICIT)
    is_missing: bool            # True if expected bar not found
    is_outlier: bool            # True if anomaly detected
    valid_bar: bool             # True ONLY if ALL checks pass
    
    # Auditability
    source_id: Optional[str]    # Raw ingestion reference
    schema_version: str         # "1.0.0"
```

### Output Guarantees

1. **Immutability:** Raw OHLC values never altered
2. **UTC Timestamps:** All timestamps in UTC, aligned to canonical schedule
3. **Explicit Flags:** Every bar has `is_missing`, `is_outlier`, `valid_bar` set
4. **NULL Safety:** Missing values are NULL, never forward-filled
5. **Schema Compliance:** Every output conforms to `CleanOHLCVSchema`
6. **Auditability:** Every output has versioned metadata

---

## Processing Pipeline

### Execution Flow

```
Raw Data
   ↓
1. Time Alignment          → UTC conversion, grid alignment
   ↓
2. Missing Detection       → Gap identification, flag setting
   ↓
3. Outlier Detection       → Anomaly flagging (never removal)
   ↓
4. Return Calculation      → Log returns (only between valid bars)
   ↓
5. Spread Estimation       → Optional spread proxy
   ↓
6. Validation              → valid_bar gate (strict rules)
   ↓
7. Schema Conformance      → Final format check
   ↓
fx_ohlcv_clean
```

### Fail-Fast Philosophy

If **any critical step fails**:
- ✋ **Abort processing**
- 📝 **Emit explicit error**
- 🚫 **Do not partially write data**
- 📊 **Log full context in metadata**

---

## Component Details

### 1. Time Alignment (`time_alignment.py`)

**Purpose:** Convert all timestamps to UTC and align to canonical schedules.

**Canonical Schedules:**
- **1H:** Every hour on the hour (00:00, 01:00, ..., 23:00)
- **4H:** Six times daily (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
- **1D:** Daily at 00:00 UTC

**Process:**
1. Convert broker timestamps to UTC (using offset if available)
2. Generate expected timestamp grid
3. Left join expected with actual bars
4. Flag missing bars where no match found

**Rules:**
- ❌ No forward-filling
- ❌ No interpolation
- ✅ Missing bars explicitly inserted with NULL OHLC

---

### 2. Missing Bar Detection (`missing_bar_detection.py`)

**Purpose:** Identify gaps and enforce quality gates.

**Detection:**
- Single missing bar → flagged
- Consecutive missing bars → counted
- High missing percentage → symbol may be excluded

**Thresholds (Configurable):**
```python
max_consecutive_missing: int = 3           # Consecutive gap limit
max_missing_percentage: float = 0.05       # 5% total missing limit
timestamp_tolerance_seconds: int = 60      # Matching tolerance
```

**Symbol Exclusion:**
Symbol excluded if:
- Missing percentage > 5%
- Consecutive missing > 3 bars
- First or last bar missing

---

### 3. Outlier Detection (`outlier_detection.py`)

**Purpose:** Flag statistical anomalies and OHLC inconsistencies.

**Detection Methods:**

**a) Price Jump Test:**
```python
abs(log_return) > k × rolling_volatility
```
Where `k = 5.0` (configurable)

**b) OHLC Consistency:**
- `high >= max(open, close)`
- `low <= min(open, close)`
- `high >= low`
- All prices > 0

**c) Extreme Returns:**
- `abs(log_return) > 0.15` (15% single-bar move)

**Philosophy:**
- 🔍 **Detection only**
- 🚫 **Never remove outliers**
- 🚫 **Never correct values**
- 🏷️ **Flag with `is_outlier = True`**

---

### 4. Return Calculation (`return_calculation.py`)

**Purpose:** Compute log returns with strict safety rules.

**Formula:**
```python
log_return_1 = log(close_t / close_{t-1})
```

**Computation Rules:**
- ✅ Only between **valid bars** (not missing, not outlier)
- ❌ Never across missing bars
- ❌ Never across outliers
- ❌ Never at first bar
- ✅ NULL return if any condition violated

**Example:**
```
Bar   Close   is_missing   is_outlier   log_return_1
0     1.2000  False        False        NULL (first bar)
1     1.2020  False        False        0.00166 (valid)
2     NULL    True         False        NULL (bar missing)
3     1.2030  False        False        NULL (prev bar missing)
4     1.2050  False        False        0.00166 (valid)
```

---

### 5. Spread Estimation (`spread_estimation.py`)

**Purpose:** Estimate bid-ask spread from OHLC (optional).

**Method:**
```python
spread_estimate = (high - low) / close
```

Optional exponential smoothing:
```python
smoothed_spread = EMA(spread_estimate, alpha=0.1)
```

**Status:** Disabled by default (`enabled: False` in config)

---

### 6. Validation (`validator.py`)

**Purpose:** Implement the `valid_bar` gate - the **mandatory filter** for downstream use.

**A bar is valid ONLY if ALL pass:**
1. ✅ Not missing
2. ✅ Not outlier
3. ✅ OHLC consistency valid
4. ✅ Timestamp aligned
5. ✅ Return computable (when required)

**Otherwise:**
```python
valid_bar = False
```

**No exceptions. No overrides.**

**Validation Report Example:**
```python
{
    "total_bars": 1000,
    "valid_bars": 945,
    "invalid_bars": 55,
    "validation_rate": 94.5,
    "failure_breakdown": {
        "missing": 30,
        "outlier": 15,
        "null_return": 10,
        "low_volume": 0
    }
}
```

---

## Usage

### Basic Usage

```python
from arbitrex.clean_data import CleanDataPipeline, CleanDataConfig
import pandas as pd

# Load raw data
raw_df = pd.read_csv("raw_ohlcv/EURUSD_1H.csv")

# Initialize pipeline
config = CleanDataConfig()  # Uses defaults
pipeline = CleanDataPipeline(config)

# Process single symbol
cleaned_df, metadata = pipeline.process_symbol(
    raw_df=raw_df,
    symbol="EURUSD",
    timeframe="1H",
    source_id="raw_layer_cycle_123"
)

# Check if accepted
if metadata.valid_bars / metadata.total_bars_processed >= 0.90:
    print(f"✓ Dataset accepted: {metadata.valid_bars} valid bars")
    cleaned_df.to_csv("clean_ohlcv/EURUSD_1H_clean.csv", index=False)
else:
    print(f"✗ Dataset rejected: {metadata.warnings}")
```

### Batch Processing

```python
# Process multiple symbols
raw_data = {
    "EURUSD": pd.read_csv("raw_ohlcv/EURUSD_1H.csv"),
    "GBPUSD": pd.read_csv("raw_ohlcv/GBPUSD_1H.csv"),
    "USDJPY": pd.read_csv("raw_ohlcv/USDJPY_1H.csv"),
}

results = pipeline.process_multiple_symbols(
    raw_data=raw_data,
    timeframe="1H",
    output_dir="clean_ohlcv/"
)

# Check results
for symbol, (df, meta) in results.items():
    if df is not None:
        print(f"✓ {symbol}: {meta.valid_bars}/{meta.total_bars_processed} valid")
    else:
        print(f"✗ {symbol}: Failed - {meta.errors}")
```

### Custom Configuration

```python
from arbitrex.clean_data.config import (
    CleanDataConfig,
    OutlierThresholds,
    MissingBarThresholds,
    ValidationRules
)

# Create custom config
config = CleanDataConfig(
    outlier_thresholds=OutlierThresholds(
        price_jump_std_multiplier=3.0,  # More sensitive
        max_abs_log_return=0.10,        # Tighter limit
    ),
    missing_bar_thresholds=MissingBarThresholds(
        max_consecutive_missing=5,       # Allow more gaps
        max_missing_percentage=0.10,     # 10% tolerance
    ),
    validation_rules=ValidationRules(
        require_valid_returns=True,
        require_non_missing=True,
        require_non_outlier=True,
        min_volume=100.0,                # Require minimum volume
    ),
    fail_on_critical_error=True          # Abort on errors
)

pipeline = CleanDataPipeline(config)
```

---

## Configuration Reference

### Default Thresholds

```python
# Outlier Detection
price_jump_std_multiplier: float = 5.0      # Price jump threshold
volatility_window: int = 20                 # Rolling window size
max_abs_log_return: float = 0.15            # 15% max single-bar move
zero_price_tolerance: float = 1e-10         # Near-zero detection

# Missing Bars
timestamp_tolerance_seconds: int = 60       # Matching tolerance
max_consecutive_missing: int = 3            # Consecutive gap limit
max_missing_percentage: float = 0.05        # 5% total missing limit

# Validation
enforce_ohlc_consistency: bool = True       # Check OHLC logic
require_valid_returns: bool = True          # Require computable returns
require_non_missing: bool = True            # Reject missing bars
require_non_outlier: bool = True            # Reject outliers
min_volume: float = 0.0                     # Minimum volume (0=disabled)

# Spread Estimation
enabled: bool = False                       # Disabled by default
use_hl_spread: bool = True                  # Use high-low method
smoothing_alpha: float = 0.1                # EMA smoothing factor
```

---

## Versioning & Auditability

### Metadata Tracking

Every cleaned dataset includes comprehensive metadata:

```python
{
    "processing_timestamp": "2025-12-22T10:30:15.123456Z",
    "config_version": "1.0.0",
    "schema_version": "1.0.0",
    "raw_source_path": "raw_layer_cycle_20251222_103000",
    "statistics": {
        "total_bars_processed": 1000,
        "valid_bars": 945,
        "missing_bars": 30,
        "outlier_bars": 15,
        "invalid_bars": 55,
        "valid_bar_percentage": 94.5
    },
    "scope": {
        "symbols_processed": ["EURUSD"],
        "timeframes_processed": ["1H"]
    },
    "config_snapshot": { /* full config dict */ },
    "issues": {
        "warnings": ["High missing percentage in 2025-12-20"],
        "errors": []
    }
}
```

### Reproducibility

To reproduce a cleaned dataset:
1. Use same raw data (referenced by `raw_source_path`)
2. Use same config (stored in `config_snapshot`)
3. Use same code version (tracked by `config_version`)

Result: **Bit-for-bit identical output**

---

## Error Handling

### Failure Modes

**1. Critical Errors (Abort):**
- Empty input dataframe
- Schema validation failure
- Missing required columns
- Corrupt data format

**2. Validation Warnings (Continue):**
- High missing percentage (but below threshold)
- Outliers detected
- Low validation rate (but above minimum)

**3. Symbol Exclusion:**
- Missing percentage > 5%
- Consecutive missing > 3 bars
- First or last bar invalid

### Fail-Fast Configuration

```python
config = CleanDataConfig(
    fail_on_critical_error=True  # Abort on any critical error
)
```

When `True`:
- Pipeline raises exception immediately
- No partial data written
- Full context logged

When `False`:
- Pipeline continues processing
- Returns `None` for failed symbols
- Errors recorded in metadata

---

## Best Practices

### For Quantitative Analysts

1. **Always check `valid_bar` flag** before using data
2. **Never use bars where `is_missing=True`**
3. **Investigate bars where `is_outlier=True`** (may be real events)
4. **Use `log_return_1` directly** (already computed safely)
5. **Verify config version** matches your expectations
6. **Review metadata warnings** before analysis

### For Production Systems

1. **Store metadata with cleaned data** (same filename + `_metadata.json`)
2. **Version control your config files**
3. **Log all pipeline executions** with timestamps
4. **Monitor validation rates** (alert if < 90%)
5. **Reject datasets below quality threshold**
6. **Audit outlier flags regularly**

### For Backtesting

1. **Use only bars where `valid_bar=True`**
2. **Never look ahead** (data already ensures this)
3. **Check for survivorship bias** (missing bars may indicate delisting)
4. **Validate timestamp alignment** (no timezone leaks)
5. **Document which config version used**

---

## Troubleshooting

### Issue: All bars marked `is_missing=True`

**Cause:** Time alignment failed, raw timestamps don't match expected grid

**Solution:**
1. Check `timestamp_utc` column exists in raw data
2. Verify broker offset conversion correct
3. Review canonical schedule for timeframe

---

### Issue: Too many outliers detected

**Cause:** Thresholds too strict, or genuine market volatility

**Solution:**
1. Review outlier timestamps (are they real events?)
2. Adjust `price_jump_std_multiplier` if needed
3. Check raw data quality
4. Consider market regime (crypto vs FX)

---

### Issue: Low validation rate (<90%)

**Cause:** Poor raw data quality, gaps, or strict thresholds

**Solution:**
1. Review metadata `failure_breakdown`
2. Check raw data source reliability
3. Relax thresholds if appropriate
4. Consider excluding problematic symbols

---

### Issue: Returns always NULL

**Cause:** All bars marked invalid due to missing/outlier flags

**Solution:**
1. Check `is_missing` and `is_outlier` counts
2. Verify OHLC consistency
3. Review outlier detection thresholds
4. Ensure at least 2 consecutive valid bars exist

---

## Testing

### Validation Tests

```python
# Test schema conformance
from arbitrex.clean_data.schemas import CleanOHLCVSchema

try:
    CleanOHLCVSchema.validate_dataframe(cleaned_df)
    print("✓ Schema valid")
except ValueError as e:
    print(f"✗ Schema invalid: {e}")

# Test validation rate
validation_rate = cleaned_df["valid_bar"].mean() * 100
assert validation_rate >= 90.0, f"Validation rate too low: {validation_rate}%"

# Test no missing OHLC on valid bars
valid_bars = cleaned_df[cleaned_df["valid_bar"]]
assert valid_bars["close"].notna().all(), "Valid bars have NULL close prices"

# Test return safety
returns_on_first_bar = cleaned_df.iloc[0]["log_return_1"]
assert pd.isna(returns_on_first_bar), "First bar has non-NULL return"
```

---

## Performance

### Typical Throughput

- **Single symbol (1000 bars):** 50-200ms
- **Batch processing (50 symbols):** 5-10 seconds
- **Large dataset (100K bars):** 1-3 seconds

### Optimization Tips

1. **Process in batches** (use `process_multiple_symbols`)
2. **Disable spread estimation** if not needed
3. **Parallelize** across symbols (pipeline is stateless)
4. **Use chunking** for very large datasets

---

## Contact & Support

**Maintainer:** Arbitrex Quantitative Research Team  
**Version:** 1.0.0  
**License:** Proprietary

---

## Appendix: Complete Example

```python
from arbitrex.clean_data import CleanDataPipeline, CleanDataConfig
from arbitrex.clean_data.schemas import CleanOHLCVSchema
import pandas as pd
from pathlib import Path

# 1. Setup
config = CleanDataConfig()
pipeline = CleanDataPipeline(config)
output_dir = Path("clean_data_output")

# 2. Load raw data
raw_df = pd.read_csv("raw_layer/EURUSD_1H.csv")
print(f"Loaded {len(raw_df)} raw bars")

# 3. Process
cleaned_df, metadata = pipeline.process_symbol(
    raw_df=raw_df,
    symbol="EURUSD",
    timeframe="1H",
    source_id="raw_cycle_20251222"
)

# 4. Validate
if cleaned_df is not None:
    CleanOHLCVSchema.validate_dataframe(cleaned_df)
    print(f"✓ Schema valid")
    
    validation_rate = (metadata.valid_bars / metadata.total_bars_processed) * 100
    print(f"✓ Validation rate: {validation_rate:.2f}%")
    
    # 5. Write
    if validation_rate >= 90.0:
        output_file = pipeline.write_clean_data(cleaned_df, metadata, output_dir)
        print(f"✓ Written to {output_file}")
    else:
        print(f"✗ Rejected: validation rate too low")
else:
    print(f"✗ Processing failed: {metadata.errors}")

# 6. Summary
print("\nPipeline Summary:")
print(f"  Total bars: {metadata.total_bars_processed}")
print(f"  Valid bars: {metadata.valid_bars}")
print(f"  Missing bars: {metadata.missing_bars}")
print(f"  Outlier bars: {metadata.outlier_bars}")
print(f"  Warnings: {len(metadata.warnings)}")
print(f"  Errors: {len(metadata.errors)}")
```

---

**End of Documentation**
