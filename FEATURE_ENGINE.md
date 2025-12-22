# ARBITREX Feature Engine

## ✅ Production-Grade Feature Computation System

Transform clean OHLCV bars into stationary, normalized feature vectors for ML models and signal generation.

---

## 🎯 Design Philosophy

### **Core Principles**

1. **Causality**: No lookahead - all windows end at time t
2. **Stationarity**: No raw prices - only returns, ratios, normalized distances
3. **Determinism**: Same input → same output, fully reproducible
4. **Timeframe Isolation**: No mixing timeframes at computation time
5. **Data Trust**: Only consume `valid_bar == True` from Clean Data Layer
6. **No Retail**: No RSI, MACD, Stochastic, CCI

**Features describe market condition, NOT next price move.**

---

## 📦 Architecture

```
Clean OHLC Bars (valid_bar == True)
   ↓
Input Validation (FeatureInputValidator)
   ↓
Primitive Transforms (PrimitiveTransforms)
   ↓
Feature Categories:
   ├─ A: Returns & Momentum
   ├─ B: Volatility Structure
   ├─ C: Trend Structure (descriptive)
   ├─ D: Range & Market Efficiency
   ├─ E: Regime Features (daily only)
   └─ F: Execution/Cost Filters (ML excluded)
   ↓
Normalization (Rolling Z-Score)
   ↓
Feature Vector Freeze
   ↓
Feature Store (Immutable, Versioned)
```

---

## 📊 Feature Categories

### **Category A: Returns & Momentum**
**Purpose**: Directional persistence

| Feature | Formula | Description |
|---------|---------|-------------|
| `rolling_return_3` | ∑(log_return, 3 bars) | 3-bar cumulative return |
| `rolling_return_6` | ∑(log_return, 6 bars) | 6-bar cumulative return |
| `rolling_return_12` | ∑(log_return, 12 bars) | 12-bar cumulative return |
| `momentum_score` | R_12 / σ_12 | Risk-adjusted momentum |

### **Category B: Volatility Structure**
**Purpose**: Risk context & regime awareness

| Feature | Formula | Description |
|---------|---------|-------------|
| `vol_6` | σ(log_return, 6 bars) | 6-bar rolling volatility |
| `vol_12` | σ(log_return, 12 bars) | 12-bar rolling volatility |
| `vol_24` | σ(log_return, 24 bars) | 24-bar rolling volatility |
| `atr_normalized` | ATR_14 / close_t | Normalized Average True Range |

### **Category C: Trend Structure (Descriptive)**
**Purpose**: Trend geometry, NOT prediction

| Feature | Formula | Description |
|---------|---------|-------------|
| `ma_12_slope` | (MA_t - MA_{t-3}) / ATR | 12-bar MA slope (normalized) |
| `ma_24_slope` | (MA_t - MA_{t-3}) / ATR | 24-bar MA slope (normalized) |
| `ma_50_slope` | (MA_t - MA_{t-3}) / ATR | 50-bar MA slope (normalized) |
| `distance_to_ma_12` | (close - MA_12) / ATR | Price distance from 12-bar MA |
| `distance_to_ma_24` | (close - MA_24) / ATR | Price distance from 24-bar MA |
| `distance_to_ma_50` | (close - MA_50) / ATR | Price distance from 50-bar MA |

### **Category D: Range & Market Efficiency**
**Purpose**: Detect chop vs flow

| Feature | Formula | Description |
|---------|---------|-------------|
| `efficiency_ratio` | \|Δprice\| / ∑\|Δprices\| | Kaufman ER (0=chop, 1=trend) |
| `range_compression` | (high_w - low_w) / ATR | Range compression ratio |

### **Category E: Regime Features (Daily Only)**
**Purpose**: Trade permission, NOT direction

| Feature | Formula | Description |
|---------|---------|-------------|
| `trend_regime` | MA_fast vs MA_slow | Binary trend flag (+1/0/-1) |
| `stress_indicator` | σ_short / σ_long | Volatility stress ratio |

⚠️ **DAILY TIMEFRAME ONLY**

### **Category F: Execution/Cost Filters (Optional)**
**Purpose**: Prevent untradable signals

| Feature | Formula | Description |
|---------|---------|-------------|
| `spread_ratio` | avg_spread / ATR | Cost-to-volatility ratio |

⚠️ **NEVER PASSED TO ML MODELS** (`ml_excluded = True`)

---

## 🔧 Configuration

```python
from arbitrex.feature_engine.config import FeatureEngineConfig

# Default configuration
config = FeatureEngineConfig()

# Custom configuration
custom_config = FeatureEngineConfig(
    returns_momentum=ReturnsMomentumConfig(
        return_windows=[3, 6, 12, 24],
        momentum_window=12
    ),
    normalization=NormalizationConfig(
        norm_window=100,
        z_score_clip=3.0
    )
)

# Configuration is versioned
config_hash = config.get_config_hash()
```

---

## 🚀 Usage

### **Basic Usage**

```python
from arbitrex.feature_engine import FeaturePipeline
from arbitrex.clean_data.pipeline import CleanDataPipeline

# 1. Get clean data
clean_pipeline = CleanDataPipeline()
clean_df, _ = clean_pipeline.process_symbol(
    raw_df, 
    symbol='EURUSD', 
    timeframe='1H', 
    source_id='mt5'
)

# 2. Compute features
feature_pipeline = FeaturePipeline()
feature_df, metadata = feature_pipeline.compute_features(
    clean_df,
    symbol='EURUSD',
    timeframe='1H',
    normalize=True
)

# 3. Freeze feature vector (for live trading)
timestamp = feature_df['timestamp_utc'].iloc[-1]
vector = feature_pipeline.freeze_feature_vector(
    feature_df,
    timestamp,
    symbol='EURUSD',
    timeframe='1H',
    ml_only=True
)

print(f"Feature vector: {len(vector.feature_values)} features")
print(f"Version: {vector.feature_version}")
```

### **Feature Store Usage**

```python
from arbitrex.feature_engine.feature_store import FeatureStore
from pathlib import Path

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

# Read features (backtest/live parity)
features = store.read_features(
    symbol='EURUSD',
    timeframe='1H',
    config_hash=config.get_config_hash()
)

# Check existence
exists = store.exists('EURUSD', '1H', config_hash)

# List versions
versions = store.list_versions('EURUSD', '1H')
```

---

## 🎛️ Normalization

All features are normalized using **rolling z-score**:

```
x_norm(t) = (x_t - μ_{t-W}) / σ_{t-W}
```

**Properties**:
- Rolling window only (no global statistics)
- No future information
- Normalization parameters stored with features
- Optional robust statistics (median/MAD)

```python
from arbitrex.feature_engine.normalization import FeatureNormalizer

normalizer = FeatureNormalizer(config.normalization)

df_norm, norm_metadata = normalizer.normalize(
    df,
    feature_columns=['rolling_return_12', 'vol_12', 'momentum_score']
)

# Normalized features have '_norm' suffix
# df_norm['rolling_return_12_norm']
# df_norm['vol_12_norm']
```

---

## 📋 Feature Vector Schema

```python
from arbitrex.feature_engine.schemas import FeatureVector, FeatureSchema

# Get ML-ready features
schema = FeatureSchema()
ml_features = schema.get_ml_features(timeframe='1H')

# Daily timeframe includes regime features
ml_features_daily = schema.get_ml_features(timeframe='1D')

# All features (including execution filters)
all_features = schema.get_all_features(timeframe='1H')
```

**Feature Counts by Timeframe**:

| Timeframe | ML Features | All Features |
|-----------|-------------|--------------|
| 1H / 4H | 16 | 17 |
| 1D | 18 | 19 |

---

## ⚡ Live Execution Timing

**At bar close only**:

```python
# Bar close event occurs
timestamp_utc = get_current_bar_close()

# 1. Clean data validated (already done)
# 2. Feature computation triggered
feature_df, metadata = feature_pipeline.compute_features(
    clean_df,
    symbol='EURUSD',
    timeframe='1H',
    normalize=True
)

# 3. Feature vector frozen
vector = feature_pipeline.freeze_feature_vector(
    feature_df,
    timestamp_utc,
    symbol='EURUSD',
    timeframe='1H',
    ml_only=True
)

# 4. Pass downstream to:
#    - ML model
#    - Signal generator
#    - Risk manager

# 5. System sleeps until next bar close
```

❌ **No mid-bar updates**  
❌ **No recomputation**

---

## 🔒 Hard Constraints (Enforced)

### **1. Causality**
- All rolling windows end at time t
- No future information in computation
- Validated in `PrimitiveTransforms`

### **2. Stationarity**
- No raw prices to ML
- Only returns, ratios, normalized distances
- Enforced in feature computation

### **3. Determinism**
- Same input → same output
- No random seeds
- Fully reproducible

### **4. Timeframe Isolation**
- No mixing timeframes at computation time
- Daily ≠ 4H ≠ 1H
- Validated in input validator

### **5. Data Trust Boundary**
- Only consume `valid_bar == True`
- No internal cleaning/repair
- Validated in `FeatureInputValidator`

### **6. No Retail Indicators**
- RSI: ❌ Forbidden
- MACD: ❌ Forbidden
- Stochastic: ❌ Forbidden
- CCI: ❌ Forbidden

---

## 📈 Testing

```python
# Run feature engine tests
pytest tests/test_feature_engine.py -v

# Test individual components
pytest tests/test_feature_engine.py::TestReturnsMomentum -v
pytest tests/test_feature_engine.py::TestVolatility -v
pytest tests/test_feature_engine.py::TestNormalization -v
```

---

## 📂 Module Structure

```
arbitrex/feature_engine/
├── __init__.py                  # Package exports
├── config.py                    # Configuration (versioned)
├── validation.py                # Input validation
├── primitives.py                # Primitive transforms
├── returns_momentum.py          # Category A features
├── volatility.py                # Category B features
├── trend.py                     # Category C features
├── efficiency.py                # Category D features
├── regime.py                    # Category E features
├── execution.py                 # Category F features
├── normalization.py             # Rolling z-score
├── schemas.py                   # Feature vector schemas
├── pipeline.py                  # Orchestration
└── feature_store.py             # Immutable storage
```

---

## 🎯 Integration Points

### **Upstream: Clean Data Layer**
```python
# Feature Engine consumes clean data
from arbitrex.clean_data.pipeline import CleanDataPipeline

clean_pipeline = CleanDataPipeline()
clean_df, _ = clean_pipeline.process_symbol(raw_df, 'EURUSD', '1H', 'mt5')

# Only valid bars passed to Feature Engine
```

### **Downstream: ML Models**
```python
# Feature vectors feed ML models
ml_features = schema.get_ml_features(timeframe='1H')
X = feature_df[[f'{col}_norm' for col in ml_features]]
```

### **Downstream: Signal Generation**
```python
# Features used for signal logic
if vector.feature_values[momentum_idx] > 1.0 and \
   vector.feature_values[vol_idx] < vol_threshold:
    signal = generate_long_signal()
```

---

## 🏆 Production Readiness

✅ **Deterministic**: Same input → same output  
✅ **Causal**: No lookahead  
✅ **Stationary**: ML-safe features  
✅ **Versioned**: Config hashing  
✅ **Immutable**: Feature store never recomputes  
✅ **Tested**: Comprehensive test coverage  
✅ **Documented**: Full API documentation  
✅ **Auditable**: Complete metadata tracking  

---

## 📖 References

- **Configuration**: [config.py](config.py)
- **Pipeline**: [pipeline.py](pipeline.py)
- **Schemas**: [schemas.py](schemas.py)
- **Feature Store**: [feature_store.py](feature_store.py)

---

**Generated**: 2025-12-22  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
