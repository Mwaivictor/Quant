# Quantitative Statistics Engine (QSE)

## Overview

The **Quantitative Statistics Engine (QSE)** is a statistical validation layer that sits between the Feature Engine and ML layers in the Arbitrex system. It acts as a **gatekeeper**, ensuring that feature vectors meet rigorous statistical criteria before being forwarded to machine learning models.

### Purpose

- **Prevent noise-based learning**: Block signals when returns lack autocorrelation or exhibit excessive randomness
- **Ensure stationarity**: Validate that time series are stationary (ADF test) before ML modeling
- **Detect outliers**: Flag extreme observations that could distort ML predictions
- **Monitor correlation**: Suppress signals when cross-pair correlation is too high (risk concentration)
- **Filter by regime**: Block signals in extreme volatility regimes (too low or too high)

### Architecture

```
┌─────────────────┐
│ Feature Engine  │
│  (17 features)  │
└────────┬────────┘
         │
         │ X_t (feature vector)
         │
         ▼
┌─────────────────────────────────────────┐
│  Quantitative Statistics Engine (QSE)   │
│  ┌─────────────────────────────────┐   │
│  │ 1. Autocorrelation & Trend      │   │
│  │ 2. Stationarity (ADF)           │   │
│  │ 3. Distribution Stability       │   │
│  │ 4. Cross-Pair Correlation       │   │
│  │ 5. Volatility Regime            │   │
│  └─────────────────────────────────┘   │
└────────┬────────────────────────────────┘
         │
         │ signal_validity_flag + metrics
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   ML Layer      │─────▶│   Signal     │
│  (if valid)     │      │  Generator   │
└─────────────────┘      └──────────────┘
```

## Statistical Gates

### 1. Autocorrelation & Trend Persistence

**Formula:**
```
ρ_k = corr(r_t, r_{t-k})
trend_persistence_score = mean(|ρ_k| for ρ_k > threshold)
```

**Purpose:** Measure trend strength and persistence. High autocorrelation indicates exploitable patterns.

**Thresholds:**
- `lags = [1, 5, 10, 20]`
- `min_threshold = 0.15`
- `min_trend_consistency = 0.20`

**Pass Criteria:** `trend_persistence_score >= 0.20`

### 2. Stationarity (ADF Test)

**Formula:**
```
Augmented Dickey-Fuller test:
Δy_t = α + β*t + γ*y_{t-1} + Σ δ_i*Δy_{t-i} + ε_t

H0: γ = 0 (non-stationary)
H1: γ < 0 (stationary)
```

**Purpose:** Ensure time series is stationary (mean/variance stable over time).

**Thresholds:**
- `significance_level = 0.05`
- `max_lag = 10`

**Pass Criteria:** `p_value < 0.05` (reject null hypothesis)

### 3. Distribution Stability & Outlier Detection

**Formula:**
```
z_t = (x_t - μ_window) / σ_window
is_outlier = |z_t| > z_threshold
```

**Purpose:** Detect outliers and ensure distribution stability.

**Thresholds:**
- `rolling_window = 60`
- `z_score_threshold = 3.0`

**Pass Criteria:** `not is_outlier AND distribution_stable`

### 4. Cross-Pair Correlation

**Formula:**
```
ρ_ij = corr(r_i, r_j) for all pairs (i, j)
max_correlation = max(|ρ_ij|)
```

**Purpose:** Identify risk concentration from highly correlated pairs.

**Thresholds:**
- `max_correlation_threshold = 0.85`
- `min_pairs = 2`

**Pass Criteria:** `max_correlation < 0.85`

### 5. Volatility Regime Filtering

**Formula:**
```
volatility_t = rolling_std(returns, window)
percentile_t = percentile_rank(volatility_t, historical_window)

regime = LOW if percentile < 10%
       = NORMAL if 10% <= percentile <= 90%
       = HIGH if percentile > 90%
```

**Purpose:** Filter signals in extreme volatility regimes.

**Thresholds:**
- `min_percentile = 10.0`
- `max_percentile = 90.0`

**Pass Criteria:** `regime == NORMAL`

## Installation

The QSE is part of the Arbitrex package:

```bash
pip install statsmodels  # Required for ADF test
```

## Usage

### Basic Usage

```python
from arbitrex.quant_stats import QuantStatsConfig, QuantitativeStatisticsEngine
import pandas as pd

# Initialize QSE with default config
config = QuantStatsConfig()
qse = QuantitativeStatisticsEngine(config)

# Process a single bar
output = qse.process_bar(
    symbol='EURUSD',
    returns=returns_series,
    bar_index=100
)

# Check validity
if output.validation.signal_validity_flag:
    print("✓ Signal is valid - forward to ML")
else:
    print("✗ Signal is invalid - suppress")
    print(f"Reasons: {output.validation.failure_reasons}")
```

### Integration with Feature Engine

```python
from arbitrex.feature_engine import FeatureEngine
from arbitrex.quant_stats import QuantitativeStatisticsEngine

# Compute features
feature_engine = FeatureEngine(fe_config)
features = feature_engine.compute_features(symbol, ohlcv_df, bar_index)

# Validate with QSE
qse = QuantitativeStatisticsEngine(qse_config)
qse_output = qse.process_bar(
    symbol=symbol,
    returns=returns,
    bar_index=bar_index
)

# Decision logic
if qse_output.validation.signal_validity_flag:
    # Forward to ML layer
    ml_prediction = ml_model.predict(features)
    signal = signal_generator.generate(ml_prediction, features)
else:
    # Use fallback logic
    signal = signal_generator.generate_fallback(features)
```

### Multi-Symbol Correlation Analysis

```python
# Prepare return series for multiple symbols
returns_dict = {
    'EURUSD': eurusd_returns,
    'GBPUSD': gbpusd_returns,
    'XAUUSD': xauusd_returns
}

# Process with cross-pair correlation
output = qse.process_bar(
    symbol='EURUSD',
    returns=returns_dict['EURUSD'],
    bar_index=bar_index,
    returns_dict=returns_dict
)

# Check correlation
print(f"Max Correlation: {output.metrics.max_correlation:.4f}")
print(f"Correlation OK: {output.validation.correlation_check}")
```

### Processing Full Series

```python
# Process entire series
results = qse.process_series(
    symbol='EURUSD',
    returns=returns_series
)

# Get summary
summary = qse.get_validation_summary(results)

print(f"Validity Rate: {summary['validity_rate']*100:.2f}%")
print(f"Total Bars: {summary['total_bars']}")
print(f"Valid Signals: {summary['valid_signals']}")
```

## Configuration

### Custom Configuration

```python
from arbitrex.quant_stats import QuantStatsConfig

config = QuantStatsConfig()

# Autocorrelation
config.autocorr.lags = [1, 5, 10, 20]
config.autocorr.min_threshold = 0.15
config.autocorr.rolling_window = 60

# Stationarity
config.stationarity.significance_level = 0.05
config.stationarity.rolling_window = 60
config.stationarity.max_lag = 10

# Distribution
config.distribution.z_score_threshold = 3.0
config.distribution.rolling_window = 60
config.distribution.min_samples = 30

# Correlation
config.correlation.max_correlation_threshold = 0.85
config.correlation.rolling_window = 60
config.correlation.min_pairs = 2

# Volatility
config.volatility.rolling_window = 60
config.volatility.min_percentile = 10.0
config.volatility.max_percentile = 90.0

# Validation
config.validation.require_trend_persistence = True
config.validation.require_stationarity = True
config.validation.require_distribution_stability = True
config.validation.require_correlation_check = False
config.validation.require_volatility_filter = True
config.validation.min_trend_consistency = 0.20
```

### Configuration Versioning

```python
# Get deterministic config hash
config_hash = config.get_config_hash()
print(f"Config Version: {config_hash}")  # e.g., "a3f5b8c2d1e4f6a7"

# Serialize/deserialize
config_dict = config.to_dict()
config_restored = QuantStatsConfig.from_dict(config_dict)

# Hashes match
assert config.get_config_hash() == config_restored.get_config_hash()
```

## Output Schema

### QuantStatsOutput

```python
@dataclass
class QuantStatsOutput:
    timestamp: datetime           # Processing timestamp
    symbol: str                   # Symbol identifier
    bar_index: int               # Bar index
    metrics: StatisticalMetrics  # All statistical metrics
    validation: SignalValidation # Validation results
    regime_state: RegimeState    # Market regime state
    config_hash: str             # Config version
```

### StatisticalMetrics

```python
@dataclass
class StatisticalMetrics:
    # Autocorrelation
    autocorr_lag1: float
    autocorr_lag5: float
    autocorr_lag10: float
    autocorr_lag20: float
    trend_persistence_score: float
    
    # Stationarity
    adf_stationary: bool
    adf_pvalue: float
    adf_test_statistic: float
    
    # Distribution
    z_score: float
    is_outlier: bool
    rolling_mean: float
    rolling_std: float
    distribution_stable: bool
    
    # Correlation
    avg_correlation: float
    max_correlation: float
    correlation_dispersion: float
    
    # Volatility
    volatility_percentile: float
    volatility_regime: str  # LOW, NORMAL, HIGH
    current_volatility: float
```

### SignalValidation

```python
@dataclass
class SignalValidation:
    signal_validity_flag: bool           # Overall validity
    trend_persistence_check: bool        # Gate 1
    stationarity_check: bool            # Gate 2
    distribution_check: bool            # Gate 3
    correlation_check: bool             # Gate 4
    volatility_regime_check: bool       # Gate 5
    trend_consistency_score: float      # 0-1 composite score
    regime_quality_score: float         # 0-1 composite score
    failure_reasons: List[str]          # Detailed failure reasons
```

### RegimeState

```python
@dataclass
class RegimeState:
    trend_regime: str                   # STRONG_TREND, WEAK_TREND, MEAN_REVERTING
    volatility_regime: str              # LOW, NORMAL, HIGH
    correlation_regime: str             # LOW_CORRELATION, NORMAL_CORRELATION, HIGH_CORRELATION
    market_phase: str                   # BREAKOUT, RANGING, VOLATILE, NORMAL
    efficiency_ratio: float             # 0-1 (trend efficiency)
    trend_regime_stable: bool
    volatility_regime_stable: bool
```

## Testing

Run QSE tests:

```bash
pytest tests/test_quant_stats.py -v
```

Run demo:

```bash
python demo_qse.py
```

## Performance Characteristics

### Computational Complexity

| Module | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Autocorrelation | O(k*w) where k=lags, w=window | O(w) |
| Stationarity | O(w*log(w)) | O(w) |
| Distribution | O(w) | O(w) |
| Correlation | O(n^2*w) where n=symbols | O(n^2) |
| Volatility | O(w) | O(w) |
| **Total** | **O(n^2*w + k*w)** | **O(n^2 + w)** |

### Typical Processing Times

| Operation | Time (ms) |
|-----------|-----------|
| Single bar (1 symbol) | 5-10 ms |
| Single bar (10 symbols) | 15-25 ms |
| 1000 bars (1 symbol) | 3-5 seconds |

## Best Practices

### 1. Causal Computation
Always use data up to and including the current bar only:
```python
# ✓ CORRECT: Causal
window_data = returns.iloc[:bar_index+1].tail(rolling_window)

# ✗ WRONG: Lookahead bias
window_data = returns.iloc[bar_index:bar_index+rolling_window]
```

### 2. Configuration Management
Version your configs for reproducibility:
```python
# Save config hash with results
output.config_hash  # "a3f5b8c2d1e4f6a7"

# Load config later
config = load_config_by_hash("a3f5b8c2d1e4f6a7")
```

### 3. Failure Handling
Always check failure reasons:
```python
if not output.validation.signal_validity_flag:
    for reason in output.validation.failure_reasons:
        logger.warning(f"QSE rejection: {reason}")
```

### 4. Multi-Symbol Analysis
Use correlation analysis for portfolio-level signals:
```python
# Prepare all symbols in portfolio
returns_dict = {sym: get_returns(sym) for sym in portfolio}

# Check correlation before position sizing
output = qse.process_bar(..., returns_dict=returns_dict)
if output.metrics.max_correlation > 0.85:
    # Reduce position size due to correlation
    size = size * 0.5
```

## Integration Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    ARBITREX PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Raw Layer → OHLCV data ingestion                         │
│  2. Feature Engine → Compute X_t (17 features)               │
│  3. QSE → Validate X_t (5 statistical gates)                 │
│       │                                                       │
│       ├─ VALID → Forward to ML Layer                         │
│       │    ├─ ML Model → Predict                             │
│       │    └─ Signal Generator → Use ML prediction           │
│       │                                                       │
│       └─ INVALID → Suppress signal                           │
│            └─ Signal Generator → Use fallback logic          │
│                                                               │
│  4. Risk Manager → Position sizing + validation              │
│  5. Execution → Order placement                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: All signals marked invalid

**Symptoms:** `validity_rate == 0%`

**Diagnosis:**
```python
summary = qse.get_validation_summary(results)
print(summary['failure_breakdown'])
```

**Solutions:**
- Relax thresholds in config
- Check data quality (sufficient history, no NaNs)
- Verify return series is properly computed

### Issue: ADF tests always fail

**Symptoms:** `stationarity_check == False` for all bars

**Diagnosis:**
```python
print(f"ADF p-value: {output.metrics.adf_pvalue}")
```

**Solutions:**
- Increase rolling window (more data for test)
- Use differenced returns (already done in most cases)
- Disable stationarity requirement: `config.validation.require_stationarity = False`

### Issue: Statsmodels not available

**Symptoms:** Warning: "statsmodels not available, ADF tests will be disabled"

**Solution:**
```bash
pip install statsmodels
```

## References

- **ADF Test:** Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root"
- **Autocorrelation:** Box, G. E. P., & Jenkins, G. M. (1970). "Time Series Analysis: Forecasting and Control"
- **Z-scores:** Standard deviation-based outlier detection
- **Volatility Regimes:** Percentile-based regime classification

## License

Internal use only - Arbitrex Trading System
