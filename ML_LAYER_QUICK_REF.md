# ML Layer - Quick Reference

## 🚀 Quick Start

```python
from arbitrex.ml_layer import MLInferenceEngine

# Initialize
ml_engine = MLInferenceEngine()

# Predict
output = ml_engine.predict(
    symbol="EURUSD",
    timeframe="4H",
    feature_df=feature_df,     # From Feature Engine
    qse_output=qse_output      # From QSE
)

# Decision
if output.prediction.allow_trade:
    # ✅ Proceed to Signal Generator
    pass
else:
    # ❌ Suppress signal
    pass
```

---

## 📊 Outputs

### Regime
```python
output.prediction.regime.regime_label      # TRENDING/RANGING/STRESSED
output.prediction.regime.regime_confidence # 0.0 - 1.0
output.prediction.regime.efficiency_ratio  # ER metric
```

### Signal
```python
output.prediction.signal.momentum_success_prob  # P(success)
output.prediction.signal.should_enter           # bool
output.prediction.signal.confidence_level       # HIGH/MEDIUM/LOW
output.prediction.signal.top_features           # Dict[str, float]
```

### Decision
```python
output.prediction.allow_trade       # Final decision (bool)
output.prediction.decision_reasons  # List[str]
```

---

## ⚙️ Configuration

```python
from arbitrex.ml_layer import MLConfig

config = MLConfig()

# Regime
config.regime.trending_min_efficiency = 0.65
config.regime.min_confidence = 0.60

# Signal
config.signal_filter.entry_threshold = 0.55
config.signal_filter.exit_threshold = 0.45
config.signal_filter.allowed_regimes = ['TRENDING']
```

---

## 🧪 Testing

```bash
# Full integration test
python test_ml_layer.py

# Quick demo
python demo_ml_layer.py
```

---

## 📈 Performance

- **Prediction Time**: ~1ms
- **Memory**: ~100MB
- **Throughput**: ~1000 pred/sec

---

## 🔍 Key Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `entry_threshold` | 0.55 | Enter if P > 0.55 |
| `exit_threshold` | 0.45 | Exit if P < 0.45 |
| `trending_min_efficiency` | 0.65 | ER > 0.65 = TRENDING |
| `regime.min_confidence` | 0.60 | Min confidence required |

---

## 🎯 Decision Logic

1. **QSE Valid?** → If NO, suppress
2. **Regime Allowed?** → Only TRENDING
3. **Regime Confidence > 0.60?** → If NO, suppress
4. **Signal Prob > 0.55?** → If NO, suppress
5. **All Pass** → Allow trade

---

## 📁 Files

- `arbitrex/ml_layer/inference.py` - Main engine
- `arbitrex/ml_layer/config.py` - Configuration
- `arbitrex/ml_layer/schemas.py` - Output structures
- `arbitrex/ml_layer/README.md` - Full documentation

---

## 🔗 Integration

### Complete Pipeline
```python
# 1. Feature Engine
feature_df, _ = feature_pipeline.compute_features(symbol, clean_data)

# 2. QSE
qse_output = qse.process_bar(symbol, returns, bar_index)

# 3. ML Layer
ml_output = ml_engine.predict(symbol, "4H", feature_df, qse_output.to_dict())

# 4. Decision
if ml_output.prediction.allow_trade:
    signal_generator.generate(symbol, ml_output)
```

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Updated:** December 22, 2025
