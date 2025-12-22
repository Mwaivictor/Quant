# ML Layer Integration Guide

## Complete Pipeline Integration

This guide shows how to integrate the ML Layer into the full ArbitreX pipeline.

---

## 🔄 Data Flow

```
Raw Data (MT5)
    ↓
Clean Data Layer (valid_bar gate, log_return_1)
    ↓
Feature Engine (16 ML features)
    ↓
Quantitative Statistics Engine (5-gate validation)
    ↓
ML Layer (regime + signal filter) ← YOU ARE HERE
    ↓
Signal Generator (if allowed)
    ↓
Risk Manager (position sizing)
    ↓
Execution
```

---

## 📝 Step-by-Step Integration

### Step 1: Initialize All Components

```python
from arbitrex.clean import CleanDataPipeline, RawToCleanBridge
from arbitrex.features import FeaturePipeline
from arbitrex.quant_stats import QuantitativeStatisticsEngine
from arbitrex.ml_layer import MLInferenceEngine

# Initialize pipeline components
clean_bridge = RawToCleanBridge()
feature_pipeline = FeaturePipeline()
qse = QuantitativeStatisticsEngine()
ml_engine = MLInferenceEngine()

print("✓ All components initialized")
```

### Step 2: Process Single Symbol

```python
def process_symbol_ml(symbol: str, raw_data_path: str):
    """Process symbol through complete ML pipeline"""
    
    # 1. Clean Data Layer
    cleaned_df, clean_meta = clean_bridge.process_csv(raw_data_path)
    valid_bars = cleaned_df[cleaned_df['valid_bar'] == True]
    
    print(f"[{symbol}] Clean: {len(valid_bars)} valid bars")
    
    # 2. Feature Engine
    feature_df, feature_meta = feature_pipeline.compute_features(
        symbol=symbol,
        clean_data=valid_bars
    )
    
    print(f"[{symbol}] Features: {len(feature_df)} bars, {len(feature_df.columns)} features")
    
    # 3. QSE Validation
    returns = feature_df['log_return_1']
    bar_index = len(returns) - 1
    
    qse_output = qse.process_bar(
        symbol=symbol,
        returns=returns,
        bar_index=bar_index
    )
    
    qse_valid = qse_output.validation.signal_validity_flag
    print(f"[{symbol}] QSE: {'✓ VALID' if qse_valid else '✗ INVALID'}")
    
    if not qse_valid:
        print(f"  Reasons: {qse_output.validation.failure_reasons}")
        return None  # Signal suppressed by QSE
    
    # 4. ML Layer
    ml_output = ml_engine.predict(
        symbol=symbol,
        timeframe="4H",
        feature_df=feature_df,
        qse_output=qse_output.to_dict(),
        bar_index=bar_index
    )
    
    regime = ml_output.prediction.regime.regime_label.value
    prob = ml_output.prediction.signal.momentum_success_prob
    allow = ml_output.prediction.allow_trade
    
    print(f"[{symbol}] ML: Regime={regime}, P={prob:.3f}, {'✓ ALLOW' if allow else '✗ SUPPRESS'}")
    
    if not allow:
        print(f"  Reasons: {ml_output.prediction.decision_reasons[0]}")
        return None  # Signal suppressed by ML
    
    # 5. Signal approved - return for Signal Generator
    return {
        'symbol': symbol,
        'feature_df': feature_df,
        'qse_output': qse_output,
        'ml_output': ml_output,
        'regime': regime,
        'signal_prob': prob,
        'confidence': ml_output.prediction.signal.confidence_level
    }


# Example usage
result = process_symbol_ml('EURUSD', 'data/EURUSD_4H.csv')

if result:
    print(f"\n✓ Signal approved for {result['symbol']}")
    print(f"  Regime: {result['regime']}")
    print(f"  Probability: {result['signal_prob']:.1%}")
    print(f"  Confidence: {result['confidence']}")
    # → Proceed to Signal Generator
else:
    print("\n✗ Signal suppressed")
```

### Step 3: Batch Processing (Multiple Symbols)

```python
def process_portfolio_ml(symbols: list, data_paths: dict):
    """Process multiple symbols through ML pipeline"""
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        result = process_symbol_ml(symbol, data_paths[symbol])
        
        if result:
            results[symbol] = result
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    print(f"Symbols processed: {len(symbols)}")
    print(f"Signals approved: {len(results)}")
    print(f"Signals suppressed: {len(symbols) - len(results)}")
    
    for symbol, result in results.items():
        print(f"  {symbol}: {result['regime']:10s} | P={result['signal_prob']:.3f}")
    
    return results


# Example usage
symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
data_paths = {
    'EURUSD': 'data/EURUSD_4H.csv',
    'GBPUSD': 'data/GBPUSD_4H.csv',
    'XAUUSD': 'data/XAUUSD_4H.csv'
}

approved_signals = process_portfolio_ml(symbols, data_paths)
```

---

## 🎯 Signal Generator Integration

Once ML Layer approves a signal, pass to Signal Generator:

```python
def generate_signals(approved_signals: dict):
    """Generate trading signals from ML-approved symbols"""
    
    signals = []
    
    for symbol, result in approved_signals.items():
        # Extract ML context
        regime = result['regime']
        prob = result['signal_prob']
        confidence = result['confidence']
        
        # Regime-based signal adjustment
        if regime == 'TRENDING':
            signal_strength = 1.0  # Full strength
            holding_period = 10    # bars
        elif regime == 'RANGING':
            continue  # Should not reach here (filtered by ML)
        elif regime == 'STRESSED':
            continue  # Should not reach here (filtered by ML)
        
        # Confidence-based adjustments
        if confidence == 'HIGH':
            position_size_multiplier = 1.2
        elif confidence == 'LOW':
            position_size_multiplier = 0.8
        else:  # MEDIUM
            position_size_multiplier = 1.0
        
        # Create signal
        signal = {
            'symbol': symbol,
            'direction': 'LONG' if result['feature_df']['momentum_score'].iloc[-1] > 0 else 'SHORT',
            'strength': signal_strength,
            'confidence': confidence,
            'probability': prob,
            'regime': regime,
            'position_size_multiplier': position_size_multiplier,
            'holding_period': holding_period,
            'ml_metadata': {
                'config_hash': result['ml_output'].config_hash,
                'processing_time_ms': result['ml_output'].processing_time_ms,
                'top_features': result['ml_output'].prediction.signal.top_features
            }
        }
        
        signals.append(signal)
    
    return signals


# Example usage
signals = generate_signals(approved_signals)

for signal in signals:
    print(f"\n{signal['symbol']} Signal:")
    print(f"  Direction: {signal['direction']}")
    print(f"  Regime: {signal['regime']}")
    print(f"  Probability: {signal['probability']:.1%}")
    print(f"  Confidence: {signal['confidence']}")
    print(f"  Position Size Multiplier: {signal['position_size_multiplier']:.1f}x")
```

---

## 🔄 Real-Time Integration

For live trading:

```python
import time

def run_ml_pipeline_realtime(symbols: list, interval_seconds: int = 3600):
    """Run ML pipeline in real-time loop"""
    
    # Initialize components (once)
    ml_engine = MLInferenceEngine()
    feature_pipeline = FeaturePipeline()
    qse = QuantitativeStatisticsEngine()
    
    print("Real-time ML pipeline started")
    
    while True:
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing...")
            
            # Fetch latest data (from MT5 or data source)
            # feature_dfs = fetch_latest_features(symbols)
            
            # Process through ML Layer
            for symbol in symbols:
                # Get features
                feature_df = fetch_features(symbol)
                
                # QSE
                returns = feature_df['log_return_1']
                qse_out = qse.process_bar(symbol, returns, len(returns)-1)
                
                # ML
                ml_out = ml_engine.predict(
                    symbol, "4H", feature_df, qse_out.to_dict()
                )
                
                # Act on decision
                if ml_out.prediction.allow_trade:
                    # Send to Signal Generator
                    send_to_signal_generator(symbol, ml_out)
                else:
                    # Log suppression
                    log_suppression(symbol, ml_out.prediction.decision_reasons)
            
            # Wait for next interval
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            print("\nPipeline stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)  # Wait before retry
```

---

## 📊 Monitoring & Logging

Track ML Layer performance:

```python
import json
from datetime import datetime

class MLPerformanceMonitor:
    """Monitor ML Layer decisions"""
    
    def __init__(self):
        self.decisions = []
    
    def log_decision(self, symbol: str, ml_output, actual_return: float = None):
        """Log ML decision for analysis"""
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'regime': ml_output.prediction.regime.regime_label.value,
            'signal_prob': ml_output.prediction.signal.momentum_success_prob,
            'allowed': ml_output.prediction.allow_trade,
            'reasons': ml_output.prediction.decision_reasons,
            'actual_return': actual_return  # To evaluate later
        }
        
        self.decisions.append(decision)
    
    def export_report(self, filepath: str):
        """Export decisions to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.decisions, f, indent=2)
    
    def get_stats(self):
        """Get summary statistics"""
        total = len(self.decisions)
        allowed = sum(1 for d in self.decisions if d['allowed'])
        suppressed = total - allowed
        
        return {
            'total_decisions': total,
            'allowed': allowed,
            'suppressed': suppressed,
            'allow_rate': allowed / max(total, 1)
        }


# Usage
monitor = MLPerformanceMonitor()

for symbol in symbols:
    ml_output = ml_engine.predict(...)
    monitor.log_decision(symbol, ml_output)

# Get stats
stats = monitor.get_stats()
print(f"ML Layer Stats: {stats['allow_rate']:.1%} allow rate")

# Export for analysis
monitor.export_report('ml_decisions.json')
```

---

## 🎯 Best Practices

### 1. Always Check QSE First
```python
if not qse_output.validation.signal_validity_flag:
    return None  # Don't even call ML Layer
```

### 2. Use Regime Context
```python
if ml_output.prediction.regime.regime_label == 'STRESSED':
    # Reduce position size even if allowed (defensive)
    position_size *= 0.5
```

### 3. Respect Confidence Levels
```python
if ml_output.prediction.signal.confidence_level == 'LOW':
    # Tighter stops, smaller position
    stop_loss_multiplier = 0.7
```

### 4. Log All Decisions
```python
# For debugging and performance analysis
logger.info(f"ML Decision: {ml_output.to_dict()}")
```

### 5. Monitor Allow Rate
```python
# Alert if allow rate drops significantly
if allow_rate < 0.10:  # Less than 10%
    alert("ML Layer suppressing too many signals")
```

---

## 🔧 Troubleshooting

### Issue: All signals suppressed
```python
# Check regime classification
print(f"Regime: {ml_output.prediction.regime.regime_label}")
# If all RANGING → adjust regime thresholds

# Check signal probability
print(f"Prob: {ml_output.prediction.signal.momentum_success_prob}")
# If all below 0.55 → lower entry_threshold or check features
```

### Issue: Processing too slow
```python
# Use batch processing
results = ml_engine.batch_predict(symbols, "4H", feature_dfs, qse_outputs)
# Should be ~3ms for 3 symbols vs 3ms if done individually
```

### Issue: Unexpected decisions
```python
# Check feature importance
print(f"Top features: {ml_output.prediction.signal.top_features}")
# Verify feature values make sense
```

---

## 📈 Performance Tuning

### Optimize Thresholds
```python
# Start conservative
config.signal_filter.entry_threshold = 0.60  # Higher = fewer trades

# Gradually lower based on backtest results
# Target: 40-60% allow rate
```

### Adjust Regime Filtering
```python
# Allow RANGING if confident
config.signal_filter.allowed_regimes = ['TRENDING', 'RANGING']
config.signal_filter.entry_threshold = 0.65  # Higher bar for RANGING
```

---

## ✅ Integration Checklist

- [ ] All components initialized
- [ ] Data flow validated (Raw → Clean → Features → QSE → ML)
- [ ] QSE validation checked before ML
- [ ] ML decision used to gate Signal Generator
- [ ] Regime context used for signal adjustments
- [ ] Confidence levels respected
- [ ] All decisions logged
- [ ] Performance monitored
- [ ] Thresholds tuned

---

**Status:** Ready for Integration  
**Next Step:** Connect to Signal Generator  
**Documentation:** See `arbitrex/ml_layer/README.md`
