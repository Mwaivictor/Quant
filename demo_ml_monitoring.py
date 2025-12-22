"""
Demo: ML Layer API + Monitoring

Shows complete workflow:
1. Start monitoring
2. Make predictions via API
3. View metrics and alerts
"""

import sys
sys.path.insert(0, '.')

from arbitrex.ml_layer import MLConfig, MLInferenceEngine, MLMonitor
import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("ML Layer API + Monitoring Demo")
print("="*70)

# 1. Initialize components
print("\n1. Initializing ML Layer with monitoring...")
config = MLConfig()
ml_engine = MLInferenceEngine(config)
monitor = MLMonitor(config, log_dir="logs/ml_layer")
print("   ✓ ML Engine initialized")
print("   ✓ Monitor initialized")

# 2. Generate test data for 3 symbols
print("\n2. Generating test data for 3 symbols...")
np.random.seed(42)
symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
n_bars = 150

predictions = []

for symbol in symbols:
    # Generate features
    feature_df = pd.DataFrame({
        'momentum_score': np.random.randn(n_bars) * 0.5,
        'log_return_1': np.random.randn(n_bars) * 0.01,
        'atr': np.random.uniform(0.001, 0.005, n_bars),
        'efficiency_ratio': np.random.uniform(0.3, 0.8, n_bars),
        'volatility': np.random.uniform(0.01, 0.03, n_bars),
        'trend_persistence': np.random.uniform(0.4, 0.7, n_bars),
        'correlation': np.random.uniform(-0.5, 0.5, n_bars),
        'returns_autocorr': np.random.uniform(-0.3, 0.3, n_bars),
    })
    
    # Mock QSE output
    qse_output = {
        'validation': {
            'signal_validity_flag': True,
            'failure_reasons': []
        },
        'metrics': {
            'trend_persistence': 0.65,
            'stationarity_pvalue': 0.03,
            'z_score_normalized': 1.2
        }
    }
    
    # Make prediction
    output = ml_engine.predict(
        symbol=symbol,
        timeframe='4H',
        feature_df=feature_df,
        qse_output=qse_output
    )
    
    # Log to monitor
    monitor.log_prediction(symbol, '4H', output)
    
    predictions.append((symbol, output))
    
    # Display result
    regime = output.prediction.regime.regime_label.value
    prob = output.prediction.signal.momentum_success_prob
    allowed = output.prediction.allow_trade
    confidence = output.prediction.signal.confidence_level
    
    print(f"   {symbol:8s} | {regime:10s} | P={prob:.3f} | {confidence:7s} | {'✓ ALLOW' if allowed else '✗ SUPPRESS'}")

# 3. View monitoring metrics
print("\n3. Monitoring Metrics")
print("-" * 70)
metrics = monitor.get_current_metrics()

print(f"   Total predictions: {metrics['total_predictions']}")
print(f"   Allowed: {metrics['allowed_predictions']}")
print(f"   Suppressed: {metrics['suppressed_predictions']}")
print(f"   Allow rate: {metrics['allow_rate']:.1%}")
print(f"   Avg processing time: {metrics.get('avg_processing_time_ms', 0):.2f}ms")

# Regime distribution
if 'regime_distribution' in metrics:
    print(f"\n   Regime Distribution:")
    for regime, pct in metrics['regime_distribution'].items():
        print(f"      {regime}: {pct:.1%}")

# 4. Decision history
print("\n4. Decision History (Last 5)")
print("-" * 70)
decisions = monitor.get_decision_history(limit=5)
for i, decision in enumerate(decisions, 1):
    print(f"   {i}. {decision['symbol']:8s} | {decision['regime']:10s} | "
          f"P={decision['signal_prob']:.3f} | "
          f"{'✓' if decision['allowed'] else '✗'} | "
          f"{decision['decision_reasons'][0] if not decision['allowed'] else 'All checks passed'}")

# 5. Check alerts
print("\n5. Active Alerts")
print("-" * 70)
alerts = monitor.get_active_alerts()
if alerts:
    for alert in alerts:
        print(f"   [{alert['severity'].upper()}] {alert['message']}")
        print(f"      Triggered: {alert['timestamp']}")
else:
    print("   No active alerts ✓")

# 6. Performance summary
print("\n6. Performance Summary")
print("-" * 70)
summary = monitor.get_performance_summary(hours=1)
print(f"   Predictions in last hour: {summary['total_predictions']}")
print(f"   Allow rate: {summary['allow_rate']:.1%}")
print(f"   Avg signal probability: {summary['avg_signal_prob']:.3f}")
print(f"   Avg regime confidence: {summary['avg_regime_confidence']:.3f}")
print(f"   Avg processing time: {summary['avg_processing_time_ms']:.2f}ms")

# 7. Export metrics
print("\n7. Exporting Metrics")
print("-" * 70)
export_file = f"ml_metrics_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
monitor.export_metrics(export_file)
print(f"   ✓ Metrics exported to: {export_file}")

# 8. Prometheus format
print("\n8. Prometheus Metrics (sample)")
print("-" * 70)
prom_output = monitor.export_prometheus()
# Show first 10 lines
for line in prom_output.split('\n')[:10]:
    print(f"   {line}")
print("   ...")

# 9. Alert rules
print("\n9. Alert Rules")
print("-" * 70)
rules = monitor.get_alert_rules()
print(f"   Total rules: {len(rules)}")
for rule in rules[:3]:  # Show first 3
    status = "✓" if rule['enabled'] else "✗"
    print(f"   {status} {rule['name']:25s} | {rule['metric']:20s} {rule['operator']} {rule['threshold']}")

print("\n" + "="*70)
print("Demo Complete!")
print("="*70)
print("\nNext Steps:")
print("1. Start API server: python start_ml_api.py")
print("2. Test API: python test_ml_api.py")
print("3. View logs: logs/ml_layer/predictions.jsonl")
print("4. View docs: arbitrex/ml_layer/MONITORING.md")
