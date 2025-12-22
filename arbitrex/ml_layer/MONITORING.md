# ML Layer Monitoring Guide

## Overview

The ML Layer monitoring system tracks all aspects of ML operations:
- **Predictions**: Latency, decisions, regime distribution
- **Performance**: Model accuracy, drift detection
- **System Health**: Memory, throughput, errors
- **Alerts**: Performance degradation, anomalies
- **Audit**: Complete prediction logs for compliance

---

## Quick Start

### Initialize Monitor

```python
from arbitrex.ml_layer import MLMonitor, MLConfig

# Create monitor
config = MLConfig()
monitor = MLMonitor(config, log_dir="logs/ml_layer")
```

### Log Predictions

```python
from arbitrex.ml_layer import MLInferenceEngine

# Initialize engine
ml_engine = MLInferenceEngine(config)

# Make prediction
output = ml_engine.predict(symbol, timeframe, feature_df, qse_output)

# Log for monitoring
monitor.log_prediction(symbol, timeframe, output)
```

### Get Metrics

```python
# Current metrics
metrics = monitor.get_current_metrics()

print(f"Total predictions: {metrics['total_predictions']}")
print(f"Allow rate: {metrics['allow_rate']:.1%}")
print(f"Avg latency: {metrics['avg_processing_time_ms']:.2f}ms")
```

---

## Metrics Tracked

### 1. Prediction Metrics

| Metric | Description |
|--------|-------------|
| `total_predictions` | Total number of predictions |
| `allowed_predictions` | Predictions that passed all gates |
| `suppressed_predictions` | Predictions that were suppressed |
| `allow_rate` | Percentage of allowed predictions |
| `suppression_rate` | Percentage of suppressed predictions |

### 2. Performance Metrics

| Metric | Description |
|--------|-------------|
| `avg_processing_time_ms` | Average prediction latency |
| `p50_processing_time_ms` | Median latency |
| `p95_processing_time_ms` | 95th percentile latency |
| `p99_processing_time_ms` | 99th percentile latency |
| `max_processing_time_ms` | Maximum latency observed |

### 3. Regime Metrics

| Metric | Description |
|--------|-------------|
| `regime_distribution` | % of predictions per regime |
| `avg_regime_confidence` | Average regime confidence score |
| `min_regime_confidence` | Minimum confidence observed |

### 4. System Metrics

| Metric | Description |
|--------|-------------|
| `predictions_per_hour` | Throughput rate |
| `symbol_distribution` | Predictions per symbol |
| `uptime_seconds` | System uptime |

---

## Alert System

### Default Alert Rules

```python
# High suppression rate (90%+)
AlertRule(
    name="high_suppression_rate",
    metric="suppression_rate",
    operator=">",
    threshold=0.90,
    severity="warning"
)

# High latency (>10ms)
AlertRule(
    name="high_latency",
    metric="avg_processing_time_ms",
    operator=">",
    threshold=10.0,
    severity="warning"
)

# Stressed regime spike (50%+)
AlertRule(
    name="stressed_regime_spike",
    metric="stressed_regime_pct",
    operator=">",
    threshold=0.50,
    severity="critical"
)

# Low confidence (<50%)
AlertRule(
    name="low_confidence",
    metric="avg_regime_confidence",
    operator="<",
    threshold=0.50,
    severity="warning"
)
```

### Add Custom Alert

```python
from arbitrex.ml_layer.monitoring import AlertRule

# Create custom alert
custom_rule = AlertRule(
    name="very_low_allow_rate",
    metric="allow_rate",
    operator="<",
    threshold=0.05,  # Less than 5% allowed
    window_minutes=30,
    severity="critical",
    enabled=True
)

# Add to monitor
monitor.add_alert_rule(custom_rule)
```

### Get Active Alerts

```python
# Get alerts from last 24 hours
alerts = monitor.get_active_alerts()

for alert in alerts:
    print(f"{alert['severity']}: {alert['message']}")
    print(f"  Triggered: {alert['timestamp']}")
    print(f"  Value: {alert['metric_value']:.2f} (threshold: {alert['threshold']})")
```

### Manage Alerts

```python
# Disable alert rule
monitor.disable_alert_rule("high_suppression_rate")

# Enable alert rule
monitor.enable_alert_rule("high_suppression_rate")

# Clear all alerts
monitor.clear_alerts()

# Clear alerts by severity
monitor.clear_alerts(severity="warning")
```

---

## Decision History

### View Recent Decisions

```python
# Get last 100 decisions
decisions = monitor.get_decision_history(limit=100)

for decision in decisions[-10:]:  # Last 10
    print(f"{decision['symbol']:8s} | {decision['regime']:10s} | "
          f"P={decision['signal_prob']:.3f} | "
          f"{'✓ ALLOW' if decision['allowed'] else '✗ SUPPRESS'}")
```

### Analyze Decisions

```python
import pandas as pd

# Convert to DataFrame for analysis
decisions_df = pd.DataFrame(monitor.get_decision_history(limit=1000))

# Group by regime
regime_stats = decisions_df.groupby('regime').agg({
    'allowed': ['count', 'sum', 'mean']
})
print(regime_stats)

# Group by symbol
symbol_stats = decisions_df.groupby('symbol').agg({
    'allowed': ['count', 'mean'],
    'signal_prob': ['mean', 'std']
})
print(symbol_stats)
```

---

## Performance Analysis

### Get Performance Summary

```python
# Last 24 hours
summary_24h = monitor.get_performance_summary(hours=24)

print(f"24-Hour Performance:")
print(f"  Total predictions: {summary_24h['total_predictions']}")
print(f"  Allow rate: {summary_24h['allow_rate']:.1%}")
print(f"  Avg latency: {summary_24h['avg_processing_time_ms']:.2f}ms")
print(f"  Avg signal prob: {summary_24h['avg_signal_prob']:.3f}")
print(f"  Regime distribution:")
for regime, pct in summary_24h['regime_distribution'].items():
    print(f"    {regime}: {pct:.1%}")
```

### Export Metrics

```python
# Export to JSON
monitor.export_metrics("ml_metrics_20251222.json")

# Data includes:
# - Current metrics
# - Active alerts
# - Recent decisions (last 50)
# - Timestamp and uptime
```

---

## Prometheus Integration

### Export Prometheus Metrics

```python
# Get Prometheus format
prometheus_output = monitor.export_prometheus()

# Output includes:
# ml_layer_predictions_total
# ml_layer_predictions_allowed
# ml_layer_predictions_suppressed
# ml_layer_allow_rate
# ml_layer_suppression_rate
# ml_layer_processing_time_ms{quantile="0.5"}
# ml_layer_processing_time_ms{quantile="0.95"}
# ml_layer_processing_time_ms{quantile="0.99"}
# ml_layer_regime_distribution{regime="TRENDING"}
# ml_layer_regime_distribution{regime="RANGING"}
# ml_layer_regime_distribution{regime="STRESSED"}
# ml_layer_predictions_per_hour
# ml_layer_active_alerts
```

### Grafana Dashboard

Create Grafana dashboard with panels:

1. **Prediction Volume**
   - Query: `rate(ml_layer_predictions_total[5m])`
   - Visualization: Graph

2. **Allow Rate**
   - Query: `ml_layer_allow_rate`
   - Visualization: Gauge (0-1)

3. **Processing Latency**
   - Query: `ml_layer_processing_time_ms`
   - Visualization: Heatmap

4. **Regime Distribution**
   - Query: `ml_layer_regime_distribution`
   - Visualization: Pie chart

5. **Active Alerts**
   - Query: `ml_layer_active_alerts`
   - Visualization: Stat panel

---

## API Integration

The ML Layer API automatically integrates monitoring:

### Start API with Monitoring

```python
# API automatically initializes monitor
python start_ml_api.py

# Access metrics via API
curl http://localhost:8003/metrics
curl http://localhost:8003/alerts
curl http://localhost:8003/metrics/decisions?limit=50
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/metrics` | Current metrics |
| `/metrics/prometheus` | Prometheus format |
| `/metrics/decisions` | Decision history |
| `/alerts` | Active alerts |
| `/status` | Detailed status |

---

## Audit Logging

### Enable Prediction Logging

```python
# In config
config.governance.enable_prediction_logging = True

# All predictions logged to disk
# File: logs/ml_layer/predictions.jsonl
```

### Read Audit Log

```python
import json

# Read predictions log
with open('logs/ml_layer/predictions.jsonl', 'r') as f:
    for line in f:
        prediction = json.loads(line)
        print(f"{prediction['timestamp']} | {prediction['symbol']} | "
              f"Regime={prediction['regime']} | "
              f"Allowed={prediction['allowed']}")
```

### Analyze Audit Trail

```python
import pandas as pd

# Load all predictions
predictions = []
with open('logs/ml_layer/predictions.jsonl', 'r') as f:
    predictions = [json.loads(line) for line in f]

df = pd.DataFrame(predictions)

# Daily statistics
df['date'] = pd.to_datetime(df['timestamp']).dt.date
daily_stats = df.groupby('date').agg({
    'allowed': ['count', 'sum', 'mean'],
    'processing_time_ms': ['mean', 'max'],
    'regime_confidence': ['mean', 'min']
})

print(daily_stats)
```

---

## Best Practices

### 1. Monitor Allow Rate
- **Target**: 40-60% allow rate
- **Too high (>80%)**: ML may be too permissive
- **Too low (<20%)**: ML may be too restrictive

### 2. Track Latency
- **Target**: <5ms per prediction
- **Alert**: >10ms sustained latency
- **Action**: Investigate feature computation bottlenecks

### 3. Watch Regime Distribution
- **Normal**: Mix of TRENDING (30-50%), RANGING (30-50%), STRESSED (5-20%)
- **Alert**: >50% STRESSED (market instability)
- **Action**: Consider reducing position sizes

### 4. Monitor Confidence
- **Target**: Avg confidence >60%
- **Alert**: Confidence <50% sustained
- **Action**: May indicate model drift or regime change

### 5. Export Regularly
```python
# Daily export
import schedule

def daily_export():
    monitor.export_metrics(f"ml_metrics_{datetime.now().date()}.json")

schedule.every().day.at("23:59").do(daily_export)
```

---

## Troubleshooting

### High Suppression Rate

```python
metrics = monitor.get_current_metrics()

# Check why signals are suppressed
decisions = monitor.get_decision_history(limit=100)
reasons = {}
for d in decisions:
    if not d['allowed']:
        for reason in d['decision_reasons']:
            reasons[reason] = reasons.get(reason, 0) + 1

print("Suppression reasons:")
for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reason}: {count}")

# Common reasons:
# - "Regime not allowed: RANGING"
# - "Signal probability too low"
# - "QSE validation failed"
# - "Regime confidence too low"
```

### High Latency

```python
# Check processing time distribution
metrics = monitor.get_current_metrics()

print(f"Avg: {metrics['avg_processing_time_ms']:.2f}ms")
print(f"P95: {metrics['p95_processing_time_ms']:.2f}ms")
print(f"P99: {metrics['p99_processing_time_ms']:.2f}ms")
print(f"Max: {metrics['max_processing_time_ms']:.2f}ms")

# If P99 >> Avg → Investigate outliers
# If all high → Optimize feature computation
```

### Model Drift Detection

```python
# Compare recent vs historical performance
recent = monitor.get_performance_summary(hours=24)
historical = monitor.get_performance_summary(hours=168)  # 7 days

drift = {
    'allow_rate_change': recent['allow_rate'] - historical['allow_rate'],
    'avg_prob_change': recent['avg_signal_prob'] - historical['avg_signal_prob'],
    'confidence_change': recent['avg_regime_confidence'] - historical['avg_regime_confidence']
}

print("Drift analysis:")
for metric, change in drift.items():
    status = "⚠️" if abs(change) > 0.1 else "✓"
    print(f"{status} {metric}: {change:+.3f}")
```

---

## Summary

The ML Layer monitoring system provides:
- ✅ Real-time metrics tracking
- ✅ Configurable alert system
- ✅ Complete audit trail
- ✅ Performance analysis tools
- ✅ Prometheus/Grafana integration
- ✅ API-accessible metrics

**Monitor continuously, alert proactively, audit comprehensively.**
