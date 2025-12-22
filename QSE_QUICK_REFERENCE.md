# QSE Quick Reference Card

## 🎯 Quick Access

### Start API Server
```bash
python start_qse_api.py
# API: http://localhost:8002
# Docs: http://localhost:8002/docs
```

### Run Tests
```bash
python test_qse_api.py              # Basic test
python test_qse_integration.py      # Full integration
python test_qse_quick.py            # Quick validation
```

---

## 📡 API Endpoints (Port 8002)

### Validation
```bash
# Validate signal
POST /validate
{
  "symbol": "EURUSD",
  "timeframe": "1H", 
  "returns": [0.001, -0.002, ...],
  "bar_index": 100
}
```

### Health
```bash
GET /health                  # Overall status
GET /health/EURUSD          # Symbol health
GET /failures               # Failure breakdown
GET /recent?limit=20        # Recent validations
```

### Info
```bash
GET /config                 # Configuration
GET /                       # API info
```

---

## 💊 Health Monitor Usage

```python
from arbitrex.quant_stats import QSEHealthMonitor

health = QSEHealthMonitor()

# Record validation
start = health.record_validation_start(symbol)

# Record result
if valid:
    health.record_validation_success(symbol, start, metrics)
else:
    health.record_validation_failure(symbol, start, reasons, metrics)

# Check status
status = health.get_health_status()
print(f"Status: {status['status']}")
print(f"Validity: {status['validity_rate']:.1%}")

# Export report
health.export_health_report('report.json')
```

---

## 🔍 Integration Example

```python
from arbitrex.quant_stats import QuantitativeStatisticsEngine, QSEHealthMonitor

qse = QuantitativeStatisticsEngine()
health = QSEHealthMonitor()

# Validate
start = health.record_validation_start(symbol)
output = qse.process_bar(symbol, returns, bar_index)

if output.validation.signal_validity_flag:
    health.record_validation_success(symbol, start, metrics)
    # ✅ Proceed to ML
else:
    health.record_validation_failure(symbol, start, reasons, metrics)
    # ❌ Suppress signal
```

---

## 📊 Health Status Levels

| Status | Validity Rate | Action |
|--------|---------------|--------|
| 🟢 HEALTHY | ≥80% | Normal operation |
| 🟡 DEGRADED | 50-80% | Monitor closely |
| 🔴 UNHEALTHY | <50% | Investigate immediately |

---

## 🏃 Performance

- **Validation Time**: 10-30ms
- **Throughput**: 50-100 val/sec
- **Memory**: ~50MB base
- **API Response**: <1ms (health) / ~20ms (validate)

---

## 📁 Files

### Core
- `arbitrex/quant_stats/api.py` - REST API
- `arbitrex/quant_stats/health_monitor.py` - Health tracking
- `arbitrex/quant_stats/engine.py` - QSE orchestrator

### Tests
- `test_qse_api.py` - Basic tests
- `test_qse_integration.py` - Integration tests
- `test_qse_quick.py` - Quick validation

### Docs
- `API_HEALTH_README.md` - Full documentation
- `QSE_API_HEALTH_SUMMARY.md` - Implementation summary
- `QSE_QUICK_REFERENCE.md` - This file

### Utils
- `start_qse_api.py` - Server startup

---

## 🚨 Monitoring Alerts

Set alerts for:
- ✋ Validity rate < 50% (UNHEALTHY)
- ✋ Consecutive failures ≥ 5
- ✋ Processing time > 50ms
- ✋ Unhealthy symbols > 2

---

## 🔧 Common Tasks

### Check System Health
```bash
curl http://localhost:8002/health
```

### Get Failure Stats
```bash
curl http://localhost:8002/failures
```

### Validate Signal
```bash
curl -X POST http://localhost:8002/validate \
  -H "Content-Type: application/json" \
  -d @signal.json
```

### Export Health Report
```python
health.export_health_report('health_report.json')
```

---

## 📈 Next Steps

1. ✅ QSE Engine - COMPLETE
2. ✅ QSE API - COMPLETE
3. ✅ Health Monitor - COMPLETE
4. ⏳ ML Pipeline Integration - NEXT
5. ⏳ Signal Generator Integration
6. ⏳ Production Monitoring

---

**Documentation**: See `API_HEALTH_README.md` for full details  
**Summary**: See `QSE_API_HEALTH_SUMMARY.md` for implementation overview
