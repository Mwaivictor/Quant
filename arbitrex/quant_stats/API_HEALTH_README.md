# QSE API & Health Monitoring

Complete REST API and health monitoring system for the Quantitative Statistics Engine.

## Components

### 1. REST API (`api.py`)
FastAPI-based REST API providing 10 endpoints for statistical validation and monitoring.

**Port:** 8002  
**Base URL:** `http://localhost:8002`

#### Endpoints

##### Core Validation
- **POST /validate** - Validate signal using 5-gate statistical tests
  ```json
  {
    "symbol": "EURUSD",
    "timeframe": "1H",
    "returns": [0.001, -0.002, 0.003, ...],
    "bar_index": 100
  }
  ```
  Returns: Full validation result with metrics, regime state, and processing time

- **GET /regime/{symbol}** - Get current market regime
  - Query params: `returns` (comma-separated), `bar_index`
  - Returns: Trend/volatility/correlation regime classification

##### Health Monitoring
- **GET /health** - Overall system health
  - Status (HEALTHY/DEGRADED/UNHEALTHY)
  - Uptime, validation counts, validity rate
  - Processing times, unhealthy symbols

- **GET /health/{symbol}** - Symbol-specific health
  - Last validation time
  - Consecutive failures
  - Validation metrics
  - Recent failure history

- **GET /failures** - Failure breakdown by type
  - Trend failures
  - Stationarity failures
  - Distribution failures
  - Correlation failures
  - Volatility failures

- **GET /recent** - Recent validation history
  - Query param: `limit` (default 20)
  - Returns last N validations with timestamps

##### Configuration
- **GET /config** - Current QSE configuration
  - Config hash for versioning
  - All 6 sub-config settings
  - Threshold values

- **POST /reset-health** - Reset health metrics (admin)

##### Root
- **GET /** - API info and endpoint list

### 2. Health Monitor (`health_monitor.py`)
Comprehensive health tracking system for QSE operations.

**Key Classes:**
- `ValidationMetrics` - Performance and quality metrics
- `SymbolHealth` - Per-symbol health tracking
- `QSEHealthMonitor` - Main monitoring orchestrator

**Tracked Metrics:**
- Total validations (valid/invalid breakdown)
- Failure types (5 categories)
- Processing times (avg/min/max)
- Quality metrics (trend score, ADF p-value, z-score)
- Consecutive failures per symbol
- Recent validation history (last 100)

**Health Status Levels:**
- **HEALTHY**: Validity rate ≥ 80%
- **DEGRADED**: Validity rate 50-80%
- **UNHEALTHY**: Validity rate < 50%

## Quick Start

### 1. Start API Server
```bash
# Activate environment
.\.venv\Scripts\activate

# Start API (port 8002)
python -m uvicorn arbitrex.quant_stats.api:app --host 0.0.0.0 --port 8002 --reload
```

### 2. Test Health Monitor
```bash
python test_qse_api.py
```

### 3. Test API Endpoints
```bash
# Get health status
curl http://localhost:8002/health

# Get config
curl http://localhost:8002/config

# Validate signal
curl -X POST http://localhost:8002/validate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "1H",
    "returns": [0.001, -0.002, 0.003, 0.001],
    "bar_index": 100
  }'

# Get regime
curl "http://localhost:8002/regime/EURUSD?returns=0.001,-0.002,0.003,0.001&bar_index=100"
```

## Integration Example

```python
from arbitrex.quant_stats import QuantitativeStatisticsEngine, QSEHealthMonitor
import pandas as pd

# Initialize
qse = QuantitativeStatisticsEngine()
health = QSEHealthMonitor()

# Process signal
symbol = "EURUSD"
returns = pd.Series([...])  # Your returns data

# Record start
start_time = health.record_validation_start(symbol)

# Validate
output = qse.process_bar(symbol, returns, bar_index=100)

# Record result
if output.validation.signal_validity_flag:
    health.record_validation_success(symbol, start_time, metrics_dict)
    # Proceed to ML prediction
else:
    health.record_validation_failure(symbol, start_time, 
                                     output.validation.failure_reasons,
                                     metrics_dict)
    # Suppress signal

# Check health
status = health.get_health_status()
print(f"System Status: {status['status']}")
print(f"Validity Rate: {status['validity_rate']:.1%}")
```

## API Response Examples

### Validate Response (Valid Signal)
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "1H",
  "signal_valid": true,
  "trend_persistence_score": 0.28,
  "adf_stationary": true,
  "adf_pvalue": 0.03,
  "z_score": -1.52,
  "is_outlier": false,
  "volatility_regime": "NORMAL",
  "volatility_percentile": 45.2,
  "autocorr_check": true,
  "stationarity_check": true,
  "distribution_check": true,
  "correlation_check": true,
  "volatility_check": true,
  "trend_regime": "TRENDING",
  "market_phase": "ACTIVE",
  "regime_stable": true,
  "failure_reasons": [],
  "processing_time_ms": 12.5,
  "config_hash": "e21b510e65e706e6"
}
```

### Health Status Response
```json
{
  "status": "HEALTHY",
  "uptime_seconds": 3600.5,
  "total_validations": 150,
  "valid_signals": 120,
  "invalid_signals": 30,
  "validity_rate": 0.80,
  "avg_processing_time_ms": 11.3,
  "symbols_tracked": 3,
  "unhealthy_symbols": 0
}
```

## Performance Characteristics

- **Processing Time**: 10-15ms per validation
- **Memory**: ~50MB base + 1MB per 1000 validations tracked
- **Throughput**: ~100 validations/second (single instance)
- **History Retention**: Last 100 validations in memory

## Health Export

Export comprehensive health report:
```python
health.export_health_report('health_report.json')
```

Report includes:
- Overall health status
- All symbol health states
- Failure breakdown
- Recent validation history (last 50)

## Monitoring Dashboard Integration

Health metrics ready for Grafana/Prometheus:
- Validity rates per symbol
- Processing time percentiles
- Failure type distribution
- Consecutive failure alerts
- System uptime

## Best Practices

1. **Monitor Validity Rates**: Alert if < 50%
2. **Track Consecutive Failures**: Alert if ≥ 5 for any symbol
3. **Watch Processing Times**: Alert if > 50ms
4. **Export Regular Reports**: Hourly/daily health exports
5. **Reset Metrics Carefully**: Only for testing/maintenance

## Next Steps

1. Run test: `python test_qse_api.py`
2. Start API: `uvicorn arbitrex.quant_stats.api:app --port 8002`
3. Integrate with ML pipeline (use `/validate` endpoint)
4. Set up monitoring dashboard (consume health endpoints)
5. Configure alerts for unhealthy states

## Dependencies

- FastAPI
- Pydantic
- pandas
- numpy
- uvicorn (for running server)

All already included in requirements.txt ✓
