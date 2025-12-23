# Signal Engine - Setup and Installation Guide

## Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- All system dependencies installed

## Installation Steps

### 1. Install Dependencies

Ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

Key dependencies for Signal Engine:
- `numpy` - Numerical computations
- `pandas` - Data handling (upstream layer dependency)
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `pytest` - Testing framework

### 2. Verify Installation

Run the validation script:

```bash
python validate_signal_engine.py
```

Expected output:
```
================================================================================
SIGNAL ENGINE INTEGRATION VALIDATION
================================================================================

Testing imports...
✓ SignalGenerationEngine imported
✓ SignalEngineConfig imported
✓ Schemas imported
✓ Filters imported
✓ SignalStateManager imported

Testing upstream layer integration...
✓ Feature Engine schemas accessible
✓ Quant Stats Engine schemas accessible
✓ ML Layer schemas accessible

Testing engine initialization...
✓ Engine initialized with default config
  Config hash: a3f9c21b
✓ Engine initialized with custom config
  Config hash: d4e8a7f2
✓ Config hashing working (different configs → different hashes)

Testing basic bar processing...
✓ Bar processed successfully
  Timestamp: 2025-12-23 10:00:00+00:00
  Symbol: EURUSD
  Trade allowed: True
  Direction: LONG
  Confidence: 0.670
  State: VALID_SIGNAL
  Processing time: 1.23ms

Testing health metrics...
✓ Health metrics retrieved
  Total bars processed: 1
  Signals generated: 1
  Active signals: 1

Testing API availability...
✓ API module importable
  Title: Signal Generation Engine API
  Version: 1.0.0
  Start with: python start_signal_api.py

================================================================================
VALIDATION SUMMARY
================================================================================
✓ PASS: Imports
✓ PASS: Upstream Integration
✓ PASS: Engine Initialization
✓ PASS: Basic Processing
✓ PASS: Health Metrics
✓ PASS: API Availability

Results: 6/6 tests passed

🎉 ALL VALIDATION TESTS PASSED!

Signal Engine is ready for integration:
  1. Start API: python start_signal_api.py
  2. Run tests: pytest test_signal_engine.py -v
  3. Run demo: python demo_signal_engine.py
  4. Check docs: SIGNAL_ENGINE.md
```

### 3. Run Tests

Execute the comprehensive test suite:

```bash
pytest test_signal_engine.py -v
```

Expected: 21 tests pass.

### 4. Run Demo

See the engine in action:

```bash
python demo_signal_engine.py
```

This demonstrates:
- Valid signal generation
- Gate filtering (regime, quant stats, ML)
- Signal suppression scenarios
- State management
- Health metrics

## Starting the Signal Engine API

### Development Mode

```bash
python start_signal_api.py
```

The API will start on `http://127.0.0.1:8004`

### Access Points

- **Swagger UI**: http://127.0.0.1:8004/docs
- **ReDoc**: http://127.0.0.1:8004/redoc
- **Health Check**: http://127.0.0.1:8004/health

### Test API

```bash
# Health check
curl http://localhost:8004/health

# Get configuration
curl http://localhost:8004/config

# Get all states
curl http://localhost:8004/state/all
```

## Integration with System Stack

### Port Assignments

The complete system uses these ports:
- Raw Layer API: **8000**
- Clean Data API: **8001**
- Feature Engine API: **8002**
- Quant Stats Engine API: **8003**
- **Signal Engine API: 8004** ← This service
- ML Layer API: **8005**

### Starting Full Stack

To start all services (including Signal Engine):

```powershell
# Start all services
.\START_STACK.ps1
```

Or start individually:

```bash
# Terminal 1: Raw Layer
python start_raw_api.py

# Terminal 2: Clean Data Layer
python start_clean_api.py

# Terminal 3: Feature Engine
python start_feature_api.py

# Terminal 4: Quant Stats Engine
python start_qse_api.py

# Terminal 5: Signal Engine
python start_signal_api.py

# Terminal 6: ML Layer
python start_ml_api.py
```

## Configuration

### Default Configuration

The Signal Engine uses these default thresholds (can be customized):

```python
from arbitrex.signal_engine.config import SignalEngineConfig

config = SignalEngineConfig()

# Regime Gate
config.regime_gate.allowed_regimes = ["TRENDING"]
config.regime_gate.min_regime_confidence = 0.6

# Quant Stats Gate
config.quant_gate.min_trend_consistency = 0.5
config.quant_gate.min_volatility_percentile = 20.0
config.quant_gate.max_volatility_percentile = 80.0
config.quant_gate.max_cross_correlation = 0.85

# ML Gate
config.ml_gate.entry_threshold = 0.55
config.ml_gate.min_confidence_level = "MEDIUM"

# Confidence Scoring
config.confidence_score.ml_confidence_weight = 0.5
config.confidence_score.trend_consistency_weight = 0.3
config.confidence_score.regime_weight_contribution = 0.2

# State Management
config.state_management.min_bars_between_signals = 5
config.state_management.allow_reversal = True
```

### Custom Configuration

```python
from arbitrex.signal_engine import SignalGenerationEngine, SignalEngineConfig

# Create custom config
config = SignalEngineConfig()
config.quant_gate.min_trend_consistency = 0.7  # More conservative

# Initialize engine with custom config
engine = SignalGenerationEngine(config)
```

## Usage Examples

### Python Integration

```python
from arbitrex.signal_engine import SignalGenerationEngine

# Initialize
engine = SignalGenerationEngine()

# Process bar (receives inputs from upstream layers)
output = engine.process_bar(
    feature_vector=fv,   # From Feature Engine
    qse_output=qse,      # From Quant Stats Engine
    ml_output=ml,        # From ML Layer
    bar_index=42
)

# Check decision
if output.decision.trade_allowed:
    intent = output.decision.trade_intent
    print(f"Trade: {intent.direction.name}")
    print(f"Confidence: {intent.confidence_score:.3f}")
    # Send to Risk Manager for position sizing
else:
    reasons = ', '.join(output.decision.suppression_reasons)
    print(f"Signal suppressed: {reasons}")

# Get health metrics
health = engine.get_health()
print(f"Signal generation rate: {health.signal_generation_rate:.2%}")
```

### REST API Integration

```python
import requests

# Process bar
response = requests.post('http://localhost:8004/process', json={
    'feature_vector': {...},
    'qse_output': {...},
    'ml_output': {...},
    'bar_index': 42
})

result = response.json()

if result['decision']['trade_allowed']:
    intent = result['decision']['trade_intent']
    print(f"Direction: {intent['direction']}")
    print(f"Confidence: {intent['confidence_score']}")
```

## Monitoring

### Health Metrics

```bash
# Get health status
curl http://localhost:8004/health
```

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-23T10:00:00Z",
  "engine_health": {
    "total_bars_processed": 1000,
    "signals_generated": 150,
    "signals_suppressed": 850,
    "signal_generation_rate": 0.15,
    "regime_gate_pass_rate": 0.40,
    "quant_gate_pass_rate": 0.60,
    "ml_gate_pass_rate": 0.75,
    "suppression_by_regime": 600,
    "suppression_by_quant": 200,
    "suppression_by_ml": 50,
    "long_signals": 80,
    "short_signals": 70,
    "avg_confidence_score": 0.67,
    "active_signals": 3,
    "avg_processing_time_ms": 1.2
  }
}
```

### State Monitoring

```bash
# Get all active signals
curl http://localhost:8004/state/active

# Get specific symbol state
curl http://localhost:8004/state/EURUSD/H1

# Get all states summary
curl http://localhost:8004/state/all
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Verify installation
python -c "import arbitrex.signal_engine; print('OK')"
```

### API Won't Start

Check port availability:

```bash
# Windows
netstat -ano | findstr :8004

# If port in use, kill process or change port in api.py
```

### Processing Errors

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs in:
- `logs/signal_engine_api.log`

## File Locations

```
arbitrex/signal_engine/
├── __init__.py          # Module exports
├── engine.py            # Core engine
├── filters.py           # Gate filters
├── state_manager.py     # State management
├── config.py            # Configuration
├── schemas.py           # Data structures
├── api.py               # REST API
└── README.md            # Module docs

Root level:
├── test_signal_engine.py              # Tests
├── demo_signal_engine.py              # Demo
├── validate_signal_engine.py          # Validation
├── start_signal_api.py                # API starter
├── SIGNAL_ENGINE.md                   # Full docs
├── SIGNAL_ENGINE_QUICK_REF.md         # Quick ref
├── SIGNAL_ENGINE_IMPLEMENTATION.md    # Implementation summary
└── SIGNAL_ENGINE_SETUP.md             # This file
```

## Next Steps

After successful setup:

1. ✓ **Validate installation** - `python validate_signal_engine.py`
2. ✓ **Run tests** - `pytest test_signal_engine.py -v`
3. ✓ **Run demo** - `python demo_signal_engine.py`
4. **Start API** - `python start_signal_api.py`
5. **Integrate with system** - Connect to Feature Engine, QSE, ML Layer
6. **Monitor health** - Check `/health` endpoint regularly

## Support

For issues or questions:
- Review documentation: [SIGNAL_ENGINE.md](SIGNAL_ENGINE.md)
- Check test examples: [test_signal_engine.py](test_signal_engine.py)
- Run demo: [demo_signal_engine.py](demo_signal_engine.py)
- Review quick reference: [SIGNAL_ENGINE_QUICK_REF.md](SIGNAL_ENGINE_QUICK_REF.md)

---

**Signal Engine v1.0.0 - Production Ready**
