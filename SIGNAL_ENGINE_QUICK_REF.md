# Signal Engine Quick Reference

## Starting the API

```bash
python start_signal_api.py
```

Server runs on: `http://127.0.0.1:8004`

## Quick Test

```bash
# Health check
curl http://localhost:8004/health

# Get configuration
curl http://localhost:8004/config

# Get all states
curl http://localhost:8004/state/all
```

## Integration Example

```python
from arbitrex.signal_engine import SignalGenerationEngine

# Initialize
engine = SignalGenerationEngine()

# Process bar (receives inputs from Feature Engine, QSE, ML Layer)
output = engine.process_bar(feature_vector, qse_output, ml_output, bar_index)

# Check result
if output.decision.trade_allowed:
    intent = output.decision.trade_intent
    print(f"{intent.direction.name}: {intent.confidence_score:.3f}")
```

## Gate Thresholds

| Gate | Threshold | Description |
|------|-----------|-------------|
| Regime | TRENDING only | Blocks RANGING/STRESSED |
| Trend Consistency | ≥ 0.5 | Quant stats validation |
| Volatility Percentile | 20-80 | Acceptable vol range |
| Cross-Correlation | < 0.85 | Prevents crowding |
| ML Confidence | ≥ 0.55 | P(momentum success) |

## State Transitions

```
NO_TRADE → VALID_SIGNAL → ACTIVE_TRADE → NO_TRADE
```

Cooldown: 5 bars between signals (default)

## Testing

```bash
# Run tests
pytest test_signal_engine.py -v

# Run demo
python demo_signal_engine.py
```

## Key Outputs

### Trade Intent
- `direction`: LONG (1) or SHORT (-1)
- `confidence_score`: 0-1 (for position sizing)
- `signal_source`: Strategy identifier
- `signal_version`: Config hash

### Signal Decision
- `trade_allowed`: bool
- `suppression_reasons`: List of blocking gates
- `gate_passed`: Individual gate results

### Health Metrics
- `signal_generation_rate`: % of bars with signals
- `gate_pass_rates`: % passing each gate
- `suppression_breakdown`: Count by gate
- `confidence_stats`: avg/min/max

## File Locations

- Engine: `arbitrex/signal_engine/engine.py`
- Config: `arbitrex/signal_engine/config.py`
- API: `arbitrex/signal_engine/api.py`
- Tests: `test_signal_engine.py`
- Demo: `demo_signal_engine.py`
- Docs: `SIGNAL_ENGINE.md`

## Port Assignments

- Raw Layer: 8000
- Clean Data: 8001
- Feature Engine: 8002
- QSE: 8003
- **Signal Engine: 8004** ← This service
- ML Layer: 8005

## Next Steps

1. ✓ Start Signal Engine API
2. Ensure Feature Engine, QSE, and ML Layer are running
3. Send bar data through pipeline
4. Signal Engine receives validated inputs
5. Generates trade intents for Risk Manager

---

**Conservative signal generation with strict filtering gates.**
