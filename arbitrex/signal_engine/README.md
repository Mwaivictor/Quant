# Signal Generation Engine

**Conservative decision layer for systematic FX trading.**

## Purpose

Converts quantitative validation and ML confidence into actionable **trade intents**.

The Signal Engine decides **WHETHER** a trade should exist, **NOT** how it executes.

## Core Responsibilities

1. **Regime Gate Enforcement** - Only allow TRENDING regimes
2. **Statistical Validation** - Verify market structure supports signal
3. **ML Confidence Filtering** - Confirm historical edge exists
4. **Direction Assignment** - Deterministic from momentum features
5. **Confidence Scoring** - Weighted combination for position sizing
6. **State Management** - Single active signal per symbol
7. **Trade Intent Emission** - Pure data objects (no execution params)

## Architecture

### Three-Gate System

```
┌─────────────────────┐
│   Feature Engine    │
│   Quant Stats Eng   │
│     ML Layer        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   REGIME GATE       │  ← Only TRENDING allowed
├─────────────────────┤
│   QUANT STATS GATE  │  ← Statistical robustness
├─────────────────────┤
│   ML CONFIDENCE     │  ← P(momentum_success) ≥ 0.55
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Trade Intent      │
│   direction: LONG   │
│   confidence: 0.67  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Risk Manager      │
└─────────────────────┘
```

### Module Structure

```
arbitrex/signal_engine/
│
├── engine.py           # Core Signal Generation Engine
│   └─ SignalGenerationEngine
│       ├─ process_bar()          # Main entry point
│       ├─ _generate_decision()   # Apply gates
│       ├─ _assign_direction()    # Deterministic direction
│       └─ _compute_confidence()  # Weighted score
│
├── filters.py          # Filtering Gates
│   ├─ RegimeGate               # Gate 1
│   ├─ QuantStatsGate           # Gate 2
│   └─ MLConfidenceGate         # Gate 3
│
├── state_manager.py    # State Management
│   └─ SignalStateManager
│       ├─ State transitions
│       ├─ Cooldown enforcement
│       └─ Exit condition checking
│
├── config.py           # Configuration
│   └─ SignalEngineConfig
│       ├─ Gate thresholds
│       ├─ Confidence weights
│       └─ State management rules
│
├── schemas.py          # Data Structures
│   ├─ TradeIntent
│   ├─ SignalDecision
│   ├─ SignalState
│   └─ SignalEngineHealth
│
└── api.py             # REST API
    └─ FastAPI endpoints
```

## Key Features

### 1. Conservative Filtering

**When in doubt, do nothing.**

Signals are suppressed unless:
- ✓ Market is TRENDING (regime stable)
- ✓ Statistics validate robustness
- ✓ ML confirms historical edge

### 2. State Machine

Prevents duplicate signals:

```
NO_TRADE → VALID_SIGNAL → ACTIVE_TRADE → NO_TRADE
```

Enforces:
- Single active signal per symbol
- Minimum cooldown between signals
- Clean exit handling

### 3. Confidence Scoring

Weighted combination:
```python
confidence = (
    ML_confidence * 0.5 +
    trend_consistency * 0.3 +
    regime_weight * 0.2
)
```

Used downstream for position sizing.

### 4. Full Auditability

Every decision includes:
- Gate-by-gate results
- Suppression reasons
- Intermediate scores
- Configuration version
- Processing metadata

### 5. Deterministic & Reproducible

- Same inputs → same outputs
- No randomness
- Config hash versioning
- Bar-close only (no intra-bar)

## Usage

### Python API

```python
from arbitrex.signal_engine import SignalGenerationEngine

# Initialize
engine = SignalGenerationEngine()

# Process bar
output = engine.process_bar(
    feature_vector,  # From Feature Engine
    qse_output,      # From Quant Stats Engine
    ml_output,       # From ML Layer
    bar_index=42
)

# Check result
if output.decision.trade_allowed:
    intent = output.decision.trade_intent
    print(f"Signal: {intent.direction.name}")
    print(f"Confidence: {intent.confidence_score:.3f}")
    # Send to Risk Manager for position sizing
else:
    print(f"Suppressed: {output.decision.suppression_reasons}")
```

### REST API

```bash
# Start server
python start_signal_api.py

# Process bar
curl -X POST http://localhost:8004/process \
  -H "Content-Type: application/json" \
  -d '{"feature_vector": {...}, "qse_output": {...}, "ml_output": {...}, "bar_index": 0}'

# Health check
curl http://localhost:8004/health

# Get state
curl http://localhost:8004/state/EURUSD/H1
```

## Configuration

Key thresholds:

```python
# Regime Gate
allowed_regimes = ["TRENDING"]
min_regime_confidence = 0.6

# Quant Stats Gate
min_trend_consistency = 0.5
vol_percentile_range = (20.0, 80.0)
max_cross_correlation = 0.85

# ML Gate
entry_threshold = 0.55  # P(momentum_success)
min_confidence_level = "MEDIUM"

# State Management
min_bars_between_signals = 5
allow_reversal = True
exit_on_regime_change = True
```

## Output Contract

### Trade Intent (Pure Data Object)

```python
{
  "timestamp": "2025-12-23T10:00:00Z",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "direction": 1,           # LONG=1, SHORT=-1
  "confidence_score": 0.67, # 0-1 for position sizing
  "signal_source": "momentum_v1",
  "signal_version": "a3f9c21b",
  "bar_index": 42
}
```

**No execution parameters:**
- ❌ No position size
- ❌ No stop loss / take profit
- ❌ No order type
- ❌ No slippage / commission

These are handled by Risk Manager.

## Testing

```bash
# Run tests
pytest test_signal_engine.py -v

# Run demo
python demo_signal_engine.py
```

Test coverage:
- ✓ Regime gate filtering
- ✓ Quant stats validation
- ✓ ML confidence filtering
- ✓ Direction assignment
- ✓ Confidence computation
- ✓ State management
- ✓ Cooldown enforcement
- ✓ Input validation
- ✓ Health tracking

## Health Monitoring

Tracked metrics:
- Signal generation rate
- Gate pass rates
- Suppression breakdown
- Direction distribution
- Confidence statistics
- Processing performance

Access via:
```python
health = engine.get_health()
print(health.to_dict())
```

Or API:
```bash
curl http://localhost:8004/health
```

## Design Principles

1. **Conservative** - Suppress aggressively, only allow high-quality signals
2. **Deterministic** - No randomness, fully reproducible
3. **Auditable** - Complete decision trail with reasons
4. **Causal** - No future leakage, bar-close only
5. **Modular** - Clean separation, independently testable
6. **Transparent** - Every decision explainable

## Integration

### Upstream Dependencies
- Feature Engine (feature vectors)
- Quant Stats Engine (statistical validation)
- ML Layer (regime + confidence predictions)

### Downstream Consumers
- Risk Manager (position sizing)
- Execution Engine (via Risk Manager)

### Data Flow
```
Feature Engine ─┐
                ├─→ Signal Engine ─→ Risk Manager ─→ Execution
Quant Stats ────┤
                │
ML Layer ───────┘
```

## Files

- `engine.py` - Core engine implementation
- `filters.py` - Three filtering gates
- `state_manager.py` - State machine and transitions
- `config.py` - Configuration and thresholds
- `schemas.py` - Data structures and types
- `api.py` - REST API endpoints

## Version

**v1.0.0** - Production release

## Documentation

- [SIGNAL_ENGINE.md](../../SIGNAL_ENGINE.md) - Full documentation
- [SIGNAL_ENGINE_QUICK_REF.md](../../SIGNAL_ENGINE_QUICK_REF.md) - Quick reference
- [test_signal_engine.py](../../test_signal_engine.py) - Test suite
- [demo_signal_engine.py](../../demo_signal_engine.py) - Demo script

---

**Built with quantitative discipline and software excellence.**
