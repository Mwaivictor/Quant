# Signal Generation Engine

**Production-Grade Signal Generation Engine for Systematic FX Trading**

## Overview

The Signal Generation Engine is a conservative decision layer that converts quantitative validation and ML confidence into actionable trade intents. It implements strict filtering gates to ensure only high-quality signals reach the risk management layer.

## Core Principle

> **The Signal Engine decides WHETHER a trade should exist, NOT how it executes.**

The engine produces **trade intents**, not orders. Trade intents contain only:
- Direction (LONG/SHORT)
- Confidence score (for position sizing)
- Metadata (timestamp, symbol, signal source)

**No execution parameters** - sizing, stops, and execution are handled downstream by the Risk Manager.

## Architecture

### System Flow
```
Feature Engine → Quant Stats Engine → ML Layer → Signal Engine → Risk Manager → Execution
```

### Core Components

1. **Signal Generation Engine** ([engine.py](arbitrex/signal_engine/engine.py))
   - Orchestrates the signal generation pipeline
   - Applies filtering gates in strict order
   - Manages state transitions
   - Tracks health metrics

2. **Filtering Gates** ([filters.py](arbitrex/signal_engine/filters.py))
   - **Regime Gate**: Only allows TRENDING regimes
   - **Quant Stats Gate**: Validates statistical robustness
   - **ML Confidence Gate**: Filters by momentum success probability

3. **State Manager** ([state_manager.py](arbitrex/signal_engine/state_manager.py))
   - Ensures single active signal per symbol
   - Prevents duplicate entries
   - Manages cooldown periods
   - Handles exit conditions

4. **Configuration** ([config.py](arbitrex/signal_engine/config.py))
   - All thresholds and decision parameters
   - Versioned and hashable for reproducibility

5. **Schemas** ([schemas.py](arbitrex/signal_engine/schemas.py))
   - Trade intents
   - Signal decisions
   - State records
   - Health metrics

## Decision Logic

### Gate Execution Order (Strict)

#### 1. Regime Gate
```python
if regime != TRENDING:
    return NO_TRADE
```
- Only TRENDING regimes allowed
- Requires regime stability (no recent changes)
- Minimum regime confidence threshold

#### 2. Quantitative Statistics Gate
```python
if not signal_validity_flag:
    return NO_TRADE

if trend_consistency < TREND_THRESHOLD:
    return NO_TRADE

if vol_percentile not in ALLOWED_VOL_RANGE:
    return NO_TRADE

if cross_corr > MAX_CORR_THRESHOLD:
    return NO_TRADE
```
- Primary signal validity flag must be True
- Trend consistency ≥ 0.5
- Volatility in acceptable range (20-80 percentile)
- Cross-correlation < 0.85 (prevents crowded trades)
- Distribution must be stable
- Autocorrelation and stationarity checks must pass

#### 3. ML Confidence Gate
```python
if P_momentum_success < ENTRY_THRESHOLD:
    return NO_TRADE
```
- Momentum success probability ≥ 0.55
- Confidence level ≥ MEDIUM
- ML layer must allow trade (combines regime + signal)

#### 4. Direction Assignment
```python
direction = sign(momentum_signal)
```
- Deterministic from feature vector
- NO PREDICTION - purely rule-based
- Typically derived from trend regime or momentum features

#### 5. Confidence Score Computation
```python
confidence = (
    P_momentum_success * 0.5 +
    trend_consistency * 0.3 +
    regime_weight * 0.2
)
```
- Weighted combination of:
  - ML confidence (50%)
  - Trend consistency (30%)
  - Regime quality (20%)
- Used downstream for position sizing

## State Machine

The engine implements a simple state machine to prevent duplicate signals:

```
NO_TRADE → VALID_SIGNAL → ACTIVE_TRADE → NO_TRADE
              ↓               ↓
          NO_TRADE       NO_TRADE
```

### State Transitions

- **NO_TRADE**: No active signal
- **VALID_SIGNAL**: Signal generated, awaiting confirmation
- **ACTIVE_TRADE**: Signal confirmed and active
- **EXITED**: Signal closed

### State Guarantees

1. **Single active signal per symbol** - No duplicate positions
2. **Cooldown period** - Minimum bars between signals (default: 5)
3. **Reversal control** - Configurable LONG ↔ SHORT transitions
4. **Exit conditions**:
   - Regime change (TRENDING → RANGING/STRESSED)
   - Quant stats failure
   - ML exit signal
   - Opposite direction signal

## Configuration

Default configuration ([config.py](arbitrex/signal_engine/config.py)):

```python
# Regime Gate
allowed_regimes = ["TRENDING"]
require_stable_regime = True
min_regime_confidence = 0.6

# Quant Stats Gate
min_trend_consistency = 0.5
allowed_volatility_regimes = ["NORMAL", "LOW"]
min_volatility_percentile = 20.0
max_volatility_percentile = 80.0
max_cross_correlation = 0.85

# ML Gate
entry_threshold = 0.55  # P(momentum_success)
exit_threshold = 0.45
min_confidence_level = "MEDIUM"

# Confidence Score
ml_confidence_weight = 0.5
trend_consistency_weight = 0.3
regime_weight_contribution = 0.2

# State Management
min_bars_between_signals = 5
allow_reversal = True
exit_on_regime_change = True
exit_on_quant_failure = True
```

## API Endpoints

The Signal Engine provides a REST API ([api.py](arbitrex/signal_engine/api.py)):

### Health Check
```bash
GET /health
```
Returns engine health metrics and status.

### Configuration
```bash
GET /config
```
Returns current engine configuration.

### Process Bar
```bash
POST /process
```
Main endpoint for signal generation. Accepts:
- Feature vector from Feature Engine
- Quant stats from QSE
- ML predictions from ML Layer
- Bar index

Returns:
- Signal decision (trade intent or suppression reasons)
- Current state
- Processing metadata

### State Management
```bash
GET /state/{symbol}/{timeframe}
GET /state/all
GET /state/active
POST /state/reset/{symbol}/{timeframe}
POST /reset
```

## Usage

### Python API

```python
from arbitrex.signal_engine import SignalGenerationEngine, SignalEngineConfig

# Initialize engine
config = SignalEngineConfig()
engine = SignalGenerationEngine(config)

# Process bar
output = engine.process_bar(
    feature_vector=fv,
    qse_output=qse,
    ml_output=ml,
    bar_index=0
)

# Check decision
if output.decision.trade_allowed:
    intent = output.decision.trade_intent
    print(f"Trade: {intent.direction.name} @ confidence {intent.confidence_score:.3f}")
else:
    print(f"Suppressed: {output.decision.suppression_reasons}")

# Get health metrics
health = engine.get_health()
print(f"Signal generation rate: {health.signal_generation_rate:.2%}")
```

### REST API

Start the server:
```bash
python -m arbitrex.signal_engine.api
```

Process a bar:
```bash
curl -X POST http://localhost:8004/process \
  -H "Content-Type: application/json" \
  -d '{
    "feature_vector": {...},
    "qse_output": {...},
    "ml_output": {...},
    "bar_index": 0
  }'
```

## Testing

Comprehensive test suite ([test_signal_engine.py](test_signal_engine.py)):

```bash
pytest test_signal_engine.py -v
```

Test coverage:
- ✓ Regime gate filtering
- ✓ Quant stats gate filtering
- ✓ ML confidence gate filtering
- ✓ Direction assignment
- ✓ Confidence score computation
- ✓ State management
- ✓ Cooldown periods
- ✓ Input validation
- ✓ Health tracking
- ✓ End-to-end integration

## Demo

Run the demonstration:
```bash
python demo_signal_engine.py
```

The demo shows:
- Valid signal generation (all gates pass)
- Regime suppression (ranging market)
- Quant stats suppression (weak trend)
- ML suppression (low confidence)
- Cooldown enforcement
- Health metrics tracking

## Design Principles

### 1. Conservative by Default
> **When in doubt, do nothing.**

The engine suppresses signals aggressively. Only when:
- Market structure is favorable (regime)
- Statistics justify continuation (quant stats)
- ML confirms historical edge (confidence)

...does a signal pass.

### 2. Fully Deterministic
- Same inputs → same outputs
- No randomness
- No inference or prediction (direction is deterministic)
- Fully reproducible with config hash

### 3. Fully Auditable
Every decision includes:
- Gate-by-gate results
- Suppression reasons
- Intermediate scores
- State history
- Configuration version

### 4. Causal and Bar-Close Only
- No intra-bar logic
- No future leakage
- All inputs from bar-close data
- Signals generated only at bar close

### 5. Modular and Testable
- Each gate independently testable
- Clean separation of concerns
- No dependencies on downstream layers
- Configuration-driven thresholds

## Output Contract

### Trade Intent Object
```python
{
  "timestamp": "2025-01-07T06:00:00Z",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "direction": 1,  # LONG=1, SHORT=-1
  "confidence_score": 0.67,
  "signal_source": "momentum_v1",
  "signal_version": "a3f9c21b",
  "bar_index": 42
}
```

**No order fields**:
- ❌ No position size
- ❌ No stop loss
- ❌ No take profit
- ❌ No execution parameters

These are handled by the Risk Manager downstream.

## Integration Points

### Upstream Dependencies
- **Feature Engine**: Provides feature vectors
- **Quant Stats Engine**: Provides statistical validation
- **ML Layer**: Provides regime and confidence predictions

### Downstream Consumers
- **Risk Manager**: Consumes trade intents for position sizing
- **Execution Engine**: Receives sized orders from Risk Manager

### Data Flow
```
[Feature Engine] ─┐
                  ├─→ [Signal Engine] ─→ [Risk Manager] ─→ [Execution]
[Quant Stats] ────┤
                  │
[ML Layer] ───────┘
```

## Health Monitoring

The engine tracks comprehensive health metrics:

### Signal Statistics
- Total bars processed
- Signals generated vs suppressed
- Signal generation rate
- Gate pass rates

### Suppression Analysis
- Suppression by regime
- Suppression by quant stats
- Suppression by ML confidence

### Direction Distribution
- Long vs short signal counts
- Direction bias detection

### Confidence Statistics
- Average confidence score
- Min/max confidence range
- Confidence distribution

### Performance Metrics
- Average processing time
- Active signal count
- Last signal timestamp

Access via API:
```bash
GET /health
```

Or programmatically:
```python
health = engine.get_health()
print(health.to_dict())
```

## File Structure

```
arbitrex/signal_engine/
├── __init__.py          # Module exports
├── engine.py            # Core Signal Generation Engine
├── filters.py           # Filtering gates (Regime, Quant, ML)
├── state_manager.py     # Signal state management
├── config.py            # Configuration and thresholds
├── schemas.py           # Data structures and types
└── api.py               # FastAPI REST interface

test_signal_engine.py    # Comprehensive test suite
demo_signal_engine.py    # Demonstration script
SIGNAL_ENGINE.md         # This documentation
```

## Version History

- **v1.0.0** (2025-12-23): Initial production release
  - Three-gate filtering system
  - State management with cooldown
  - Confidence scoring
  - REST API
  - Comprehensive testing

## Future Enhancements

Potential improvements:
1. **Multi-timeframe signals**: Coordinate signals across timeframes
2. **Portfolio-level filtering**: Account for cross-pair exposure
3. **Adaptive thresholds**: Machine-learned gate parameters
4. **Signal strength tiers**: Multiple confidence bands
5. **Advanced exit logic**: Trailing stops, profit targets

## Support

For issues, questions, or contributions:
- Review test suite for usage examples
- Run demo script to see engine in action
- Check health metrics for debugging
- Examine decision audit trail for transparency

---

**Built with quantitative discipline and software excellence.**
