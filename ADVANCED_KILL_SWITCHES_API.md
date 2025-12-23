# Advanced Kill Switches API Documentation
**Version**: 2.0.1 Enterprise Edition  
**Module**: `advanced_kill_switches.py`  
**API Endpoints**: 10 comprehensive endpoints

---

## Overview

Advanced Kill Switches provide **institutional-grade circuit breakers** that detect:
- **Rejection Velocity**: Strategy degradation or market breakdown
- **Exposure Velocity**: Runaway risk accumulation
- **Leverage Acceleration**: Uncontrolled leverage growth
- **Per-Strategy Failures**: Localized strategy issues

These are **NON-BYPASSABLE** controls with 4 severity levels:
- `WARNING` - Monitor and alert
- `THROTTLE` - Reduce position size
- `HALT` - Stop new trades, maintain positions
- `LIQUIDATE` - Close all positions immediately

---

## 🆕 New API Endpoints (10 Total)

### 1. Get Advanced Kill Switches Status
**Endpoint**: `GET /advanced_kill_switches/status`  
**Purpose**: Comprehensive status of all advanced kill switches

**Response**:
```json
{
  "advanced_kill_switches": {
    "rejection_velocity": {
      "is_active": false,
      "rejection_count": 3,
      "time_window_minutes": 5,
      "max_rejections": 10,
      "threshold_pct": 30.0
    },
    "exposure_velocity": {
      "is_active": false,
      "current_exposure_rate": 0.05,
      "max_exposure_rate_pct_per_min": 10.0,
      "snapshots_count": 15
    },
    "per_strategy": [
      {
        "strategy_id": "momentum_strategy",
        "is_enabled": true,
        "total_rejections": 2,
        "consecutive_rejections": 0
      }
    ],
    "recent_events": [...],
    "total_events": 5
  },
  "timestamp": "2025-12-23T11:45:00"
}
```

---

### 2. Record Rejection Event
**Endpoint**: `POST /advanced_kill_switches/rejection/record`  
**Purpose**: Record trade rejection for velocity tracking

**Request Body**:
```json
{
  "symbol": "EURUSD",
  "reason": "MAX_DRAWDOWN_EXCEEDED",
  "strategy_id": "momentum_strategy",
  "regime": "STRESSED"
}
```

**Response**:
```json
{
  "status": "rejection_recorded",
  "symbol": "EURUSD",
  "kill_switch_triggered": false,
  "event": null
}
```

**Triggered Response** (when threshold exceeded):
```json
{
  "status": "rejection_recorded",
  "symbol": "EURUSD",
  "kill_switch_triggered": true,
  "event": {
    "switch_type": "REJECTION_VELOCITY",
    "severity": "HALT",
    "reason": "10 rejections in 5 minutes (STRESSED regime)",
    "metrics": {
      "rejection_count": 10,
      "time_window_minutes": 5,
      "threshold_pct": 100.0
    },
    "timestamp": "2025-12-23T11:45:00",
    "strategy_id": "momentum_strategy"
  }
}
```

---

### 3. Get Rejection Velocity Stats
**Endpoint**: `GET /advanced_kill_switches/rejection/stats`  
**Purpose**: View rejection rate and velocity statistics

**Response**:
```json
{
  "is_active": false,
  "rejection_count": 3,
  "time_window_minutes": 5,
  "max_rejections": 10,
  "throttle_threshold": 6,
  "threshold_pct": 30.0,
  "last_trigger_time": null,
  "trigger_count": 0,
  "rejections_per_minute": 0.6
}
```

---

### 4. Record Exposure Snapshot
**Endpoint**: `POST /advanced_kill_switches/exposure/snapshot`  
**Purpose**: Record exposure snapshot for velocity tracking

**Request Body**:
```json
{
  "gross_exposure": 150000.0,
  "net_exposure": 50000.0,
  "leverage": 1.5,
  "num_positions": 8
}
```

**Response**:
```json
{
  "status": "snapshot_recorded",
  "kill_switch_triggered": false,
  "event": null
}
```

**Triggered Response** (when exposure velocity exceeded):
```json
{
  "status": "snapshot_recorded",
  "kill_switch_triggered": true,
  "event": {
    "switch_type": "EXPOSURE_VELOCITY",
    "severity": "THROTTLE",
    "reason": "Exposure increasing at 15.0% per minute (max: 10.0%)",
    "metrics": {
      "exposure_rate_pct_per_min": 15.0,
      "leverage_acceleration": 0.05,
      "current_gross_exposure": 150000.0
    },
    "timestamp": "2025-12-23T11:45:00"
  }
}
```

---

### 5. Get Exposure Velocity Stats
**Endpoint**: `GET /advanced_kill_switches/exposure/stats`  
**Purpose**: View exposure velocity and leverage acceleration

**Response**:
```json
{
  "is_active": false,
  "snapshots_count": 15,
  "time_window_minutes": 10,
  "max_exposure_rate_pct_per_min": 10.0,
  "current_exposure_rate_pct_per_min": 3.5,
  "leverage_acceleration": 0.02,
  "max_leverage_acceleration": 0.5,
  "last_snapshot": {
    "timestamp": "2025-12-23T11:44:00",
    "gross_exposure": 145000.0,
    "leverage": 1.45
  }
}
```

---

### 6. Control Strategy (Enable/Disable)
**Endpoint**: `POST /advanced_kill_switches/strategy/control`  
**Purpose**: Enable or disable a specific strategy

**Request Body (Disable)**:
```json
{
  "strategy_id": "momentum_strategy",
  "action": "disable",
  "reason": "Poor performance in current regime"
}
```

**Request Body (Enable)**:
```json
{
  "strategy_id": "momentum_strategy",
  "action": "enable"
}
```

**Response**:
```json
{
  "status": "strategy_updated",
  "strategy_id": "momentum_strategy",
  "action": "disable",
  "reason": "Poor performance in current regime",
  "is_enabled": false
}
```

---

### 7. Get Strategy Kill Switch Status
**Endpoint**: `GET /advanced_kill_switches/strategy/{strategy_id}/status`  
**Purpose**: Get detailed kill switch status for specific strategy

**Response**:
```json
{
  "strategy_id": "momentum_strategy",
  "is_enabled": true,
  "total_trades": 150,
  "total_rejections": 8,
  "rejection_rate": 0.053,
  "consecutive_rejections": 0,
  "max_consecutive_rejections": 5,
  "last_rejection_time": "2025-12-23T10:30:00",
  "disabled_since": null,
  "disable_reason": null
}
```

**Response (Disabled Strategy)**:
```json
{
  "strategy_id": "failing_strategy",
  "is_enabled": false,
  "total_trades": 50,
  "total_rejections": 12,
  "rejection_rate": 0.24,
  "consecutive_rejections": 5,
  "max_consecutive_rejections": 5,
  "last_rejection_time": "2025-12-23T11:40:00",
  "disabled_since": "2025-12-23T11:40:00",
  "disable_reason": "5 consecutive rejections exceeded threshold"
}
```

---

### 8. Get All Strategy Kill Switches
**Endpoint**: `GET /advanced_kill_switches/strategies/all`  
**Purpose**: Status for all tracked strategies

**Response**:
```json
{
  "strategies": [
    {
      "strategy_id": "momentum_strategy",
      "is_enabled": true,
      "total_trades": 150,
      "total_rejections": 8,
      "rejection_rate": 0.053
    },
    {
      "strategy_id": "mean_reversion_strategy",
      "is_enabled": true,
      "total_trades": 200,
      "total_rejections": 15,
      "rejection_rate": 0.075
    },
    {
      "strategy_id": "breakout_strategy",
      "is_enabled": false,
      "total_trades": 80,
      "total_rejections": 20,
      "rejection_rate": 0.25,
      "disabled_since": "2025-12-23T09:00:00"
    }
  ],
  "count": 3
}
```

---

### 9. Get Recent Kill Switch Events
**Endpoint**: `GET /advanced_kill_switches/events/recent?limit=50`  
**Purpose**: History of kill switch trigger events

**Response**:
```json
{
  "events": [
    {
      "switch_type": "REJECTION_VELOCITY",
      "severity": "THROTTLE",
      "reason": "6 rejections in 5 minutes (60% threshold)",
      "metrics": {...},
      "timestamp": "2025-12-23T11:30:00",
      "strategy_id": "momentum_strategy"
    },
    {
      "switch_type": "STRATEGY_FAILURE",
      "severity": "HALT",
      "reason": "5 consecutive rejections for breakout_strategy",
      "metrics": {...},
      "timestamp": "2025-12-23T09:00:00",
      "strategy_id": "breakout_strategy"
    }
  ],
  "count": 2,
  "total_events": 47
}
```

---

### 10. Check All Advanced Kill Switches
**Endpoint**: `POST /advanced_kill_switches/check?regime=STRESSED`  
**Purpose**: Manually trigger check of all advanced kill switches

**Response (All Clear)**:
```json
{
  "triggered_events": [],
  "count": 0,
  "all_clear": true
}
```

**Response (Triggered)**:
```json
{
  "triggered_events": [
    {
      "switch_type": "EXPOSURE_VELOCITY",
      "severity": "THROTTLE",
      "reason": "Exposure increasing at 12.5% per minute",
      "metrics": {
        "exposure_rate_pct_per_min": 12.5,
        "max_allowed": 10.0
      },
      "timestamp": "2025-12-23T11:45:00"
    }
  ],
  "count": 1,
  "all_clear": false
}
```

---

## 📊 Kill Switch Types & Thresholds

### Rejection Velocity Kill Switch

**Purpose**: Detects strategy degradation or market breakdown

**Configuration**:
- `max_rejections`: 10 rejections
- `time_window_minutes`: 5 minutes
- `throttle_threshold_pct`: 60% of max (6 rejections = THROTTLE)

**Regime Multipliers**:
- TRENDING: 1.0× (10 rejections allowed)
- RANGING: 1.0× (10 rejections)
- VOLATILE: 1.5× (15 rejections - more lenient)
- STRESSED: 0.5× (5 rejections - stricter)

**Severity Levels**:
- `WARNING`: 40% of threshold (4 rejections)
- `THROTTLE`: 60% of threshold (6 rejections)
- `HALT`: 100% of threshold (10 rejections)

---

### Exposure Velocity Kill Switch

**Purpose**: Detects runaway risk accumulation

**Configuration**:
- `max_exposure_rate_pct_per_min`: 10.0% per minute
- `max_leverage_acceleration`: 0.5 per minute
- `time_window_minutes`: 10 minutes

**Triggers**:
- Gross exposure increasing >10% per minute
- Leverage accelerating >0.5 per minute
- Net exposure spiking rapidly

**Severity Levels**:
- `WARNING`: Exposure rate 7-10% per minute
- `THROTTLE`: Exposure rate 10-15% per minute
- `HALT`: Exposure rate >15% per minute

---

### Per-Strategy Kill Switch

**Purpose**: Isolates failing strategies

**Configuration**:
- `max_consecutive_rejections`: 5
- `max_rejection_rate`: 20% (over 30 trades)

**Auto-Disable Triggers**:
- 5 consecutive rejections
- >20% rejection rate (with min 30 trades)

**Manual Override**: Can be re-enabled via API

---

## 🔄 Integration Flow

### 1. Record Rejections Automatically
```python
# After trade rejection
response = requests.post(
    'http://localhost:8005/advanced_kill_switches/rejection/record',
    json={
        'symbol': 'EURUSD',
        'reason': 'MAX_DRAWDOWN_EXCEEDED',
        'strategy_id': 'momentum_strategy',
        'regime': 'STRESSED'
    }
)

if response.json()['kill_switch_triggered']:
    print("⚠️ REJECTION VELOCITY KILL SWITCH TRIGGERED")
```

### 2. Monitor Exposure Velocity
```python
# After portfolio update
response = requests.post(
    'http://localhost:8005/advanced_kill_switches/exposure/snapshot',
    json={
        'gross_exposure': portfolio.gross_exposure,
        'net_exposure': portfolio.net_exposure,
        'leverage': portfolio.leverage,
        'num_positions': len(portfolio.positions)
    }
)

if response.json()['kill_switch_triggered']:
    print("⚠️ EXPOSURE VELOCITY KILL SWITCH TRIGGERED")
```

### 3. Check Before Trading
```python
# Before submitting trade
status = requests.get(
    'http://localhost:8005/advanced_kill_switches/status'
).json()

if status['advanced_kill_switches']['rejection_velocity']['is_active']:
    print("❌ Trading halted - rejection velocity kill switch active")
    return

if not status['advanced_kill_switches']['per_strategy'][0]['is_enabled']:
    print("❌ Strategy disabled by kill switch")
    return
```

### 4. Manual Strategy Control
```python
# Disable underperforming strategy
requests.post(
    'http://localhost:8005/advanced_kill_switches/strategy/control',
    json={
        'strategy_id': 'failing_strategy',
        'action': 'disable',
        'reason': 'Manual disable - poor regime fit'
    }
)

# Re-enable after conditions improve
requests.post(
    'http://localhost:8005/advanced_kill_switches/strategy/control',
    json={
        'strategy_id': 'failing_strategy',
        'action': 'enable'
    }
)
```

---

## 📈 Monitoring Dashboard Example

```python
# Get comprehensive status
status = requests.get(
    'http://localhost:8005/advanced_kill_switches/status'
).json()

rejection_stats = requests.get(
    'http://localhost:8005/advanced_kill_switches/rejection/stats'
).json()

exposure_stats = requests.get(
    'http://localhost:8005/advanced_kill_switches/exposure/stats'
).json()

strategies = requests.get(
    'http://localhost:8005/advanced_kill_switches/strategies/all'
).json()

print(f"Rejection Rate: {rejection_stats['threshold_pct']:.1f}%")
print(f"Exposure Velocity: {exposure_stats['current_exposure_rate_pct_per_min']:.1f}%/min")
print(f"Active Strategies: {sum(1 for s in strategies['strategies'] if s['is_enabled'])}")
print(f"Disabled Strategies: {sum(1 for s in strategies['strategies'] if not s['is_enabled'])}")
```

---

## 🚨 Alert Configuration

**Recommended Alerting**:
1. **CRITICAL**: Any `HALT` or `LIQUIDATE` severity event
2. **WARNING**: `THROTTLE` events (monitor for escalation)
3. **INFO**: Strategy disable/enable events

**Alert Channels**:
- Slack/Discord webhook for real-time alerts
- Email for `HALT` severity
- SMS for `LIQUIDATE` severity
- Dashboard monitoring for `WARNING`

---

## 🎯 Complete API Summary

### Basic Kill Switches (3 endpoints)
- GET /kill_switches - Basic kill switch status
- POST /halt - Manual emergency halt
- POST /resume - Resume trading

### Advanced Kill Switches (10 endpoints)
1. GET /advanced_kill_switches/status
2. POST /advanced_kill_switches/rejection/record
3. GET /advanced_kill_switches/rejection/stats
4. POST /advanced_kill_switches/exposure/snapshot
5. GET /advanced_kill_switches/exposure/stats
6. POST /advanced_kill_switches/strategy/control
7. GET /advanced_kill_switches/strategy/{strategy_id}/status
8. GET /advanced_kill_switches/strategies/all
9. GET /advanced_kill_switches/events/recent
10. POST /advanced_kill_switches/check

**Total Kill Switch Endpoints**: 13 (3 basic + 10 advanced)

---

## ✅ Validation

```bash
# Start API
python -m arbitrex.risk_portfolio_manager.api

# Test advanced kill switch status
curl http://localhost:8005/advanced_kill_switches/status

# Record rejection
curl -X POST http://localhost:8005/advanced_kill_switches/rejection/record \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD", "reason": "TEST", "regime": "TRENDING"}'

# Check all switches
curl -X POST http://localhost:8005/advanced_kill_switches/check?regime=STRESSED
```

---

## 🎉 Key Benefits

1. **Institutional-Grade Protection**: Multi-layered circuit breakers
2. **Early Warning System**: THROTTLE before HALT
3. **Strategy Isolation**: Per-strategy kill switches prevent portfolio-wide damage
4. **Velocity Detection**: Catches runaway scenarios early
5. **Regime-Aware**: Adjusts thresholds based on market conditions
6. **Full Audit Trail**: Event history for compliance
7. **Manual Override**: Emergency controls when needed
8. **Real-Time Monitoring**: API endpoints for dashboards

---

**Version**: 2.0.1 Enterprise  
**Status**: Production Ready  
**Total Endpoints**: 41 (32 core + 3 basic kill + 10 advanced kill - 4 overlap = 41 unique)
