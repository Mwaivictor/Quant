# RPM API Complete Reference
**Version**: 2.0.1 Enterprise Edition  
**Last Updated**: December 23, 2025

## Overview

Complete REST API reference for Risk & Portfolio Manager (RPM). All endpoints are organized by functional category with comprehensive coverage of the entire RPM codebase.

**Base URL**: `http://localhost:8005`  
**API Framework**: FastAPI with automatic OpenAPI docs at `/docs`

---

## 🎯 Core Trading Operations

### 1. Process Trade Intent
**Endpoint**: `POST /process_trade`  
**Purpose**: THE CRITICAL ENDPOINT - All trades pass through RPM's absolute veto authority

**Request Body**:
```json
{
  "symbol": "EURUSD",
  "direction": 1,
  "confidence_score": 0.85,
  "regime": "TRENDING",
  "atr": 0.0012,
  "vol_percentile": 0.4,
  "current_price": 1.1000,
  "win_rate": 0.65,
  "avg_win": 0.030,
  "avg_loss": 0.018,
  "num_trades": 100,
  "adv_units": 10000000,
  "spread_pct": 0.0002,
  "daily_volatility": 0.008
}
```

**Response**:
```json
{
  "decision": {
    "status": "APPROVED",
    "approved_trade": {
      "position_units": 1524.5,
      "confidence_score": 0.85,
      "regime": "TRENDING"
    }
  },
  "portfolio_state": {...},
  "risk_metrics": {...},
  "config_hash": "abc123",
  "rpm_version": "2.0.1",
  "timestamp": "2025-12-23T10:30:00"
}
```

---

## 📊 Portfolio & Risk Monitoring

### 2. Get Health Status
**Endpoint**: `GET /health`  
**Purpose**: Complete system health check with portfolio state and kill switch status

**Response**:
```json
{
  "status": "HEALTHY",
  "portfolio_state": {...},
  "risk_metrics": {...},
  "kill_switches": {...},
  "timestamp": "2025-12-23T10:30:00"
}
```

### 3. Get Portfolio State
**Endpoint**: `GET /portfolio`  
**Purpose**: Current portfolio positions and P&L

### 4. Get Risk Metrics
**Endpoint**: `GET /metrics`  
**Purpose**: Real-time risk metrics (VaR, Sharpe, drawdown)

### 5. Get Detailed Positions
**Endpoint**: `GET /positions/detailed`  
**Purpose**: Per-position breakdown with unrealized P&L and risk contributions

**Response**:
```json
{
  "positions": [
    {
      "symbol": "EURUSD",
      "direction": 1,
      "units": 10000,
      "entry_price": 1.0950,
      "current_price": 1.1000,
      "unrealized_pnl": 500.0,
      "unrealized_pnl_pct": 0.0457,
      "position_value": 11000.0,
      "entry_timestamp": "2025-12-23T09:00:00",
      "regime_at_entry": "TRENDING"
    }
  ],
  "count": 5,
  "total_unrealized_pnl": 2500.0
}
```

### 6. Get Comprehensive Risk
**Endpoint**: `GET /risk/comprehensive`  
**Purpose**: Full risk analysis including VaR, portfolio volatility, correlation risk

**Response**:
```json
{
  "risk_metrics": {
    "var_95": 5000.0,
    "sharpe_ratio": 1.85,
    "max_drawdown_pct": -8.5
  },
  "portfolio_volatility": 0.145,
  "diversification_benefit": 0.72,
  "timestamp": "2025-12-23T10:30:00"
}
```

---

## 🔒 Kill Switches & Circuit Breakers

### 7. Get Kill Switch Status
**Endpoint**: `GET /kill_switches`  
**Purpose**: Current kill switch state and thresholds

### 8. Manual Halt
**Endpoint**: `POST /halt?reason=Emergency+market+conditions`  
**Purpose**: Emergency stop - all trading ceases immediately

**Response**:
```json
{
  "status": "HALTED",
  "reason": "Emergency market conditions",
  "timestamp": "2025-12-23T10:30:00"
}
```

### 9. Manual Resume
**Endpoint**: `POST /resume`  
**Purpose**: Resume trading after manual halt

---

## 📈 Adaptive Kelly & Edge Tracking

### 10. Calculate Kelly Criterion
**Endpoint**: `POST /kelly/calculate`  
**Purpose**: Calculate Kelly Criterion with adaptive regime caps

**Request Body**:
```json
{
  "win_rate": 0.58,
  "avg_win": 0.025,
  "avg_loss": 0.018,
  "num_trades": 50,
  "regime": "STRESSED"
}
```

**Response**:
```json
{
  "kelly_fraction": 0.0234,
  "fractional_kelly": 0.00585,
  "kelly_cap": 0.002,
  "is_valid": true,
  "regime": "STRESSED",
  "adaptive_cap_enabled": true
}
```

### 11. Get Strategy Metrics
**Endpoint**: `GET /strategy/{strategy_id}/metrics`  
**Purpose**: Comprehensive metrics including EWMA, regime-conditional stats, edge decay

**Response**:
```json
{
  "strategy_id": "momentum_strategy",
  "total_trades": 70,
  "win_rate": 0.464,
  "expectancy": 5.65,
  "ewma_win_rate": 0.102,
  "ewma_expectancy": -87.50,
  "edge_is_decaying": true,
  "edge_decay_pct": -17.66,
  "edge_decay_multiplier": 0.5,
  "health_status": "CRITICAL",
  "health_score": 0.15,
  "regime_metrics": {
    "TRENDING": {
      "trades": 30,
      "win_rate": 0.50,
      "expectancy": 10.0
    }
  }
}
```

### 12. Record Trade
**Endpoint**: `POST /strategy/record_trade`  
**Purpose**: Record completed trade for strategy intelligence tracking

**Request Body**:
```json
{
  "strategy_id": "momentum_strategy",
  "symbol": "EURUSD",
  "pnl": 150.0,
  "return_pct": 0.015,
  "size": 10000,
  "regime": "TRENDING",
  "commission": 2.5
}
```

### 13. Get All Strategies
**Endpoint**: `GET /strategies/all`  
**Purpose**: Metrics for all tracked strategies

### 14. Get Edge Tracking Status
**Endpoint**: `GET /edge_tracking/status`  
**Purpose**: EWMA configuration and current edge tracking parameters

**Response**:
```json
{
  "ewma_enabled": true,
  "ewma_halflife_days": 30.0,
  "ewma_alpha": 0.05,
  "regime_specific": true,
  "min_trades_per_regime": 10,
  "decay_threshold_pct": 0.30,
  "auto_reduce_on_decay": true,
  "decay_multiplier": 0.5,
  "vol_adjusted": true
}
```

---

## 💧 Liquidity Constraints

### 15. Get Liquidity Config
**Endpoint**: `GET /liquidity/config`  
**Purpose**: ADV limits, spread limits, market impact parameters

**Response**:
```json
{
  "max_adv_pct": 0.01,
  "max_spread_bps": 20.0,
  "max_market_impact_pct": 0.005,
  "impact_coefficient": 0.1,
  "min_adv_units": 10000.0
}
```

---

## 📦 Order Management

### 16. Get Pending Orders
**Endpoint**: `GET /orders/pending`  
**Purpose**: All orders approved by RPM but not yet filled

**Response**:
```json
{
  "orders": [
    {
      "order_id": "ORD-20251223-001",
      "symbol": "EURUSD",
      "direction": 1,
      "approved_units": 10000,
      "filled_units": 5000,
      "remaining_units": 5000,
      "status": "PARTIAL",
      "created_at": "2025-12-23T09:00:00"
    }
  ],
  "count": 3
}
```

### 17. Record Order Fill
**Endpoint**: `POST /orders/{order_id}/fill`  
**Purpose**: Record order fill (complete or partial)

**Request Body**:
```json
{
  "order_id": "ORD-20251223-001",
  "fill_units": 5000,
  "fill_price": 1.1005,
  "fill_timestamp": "2025-12-23T10:30:00"
}
```

### 18. Get Order Stats
**Endpoint**: `GET /orders/stats`  
**Purpose**: Order execution statistics including fill rates and slippage

**Response**:
```json
{
  "order_stats": {
    "total_orders": 150,
    "filled": 140,
    "partial": 5,
    "rejected": 3,
    "cancelled": 2,
    "fill_rate": 0.933
  },
  "slippage_stats": {
    "avg_slippage_bps": 1.2,
    "max_slippage_bps": 5.8,
    "positive_slippage_pct": 0.45
  }
}
```

---

## 🔗 Correlation & Portfolio Risk

### 19. Get Correlation Matrix
**Endpoint**: `GET /correlation/matrix?regime=STRESSED`  
**Purpose**: Pairwise correlations for all portfolio positions

**Response**:
```json
{
  "correlations": {
    "EURUSD-GBPUSD": 0.85,
    "EURUSD-USDJPY": -0.40,
    "GBPUSD-USDJPY": -0.35
  },
  "regime": "STRESSED",
  "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
}
```

### 20. Update Correlation
**Endpoint**: `POST /correlation/update`  
**Purpose**: Update correlation between two symbols

**Request Body**:
```json
{
  "symbol1": "EURUSD",
  "symbol2": "GBPUSD",
  "correlation": 0.85,
  "regime": "STRESSED"
}
```

### 21. Get Portfolio Volatility
**Endpoint**: `GET /portfolio/volatility?regime=RANGING`  
**Purpose**: Portfolio-level volatility considering correlations

**Response**:
```json
{
  "portfolio_volatility": 0.145,
  "regime": "RANGING",
  "annualized": true,
  "timestamp": "2025-12-23T10:30:00"
}
```

### 22. Get Diversification Benefit
**Endpoint**: `GET /portfolio/diversification?regime=RANGING`  
**Purpose**: Risk reduction from correlation < 1.0 between positions

**Response**:
```json
{
  "diversification_benefit": 0.72,
  "risk_reduction_pct": 28.0,
  "regime": "RANGING",
  "interpretation": "Lower is better (more diversification)"
}
```

---

## 🧪 Stress Testing

### 23. Run Stress Test
**Endpoint**: `POST /stress_test/run`  
**Purpose**: Simulate crisis scenarios against current portfolio

**Request Body (Historical)**:
```json
{
  "scenario_type": "HISTORICAL",
  "scenario_name": "GFC_2008",
  "initial_portfolio_value": 100000.0,
  "initial_positions": {
    "EURUSD": 10000,
    "GBPUSD": 5000,
    "USDJPY": -8000
  }
}
```

**Request Body (Synthetic)**:
```json
{
  "scenario_type": "SYNTHETIC",
  "scenario_name": "VOLATILITY_SPIKE",
  "initial_portfolio_value": 100000.0,
  "initial_positions": {...}
}
```

**Response**:
```json
{
  "scenario": "GFC_2008",
  "scenario_type": "HISTORICAL",
  "max_drawdown_pct": -18.5,
  "final_portfolio_value": 81500.0,
  "var_breaches": 3,
  "kill_switches_triggered": ["MAX_DRAWDOWN", "GROSS_EXPOSURE"],
  "passed": false,
  "failure_reason": "Max drawdown exceeded threshold"
}
```

**Available Historical Scenarios**:
- `GFC_2008` - Global Financial Crisis
- `FLASH_CRASH_2010` - Flash Crash
- `EURO_CRISIS_2011` - European Debt Crisis
- `TAPER_TANTRUM_2013` - Fed Taper Tantrum
- `CHF_UNPEGGING_2015` - Swiss Franc Unpegging
- `COVID_2020` - COVID-19 Pandemic
- `SVB_MARCH_2023` - Silicon Valley Bank Collapse

---

## 🔄 MT5 Synchronization

### 24. Get MT5 Sync Status
**Endpoint**: `GET /mt5/sync_status`  
**Purpose**: MT5 synchronization status and position mismatches

**Response**:
```json
{
  "enabled": true,
  "last_sync": "2025-12-23T10:25:00",
  "sync_interval_seconds": 300,
  "position_mismatches": 0,
  "balance_mismatch": 0.0,
  "sync_health": "HEALTHY"
}
```

### 25. Trigger MT5 Sync
**Endpoint**: `POST /mt5/sync`  
**Purpose**: Manually trigger MT5 account synchronization

**Response**:
```json
{
  "status": "sync_completed",
  "timestamp": "2025-12-23T10:30:00",
  "portfolio_state": {...}
}
```

---

## 💾 State Management

### 26. Save State
**Endpoint**: `POST /state/save`  
**Purpose**: Manually save portfolio state to disk

### 27. Create Backup
**Endpoint**: `POST /state/backup`  
**Purpose**: Create timestamped backup for disaster recovery

**Response**:
```json
{
  "status": "backup_created",
  "timestamp": "2025-12-23T10:30:00"
}
```

---

## ⚙️ Configuration Management

### 28. Get Config
**Endpoint**: `GET /config`  
**Purpose**: Current RPM configuration parameters

### 29. Update Config
**Endpoint**: `POST /config/update`  
**Purpose**: Update configuration parameter at runtime (use with caution)

**Request Body**:
```json
{
  "parameter_name": "kelly_base_max_pct",
  "parameter_value": 0.012,
  "reason": "Increasing Kelly cap for lower volatility regime"
}
```

**Response**:
```json
{
  "status": "config_updated",
  "parameter": "kelly_base_max_pct",
  "old_value": 0.01,
  "new_value": 0.012,
  "reason": "Increasing Kelly cap for lower volatility regime",
  "timestamp": "2025-12-23T10:30:00"
}
```

---

## 🔄 Daily/Weekly Resets

### 30. Reset Daily Metrics
**Endpoint**: `POST /reset/daily`  
**Purpose**: Reset daily metrics at start of new trading day

### 31. Reset Weekly Metrics
**Endpoint**: `POST /reset/weekly`  
**Purpose**: Reset weekly metrics at start of new trading week

---

## 📋 API Summary by Category

| Category | Endpoints | Coverage |
|----------|-----------|----------|
| **Core Trading** | 1 | process_trade_intent |
| **Monitoring** | 6 | health, portfolio, metrics, positions, risk |
| **Kill Switches** | 3 | status, halt, resume |
| **Kelly & Edge** | 5 | calculate, metrics, record, all strategies, status |
| **Liquidity** | 1 | config |
| **Orders** | 3 | pending, fill, stats |
| **Correlation** | 4 | matrix, update, volatility, diversification |
| **Stress Testing** | 1 | run |
| **MT5 Sync** | 2 | status, trigger |
| **State** | 2 | save, backup |
| **Config** | 2 | get, update |
| **Resets** | 2 | daily, weekly |
| **TOTAL** | **32** | **Complete RPM Coverage** |

---

## 🚀 Quick Start

### Start API Server
```bash
python -m arbitrex.risk_portfolio_manager.api
```

Server runs at: `http://localhost:8005`

### Interactive API Docs
Navigate to: `http://localhost:8005/docs`

FastAPI provides automatic Swagger UI for testing all endpoints.

---

## 🔐 Authentication & Security

**Current Status**: No authentication (development mode)

**Production Requirements**:
- API key authentication
- Rate limiting
- IP whitelisting
- TLS/HTTPS encryption
- Request signing

---

## 📊 Monitoring & Observability

All endpoints support:
- Structured JSON logging
- Correlation IDs for request tracing
- Performance metrics
- Error tracking
- Audit trail generation

---

## ⚠️ Critical Endpoints

**Highest Priority**:
1. `POST /process_trade` - Core trade approval
2. `POST /halt` - Emergency circuit breaker
3. `GET /health` - System monitoring
4. `POST /orders/{order_id}/fill` - Execution confirmation
5. `POST /mt5/sync` - Account synchronization

**Never Bypass**: RPM has absolute veto authority. All trades MUST pass through `/process_trade`.

---

## 📈 Integration Examples

### Python Client
```python
import requests

# Process trade
response = requests.post(
    'http://localhost:8005/process_trade',
    json={
        'symbol': 'EURUSD',
        'direction': 1,
        'confidence_score': 0.85,
        'regime': 'TRENDING',
        'atr': 0.0012,
        'vol_percentile': 0.4,
        'current_price': 1.1000
    }
)

decision = response.json()
if decision['decision']['status'] == 'APPROVED':
    units = decision['decision']['approved_trade']['position_units']
    print(f"Trade approved: {units} units")
```

### Check Health
```python
health = requests.get('http://localhost:8005/health').json()
if health['status'] != 'HEALTHY':
    print(f"WARNING: System health degraded")
```

---

## 🎯 API Design Philosophy

1. **Comprehensive Coverage**: Every RPM function accessible via API
2. **RESTful Design**: Standard HTTP methods (GET/POST)
3. **Rich Responses**: Complete context in every response
4. **Error Handling**: Detailed error messages with HTTP status codes
5. **Validation**: Pydantic schemas ensure type safety
6. **Documentation**: Self-documenting via FastAPI/OpenAPI

---

## 📝 Version History

- **v2.0.1** (Dec 2025) - Complete API coverage with 32 endpoints
- **v2.0.0** (Dec 2025) - Adaptive Kelly, EWMA, correlation risk
- **v1.0.0** (Nov 2025) - Initial RPM release

---

## 🛠️ Troubleshooting

### API Won't Start
```bash
# Check if port 8005 is already in use
netstat -ano | findstr :8005

# Try alternative port
python -m arbitrex.risk_portfolio_manager.api --port 8006
```

### RPM Not Initialized (503 Error)
- Restart API server to trigger `startup_event()`
- Check RPMConfig validation passes

### MT5 Sync Fails
- Verify MT5 connection pool configured
- Check MT5 terminal is running
- Validate account credentials

---

**End of RPM API Complete Reference**
