# RPM API Coverage Audit - Complete Analysis
**Date**: December 23, 2025  
**Total RPM Files**: 20 Python modules  
**Total API Endpoints**: 42

---

## 📁 RPM File Inventory

1. ✅ **adaptive_thresholds.py** - Regime-aware risk limits
2. ✅ **advanced_kill_switches.py** - Institutional circuit breakers
3. ✅ **api.py** - REST API interface
4. ✅ **config.py** - Configuration management
5. ✅ **constraints.py** - Portfolio constraints
6. ✅ **correlation_risk.py** - Correlation-aware sizing
7. ✅ **engine.py** - Core RPM engine
8. ⚠️  **expectancy.py** - Expectancy-based sizing (MISSING API)
9. ✅ **factor_exposure.py** - Factor/sector exposure (PARTIAL - needs API)
10. ✅ **kelly_criterion.py** - Kelly Criterion
11. ✅ **kill_switches.py** - Basic kill switches
12. ✅ **liquidity_constraints.py** - Liquidity limits
13. ✅ **mt5_sync.py** - MT5 synchronization
14. ⚠️  **observability.py** - Logging/metrics (MISSING API)
15. ✅ **order_manager.py** - Order management
16. ⚠️  **portfolio_risk.py** - VaR/CVaR calculations (MISSING API)
17. ✅ **position_sizing.py** - Position sizing (used internally)
18. ✅ **schemas.py** - Data structures
19. ✅ **state_manager.py** - State persistence
20. ✅ **strategy_intelligence.py** - Strategy tracking
21. ✅ **stress_testing.py** - Stress testing

---

## 🗺️ API ENDPOINT → SOURCE FILE MAPPING

### ✅ FULLY COVERED (36 endpoints)

#### Core Trading (1)
- `POST /process_trade` → **engine.py** (`process_trade_intent()`)

#### Monitoring & Health (6)
- `GET /health` → **engine.py** (`get_health_status()`)
- `GET /portfolio` → **schemas.py** (`PortfolioState.to_dict()`)
- `GET /metrics` → **schemas.py** (`RiskMetrics.to_dict()`)
- `GET /positions/detailed` → **engine.py** (portfolio_state.positions)
- `GET /risk/comprehensive` → **engine.py** (risk_metrics + volatility + diversification)
- `GET /kill_switches` → **kill_switches.py** (`get_kill_switch_status()`)

#### Kill Switches - Basic (2)
- `POST /halt` → **kill_switches.py** (`manual_halt()`)
- `POST /resume` → **kill_switches.py** (`manual_resume()`)

#### Kill Switches - Advanced (10)
- `GET /advanced_kill_switches/status` → **advanced_kill_switches.py** (`get_comprehensive_stats()`)
- `POST /advanced_kill_switches/rejection/record` → **advanced_kill_switches.py** (`record_rejection()`)
- `GET /advanced_kill_switches/rejection/stats` → **advanced_kill_switches.py** (`get_stats()`)
- `POST /advanced_kill_switches/exposure/snapshot` → **advanced_kill_switches.py** (`record_snapshot()`)
- `GET /advanced_kill_switches/exposure/stats` → **advanced_kill_switches.py** (`get_stats()`)
- `POST /advanced_kill_switches/strategy/control` → **advanced_kill_switches.py** (`disable_strategy()`, `enable_strategy()`)
- `GET /advanced_kill_switches/strategy/{id}/status` → **advanced_kill_switches.py** (`get_strategy_stats()`)
- `GET /advanced_kill_switches/strategies/all` → **advanced_kill_switches.py** (`get_all_strategies_stats()`)
- `GET /advanced_kill_switches/events/recent` → **advanced_kill_switches.py** (events list)
- `POST /advanced_kill_switches/check` → **advanced_kill_switches.py** (`check_all()`)

#### Kelly & Strategy Intelligence (5)
- `POST /kelly/calculate` → **kelly_criterion.py** (`calculate()`)
- `GET /strategy/{id}/metrics` → **strategy_intelligence.py** (`calculate_metrics()`)
- `POST /strategy/record_trade` → **strategy_intelligence.py** (`record_trade()`)
- `GET /strategies/all` → **strategy_intelligence.py** (get_all_strategy_metrics)
- `GET /edge_tracking/status` → **config.py** (edge tracking config)

#### Liquidity (1)
- `GET /liquidity/config` → **config.py** (liquidity config)

#### Orders (3)
- `GET /orders/pending` → **order_manager.py** (get_pending_orders)
- `POST /orders/{id}/fill` → **order_manager.py** (`add_fill()`)
- `GET /orders/stats` → **order_manager.py** (get_order_stats, get_slippage_stats)

#### Correlation & Risk (4)
- `GET /correlation/matrix` → **correlation_risk.py** (`get_correlation()`)
- `POST /correlation/update` → **correlation_risk.py** (`set_correlation()`)
- `GET /portfolio/volatility` → **correlation_risk.py** (`calculate_portfolio_volatility()`)
- `GET /portfolio/diversification` → **correlation_risk.py** (diversification benefit)

#### Stress Testing (1)
- `POST /stress_test/run` → **stress_testing.py** (`run_historical_crisis_test()`, `run_synthetic_stress_test()`)

#### MT5 Sync (2)
- `GET /mt5/sync_status` → **mt5_sync.py** (`get_sync_stats()`)
- `POST /mt5/sync` → **mt5_sync.py** (`sync_positions()`)

#### State Management (2)
- `POST /state/save` → **state_manager.py** (`save_state()`)
- `POST /state/backup` → **state_manager.py** (`create_backup()`)

#### Configuration (2)
- `GET /config` → **config.py** (`to_dict()`)
- `POST /config/update` → **config.py** (runtime parameter update)

#### Resets (2)
- `POST /reset/daily` → **engine.py** (`reset_daily_metrics()`)
- `POST /reset/weekly` → **engine.py** (`reset_weekly_metrics()`)

---

## ⚠️ MISSING API COVERAGE (6 endpoints needed)

### 1. ❌ EXPECTANCY.PY - NO API ENDPOINTS

**File Purpose**: Expectancy-based position sizing (E = p·W - (1-p)·L)

**Key Functions**:
- `ExpectancyCalculator.calculate()` - Calculate expectancy from win rate, avg win, avg loss
- Returns: expectancy, expectancy_multiplier, is_valid, profit_factor

**Missing Endpoints** (2 needed):
```
POST /expectancy/calculate
  Request: { win_rate, avg_win, avg_loss, num_trades }
  Response: { expectancy, multiplier, is_valid, profit_factor }

GET /expectancy/config
  Response: { min_expectancy, thresholds, multipliers }
```

**Used By**: position_sizing.py (internally) but not exposed via API

---

### 2. ⚠️ PORTFOLIO_RISK.PY - PARTIAL COVERAGE

**File Purpose**: Portfolio-level VaR, CVaR, covariance matrix estimation

**Key Functions**:
- `CovarianceMatrixEstimator` - Estimate rolling covariance
- `VaRCalculator` - Calculate VaR at 95%, 99% confidence
- `CVaRCalculator` - Calculate Expected Shortfall (CVaR)
- `PortfolioVolatilityTargeter` - Target volatility management
- `FatTailModeler` - Student-t distribution fitting

**Current Coverage**: Basic VaR in /metrics, but missing:
- Covariance matrix estimation
- CVaR calculation
- Volatility targeting
- Fat-tail modeling

**Missing Endpoints** (3 needed):
```
GET /portfolio/var_cvar
  Query: confidence_level (95, 99)
  Response: { var_95, var_99, cvar_95, cvar_99, method }

GET /portfolio/covariance_matrix
  Response: { covariance_matrix, method (EWMA, Ledoit-Wolf), timestamp }

GET /portfolio/volatility_target
  Response: { target_vol, current_vol, utilization, scaling_factor }
```

---

### 3. ❌ ADAPTIVE_THRESHOLDS.PY - NO API ENDPOINTS

**File Purpose**: Regime-aware dynamic risk thresholds

**Key Classes**:
- `RegimeAwareRiskLimits` - Different limits per regime
- `AdaptiveVolatilityThresholds` - Rolling percentile thresholds
- `StressAdjustedLimits` - Stress-based adjustments
- `AdaptiveRiskManager` - Unified interface

**Key Functions**:
- `get_regime_parameters()` - Get limits for current regime
- `get_adaptive_thresholds()` - Calculate adaptive thresholds
- `calculate_stress_score()` - Calculate market stress
- `get_stress_adjusted_limits()` - Apply stress adjustments

**Missing Endpoints** (4 needed):
```
GET /adaptive_thresholds/regime/{regime}
  Response: { risk_per_trade, max_position_size, max_exposure, leverage }

GET /adaptive_thresholds/volatility
  Response: { current_percentile, thresholds, adaptive_limits }

GET /adaptive_thresholds/stress
  Response: { stress_score, correlation_stress, volatility_stress, adjustments }

GET /adaptive_thresholds/current
  Query: regime
  Response: { complete limits for current market state }
```

---

### 4. ⚠️ FACTOR_EXPOSURE.PY - PARTIAL COVERAGE

**File Purpose**: Factor & sector exposure tracking

**Key Functions**:
- `AssetFactorDatabase` - Factor profiles for assets
- `FactorExposureCalculator` - Calculate portfolio factor exposures
- `check_exposure_limits()` - Validate sector/factor limits
- `get_factor_diversification_score()` - Measure diversification

**Current Coverage**: None - completely missing

**Missing Endpoints** (2 needed):
```
GET /portfolio/factor_exposure
  Response: { 
    market_beta, momentum, value, size, volatility, quality,
    sector_exposures, macro_themes, diversification_score
  }

GET /portfolio/sector_limits
  Response: { 
    sector_exposures, limits, breaches, concentration_score
  }
```

---

### 5. ❌ OBSERVABILITY.PY - NO API ENDPOINTS

**File Purpose**: Structured logging, metrics, alerting

**Key Classes**:
- `StructuredLogger` - JSON logging with correlation IDs
- `PrometheusMetrics` - Metrics export
- `AlertingSystem` - Alert management
- `ObservabilityManager` - Unified interface

**Key Functions**:
- `log_trade_decision()` - Log trade decisions
- `log_kill_switch()` - Log kill switch events
- `record_execution_time()` - Performance metrics
- Alert severity levels: INFO, WARNING, CRITICAL, EMERGENCY

**Missing Endpoints** (2 needed):
```
GET /observability/metrics
  Response: { prometheus_metrics, recent_logs, alerts }

GET /observability/alerts/active
  Response: { active_alerts, severity, correlation_ids }
```

---

### 6. ⚠️ CONSTRAINTS.PY - IMPLICIT COVERAGE

**File Purpose**: Portfolio constraint validation

**Current Coverage**: Used internally by engine.py, results visible in /health
**Status**: SUFFICIENT - no separate API needed (integrated into /process_trade response)

---

## 📊 COVERAGE SUMMARY

| Category | Files | API Endpoints | Coverage |
|----------|-------|---------------|----------|
| **Fully Covered** | 15 | 36 | ✅ 100% |
| **Partially Covered** | 2 | 5 needed | ⚠️ 50% |
| **No Coverage** | 3 | 8 needed | ❌ 0% |
| **Internal Only** | 3 | N/A | ✅ OK |
| **TOTAL** | 23 | 42 + 13 missing = **55** | 76% |

---

## 🎯 PRIORITY INTEGRATION RECOMMENDATIONS

### HIGH PRIORITY (Add 6 endpoints - Critical Gaps)

1. **Expectancy Calculator** (2 endpoints)
   - POST /expectancy/calculate
   - GET /expectancy/config

2. **Portfolio Risk Analytics** (3 endpoints)
   - GET /portfolio/var_cvar
   - GET /portfolio/covariance_matrix
   - GET /portfolio/volatility_target

3. **Factor Exposure** (2 endpoints)
   - GET /portfolio/factor_exposure
   - GET /portfolio/sector_limits

### MEDIUM PRIORITY (Add 5 endpoints - Enhanced Control)

4. **Adaptive Thresholds** (4 endpoints)
   - GET /adaptive_thresholds/regime/{regime}
   - GET /adaptive_thresholds/volatility
   - GET /adaptive_thresholds/stress
   - GET /adaptive_thresholds/current

5. **Observability** (2 endpoints)
   - GET /observability/metrics
   - GET /observability/alerts/active

### LOW PRIORITY (Already Sufficient)

- **constraints.py** - Integrated into /process_trade
- **position_sizing.py** - Internal calculation, results in /process_trade
- **schemas.py** - Data structures, no API needed

---

## 📈 IMPLEMENTATION PLAN

### Phase 1: Critical Gaps (Expectancy + VaR/CVaR)
**Impact**: High - Essential analytics missing
**Effort**: 2 hours
**Endpoints**: 5

### Phase 2: Factor & Adaptive Thresholds
**Impact**: Medium - Enhanced risk management
**Effort**: 3 hours
**Endpoints**: 6

### Phase 3: Observability & Monitoring
**Impact**: Medium - Production observability
**Effort**: 2 hours
**Endpoints**: 2

---

## ✅ VALIDATION COMMANDS

```powershell
# List all RPM files
Get-ChildItem "arbitrex/risk_portfolio_manager/*.py" | Select-Object Name

# Count API endpoints
python -c "from arbitrex.risk_portfolio_manager import api; print(len([r for r in api.app.routes if hasattr(r, 'methods')]))"

# Check for missing functions
python -c "from arbitrex.risk_portfolio_manager import expectancy, portfolio_risk, adaptive_thresholds, factor_exposure, observability; print('All modules loaded successfully')"
```

---

## 🎯 CURRENT STATE

**Total Potential Endpoints**: 55 (42 existing + 13 missing)  
**Current Coverage**: 76% (42/55)  
**Files Without API**: 3 (expectancy, observability, adaptive_thresholds)  
**Files With Partial API**: 2 (portfolio_risk, factor_exposure)

**Next Action**: Implement Phase 1 (Critical Gaps) - 5 endpoints for expectancy and advanced VaR/CVaR analytics.
