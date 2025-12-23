"""
Risk & Portfolio Manager REST API

FastAPI interface for RPM - provides endpoints for trade approval and monitoring.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uvicorn
import logging

from .engine import RiskPortfolioManager
from .config import RPMConfig
from .expectancy import ExpectancyCalculator
from .portfolio_risk import CovarianceMatrixEstimator, VaRCalculator
from .adaptive_thresholds import RegimeAwareRiskLimits, AdaptiveVolatilityThresholds, StressAdjustedLimits
from .factor_exposure import FactorExposureCalculator, AssetFactorDatabase
from .observability import StructuredLogger, PrometheusMetrics, AlertingSystem

try:
    from arbitrex.event_bus import get_event_bus, Event, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

LOG = logging.getLogger(__name__)


# ========================================
# REQUEST/RESPONSE SCHEMAS
# ========================================

class TradeIntentRequest(BaseModel):
    """Trade intent from Signal Engine"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    direction: int = Field(..., description="Trade direction (1=LONG, -1=SHORT)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Signal confidence [0-1]")
    regime: str = Field(..., description="Market regime (TRENDING, RANGING, VOLATILE, STRESSED)")
    atr: float = Field(..., gt=0.0, description="Average True Range for position sizing")
    vol_percentile: float = Field(..., ge=0.0, le=1.0, description="Volatility percentile [0-1]")
    current_price: Optional[float] = Field(None, description="Current market price (optional)")
    # Kelly/expectancy stats (optional)
    win_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Historical win rate")
    avg_win: Optional[float] = Field(None, gt=0.0, description="Average win %")
    avg_loss: Optional[float] = Field(None, gt=0.0, description="Average loss %")
    num_trades: Optional[int] = Field(None, ge=0, description="Number of trades")
    # Liquidity (optional)
    adv_units: Optional[float] = Field(None, gt=0.0, description="Average daily volume")
    spread_pct: Optional[float] = Field(None, ge=0.0, description="Bid-ask spread %")
    daily_volatility: Optional[float] = Field(None, gt=0.0, description="Daily volatility")


class KellyCalculationRequest(BaseModel):
    """Request Kelly Criterion calculation"""
    win_rate: float = Field(..., ge=0.0, le=1.0)
    avg_win: float = Field(..., gt=0.0)
    avg_loss: float = Field(..., gt=0.0)
    num_trades: Optional[int] = Field(None, ge=0)
    regime: Optional[str] = Field(None, description="Market regime for adaptive cap")


class TradeRecordRequest(BaseModel):
    """Record completed trade for strategy intelligence"""
    strategy_id: str
    symbol: str
    pnl: float
    return_pct: float
    size: float
    regime: Optional[str] = None
    commission: float = 0.0


class OrderFillRequest(BaseModel):
    """Record order fill"""
    order_id: str
    fill_units: float
    fill_price: float
    fill_timestamp: Optional[str] = None


class CorrelationUpdateRequest(BaseModel):
    """Update correlation between two symbols"""
    symbol1: str
    symbol2: str
    correlation: float = Field(..., ge=-1.0, le=1.0)
    regime: Optional[str] = None


class StressTestRequest(BaseModel):
    """Run stress test scenario"""
    scenario_type: str = Field(..., description="HISTORICAL or SYNTHETIC")
    scenario_name: Optional[str] = Field(None, description="e.g., GFC_2008, COVID_2020")
    initial_portfolio_value: float = Field(..., gt=0.0)
    initial_positions: Dict[str, float] = Field(default_factory=dict)


class ConfigUpdateRequest(BaseModel):
    """Update RPM configuration parameters"""
    parameter_name: str
    parameter_value: Any
    reason: Optional[str] = None


class RejectionRecordRequest(BaseModel):
    """Record trade rejection for velocity tracking"""
    symbol: str
    reason: str
    strategy_id: Optional[str] = None
    regime: Optional[str] = 'TRENDING'


class ExposureSnapshotRequest(BaseModel):
    """Record exposure snapshot for velocity tracking"""
    gross_exposure: float
    net_exposure: float
    leverage: float
    num_positions: int


class StrategyControlRequest(BaseModel):
    """Enable/disable strategy"""
    strategy_id: str
    action: str = Field(..., pattern="^(enable|disable)$")
    reason: Optional[str] = None


class ExpectancyCalculationRequest(BaseModel):
    """Calculate expectancy from trading statistics"""
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate [0-1]")
    avg_win: float = Field(..., gt=0.0, description="Average win (decimal)")
    avg_loss: float = Field(..., gt=0.0, description="Average loss (decimal)")
    num_trades: Optional[int] = Field(None, ge=0, description="Number of trades")


class PortfolioVaRRequest(BaseModel):
    """Request portfolio VaR/CVaR calculation"""
    confidence_level: float = Field(95.0, ge=90.0, le=99.9, description="Confidence level (%)")
    positions: Optional[Dict[str, float]] = Field(None, description="Position overrides")


class TradeDecisionResponse(BaseModel):
    """RPM trade decision response"""
    decision: dict
    portfolio_state: dict
    risk_metrics: dict
    config_hash: str
    rpm_version: str
    timestamp: str


# ========================================
# API INITIALIZATION
# ========================================

app = FastAPI(
    title="Risk & Portfolio Manager API",
    description="RPM - The Gatekeeper with absolute veto authority over all trades",
    version="1.0.0",
)

# Global RPM instance
rpm: Optional[RiskPortfolioManager] = None


def get_rpm() -> RiskPortfolioManager:
    """Get RPM instance (lazy initialization)"""
    global rpm
    if rpm is None:
        raise HTTPException(status_code=503, detail="RPM not initialized")
    return rpm


# ========================================
# ENDPOINTS
# ========================================

@app.post("/process_trade", response_model=TradeDecisionResponse)
async def process_trade(request: TradeIntentRequest):
    """
    Process trade intent from Signal Engine.
    
    THIS IS THE CRITICAL ENDPOINT - All trades pass through here.
    RPM exercises absolute veto authority.
    
    Returns:
        TradeDecisionResponse: APPROVED or REJECTED with full audit trail
    """
    rpm_instance = get_rpm()
    
    try:
        output = rpm_instance.process_trade_intent(
            symbol=request.symbol,
            direction=request.direction,
            confidence_score=request.confidence_score,
            regime=request.regime,
            atr=request.atr,
            vol_percentile=request.vol_percentile,
            current_price=request.current_price,
            # Kelly/expectancy stats (optional)
            win_rate=request.win_rate,
            avg_win=request.avg_win,
            avg_loss=request.avg_loss,
            num_trades=request.num_trades,
            # Liquidity (optional)
            adv_units=request.adv_units,
            spread_pct=request.spread_pct,
            daily_volatility=request.daily_volatility,
        )
        
        return TradeDecisionResponse(
            decision=output.decision.to_dict(),
            portfolio_state=output.portfolio_state.to_dict(),
            risk_metrics=output.risk_metrics.to_dict(),
            config_hash=output.config_hash,
            rpm_version=output.rpm_version,
            timestamp=output.timestamp.isoformat(),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RPM processing error: {str(e)}"
        )


@app.get("/health")
async def get_health():
    """
    Get RPM health status.
    
    Returns complete portfolio state, risk metrics, and kill switch status.
    """
    rpm_instance = get_rpm()
    
    try:
        health = rpm_instance.get_health_status()
        return health
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check error: {str(e)}"
        )


@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio state"""
    rpm_instance = get_rpm()
    return rpm_instance.portfolio_state.to_dict()


@app.get("/metrics")
async def get_metrics():
    """Get risk metrics"""
    rpm_instance = get_rpm()
    return rpm_instance.risk_metrics.to_dict()


# ========================================
# BROKER RECONCILIATION ENDPOINTS (NEW)
# ========================================

@app.get("/reconciliation/status")
async def get_reconciliation_status():
    """
    Get broker reconciliation status and latest report.
    
    Returns:
        - Reconciliation statistics
        - Last reconciliation report
        - Drift detection history
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'state_manager') or rpm_instance.state_manager is None:
            return {
                'enabled': False,
                'message': 'Portfolio state manager not initialized'
            }
        
        stats = rpm_instance.state_manager.get_stats()
        report = rpm_instance.state_manager.get_reconciliation_report()
        
        return {
            'enabled': True,
            'reconciliation': stats.get('reconciliation', {}),
            'last_report': report,
        }
    except Exception as e:
        LOG.error(f"Failed to get reconciliation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reconciliation/force")
async def force_reconciliation():
    """
    Force immediate broker reconciliation (for testing or manual intervention).
    
    Returns:
        Reconciliation report with drift analysis
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'state_manager') or rpm_instance.state_manager is None:
            raise HTTPException(
                status_code=400, 
                detail="Portfolio state manager not initialized"
            )
        
        report = rpm_instance.state_manager.force_reconciliation()
        
        if report:
            return {
                'success': True,
                'report': report.to_dict(),
                'message': f"Reconciliation completed - Severity: {report.overall_severity.value}"
            }
        else:
            return {
                'success': False,
                'message': 'Reconciliation failed - check logs'
            }
    
    except Exception as e:
        LOG.error(f"Force reconciliation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reconciliation/history")
async def get_reconciliation_history(limit: int = 10):
    """
    Get reconciliation history.
    
    Args:
        limit: Maximum number of reports to return (default 10)
    
    Returns:
        List of recent reconciliation reports
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'state_manager') or rpm_instance.state_manager is None:
            return {'enabled': False, 'history': []}
        
        engine = rpm_instance.state_manager._reconciliation_engine
        history = engine.reconciliation_history[-limit:]
        
        return {
            'enabled': True,
            'total_reconciliations': engine.total_reconciliations,
            'drift_detected_count': engine.drift_detected_count,
            'auto_corrections': engine.auto_corrections_applied,
            'halts_triggered': engine.halts_triggered,
            'history': [r.to_dict() for r in history],
        }
    
    except Exception as e:
        LOG.error(f"Failed to get reconciliation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kill_switches")
async def get_kill_switches():
    """Get unified kill switch status"""
    rpm_instance = get_rpm()
    
    kill_switch_status = rpm_instance.kill_switches.get_kill_switch_status(
        portfolio_state=rpm_instance.portfolio_state,
        regime="UNKNOWN",  # Would need to fetch latest
    )
    
    return kill_switch_status


@app.post("/halt")
async def manual_halt(reason: str):
    """
    Manually trigger trading halt.
    
    Emergency stop - all trading ceases immediately.
    """
    rpm_instance = get_rpm()
    
    try:
        rpm_instance.kill_switches.manual_halt(reason=reason)
        
        return {
            'status': 'HALTED',
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
            'global_state': rpm_instance.kill_switches.get_state(
                rpm_instance.kill_switches.KillSwitchLevel.GLOBAL if hasattr(rpm_instance.kill_switches, 'KillSwitchLevel') else 'GLOBAL',
                'global'
            ).action.value if hasattr(rpm_instance.kill_switches.get_state(
                rpm_instance.kill_switches.KillSwitchLevel.GLOBAL if hasattr(rpm_instance.kill_switches, 'KillSwitchLevel') else 'GLOBAL',
                'global'
            ), 'action') else 'shutdown'
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Halt error: {str(e)}"
        )


@app.post("/resume")
async def manual_resume():
    """
    Manually resume trading after halt.
    
    Use with caution - ensure conditions are safe.
    """
    rpm_instance = get_rpm()
    
    try:
        rpm_instance.kill_switches.manual_resume(resumed_by="api")
        
        return {
            'status': 'RESUMED',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resume error: {str(e)}"
        )


# ========================================
# UNIFIED KILL-SWITCH ENDPOINTS
# ========================================

@app.get("/kill_switches/summary")
async def get_kill_switches_summary():
    """
    Get comprehensive kill-switch summary.
    
    Returns status of all kill-switches (global, venue, symbol, strategy).
    """
    rpm_instance = get_rpm()
    
    try:
        summary = rpm_instance.kill_switches.get_summary()
        return {
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Kill-switch summary error: {str(e)}"
        )


@app.post("/kill_switches/activate")
async def activate_kill_switch(
    level: str,
    scope_id: str,
    action: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None
):
    """
    Manually activate kill-switch.
    
    Args:
        level: Kill-switch level (global, venue, symbol, strategy)
        scope_id: Scope identifier (e.g., strategy name, symbol)
        action: Response action (throttle, suspend, shutdown)
        reason: Trigger reason
        details: Additional context
    """
    rpm_instance = get_rpm()
    
    try:
        from .kill_switch import KillSwitchLevel, ResponseAction, TriggerReason
        
        rpm_instance.kill_switches.activate_kill_switch(
            level=KillSwitchLevel(level.upper()),
            scope_id=scope_id,
            action=ResponseAction(action.upper()),
            reason=TriggerReason(reason.upper()),
            triggered_by="api",
            details=details or {}
        )
        
        return {
            'status': 'activated',
            'level': level,
            'scope_id': scope_id,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Activate kill-switch error: {str(e)}"
        )


@app.post("/kill_switches/deactivate")
async def deactivate_kill_switch(level: str, scope_id: str):
    """
    Manually deactivate kill-switch.
    
    Args:
        level: Kill-switch level (global, venue, symbol, strategy)
        scope_id: Scope identifier
    """
    rpm_instance = get_rpm()
    
    try:
        from .kill_switch import KillSwitchLevel
        
        rpm_instance.kill_switches.deactivate_kill_switch(
            level=KillSwitchLevel(level.upper()),
            scope_id=scope_id,
            deactivated_by="api"
        )
        
        return {
            'status': 'deactivated',
            'level': level,
            'scope_id': scope_id,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deactivate kill-switch error: {str(e)}"
        )


@app.post("/kill_switches/rejection/record")
async def record_rejection_event(symbol: str, reason: str, strategy_id: Optional[str] = None):
    """
    Record trade rejection for velocity tracking.
    
    Feeds rejection-velocity monitoring in unified kill-switch system.
    """
    rpm_instance = get_rpm()
    
    try:
        rpm_instance.kill_switches.record_rejection(
            symbol=symbol,
            reason=reason,
            strategy_id=strategy_id
        )
        
        # Check if velocity threshold exceeded
        velocity = rpm_instance.kill_switches._check_rejection_velocity()
        
        return {
            'status': 'rejection_recorded',
            'symbol': symbol,
            'rejection_velocity': velocity,
            'kill_switch_triggered': velocity >= 1.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Record rejection error: {str(e)}"
        )


# Deprecated endpoints - maintained for backward compatibility
@app.get("/advanced_kill_switches/status")
async def get_advanced_kill_switches_status():
    """
    DEPRECATED: Use /kill_switches/summary instead.
    
    Legacy endpoint maintained for backward compatibility.
    """
    return await get_kill_switches_summary()


@app.post("/advanced_kill_switches/rejection/record")
async def record_rejection_event_legacy(request: RejectionRecordRequest):
    """
    DEPRECATED: Use /kill_switches/rejection/record instead.
    
    Legacy endpoint maintained for backward compatibility.
    """
    return await record_rejection_event(
        symbol=request.symbol,
        reason=request.reason,
        strategy_id=request.strategy_id
    )





@app.get("/config")
async def get_config():
    """Get RPM configuration"""
    rpm_instance = get_rpm()
    return rpm_instance.config.to_dict()


@app.post("/kelly/calculate")
async def calculate_kelly(request: KellyCalculationRequest):
    """
    Calculate Kelly Criterion with adaptive regime caps.
    
    Returns raw Kelly, fractional Kelly, and regime-adjusted cap.
    """
    rpm_instance = get_rpm()
    
    try:
        from .kelly_criterion import KellyCriterion
        kelly = KellyCriterion(
            safety_factor=rpm_instance.config.kelly_safety_factor,
            max_kelly_pct=rpm_instance.config.kelly_base_max_pct,
            use_adaptive_cap=rpm_instance.config.kelly_use_adaptive_cap
        )
        
        result = kelly.calculate(
            win_rate=request.win_rate,
            avg_win=request.avg_win,
            avg_loss=request.avg_loss,
            num_trades=request.num_trades,
            regime=request.regime
        )
        
        return {
            'kelly_fraction': result.kelly_fraction,
            'fractional_kelly': result.fractional_kelly,
            'kelly_cap': result.kelly_cap,
            'is_valid': result.is_valid,
            'rejection_reason': result.rejection_reason,
            'regime': request.regime,
            'adaptive_cap_enabled': rpm_instance.config.kelly_use_adaptive_cap
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Kelly calculation error: {str(e)}"
        )


@app.get("/strategy/{strategy_id}/metrics")
async def get_strategy_metrics(strategy_id: str):
    """
    Get comprehensive metrics for a specific strategy.
    
    Includes EWMA, regime-conditional stats, and edge decay detection.
    """
    rpm_instance = get_rpm()
    
    try:
        # Access strategy intelligence engine if available
        if hasattr(rpm_instance, 'strategy_intelligence'):
            metrics = rpm_instance.strategy_intelligence.get_strategy_metrics(strategy_id)
            if metrics:
                return metrics.to_dict()
        
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_id}' not found or no metrics available"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy metrics error: {str(e)}"
        )


@app.post("/strategy/record_trade")
async def record_trade(request: TradeRecordRequest):
    """
    Record completed trade for strategy intelligence tracking.
    
    Updates EWMA statistics, regime-specific performance, and edge decay detection.
    """
    rpm_instance = get_rpm()
    
    try:
        from .strategy_intelligence import TradeRecord
        from datetime import datetime
        
        trade = TradeRecord(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            entry_time=datetime.utcnow(),  # Would be provided in real impl
            exit_time=datetime.utcnow(),
            pnl=request.pnl,
            return_pct=request.return_pct,
            size=request.size,
            commission=request.commission
        )
        
        # Record trade with strategy intelligence engine
        if hasattr(rpm_instance, 'strategy_intelligence'):
            rpm_instance.strategy_intelligence.record_trade(
                strategy_id=request.strategy_id,
                trade=trade,
                regime=request.regime
            )
            
            return {
                'status': 'recorded',
                'strategy_id': request.strategy_id,
                'pnl': request.pnl,
                'regime': request.regime
            }
        
        raise HTTPException(
            status_code=503,
            detail="Strategy intelligence not available"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trade recording error: {str(e)}"
        )


@app.get("/strategies/all")
async def get_all_strategies():
    """
    Get metrics for all tracked strategies.
    
    Returns list of strategy IDs with their health status and key metrics.
    """
    rpm_instance = get_rpm()
    
    try:
        if hasattr(rpm_instance, 'strategy_intelligence'):
            all_metrics = rpm_instance.strategy_intelligence.get_all_strategy_metrics()
            return {
                'strategies': [m.to_dict() for m in all_metrics],
                'count': len(all_metrics)
            }
        
        return {'strategies': [], 'count': 0}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"All strategies error: {str(e)}"
        )


@app.get("/edge_tracking/status")
async def get_edge_tracking_status():
    """
    Get edge tracking configuration and current status.
    
    Shows EWMA settings, regime-conditional tracking, and decay detection parameters.
    """
    rpm_instance = get_rpm()
    
    return {
        'ewma_enabled': rpm_instance.config.edge_use_ewma,
        'ewma_halflife_days': rpm_instance.config.edge_ewma_halflife_days,
        'ewma_alpha': rpm_instance.config.edge_ewma_alpha,
        'regime_specific': rpm_instance.config.edge_regime_specific,
        'min_trades_per_regime': rpm_instance.config.edge_min_trades_per_regime,
        'decay_threshold_pct': rpm_instance.config.edge_decay_threshold_pct,
        'auto_reduce_on_decay': rpm_instance.config.edge_auto_reduce_on_decay,
        'decay_multiplier': rpm_instance.config.edge_decay_multiplier,
        'vol_adjusted': rpm_instance.config.edge_vol_adjusted
    }


@app.get("/liquidity/config")
async def get_liquidity_config():
    """
    Get liquidity constraints configuration.
    
    Shows ADV limits, spread limits, and market impact parameters.
    """
    rpm_instance = get_rpm()
    
    return {
        'max_adv_pct': rpm_instance.config.max_adv_pct,
        'max_spread_bps': rpm_instance.config.max_spread_bps,
        'max_market_impact_pct': rpm_instance.config.max_market_impact_pct,
        'impact_coefficient': rpm_instance.config.impact_coefficient,
        'min_adv_units': rpm_instance.config.min_adv_units
    }


@app.get("/orders/pending")
async def get_pending_orders():
    """
    Get all pending orders.
    
    Returns orders approved by RPM but not yet filled.
    """
    rpm_instance = get_rpm()
    
    try:
        pending = rpm_instance.get_pending_orders()
        return {
            'orders': [{
                'order_id': o.order_id,
                'symbol': o.symbol,
                'direction': o.direction,
                'approved_units': o.approved_units,
                'filled_units': o.filled_units,
                'remaining_units': o.remaining_units,
                'status': o.status.value,
                'created_at': o.created_at.isoformat()
            } for o in pending],
            'count': len(pending)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pending orders error: {str(e)}"
        )


@app.post("/orders/{order_id}/fill")
async def record_order_fill(order_id: str, request: OrderFillRequest):
    """
    Record order fill (complete or partial).
    
    Updates order status and portfolio state.
    """
    rpm_instance = get_rpm()
    
    try:
        from datetime import datetime
        fill_time = datetime.fromisoformat(request.fill_timestamp) if request.fill_timestamp else None
        
        rpm_instance.update_order_fill(
            order_id=order_id,
            fill_units=request.fill_units,
            fill_price=request.fill_price,
            fill_timestamp=fill_time
        )
        
        return {
            'status': 'fill_recorded',
            'order_id': order_id,
            'fill_units': request.fill_units,
            'fill_price': request.fill_price
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Order fill error: {str(e)}"
        )


@app.get("/orders/stats")
async def get_order_stats():
    """
    Get order execution statistics.
    
    Includes fill rates, slippage, and rejection rates.
    """
    rpm_instance = get_rpm()
    
    try:
        order_stats = rpm_instance.get_order_stats()
        slippage_stats = rpm_instance.get_slippage_stats()
        
        return {
            'order_stats': order_stats,
            'slippage_stats': slippage_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Order stats error: {str(e)}"
        )


@app.get("/correlation/matrix")
async def get_correlation_matrix(regime: Optional[str] = None):
    """
    Get current correlation matrix.
    
    Returns pairwise correlations for all portfolio positions.
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'correlation_matrix'):
            raise HTTPException(
                status_code=503,
                detail="Correlation tracking not available"
            )
        
        # Get all unique symbols from portfolio
        symbols = list(set(pos.symbol for pos in rpm_instance.portfolio_state.positions.values()))
        
        # Build correlation matrix
        matrix = {}
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i:]:
                corr = rpm_instance.correlation_matrix.get_correlation(
                    sym1, sym2, regime=regime or 'RANGING'
                )
                matrix[f"{sym1}-{sym2}"] = corr
        
        return {
            'correlations': matrix,
            'regime': regime or 'RANGING',
            'symbols': symbols
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Correlation matrix error: {str(e)}"
        )


@app.post("/correlation/update")
async def update_correlation(request: CorrelationUpdateRequest):
    """
    Update correlation between two symbols.
    
    Allows manual override or integration with external correlation engine.
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'correlation_matrix'):
            raise HTTPException(
                status_code=503,
                detail="Correlation tracking not available"
            )
        
        rpm_instance.correlation_matrix.set_correlation(
            symbol1=request.symbol1,
            symbol2=request.symbol2,
            correlation=request.correlation
        )
        
        return {
            'status': 'correlation_updated',
            'symbol1': request.symbol1,
            'symbol2': request.symbol2,
            'correlation': request.correlation,
            'regime': request.regime
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Correlation update error: {str(e)}"
        )


@app.get("/portfolio/volatility")
async def get_portfolio_volatility(regime: str = 'RANGING'):
    """
    Calculate portfolio-level volatility considering correlations.
    
    Critical metric that accounts for diversification effects.
    """
    rpm_instance = get_rpm()
    
    try:
        volatility = rpm_instance.get_portfolio_volatility(regime=regime)
        
        return {
            'portfolio_volatility': volatility,
            'regime': regime,
            'annualized': True,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio volatility error: {str(e)}"
        )


@app.get("/portfolio/diversification")
async def get_diversification_benefit(regime: str = 'RANGING'):
    """
    Calculate diversification benefit.
    
    Shows risk reduction from correlation < 1.0 between positions.
    """
    rpm_instance = get_rpm()
    
    try:
        benefit = rpm_instance.get_diversification_benefit(regime=regime)
        
        return {
            'diversification_benefit': benefit,
            'risk_reduction_pct': (1 - benefit) * 100 if benefit < 1 else 0,
            'regime': regime,
            'interpretation': 'Lower is better (more diversification)'
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diversification benefit error: {str(e)}"
        )


@app.post("/stress_test/run")
async def run_stress_test(request: StressTestRequest):
    """
    Run stress test scenario against current portfolio.
    
    Simulates crisis scenarios and validates risk limits.
    """
    rpm_instance = get_rpm()
    
    try:
        from .stress_testing import StressTestEngine, HistoricalCrisisLibrary, CrisisScenario
        
        engine = StressTestEngine(
            max_acceptable_loss_pct=-15.0,
            max_acceptable_var_breach=2.0,
            max_decision_time_ms=100.0
        )
        
        if request.scenario_type.upper() == 'HISTORICAL':
            scenario_enum = CrisisScenario[request.scenario_name.upper()]
            result = engine.run_historical_crisis_test(
                scenario=scenario_enum,
                initial_portfolio_value=request.initial_portfolio_value,
                initial_positions=request.initial_positions,
                rpm_system=rpm_instance
            )
        else:
            # Synthetic stress test
            result = engine.run_synthetic_stress_test(
                stress_type=request.scenario_name or 'VOLATILITY_SPIKE',
                initial_portfolio_value=request.initial_portfolio_value,
                initial_positions=request.initial_positions,
                rpm_system=rpm_instance
            )
        
        return {
            'scenario': request.scenario_name,
            'scenario_type': request.scenario_type,
            'max_drawdown_pct': result.max_drawdown_pct,
            'final_portfolio_value': result.final_portfolio_value,
            'var_breaches': result.var_breaches,
            'kill_switches_triggered': result.kill_switches_triggered,
            'passed': result.passed,
            'failure_reason': result.failure_reason
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stress test error: {str(e)}"
        )


@app.get("/mt5/sync_status")
async def get_mt5_sync_status():
    """
    Get MT5 synchronization status.
    
    Shows last sync time, position mismatches, and sync health.
    """
    rpm_instance = get_rpm()
    
    try:
        sync_stats = rpm_instance.get_mt5_sync_stats()
        
        return sync_stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MT5 sync status error: {str(e)}"
        )


@app.post("/mt5/sync")
async def trigger_mt5_sync():
    """
    Manually trigger MT5 account synchronization.
    
    Fetches live positions and P&L from MT5.
    """
    rpm_instance = get_rpm()
    
    try:
        success = rpm_instance.sync_with_mt5_account()
        
        if success:
            return {
                'status': 'sync_completed',
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio_state': rpm_instance.portfolio_state.to_dict()
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="MT5 sync failed or not configured"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MT5 sync error: {str(e)}"
        )


@app.post("/state/save")
async def save_state():
    """
    Manually trigger state persistence.
    
    Saves current portfolio state to disk.
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'state_manager'):
            raise HTTPException(
                status_code=503,
                detail="State persistence not enabled"
            )
        
        success = rpm_instance.save_state()
        
        if success:
            return {
                'status': 'state_saved',
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="State save failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"State save error: {str(e)}"
        )


@app.post("/state/backup")
async def create_backup():
    """
    Create backup of current state.
    
    Creates timestamped backup file for disaster recovery.
    """
    rpm_instance = get_rpm()
    
    try:
        if not hasattr(rpm_instance, 'state_manager'):
            raise HTTPException(
                status_code=503,
                detail="State persistence not enabled"
            )
        
        success = rpm_instance.create_backup()
        
        if success:
            return {
                'status': 'backup_created',
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Backup creation failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backup error: {str(e)}"
        )


@app.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    """
    Update RPM configuration parameter at runtime.
    
    WARNING: Changes take effect immediately. Use with caution.
    """
    rpm_instance = get_rpm()
    
    try:
        # Validate parameter exists
        if not hasattr(rpm_instance.config, request.parameter_name):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown parameter: {request.parameter_name}"
            )
        
        # Store old value
        old_value = getattr(rpm_instance.config, request.parameter_name)
        
        # Update parameter
        setattr(rpm_instance.config, request.parameter_name, request.parameter_value)
        
        # Validate updated config
        rpm_instance.config.validate()
        
        # Log change
        LOG.warning(
            f"RPM config updated: {request.parameter_name} = {old_value} -> {request.parameter_value}"
            f" (reason: {request.reason or 'none provided'})"
        )
        
        return {
            'status': 'config_updated',
            'parameter': request.parameter_name,
            'old_value': old_value,
            'new_value': request.parameter_value,
            'reason': request.reason,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Config update error: {str(e)}"
        )


@app.get("/positions/detailed")
async def get_detailed_positions():
    """
    Get detailed position information with risk metrics.
    
    Includes per-position P&L, risk, correlation contributions.
    """
    rpm_instance = get_rpm()
    
    try:
        positions = []
        for pos_id, pos in rpm_instance.portfolio_state.positions.items():
            positions.append({
                'symbol': pos.symbol,
                'direction': pos.direction,
                'units': pos.units,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'position_value': pos.position_value,
                'entry_timestamp': pos.entry_timestamp.isoformat(),
                'regime_at_entry': pos.regime_at_entry
            })
        
        return {
            'positions': positions,
            'count': len(positions),
            'total_unrealized_pnl': sum(p['unrealized_pnl'] for p in positions),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detailed positions error: {str(e)}"
        )


@app.get("/risk/comprehensive")
async def get_comprehensive_risk():
    """
    Get comprehensive risk metrics.
    
    Includes VaR, portfolio volatility, correlation risk, and factor exposures.
    """
    rpm_instance = get_rpm()
    
    try:
        risk_metrics = rpm_instance.risk_metrics.to_dict()
        portfolio_vol = rpm_instance.get_portfolio_volatility()
        diversification = rpm_instance.get_diversification_benefit()
        
        return {
            'risk_metrics': risk_metrics,
            'portfolio_volatility': portfolio_vol,
            'diversification_benefit': diversification,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comprehensive risk error: {str(e)}"
        )


# ========================================
# EXPECTANCY ENDPOINTS (2)
# ========================================

@app.post("/expectancy/calculate")
async def calculate_expectancy(request: ExpectancyCalculationRequest):
    """
    Calculate trading expectancy and position size multiplier.
    
    Formula: E = p·W - (1-p)·L
    Returns expectancy, multiplier, profit factor.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create calculator if not exists
        if not hasattr(rpm_instance, 'expectancy_calculator'):
            rpm_instance.expectancy_calculator = ExpectancyCalculator()
        
        result = rpm_instance.expectancy_calculator.calculate(
            win_rate=request.win_rate,
            avg_win=request.avg_win,
            avg_loss=request.avg_loss,
            num_trades=request.num_trades
        )
        
        return result.to_dict()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Expectancy calculation error: {str(e)}"
        )


@app.get("/expectancy/config")
async def get_expectancy_config():
    """
    Get expectancy calculator configuration.
    
    Returns thresholds and multipliers.
    """
    try:
        rpm_instance = get_rpm()
        
        if not hasattr(rpm_instance, 'expectancy_calculator'):
            rpm_instance.expectancy_calculator = ExpectancyCalculator()
        
        calc = rpm_instance.expectancy_calculator
        
        return {
            'min_expectancy': calc.min_expectancy,
            'high_expectancy_threshold': calc.high_expectancy_threshold,
            'medium_expectancy_threshold': calc.medium_expectancy_threshold,
            'high_multiplier': calc.high_multiplier,
            'medium_multiplier': calc.medium_multiplier,
            'low_multiplier': calc.low_multiplier,
            'min_sample_size': calc.min_sample_size
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Get expectancy config error: {str(e)}"
        )


# ========================================
# PORTFOLIO RISK ENDPOINTS (3)
# ========================================

@app.get("/portfolio/var_cvar")
async def get_portfolio_var_cvar(confidence_level: float = 95.0):
    """
    Calculate advanced portfolio VaR and CVaR.
    
    Uses sophisticated risk models with fat-tail modeling.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create calculator if not exist
        if not hasattr(rpm_instance, 'var_calculator'):
            rpm_instance.var_calculator = VaRCalculator()
        
        # Calculate VaR
        var = rpm_instance.var_calculator.calculate_var(
            portfolio_value=rpm_instance.portfolio_state.nav,
            confidence_level=confidence_level / 100.0
        )
        
        # Calculate CVaR (Expected Shortfall) - same calculator
        cvar = rpm_instance.var_calculator.calculate_cvar(
            portfolio_value=rpm_instance.portfolio_state.nav,
            confidence_level=confidence_level / 100.0
        )
        
        return {
            'confidence_level': confidence_level,
            'var': var,
            'cvar': cvar,
            'portfolio_value': rpm_instance.portfolio_state.nav,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"VaR/CVaR calculation error: {str(e)}"
        )


@app.get("/portfolio/covariance_matrix")
async def get_covariance_matrix(method: str = 'sample'):
    """
    Get rolling covariance matrix estimation.
    
    Methods: 'sample', 'ewma', 'stressed'
    """
    try:
        rpm_instance = get_rpm()
        
        # Create estimator if not exists
        if not hasattr(rpm_instance, 'cov_estimator'):
            rpm_instance.cov_estimator = CovarianceMatrixEstimator()
        
        symbols = list(rpm_instance.portfolio_state.positions.keys())
        if not symbols:
            return {
                'covariance_matrix': {},
                'method': method,
                'symbols': [],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        cov_matrix = rpm_instance.cov_estimator.estimate_covariance_matrix(
            symbols=symbols,
            method=method
        )
        
        return {
            'covariance_matrix': cov_matrix.to_dict(),
            'method': method,
            'symbols': symbols,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Covariance matrix error: {str(e)}"
        )


@app.get("/portfolio/volatility_target")
async def get_volatility_target():
    """
    Get target volatility management metrics.
    
    Shows current vs target volatility and utilization.
    """
    try:
        rpm_instance = get_rpm()
        
        # Get current portfolio volatility
        portfolio_vol = rpm_instance.get_portfolio_volatility()
        target_vol = rpm_instance.config.target_volatility if hasattr(rpm_instance.config, 'target_volatility') else 0.15
        
        return {
            'current_volatility': portfolio_vol,
            'target_volatility': target_vol,
            'volatility_utilization': portfolio_vol / target_vol if target_vol > 0 else 0.0,
            'breaches_target': portfolio_vol > target_vol,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Volatility target error: {str(e)}"
        )


# ========================================
# ADAPTIVE THRESHOLDS ENDPOINTS (4)
# ========================================

@app.get("/adaptive_thresholds/regime/{regime}")
async def get_regime_parameters(regime: str):
    """
    Get risk parameters for specific market regime.
    
    Regimes: TRENDING, RANGING, VOLATILE, STRESSED
    """
    try:
        rpm_instance = get_rpm()
        
        # Create regime manager if not exists
        if not hasattr(rpm_instance, 'regime_limits'):
            rpm_instance.regime_limits = RegimeAwareRiskLimits()
        
        params = rpm_instance.regime_limits.get_regime_parameters(regime)
        
        return params.to_dict()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Get regime parameters error: {str(e)}"
        )


@app.get("/adaptive_thresholds/volatility")
async def get_adaptive_volatility_thresholds():
    """
    Get adaptive volatility thresholds based on rolling percentiles.
    
    Returns high/extreme/crisis volatility levels.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create adaptive vol manager if not exists
        if not hasattr(rpm_instance, 'adaptive_vol'):
            rpm_instance.adaptive_vol = AdaptiveVolatilityThresholds()
        
        thresholds = rpm_instance.adaptive_vol.get_adaptive_thresholds()
        
        return {
            'thresholds': thresholds,
            'lookback_days': rpm_instance.adaptive_vol.lookback_days,
            'high_vol_percentile': rpm_instance.adaptive_vol.high_vol_percentile,
            'extreme_vol_percentile': rpm_instance.adaptive_vol.extreme_vol_percentile,
            'crisis_vol_percentile': rpm_instance.adaptive_vol.crisis_vol_percentile,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Adaptive volatility thresholds error: {str(e)}"
        )


@app.get("/adaptive_thresholds/stress")
async def get_stress_score():
    """
    Get market stress score and stress-adjusted limits.
    
    Stress score: 0.0 = calm, 1.0 = crisis
    """
    try:
        rpm_instance = get_rpm()
        
        # Create stress manager if not exists
        if not hasattr(rpm_instance, 'stress_limits'):
            rpm_instance.stress_limits = StressAdjustedLimits()
        
        stress_score = rpm_instance.stress_limits.calculate_stress_score()
        adjusted_limits = rpm_instance.stress_limits.get_stress_adjusted_limits()
        
        return {
            'stress_score': stress_score,
            'adjusted_limits': adjusted_limits,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stress score error: {str(e)}"
        )


@app.get("/adaptive_thresholds/current")
async def get_current_adaptive_thresholds(regime: str = 'TRENDING'):
    """
    Get comprehensive current adaptive thresholds.
    
    Combines regime parameters, adaptive volatility, and stress adjustments.
    """
    try:
        rpm_instance = get_rpm()
        
        # Initialize components if needed
        if not hasattr(rpm_instance, 'regime_limits'):
            rpm_instance.regime_limits = RegimeAwareRiskLimits()
        if not hasattr(rpm_instance, 'adaptive_vol'):
            rpm_instance.adaptive_vol = AdaptiveVolatilityThresholds()
        if not hasattr(rpm_instance, 'stress_limits'):
            rpm_instance.stress_limits = StressAdjustedLimits()
        
        regime_params = rpm_instance.regime_limits.get_regime_parameters(regime)
        vol_thresholds = rpm_instance.adaptive_vol.get_adaptive_thresholds()
        stress_score = rpm_instance.stress_limits.calculate_stress_score()
        
        return {
            'regime': regime,
            'regime_parameters': regime_params.to_dict(),
            'volatility_thresholds': vol_thresholds,
            'stress_score': stress_score,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Current adaptive thresholds error: {str(e)}"
        )


# ========================================
# FACTOR EXPOSURE ENDPOINTS (2)
# ========================================

@app.get("/portfolio/factor_exposure")
async def get_factor_exposure():
    """
    Get portfolio factor exposure analysis.
    
    Includes beta, momentum, value, size, sector, and macro themes.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create factor calculator if not exists
        if not hasattr(rpm_instance, 'factor_calculator'):
            rpm_instance.factor_calculator = FactorExposureCalculator()
        
        positions = rpm_instance.portfolio_state.positions
        
        if not positions:
            return {
                'factor_exposure': None,
                'positions': {},
                'timestamp': datetime.utcnow().isoformat()
            }
        
        exposure = rpm_instance.factor_calculator.calculate_portfolio_exposure(positions)
        
        return {
            'factor_exposure': exposure.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Factor exposure error: {str(e)}"
        )


@app.get("/portfolio/sector_limits")
async def get_sector_limits():
    """
    Get sector concentration and limit breach detection.
    
    Shows sector exposures and which sectors exceed limits.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create factor calculator if not exists
        if not hasattr(rpm_instance, 'factor_calculator'):
            rpm_instance.factor_calculator = FactorExposureCalculator()
        
        positions = rpm_instance.portfolio_state.positions
        
        if not positions:
            return {
                'sector_exposures': {},
                'breaches': [],
                'within_limits': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        exposure = rpm_instance.factor_calculator.calculate_portfolio_exposure(positions)
        within_limits, breaches = rpm_instance.factor_calculator.check_exposure_limits(exposure)
        
        return {
            'sector_exposures': {s.value: v for s, v in exposure.sector_exposures.items()},
            'max_sector': exposure.max_sector_exposure[0].value if exposure.max_sector_exposure else None,
            'max_sector_pct': exposure.max_sector_exposure[1] if exposure.max_sector_exposure else 0.0,
            'breaches': breaches,
            'within_limits': within_limits,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sector limits error: {str(e)}"
        )


# ========================================
# OBSERVABILITY ENDPOINTS (2)
# ========================================

@app.get("/observability/metrics")
async def get_observability_metrics():
    """
    Get Prometheus metrics export and recent structured logs.
    
    Includes performance metrics, event counts, and log buffer.
    """
    try:
        rpm_instance = get_rpm()
        
        # Create observability components if not exist
        if not hasattr(rpm_instance, 'structured_logger'):
            rpm_instance.structured_logger = StructuredLogger(console_output=False)
        if not hasattr(rpm_instance, 'prometheus_metrics'):
            rpm_instance.prometheus_metrics = PrometheusMetrics()
        
        # Get recent logs
        recent_logs = [log.to_dict() for log in list(rpm_instance.structured_logger.log_buffer)[-50:]]
        
        # Get metrics
        metrics = rpm_instance.prometheus_metrics.export_metrics()
        
        return {
            'prometheus_metrics': metrics,
            'recent_logs': recent_logs,
            'log_buffer_size': len(rpm_instance.structured_logger.log_buffer),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Observability metrics error: {str(e)}"
        )


@app.get("/observability/alerts/active")
async def get_active_alerts():
    """
    Get active alerts with severity levels.
    
    Severity: INFO, WARNING, CRITICAL, EMERGENCY
    """
    try:
        rpm_instance = get_rpm()
        
        # Create alerting system if not exists
        if not hasattr(rpm_instance, 'alerting_system'):
            rpm_instance.alerting_system = AlertingSystem()
        
        active_alerts = rpm_instance.alerting_system.get_active_alerts()
        
        return {
            'active_alerts': [alert.__dict__ for alert in active_alerts],
            'total_active': len(active_alerts),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Active alerts error: {str(e)}"
        )


# ========================================
# RESET ENDPOINTS
# ========================================

@app.post("/reset/daily")
async def reset_daily():
    """Reset daily metrics (call at start of new trading day)"""
    rpm_instance = get_rpm()
    rpm_instance.reset_daily_metrics()
    return {'status': 'daily_metrics_reset'}


@app.post("/reset/weekly")
async def reset_weekly():
    """Reset weekly metrics (call at start of new trading week)"""
    rpm_instance = get_rpm()
    rpm_instance.reset_weekly_metrics()
    return {'status': 'weekly_metrics_reset'}


# ========================================
# EVENT BUS ENDPOINTS
# ========================================

@app.get("/events/metrics")
async def get_event_metrics():
    """Get event bus metrics and statistics"""
    if not EVENT_BUS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Event bus not available"
        )
    
    try:
        event_bus = get_event_bus()
        metrics = event_bus.get_metrics()
        return {
            'event_bus_metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Event metrics error: {str(e)}"
        )


@app.get("/events/health")
async def get_event_health():
    """Check event bus health status"""
    if not EVENT_BUS_AVAILABLE:
        return {
            'event_bus_available': False,
            'status': 'not_available',
            'message': 'Event bus module not installed'
        }
    
    try:
        event_bus = get_event_bus()
        metrics = event_bus.get_metrics()
        
        # Determine health based on metrics
        is_healthy = (
            metrics['running'] and
            metrics['events_dropped'] < metrics['events_published'] * 0.1  # Less than 10% dropped
        )
        
        return {
            'event_bus_available': True,
            'status': 'healthy' if is_healthy else 'degraded',
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            'event_bus_available': True,
            'status': 'error',
            'error': str(e)
        }


# ========================================
# LIFECYCLE EVENTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize RPM on startup"""
    global rpm
    config = RPMConfig()
    rpm = RiskPortfolioManager(config=config, emit_events=True)
    
    # Start event bus
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.start()
        LOG.info("✓ Event bus started for RPM")
    
    LOG.info("RPM initialized successfully")
    LOG.info(f"Config hash: {rpm.config_hash}")
    LOG.info(f"Version: {rpm.version}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    LOG.info("RPM shutting down...")
    
    # Stop event bus
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.stop()
        LOG.info("✓ Event bus stopped")


# ========================================
# MAIN ENTRY POINT
# ========================================

def run_api(host: str = "0.0.0.0", port: int = 8005):
    """
    Run RPM API server.
    
    Args:
        host: Host address
        port: Port number (default 8005)
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
