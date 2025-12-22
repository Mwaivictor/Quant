"""
QSE REST API

Provides HTTP endpoints for statistical validation, health monitoring, and regime analysis.
"""

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from arbitrex.quant_stats.config import QuantStatsConfig
from arbitrex.quant_stats.engine import QuantitativeStatisticsEngine
from arbitrex.quant_stats.health_monitor import QSEHealthMonitor

LOG = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArbitreX Quantitative Statistics Engine API",
    description="REST API for statistical validation and regime analysis",
    version="1.0.0"
)

# Global instances
config = QuantStatsConfig()
qse = QuantitativeStatisticsEngine(config)
health_monitor = QSEHealthMonitor()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ValidateRequest(BaseModel):
    """Request to validate returns series"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1H", description="Timeframe")
    returns: List[float] = Field(..., description="Return series")
    bar_index: int = Field(..., description="Current bar index")
    returns_dict: Optional[Dict[str, List[float]]] = Field(None, description="Multi-symbol returns for correlation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "returns": [0.001, -0.002, 0.003, 0.001],
                "bar_index": 100
            }
        }


class ValidateResponse(BaseModel):
    """Response from validation"""
    success: bool
    symbol: str
    timeframe: str
    signal_valid: bool
    
    # Metrics
    trend_persistence_score: float
    adf_stationary: bool
    adf_pvalue: float
    z_score: float
    is_outlier: bool
    volatility_regime: str
    volatility_percentile: float
    
    # Validation details
    autocorr_check: bool
    stationarity_check: bool
    distribution_check: bool
    correlation_check: bool
    volatility_check: bool
    
    # Regime
    trend_regime: str
    market_phase: str
    regime_stable: bool
    
    # Failure reasons (if invalid)
    failure_reasons: List[str]
    
    # Performance
    processing_time_ms: float
    config_hash: str
    
    error: Optional[str] = None


class RegimeResponse(BaseModel):
    """Response with regime state"""
    symbol: str
    timeframe: str
    trend_regime: str
    trend_strength: float
    volatility_regime: str
    volatility_level: float
    market_phase: str
    efficiency_ratio: float
    regime_stable: bool
    correlation_regime: str
    avg_correlation: float


class HealthResponse(BaseModel):
    """Health status response"""
    status: str
    uptime_seconds: float
    total_validations: int
    valid_signals: int
    invalid_signals: int
    validity_rate: float
    avg_processing_time_ms: float
    symbols_tracked: int
    unhealthy_symbols: int


class SymbolHealthResponse(BaseModel):
    """Symbol-specific health response"""
    symbol: str
    last_validation_time: Optional[str]
    consecutive_failures: int
    total_validations: int
    valid_signals: int
    invalid_signals: int
    validity_rate: float
    avg_processing_time_ms: float
    avg_trend_score: float
    recent_failures: List[Dict]


class ConfigResponse(BaseModel):
    """Configuration response"""
    config_hash: str
    config_version: str
    autocorrelation: Dict
    stationarity: Dict
    distribution: Dict
    correlation: Dict
    volatility: Dict
    validation: Dict


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ArbitreX Quantitative Statistics Engine API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "validate": "/validate",
            "regime": "/regime/{symbol}",
            "health": "/health",
            "health_symbol": "/health/{symbol}",
            "failure_breakdown": "/failures",
            "config": "/config",
            "recent": "/recent"
        }
    }


@app.post("/validate", response_model=ValidateResponse)
async def validate_signal(request: ValidateRequest):
    """
    Validate signal using statistical tests.
    
    Args:
        request: Validation request with returns series
    
    Returns:
        Validation result with metrics and regime state
    """
    start_time = health_monitor.record_validation_start(request.symbol)
    
    try:
        # Convert to pandas Series
        returns = pd.Series(request.returns)
        
        # Prepare returns dict if provided
        returns_dict = None
        if request.returns_dict:
            returns_dict = {
                sym: pd.Series(rets)
                for sym, rets in request.returns_dict.items()
            }
        
        # Process through QSE
        output = qse.process_bar(
            symbol=request.symbol,
            returns=returns,
            bar_index=request.bar_index,
            returns_dict=returns_dict
        )
        
        # Calculate processing time
        elapsed_ms = (datetime.now() - datetime.fromtimestamp(start_time)).total_seconds() * 1000
        
        # Record success/failure
        metrics_dict = {
            'trend_persistence_score': output.metrics.trend_persistence_score,
            'adf_pvalue': output.metrics.adf_pvalue,
            'z_score': output.metrics.z_score
        }
        
        if output.validation.signal_validity_flag:
            health_monitor.record_validation_success(
                request.symbol,
                start_time,
                metrics_dict
            )
        else:
            health_monitor.record_validation_failure(
                request.symbol,
                start_time,
                output.validation.failure_reasons,
                metrics_dict
            )
        
        # Build response
        return ValidateResponse(
            success=True,
            symbol=request.symbol,
            timeframe=request.timeframe,
            signal_valid=output.validation.signal_validity_flag,
            
            # Metrics
            trend_persistence_score=float(output.metrics.trend_persistence_score),
            adf_stationary=bool(output.metrics.adf_stationary),
            adf_pvalue=float(output.metrics.adf_pvalue),
            z_score=float(output.metrics.z_score),
            is_outlier=bool(output.metrics.is_outlier),
            volatility_regime=output.metrics.volatility_regime,
            volatility_percentile=float(output.metrics.volatility_percentile),
            
            # Validation checks
            autocorr_check=output.validation.autocorr_check_passed,
            stationarity_check=output.validation.stationarity_check_passed,
            distribution_check=output.validation.distribution_check_passed,
            correlation_check=output.validation.correlation_check_passed,
            volatility_check=output.validation.volatility_check_passed,
            
            # Regime
            trend_regime=output.regime.trend_regime,
            market_phase=output.regime.market_phase,
            regime_stable=output.regime.regime_stable,
            
            # Failure reasons
            failure_reasons=output.validation.failure_reasons,
            
            # Performance
            processing_time_ms=elapsed_ms,
            config_hash=output.config_hash
        )
        
    except Exception as e:
        LOG.error(f"Validation failed: {e}", exc_info=True)
        return ValidateResponse(
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            signal_valid=False,
            trend_persistence_score=0.0,
            adf_stationary=False,
            adf_pvalue=1.0,
            z_score=0.0,
            is_outlier=False,
            volatility_regime="UNKNOWN",
            volatility_percentile=0.0,
            autocorr_check=False,
            stationarity_check=False,
            distribution_check=False,
            correlation_check=False,
            volatility_check=False,
            trend_regime="UNKNOWN",
            market_phase="UNKNOWN",
            regime_stable=False,
            failure_reasons=[],
            processing_time_ms=0.0,
            config_hash="",
            error=str(e)
        )


@app.get("/regime/{symbol}", response_model=RegimeResponse)
async def get_regime(
    symbol: str = PathParam(..., description="Trading symbol"),
    returns: str = Query(..., description="Comma-separated returns"),
    bar_index: int = Query(..., description="Current bar index")
):
    """
    Get current market regime for symbol.
    
    Args:
        symbol: Trading symbol
        returns: Comma-separated return values
        bar_index: Current bar index
    
    Returns:
        Regime state information
    """
    try:
        # Parse returns
        returns_list = [float(x) for x in returns.split(',')]
        returns_series = pd.Series(returns_list)
        
        # Process through QSE
        output = qse.process_bar(
            symbol=symbol,
            returns=returns_series,
            bar_index=bar_index
        )
        
        return RegimeResponse(
            symbol=symbol,
            timeframe="unknown",
            trend_regime=output.regime.trend_regime,
            trend_strength=float(output.regime.trend_strength),
            volatility_regime=output.regime.volatility_regime,
            volatility_level=float(output.regime.volatility_level),
            market_phase=output.regime.market_phase,
            efficiency_ratio=float(output.regime.efficiency_ratio),
            regime_stable=output.regime.regime_stable,
            correlation_regime=output.regime.correlation_regime,
            avg_correlation=float(output.regime.avg_correlation)
        )
        
    except Exception as e:
        LOG.error(f"Regime retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Get overall health status.
    
    Returns:
        Health metrics and status
    """
    health_status = health_monitor.get_health_status()
    
    return HealthResponse(
        status=health_status['status'],
        uptime_seconds=health_status['uptime_seconds'],
        total_validations=health_status['global_metrics']['total_validations'],
        valid_signals=health_status['global_metrics']['valid_signals'],
        invalid_signals=health_status['global_metrics']['invalid_signals'],
        validity_rate=health_status['validity_rate'],
        avg_processing_time_ms=health_status['avg_processing_time_ms'],
        symbols_tracked=health_status['symbols_tracked'],
        unhealthy_symbols=health_status['unhealthy_symbols']
    )


@app.get("/health/{symbol}", response_model=SymbolHealthResponse)
async def get_symbol_health(
    symbol: str = PathParam(..., description="Trading symbol")
):
    """
    Get health status for specific symbol.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Symbol health metrics
    """
    health = health_monitor.get_symbol_health(symbol)
    
    if health is None:
        raise HTTPException(
            status_code=404,
            detail=f"No health data found for {symbol}"
        )
    
    metrics = health['metrics']
    validity_rate = metrics['valid_signals'] / max(1, metrics['total_validations'])
    
    return SymbolHealthResponse(
        symbol=health['symbol'],
        last_validation_time=health['last_validation_time'],
        consecutive_failures=health['consecutive_failures'],
        total_validations=metrics['total_validations'],
        valid_signals=metrics['valid_signals'],
        invalid_signals=metrics['invalid_signals'],
        validity_rate=validity_rate,
        avg_processing_time_ms=metrics['avg_processing_time_ms'],
        avg_trend_score=metrics['avg_trend_score'],
        recent_failures=health['recent_failures']
    )


@app.get("/failures")
async def get_failure_breakdown():
    """
    Get breakdown of failure types.
    
    Returns:
        Failure counts by type
    """
    return health_monitor.get_failure_breakdown()


@app.get("/recent")
async def get_recent_validations(
    limit: int = Query(20, description="Number of recent validations to return")
):
    """
    Get recent validation history.
    
    Args:
        limit: Maximum number of validations
    
    Returns:
        Recent validation records
    """
    return {
        "validations": health_monitor.get_recent_validations(limit),
        "count": len(health_monitor.get_recent_validations(limit))
    }


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get current QSE configuration.
    
    Returns:
        Configuration settings
    """
    config_dict = config.to_dict()
    
    return ConfigResponse(
        config_hash=config.get_config_hash(),
        config_version=config.config_version,
        autocorrelation=config_dict['autocorrelation'],
        stationarity=config_dict['stationarity'],
        distribution=config_dict['distribution'],
        correlation=config_dict['correlation'],
        volatility=config_dict['volatility'],
        validation=config_dict['validation']
    )


@app.post("/reset-health")
async def reset_health():
    """
    Reset health monitoring metrics (admin only).
    
    Returns:
        Confirmation message
    """
    health_monitor.reset_metrics()
    return {
        "success": True,
        "message": "Health monitoring metrics reset"
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    LOG.info("QSE API starting up...")
    LOG.info(f"Config hash: {config.get_config_hash()}")
    LOG.info(f"Config version: {config.config_version}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    LOG.info("QSE API shutting down...")
    
    # Export final health report
    try:
        health_monitor.export_health_report("qse_health_report_final.json")
        LOG.info("Final health report exported")
    except Exception as e:
        LOG.error(f"Failed to export final health report: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
