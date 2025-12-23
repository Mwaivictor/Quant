"""
Signal Engine FastAPI Interface

RESTful API for Signal Generation Engine.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional, Dict
from datetime import datetime
import logging

from arbitrex.signal_engine.engine import SignalGenerationEngine
from arbitrex.signal_engine.config import SignalEngineConfig
from arbitrex.feature_engine.schemas import FeatureVector
from arbitrex.quant_stats.schemas import QuantStatsOutput
from arbitrex.ml_layer.schemas import MLOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Signal Generation Engine API",
    description="Conservative decision layer for systematic FX trading",
    version="1.0.0"
)

# Global engine instance
_engine: Optional[SignalGenerationEngine] = None


def get_engine() -> SignalGenerationEngine:
    """Get or create engine instance"""
    global _engine
    if _engine is None:
        _engine = SignalGenerationEngine()
        logger.info("Signal Generation Engine initialized")
    return _engine


@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    get_engine()
    logger.info("Signal Engine API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Signal Engine API shutting down")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        API status and engine health metrics
    """
    try:
        engine = get_engine()
        health = engine.get_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "engine_health": health.to_dict(),
            "config_hash": engine.config_hash
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.get("/config")
async def get_config():
    """
    Get current engine configuration.
    
    Returns:
        Complete engine configuration
    """
    try:
        engine = get_engine()
        return {
            "config": engine.config.to_dict(),
            "config_hash": engine.config_hash,
            "engine_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Failed to retrieve config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/process")
async def process_bar(
    feature_vector: Dict,
    qse_output: Dict,
    ml_output: Dict,
    bar_index: int
):
    """
    Process a single bar and generate signal decision.
    
    This is the main endpoint for signal generation.
    
    Args:
        feature_vector: Feature vector from Feature Engine
        qse_output: Quant Stats output from QSE
        ml_output: ML predictions from ML Layer
        bar_index: Current bar index
        
    Returns:
        SignalEngineOutput with decision and state
    """
    try:
        engine = get_engine()
        
        # Reconstruct objects from dictionaries
        # Note: In production, use proper Pydantic models or deserialization
        from arbitrex.feature_engine.schemas import FeatureVector
        from arbitrex.quant_stats.schemas import QuantStatsOutput, StatisticalMetrics, SignalValidation, RegimeState
        from arbitrex.ml_layer.schemas import MLOutput, MLPrediction, RegimePrediction, SignalPrediction, ModelMetadata, RegimeLabel
        import numpy as np
        
        # Reconstruct FeatureVector
        fv = FeatureVector(
            timestamp_utc=datetime.fromisoformat(feature_vector['timestamp_utc']),
            symbol=feature_vector['symbol'],
            timeframe=feature_vector['timeframe'],
            feature_values=np.array(feature_vector['feature_values']),
            feature_names=feature_vector['feature_names'],
            feature_version=feature_vector['feature_version'],
            schema_version=feature_vector.get('schema_version', '1.0.0'),
            is_ml_ready=feature_vector.get('is_ml_ready', True)
        )
        
        # Reconstruct QuantStatsOutput
        metrics = StatisticalMetrics(**qse_output['metrics'])
        validation = SignalValidation(**qse_output['validation'])
        regime = RegimeState(**qse_output['regime'])
        
        qse = QuantStatsOutput(
            timestamp=datetime.fromisoformat(qse_output['timestamp']),
            symbol=qse_output['symbol'],
            timeframe=qse_output['timeframe'],
            metrics=metrics,
            validation=validation,
            regime=regime,
            config_hash=qse_output['config_hash'],
            config_version=qse_output['config_version']
        )
        
        # Reconstruct MLOutput
        regime_pred = RegimePrediction(
            regime_label=RegimeLabel(ml_output['prediction']['regime']['regime_label']),
            regime_confidence=ml_output['prediction']['regime']['regime_confidence'],
            prob_trending=ml_output['prediction']['regime']['prob_trending'],
            prob_ranging=ml_output['prediction']['regime']['prob_ranging'],
            prob_stressed=ml_output['prediction']['regime']['prob_stressed'],
            efficiency_ratio=ml_output['prediction']['regime']['efficiency_ratio'],
            volatility_percentile=ml_output['prediction']['regime']['volatility_percentile'],
            correlation_regime=ml_output['prediction']['regime']['correlation_regime'],
            regime_stable=ml_output['prediction']['regime']['regime_stable']
        )
        
        signal_pred = SignalPrediction(
            momentum_success_prob=ml_output['prediction']['signal']['momentum_success_prob'],
            should_enter=ml_output['prediction']['signal']['should_enter'],
            should_exit=ml_output['prediction']['signal']['should_exit'],
            confidence_level=ml_output['prediction']['signal']['confidence_level'],
            top_features=ml_output['prediction']['signal']['top_features']
        )
        
        prediction = MLPrediction(
            regime=regime_pred,
            signal=signal_pred,
            allow_trade=ml_output['prediction']['allow_trade'],
            decision_reasons=ml_output['prediction']['decision_reasons']
        )
        
        regime_model = ModelMetadata(**ml_output['regime_model'])
        signal_model = ModelMetadata(**ml_output['signal_model'])
        
        ml = MLOutput(
            timestamp=datetime.fromisoformat(ml_output['timestamp']),
            symbol=ml_output['symbol'],
            timeframe=ml_output['timeframe'],
            bar_index=ml_output['bar_index'],
            prediction=prediction,
            regime_model=regime_model,
            signal_model=signal_model,
            config_hash=ml_output['config_hash'],
            ml_version=ml_output['ml_version'],
            processing_time_ms=ml_output['processing_time_ms']
        )
        
        # Process bar
        output = engine.process_bar(fv, qse, ml, bar_index)
        
        logger.info(
            f"Processed bar {bar_index} for {fv.symbol} - "
            f"Trade allowed: {output.decision.trade_allowed}"
        )
        
        return output.to_dict()
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/state/{symbol}/{timeframe}")
async def get_state(symbol: str, timeframe: str):
    """
    Get current signal state for symbol/timeframe.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
        timeframe: Timeframe (e.g., H1)
        
    Returns:
        Current signal state
    """
    try:
        engine = get_engine()
        state = engine.state_manager.get_state(symbol, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "state": state.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to retrieve state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/state/all")
async def get_all_states():
    """
    Get all signal states.
    
    Returns:
        Summary of all tracked states
    """
    try:
        engine = get_engine()
        summary = engine.get_state_summary()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to retrieve states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/state/active")
async def get_active_signals():
    """
    Get all active signal states.
    
    Returns:
        Dictionary of active signals
    """
    try:
        engine = get_engine()
        active = engine.state_manager.get_all_active_signals()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(active),
            "active_signals": {k: v.to_dict() for k, v in active.items()}
        }
    except Exception as e:
        logger.error(f"Failed to retrieve active signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/state/reset/{symbol}/{timeframe}")
async def reset_state(symbol: str, timeframe: str):
    """
    Reset signal state for symbol/timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Confirmation
    """
    try:
        engine = get_engine()
        engine.state_manager.reset(symbol, timeframe)
        
        logger.info(f"Reset state for {symbol} {timeframe}")
        
        return {
            "status": "success",
            "message": f"State reset for {symbol} {timeframe}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/reset")
async def reset_engine():
    """
    Reset entire engine (all states and health metrics).
    
    Returns:
        Confirmation
    """
    try:
        engine = get_engine()
        engine.reset()
        
        logger.warning("Engine reset performed")
        
        return {
            "status": "success",
            "message": "Engine reset complete",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
