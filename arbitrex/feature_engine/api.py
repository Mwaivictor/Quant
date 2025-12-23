"""
Feature Engine REST API

Provides HTTP endpoints for feature computation, retrieval, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from arbitrex.feature_engine.config import FeatureEngineConfig
from arbitrex.feature_engine.pipeline import FeaturePipeline
from arbitrex.feature_engine.feature_store import FeatureStore
from arbitrex.feature_engine.health_monitor import FeatureEngineHealthMonitor
from arbitrex.feature_engine.schemas import FeatureVector

try:
    from arbitrex.event_bus import get_event_bus, Event, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

LOG = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArbitreX Feature Engine API",
    description="REST API for ML feature computation and retrieval",
    version="1.0.0"
)

# Global instances
config = FeatureEngineConfig()
pipeline = FeaturePipeline(config)
feature_store = FeatureStore(base_path=Path("./arbitrex/data/features"))
health_monitor = FeatureEngineHealthMonitor()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ComputeFeaturesRequest(BaseModel):
    """Request to compute features from OHLCV data"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (1H, 4H, 1D)")
    ohlcv_data: List[Dict] = Field(..., description="List of OHLCV bars")
    normalize: bool = Field(True, description="Apply normalization")
    store_features: bool = Field(True, description="Store computed features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "normalize": True,
                "store_features": True,
                "ohlcv_data": [
                    {
                        "timestamp_utc": "2025-01-01T00:00:00Z",
                        "open": 1.0500,
                        "high": 1.0520,
                        "low": 1.0495,
                        "close": 1.0510,
                        "volume": 1000,
                        "spread": 0.0002,
                        "log_return_1": 0.0001,
                        "valid_bar": True
                    }
                ]
            }
        }


class ComputeFeaturesResponse(BaseModel):
    """Response from feature computation"""
    success: bool
    symbol: str
    timeframe: str
    features_computed: int
    bars_processed: int
    config_version: str
    computation_time_ms: float
    features: Optional[List[Dict]] = None
    error: Optional[str] = None


class GetFeaturesRequest(BaseModel):
    """Request to retrieve features from store"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    config_version: Optional[str] = Field(None, description="Specific config version")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")


class FeatureVectorResponse(BaseModel):
    """Response containing a feature vector"""
    timestamp_utc: str
    symbol: str
    timeframe: str
    feature_values: List[float]
    feature_names: List[str]
    config_version: str
    is_ml_ready: bool


class HealthResponse(BaseModel):
    """Health status response"""
    status: str
    success_rate_pct: float
    total_computations: int
    uptime_seconds: float
    metrics: Dict


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ArbitreX Feature Engine API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "compute": "/compute",
            "features": "/features/{symbol}/{timeframe}",
            "vector": "/vector/{symbol}/{timeframe}/{timestamp}",
            "health": "/health",
            "config": "/config"
        }
    }


@app.post("/compute", response_model=ComputeFeaturesResponse)
async def compute_features(request: ComputeFeaturesRequest):
    """
    Compute features from OHLCV data.
    
    Args:
        request: Feature computation request
    
    Returns:
        Computed features and metadata
    """
    start_time = health_monitor.record_computation_start(request.symbol, request.timeframe)
    
    try:
        # Convert OHLCV data to DataFrame
        df = pd.DataFrame(request.ohlcv_data)
        
        # Parse timestamp if string
        if 'timestamp_utc' in df.columns and df['timestamp_utc'].dtype == 'object':
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        
        # Add symbol and timeframe if not present
        df['symbol'] = request.symbol
        df['timeframe'] = request.timeframe
        
        # Compute features
        feature_df, metadata = pipeline.compute_features(
            df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            normalize=request.normalize
        )
        
        # Store features if requested
        if request.store_features:
            feature_store.write_features(
                feature_df,
                symbol=request.symbol,
                timeframe=request.timeframe,
                config_version=metadata.config_hash
            )
            health_monitor.record_storage_write(True, metadata.config_hash)
        
        # Calculate feature coverage
        feature_cols = metadata.feature_names
        total_features = len(feature_cols) * len(feature_df)
        non_null_features = sum(feature_df[col].notna().sum() for col in feature_cols if col in feature_df.columns)
        coverage_pct = (non_null_features / total_features * 100) if total_features > 0 else 0.0
        
        # Record success
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        health_monitor.record_computation_success(
            request.symbol,
            request.timeframe,
            start_time,
            metadata.features_computed,
            metadata.valid_bars_processed,
            coverage_pct,
            metadata.normalization_applied
        )
        
        # Convert features to list of dicts
        feature_list = feature_df.to_dict('records')
        
        # Convert numpy types to native Python types
        for record in feature_list:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
        
        return ComputeFeaturesResponse(
            success=True,
            symbol=request.symbol,
            timeframe=request.timeframe,
            features_computed=metadata.features_computed,
            bars_processed=metadata.valid_bars_processed,
            config_version=metadata.config_hash,
            computation_time_ms=elapsed_ms,
            features=feature_list
        )
        
    except Exception as e:
        LOG.error(f"Feature computation failed: {e}", exc_info=True)
        health_monitor.record_computation_failure(request.symbol, request.timeframe, str(e))
        
        return ComputeFeaturesResponse(
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            features_computed=0,
            bars_processed=0,
            config_version="",
            computation_time_ms=0.0,
            error=str(e)
        )


@app.get("/features/{symbol}/{timeframe}")
async def get_features(
    symbol: str = PathParam(..., description="Trading symbol"),
    timeframe: str = PathParam(..., description="Timeframe"),
    config_version: Optional[str] = Query(None, description="Config version"),
    limit: int = Query(100, ge=1, le=10000, description="Max records to return")
):
    """
    Retrieve computed features from store.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        config_version: Optional config version
        limit: Maximum records to return
    
    Returns:
        Feature data
    """
    try:
        # Check if features exist
        if not feature_store.feature_exists(symbol, timeframe, config_version or config.get_config_hash()):
            raise HTTPException(
                status_code=404,
                detail=f"Features not found for {symbol} {timeframe}"
            )
        
        # Read features
        if config_version:
            feature_df = feature_store.read_features(symbol, timeframe, config_version)
        else:
            feature_df = feature_store.get_latest_features(symbol, timeframe)
        
        health_monitor.record_storage_read(True)
        
        # Limit records
        if len(feature_df) > limit:
            feature_df = feature_df.tail(limit)
        
        # Convert to records
        records = feature_df.to_dict('records')
        
        # Convert numpy types
        for record in records:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "records": len(records),
            "features": records
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOG.error(f"Failed to retrieve features: {e}", exc_info=True)
        health_monitor.record_storage_read(False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector/{symbol}/{timeframe}/{timestamp}", response_model=FeatureVectorResponse)
async def get_feature_vector(
    symbol: str = PathParam(..., description="Trading symbol"),
    timeframe: str = PathParam(..., description="Timeframe"),
    timestamp: str = PathParam(..., description="Timestamp (ISO format)"),
    ml_only: bool = Query(True, description="ML-ready features only")
):
    """
    Get feature vector for specific timestamp.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        timestamp: Target timestamp
        ml_only: Return only ML-ready features
    
    Returns:
        Feature vector
    """
    try:
        # Parse timestamp
        ts = pd.Timestamp(timestamp, tz='UTC')
        
        # Get features from store
        feature_df = feature_store.get_latest_features(symbol, timeframe)
        
        if len(feature_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for {symbol} {timeframe}"
            )
        
        # Freeze vector at timestamp
        vector = pipeline.freeze_feature_vector(
            feature_df,
            ts,
            symbol,
            timeframe,
            ml_only=ml_only
        )
        
        # Convert to response model
        return FeatureVectorResponse(
            timestamp_utc=vector.timestamp_utc.isoformat(),
            symbol=vector.symbol,
            timeframe=vector.timeframe,
            feature_values=[float(v) for v in vector.feature_values],
            feature_names=vector.feature_names,
            config_version=vector.feature_version,
            is_ml_ready=vector.is_ml_ready
        )
        
    except HTTPException:
        raise
    except Exception as e:
        LOG.error(f"Failed to get feature vector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/versions/{symbol}/{timeframe}")
async def list_feature_versions(
    symbol: str = PathParam(..., description="Trading symbol"),
    timeframe: str = PathParam(..., description="Timeframe")
):
    """
    List available feature versions.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
    
    Returns:
        List of available versions
    """
    try:
        versions = feature_store.list_versions(symbol, timeframe)
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "versions": versions,
            "count": len(versions)
        }
        
    except Exception as e:
        LOG.error(f"Failed to list versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Get health status of Feature Engine.
    
    Returns:
        Health metrics and status
    """
    try:
        status = health_monitor.get_health_status()
        
        return HealthResponse(
            status=status['status'],
            success_rate_pct=status['success_rate_pct'],
            total_computations=status['total_computations'],
            uptime_seconds=status['metrics']['timestamps']['uptime_seconds'],
            metrics=status['metrics']
        )
        
    except Exception as e:
        LOG.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="UNHEALTHY",
            success_rate_pct=0.0,
            total_computations=0,
            uptime_seconds=0.0,
            metrics={"error": str(e)}
        )


@app.get("/config")
async def get_config():
    """
    Get current Feature Engine configuration.
    
    Returns:
        Configuration details
    """
    try:
        config_dict = config.to_dict()
        
        return {
            "success": True,
            "config_version": config.config_version,
            "config_hash": config.get_config_hash(),
            "configuration": config_dict
        }
        
    except Exception as e:
        LOG.error(f"Failed to get config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/{timeframe}")
async def get_feature_schema(
    timeframe: str = PathParam(..., description="Timeframe"),
    ml_only: bool = Query(True, description="ML features only")
):
    """
    Get feature schema for timeframe.
    
    Args:
        timeframe: Timeframe
        ml_only: Return only ML-ready features
    
    Returns:
        Feature schema
    """
    try:
        if ml_only:
            features = pipeline.schema.get_ml_features(timeframe)
        else:
            features = pipeline.schema.get_all_features(timeframe)
        
        return {
            "success": True,
            "timeframe": timeframe,
            "ml_only": ml_only,
            "feature_count": len(features),
            "features": features
        }
        
    except Exception as e:
        LOG.error(f"Failed to get schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EVENT BUS ENDPOINTS
# ============================================================================

@app.get("/events/metrics")
async def get_event_metrics():
    """Get event bus metrics"""
    if not EVENT_BUS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Event bus not available")
    
    try:
        event_bus = get_event_bus()
        metrics = event_bus.get_metrics()
        return {"event_bus_metrics": metrics, "timestamp": datetime.utcnow(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event metrics error: {str(e)}")


@app.get("/events/health")
async def get_event_health():
    """Check event bus health"""
    if not EVENT_BUS_AVAILABLE:
        return {"event_bus_available": False, "status": "not_available"}
    
    try:
        event_bus = get_event_bus()
        metrics = event_bus.get_metrics()
        is_healthy = metrics['running'] and metrics['events_dropped'] < metrics['events_published'] * 0.1
        return {
            "event_bus_available": True,
            "status": "healthy" if is_healthy else "degraded",
            "metrics": metrics
        }
    except Exception as e:
        return {"event_bus_available": True, "status": "error", "error": str(e)}


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.start()
        LOG.info("✓ Event bus started for Feature Engine")
    
    LOG.info("Feature Engine API starting up...")
    LOG.info(f"Config version: {config.config_version}")
    LOG.info(f"Config hash: {config.get_config_hash()}")
    LOG.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    LOG.info("Feature Engine API shutting down...")
    
    # Export final health metrics
    health_monitor.export_metrics()
    LOG.info("Health metrics exported")
    
    # Stop event bus
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.stop()
        LOG.info("✓ Event bus stopped")
    
    LOG.info("API shutdown complete")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    LOG.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "Internal server error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
