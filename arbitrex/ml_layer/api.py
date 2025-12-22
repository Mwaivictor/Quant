"""
ML Layer FastAPI Interface

Provides REST API endpoints for ML Layer operations:
- Predictions (single & batch)
- Model management (register, load, list)
- Configuration management
- Health & monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from arbitrex.ml_layer.inference import MLInferenceEngine
from arbitrex.ml_layer.config import MLConfig
from arbitrex.ml_layer.model_registry import ModelRegistry
from arbitrex.ml_layer.monitoring import MLMonitor
from arbitrex.ml_layer.schemas import MLOutput, RegimeLabel

LOG = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ArbitreX ML Layer API",
    description="Machine Learning inference and model management API",
    version="1.0.0"
)

# Global instances (initialized on startup)
ml_engine: Optional[MLInferenceEngine] = None
ml_monitor: Optional[MLMonitor] = None
model_registry: Optional[ModelRegistry] = None
config: Optional[MLConfig] = None


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class PredictRequest(BaseModel):
    """Single prediction request"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (e.g., 4H)")
    features: Dict[str, List[float]] = Field(..., description="Feature DataFrame as dict")
    qse_output: Dict = Field(..., description="QSE output dictionary")
    bar_index: Optional[int] = Field(None, description="Bar index (default: last)")


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    symbols: List[str] = Field(..., description="List of symbols")
    timeframe: str = Field(..., description="Timeframe")
    features: Dict[str, Dict[str, List[float]]] = Field(..., description="Features per symbol")
    qse_outputs: Dict[str, Dict] = Field(..., description="QSE outputs per symbol")


class ModelRegisterRequest(BaseModel):
    """Model registration request"""
    model_name: str = Field(..., description="Model name (regime_classifier or signal_filter)")
    version: str = Field(..., description="Semantic version (e.g., v1.0.0)")
    model_path: str = Field(..., description="Path to serialized model file")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Model metadata")


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    regime: Optional[Dict] = None
    signal_filter: Optional[Dict] = None
    model: Optional[Dict] = None
    training: Optional[Dict] = None
    governance: Optional[Dict] = None


# PredictResponse removed - using JSONResponse with dict directly


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize ML Layer on startup"""
    global ml_engine, ml_monitor, model_registry, config
    
    try:
        # Initialize components
        config = MLConfig()
        ml_engine = MLInferenceEngine(config)
        ml_monitor = MLMonitor(config)
        model_registry = ModelRegistry()
        
        LOG.info("✓ ML Layer API started successfully")
        LOG.info(f"  Config hash: {config.get_config_hash()}")
        LOG.info(f"  Monitoring enabled: {config.governance.enable_prediction_logging}")
        
    except Exception as e:
        LOG.error(f"Failed to initialize ML Layer API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global ml_monitor
    
    try:
        if ml_monitor:
            # Export final metrics
            ml_monitor.export_metrics("ml_layer_shutdown_metrics.json")
            LOG.info("✓ ML Layer API shutdown complete")
    except Exception as e:
        LOG.error(f"Error during shutdown: {e}")


# ============================================================================
# Health & Status
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ml_engine": ml_engine is not None,
                "monitor": ml_monitor is not None,
                "registry": model_registry is not None
            },
            "config_hash": config.get_config_hash() if config else None,
            "uptime_seconds": ml_monitor.get_uptime() if ml_monitor else 0
        }
        
        return JSONResponse(content=health_status, status_code=200)
    
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.get("/status")
async def get_status():
    """Detailed status and metrics"""
    try:
        if not ml_monitor:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        metrics = ml_monitor.get_current_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "config_hash": config.get_config_hash(),
            "metrics": metrics,
            "models": {
                "regime_classifier": model_registry.get_latest_version("regime_classifier"),
                "signal_filter": model_registry.get_latest_version("signal_filter")
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict")
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """Single symbol prediction"""
    try:
        if not ml_engine:
            raise HTTPException(status_code=503, detail="ML Engine not initialized")
        
        # Convert features dict to DataFrame
        feature_df = pd.DataFrame(request.features)
        
        # Make prediction
        start_time = datetime.now()
        output = ml_engine.predict(
            symbol=request.symbol,
            timeframe=request.timeframe,
            feature_df=feature_df,
            qse_output=request.qse_output,
            bar_index=request.bar_index
        )
        
        # Log prediction (async)
        if ml_monitor:
            background_tasks.add_task(
                ml_monitor.log_prediction,
                request.symbol,
                request.timeframe,
                output
            )
        
        # Convert to JSON-serializable dict
        response_dict = {
            'symbol': str(request.symbol),
            'timeframe': str(request.timeframe),
            'prediction': output.prediction.to_dict(),
            'processing_time_ms': float(output.processing_time_ms),
            'timestamp': str(output.timestamp),
            'config_hash': str(output.config_hash)
        }
        
        return JSONResponse(content=response_dict)
    
    except Exception as e:
        LOG.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Batch prediction for multiple symbols"""
    try:
        if not ml_engine:
            raise HTTPException(status_code=503, detail="ML Engine not initialized")
        
        # Convert features to DataFrames
        feature_dfs = {
            symbol: pd.DataFrame(features)
            for symbol, features in request.features.items()
        }
        
        # Batch prediction
        outputs = ml_engine.batch_predict(
            symbols=request.symbols,
            timeframe=request.timeframe,
            feature_dfs=feature_dfs,
            qse_outputs=request.qse_outputs
        )
        
        # Log batch predictions (async)
        if ml_monitor:
            for symbol, output in outputs.items():
                background_tasks.add_task(
                    ml_monitor.log_prediction,
                    symbol,
                    request.timeframe,
                    output
                )
        
        # Format response - ensure all types are JSON serializable
        response = {
            symbol: {
                "prediction": output.prediction.to_dict(),
                "processing_time_ms": float(output.processing_time_ms),
                "timestamp": str(output.timestamp),
                "config_hash": output.config_hash
            }
            for symbol, output in outputs.items()
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        LOG.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Management
# ============================================================================

@app.get("/models/list")
async def list_models():
    """List all registered models"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not initialized")
        
        models = model_registry.list_models()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "models": models
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/register")
async def register_model(request: ModelRegisterRequest):
    """Register a new model"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not initialized")
        
        # Load model from path
        import pickle
        with open(request.model_path, 'rb') as f:
            model_object = pickle.load(f)
        
        # Register model
        success = model_registry.register_model(
            model_name=request.model_name,
            version=request.version,
            model_object=model_object,
            metadata=request.metadata
        )
        
        if success:
            LOG.info(f"Model registered: {request.model_name} {request.version}")
            return {
                "status": "success",
                "model_name": request.model_name,
                "version": request.version,
                "message": "Model registered successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Model registration failed")
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        LOG.error(f"Model registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Get all versions of a model"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not initialized")
        
        models = model_registry.list_models()
        versions = [m for m in models if m['model_name'] == model_name]
        
        if not versions:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return {
            "model_name": model_name,
            "versions": versions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/{version}")
async def delete_model(model_name: str, version: str):
    """Delete a model version"""
    try:
        if not model_registry:
            raise HTTPException(status_code=503, detail="Model registry not initialized")
        
        success = model_registry.delete_model(model_name, version)
        
        if success:
            return {
                "status": "success",
                "message": f"Deleted {model_name} {version}"
            }
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Configuration Management
# ============================================================================

@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        if not config:
            raise HTTPException(status_code=503, detail="Config not initialized")
        
        return {
            "config": config.to_dict(),
            "config_hash": config.get_config_hash()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update configuration (requires restart to take effect)"""
    try:
        global config, ml_engine
        
        if not config:
            raise HTTPException(status_code=503, detail="Config not initialized")
        
        # Update config sections
        config_dict = config.to_dict()
        
        if request.regime:
            config_dict['regime'].update(request.regime)
        if request.signal_filter:
            config_dict['signal_filter'].update(request.signal_filter)
        if request.model:
            config_dict['model'].update(request.model)
        if request.training:
            config_dict['training'].update(request.training)
        if request.governance:
            config_dict['governance'].update(request.governance)
        
        # Reload config
        new_config = MLConfig.from_dict(config_dict)
        old_hash = config.get_config_hash()
        new_hash = new_config.get_config_hash()
        
        # Re-initialize engine with new config
        config = new_config
        ml_engine = MLInferenceEngine(config)
        
        LOG.info(f"Config updated: {old_hash[:8]} → {new_hash[:8]}")
        
        return {
            "status": "success",
            "old_config_hash": old_hash,
            "new_config_hash": new_hash,
            "message": "Configuration updated and engine reloaded"
        }
    
    except Exception as e:
        LOG.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Monitoring & Metrics
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """Get current monitoring metrics"""
    try:
        if not ml_monitor:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        metrics = ml_monitor.get_current_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        if not ml_monitor:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        prometheus_format = ml_monitor.export_prometheus()
        
        return JSONResponse(
            content=prometheus_format,
            media_type="text/plain"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/decisions")
async def get_decision_history(limit: int = 100):
    """Get recent prediction decisions"""
    try:
        if not ml_monitor:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        decisions = ml_monitor.get_decision_history(limit=limit)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "count": len(decisions),
            "decisions": decisions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts():
    """Get active alerts"""
    try:
        if not ml_monitor:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        alerts = ml_monitor.get_active_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "count": len(alerts),
            "alerts": alerts
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.post("/reset")
async def reset_engine():
    """Reset ML engine state (clears smoothing buffers)"""
    try:
        if not ml_engine:
            raise HTTPException(status_code=503, detail="ML Engine not initialized")
        
        ml_engine.reset()
        
        return {
            "status": "success",
            "message": "ML Engine state reset"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API root - basic info"""
    return {
        "service": "ArbitreX ML Layer API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "arbitrex.ml_layer.api:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
