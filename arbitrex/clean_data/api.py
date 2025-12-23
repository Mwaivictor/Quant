"""
Clean Data Layer REST API

Provides HTTP endpoints for:
1. Clean data access (query by symbol, timeframe, date range)
2. Health monitoring (validation metrics, processing status)
3. Pipeline orchestration (trigger processing jobs)
4. Metadata inspection (config, schemas, processing history)

Usage:
    uvicorn arbitrex.clean_data.api:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json
import logging

from arbitrex.clean_data.pipeline import CleanDataPipeline
from arbitrex.clean_data.integration import RawToCleanBridge
from arbitrex.clean_data.config import CleanDataConfig
from arbitrex.clean_data.schemas import CleanOHLCVSchema, CleanDataMetadata

try:
    from arbitrex.event_bus import get_event_bus, Event, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class HealthStatus(BaseModel):
    """Overall system health status"""
    status: str = Field(..., description="healthy, degraded, or critical")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    
    clean_data_available: bool = Field(..., description="Clean data files accessible")
    raw_data_available: bool = Field(..., description="Raw data files accessible")
    
    total_symbols: int = Field(..., description="Number of symbols with clean data")
    total_bars: int = Field(..., description="Total clean bars across all symbols")


class ValidationMetrics(BaseModel):
    """Validation metrics for a symbol"""
    symbol: str
    timeframe: str
    
    total_bars: int = Field(..., description="Total bars processed")
    valid_bars: int = Field(..., description="Bars passing all validations")
    missing_bars: int = Field(..., description="Bars flagged as missing")
    outlier_bars: int = Field(..., description="Bars flagged as outliers")
    invalid_bars: int = Field(..., description="Bars failing validation gate")
    
    validation_rate: float = Field(..., description="valid_bars / total_bars")
    last_processed: Optional[datetime] = Field(None, description="Last processing timestamp")
    
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class CleanDataQuery(BaseModel):
    """Query parameters for clean data"""
    symbol: str = Field(..., description="Symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (1H, 4H, 1D)")
    
    start_date: Optional[datetime] = Field(None, description="Start date (UTC)")
    end_date: Optional[datetime] = Field(None, description="End date (UTC)")
    
    only_valid: bool = Field(False, description="Return only valid_bar=True rows")
    limit: Optional[int] = Field(None, description="Maximum rows to return", ge=1, le=10000)
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid = ["1H", "4H", "1D", "1M"]
        if v not in valid:
            raise ValueError(f"Timeframe must be one of {valid}")
        return v


class ProcessingRequest(BaseModel):
    """Request to process raw data through clean pipeline"""
    symbols: Optional[List[str]] = Field(None, description="Symbols to process (None = all)")
    timeframe: str = Field(..., description="Timeframe to process")
    
    force_reprocess: bool = Field(False, description="Reprocess even if clean data exists")
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid = ["1H", "4H", "1D", "1M"]
        if v not in valid:
            raise ValueError(f"Timeframe must be one of {valid}")
        return v


class ProcessingStatus(BaseModel):
    """Status of a processing job"""
    job_id: str
    status: str = Field(..., description="pending, running, completed, or failed")
    
    symbols_total: int
    symbols_processed: int
    symbols_succeeded: int
    symbols_failed: int
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ArbitreX Clean Data API",
    description="REST API for accessing and monitoring clean financial data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use Redis or database)
APP_START_TIME = datetime.utcnow()
PROCESSING_JOBS = {}  # job_id -> ProcessingStatus


# ============================================================================
# Helper Functions
# ============================================================================

def get_clean_data_path(symbol: str, timeframe: str) -> Path:
    """Get path to clean data CSV for symbol/timeframe"""
    base_dir = Path("arbitrex/data/clean/ohlcv/fx")
    return base_dir / symbol / timeframe


def load_clean_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    only_valid: bool = False,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Load clean data with optional filtering"""
    
    data_path = get_clean_data_path(symbol, timeframe)
    
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No clean data found for {symbol} {timeframe}"
        )
    
    # Find latest CSV file
    csv_files = sorted(data_path.glob("*.csv"), reverse=True)
    if not csv_files:
        raise HTTPException(
            status_code=404,
            detail=f"No clean data files found for {symbol} {timeframe}"
        )
    
    latest_file = csv_files[0]
    logger.info(f"Loading clean data from {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    
    # Apply filters
    if start_date:
        df = df[df['timestamp_utc'] >= start_date]
    
    if end_date:
        df = df[df['timestamp_utc'] <= end_date]
    
    if only_valid:
        df = df[df['valid_bar'] == True]
    
    if limit:
        df = df.head(limit)
    
    return df


def load_metadata(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """Load metadata JSON for symbol/timeframe"""
    
    data_path = get_clean_data_path(symbol, timeframe)
    
    if not data_path.exists():
        return None
    
    # Find latest metadata file
    meta_files = sorted(data_path.glob("*_metadata.json"), reverse=True)
    if not meta_files:
        return None
    
    latest_meta = meta_files[0]
    
    with open(latest_meta, 'r') as f:
        return json.load(f)


def get_available_symbols(timeframe: str) -> List[str]:
    """Get list of symbols with clean data for timeframe"""
    
    base_dir = Path("arbitrex/data/clean/ohlcv/fx")
    
    if not base_dir.exists():
        return []
    
    symbols = []
    for symbol_dir in base_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
        
        tf_dir = symbol_dir / timeframe
        if tf_dir.exists() and list(tf_dir.glob("*.csv")):
            symbols.append(symbol_dir.name)
    
    return sorted(symbols)


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def get_health():
    """
    Get overall system health status.
    
    Returns information about data availability, symbol count, and uptime.
    """
    
    try:
        # Check data directories
        clean_dir = Path("arbitrex/data/clean/ohlcv/fx")
        raw_dir = Path("arbitrex/data/raw/ohlcv/fx")
        
        clean_available = clean_dir.exists()
        raw_available = raw_dir.exists()
        
        # Count symbols and bars
        total_symbols = 0
        total_bars = 0
        
        if clean_available:
            for symbol_dir in clean_dir.iterdir():
                if symbol_dir.is_dir():
                    total_symbols += 1
                    
                    # Count bars in latest file for each timeframe
                    for tf_dir in symbol_dir.iterdir():
                        if tf_dir.is_dir():
                            csv_files = list(tf_dir.glob("*.csv"))
                            if csv_files:
                                latest = sorted(csv_files, reverse=True)[0]
                                try:
                                    df = pd.read_csv(latest)
                                    total_bars += len(df)
                                except Exception:
                                    pass
        
        # Determine status
        if clean_available and raw_available and total_symbols > 0:
            status = "healthy"
        elif clean_available or raw_available:
            status = "degraded"
        else:
            status = "critical"
        
        # Calculate uptime
        uptime = (datetime.utcnow() - APP_START_TIME).total_seconds()
        
        return HealthStatus(
            status=status,
            uptime_seconds=uptime,
            clean_data_available=clean_available,
            raw_data_available=raw_available,
            total_symbols=total_symbols,
            total_bars=total_bars
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/validation/{symbol}/{timeframe}", response_model=ValidationMetrics, tags=["Health"])
async def get_validation_metrics(symbol: str, timeframe: str):
    """
    Get validation metrics for a specific symbol and timeframe.
    
    Returns detailed statistics about data quality flags.
    """
    
    try:
        # Load metadata
        metadata = load_metadata(symbol, timeframe)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"No metadata found for {symbol} {timeframe}"
            )
        
        # Extract metrics
        return ValidationMetrics(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=metadata.get('total_bars_processed', 0),
            valid_bars=metadata.get('valid_bars', 0),
            missing_bars=metadata.get('missing_bars', 0),
            outlier_bars=metadata.get('outlier_bars', 0),
            invalid_bars=metadata.get('invalid_bars', 0),
            validation_rate=metadata.get('valid_bars', 0) / max(metadata.get('total_bars_processed', 1), 1),
            last_processed=datetime.fromisoformat(metadata['processing_timestamp']) if 'processing_timestamp' in metadata else None,
            warnings=metadata.get('warnings', []),
            errors=metadata.get('errors', [])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get validation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/symbols", tags=["Health"])
async def get_symbols_health(timeframe: str = Query("1H", description="Timeframe to check")):
    """
    Get health metrics for all symbols at a given timeframe.
    
    Returns a list of validation metrics for each available symbol.
    """
    
    try:
        symbols = get_available_symbols(timeframe)
        
        metrics_list = []
        for symbol in symbols:
            try:
                metadata = load_metadata(symbol, timeframe)
                if metadata:
                    metrics_list.append(ValidationMetrics(
                        symbol=symbol,
                        timeframe=timeframe,
                        total_bars=metadata.get('total_bars_processed', 0),
                        valid_bars=metadata.get('valid_bars', 0),
                        missing_bars=metadata.get('missing_bars', 0),
                        outlier_bars=metadata.get('outlier_bars', 0),
                        invalid_bars=metadata.get('invalid_bars', 0),
                        validation_rate=metadata.get('valid_bars', 0) / max(metadata.get('total_bars_processed', 1), 1),
                        last_processed=datetime.fromisoformat(metadata['processing_timestamp']) if 'processing_timestamp' in metadata else None,
                        warnings=metadata.get('warnings', []),
                        errors=metadata.get('errors', [])
                    ))
            except Exception as e:
                logger.warning(f"Failed to load metrics for {symbol}: {e}")
                continue
        
        return {
            "timeframe": timeframe,
            "total_symbols": len(symbols),
            "symbols": metrics_list
        }
    
    except Exception as e:
        logger.error(f"Failed to get symbols health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Clean Data Endpoints
# ============================================================================

@app.get("/clean/data/{symbol}/{timeframe}", tags=["Clean Data"])
async def get_clean_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = Query(None, description="Start date (UTC)"),
    end_date: Optional[datetime] = Query(None, description="End date (UTC)"),
    only_valid: bool = Query(False, description="Return only valid bars"),
    limit: Optional[int] = Query(None, description="Max rows (1-10000)", ge=1, le=10000)
):
    """
    Get clean OHLCV data for a symbol and timeframe.
    
    Returns data with quality flags (is_missing, is_outlier, valid_bar).
    """
    
    try:
        df = load_clean_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            only_valid=only_valid,
            limit=limit
        )
        
        # Convert to JSON-friendly format
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "start_date": df['timestamp_utc'].min().isoformat() if len(df) > 0 else None,
            "end_date": df['timestamp_utc'].max().isoformat() if len(df) > 0 else None,
            "data": df.to_dict(orient='records')
        }
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clean data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clean/symbols", tags=["Clean Data"])
async def list_symbols(timeframe: str = Query("1H", description="Timeframe to filter by")):
    """
    List all symbols with available clean data.
    
    Returns symbols that have clean data files for the specified timeframe.
    """
    
    try:
        symbols = get_available_symbols(timeframe)
        
        return {
            "timeframe": timeframe,
            "count": len(symbols),
            "symbols": symbols
        }
    
    except Exception as e:
        logger.error(f"Failed to list symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clean/metadata/{symbol}/{timeframe}", tags=["Clean Data"])
async def get_metadata(symbol: str, timeframe: str):
    """
    Get processing metadata for a symbol and timeframe.
    
    Returns complete metadata including config versions, statistics, and audit trail.
    """
    
    try:
        metadata = load_metadata(symbol, timeframe)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"No metadata found for {symbol} {timeframe}"
            )
        
        return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clean/latest/{symbol}/{timeframe}", tags=["Clean Data"])
async def get_latest_bars(
    symbol: str,
    timeframe: str,
    count: int = Query(100, description="Number of latest bars", ge=1, le=1000)
):
    """
    Get the most recent clean bars for a symbol.
    
    Returns the N most recent bars ordered by timestamp descending.
    """
    
    try:
        df = load_clean_data(symbol=symbol, timeframe=timeframe)
        
        # Get latest bars
        latest = df.nlargest(count, 'timestamp_utc')
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(latest),
            "latest_timestamp": latest['timestamp_utc'].max().isoformat(),
            "data": latest.to_dict(orient='records')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest bars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Processing Endpoints
# ============================================================================

@app.post("/processing/trigger", response_model=ProcessingStatus, tags=["Processing"])
async def trigger_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Trigger raw → clean data processing in the background.
    
    Processes specified symbols (or all) through the clean data pipeline.
    Returns a job_id for tracking progress.
    """
    
    try:
        # Generate job ID
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize job status
        symbols_to_process = request.symbols or get_available_symbols(request.timeframe)
        
        job_status = ProcessingStatus(
            job_id=job_id,
            status="pending",
            symbols_total=len(symbols_to_process),
            symbols_processed=0,
            symbols_succeeded=0,
            symbols_failed=0,
            started_at=datetime.utcnow()
        )
        
        PROCESSING_JOBS[job_id] = job_status
        
        # Schedule background processing
        background_tasks.add_task(
            process_symbols_background,
            job_id=job_id,
            symbols=symbols_to_process,
            timeframe=request.timeframe,
            force_reprocess=request.force_reprocess
        )
        
        return job_status
    
    except Exception as e:
        logger.error(f"Failed to trigger processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/status/{job_id}", response_model=ProcessingStatus, tags=["Processing"])
async def get_processing_status(job_id: str):
    """
    Get status of a processing job.
    
    Returns current progress and any errors encountered.
    """
    
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return PROCESSING_JOBS[job_id]


async def process_symbols_background(
    job_id: str,
    symbols: List[str],
    timeframe: str,
    force_reprocess: bool
):
    """
    Background task to process symbols through clean pipeline.
    """
    
    job = PROCESSING_JOBS[job_id]
    job.status = "running"
    
    try:
        # Initialize bridge
        bridge = RawToCleanBridge()
        
        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol} {timeframe}...")
                
                # Check if clean data exists
                clean_path = get_clean_data_path(symbol, timeframe)
                if clean_path.exists() and not force_reprocess:
                    logger.info(f"Clean data exists for {symbol}, skipping")
                    job.symbols_succeeded += 1
                    job.symbols_processed += 1
                    continue
                
                # Process through pipeline
                result = bridge.process_symbol(
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                if result['status'] == 'success':
                    job.symbols_succeeded += 1
                else:
                    job.symbols_failed += 1
                    job.errors.append(f"{symbol}: {result.get('error', 'Unknown error')}")
                
                job.symbols_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                job.symbols_failed += 1
                job.symbols_processed += 1
                job.errors.append(f"{symbol}: {str(e)}")
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        job.status = "failed"
        job.errors.append(f"Pipeline error: {str(e)}")
        job.completed_at = datetime.utcnow()


# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/config", tags=["Configuration"])
async def get_config():
    """
    Get current clean data pipeline configuration.
    
    Returns thresholds, validation rules, and processing settings.
    """
    
    try:
        config = CleanDataConfig()
        return config.to_dict()
    
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/schema", tags=["Configuration"])
async def get_schema():
    """
    Get clean data output schema specification.
    
    Returns column names, types, and constraints.
    """
    
    try:
        schema_info = {
            "version": "1.0.0",
            "columns": [
                {"name": "timestamp_utc", "type": "datetime", "required": True, "description": "UTC timestamp (canonical)"},
                {"name": "symbol", "type": "string", "required": True, "description": "Symbol identifier"},
                {"name": "timeframe", "type": "string", "required": True, "description": "Bar timeframe (1H, 4H, etc.)"},
                {"name": "open", "type": "float", "required": True, "description": "Open price (never modified)"},
                {"name": "high", "type": "float", "required": True, "description": "High price (never modified)"},
                {"name": "low", "type": "float", "required": True, "description": "Low price (never modified)"},
                {"name": "close", "type": "float", "required": True, "description": "Close price (never modified)"},
                {"name": "volume", "type": "float", "required": True, "description": "Volume (never modified)"},
                {"name": "log_return_1", "type": "float", "required": False, "description": "Log return vs previous valid bar"},
                {"name": "spread_estimate", "type": "float", "required": False, "description": "Estimated bid-ask spread"},
                {"name": "is_missing", "type": "bool", "required": True, "description": "True if bar was missing from raw data"},
                {"name": "is_outlier", "type": "bool", "required": True, "description": "True if bar flagged as outlier"},
                {"name": "valid_bar", "type": "bool", "required": True, "description": "True if bar passes all validations"},
                {"name": "source_id", "type": "string", "required": True, "description": "Link to raw data source"},
                {"name": "schema_version", "type": "string", "required": True, "description": "Schema version (1.0.0)"}
            ],
            "validation_rules": {
                "valid_bar": "AND(NOT is_missing, NOT is_outlier, OHLC_consistent, timestamp_aligned)"
            }
        }
        
        return schema_info
    
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root with basic information and links"""
    
    return {
        "name": "ArbitreX Clean Data API",
        "version": "1.0.0",
        "description": "REST API for accessing and monitoring clean financial data",
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "clean_data": "/clean/data/{symbol}/{timeframe}",
            "symbols": "/clean/symbols",
            "validation": "/health/validation/{symbol}/{timeframe}",
            "events": "/events/metrics"
        },
        "status": "operational"
    }


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
        return {"event_bus_metrics": metrics, "timestamp": datetime.utcnow().isoformat()}
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
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize Clean Data API and event bus"""
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.start()
        logger.info("✓ Event bus started for Clean Data Layer")
    logger.info("Clean Data API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.stop()
        logger.info("✓ Event bus stopped")
    logger.info("Clean Data API shutdown complete")


# ============================================================================
# Run with: uvicorn arbitrex.clean_data.api:app --reload --port 8001
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
