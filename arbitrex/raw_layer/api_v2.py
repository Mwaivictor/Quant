"""
Raw Data Layer Complete REST API with Swagger Documentation

Comprehensive FastAPI server providing:
- Raw data access endpoints
- Health monitoring (MT5, ingestion, data quality)
- Ingestion orchestration
- Symbol/universe management
- Configuration inspection
- Full OpenAPI/Swagger documentation

Port: 8000 (Clean Data API uses 8001)

Usage:
    python -m arbitrex.scripts.run_raw_api
    
Access:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - Health: http://localhost:8000/health
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json
import logging

from arbitrex.raw_layer.config import TRADING_UNIVERSE, DEFAULT_TIMEFRAMES
from arbitrex.raw_layer.health import get_health_monitor, init_health_monitor

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
    """Overall raw layer health status"""
    status: str = Field(..., description="healthy, degraded, or critical")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    
    mt5_connected: bool = Field(..., description="MT5 terminal connected")
    raw_data_available: bool = Field(..., description="Raw data files accessible")
    redis_available: bool = Field(..., description="Redis connection active")
    
    total_symbols: int = Field(..., description="Symbols in universe")
    total_raw_bars: int = Field(..., description="Total bars stored")
    last_ingestion: Optional[datetime] = Field(None, description="Last ingestion time")


class MT5Status(BaseModel):
    """MT5 connection status"""
    connected: bool
    account: Optional[int] = None
    server: Optional[str] = None
    company: Optional[str] = None
    symbols_available: int = 0
    error: Optional[str] = None


class IngestionMetrics(BaseModel):
    """Ingestion metrics for a symbol"""
    symbol: str
    timeframe: str
    total_bars: int
    last_ingestion: Optional[datetime] = None
    last_bar_time: Optional[datetime] = None
    freshness_minutes: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class SymbolInfo(BaseModel):
    """Symbol information"""
    symbol: str
    description: Optional[str] = None
    asset_class: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ArbitreX Raw Data Layer API",
    description="Complete REST API for raw financial data ingestion, access, and monitoring with full Swagger documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health monitoring and system status"},
        {"name": "Raw Data", "description": "Access raw OHLCV data"},
        {"name": "Ingestion", "description": "Data ingestion orchestration"},
        {"name": "Symbols", "description": "Symbol information and universe"},
        {"name": "Configuration", "description": "Configuration inspection"},
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
APP_START_TIME = datetime.utcnow()


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Raw Data API...")
    init_health_monitor()
    logger.info("Raw Data API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Raw Data API...")


# ============================================================================
# Helper Functions
# ============================================================================

def get_raw_data_path(symbol: str, timeframe: str) -> Path:
    """Get path to raw data directory"""
    return Path("arbitrex/data/raw/ohlcv/fx") / symbol / timeframe


def load_raw_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """Load raw data with filtering"""
    
    data_path = get_raw_data_path(symbol, timeframe)
    
    if not data_path.exists():
        raise HTTPException(404, f"No data found for {symbol} {timeframe}")
    
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise HTTPException(404, f"No CSV files for {symbol} {timeframe}")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    
    if not dfs:
        raise HTTPException(500, "Failed to load any data files")
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.drop_duplicates(subset=['timestamp_utc']).sort_values('timestamp_utc')
    
    if start_date:
        df = df[df['timestamp_utc'] >= start_date]
    if end_date:
        df = df[df['timestamp_utc'] <= end_date]
    if limit:
        df = df.head(limit)
    
    return df


def get_available_symbols(timeframe: Optional[str] = None) -> List[str]:
    """Get symbols with raw data"""
    
    base_dir = Path("arbitrex/data/raw/ohlcv/fx")
    if not base_dir.exists():
        return []
    
    symbols = []
    for symbol_dir in base_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
        
        if timeframe:
            if (symbol_dir / timeframe).exists():
                symbols.append(symbol_dir.name)
        else:
            symbols.append(symbol_dir.name)
    
    return sorted(symbols)


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root with information and links"""
    return {
        "name": "ArbitreX Raw Data Layer API",
        "version": "1.0.0",
        "description": "REST API with Swagger documentation",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "mt5_status": "/health/mt5",
            "raw_data": "/raw/data/{symbol}/{timeframe}",
            "symbols": "/raw/symbols"
        },
        "status": "operational"
    }


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def get_health():
    """
    Get overall raw layer health status.
    
    Returns system status, MT5 connection, data availability, and key metrics.
    """
    
    try:
        monitor = get_health_monitor()
        summary = monitor.get_health_summary()
        
        # Get data statistics
        total_symbols = len(TRADING_UNIVERSE)
        total_bars = 0
        last_ingestion = None
        
        raw_dir = Path("arbitrex/data/raw/ohlcv/fx")
        if raw_dir.exists():
            for symbol_dir in raw_dir.iterdir():
                if symbol_dir.is_dir():
                    for tf_dir in symbol_dir.iterdir():
                        if tf_dir.is_dir():
                            for csv_file in tf_dir.glob("*.csv"):
                                try:
                                    df = pd.read_csv(csv_file)
                                    total_bars += len(df)
                                    mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
                                    if not last_ingestion or mtime > last_ingestion:
                                        last_ingestion = mtime
                                except Exception:
                                    pass
        
        # Check Redis
        redis_available = False
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            redis_available = True
        except Exception:
            pass
        
        uptime = (datetime.utcnow() - APP_START_TIME).total_seconds()
        
        return HealthStatus(
            status=summary.get('status', 'unknown'),
            uptime_seconds=uptime,
            mt5_connected=summary.get('components', {}).get('mt5', {}).get('status') == 'healthy',
            raw_data_available=raw_dir.exists(),
            redis_available=redis_available,
            total_symbols=total_symbols,
            total_raw_bars=total_bars,
            last_ingestion=last_ingestion
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/health/detailed", tags=["Health"])
async def get_detailed_health():
    """
    Get detailed health report with all components.
    
    Returns comprehensive health information including:
    - Component-level status (MT5, Redis, filesystem)
    - Performance metrics
    - Recent errors and warnings
    """
    
    try:
        monitor = get_health_monitor()
        report = monitor.get_health_report()
        return report.to_dict()
    except Exception as e:
        logger.error(f"Detailed health failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/health/mt5", response_model=MT5Status, tags=["Health"])
async def get_mt5_status():
    """
    Get MT5 terminal connection status.
    
    Returns connection details, account info, and available symbols count.
    """
    
    try:
        monitor = get_health_monitor()
        summary = monitor.get_health_summary()
        mt5_info = summary.get('components', {}).get('mt5', {})
        
        return MT5Status(
            connected=mt5_info.get('status') == 'healthy',
            error=mt5_info.get('message') if mt5_info.get('status') != 'healthy' else None
        )
    except Exception as e:
        logger.error(f"MT5 status failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/health/metrics", tags=["Health"])
async def get_prometheus_metrics():
    """
    Get Prometheus-compatible metrics.
    
    Returns metrics in Prometheus exposition format for monitoring systems.
    """
    
    try:
        monitor = get_health_monitor()
        report = monitor.get_health_report()
        
        metrics = [
            "# HELP arbitrex_raw_layer_up System up status",
            "# TYPE arbitrex_raw_layer_up gauge",
            "arbitrex_raw_layer_up 1",
            "",
            "# HELP arbitrex_raw_layer_uptime_seconds System uptime",
            "# TYPE arbitrex_raw_layer_uptime_seconds counter",
            f"arbitrex_raw_layer_uptime_seconds {report.uptime_seconds}",
        ]
        
        return Response(content="\n".join(metrics), media_type="text/plain")
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(500, str(e))


# ============================================================================
# Raw Data Endpoints
# ============================================================================

@app.get("/raw/data/{symbol}/{timeframe}", tags=["Raw Data"])
async def get_raw_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = Query(None, description="Start date (UTC)"),
    end_date: Optional[datetime] = Query(None, description="End date (UTC)"),
    limit: Optional[int] = Query(None, description="Max rows (1-10000)", ge=1, le=10000)
):
    """
    Get raw OHLCV data for a symbol and timeframe.
    
    Returns untransformed data as ingested from MT5 with UTC timestamps.
    
    **Parameters:**
    - **symbol**: Trading symbol (e.g., EURUSD, GBPUSD, XAUUSD)
    - **timeframe**: Bar timeframe (1H, 4H, 1D, 1M)
    - **start_date**: Filter bars >= this date
    - **end_date**: Filter bars <= this date
    - **limit**: Maximum number of rows to return
    
    **Returns:**
    - symbol, timeframe, bar count
    - date range
    - OHLCV data array
    """
    
    try:
        df = load_raw_data(symbol, timeframe, start_date, end_date, limit)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": len(df),
            "start_date": df['timestamp_utc'].min().isoformat() if len(df) > 0 else None,
            "end_date": df['timestamp_utc'].max().isoformat() if len(df) > 0 else None,
            "data": df.to_dict(orient='records')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get raw data failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/raw/symbols", tags=["Raw Data"])
async def list_symbols(timeframe: Optional[str] = Query(None, description="Filter by timeframe")):
    """
    List all symbols with available raw data.
    
    **Parameters:**
    - **timeframe**: Optional filter (1H, 4H, 1D, 1M)
    
    **Returns:**
    - List of symbol names
    - Count
    """
    
    try:
        symbols = get_available_symbols(timeframe)
        return {
            "timeframe": timeframe or "all",
            "count": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        logger.error(f"List symbols failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/raw/latest/{symbol}/{timeframe}", tags=["Raw Data"])
async def get_latest_bars(
    symbol: str,
    timeframe: str,
    count: int = Query(100, description="Number of latest bars", ge=1, le=1000)
):
    """
    Get the most recent raw bars for a symbol.
    
    Returns N most recent bars ordered by timestamp descending.
    """
    
    try:
        df = load_raw_data(symbol, timeframe)
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
        logger.error(f"Get latest bars failed: {e}")
        raise HTTPException(500, str(e))


# ============================================================================
# Symbol/Universe Endpoints
# ============================================================================

@app.get("/symbols/universe", tags=["Symbols"])
async def get_trading_universe():
    """
    Get the complete trading universe configuration.
    
    Returns all symbols configured for ingestion with asset class grouping.
    """
    
    return {
        "total_symbols": len(TRADING_UNIVERSE),
        "asset_classes": {
            "FX": [s for s in TRADING_UNIVERSE if "USD" in s and not s.startswith("X")],
            "Metals": [s for s in TRADING_UNIVERSE if s.startswith("XAU") or s.startswith("XAG")],
        },
        "universe": list(TRADING_UNIVERSE),
        "default_timeframes": list(DEFAULT_TIMEFRAMES)
    }


# ============================================================================
# Configuration Endpoints
# ============================================================================

@app.get("/config", tags=["Configuration"])
async def get_config():
    """
    Get current raw layer configuration.
    
    Returns trading universe, timeframes, and system settings.
    """
    
    return {
        "trading_universe": list(TRADING_UNIVERSE),
        "default_timeframes": list(DEFAULT_TIMEFRAMES),
        "api_version": "1.0.0",
        "raw_data_path": "arbitrex/data/raw/ohlcv/fx"
    }


# ============================================================================
# Run with: uvicorn arbitrex.raw_layer.api_v2:app --reload --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
