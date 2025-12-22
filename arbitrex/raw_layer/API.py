"""
Raw Layer API Reference for Frontend Integration

This file documents all available APIs, endpoints, data formats, and integration
patterns for the Arbitrex Raw Data Layer. Use this as a reference when building
frontend applications, data consumers, or downstream analytics layers.

IMPORTANT: All timestamps are in UTC (Unix epoch seconds) unless specified otherwise.
See TIME_NORMALIZATION.md for details.

Author: Arbitrex Team
Last Updated: 2025-12-22
"""

# =============================================================================
# WEBSOCKET API - REAL-TIME TICK STREAMING
# =============================================================================

WEBSOCKET_BASE_URL = "ws://localhost:8765"

WEBSOCKET_ENDPOINTS = {
    "tick_stream": {
        "url": f"{WEBSOCKET_BASE_URL}/ws",
        "protocol": "WebSocket",
        "description": "Real-time tick data stream with sub-second latency",
        "connection_type": "persistent",
        "message_format": "JSON",
    }
}


# -----------------------------------------------------------------------------
# WebSocket Tick Stream Protocol
# -----------------------------------------------------------------------------

class WebSocketTickStreamAPI:
    """
    Real-time tick streaming via WebSocket connection.
    
    Connection: ws://localhost:8765/ws
    
    Message Flow:
    1. Client connects
    2. Client sends subscribe message with symbol list
    3. Server sends tick updates as they arrive
    4. Client can send unsubscribe or additional subscribe messages
    5. Client closes connection when done
    """
    
    # Message from Client -> Server
    CLIENT_MESSAGES = {
        "subscribe": {
            "description": "Subscribe to tick updates for specified symbols",
            "format": {
                "action": "subscribe",
                "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]
            },
            "example": '{"action": "subscribe", "symbols": ["EURUSD", "GBPUSD"]}'
        },
        
        "unsubscribe": {
            "description": "Unsubscribe from specific symbols",
            "format": {
                "action": "unsubscribe",
                "symbols": ["EURUSD"]
            },
            "example": '{"action": "unsubscribe", "symbols": ["EURUSD"]}'
        }
    }
    
    # Messages from Server -> Client
    SERVER_MESSAGES = {
        "tick": {
            "description": "Real-time tick update",
            "format": {
                "symbol": "str",           # Trading symbol (e.g., "EURUSD")
                "ts": "int",               # Timestamp in UTC (Unix epoch seconds) - PRIMARY
                "ts_broker": "int",        # Broker local timestamp (for audit only)
                "bid": "float | None",     # Bid price (buy side)
                "ask": "float | None",     # Ask price (sell side)
                "last": "float | None",    # Last traded price (may be None)
                "volume": "int"            # Tick volume
            },
            "example": {
                "symbol": "EURUSD",
                "ts": 1703264401,
                "ts_broker": 1703271601,
                "bid": 1.12340,
                "ask": 1.12350,
                "last": 1.12345,
                "volume": 100
            }
        },
        
        "subscribed": {
            "description": "Confirmation of subscription",
            "format": {
                "status": "subscribed",
                "symbols": ["EURUSD", "GBPUSD"]
            }
        },
        
        "error": {
            "description": "Error message",
            "format": {
                "error": "str"
            }
        }
    }


# -----------------------------------------------------------------------------
# WebSocket Client Example (Python)
# -----------------------------------------------------------------------------

WEBSOCKET_CLIENT_EXAMPLE_PYTHON = """
import asyncio
import websockets
import json

async def tick_stream_client():
    uri = "ws://localhost:8765/ws"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to symbols
        subscribe_msg = {
            "action": "subscribe",
            "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]
        }
        await websocket.send(json.dumps(subscribe_msg))
        print(f"Subscribed to: {subscribe_msg['symbols']}")
        
        # Receive and process ticks
        async for message in websocket:
            tick = json.loads(message)
            
            # Use ts (UTC timestamp) for all analysis
            symbol = tick.get("symbol")
            ts_utc = tick.get("ts")
            bid = tick.get("bid")
            ask = tick.get("ask")
            
            if bid and ask:
                spread = ask - bid
                mid = (bid + ask) / 2
                print(f"{symbol}: mid={mid:.5f}, spread={spread:.5f}, ts={ts_utc}")
            
            # Unsubscribe example (optional)
            # await websocket.send(json.dumps({
            #     "action": "unsubscribe",
            #     "symbols": ["XAUUSD"]
            # }))

if __name__ == "__main__":
    asyncio.run(tick_stream_client())
"""


# -----------------------------------------------------------------------------
# WebSocket Client Example (JavaScript/TypeScript)
# -----------------------------------------------------------------------------

WEBSOCKET_CLIENT_EXAMPLE_JS = """
// JavaScript/TypeScript WebSocket client example

class TickStreamClient {
    constructor(url = 'ws://localhost:8765/ws') {
        this.url = url;
        this.ws = null;
        this.handlers = {
            onTick: null,
            onError: null,
            onClose: null
        };
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                const tick = JSON.parse(event.data);
                if (this.handlers.onTick) {
                    this.handlers.onTick(tick);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (this.handlers.onError) {
                    this.handlers.onError(error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket closed');
                if (this.handlers.onClose) {
                    this.handlers.onClose();
                }
            };
        });
    }
    
    subscribe(symbols) {
        const message = {
            action: 'subscribe',
            symbols: symbols
        };
        this.ws.send(JSON.stringify(message));
    }
    
    unsubscribe(symbols) {
        const message = {
            action: 'unsubscribe',
            symbols: symbols
        };
        this.ws.send(JSON.stringify(message));
    }
    
    onTick(callback) {
        this.handlers.onTick = callback;
    }
    
    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage example
async function main() {
    const client = new TickStreamClient();
    
    client.onTick((tick) => {
        // Use tick.ts (UTC timestamp) for all analysis
        const { symbol, ts, ts_broker, bid, ask, volume } = tick;
        
        if (bid && ask) {
            const spread = ask - bid;
            const mid = (bid + ask) / 2;
            console.log(`${symbol}: mid=${mid.toFixed(5)}, spread=${spread.toFixed(5)}, ts=${ts}`);
        }
    });
    
    await client.connect();
    client.subscribe(['EURUSD', 'GBPUSD', 'XAUUSD']);
}

main().catch(console.error);
"""


# =============================================================================
# FILE-BASED DATA ACCESS (DIRECT)
# =============================================================================

FILE_API = {
    "base_path": "arbitrex/data/raw",
    
    "ohlcv": {
        "location": "arbitrex/data/raw/ohlcv/fx/{SYMBOL}/{TIMEFRAME}/{YYYY-MM-DD}.csv",
        "format": "CSV",
        "headers": ["timestamp_utc", "timestamp_broker", "open", "high", "low", "close", "volume"],
        "description": "Historical OHLCV bars grouped by UTC date",
        "example_path": "arbitrex/data/raw/ohlcv/fx/EURUSD/1H/2025-12-22.csv",
        "example_row": "1703246400,1703253600,1.12345,1.12450,1.12300,1.12400,1523",
        "notes": [
            "Files are immutable (never overwritten)",
            "Use timestamp_utc column for all analysis",
            "timestamp_broker is for audit/reconciliation only",
            "Files grouped by UTC date, not broker date"
        ]
    },
    
    "ticks": {
        "location": "arbitrex/data/raw/ticks/fx/{SYMBOL}/{YYYY-MM-DD}.csv",
        "format": "CSV",
        "headers": ["timestamp_utc", "timestamp_broker", "bid", "ask", "last", "volume"],
        "description": "Raw tick data for diagnostic purposes",
        "example_path": "arbitrex/data/raw/ticks/fx/EURUSD/2025-12-22.csv",
        "example_row": "1703264400,1703271600,1.12340,1.12350,1.12345,100",
        "notes": [
            "Tick capture is optional/diagnostic only",
            "Not intended for production signal generation",
            "Use timestamp_utc for chronological ordering"
        ]
    },
    
    "metadata": {
        "location": "arbitrex/data/raw/metadata/ingestion_logs/{CYCLE_ID}.json",
        "format": "JSON",
        "description": "Per-cycle ingestion metadata with timezone info",
        "example_path": "arbitrex/data/raw/metadata/ingestion_logs/2025-12-22T094532Z_EURUSD_1H.json",
        "schema": {
            "cycle_id": "str",
            "symbol": "str",
            "timeframe": "str",
            "broker_utc_offset_hours": "int",
            "timestamps_normalized": "bool",
            "bars_received": "int",
            "status": "str",
            "files": ["str"],
            "written_at": "str (ISO8601 UTC)"
        }
    },
    
    "universe": {
        "location": "arbitrex/data/raw/metadata/source_registry/universe_latest.json",
        "format": "JSON",
        "description": "Current trading universe (canonical)",
        "schema": {
            "generated_at_utc": "str (ISO8601)",
            "symbols": [
                {
                    "symbol": "str",
                    "symbol_raw": "str",
                    "market": "str",
                    "digits": "int | None",
                    "currency_base": "str | None",
                    "currency_profit": "str | None",
                    "contract_size": "float | None",
                    "tradeable": "bool",
                    "raw": "dict"
                }
            ]
        }
    }
}


# -----------------------------------------------------------------------------
# File Access Example (Python)
# -----------------------------------------------------------------------------

FILE_ACCESS_EXAMPLE_PYTHON = """
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def read_ohlcv(symbol, timeframe, date):
    '''
    Read OHLCV data for a specific symbol, timeframe, and date.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1H", "4H", "1D")
        date: Date string "YYYY-MM-DD" in UTC
    
    Returns:
        DataFrame with UTC-indexed timestamps
    '''
    path = Path(f"arbitrex/data/raw/ohlcv/fx/{symbol}/{timeframe}/{date}.csv")
    
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} {timeframe} on {date}")
    
    df = pd.read_csv(path)
    
    # CRITICAL: Always use timestamp_utc for analysis
    df['datetime'] = pd.to_datetime(df['timestamp_utc'], unit='s', utc=True)
    df = df.set_index('datetime')
    
    return df


def read_ticks(symbol, date):
    '''Read tick data for a specific symbol and UTC date.'''
    path = Path(f"arbitrex/data/raw/ticks/fx/{symbol}/{date}.csv")
    
    if not path.exists():
        raise FileNotFoundError(f"No tick data for {symbol} on {date}")
    
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['timestamp_utc'], unit='s', utc=True)
    df = df.set_index('datetime')
    
    return df


def read_universe():
    '''Read the current trading universe.'''
    import json
    path = Path("arbitrex/data/raw/metadata/source_registry/universe_latest.json")
    
    with open(path) as f:
        universe = json.load(f)
    
    return pd.DataFrame(universe['symbols'])


# Usage examples
if __name__ == "__main__":
    # Read OHLCV data
    df = read_ohlcv("EURUSD", "1H", "2025-12-22")
    print(df.head())
    
    # Calculate features using UTC timestamps
    df['hour_utc'] = df.index.hour
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Read trading universe
    universe = read_universe()
    print(f"Trading {len(universe)} symbols")
"""


# =============================================================================
# HEALTH MONITORING API (IMPLEMENTED)
# =============================================================================

HEALTH_API = {
    "base_url": "http://localhost:8766",
    "description": "System health monitoring and diagnostics",
    
    "endpoints": {
        "/health": {
            "method": "GET",
            "description": "Quick health check (suitable for load balancers)",
            "returns": "200 OK if healthy, 503 if degraded/critical",
            "response_example": {
                "status": "healthy",
                "timestamp": 1703264401,
                "uptime": 3600.5,
                "components": {
                    "mt5": {"status": "healthy", "message": "All MT5 sessions connected"},
                    "tick_collection": {"status": "healthy", "message": "Collecting 120 ticks/min"},
                    "queue": {"status": "healthy", "message": "Queue healthy: 42 messages"},
                    "filesystem": {"status": "healthy", "message": "Disk healthy: 250GB free"},
                    "data_quality": {"status": "healthy", "message": "Good data quality: 100% success"},
                    "timezone": {"status": "healthy", "message": "UTC normalization enabled"}
                },
                "warnings": 0,
                "errors": 0
            }
        },
        
        "/health/detailed": {
            "method": "GET",
            "description": "Comprehensive health report with full diagnostics",
            "response_includes": [
                "Component-level metrics with thresholds",
                "Tick collection rates per symbol",
                "Queue depth and processing lag",
                "Filesystem usage and writability",
                "Recent errors and warnings",
                "Uptime and performance metrics"
            ]
        },
        
        "/health/metrics": {
            "method": "GET",
            "description": "Prometheus-compatible metrics endpoint",
            "content_type": "text/plain; version=0.0.4",
            "metrics": [
                "arbitrex_raw_layer_up",
                "arbitrex_raw_layer_uptime_seconds",
                "arbitrex_raw_layer_health_status{component}",
                "arbitrex_raw_layer_ticks_total",
                "arbitrex_raw_layer_symbols_tracked",
                "arbitrex_raw_layer_errors_total",
                "arbitrex_raw_layer_warnings_total"
            ]
        },
        
        "/health/components/{component}": {
            "method": "GET",
            "description": "Get health details for specific component",
            "path_params": {
                "component": "mt5 | tick_collection | queue | filesystem | data_quality | timezone"
            },
            "response_example": {
                "component": "mt5",
                "status": "healthy",
                "value": {
                    "connected": 1,
                    "total": 1,
                    "initialized": 1
                },
                "threshold": None,
                "message": "All MT5 sessions connected and initialized",
                "last_updated": 1703264401
            }
        }
    },
    
    "cli_tool": {
        "command": "python -m arbitrex.raw_layer.health_cli",
        "options": [
            "--detailed      Show detailed health report",
            "--json          Output in JSON format",
            "--component     Show specific component (mt5, tick_collection, etc.)",
            "--watch         Watch mode: continuously refresh",
            "--interval N    Refresh interval for watch mode (default: 5s)"
        ],
        "examples": [
            "python -m arbitrex.raw_layer.health_cli",
            "python -m arbitrex.raw_layer.health_cli --detailed",
            "python -m arbitrex.raw_layer.health_cli --watch --interval 3",
            "python -m arbitrex.raw_layer.health_cli --component mt5 --json"
        ]
    },
    
    "start_server": {
        "command": "python -m arbitrex.raw_layer.health_api",
        "description": "Start standalone health API server",
        "port": 8766,
        "note": "Health monitor is also automatically integrated when running the streaming stack"
    }
}


# =============================================================================
# REST API (FUTURE/PROPOSED)
# =============================================================================

# Note: Additional REST API endpoints are not yet implemented. These are proposed
# for future development to enable easier frontend integration.

PROPOSED_REST_API = {
    "base_url": "http://localhost:8000/api/v1",
    
    "endpoints": {
        "/health": {
            "method": "GET",
            "description": "Health check and system status",
            "response": {
                "status": "healthy",
                "mt5_connected": True,
                "tick_collector_running": True,
                "broker_utc_offset": 2,
                "symbols_tracked": 30,
                "ticks_collected_last_minute": 1250
            }
        },
        
        "/symbols": {
            "method": "GET",
            "description": "List all available symbols in trading universe",
            "query_params": {
                "market": "FX | Metals | ETFs_Indices (optional filter)"
            },
            "response": [
                {
                    "symbol": "EURUSD",
                    "market": "FX",
                    "digits": 5,
                    "tradeable": True
                }
            ]
        },
        
        "/ohlcv/{symbol}": {
            "method": "GET",
            "description": "Get OHLCV bars for a symbol",
            "path_params": {
                "symbol": "Trading symbol (e.g., EURUSD)"
            },
            "query_params": {
                "timeframe": "1H | 4H | 1D | 1M",
                "start": "Unix timestamp UTC or ISO8601",
                "end": "Unix timestamp UTC or ISO8601",
                "limit": "Max number of bars (default 1000)"
            },
            "response": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "timezone": "UTC",
                "broker_offset_hours": 2,
                "bars": [
                    {
                        "timestamp_utc": 1703246400,
                        "timestamp_broker": 1703253600,
                        "open": 1.12345,
                        "high": 1.12450,
                        "low": 1.12300,
                        "close": 1.12400,
                        "volume": 1523
                    }
                ]
            }
        },
        
        "/ticks/{symbol}": {
            "method": "GET",
            "description": "Get recent ticks for a symbol",
            "path_params": {
                "symbol": "Trading symbol"
            },
            "query_params": {
                "limit": "Max number of ticks (default 100, max 10000)"
            },
            "response": {
                "symbol": "EURUSD",
                "timezone": "UTC",
                "ticks": [
                    {
                        "timestamp_utc": 1703264401,
                        "timestamp_broker": 1703271601,
                        "bid": 1.12340,
                        "ask": 1.12350,
                        "last": 1.12345,
                        "volume": 100
                    }
                ]
            }
        },
        
        "/metadata/ingestion/{cycle_id}": {
            "method": "GET",
            "description": "Get ingestion cycle metadata",
            "response": {
                "cycle_id": "2025-12-22T094532Z_EURUSD_1H",
                "symbol": "EURUSD",
                "timeframe": "1H",
                "broker_utc_offset_hours": 2,
                "timestamps_normalized": True,
                "status": "SUCCESS",
                "bars_received": 240
            }
        }
    }
}


# -----------------------------------------------------------------------------
# REST API Client Example (Python) - FOR FUTURE USE
# -----------------------------------------------------------------------------

REST_CLIENT_EXAMPLE_PYTHON = """
import requests
from datetime import datetime, timezone

class ArbitrexRawAPI:
    '''Client for Arbitrex Raw Layer REST API (future implementation).'''
    
    def __init__(self, base_url='http://localhost:8000/api/v1'):
        self.base_url = base_url
    
    def health(self):
        '''Check system health.'''
        response = requests.get(f'{self.base_url}/health')
        response.raise_for_status()
        return response.json()
    
    def get_symbols(self, market=None):
        '''Get trading universe.'''
        params = {'market': market} if market else {}
        response = requests.get(f'{self.base_url}/symbols', params=params)
        response.raise_for_status()
        return response.json()
    
    def get_ohlcv(self, symbol, timeframe='1H', start=None, end=None, limit=1000):
        '''Get OHLCV bars.'''
        params = {
            'timeframe': timeframe,
            'limit': limit
        }
        if start:
            params['start'] = start if isinstance(start, int) else int(datetime.fromisoformat(start).timestamp())
        if end:
            params['end'] = end if isinstance(end, int) else int(datetime.fromisoformat(end).timestamp())
        
        response = requests.get(f'{self.base_url}/ohlcv/{symbol}', params=params)
        response.raise_for_status()
        return response.json()
    
    def get_ticks(self, symbol, limit=100):
        '''Get recent ticks.'''
        params = {'limit': limit}
        response = requests.get(f'{self.base_url}/ticks/{symbol}', params=params)
        response.raise_for_status()
        return response.json()

# Usage (once REST API is implemented)
# api = ArbitrexRawAPI()
# print(api.health())
# bars = api.get_ohlcv('EURUSD', timeframe='1H', limit=100)
"""


# =============================================================================
# DATA FORMATS & CONVENTIONS
# =============================================================================

DATA_CONVENTIONS = {
    "timestamps": {
        "storage": "Unix epoch seconds (integer)",
        "timezone": "UTC (primary), broker local (audit)",
        "columns": {
            "timestamp_utc": "Use this for ALL analysis, features, backtesting",
            "timestamp_broker": "Use ONLY for broker trade reconciliation"
        },
        "conversion": "Use pandas: pd.to_datetime(df['timestamp_utc'], unit='s', utc=True)"
    },
    
    "prices": {
        "format": "float (broker precision, typically 5 decimals for FX)",
        "fields": {
            "open": "Opening price of bar",
            "high": "Highest price during bar",
            "low": "Lowest price during bar",
            "close": "Closing price of bar",
            "bid": "Bid price (buy side)",
            "ask": "Ask price (sell side)",
            "last": "Last traded price (may be None)"
        }
    },
    
    "volume": {
        "format": "integer",
        "type": "tick_volume",
        "description": "Number of price changes (not notional volume)",
        "note": "FX typically reports tick volume, not actual traded volume"
    },
    
    "symbols": {
        "format": "Uppercase alphanumeric (e.g., EURUSD, XAUUSD)",
        "normalization": "Stripped of broker suffixes (.r, .m, etc.)",
        "length": "Typically 6 characters for FX pairs"
    },
    
    "timeframes": {
        "supported": ["1H", "4H", "1D", "1M"],
        "format": "String (e.g., '1H' for 1-hour bars)",
        "mapping": {
            "1H": "1 hour",
            "4H": "4 hours",
            "1D": "1 day",
            "1M": "1 month"
        }
    }
}


# =============================================================================
# IMPORTANT NOTES FOR FRONTEND DEVELOPERS
# =============================================================================

FRONTEND_INTEGRATION_NOTES = """
CRITICAL GUIDELINES FOR FRONTEND INTEGRATION:

1. TIMESTAMP HANDLING
   - Always use `ts` or `timestamp_utc` field for display and analysis
   - Ignore `ts_broker` or `timestamp_broker` unless doing broker reconciliation
   - Convert to local timezone for display ONLY: new Date(ts * 1000)
   - Do NOT use broker timestamps for charting, calculations, or comparisons

2. WEBSOCKET CONNECTION
   - WebSocket URL: ws://localhost:8765/ws
   - Always send subscribe message after connection
   - Handle reconnection logic (exponential backoff recommended)
   - Unsubscribe before closing connection (clean shutdown)
   - Process messages in order received (no buffering needed)

3. DATA DISPLAY
   - Use timestamp_utc for X-axis on charts
   - Display prices with appropriate precision (5 decimals for FX)
   - Calculate spread: spread = ask - bid
   - Calculate mid: mid = (bid + ask) / 2
   - Show volume as integer (tick count)

4. ERROR HANDLING
   - WebSocket connection failures: retry with exponential backoff
   - Missing data fields: check for null/undefined before using
   - Invalid symbols: subscribe may fail silently, monitor for ticks
   - Connection timeout: implement ping/pong or heartbeat

5. PERFORMANCE
   - Tick rate: expect 10-100 ticks/second per symbol during active hours
   - Buffer ticks client-side if rendering is expensive (e.g., 100ms batches)
   - Use requestAnimationFrame for smooth chart updates
   - Consider WebWorker for tick processing to avoid blocking UI

6. TIMEZONE DISPLAY
   - Show UTC time prominently for consistency
   - Optionally show user's local time as secondary
   - Display broker timezone offset in metadata/settings
   - Example: "2025-12-22 09:45:32 UTC (Broker: GMT+2)"

7. TESTING
   - Use demo.html in arbitrex/stream/ as reference
   - Test with multiple symbols simultaneously
   - Test reconnection behavior
   - Test subscribe/unsubscribe flow
   - Monitor for memory leaks (unsubscribe old symbols)
"""


# =============================================================================
# RUNNING THE SERVER
# =============================================================================

SERVER_COMMANDS = {
    "start_streaming_server": {
        "command": "python -m arbitrex.scripts.run_streaming_stack",
        "description": "Start WebSocket server with MT5 tick collector",
        "port": 8765,
        "requirements": ["MT5 connection", "Redis (optional)"]
    },
    
    "run_ingestion": {
        "command": "python -m arbitrex.raw_layer.runner --timeframes 1H,4H,1D",
        "description": "Run one-time OHLCV ingestion",
        "requirements": ["MT5 connection", ".env with credentials"]
    },
    
    "export_universe": {
        "command": "python -m arbitrex.raw_layer.runner --universe-only",
        "description": "Export trading universe to JSON",
        "output": "arbitrex/data/raw/metadata/source_registry/universe_latest.json"
    }
}


# =============================================================================
# COMPLETE FRONTEND INTEGRATION CHECKLIST
# =============================================================================

INTEGRATION_CHECKLIST = """
□ WebSocket Client Implementation
  □ Connection management (connect, reconnect, disconnect)
  □ Subscribe/unsubscribe messaging
  □ Tick data parsing and validation
  □ Error handling and logging
  □ Connection status indicator in UI

□ Data Display
  □ Real-time tick display (symbol, price, timestamp)
  □ Live chart rendering (candlesticks or line)
  □ Bid/ask spread calculation and display
  □ Volume indicator
  □ Timestamp display (UTC primary, local secondary)

□ Performance Optimization
  □ Tick batching for rendering (if needed)
  □ Chart update throttling (requestAnimationFrame)
  □ Memory management (limit history buffer)
  □ WebWorker for heavy processing (optional)

□ User Experience
  □ Symbol selection UI (dropdown/search)
  □ Connection status indicator (connected, disconnected, error)
  □ Loading states during connection
  □ Error messages for failed operations
  □ Timezone display preference (UTC/local)

□ Testing
  □ Multiple symbol subscriptions
  □ Connection loss and recovery
  □ High tick rate scenarios
  □ Browser compatibility (Chrome, Firefox, Safari)
  □ Mobile responsiveness (if applicable)
"""


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ARBITREX RAW LAYER API REFERENCE")
    print("=" * 70)
    print()
    print("HEALTH MONITORING:")
    print(f"  URL: {HEALTH_API['base_url']}/health")
    print("  Detailed: /health/detailed")
    print("  Metrics: /health/metrics (Prometheus)")
    print("  CLI: python -m arbitrex.raw_layer.health_cli")
    print()
    print("WEBSOCKET STREAMING:")
    print(f"  URL: {WEBSOCKET_BASE_URL}/ws")
    print("  Protocol: Subscribe to symbols, receive real-time ticks")
    print()
    print("FILE ACCESS:")
    print(f"  OHLCV: {FILE_API['ohlcv']['location']}")
    print(f"  Ticks: {FILE_API['ticks']['location']}")
    print(f"  Metadata: {FILE_API['metadata']['location']}")
    print()
    print("DATA CONVENTIONS:")
    print("  - Use timestamp_utc for ALL analysis")
    print("  - Use timestamp_broker ONLY for broker reconciliation")
    print("  - All timestamps in Unix epoch seconds")
    print("  - All files immutable (append-only)")
    print()
    print("EXAMPLE CLIENTS:")
    print("  - Python: See WEBSOCKET_CLIENT_EXAMPLE_PYTHON")
    print("  - JavaScript: See WEBSOCKET_CLIENT_EXAMPLE_JS")
    print("  - Demo: arbitrex/stream/demo.html")
    print()
    print("DOCUMENTATION:")
    print("  - Time Normalization: TIME_NORMALIZATION.md")
    print("  - Raw Layer Guide: arbitrex/raw_layer/README.md")
    print()
    print("=" * 70)
