from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RawConfig:
    """Configuration for the Raw Data Layer.

    - `base_dir` points to the immutable raw data store root. It defaults
      to the project `data/raw` folder adjacent to this package.
    - `bar_close_buffer_minutes` recommends how many minutes to wait after
      a bar close before pulling the bar; ingestion callers should respect it.
    - `tick_logging` enables optional tick capture (diagnostic only).
    - `broker_utc_offset_hours` is the broker server timezone offset from UTC.
      MT5 timestamps are in broker local time; we convert to UTC for storage.
      Set to None for auto-detection from MT5 terminal_info (recommended).
    - `normalize_timestamps` enables UTC normalization (default True).
    """

    base_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    bar_close_buffer_minutes: int = 3
    tick_logging: bool = False
    broker_utc_offset_hours: int | None = None  # None = auto-detect
    normalize_timestamps: bool = True


DEFAULT_CONFIG = RawConfig()

# Timeframes supported by the raw layer
DEFAULT_TIMEFRAMES: List[str] = ["1H", "4H", "1D", "1M"]

# Canonical, versioned trading universe file (JSON). This is the single
# source of truth for which symbols the system ingests and trades. Tools
# should update this file via a controlled maintenance task; ingestion
# always reads from it.
DEFAULT_UNIVERSE_FILE: Path = DEFAULT_CONFIG.base_dir / "metadata" / "source_registry" / "universe_latest.json"

# Optional hard-coded universe (fallback). Prefer updating DEFAULT_UNIVERSE_FILE.
DEFAULT_UNIVERSE: List[str] = []

# Canonical, in-code trading universe (single-source-of-truth). This dictionary
# is authoritative for ingestion and should be version-controlled. If populated
# the runner will use these symbols directly and will NOT perform a full MT5
# discovery unless `--discover` is passed to the runner.
TRADING_UNIVERSE = {
  "FX": [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
    "CADCHF", "CADJPY",
    "CHFJPY",
    "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPUSD",
    "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY"
  ],
  "Metals": [
    "XAUUSD", "XAUEUR", "XAUAUD",
    "XAGUSD", "XAGEUR",
    "XAUCHF", "XAUGBP",
    "XPDUSD", "XPTUSD",
    "XAGAUD"
  ],
  "ETFs_Indices": [
    "MAYM",
    "GDAXIm",
    "MXSHAR",
    "CHINAH"
  ]
}


def detect_broker_utc_offset() -> int:
    """Detect broker timezone offset from MT5 terminal.
    
    Returns offset in hours from UTC (e.g., +2 for GMT+2, -5 for EST).
    Falls back to 0 (UTC) if detection fails.
    """
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return 0
        
        # MT5 doesn't directly expose timezone, but we can infer from server time vs UTC
        import time
        from datetime import datetime, timezone as tz
        
        # Get broker server time (last quote time or current time)
        terminal = mt5.terminal_info()
        if terminal and hasattr(terminal, 'community_balance'):
            # terminal_info doesn't have timezone, try symbol quote time
            symbols = mt5.symbols_get()
            if symbols and len(symbols) > 0:
                symbol_info = mt5.symbol_info_tick(symbols[0].name)
                if symbol_info and hasattr(symbol_info, 'time'):
                    broker_ts = symbol_info.time
                    utc_ts = int(time.time())
                    offset_seconds = broker_ts - utc_ts
                    offset_hours = round(offset_seconds / 3600)
                    return offset_hours
        
        # Fallback: assume GMT+2 (common for European brokers)
        return 2
    except Exception:
        return 0


def broker_to_utc(broker_timestamp: int, offset_hours: int | None = None) -> int:
    """Convert broker local timestamp to UTC.
    
    Args:
        broker_timestamp: Unix epoch seconds in broker timezone
        offset_hours: Broker UTC offset in hours (None = use config default)
    
    Returns:
        Unix epoch seconds in UTC
    """
    if not DEFAULT_CONFIG.normalize_timestamps:
        return broker_timestamp
    
    if offset_hours is None:
        offset_hours = DEFAULT_CONFIG.broker_utc_offset_hours
        if offset_hours is None:
            offset_hours = detect_broker_utc_offset()
    
    return broker_timestamp - (offset_hours * 3600)


def utc_to_broker(utc_timestamp: int, offset_hours: int | None = None) -> int:
    """Convert UTC timestamp to broker local time.
    
    Args:
        utc_timestamp: Unix epoch seconds in UTC
        offset_hours: Broker UTC offset in hours (None = use config default)
    
    Returns:
        Unix epoch seconds in broker timezone
    """
    if not DEFAULT_CONFIG.normalize_timestamps:
        return utc_timestamp
    
    if offset_hours is None:
        offset_hours = DEFAULT_CONFIG.broker_utc_offset_hours
        if offset_hours is None:
            offset_hours = detect_broker_utc_offset()
    
    return utc_timestamp + (offset_hours * 3600)


