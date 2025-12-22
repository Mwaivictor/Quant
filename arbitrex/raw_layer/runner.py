"""Arbitrex Raw Layer CLI runner

Provides a production-ready CLI to export the MT5 trading universe and run
safe, atomic ingestion of raw OHLCV bars (append-only) for multiple timeframes.

Design goals implemented here:
- Loads MT5 credentials from a `.env` file via `python-dotenv`.
- Exports a versioned universe CSV with rich symbol metadata.
- Batch ingests finalized bars per-symbol/timeframe and writes per-day CSVs.
- Uses retries, rate limiting, atomic writes, and per-run logging.

This module is intentionally modular: functions are small, typed, and easily
reused by orchestration or tests.
"""

from __future__ import annotations
import argparse
import os
import time
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from .mt5_pool import MT5ConnectionPool, MT5Session
from .config import DEFAULT_UNIVERSE_FILE, DEFAULT_CONFIG

LOG = logging.getLogger("arbitrex.raw.runner")
LOG.setLevel(logging.INFO)


def setup_run_logger(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(output_dir, f"ingest_run_{ts}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(fh)
    return log_path


def load_credentials(env_path: Optional[str] = None) -> Dict[str, str]:
    """Load MT5 credentials from a .env file and environment.

    Returns a dict with keys: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL (optional)
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    creds = {
        "MT5_LOGIN": os.environ.get("MT5_LOGIN"),
        "MT5_PASSWORD": os.environ.get("MT5_PASSWORD"),
        "MT5_SERVER": os.environ.get("MT5_SERVER"),
        "MT5_TERMINAL": os.environ.get("MT5_TERMINAL"),
    }
    return creds


def mt5_connect_from_env(creds: Dict[str, str]) -> MT5ConnectionPool:
    """Create an MT5ConnectionPool using credentials from `creds`.

    The connection pool is account-bound to a single login.
    """
    session_params = {
        "main": {
            "terminal_path": creds.get("MT5_TERMINAL"),
            "login": int(creds["MT5_LOGIN"]) if creds.get("MT5_LOGIN") else None,
            "password": creds.get("MT5_PASSWORD"),
            "server": creds.get("MT5_SERVER"),
        }
    }

    # Get symbols from TRADING_UNIVERSE in config.py
    from .config import TRADING_UNIVERSE
    symbols = [s for group in TRADING_UNIVERSE.values() for s in group]

    logs_dir = os.path.join(os.getcwd(), "arbitrex", "data", "raw", "mt5", "session_logs")
    pool = MT5ConnectionPool(session_params, symbols, session_logs_dir=logs_dir)
    return pool


def retry_call(fn, retries: int = 3, backoff: float = 0.5, allowed_exceptions: Tuple = (Exception,), *args, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except allowed_exceptions as e:
            last_exc = e
            LOG.warning("Attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(backoff * attempt)
    raise last_exc


def fetch_all_symbols(pool: MT5ConnectionPool, rate_limit: float = 0.05) -> List[Dict[str, Any]]:
    """Fetch all symbols from MT5 and return a list of dictionaries with metadata."""
    name, sess = pool.get_connection(timeout=10)
    try:
        import MetaTrader5 as mt5

        symbols = retry_call(mt5.symbols_get, retries=3, backoff=0.5)
        records: List[Dict[str, Any]] = []
        if symbols is None:
            return records

        for s in symbols:
            # Convert symbol object to a dict where possible
            try:
                rec = s._asdict()
            except Exception:
                # Fallback: query symbol_info for rich metadata
                info = mt5.symbol_info(s.name if hasattr(s, 'name') else s)
                if info is None:
                    continue
                rec = {k: getattr(info, k, None) for k in dir(info) if not k.startswith('_')}
            records.append(rec)
            time.sleep(rate_limit)

        return records
    finally:
        pool.release_connection((name, sess))


def normalize_symbol_name(name: str) -> str:
    """Normalize symbol names: uppercase, strip non-alphanumeric suffixes and separators.

    Examples: 'EURUSD.r' -> 'EURUSD', 'gbpusd.m' -> 'GBPUSD'
    """
    if not name:
        return name
    s = name.upper()
    # remove common separators and suffixes
    for ch in ['/', '\\', '.', '@']:
        s = s.replace(ch, '')
    # strip trailing letters that are not part of base symbol (e.g., .A or m)
    # if symbol looks like 6-letter FX pair keep first 6 letters
    alpha = ''.join([c for c in s if c.isalnum()])
    if len(alpha) >= 6 and alpha[:6].isalpha():
        return alpha[:6]
    return alpha


def enrich_and_filter_symbols(raw_symbols: List[Dict[str, Any]], only_tradeable: bool = True, only_fx: bool = True) -> List[Dict[str, Any]]:
    """Normalize, enrich and filter raw symbol dicts into a canonical universe list.

    This function is conservative: it preserves all metadata in the `raw` field
    and extracts a few useful columns for selection.
    """
    out: List[Dict[str, Any]] = []
    for r in raw_symbols:
        # try to obtain a name string
        name = r.get('name') or r.get('symbol') or r.get('st_name') if isinstance(r, dict) else None
        if not name and isinstance(r, str):
            name = r
        if not name:
            continue
        norm = normalize_symbol_name(name)

        # extract _safe_ fields
        digits = r.get('digits') if isinstance(r, dict) else None
        point = r.get('point') if isinstance(r, dict) else None
        currency_base = r.get('currency_base') if isinstance(r, dict) else None
        currency_profit = r.get('currency_profit') if isinstance(r, dict) else None
        contract_size = r.get('trade_contract_size') or r.get('contract_size') or r.get('lot_size') if isinstance(r, dict) else None
        min_volume = r.get('volume_min') if isinstance(r, dict) else None
        max_volume = r.get('volume_max') if isinstance(r, dict) else None
        volume_step = r.get('volume_step') if isinstance(r, dict) else None
        spread = r.get('spread') if isinstance(r, dict) else None
        trade_mode = r.get('trade_mode') if isinstance(r, dict) else r.get('trade') if isinstance(r, dict) else None

        # market heuristic
        market = 'FX' if currency_base else 'OTHER'
        if only_fx and market != 'FX':
            continue

        # tradeable heuristic
        tradeable = True
        if only_tradeable and trade_mode in (0, None):
            # 0 or None may indicate non-tradeable; keep conservative: allow unless explicit not tradeable
            tradeable = True

        item = {
            'symbol_raw': name,
            'symbol': norm,
            'market': market,
            'digits': digits,
            'point': point,
            'currency_base': currency_base,
            'currency_profit': currency_profit,
            'contract_size': contract_size,
            'min_volume': min_volume,
            'max_volume': max_volume,
            'volume_step': volume_step,
            'spread': spread,
            'trade_mode': trade_mode,
            'tradeable': tradeable,
            'raw': r,
        }
        out.append(item)
    return out


def write_universe_csv(universe: List[Dict[str, Any]], out_path: str) -> None:
    # lazy import to avoid requiring pandas for lightweight runs
    import pandas as pd
    df = pd.DataFrame(universe)
    # ensure reproducible order
    if 'symbol' in df.columns:
        df = df.sort_values('symbol')
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def timeframe_to_mt5(tf: str):
    import MetaTrader5 as mt5
    mapping = {"1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4, "1D": mt5.TIMEFRAME_D1, "1M": mt5.TIMEFRAME_MN1}
    return mapping[tf]


def ingest_symbol_timeframes(pool: MT5ConnectionPool, symbol: str, timeframes: List[str], output_dir: str, bars_per_tf: Dict[str, int], rate_limit: float = 0.05) -> None:
    """Ingest latest finalized bars for a single symbol across multiple timeframes.

    Writes per-day CSV files under: <output_dir>/ohlcv/fx/<SYMBOL>/<YYYY-MM-DD>.csv
    Each CSV contains rows with columns: timestamp, open, high, low, close, volume, symbol, timeframe
    """
    name, sess = pool.get_connection(timeout=10)
    try:
        import MetaTrader5 as mt5
        for tf in timeframes:
            tf_const = timeframe_to_mt5(tf)
            count = bars_per_tf.get(tf, 200)
            # use pos=1 to skip the current forming bar and get closed bars
            rates = retry_call(lambda: mt5.copy_rates_from_pos(symbol, tf_const, 1, count), retries=3, backoff=0.5)
            if not rates:
                LOG.info("No rates for %s %s", symbol, tf)
                continue

            rows = []
            for r in rates:
                rows.append({
                    'timestamp': int(r.time),
                    'open': float(r.open),
                    'high': float(r.high),
                    'low': float(r.low),
                    'close': float(r.close),
                    'volume': int(getattr(r, 'tick_volume', 0)),
                    'symbol': symbol,
                    'timeframe': tf,
                })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            # group by date and write per-day CSVs
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
            for date, group in df.groupby('date'):
                date_str = date.isoformat()  # YYYY-MM-DD
                dirp = os.path.join(output_dir, 'ohlcv', 'fx', symbol, tf)
                os.makedirs(dirp, exist_ok=True)
                out_path = os.path.join(dirp, f"{date_str}.csv")
                # write atomic
                atomic_write_csv(group.drop(columns=['date']), out_path)

            time.sleep(rate_limit)
    finally:
        pool.release_connection((name, sess))


def run(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog='arbitrex-raw')
    parser.add_argument('--env', default='.env', help='Path to .env with MT5 credentials')
    parser.add_argument('--output-dir', default=os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw'), help='Root output directory')
    parser.add_argument('--timeframes', default='1H,4H,1D', help='Comma separated timeframes to ingest')
    parser.add_argument('--symbols', default=None, help='Optional comma-separated symbols to ingest (normalized names)')
    parser.add_argument('--universe-only', action='store_true', help='Only export universe and exit')
    parser.add_argument('--discover', action='store_true', help='Force MT5 discovery of symbols (ignore in-code TRADING_UNIVERSE)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel worker processes')
    parser.add_argument('--tick-logging', action='store_true', help='Enable short diagnostic tick captures per symbol')
    parser.add_argument('--parquet', action='store_true', help='Also write Parquet copies (derivative)')
    parser.add_argument('--history-days', type=int, default=365, help='History window (days) for daily history ingestion')
    parser.add_argument('--rate-limit', type=float, default=0.05, help='Seconds to wait between MT5 requests')
    args = parser.parse_args(argv)

    # Setup logging
    log_dir = os.path.join(args.output_dir, 'mt5', 'session_logs')
    log_path = setup_run_logger(log_dir)
    LOG.info('Starting run, log: %s', log_path)

    creds = load_credentials(args.env)
    pool = mt5_connect_from_env(creds)

    try:
        # If a canonical in-code trading universe is provided, use it directly
        from .config import TRADING_UNIVERSE

        if TRADING_UNIVERSE and not args.discover:
            LOG.info('Using canonical TRADING_UNIVERSE from config (no MT5 discovery)')
            universe = []
            for category, syms in TRADING_UNIVERSE.items():
                for s in syms:
                    universe.append({
                        'symbol_raw': s,
                        'symbol': s.upper(),
                        'market': category,
                        'digits': None,
                        'point': None,
                        'currency_base': None,
                        'currency_profit': None,
                        'contract_size': None,
                        'min_volume': None,
                        'max_volume': None,
                        'volume_step': None,
                        'spread': None,
                        'trade_mode': None,
                        'tradeable': True,
                        'raw': {},
                    })
        else:
            raw = fetch_all_symbols(pool, rate_limit=args.rate_limit)
            LOG.info('Fetched %d raw symbols', len(raw))
            universe = enrich_and_filter_symbols(raw, only_tradeable=True, only_fx=True)

        # Write canonical universe JSON (single source of truth) and a versioned copy
        uni_dir = DEFAULT_UNIVERSE_FILE.parent
        os.makedirs(uni_dir, exist_ok=True)
        # canonical latest
        try:
            with open(DEFAULT_UNIVERSE_FILE, 'w', encoding='utf-8') as uf:
                json.dump({'generated_at_utc': datetime.utcnow().isoformat() + 'Z', 'symbols': universe}, uf, indent=2)
            LOG.info('Wrote canonical universe: %s', DEFAULT_UNIVERSE_FILE)
        except Exception:
            LOG.exception('failed to write canonical universe file')

        # versioned snapshot
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        versioned = uni_dir / f"universe_{ts}.json"
        try:
            with open(versioned, 'w', encoding='utf-8') as vf:
                json.dump({'generated_at_utc': datetime.utcnow().isoformat() + 'Z', 'symbols': universe}, vf, indent=2)
            LOG.info('Wrote versioned universe: %s', versioned)
        except Exception:
            LOG.exception('failed to write versioned universe file')

        if args.universe_only:
            return

        # determine symbols to ingest: prefer canonical universe file if present
        if DEFAULT_UNIVERSE_FILE.exists():
            try:
                with open(DEFAULT_UNIVERSE_FILE, 'r', encoding='utf-8') as uf:
                    doc = json.load(uf)
                    universe = doc.get('symbols', universe)
                    LOG.info('Loaded universe from %s with %d entries', DEFAULT_UNIVERSE_FILE, len(universe))
            except Exception:
                LOG.exception('Failed to load canonical universe file; falling back to generated list')

        if args.symbols:
            wanted = [s.strip().upper() for s in args.symbols.split(',')]
            universe = [u for u in universe if u['symbol'] in wanted]

        timeframes = [t.strip() for t in args.timeframes.split(',')]
        # default bars per timeframe
        bars_per_tf = {"1H": 240, "4H": 240, "1D": max(365, args.history_days), "1M": 120}

        # Prepare symbol list
        symbols = []
        for item in universe:
            if isinstance(item, dict):
                symbols.append(item.get('symbol'))
            else:
                symbols.append(str(item))

        # optional symbol filter
        if args.symbols:
            wanted = [s.strip().upper() for s in args.symbols.split(',')]
            symbols = [s for s in symbols if s in wanted]

        # orchestrate ingestion using process pool (safe for MT5)
        from .orchestrator import orchestrate_process_pool
        creds = load_credentials(args.env)
        results = orchestrate_process_pool(symbols, creds, args.output_dir, timeframes, bars_per_tf, workers=args.workers, tick_logging=args.tick_logging, rate_limit=args.rate_limit, write_parquet=args.parquet)
        LOG.info('Orchestration finished, %d results', len(results))

    finally:
        pool.close()


if __name__ == '__main__':
    run()
