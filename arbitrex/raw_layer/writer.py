"""Immutable writers for raw OHLCV and tick data and metadata.

Writers enforce immutability by never overwriting an existing file. If a
target path exists the writer creates a uniquely suffixed file instead.

All data is written exactly as provided; no normalization, validation, or
feature generation is performed here.
"""

from __future__ import annotations
import os
import csv
import json
import time
import tempfile
from datetime import datetime
from typing import Sequence, Dict, List, Optional
import logging


CSV_OHLCV_HEADER = ["timestamp_utc", "timestamp_broker", "open", "high", "low", "close", "volume"]
CSV_TICK_HEADER = ["timestamp_utc", "timestamp_broker", "bid", "ask", "last", "volume"]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    for i in range(1, 1000):
        candidate = f"{root}__{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return f"{root}__{int(time.time())}{ext}"


def _atomic_write_rows_csv(header: Sequence[str], rows: Sequence[Sequence], final_path: str) -> str:
    """Atomically write rows (including header) to `final_path`.

    - Writes to a temporary file next to `final_path` and then os.replace.
    - If `final_path` exists, return a uniquely suffixed path instead.
    Returns the path actually written.
    """
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if os.path.exists(final_path):
        final_path = _unique_path(final_path)

    dirn = os.path.dirname(final_path)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', dir=dirn, text=True)
    os.close(fd)
    try:
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, final_path)
        return final_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


LOG = logging.getLogger("arbitrex.raw.writer")


def _write_parquet_copy(final_csv_path: str, parquet_path: str) -> Optional[str]:
    """Write a Parquet copy alongside the CSV. Returns path or None if failed."""
    try:
        import pandas as pd
    except Exception:
        LOG.debug("pandas not available; skipping parquet write for %s", final_csv_path)
        return None

    try:
        df = pd.read_csv(final_csv_path)
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        # try pyarrow by default
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            # fallback to fastparquet if installed
            df.to_parquet(parquet_path, index=False, engine='fastparquet')
        return parquet_path
    except Exception as e:
        LOG.exception("Failed to write parquet for %s: %s", final_csv_path, e)
        return None


def write_ohlcv(base_dir: str, symbol: str, timeframe: str, rows: Sequence[Sequence], cycle_id: str, write_parquet: bool = False, broker_utc_offset: int | None = None):
    """Write OHLCV rows with normalized UTC timestamps, split per-day files.

    Rows are written to: <base_dir>/ohlcv/fx/<SYMBOL>/<TIMEFRAME>/YYYY-MM-DD.csv
    Each row should contain [timestamp_utc, timestamp_broker, open, high, low, close, volume].
    Files are grouped by UTC date for consistency.
    
    Args:
        base_dir: Root directory for raw data
        symbol: Trading symbol
        timeframe: Timeframe string (1H, 4H, 1D, etc.)
        rows: Sequence of rows with dual timestamps
        cycle_id: Unique ingestion cycle identifier
        write_parquet: Whether to write Parquet derivative
        broker_utc_offset: Broker timezone offset (for metadata only)
    """
    if not rows:
        return

    # group rows by UTC date (first column is timestamp_utc)
    buckets: Dict[str, List[Sequence]] = {}
    for r in rows:
        try:
            ts_utc = int(r[0])
            date_str = datetime.utcfromtimestamp(ts_utc).date().isoformat()  # YYYY-MM-DD
        except Exception:
            date_str = datetime.utcnow().date().isoformat()
        buckets.setdefault(date_str, []).append(r)

    written_files: List[str] = []
    for date_str, group in buckets.items():
        dir_path = os.path.join(base_dir, "ohlcv", "fx", symbol, timeframe)
        _ensure_dir(dir_path)
        target = os.path.join(dir_path, f"{date_str}.csv")
        path = _atomic_write_rows_csv(CSV_OHLCV_HEADER, group, target)
        written_files.append(path)
        # optionally write parquet copy in parallel location (derivative)
        if write_parquet:
            try:
                parquet_path = os.path.join(base_dir, "parquet", "ohlcv", "fx", symbol, timeframe, f"{date_str}.parquet")
                _write_parquet_copy(path, parquet_path)
            except Exception:
                LOG.debug("parquet copy skipped or failed for %s", path)

    # write ingestion metadata stub for this cycle
    meta_dir = os.path.join(base_dir, "metadata", "ingestion_logs")
    _ensure_dir(meta_dir)
    meta_path = os.path.join(meta_dir, f"{cycle_id}.json")
    metadata = {
        "cycle_id": cycle_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "files": [os.path.relpath(p, base_dir) for p in written_files],
        "written_at": datetime.utcnow().isoformat() + "Z",
        "broker_utc_offset_hours": broker_utc_offset,
        "timestamps_normalized": True,
    }
    with open(_unique_path(meta_path), "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)


def write_ticks(base_dir: str, symbol: str, rows: Sequence[Sequence], cycle_id: str, write_parquet: bool = False, broker_utc_offset: int | None = None):
    """Write tick rows with normalized UTC timestamps.

    Stored under: <base_dir>/ticks/fx/<SYMBOL>/YYYYMMDD.csv
    First column is timestamp_utc used for file grouping.
    """
    if not rows:
        return
    
    # Group by UTC date (first column is timestamp_utc)
    buckets: Dict[str, List[Sequence]] = {}
    for r in rows:
        try:
            ts_utc = int(r[0])
            date_str = datetime.utcfromtimestamp(ts_utc).date().isoformat()
        except Exception:
            date_str = datetime.utcnow().date().isoformat()
        buckets.setdefault(date_str, []).append(r)
    
    written_files: List[str] = []
    for date_str, group in buckets.items():
        dir_path = os.path.join(base_dir, "ticks", "fx", symbol)
        _ensure_dir(dir_path)
        target = os.path.join(dir_path, f"{date_str}.csv")
        path = _atomic_write_rows_csv(CSV_TICK_HEADER, group, target)
        written_files.append(path)

    # optionally write parquet copy
    if write_parquet:
        try:
            parquet_path = os.path.join(base_dir, "parquet", "ticks", "fx", symbol, f"{date_str}.parquet")
            _write_parquet_copy(path, parquet_path)
        except Exception:
            LOG.debug("parquet copy skipped or failed for %s", path)

    meta_dir = os.path.join(base_dir, "metadata", "ingestion_logs")
    _ensure_dir(meta_dir)
    meta_path = os.path.join(meta_dir, f"{cycle_id}.ticks.json")
    meta = {
        "cycle_id": cycle_id,
        "symbol": symbol,
        "files": [os.path.relpath(p, base_dir) for p in written_files],
        "written_at": datetime.utcnow().isoformat() + "Z",
        "broker_utc_offset_hours": broker_utc_offset,
        "timestamps_normalized": True,
    }
    with open(_unique_path(meta_path), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

