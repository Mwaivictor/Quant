"""Ingestion entrypoints for the Raw Data Layer.

These functions illustrate how to pull finalized OHLCV bars and optional
ticks from MT5 using the `MT5ConnectionPool` and persist them using the
immutable writers. The ingestion layer performs no validation or cleaning;
it records exactly what the broker returned and writes metadata describing
the ingestion cycle.
"""

from __future__ import annotations
from datetime import datetime, timedelta
import time
import json
import os
import logging
from typing import List, Sequence, Dict, Any

from .config import DEFAULT_CONFIG, broker_to_utc, detect_broker_utc_offset
from .mt5_pool import MT5ConnectionPool
from .writer import write_ohlcv, write_ticks

LOG = logging.getLogger("arbitrex.raw.ingest")


TF_MAP = {
	"1H": "H1",
	"4H": "H4",
	"1D": "D1",
}


def _tf_to_mt5_constant(tf: str):
	try:
		import MetaTrader5 as mt5
	except Exception:
		raise
	mapping = {"1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4, "1D": mt5.TIMEFRAME_D1}
	return mapping[tf]


def ingest_ohlcv_once(pool: MT5ConnectionPool, symbol: str, timeframe: str, cycle_id: str, bars_expected: int = 1, base_dir: str | None = None) -> Dict[str, Any]:
	"""Ingest finalized OHLCV bars from MT5 and write them immutably.

	- `bars_expected` indicates how many bars we asked for (for metadata only).
	- The caller is responsible for calling this after the bar close + buffer.
	Returns ingestion metadata dict.
	"""
	base_dir = base_dir or DEFAULT_CONFIG.base_dir
	name, sess = pool.get_connection(timeout=10)
	
	# Detect broker timezone offset once per ingestion
	broker_offset = DEFAULT_CONFIG.broker_utc_offset_hours
	if broker_offset is None:
		broker_offset = detect_broker_utc_offset()
	
	result: Dict[str, Any] = {
		"cycle_id": cycle_id,
		"symbol": symbol,
		"timeframe": timeframe,
		"source": "MT5",
		"account_id": None,
		"broker_utc_offset_hours": broker_offset,
		"ingestion_time_utc": datetime.utcnow().isoformat() + "Z",
		"bars_expected": bars_expected,
		"bars_received": 0,
		"status": "FAILED",
		"error": None,
		"timestamps_normalized": DEFAULT_CONFIG.normalize_timestamps,
	}

	try:
		import MetaTrader5 as mt5

		tf_const = _tf_to_mt5_constant(timeframe)

		info = mt5.account_info()
		if info is not None:
			result["account_id"] = getattr(info, "login", None)

		terminal = mt5.terminal_info()
		if terminal is not None:
			result["broker_time_zone"] = getattr(terminal, "company", None)

		# request `bars_expected` bars ending at the most recent closed bar
		rates = mt5.copy_rates_from_pos(symbol, tf_const, 1, bars_expected)

		rows: List[Sequence] = []
		if rates is not None:
			for r in rates:
				# Each r has fields: time, open, high, low, close, tick_volume
				# r.time is in broker local time; convert to UTC
				ts_broker = int(r.time)
				ts_utc = broker_to_utc(ts_broker, broker_offset)
				# Store [timestamp_utc, timestamp_broker, OHLCV...]
				rows.append([ts_utc, ts_broker, r.open, r.high, r.low, r.close, r.tick_volume])

		result["bars_received"] = len(rows)
		result["status"] = "SUCCESS" if len(rows) == bars_expected else ("PARTIAL" if len(rows) > 0 else "FAILED")

		# write rows with normalized timestamps
		if rows:
			write_ohlcv(base_dir, symbol, timeframe, rows, cycle_id, broker_utc_offset=broker_offset)

	except Exception as e:
		LOG.exception("ingest_ohlcv_once failed")
		result["error"] = str(e)
		result["status"] = "FAILED"
	finally:
		pool.release_connection((name, sess))

	# write cycle metadata to metadata/ingestion_logs
	try:
		meta_dir = os.path.join(base_dir, "metadata", "ingestion_logs")
		os.makedirs(meta_dir, exist_ok=True)
		meta_path = os.path.join(meta_dir, f"{cycle_id}.meta.json")
		with open(meta_path, "w", encoding="utf-8") as mf:
			json.dump(result, mf, indent=2)
	except Exception:
		LOG.debug("failed to write cycle metadata")

	return result


def ingest_ticks_once(pool: MT5ConnectionPool, symbol: str, cycle_id: str, duration_seconds: int = 10, base_dir: str | None = None) -> Dict[str, Any]:
	"""Capture raw ticks for `duration_seconds`. Stored verbatim for diagnostics.

	The tick capture is optional and must not be used by signal generation.
	"""
	base_dir = base_dir or DEFAULT_CONFIG.base_dir
	name, sess = pool.get_connection(timeout=10)
	
	# Detect broker timezone offset
	broker_offset = DEFAULT_CONFIG.broker_utc_offset_hours
	if broker_offset is None:
		broker_offset = detect_broker_utc_offset()
	
	result = {
		"cycle_id": cycle_id,
		"symbol": symbol,
		"seconds": duration_seconds,
		"ticks_captured": 0,
		"status": "FAILED",
		"error": None,
		"broker_utc_offset_hours": broker_offset,
		"timestamps_normalized": DEFAULT_CONFIG.normalize_timestamps,
	}
	try:
		import MetaTrader5 as mt5
		end_time = time.time() + duration_seconds
		rows = []
		while time.time() < end_time:
			tick = mt5.copy_ticks_from(symbol, int(time.time()) - 1, 1, mt5.COPY_TICKS_ALL)
			if tick is not None and len(tick) > 0:
				for t in tick:
					ts_broker = int(t.time)
					ts_utc = broker_to_utc(ts_broker, broker_offset)
					rows.append([ts_utc, ts_broker, getattr(t, "bid", None), getattr(t, "ask", None), getattr(t, "last", None), getattr(t, "volume", None)])
			time.sleep(0.2)

		result["ticks_captured"] = len(rows)
		result["status"] = "SUCCESS"
		if rows:
			write_ticks(base_dir, symbol, rows, cycle_id, broker_utc_offset=broker_offset)
	except Exception as e:
		result["error"] = str(e)
		result["status"] = "FAILED"
	finally:
		pool.release_connection((name, sess))

	try:
		meta_dir = os.path.join(base_dir, "metadata", "ingestion_logs")
		os.makedirs(meta_dir, exist_ok=True)
		with open(os.path.join(meta_dir, f"{cycle_id}.ticks.meta.json"), "w", encoding="utf-8") as mf:
			json.dump(result, mf, indent=2)
	except Exception:
		LOG.debug("failed to write tick cycle metadata")

	return result


def generate_trading_universe_from_json(json_path: str, out_csv: str, only_fx: bool = True) -> int:
	"""Read an MT5 symbols JSON (export) and produce a simple CSV trading universe.

	- `only_fx`: if True, include symbols that look like FX pairs (6 alpha characters).
	Returns number of symbols written.
	"""
	import csv
	import json

	with open(json_path, "r", encoding="utf-8") as f:
		doc = json.load(f)

	symbols = doc.get("symbols", []) if isinstance(doc, dict) else doc
	rows = []
	for s in symbols:
		name = s.get("name") if isinstance(s, dict) else None
		if not name and isinstance(s, str):
			name = s
		if not name:
			continue
		if only_fx:
			if len(name) != 6 or not name.isalpha():
				continue
		rows.append({"symbol": name, "source": "MT5", "raw": json.dumps(s)})

	os.makedirs(os.path.dirname(out_csv), exist_ok=True)
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["symbol", "source", "raw"])
		writer.writeheader()
		for r in rows:
			writer.writerow(r)

	return len(rows)


def ingest_historical_range(pool: MT5ConnectionPool, symbol: str, timeframe: str, start_time_unix: int, end_time_unix: int, cycle_id_prefix: str, base_dir: str | None = None) -> Dict[str, Any]:
	"""Fetch raw bars between start and end (inclusive) and write them as daily files.

	Note: this function does NOT resample bars. It requests bars at MT5 timeframe
	corresponding to `timeframe` (e.g. '1D' -> D1) and writes rows exactly as received.
	For long-window overviews (3 months, 6 months, 1 year) callers should request
	sufficient daily history and downstream layers compute aggregates.
	"""
	base_dir = base_dir or DEFAULT_CONFIG.base_dir
	name, sess = pool.get_connection(timeout=10)
	
	# Detect broker timezone offset
	broker_offset = DEFAULT_CONFIG.broker_utc_offset_hours
	if broker_offset is None:
		broker_offset = detect_broker_utc_offset()
	
	result = {
		"symbol": symbol,
		"timeframe": timeframe,
		"start": start_time_unix,
		"end": end_time_unix,
		"bars_written": 0,
		"broker_utc_offset_hours": broker_offset,
		"timestamps_normalized": DEFAULT_CONFIG.normalize_timestamps,
	}
	try:
		import MetaTrader5 as mt5
		tf_const = _tf_to_mt5_constant(timeframe)
		# fetch bars: use copy_rates_range if available
		try:
			rates = mt5.copy_rates_range(symbol, tf_const, start_time_unix, end_time_unix)
		except Exception:
			# fallback to copy_rates_from_pos reading many bars
			rates = None

		rows = []
		if rates is not None:
			for r in rates:
				ts_broker = int(r.time)
				ts_utc = broker_to_utc(ts_broker, broker_offset)
				rows.append([ts_utc, ts_broker, r.open, r.high, r.low, r.close, r.tick_volume])

		# write rows using write_ohlcv (which writes one CSV per day based on first timestamp)
		if rows:
			cid = f"{cycle_id_prefix}_{symbol}_{time.time():.0f}"
			write_ohlcv(base_dir, symbol, timeframe, rows, cid, broker_utc_offset=broker_offset)
			result["bars_written"] = len(rows)
	except Exception as e:
		result["error"] = str(e)
	finally:
		pool.release_connection((name, sess))

	return result

