"""Orchestrator for process-based parallel ingestion.

Each worker process initializes its own MT5 connection (safe) and processes
assigned symbols, fetching finalized bars for requested timeframes and
writing immutable CSVs via the `writer` module. Timestamps are normalized
to UTC during ingestion.
"""
from __future__ import annotations
import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

LOG = logging.getLogger("arbitrex.raw.orchestrator")


def _worker_init_and_ingest(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint executed in each process.

    `task` contains: symbols (list), timeframes, bars_per_tf, creds, output_dir, tick_logging, rate_limit, write_parquet
    The worker initializes MT5 once and processes each assigned symbol sequentially.
    """
    # import inside worker to avoid pickling issues
    import MetaTrader5 as mt5
    from .writer import write_ohlcv, write_ticks
    from .config import broker_to_utc, detect_broker_utc_offset
    import logging as _logging

    log = _logging.getLogger(f"arbitrex.raw.worker.{os.getpid()}")

    symbols = task.get('symbols', [])
    timeframes = task['timeframes']
    bars_per_tf = task['bars_per_tf']
    creds = task['creds']
    output_dir = task['output_dir']
    tick_logging = task.get('tick_logging', False)
    write_parquet = task.get('write_parquet', False)
    rate_limit = task.get('rate_limit', 0.05)
    
    # Detect broker timezone offset for this worker process
    broker_offset = detect_broker_utc_offset()
    log.info(f"Worker {os.getpid()} detected broker UTC offset: %+d hours", broker_offset)

    summary = {'symbols': symbols, 'status': 'FAILED', 'details': []}
    try:
        # initialize and login once per process
        if creds.get('MT5_TERMINAL'):
            mt5.initialize(creds.get('MT5_TERMINAL'))
        else:
            mt5.initialize()

        if creds.get('MT5_LOGIN'):
            mt5.login(int(creds['MT5_LOGIN']), creds.get('MT5_PASSWORD'), creds.get('MT5_SERVER'))

        account = mt5.account_info()
        account_id = getattr(account, 'login', None) if account is not None else None

        for symbol in symbols:
            sym_summary = {'symbol': symbol, 'timeframes': []}
            for tf in timeframes:
                tf_const = {"1H": mt5.TIMEFRAME_H1, "4H": mt5.TIMEFRAME_H4, "1D": mt5.TIMEFRAME_D1, "1M": mt5.TIMEFRAME_MN1}[tf]
                count = bars_per_tf.get(tf, 200)
                rates = mt5.copy_rates_from_pos(symbol, tf_const, 1, count)
                rows = []
                if rates is not None:
                    for r in rates:
                        # mt5 returns numpy.void / structured arrays in some environments;
                        # prefer attribute access but fallback to index-based fields.
                        try:
                            ts_broker = int(r.time)
                            ts_utc = broker_to_utc(ts_broker, broker_offset)
                            o = float(r.open)
                            h = float(r.high)
                            l = float(r.low)
                            c = float(r.close)
                            v = int(getattr(r, 'tick_volume', 0))
                        except Exception:
                            # fallback by positional indices: (time, open, high, low, close, tick_volume)
                            try:
                                ts_broker = int(r[0])
                                ts_utc = broker_to_utc(ts_broker, broker_offset)
                                o = float(r[1])
                                h = float(r[2])
                                l = float(r[3])
                                c = float(r[4])
                                v = int(r[5]) if len(r) > 5 else 0
                            except Exception:
                                # last resort: skip malformed row
                                continue
                        rows.append([ts_utc, ts_broker, o, h, l, c, v])

                cycle_id = f"{datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ')}_{symbol}_{tf}"
                try:
                    write_ohlcv(output_dir, symbol, tf, rows, cycle_id, write_parquet=write_parquet, broker_utc_offset=broker_offset)
                    sym_summary['timeframes'].append({'tf': tf, 'bars': len(rows), 'cycle_id': cycle_id})
                except Exception as e:
                    sym_summary['timeframes'].append({'tf': tf, 'error': str(e)})

                time.sleep(rate_limit)

            if tick_logging:
                # small diagnostic capture with UTC normalization
                rows = []
                end_time = time.time() + 5
                while time.time() < end_time:
                    ticks = mt5.copy_ticks_from(symbol, int(time.time()) - 1, 1, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        for t in ticks:
                            ts_broker = int(t.time)
                            ts_utc = broker_to_utc(ts_broker, broker_offset)
                            rows.append([ts_utc, ts_broker, getattr(t, 'bid', None), getattr(t, 'ask', None), getattr(t, 'last', None), getattr(t, 'volume', None)])
                    time.sleep(0.2)
                if rows:
                    write_ticks(output_dir, symbol, rows, f"ticks_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{symbol}", write_parquet=write_parquet, broker_utc_offset=broker_offset)

            summary['details'].append(sym_summary)

        summary['status'] = 'SUCCESS'

    except Exception as e:
        summary['status'] = 'FAILED'
        summary['error'] = str(e)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    return summary


def _partition_symbols(symbols: List[str], workers: int) -> List[List[str]]:
    """Partition symbol list into `workers` roughly-equal chunks."""
    if workers <= 1:
        return [symbols]
    chunks: List[List[str]] = [[] for _ in range(workers)]
    for i, s in enumerate(symbols):
        chunks[i % workers].append(s)
    # remove empty chunks
    return [c for c in chunks if c]


def orchestrate_process_pool(symbols: List[str], creds: Dict[str, Any], output_dir: str, timeframes: List[str], bars_per_tf: Dict[str, int], workers: int = 4, tick_logging: bool = False, rate_limit: float = 0.05, write_parquet: bool = False) -> List[Dict[str, Any]]:
    # create shards
    shards = _partition_symbols(symbols, workers)
    tasks = []
    for shard in shards:
        tasks.append({'symbols': shard, 'timeframes': timeframes, 'bars_per_tf': bars_per_tf, 'creds': creds, 'output_dir': output_dir, 'tick_logging': tick_logging, 'rate_limit': rate_limit, 'write_parquet': write_parquet})

    results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futs = {ex.submit(_worker_init_and_ingest, t): t for t in tasks}
        for f in as_completed(futs):
            try:
                res = f.result()
                results.append(res)
                LOG.info('Worker finished: symbols=%s status=%s', res.get('symbols'), res.get('status'))
            except Exception as e:
                LOG.exception('Worker crashed: %s', e)

    return results
