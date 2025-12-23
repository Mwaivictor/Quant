import sqlite3
import threading
from typing import List, Tuple, Optional
from datetime import datetime


class TickQueue:
    """A lightweight durable on-disk tick queue using SQLite.

    Schema: ticks(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts INTEGER,
    bid REAL, ask REAL, last REAL, volume REAL, seq TEXT)
    
    Features:
    - Per-symbol isolation (no cross-symbol blocking)
    - Event emission on tick receipt (optional)
    """
    def __init__(self, path: str, emit_events: bool = False):
        self._path = path
        self._lock = threading.RLock()
        self._emit_events = emit_events
        self._event_bus = None
        
        if emit_events:
            try:
                from arbitrex.event_bus import get_event_bus, Event, EventType
                self._event_bus = get_event_bus()
                self._Event = Event
                self._EventType = EventType
            except ImportError:
                self._emit_events = False
        
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                ts INTEGER NOT NULL,
                bid REAL,
                ask REAL,
                last REAL,
                volume REAL,
                seq TEXT
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_ticks_symbol_ts ON ticks(symbol, ts)")
        self._conn.commit()

    def enqueue(self, symbol: str, ts: int, bid: Optional[float], ask: Optional[float], last: Optional[float], volume: Optional[float], seq: Optional[str] = None):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO ticks(symbol, ts, bid, ask, last, volume, seq) VALUES (?,?,?,?,?,?,?)",
                        (symbol, int(ts), bid, ask, last, volume, seq))
            self._conn.commit()
            tick_id = cur.lastrowid
            
            # Emit event (non-blocking)
            if self._emit_events and self._event_bus:
                event = self._Event(
                    event_type=self._EventType.TICK_RECEIVED,
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    data={
                        'tick_id': tick_id,
                        'ts': ts,
                        'bid': bid,
                        'ask': ask,
                        'last': last,
                        'volume': volume,
                        'seq': seq
                    }
                )
                self._event_bus.publish(event)
            
            return tick_id

    def dequeue_all_for_symbol(self, symbol: str) -> List[Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, ts, bid, ask, last, volume, seq FROM ticks WHERE symbol=? ORDER BY ts ASC, id ASC", (symbol,))
            rows = cur.fetchall()
            return rows

    def delete_ids(self, ids: List[int]):
        if not ids:
            return
        with self._lock:
            cur = self._conn.cursor()
            q = ",".join(["?" for _ in ids])
            cur.execute(f"DELETE FROM ticks WHERE id IN ({q})", ids)
            self._conn.commit()

    def count(self, symbol: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if symbol:
                cur.execute("SELECT COUNT(*) FROM ticks WHERE symbol=?", (symbol,))
            else:
                cur.execute("SELECT COUNT(*) FROM ticks")
            return int(cur.fetchone()[0])

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
