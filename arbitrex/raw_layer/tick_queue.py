import sqlite3
import threading
from typing import List, Tuple, Optional


class TickQueue:
    """A lightweight durable on-disk tick queue using SQLite.

    Schema: ticks(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts INTEGER,
    bid REAL, ask REAL, last REAL, volume REAL, seq TEXT)
    """
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
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
            return cur.lastrowid

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
