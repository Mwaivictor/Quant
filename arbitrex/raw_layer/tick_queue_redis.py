"""Redis-backed durable tick queue using Redis Streams.

This adapter uses Redis Streams (`XADD`, `XRANGE`, `XDEL`) so multiple
producers/consumers can operate. It provides the same interface as
`TickQueue` used by the code: `enqueue`, `dequeue_all_for_symbol`, `delete_ids`, `count`, `close`.

Requires `redis` Python package. Use `REDIS_URL` environment variable
or pass a URL to the constructor.
"""
import json
import threading
from typing import List, Tuple, Optional

try:
    import redis
except Exception:
    redis = None


class RedisTickQueue:
    def __init__(self, url: Optional[str] = None):
        if redis is None:
            raise RuntimeError("redis package not available")
        self._url = url or "redis://localhost:6379/0"
        self._r = redis.from_url(self._url, decode_responses=True)
        self._lock = threading.RLock()

    def _stream_key(self, symbol: str) -> str:
        return f"ticks:{symbol}"

    def enqueue(self, symbol: str, ts: int, bid: Optional[float], ask: Optional[float], last: Optional[float], volume: Optional[float], seq: Optional[str] = None):
        with self._lock:
            key = self._stream_key(symbol)
            # Convert numpy types to native Python types for JSON serialization
            def to_py(val):
                if hasattr(val, 'item'):
                    return val.item()
                return float(val) if isinstance(val, (float,)) else int(val) if isinstance(val, (int,)) else val

            fields = {
                "ts": str(int(ts)),
                "bid": json.dumps(to_py(bid)),
                "ask": json.dumps(to_py(ask)),
                "last": json.dumps(to_py(last)),
                "volume": json.dumps(to_py(volume))
            }
            if seq:
                fields['seq'] = seq
            # XADD returns the entry id
            return self._r.xadd(key, fields)

    def dequeue_all_for_symbol(self, symbol: str) -> List[Tuple[str, int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]]:
        with self._lock:
            key = self._stream_key(symbol)
            # XRANGE from beginning to end
            entries = self._r.xrange(key, min='-', max='+')
            out = []
            for eid, data in entries:
                ts = int(data.get('ts', 0))
                bid = json.loads(data.get('bid', 'null'))
                ask = json.loads(data.get('ask', 'null'))
                last = json.loads(data.get('last', 'null'))
                vol = json.loads(data.get('volume', 'null'))
                seq = data.get('seq')
                out.append((eid, ts, bid, ask, last, vol, seq))
            return out

    def delete_ids(self, ids: List[str]):
        if not ids:
            return
        with self._lock:
            # group by stream key prefix (id includes nothing about key), we expect ids for a single stream per call
            # require caller to call delete_ids only with ids from a single symbol
            # use XDEL with stream and ids
            # Caller must pass ids as list of strings; we cannot infer stream key here — user should delete per-symbol via XRANGE ID mapping
            # To keep parity with TickQueue API, accept tuples (key, id) pairs optionally
            # Simple approach: assume ids are for a single symbol and try deleting across all known symbols
            # Not perfect — for robust usage prefer the Redis stream ids be passed along with symbol in caller
            # We'll attempt to delete across common known keys by scanning prefixes
            # This is a pragmatic implementation for the demo environment.
            for key in self._r.scan_iter(match='ticks:*'):
                try:
                    self._r.xdel(key, *ids)
                except Exception:
                    pass

    def count(self, symbol: Optional[str] = None) -> int:
        with self._lock:
            if symbol:
                key = self._stream_key(symbol)
                return self._r.xlen(key)
            # total across streams
            total = 0
            for key in self._r.scan_iter(match='ticks:*'):
                total += self._r.xlen(key)
            return total

    def close(self):
        try:
            self._r.close()
        except Exception:
            pass
