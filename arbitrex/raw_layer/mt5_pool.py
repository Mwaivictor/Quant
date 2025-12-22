from __future__ import annotations
import threading
import queue
import time
import json
import os
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime

try:
    from .tick_queue import TickQueue
except Exception:
    TickQueue = None

try:
    from .tick_queue_redis import RedisTickQueue
except Exception:
    RedisTickQueue = None

try:
    from .tick_queue_kafka import KafkaTickQueue
except Exception:
    KafkaTickQueue = None

try:
    from .config import broker_to_utc, detect_broker_utc_offset
except Exception:
    # Fallback if config import fails
    def broker_to_utc(ts, offset=0):
        return ts - (offset * 3600) if offset else ts
    def detect_broker_utc_offset():
        return 0

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False


LOG = logging.getLogger("arbitrex.raw.mt5_pool")


# Minimal MT5Session stub
class MT5Session:
    def __init__(self, terminal_path, login, password, server):
        self.terminal_path = terminal_path
        self.login = login
        self.password = password
        self.server = server
        self.lock = threading.RLock()
        self.last_heartbeat = time.time()
        self.status = "DISCONNECTED"
        self.mt5_initialized = False

    def connect(self):
        try:
            import MetaTrader5 as mt5
            # Shutdown any existing connection first
            if self.mt5_initialized:
                try:
                    mt5.shutdown()
                    LOG.debug("Shutdown previous MT5 connection")
                except Exception:
                    pass
            
            # Try to initialize with all parameters
            LOG.info("Initializing MT5 connection (login=%s, server=%s)", self.login, self.server)
            ok = mt5.initialize(
                path=self.terminal_path,
                login=self.login,
                password=self.password,
                server=self.server,
            )
            if not ok:
                err = mt5.last_error()
                LOG.error("MT5 initialization failed: %s", err)
                self.status = "DISCONNECTED"
                self.mt5_initialized = False
                raise RuntimeError(f"MT5 init failed: {err}")
            
            self.status = "CONNECTED"
            self.mt5_initialized = True
            self.last_heartbeat = time.time()
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info:
                LOG.info("MT5 session connected successfully (login=%s, server=%s, balance=%.2f)", 
                        self.login, account_info.server, account_info.balance)
            else:
                LOG.warning("MT5 initialized but account_info is None")
        except Exception as e:
            LOG.exception("MT5 connect() exception: %s", e)
            self.status = "DISCONNECTED"
            self.mt5_initialized = False
            raise

    def heartbeat(self) -> bool:
        with self.lock:
            try:
                import MetaTrader5 as mt5
                if not self.mt5_initialized:
                    LOG.debug("Heartbeat: MT5 not initialized")
                    self.status = "DISCONNECTED"
                    return False
                
                info = mt5.account_info()
                if info is None:
                    LOG.debug("Heartbeat: account_info returned None")
                    self.status = "DISCONNECTED"
                    self.mt5_initialized = False
                    return False
                
                self.last_heartbeat = time.time()
                self.status = "CONNECTED"
                return True
            except Exception as e:
                LOG.debug("Heartbeat exception: %s", e)
                self.status = "DISCONNECTED"
                self.mt5_initialized = False
                LOG.debug("heartbeat check failed")
                return False

    def shutdown(self):
        with self.lock:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                LOG.debug("mt5.shutdown failed or mt5 not available")
            self.status = "DISCONNECTED"


class MT5ConnectionPool:
    def __init__(self, sessions: Dict[str, Dict], symbols: list, session_logs_dir: Optional[str] = None):
        self._queue: queue.Queue[Tuple[str, MT5Session]] = queue.Queue()
        self._sessions = []
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.session_logs_dir = session_logs_dir

        for name, params in sessions.items():
            sess = MT5Session(params.get("terminal_path"), params.get("login"),
                              params.get("password"), params.get("server"))
            try:
                sess.connect()
            except Exception:
                LOG.exception("initial connect failed for session %s", name)
            self._sessions.append((name, sess))
            self._queue.put((name, sess))

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        self._tick_thread: Optional[threading.Thread] = None
        self._tick_stop_event = threading.Event()
        self._tick_symbols: list = symbols
        self._tick_poll_interval: float = 0.5
        self._tick_flush_interval: float = 5.0
        self._tick_queue = None
        self._tick_last_ts: Dict[str, int] = {}
        self._tick_lock = threading.RLock()
        self._tick_buffers: Dict[str, list] = {}
        self._tick_publish_cb = None

        # Initialize durable queue (Redis or SQLite)
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            # Strip quotes from environment variable if present
            redis_url = redis_url.strip("'\"")
        if redis_url and RedisTickQueue is not None:
            try:
                self._tick_queue = RedisTickQueue(redis_url)
                LOG.info('Using RedisTickQueue (REDIS_URL=%s)', redis_url)
            except Exception as e:
                LOG.exception('Failed to init RedisTickQueue: %s; falling back', e)

        if self._tick_queue is None and TickQueue is not None:
            try:
                db_path = os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'ticks', 'ticks_queue.db')
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self._tick_queue = TickQueue(db_path)
                LOG.info('Using SQLite TickQueue at %s', db_path)
            except Exception:
                LOG.exception('Failed to init SQLite TickQueue')

        # Kafka producer (optional, disabled if DISABLE_KAFKA=1)
        self._kafka_producer = None
        kafka_bs = os.environ.get('KAFKA_BOOTSTRAP_SERVERS')
        disable_kafka = os.environ.get('DISABLE_KAFKA', '0') == '1'
        if disable_kafka:
            LOG.info('Kafka disabled via DISABLE_KAFKA=1')
        elif kafka_bs and KafkaTickQueue is not None:
            # Strip quotes from environment variable if present
            kafka_bs = kafka_bs.strip("'\"")
            try:
                self._kafka_producer = KafkaTickQueue(bootstrap_servers=kafka_bs)
                LOG.info('Initialized KafkaTickQueue producer (bootstrap=%s)', kafka_bs)
            except Exception as e:
                LOG.exception('Failed to init KafkaTickQueue: %s', e)

        # Prometheus metrics (optional)
        self._metrics = {}
        if PROM_AVAILABLE:
            try:
                self._metrics['received'] = Counter('arbitrex_ticks_received_total', 'Ticks received')
                self._metrics['published'] = Counter('arbitrex_ticks_published_total', 'Ticks published to WS')
                self._metrics['flushed'] = Counter('arbitrex_ticks_flushed_total', 'Ticks flushed to disk')
                self._metrics['queue_size'] = Gauge('arbitrex_ticks_queue_size', 'Ticks in durable queue')
                self._metrics['flush_duration'] = Histogram('arbitrex_tick_flush_seconds', 'Tick flush duration seconds')
                prom_port = int(os.environ.get('PROMETHEUS_PORT', '0'))
                if prom_port:
                    start_http_server(prom_port)
            except Exception:
                LOG.debug("prometheus metrics initialization failed")

    # -------------------------
    # Public methods
    # -------------------------

    def set_tick_publisher(self, cb):
        """Set a callback to publish tick dicts in real time."""
        self._tick_publish_cb = cb

    def get_connection(self, timeout: Optional[float] = None) -> Tuple[str, MT5Session]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("No MT5 sessions available")

    def release_connection(self, conn_tuple: Tuple[str, MT5Session]):
        self._queue.put(conn_tuple)

    def start_tick_collector(self, symbols: Optional[list], base_dir: str, poll_interval: float = 0.5, flush_interval: float = 5.0):
        with self._tick_lock:
            if self._tick_thread and self._tick_thread.is_alive():
                LOG.info("Tick collector already running")
                return
            if symbols:
                self._tick_symbols = symbols
            self._tick_poll_interval = poll_interval
            self._tick_flush_interval = flush_interval
            self._tick_stop_event.clear()
            self._tick_thread = threading.Thread(target=self._tick_loop, args=(base_dir,), daemon=True)
            self._tick_thread.start()
            LOG.info("Started tick collector for %d symbols", len(self._tick_symbols))

    def stop_tick_collector(self):
        with self._tick_lock:
            if self._tick_thread:
                self._tick_stop_event.set()
                self._tick_thread.join(timeout=5)
                self._tick_thread = None

    def close(self):
        self._stop_event.set()
        self.stop_tick_collector()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
        for _, sess in self._sessions:
            try:
                sess.shutdown()
            except Exception:
                LOG.debug("shutdown failed")
        if self._tick_queue:
            try:
                self._tick_queue.close()
            except Exception:
                LOG.debug("failed to close tick queue")

    # -------------------------
    # Internal methods
    # -------------------------

    def _heartbeat_loop(self):
        while not self._stop_event.is_set():
            for name, sess in self._sessions:
                ok = sess.heartbeat()
                if not ok:
                    LOG.warning("Session %s disconnected, attempting reconnect", name)
                    try:
                        sess.connect()
                    except Exception:
                        LOG.exception("Reconnect failed for %s", name)
            time.sleep(10)

    def _tick_loop(self, base_dir: str):
        """Simplified tick loop; assumes MetaTrader5 available"""
        try:
            import MetaTrader5 as mt5
        except Exception:
            LOG.exception("MetaTrader5 not available")
            return

        LOG.info("Tick collection loop started for %d symbols", len(self._tick_symbols))
        last_flush = time.time()
        tick_count = 0
        poll_count = 0
        last_log_time = time.time()
        
        # Ensure at least one session is connected
        connected = False
        for name, sess in self._sessions:
            if sess.status == "CONNECTED" and sess.mt5_initialized:
                connected = True
                break
        
        if not connected:
            LOG.warning("No MT5 sessions connected at tick loop start, attempting to connect...")
            for name, sess in self._sessions:
                try:
                    sess.connect()
                    connected = True
                    break
                except Exception as e:
                    LOG.error("Failed to connect session %s: %s", name, e)
        
        if not connected:
            LOG.error("Cannot start tick collection: No MT5 sessions available")
            return
        
        while not self._tick_stop_event.is_set():
            with self._tick_lock:
                symbols = list(self._tick_symbols)
            
            poll_count += 1
            
            # Log status every 30 seconds
            now = time.time()
            if now - last_log_time >= 30:
                LOG.info("Tick loop status: %d polls, %d ticks collected, checking %d symbols", 
                        poll_count, tick_count, len(symbols))
                last_log_time = now

            for name, sess in self._sessions:
                # Check session is connected
                if sess.status != "CONNECTED" or not sess.mt5_initialized:
                    LOG.debug("Session %s not connected, skipping tick poll", name)
                    continue
                
                for sym in symbols:
                    try:
                        from_ts = int(time.time()) - 60
                        ticks = mt5.copy_ticks_from(sym, from_ts, 10000, mt5.COPY_TICKS_ALL)
                        
                        if ticks is None:
                            err = mt5.last_error()
                            if poll_count % 100 == 0:  # Log occasionally
                                LOG.debug(f"No ticks for {sym}: {err}")
                            continue
                        
                        if len(ticks) == 0:
                            if poll_count % 100 == 0:  # Log occasionally
                                LOG.debug(f"{sym}: Retrieved 0 ticks (market may be closed)")
                            continue
                        
                        if tick_count % 100 == 0:
                            LOG.info(f"Retrieved {len(ticks)} ticks for {sym}")
                        tick_count += len(ticks)

                        for t in ticks:
                            try:
                                ts_broker = int(getattr(t, 'time', t[0]))
                                ts_utc = broker_to_utc(ts_broker, self._broker_utc_offset)
                                bid = getattr(t, 'bid', None) if hasattr(t, 'bid') else t[1]
                                ask = getattr(t, 'ask', None) if hasattr(t, 'ask') else t[2]
                                last = getattr(t, 'last', None) if hasattr(t, 'last') else t[3]
                                vol = getattr(t, 'volume', None) if hasattr(t, 'volume') else t[4]

                                # enqueue to durable queue with UTC timestamp
                                if self._tick_queue:
                                    try:
                                        self._tick_queue.enqueue(sym, ts_utc, bid, ask, last, vol)
                                    except Exception:
                                        LOG.exception('enqueue to durable queue failed')
                                else:
                                    self._tick_buffers.setdefault(sym, []).append([ts_utc, bid, ask, last, vol])

                                # publish to WS with both timestamps for client info
                                if self._tick_publish_cb:
                                    try:
                                        payload = {
                                            "symbol": sym,
                                            "ts": ts_utc,
                                            "ts_broker": ts_broker,
                                            "bid": float(bid) if bid else None,
                                            "ask": float(ask) if ask else None, 
                                            "last": float(last) if last else None, 
                                            "volume": int(vol) if vol else 0
                                        }
                                        self._tick_publish_cb(payload)
                                        
                                        # Log first few publishes for debugging
                                        if tick_count <= 5:
                                            LOG.info("Published tick to WebSocket: %s", payload)
                                    except Exception as e:
                                        if tick_count <= 10:
                                            LOG.error("tick publish callback failed: %s", e, exc_info=True)
                                        else:
                                            LOG.debug("tick publish callback failed: %s", e)

                            except Exception:
                                LOG.debug("tick unpack failed for %s", sym)

                    except Exception:
                        LOG.debug("tick poll failed for %s", sym)

            # flush interval
            now = time.time()
            if now - last_flush >= self._tick_flush_interval:
                last_flush = now

            time.sleep(self._tick_poll_interval)

