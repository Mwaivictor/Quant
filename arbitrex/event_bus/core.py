"""
Core event bus implementation with per-symbol isolation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import threading
import logging
import uuid

LOG = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in system"""
    # Data pipeline events
    TICK_RECEIVED = "tick_received"
    NORMALIZED_BAR_READY = "normalized_bar_ready"
    FEATURE_TIER1_READY = "feature_tier1_ready"
    FEATURE_TIER2_READY = "feature_tier2_ready"
    
    # Signal engine events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_APPROVED = "signal_approved"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Portfolio/Risk events
    POSITION_UPDATED = "position_updated"
    RESERVATION_CREATED = "reservation_created"
    RESERVATION_COMMITTED = "reservation_committed"
    RESERVATION_RELEASED = "reservation_released"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    
    # Execution events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    
    # System events
    BROKER_SYNC_COMPLETE = "broker_sync_complete"
    HEALTH_CHECK = "health_check"


@dataclass
class Event:
    """Base event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.TICK_RECEIVED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = 0


class EventBus:
    """Thread-safe event bus with per-symbol buffers"""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self._symbol_buffers: Dict[str, deque] = {}
        self._global_buffer = deque(maxlen=buffer_size)
        self._subscribers: Dict[EventType, List] = defaultdict(list)
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._events_published = 0
        self._events_dispatched = 0
        self._events_dropped = 0
    
    def start(self):
        """Start dispatcher"""
        if self._running:
            return
        self._running = True
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventBusDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        LOG.info("EventBus started")
    
    def stop(self):
        """Stop dispatcher"""
        if not self._running:
            return
        self._running = False
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)
        LOG.info("EventBus stopped")
    
    def publish(self, event: Event) -> bool:
        """Publish event (non-blocking)"""
        self._events_published += 1
        
        with self._lock:
            if event.symbol:
                if event.symbol not in self._symbol_buffers:
                    self._symbol_buffers[event.symbol] = deque(maxlen=self.buffer_size)
                buffer = self._symbol_buffers[event.symbol]
            else:
                buffer = self._global_buffer
            
            if len(buffer) >= self.buffer_size:
                self._events_dropped += 1
                return False
            
            buffer.append(event)
            return True
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None], symbols: Optional[List[str]] = None):
        """Subscribe to events"""
        subscriber = {'callback': callback, 'symbols': set(symbols) if symbols else None}
        self._subscribers[event_type].append(subscriber)
        return len(self._subscribers[event_type]) - 1
    
    def _dispatch_loop(self):
        """Background dispatcher"""
        while self._running:
            dispatched = 0
            
            with self._lock:
                # Dispatch global events
                while self._global_buffer:
                    event = self._global_buffer.popleft()
                    self._dispatch_event(event)
                    dispatched += 1
                
                # Dispatch per-symbol events
                for symbol in list(self._symbol_buffers.keys()):
                    buffer = self._symbol_buffers[symbol]
                    batch_size = min(100, len(buffer))
                    for _ in range(batch_size):
                        if buffer:
                            event = buffer.popleft()
                            self._dispatch_event(event)
                            dispatched += 1
            
            if dispatched == 0:
                threading.Event().wait(0.001)  # 1ms
    
    def _dispatch_event(self, event: Event):
        """Dispatch to subscribers"""
        subscribers = self._subscribers.get(event.event_type, [])
        for sub in subscribers:
            if sub['symbols'] is None or (event.symbol and event.symbol in sub['symbols']):
                try:
                    sub['callback'](event)
                    self._events_dispatched += 1
                except Exception as e:
                    LOG.error(f"Subscriber callback failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics"""
        with self._lock:
            total_buffer = len(self._global_buffer)
            for buf in self._symbol_buffers.values():
                total_buffer += len(buf)
            
            return {
                'events_published': self._events_published,
                'events_dispatched': self._events_dispatched,
                'events_dropped': self._events_dropped,
                'buffer_depth': total_buffer,
                'symbol_buffers': len(self._symbol_buffers),
                'running': self._running
            }


_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
        _global_bus.start()
    return _global_bus
