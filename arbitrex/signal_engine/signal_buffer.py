"""
Thread-Safe Signal Buffer

Provides thread-safe buffering and routing of signals from multiple strategies.

Features:
- Per-strategy signal buffers (isolation)
- Global signal queue for downstream consumers
- Non-blocking signal publication
- Thread-safe signal retrieval
- Signal filtering and routing
- Buffer overflow protection
- Metrics and monitoring

Similar to EventBus pattern but specialized for signals.
"""

import threading
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field

from arbitrex.signal_engine.signal_schemas import Signal, SignalStatus, SignalType

LOG = logging.getLogger(__name__)


@dataclass
class SignalBufferMetrics:
    """Signal buffer metrics for monitoring"""
    signals_published: int = 0
    signals_consumed: int = 0
    signals_dropped: int = 0
    signals_filtered: int = 0
    buffer_depth: int = 0
    per_strategy_depth: Dict[str, int] = field(default_factory=dict)
    oldest_signal_age_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'signals_published': self.signals_published,
            'signals_consumed': self.signals_consumed,
            'signals_dropped': self.signals_dropped,
            'signals_filtered': self.signals_filtered,
            'buffer_depth': self.buffer_depth,
            'per_strategy_depth': dict(self.per_strategy_depth),
            'oldest_signal_age_seconds': self.oldest_signal_age_seconds,
        }


class SignalFilter:
    """Filter for signal routing"""
    
    def __init__(
        self,
        strategy_ids: Optional[Set[str]] = None,
        signal_types: Optional[Set[SignalType]] = None,
        symbols: Optional[Set[str]] = None,
        min_confidence: float = 0.0
    ):
        """
        Initialize signal filter.
        
        Args:
            strategy_ids: Filter by strategy IDs (None = all)
            signal_types: Filter by signal types (None = all)
            symbols: Filter by symbols (None = all)
            min_confidence: Minimum confidence threshold
        """
        self.strategy_ids = strategy_ids
        self.signal_types = signal_types
        self.symbols = symbols
        self.min_confidence = min_confidence
    
    def matches(self, signal: Signal) -> bool:
        """Check if signal matches filter"""
        # Check strategy ID
        if self.strategy_ids is not None and signal.strategy_id not in self.strategy_ids:
            return False
        
        # Check signal type
        if self.signal_types is not None and signal.signal_type not in self.signal_types:
            return False
        
        # Check symbols (any overlap)
        if self.symbols is not None:
            signal_symbols = set(signal.symbols)
            if not signal_symbols.intersection(self.symbols):
                return False
        
        # Check confidence
        if signal.confidence_score < self.min_confidence:
            return False
        
        return True


class SignalSubscriber:
    """Signal subscriber with callback"""
    
    def __init__(
        self,
        callback: Callable[[Signal], None],
        signal_filter: Optional[SignalFilter] = None,
        name: str = ""
    ):
        """
        Initialize subscriber.
        
        Args:
            callback: Function to call with signal
            signal_filter: Optional filter for signals
            name: Subscriber name for logging
        """
        self.callback = callback
        self.signal_filter = signal_filter
        self.name = name or f"subscriber_{id(self)}"
        self.signals_received = 0
        self.signals_filtered = 0
    
    def should_receive(self, signal: Signal) -> bool:
        """Check if subscriber should receive signal"""
        if self.signal_filter is None:
            return True
        return self.signal_filter.matches(signal)
    
    def notify(self, signal: Signal):
        """Notify subscriber of signal"""
        if self.should_receive(signal):
            try:
                self.callback(signal)
                self.signals_received += 1
            except Exception as e:
                LOG.error(f"Subscriber {self.name} callback failed: {e}")
        else:
            self.signals_filtered += 1


class SignalBuffer:
    """
    Thread-safe signal buffer with per-strategy isolation.
    
    Manages signal flow from multiple strategies to downstream consumers.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        per_strategy_buffer_size: int = 1000,
        enable_expiry_check: bool = True,
        expiry_check_interval: float = 1.0
    ):
        """
        Initialize signal buffer.
        
        Args:
            buffer_size: Maximum size of global buffer
            per_strategy_buffer_size: Maximum size per strategy buffer
            enable_expiry_check: Enable automatic expiry checking
            expiry_check_interval: Expiry check interval in seconds
        """
        self.buffer_size = buffer_size
        self.per_strategy_buffer_size = per_strategy_buffer_size
        
        # Buffers
        self._global_buffer: deque = deque(maxlen=buffer_size)
        self._strategy_buffers: Dict[str, deque] = {}
        
        # Subscribers
        self._subscribers: List[SignalSubscriber] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self._signals_published = 0
        self._signals_consumed = 0
        self._signals_dropped = 0
        self._signals_filtered = 0
        
        # Expiry checking
        self._enable_expiry_check = enable_expiry_check
        self._expiry_check_interval = expiry_check_interval
        self._expiry_thread: Optional[threading.Thread] = None
        self._running = False
        
        if enable_expiry_check:
            self.start_expiry_checker()
    
    def start_expiry_checker(self):
        """Start background expiry checker thread"""
        if self._running:
            return
        
        self._running = True
        self._expiry_thread = threading.Thread(
            target=self._expiry_check_loop,
            name="SignalBufferExpiryChecker",
            daemon=True
        )
        self._expiry_thread.start()
        LOG.info("SignalBuffer expiry checker started")
    
    def stop_expiry_checker(self):
        """Stop background expiry checker"""
        if not self._running:
            return
        
        self._running = False
        if self._expiry_thread:
            self._expiry_thread.join(timeout=5.0)
        LOG.info("SignalBuffer expiry checker stopped")
    
    def _expiry_check_loop(self):
        """Background loop to check and remove expired signals"""
        while self._running:
            try:
                self._remove_expired_signals()
            except Exception as e:
                LOG.error(f"Expiry check failed: {e}")
            
            threading.Event().wait(self._expiry_check_interval)
    
    def _remove_expired_signals(self):
        """Remove expired signals from buffers"""
        with self._lock:
            # Check global buffer
            expired_count = 0
            for _ in range(len(self._global_buffer)):
                signal = self._global_buffer[0]
                if signal.is_expired():
                    self._global_buffer.popleft()
                    expired_count += 1
                else:
                    # Since deque is ordered, if first is not expired, rest aren't either
                    break
            
            # Check per-strategy buffers
            for strategy_id, buffer in list(self._strategy_buffers.items()):
                for _ in range(len(buffer)):
                    signal = buffer[0]
                    if signal.is_expired():
                        buffer.popleft()
                        expired_count += 1
                    else:
                        break
                
                # Remove empty strategy buffers
                if len(buffer) == 0:
                    del self._strategy_buffers[strategy_id]
            
            if expired_count > 0:
                LOG.debug(f"Removed {expired_count} expired signals")
    
    def publish(self, signal: Signal) -> bool:
        """
        Publish signal to buffer (non-blocking).
        
        Args:
            signal: Signal to publish
            
        Returns:
            True if published, False if dropped (buffer full)
        """
        with self._lock:
            # Check if signal already expired
            if signal.is_expired():
                LOG.warning(f"Signal {signal.signal_id} already expired, not publishing")
                self._signals_dropped += 1
                return False
            
            # Add to global buffer
            if len(self._global_buffer) >= self.buffer_size:
                LOG.warning(f"Global signal buffer full ({self.buffer_size}), dropping signal")
                self._signals_dropped += 1
                return False
            
            self._global_buffer.append(signal)
            
            # Add to per-strategy buffer
            strategy_id = signal.strategy_id
            if strategy_id not in self._strategy_buffers:
                self._strategy_buffers[strategy_id] = deque(maxlen=self.per_strategy_buffer_size)
            
            strategy_buffer = self._strategy_buffers[strategy_id]
            if len(strategy_buffer) >= self.per_strategy_buffer_size:
                LOG.warning(f"Strategy {strategy_id} buffer full ({self.per_strategy_buffer_size})")
                # Already added to global, so don't drop, just don't add to strategy buffer
            else:
                strategy_buffer.append(signal)
            
            self._signals_published += 1
            
            # Notify subscribers
            self._notify_subscribers(signal)
            
            return True
    
    def _notify_subscribers(self, signal: Signal):
        """Notify all subscribers of new signal"""
        for subscriber in self._subscribers:
            try:
                subscriber.notify(signal)
            except Exception as e:
                LOG.error(f"Failed to notify subscriber {subscriber.name}: {e}")
    
    def subscribe(
        self,
        callback: Callable[[Signal], None],
        signal_filter: Optional[SignalFilter] = None,
        name: str = ""
    ) -> SignalSubscriber:
        """
        Subscribe to signals.
        
        Args:
            callback: Function to call with signal
            signal_filter: Optional filter
            name: Subscriber name
            
        Returns:
            SignalSubscriber object
        """
        with self._lock:
            subscriber = SignalSubscriber(callback, signal_filter, name)
            self._subscribers.append(subscriber)
            LOG.info(f"Subscriber {subscriber.name} registered")
            return subscriber
    
    def unsubscribe(self, subscriber: SignalSubscriber):
        """Remove subscriber"""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
                LOG.info(f"Subscriber {subscriber.name} removed")
    
    def get_signals(
        self,
        strategy_id: Optional[str] = None,
        max_count: int = 100,
        remove: bool = True
    ) -> List[Signal]:
        """
        Get signals from buffer.
        
        Args:
            strategy_id: Get signals from specific strategy (None = global)
            max_count: Maximum number of signals to retrieve
            remove: Remove signals from buffer after retrieval
            
        Returns:
            List of signals
        """
        with self._lock:
            signals = []
            
            if strategy_id is None:
                # Get from global buffer
                buffer = self._global_buffer
            else:
                # Get from strategy-specific buffer
                buffer = self._strategy_buffers.get(strategy_id)
                if buffer is None:
                    return []
            
            # Retrieve signals
            count = min(max_count, len(buffer))
            if remove:
                for _ in range(count):
                    signal = buffer.popleft()
                    signals.append(signal)
                    self._signals_consumed += 1
            else:
                # Non-destructive read
                for i in range(count):
                    signals.append(buffer[i])
            
            return signals
    
    def peek(self, strategy_id: Optional[str] = None) -> Optional[Signal]:
        """
        Peek at next signal without removing.
        
        Args:
            strategy_id: Strategy ID (None = global)
            
        Returns:
            Next signal or None
        """
        with self._lock:
            if strategy_id is None:
                buffer = self._global_buffer
            else:
                buffer = self._strategy_buffers.get(strategy_id)
                if buffer is None:
                    return None
            
            return buffer[0] if buffer else None
    
    def get_metrics(self) -> SignalBufferMetrics:
        """Get buffer metrics"""
        with self._lock:
            # Calculate buffer depth
            total_depth = len(self._global_buffer)
            per_strategy_depth = {
                sid: len(buf) for sid, buf in self._strategy_buffers.items()
            }
            
            # Calculate oldest signal age
            oldest_age = 0.0
            if self._global_buffer:
                oldest_signal = self._global_buffer[0]
                age_delta = datetime.utcnow() - oldest_signal.created_at
                oldest_age = age_delta.total_seconds()
            
            return SignalBufferMetrics(
                signals_published=self._signals_published,
                signals_consumed=self._signals_consumed,
                signals_dropped=self._signals_dropped,
                signals_filtered=self._signals_filtered,
                buffer_depth=total_depth,
                per_strategy_depth=per_strategy_depth,
                oldest_signal_age_seconds=oldest_age
            )
    
    def clear(self, strategy_id: Optional[str] = None):
        """
        Clear buffer.
        
        Args:
            strategy_id: Clear specific strategy buffer (None = all)
        """
        with self._lock:
            if strategy_id is None:
                self._global_buffer.clear()
                self._strategy_buffers.clear()
                LOG.info("All signal buffers cleared")
            else:
                if strategy_id in self._strategy_buffers:
                    self._strategy_buffers[strategy_id].clear()
                    LOG.info(f"Strategy {strategy_id} buffer cleared")


# Global signal buffer singleton
_global_signal_buffer: Optional[SignalBuffer] = None


def get_signal_buffer() -> SignalBuffer:
    """Get global signal buffer instance"""
    global _global_signal_buffer
    if _global_signal_buffer is None:
        _global_signal_buffer = SignalBuffer()
    return _global_signal_buffer
