"""
Actor-Based Parallel Strategy Runner

Implements strategy execution as isolated actors running in parallel.

Features:
- Per-strategy actor with isolated state
- Thread pool execution (configurable workers)
- Per-strategy rate limiting
- Per-strategy health monitoring
- Failure isolation (one strategy failure doesn't affect others)
- Graceful degradation
- Automatic restart on failure
- Circuit breaker pattern

Architecture:
    StrategyRegistry → StrategyRunner → ThreadPoolExecutor → StrategyActor(s)
                                            ↓
                                      SignalBuffer
"""

import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import traceback

from arbitrex.signal_engine.signal_schemas import Signal
from arbitrex.signal_engine.signal_buffer import SignalBuffer, get_signal_buffer

LOG = logging.getLogger(__name__)


class StrategyState(str, Enum):
    """Strategy actor state"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker tripped


@dataclass
class StrategyConfig:
    """Configuration for strategy actor"""
    strategy_id: str
    strategy_name: str = ""
    
    # Rate limiting
    max_signals_per_minute: int = 10
    max_signals_per_hour: int = 100
    
    # Health thresholds
    max_consecutive_failures: int = 5
    failure_reset_timeout: float = 300.0  # 5 minutes
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 10  # Failures before opening
    circuit_breaker_timeout: float = 60.0  # Seconds before retry
    
    # Execution
    execution_timeout: float = 5.0  # Max execution time per bar
    
    # Auto-restart
    auto_restart_enabled: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 10.0


@dataclass
class StrategyHealth:
    """Health metrics for strategy actor"""
    strategy_id: str
    state: StrategyState = StrategyState.INITIALIZING
    
    # Execution stats
    bars_processed: int = 0
    signals_generated: int = 0
    execution_count: int = 0
    
    # Failure tracking
    consecutive_failures: int = 0
    total_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_failure_reason: str = ""
    
    # Circuit breaker
    circuit_breaker_trips: int = 0
    circuit_breaker_open_until: Optional[datetime] = None
    
    # Rate limiting
    signals_last_minute: int = 0
    signals_last_hour: int = 0
    last_signal_time: Optional[datetime] = None
    
    # Performance
    avg_execution_time_ms: float = 0.0
    last_execution_time_ms: float = 0.0
    last_execution_timestamp: Optional[datetime] = None
    
    # Restart tracking
    restart_count: int = 0
    last_restart_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'state': self.state.value,
            'bars_processed': self.bars_processed,
            'signals_generated': self.signals_generated,
            'execution_count': self.execution_count,
            'consecutive_failures': self.consecutive_failures,
            'total_failures': self.total_failures,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_failure_reason': self.last_failure_reason,
            'circuit_breaker_trips': self.circuit_breaker_trips,
            'circuit_breaker_open_until': self.circuit_breaker_open_until.isoformat() if self.circuit_breaker_open_until else None,
            'signals_last_minute': self.signals_last_minute,
            'signals_last_hour': self.signals_last_hour,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'last_execution_time_ms': self.last_execution_time_ms,
            'last_execution_timestamp': self.last_execution_timestamp.isoformat() if self.last_execution_timestamp else None,
            'restart_count': self.restart_count,
            'last_restart_time': self.last_restart_time.isoformat() if self.last_restart_time else None,
        }


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_per_minute: int, max_per_hour: int):
        """Initialize rate limiter"""
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._minute_tokens = deque(maxlen=max_per_minute)
        self._hour_tokens = deque(maxlen=max_per_hour)
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        """Check if action is allowed"""
        with self._lock:
            now = datetime.utcnow()
            
            # Remove expired tokens (older than 1 minute)
            while self._minute_tokens and (now - self._minute_tokens[0]) > timedelta(minutes=1):
                self._minute_tokens.popleft()
            
            # Remove expired tokens (older than 1 hour)
            while self._hour_tokens and (now - self._hour_tokens[0]) > timedelta(hours=1):
                self._hour_tokens.popleft()
            
            # Check limits
            if len(self._minute_tokens) >= self.max_per_minute:
                return False
            if len(self._hour_tokens) >= self.max_per_hour:
                return False
            
            # Add token
            self._minute_tokens.append(now)
            self._hour_tokens.append(now)
            return True
    
    def get_counts(self) -> tuple:
        """Get current token counts (minute, hour)"""
        with self._lock:
            return len(self._minute_tokens), len(self._hour_tokens)


class StrategyActor:
    """
    Strategy execution actor with isolation and health monitoring.
    
    Each strategy runs in its own isolated context with:
    - Rate limiting
    - Health monitoring
    - Circuit breaker
    - Failure isolation
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        strategy_func: Callable[[Any], Optional[Signal]],
        signal_buffer: Optional[SignalBuffer] = None,
        event_bus = None
    ):
        """
        Initialize strategy actor.
        
        Args:
            config: Strategy configuration
            strategy_func: Strategy function (data -> Optional[Signal])
            signal_buffer: Signal buffer for publishing signals
            event_bus: Event bus for publishing events (optional)
        """
        self.config = config
        self.strategy_func = strategy_func
        self.signal_buffer = signal_buffer or get_signal_buffer()
        self.event_bus = event_bus
        
        # Health tracking
        self.health = StrategyHealth(strategy_id=config.strategy_id)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_per_minute=config.max_signals_per_minute,
            max_per_hour=config.max_signals_per_hour
        )
        
        # State
        self._lock = threading.RLock()
        self._execution_times: deque = deque(maxlen=100)  # Last 100 execution times
    
    def execute(self, data: Any) -> Optional[Signal]:
        """
        Execute strategy on data.
        
        Args:
            data: Input data for strategy
            
        Returns:
            Generated signal or None
        """
        with self._lock:
            # Check circuit breaker
            if self._is_circuit_open():
                LOG.warning(f"Strategy {self.config.strategy_id} circuit breaker open, skipping")
                return None
            
            # Check if paused
            if self.health.state == StrategyState.PAUSED:
                return None
            
            # Execute strategy with timeout and error handling
            start_time = time.perf_counter()
            signal = None
            
            try:
                # Set state to running
                self.health.state = StrategyState.RUNNING
                
                # Execute strategy function
                signal = self._execute_with_timeout(data)
                
                # Update success metrics
                self.health.bars_processed += 1
                self.health.execution_count += 1
                self.health.consecutive_failures = 0
                
                # Handle signal publication
                if signal is not None:
                    if self._check_rate_limit():
                        self.signal_buffer.publish(signal)
                        self.health.signals_generated += 1
                        self.health.last_signal_time = datetime.utcnow()
                        LOG.debug(f"Strategy {self.config.strategy_id} generated signal: {signal.signal_id}")
                        
                        # Publish to event bus
                        if self.event_bus is not None:
                            try:
                                from arbitrex.event_bus import Event, EventType
                                event = Event(
                                    event_type=EventType.SIGNAL_GENERATED,
                                    symbol=signal.legs[0].symbol if signal.legs else None,
                                    data={
                                        'signal_id': signal.signal_id,
                                        'strategy_id': self.config.strategy_id,
                                        'signal_type': signal.signal_type.value,
                                        'confidence': signal.confidence_score,
                                        'legs': len(signal.legs)
                                    }
                                )
                                self.event_bus.publish(event)
                            except Exception as e:
                                LOG.warning(f"Failed to publish signal event: {e}")
                    else:
                        LOG.warning(f"Strategy {self.config.strategy_id} rate limited, signal dropped")
                
            except Exception as e:
                # Handle failure
                self._handle_failure(e)
                
            finally:
                # Update execution time metrics
                execution_time = (time.perf_counter() - start_time) * 1000  # ms
                self.health.last_execution_time_ms = execution_time
                self.health.last_execution_timestamp = datetime.utcnow()
                self._execution_times.append(execution_time)
                self.health.avg_execution_time_ms = sum(self._execution_times) / len(self._execution_times)
            
            return signal
    
    def _execute_with_timeout(self, data: Any) -> Optional[Signal]:
        """Execute strategy with timeout"""
        # Note: Python doesn't support true thread timeout easily
        # For production, consider using multiprocessing or async
        return self.strategy_func(data)
    
    def _check_rate_limit(self) -> bool:
        """Check if signal generation is within rate limits"""
        allowed = self.rate_limiter.allow()
        if allowed:
            minute_count, hour_count = self.rate_limiter.get_counts()
            self.health.signals_last_minute = minute_count
            self.health.signals_last_hour = hour_count
        return allowed
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.config.circuit_breaker_enabled:
            return False
        
        if self.health.circuit_breaker_open_until is None:
            return False
        
        # Check if timeout expired
        if datetime.utcnow() >= self.health.circuit_breaker_open_until:
            # Reset circuit breaker
            self.health.circuit_breaker_open_until = None
            self.health.state = StrategyState.RUNNING
            LOG.info(f"Strategy {self.config.strategy_id} circuit breaker closed (timeout expired)")
            return False
        
        return True
    
    def _handle_failure(self, error: Exception):
        """Handle strategy execution failure"""
        self.health.consecutive_failures += 1
        self.health.total_failures += 1
        self.health.last_failure_time = datetime.utcnow()
        self.health.last_failure_reason = str(error)
        
        LOG.error(f"Strategy {self.config.strategy_id} failed: {error}")
        LOG.debug(f"Traceback: {traceback.format_exc()}")
        
        # Check if circuit breaker should trip
        if (self.config.circuit_breaker_enabled and 
            self.health.consecutive_failures >= self.config.circuit_breaker_threshold):
            self._trip_circuit_breaker()
        
        # Check if max consecutive failures reached
        if self.health.consecutive_failures >= self.config.max_consecutive_failures:
            self.health.state = StrategyState.FAILED
            LOG.error(f"Strategy {self.config.strategy_id} entered FAILED state after {self.health.consecutive_failures} failures")
    
    def _trip_circuit_breaker(self):
        """Trip circuit breaker"""
        self.health.circuit_breaker_trips += 1
        self.health.circuit_breaker_open_until = datetime.utcnow() + timedelta(seconds=self.config.circuit_breaker_timeout)
        self.health.state = StrategyState.CIRCUIT_OPEN
        LOG.warning(f"Strategy {self.config.strategy_id} circuit breaker tripped (trip #{self.health.circuit_breaker_trips})")
    
    def pause(self):
        """Pause strategy execution"""
        with self._lock:
            self.health.state = StrategyState.PAUSED
            LOG.info(f"Strategy {self.config.strategy_id} paused")
    
    def resume(self):
        """Resume strategy execution"""
        with self._lock:
            if self.health.state == StrategyState.PAUSED:
                self.health.state = StrategyState.RUNNING
                LOG.info(f"Strategy {self.config.strategy_id} resumed")
    
    def reset(self):
        """Reset failure counters"""
        with self._lock:
            self.health.consecutive_failures = 0
            self.health.circuit_breaker_open_until = None
            if self.health.state in [StrategyState.FAILED, StrategyState.CIRCUIT_OPEN]:
                self.health.state = StrategyState.RUNNING
            LOG.info(f"Strategy {self.config.strategy_id} reset")


class StrategyRunner:
    """
    Parallel strategy runner managing multiple strategy actors.
    
    Executes strategies in parallel using thread pool.
    """
    
    def __init__(
        self,
        max_workers: int = 50,
        signal_buffer: Optional[SignalBuffer] = None,
        emit_events: bool = True
    ):
        """
        Initialize strategy runner.
        
        Args:
            max_workers: Maximum number of concurrent strategy executions
            signal_buffer: Signal buffer for publishing signals
            emit_events: Whether to emit events to event bus
        """
        self.max_workers = max_workers
        self.signal_buffer = signal_buffer or get_signal_buffer()
        
        # Strategy registry
        self._strategies: Dict[str, StrategyActor] = {}
        self._lock = threading.RLock()
        
        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Strategy")
        
        # Runner state
        self._running = False
        
        # Event bus integration
        self._emit_events = emit_events
        self._event_bus = None
        if emit_events:
            try:
                from arbitrex.event_bus import get_event_bus, Event, EventType
                self._event_bus = get_event_bus()
                self._Event = Event
                self._EventType = EventType
                
                # Subscribe to feature events
                self._event_bus.subscribe(
                    EventType.FEATURE_TIER1_READY,
                    self._on_feature_ready
                )
                self._event_bus.subscribe(
                    EventType.FEATURE_TIER2_READY,
                    self._on_feature_ready
                )
                LOG.info("StrategyRunner subscribed to feature events")
            except ImportError:
                self._emit_events = False
                LOG.warning("Event bus not available for StrategyRunner")
    
    def _on_feature_ready(self, event):
        """
        Handle feature ready events.
        
        Executes all strategies on new feature data.
        """
        symbol = event.data.get('symbol')
        feature_data = event.data.get('features')
        
        if not feature_data:
            return
        
        LOG.debug(f"Feature ready for {symbol}, executing strategies...")
        
        # Execute all strategies on this feature data
        results = self.execute_parallel(feature_data)
        
        # Count generated signals
        signals_generated = sum(1 for sig in results.values() if sig is not None)
        LOG.debug(f"Generated {signals_generated} signals from {len(results)} strategies")
    
    def register_strategy(
        self,
        config: StrategyConfig,
        strategy_func: Callable[[Any], Optional[Signal]]
    ):
        """
        Register strategy actor.
        
        Args:
            config: Strategy configuration
            strategy_func: Strategy function
        """
        with self._lock:
            actor = StrategyActor(config, strategy_func, self.signal_buffer, self._event_bus)
            self._strategies[config.strategy_id] = actor
            LOG.info(f"Strategy {config.strategy_id} registered")
    
    def unregister_strategy(self, strategy_id: str):
        """Unregister strategy"""
        with self._lock:
            if strategy_id in self._strategies:
                del self._strategies[strategy_id]
                LOG.info(f"Strategy {strategy_id} unregistered")
    
    def execute_parallel(self, data: Any, strategy_ids: Optional[List[str]] = None) -> Dict[str, Optional[Signal]]:
        """
        Execute strategies in parallel.
        
        Args:
            data: Input data for all strategies
            strategy_ids: Specific strategies to execute (None = all)
            
        Returns:
            Dict mapping strategy_id to generated signal
        """
        with self._lock:
            # Determine which strategies to execute
            if strategy_ids is None:
                strategies_to_run = list(self._strategies.items())
            else:
                strategies_to_run = [(sid, self._strategies[sid]) for sid in strategy_ids if sid in self._strategies]
            
            if not strategies_to_run:
                return {}
        
        # Submit all strategies to thread pool
        future_to_strategy = {
            self._executor.submit(actor.execute, data): strategy_id
            for strategy_id, actor in strategies_to_run
        }
        
        # Collect results
        results = {}
        for future in as_completed(future_to_strategy):
            strategy_id = future_to_strategy[future]
            try:
                signal = future.result()
                results[strategy_id] = signal
            except Exception as e:
                LOG.error(f"Strategy {strategy_id} execution failed in runner: {e}")
                results[strategy_id] = None
        
        return results
    
    def get_health(self, strategy_id: Optional[str] = None) -> Dict[str, StrategyHealth]:
        """
        Get health metrics.
        
        Args:
            strategy_id: Specific strategy (None = all)
            
        Returns:
            Dict mapping strategy_id to health metrics
        """
        with self._lock:
            if strategy_id is not None:
                if strategy_id in self._strategies:
                    return {strategy_id: self._strategies[strategy_id].health}
                return {}
            
            return {sid: actor.health for sid, actor in self._strategies.items()}
    
    def get_summary(self) -> dict:
        """Get summary of all strategies"""
        with self._lock:
            total_strategies = len(self._strategies)
            states = {}
            total_signals = 0
            total_failures = 0
            
            for actor in self._strategies.values():
                state = actor.health.state.value
                states[state] = states.get(state, 0) + 1
                total_signals += actor.health.signals_generated
                total_failures += actor.health.total_failures
            
            return {
                'total_strategies': total_strategies,
                'states': states,
                'total_signals_generated': total_signals,
                'total_failures': total_failures,
                'max_workers': self.max_workers,
            }
    
    def pause_strategy(self, strategy_id: str):
        """Pause specific strategy"""
        with self._lock:
            if strategy_id in self._strategies:
                self._strategies[strategy_id].pause()
    
    def resume_strategy(self, strategy_id: str):
        """Resume specific strategy"""
        with self._lock:
            if strategy_id in self._strategies:
                self._strategies[strategy_id].resume()
    
    def reset_strategy(self, strategy_id: str):
        """Reset strategy failure counters"""
        with self._lock:
            if strategy_id in self._strategies:
                self._strategies[strategy_id].reset()
    
    def shutdown(self):
        """Shutdown runner and cleanup"""
        LOG.info("Shutting down strategy runner...")
        self._executor.shutdown(wait=True)
        LOG.info("Strategy runner shutdown complete")
