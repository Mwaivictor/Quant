"""
Parallel Execution Engine with Thread-Pool Groups

Features:
- Thread-pool execution groups for parallel order execution
- Async fill callback queue with <1ms processing
- Multi-venue support with venue failover
- Backpressure handling for execution saturation
- Event bus integration

Architecture:
    Signal → RPM → Parallel Executor → VenueRouter → [Venue1, Venue2, ...]
                         ↓                              ↓
                   ExecutionGroup(s)              FillProcessor
"""

import threading
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import queue

# Kill-switch integration
try:
    from arbitrex.risk_portfolio_manager.kill_switch import KillSwitchManager, KillSwitchLevel
except ImportError:
    KillSwitchManager = None
    KillSwitchLevel = None

LOG = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ERROR = "error"


class VenueStatus(str, Enum):
    """Venue health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ExecutionOrder:
    """Order ready for execution"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: Optional[float] = None  # None = market order
    
    # Metadata
    signal_id: str = ""
    strategy_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Execution preferences
    preferred_venue: Optional[str] = None
    max_slippage: float = 0.001  # 10 bps
    timeout_seconds: float = 30.0
    
    # State
    status: OrderStatus = OrderStatus.PENDING
    venue: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    rejection_reason: Optional[str] = None


@dataclass
class FillEvent:
    """Order fill notification"""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    venue: str = ""
    
    # Fill details
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0  # Time from order submission to fill
    
    # Metadata
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class VenueConnector:
    """Abstract venue connector"""
    venue_id: str
    venue_name: str
    status: VenueStatus = VenueStatus.HEALTHY
    
    # Health metrics
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    avg_latency_ms: float = 0.0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    
    # Connector methods (override in subclass)
    def submit_order(self, order: ExecutionOrder) -> bool:
        """Submit order to venue. Returns True if submitted successfully."""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order. Returns True if cancelled successfully."""
        raise NotImplementedError
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from venue."""
        raise NotImplementedError


class BackpressureController:
    """
    Backpressure handling for execution saturation.
    
    Prevents overload by:
    - Tracking pending order count
    - Rate limiting order submission
    - Queue depth monitoring
    """
    
    def __init__(
        self,
        max_pending_orders: int = 100,
        max_orders_per_second: int = 50,
        queue_warning_threshold: int = 50
    ):
        self.max_pending_orders = max_pending_orders
        self.max_orders_per_second = max_orders_per_second
        self.queue_warning_threshold = queue_warning_threshold
        
        # State
        self._pending_orders: Set[str] = set()
        self._submission_times: deque = deque(maxlen=max_orders_per_second)
        self._lock = threading.RLock()
        
        # Metrics
        self.orders_throttled = 0
        self.orders_rejected_backpressure = 0
    
    def can_accept_order(self) -> bool:
        """Check if system can accept new order (backpressure check)"""
        with self._lock:
            # Check pending order limit
            if len(self._pending_orders) >= self.max_pending_orders:
                LOG.warning(f"Backpressure: {len(self._pending_orders)} pending orders (max={self.max_pending_orders})")
                self.orders_rejected_backpressure += 1
                return False
            
            # Check rate limit
            now = datetime.utcnow()
            # Remove submissions older than 1 second
            while self._submission_times and (now - self._submission_times[0]) > timedelta(seconds=1):
                self._submission_times.popleft()
            
            if len(self._submission_times) >= self.max_orders_per_second:
                self.orders_throttled += 1
                return False
            
            return True
    
    def register_submission(self, order_id: str):
        """Register order submission"""
        with self._lock:
            self._pending_orders.add(order_id)
            self._submission_times.append(datetime.utcnow())
    
    def register_completion(self, order_id: str):
        """Register order completion (fill/reject/cancel)"""
        with self._lock:
            self._pending_orders.discard(order_id)
    
    def get_queue_depth(self) -> int:
        """Get current pending order count"""
        with self._lock:
            return len(self._pending_orders)
    
    def is_saturated(self) -> bool:
        """Check if execution system is saturated"""
        depth = self.get_queue_depth()
        return depth >= self.queue_warning_threshold


class FillProcessor:
    """
    Async fill callback queue with <1ms processing.
    
    Processes order fills in dedicated thread with callback notification.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        
        # Fill queue (thread-safe)
        self._fill_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Callbacks
        self._callbacks: List[Callable[[FillEvent], None]] = []
        self._callback_lock = threading.RLock()
        
        # Processor thread
        self._processor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Metrics
        self.fills_processed = 0
        self.fills_dropped = 0
        self.avg_processing_time_ms = 0.0
        self._processing_times: deque = deque(maxlen=1000)
    
    def start(self):
        """Start fill processor thread"""
        if self._running:
            return
        
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="FillProcessor",
            daemon=True
        )
        self._processor_thread.start()
        LOG.info("FillProcessor started")
    
    def stop(self):
        """Stop fill processor"""
        if not self._running:
            return
        
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        LOG.info("FillProcessor stopped")
    
    def submit_fill(self, fill: FillEvent) -> bool:
        """
        Submit fill for processing.
        
        Returns False if queue is full (fill dropped).
        """
        try:
            self._fill_queue.put_nowait(fill)
            return True
        except queue.Full:
            self.fills_dropped += 1
            LOG.error(f"Fill queue full, dropped fill: {fill.fill_id}")
            return False
    
    def register_callback(self, callback: Callable[[FillEvent], None]):
        """Register fill callback"""
        with self._callback_lock:
            self._callbacks.append(callback)
            LOG.debug(f"Registered fill callback (total={len(self._callbacks)})")
    
    def _processor_loop(self):
        """Process fills from queue"""
        while self._running:
            try:
                # Get fill with timeout
                fill = self._fill_queue.get(timeout=0.1)
                
                # Process fill
                start_time = time.perf_counter()
                self._process_fill(fill)
                processing_time = (time.perf_counter() - start_time) * 1000  # ms
                
                # Update metrics
                self.fills_processed += 1
                self._processing_times.append(processing_time)
                if self._processing_times:
                    self.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)
                
            except queue.Empty:
                continue
            except Exception as e:
                LOG.error(f"Fill processing error: {e}")
    
    def _process_fill(self, fill: FillEvent):
        """Process single fill (call all callbacks)"""
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    LOG.error(f"Fill callback error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fill processor metrics"""
        return {
            'fills_processed': self.fills_processed,
            'fills_dropped': self.fills_dropped,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'queue_depth': self._fill_queue.qsize(),
            'callbacks_registered': len(self._callbacks)
        }


class VenueRouter:
    """
    Multi-venue support with venue failover.
    
    Routes orders to venues with automatic failover on venue failure.
    """
    
    def __init__(self, default_failover_attempts: int = 3):
        self.default_failover_attempts = default_failover_attempts
        
        # Venue registry
        self._venues: Dict[str, VenueConnector] = {}
        self._venue_priority: List[str] = []  # Ordered by priority
        self._lock = threading.RLock()
        
        # Metrics
        self.orders_routed = 0
        self.failover_count = 0
        self.venue_failures: Dict[str, int] = {}
    
    def register_venue(self, connector: VenueConnector, priority: int = 100):
        """
        Register venue connector.
        
        Args:
            connector: Venue connector instance
            priority: Lower number = higher priority
        """
        with self._lock:
            self._venues[connector.venue_id] = connector
            
            # Insert into priority list (sorted by priority)
            insert_idx = 0
            for idx, venue_id in enumerate(self._venue_priority):
                if priority < getattr(self._venues[venue_id], '_priority', 100):
                    insert_idx = idx
                    break
                insert_idx = idx + 1
            
            self._venue_priority.insert(insert_idx, connector.venue_id)
            connector._priority = priority
            
            LOG.info(f"Registered venue: {connector.venue_name} (id={connector.venue_id}, priority={priority})")
    
    def route_order(
        self,
        order: ExecutionOrder,
        failover_attempts: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Route order to venue with failover.
        
        Args:
            order: Order to execute
            failover_attempts: Max failover attempts (None = use default)
            
        Returns:
            (success, venue_id, failure_reason)
        """
        attempts = failover_attempts or self.default_failover_attempts
        
        with self._lock:
            # Get venue list (prefer order.preferred_venue if specified)
            venue_order = self._get_venue_order(order.preferred_venue)
            
            if not venue_order:
                return False, None, "no_venues_available"
            
            # Try venues in order
            for attempt in range(attempts):
                for venue_id in venue_order:
                    venue = self._venues.get(venue_id)
                    
                    if not venue or venue.status == VenueStatus.UNAVAILABLE:
                        continue
                    
                    # Attempt submission
                    try:
                        order.venue = venue_id
                        success = venue.submit_order(order)
                        
                        if success:
                            venue.orders_submitted += 1
                            self.orders_routed += 1
                            LOG.debug(f"Order {order.order_id} routed to {venue.venue_name}")
                            return True, venue_id, None
                        else:
                            # Submission failed, try next venue
                            venue.orders_rejected += 1
                            venue.consecutive_failures += 1
                            venue.last_failure_time = datetime.utcnow()
                            self.venue_failures[venue_id] = self.venue_failures.get(venue_id, 0) + 1
                            
                            # Mark venue as degraded after consecutive failures
                            if venue.consecutive_failures >= 5:
                                venue.status = VenueStatus.DEGRADED
                                LOG.warning(f"Venue {venue.venue_name} marked as degraded")
                            
                    except Exception as e:
                        LOG.error(f"Venue {venue_id} submission error: {e}")
                        venue.consecutive_failures += 1
                        continue
                
                # If we get here, all venues failed this round
                if attempt < attempts - 1:
                    self.failover_count += 1
                    LOG.warning(f"All venues failed, retrying (attempt {attempt + 1}/{attempts})")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
            return False, None, "all_venues_failed"
    
    def _get_venue_order(self, preferred_venue: Optional[str]) -> List[str]:
        """Get ordered list of venues to try"""
        with self._lock:
            # Filter out unavailable venues
            available = [
                vid for vid in self._venue_priority
                if self._venues[vid].status != VenueStatus.UNAVAILABLE
            ]
            
            if not available:
                return []
            
            # If preferred venue specified and available, try it first
            if preferred_venue and preferred_venue in available:
                result = [preferred_venue]
                result.extend([v for v in available if v != preferred_venue])
                return result
            
            return available
    
    def mark_venue_healthy(self, venue_id: str):
        """Mark venue as healthy (reset failure counter)"""
        with self._lock:
            venue = self._venues.get(venue_id)
            if venue:
                venue.status = VenueStatus.HEALTHY
                venue.consecutive_failures = 0
                LOG.info(f"Venue {venue.venue_name} marked healthy")
    
    def get_venue_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all venues"""
        with self._lock:
            return {
                venue_id: {
                    'name': venue.venue_name,
                    'status': venue.status.value,
                    'orders_submitted': venue.orders_submitted,
                    'orders_filled': venue.orders_filled,
                    'orders_rejected': venue.orders_rejected,
                    'consecutive_failures': venue.consecutive_failures,
                    'fill_rate': venue.orders_filled / max(venue.orders_submitted, 1)
                }
                for venue_id, venue in self._venues.items()
            }


class ExecutionGroup:
    """
    Thread-pool execution group.
    
    Executes orders in parallel within the group using thread pool.
    """
    
    def __init__(
        self,
        group_id: str,
        max_workers: int = 10,
        venue_router: Optional[VenueRouter] = None,
        fill_processor: Optional[FillProcessor] = None,
        backpressure: Optional[BackpressureController] = None
    ):
        self.group_id = group_id
        self.max_workers = max_workers
        self.venue_router = venue_router
        self.fill_processor = fill_processor
        self.backpressure = backpressure
        
        # Thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"ExecGroup_{group_id}"
        )
        
        # Active orders
        self._active_orders: Dict[str, ExecutionOrder] = {}
        self._lock = threading.RLock()
        
        # Metrics
        self.orders_submitted = 0
        self.orders_completed = 0
        self.orders_failed = 0
    
    def submit_order(self, order: ExecutionOrder) -> Optional[Future]:
        """
        Submit order for execution.
        
        Returns Future for tracking execution or None if rejected by backpressure.
        """
        # Check backpressure
        if self.backpressure and not self.backpressure.can_accept_order():
            LOG.warning(f"Order {order.order_id} rejected due to backpressure")
            return None
        
        # Register with backpressure
        if self.backpressure:
            self.backpressure.register_submission(order.order_id)
        
        # Track order
        with self._lock:
            self._active_orders[order.order_id] = order
            self.orders_submitted += 1
        
        # Submit to thread pool
        future = self._executor.submit(self._execute_order, order)
        return future
    
    def _execute_order(self, order: ExecutionOrder) -> ExecutionOrder:
        """Execute order (runs in thread pool)"""
        try:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            
            # Route to venue
            if self.venue_router:
                success, venue_id, failure_reason = self.venue_router.route_order(order)
                
                if success:
                    order.status = OrderStatus.FILLED
                    order.filled_at = datetime.utcnow()
                    order.venue = venue_id
                    self.orders_completed += 1
                    
                    # Create fill event
                    if self.fill_processor:
                        latency_ms = (order.filled_at - order.submitted_at).total_seconds() * 1000
                        fill = FillEvent(
                            order_id=order.order_id,
                            venue=venue_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            price=order.price or 0.0,
                            latency_ms=latency_ms
                        )
                        self.fill_processor.submit_fill(fill)
                else:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = failure_reason
                    self.orders_failed += 1
            else:
                # No venue router, mock execution
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self.orders_completed += 1
            
        except Exception as e:
            LOG.error(f"Order execution error: {e}")
            order.status = OrderStatus.ERROR
            order.rejection_reason = str(e)
            self.orders_failed += 1
        
        finally:
            # Cleanup
            with self._lock:
                self._active_orders.pop(order.order_id, None)
            
            if self.backpressure:
                self.backpressure.register_completion(order.order_id)
        
        return order
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution group metrics"""
        with self._lock:
            return {
                'group_id': self.group_id,
                'max_workers': self.max_workers,
                'active_orders': len(self._active_orders),
                'orders_submitted': self.orders_submitted,
                'orders_completed': self.orders_completed,
                'orders_failed': self.orders_failed,
                'success_rate': self.orders_completed / max(self.orders_submitted, 1)
            }
    
    def shutdown(self):
        """Shutdown execution group"""
        LOG.info(f"Shutting down execution group {self.group_id}")
        self._executor.shutdown(wait=True)


class ParallelExecutionEngine:
    """
    Parallel execution engine managing multiple execution groups.
    
    Coordinates execution groups, venue routing, fill processing, and backpressure.
    """
    
    def __init__(
        self,
        num_groups: int = 20,
        workers_per_group: int = 5,
        enable_backpressure: bool = True,
        emit_events: bool = True,
        kill_switch_manager: Optional['KillSwitchManager'] = None
    ):
        self.num_groups = num_groups
        self.workers_per_group = workers_per_group
        
        # Core components
        self.backpressure = BackpressureController() if enable_backpressure else None
        self.fill_processor = FillProcessor()
        self.venue_router = VenueRouter()
        
        # Kill-switch integration
        self.kill_switch_manager = kill_switch_manager
        
        # Execution groups
        self._groups: List[ExecutionGroup] = []
        self._next_group_idx = 0
        self._lock = threading.RLock()
        
        # Initialize groups
        for i in range(num_groups):
            group = ExecutionGroup(
                group_id=f"group_{i}",
                max_workers=workers_per_group,
                venue_router=self.venue_router,
                fill_processor=self.fill_processor,
                backpressure=self.backpressure
            )
            self._groups.append(group)
        
        # Event bus integration
        self._emit_events = emit_events
        self._event_bus = None
        if emit_events:
            try:
                from arbitrex.event_bus import get_event_bus, Event, EventType
                self._event_bus = get_event_bus()
                self._Event = Event
                self._EventType = EventType
                
                # Subscribe to signal events
                self._event_bus.subscribe(
                    EventType.SIGNAL_GENERATED,
                    self._on_signal_generated
                )
                self._event_bus.subscribe(
                    EventType.SIGNAL_APPROVED,
                    self._on_signal_approved
                )
                self._event_bus.subscribe(
                    EventType.RESERVATION_COMMITTED,
                    self._on_reservation_committed
                )
                LOG.info("ParallelExecutionEngine subscribed to signal and portfolio events")
            except ImportError:
                self._emit_events = False
                LOG.warning("Event bus not available for ParallelExecutionEngine")
        
        # Start fill processor
        self.fill_processor.start()
        
        # Register fill callback for event publishing
        if self._emit_events:
            self.fill_processor.register_callback(self._on_fill_event)
        
        LOG.info(f"ParallelExecutionEngine initialized: {num_groups} groups × {workers_per_group} workers = {num_groups * workers_per_group} total workers")
    
    def _on_signal_generated(self, event):
        """
        Handle signal generated events.
        
        Signals are generated by strategies and need risk approval.
        This is informational - we wait for SIGNAL_APPROVED to execute.
        """
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        symbol = event.symbol
        
        LOG.debug(f"Signal {signal_id} generated by {strategy_id} for {symbol} (awaiting approval)")
    
    def _on_signal_approved(self, event):
        """
        Handle approved signal events.
        
        Creates execution order from approved signal and submits for execution.
        """
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        symbol = event.symbol
        
        # Extract order details from event
        side_str = event.data.get('side', 'buy')
        quantity = event.data.get('quantity', 0.0)
        price = event.data.get('price')  # None = market order
        reservation_id = event.data.get('reservation_id')
        
        LOG.info(f"Signal {signal_id} approved for {symbol}, executing order...")
        
        # Check kill-switches before executing
        if self.kill_switch_manager:
            # Extract venue from event data if available
            venue = event.data.get('venue')
            
            if not self.kill_switch_manager.is_trading_allowed(
                strategy_id=strategy_id,
                symbol=symbol,
                venue=venue
            ):
                LOG.warning(f"Kill-switch blocked order for signal {signal_id} ({strategy_id}/{symbol}/{venue})")
                # Publish rejection event
                if self._event_bus:
                    try:
                        rejection_event = self._Event(
                            event_type=self._EventType.ORDER_REJECTED,
                            symbol=symbol,
                            data={
                                'signal_id': signal_id,
                                'strategy_id': strategy_id,
                                'reason': 'kill_switch_active',
                                'venue': venue
                            }
                        )
                        self._event_bus.publish(rejection_event)
                    except Exception as e:
                        LOG.error(f"Failed to publish rejection event: {e}")
                return
        
        # Create execution order
        order = ExecutionOrder(
            symbol=symbol,
            side=OrderSide.BUY if side_str.lower() == 'buy' else OrderSide.SELL,
            quantity=float(quantity),
            price=float(price) if price else None,
            signal_id=signal_id,
            strategy_id=strategy_id
        )
        
        # Submit for execution
        future = self.submit_order(order)
        
        if future:
            LOG.info(f"Order {order.order_id} submitted for signal {signal_id}")
        else:
            LOG.error(f"Failed to submit order for signal {signal_id} (backpressure)")
    
    def _on_reservation_committed(self, event):
        """
        Handle reservation committed events.
        
        This means RPM has approved and reserved capacity for the trade.
        We can now execute with confidence that we have the capital.
        """
        reservation_id = event.data.get('reservation_id')
        position_id = event.data.get('position_id')
        symbol = event.symbol
        
        LOG.debug(f"Reservation {reservation_id} committed to position {position_id} for {symbol}")
    
    def _on_fill_event(self, fill: FillEvent):
        """Handle fill events (publish to event bus)"""
        if self._event_bus:
            try:
                event = self._Event(
                    event_type=self._EventType.ORDER_FILLED,
                    symbol=fill.symbol,
                    data={
                        'fill_id': fill.fill_id,
                        'order_id': fill.order_id,
                        'venue': fill.venue,
                        'side': fill.side.value,
                        'quantity': fill.quantity,
                        'price': fill.price,
                        'latency_ms': fill.latency_ms,
                        'commission': fill.commission,
                        'slippage': fill.slippage
                    }
                )
                self._event_bus.publish(event)
            except Exception as e:
                LOG.warning(f"Failed to publish fill event: {e}")
    
    def submit_order(self, order: ExecutionOrder) -> Optional[Future]:
        """
        Submit order to execution engine.
        
        Uses round-robin to distribute orders across groups.
        """
        # Check kill-switches before submitting
        if self.kill_switch_manager:
            if not self.kill_switch_manager.is_trading_allowed(
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                venue=None  # Venue checked at routing level
            ):
                LOG.warning(f"Kill-switch blocked order {order.order_id} ({order.strategy_id}/{order.symbol})")
                return None
        
        with self._lock:
            # Select group (round-robin)
            group = self._groups[self._next_group_idx]
            self._next_group_idx = (self._next_group_idx + 1) % len(self._groups)
        
        # Submit to group
        future = group.submit_order(order)
        
        # Publish event
        if future and self._emit_events and self._event_bus:
            try:
                event = self._Event(
                    event_type=self._EventType.ORDER_SUBMITTED,
                    symbol=order.symbol,
                    data={
                        'order_id': order.order_id,
                        'signal_id': order.signal_id,
                        'strategy_id': order.strategy_id,
                        'side': order.side.value,
                        'quantity': order.quantity,
                        'group_id': group.group_id
                    }
                )
                self._event_bus.publish(event)
            except Exception as e:
                LOG.warning(f"Failed to publish order submitted event: {e}")
        
        return future
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        group_metrics = [g.get_metrics() for g in self._groups]
        
        total_submitted = sum(g['orders_submitted'] for g in group_metrics)
        total_completed = sum(g['orders_completed'] for g in group_metrics)
        total_failed = sum(g['orders_failed'] for g in group_metrics)
        
        return {
            'num_groups': self.num_groups,
            'workers_per_group': self.workers_per_group,
            'total_workers': self.num_groups * self.workers_per_group,
            'total_orders_submitted': total_submitted,
            'total_orders_completed': total_completed,
            'total_orders_failed': total_failed,
            'overall_success_rate': total_completed / max(total_submitted, 1),
            'backpressure': self.backpressure.__dict__ if self.backpressure else None,
            'fill_processor': self.fill_processor.get_metrics(),
            'venue_health': self.venue_router.get_venue_health(),
            'groups': group_metrics
        }
    
    def shutdown(self):
        """Shutdown execution engine"""
        LOG.info("Shutting down ParallelExecutionEngine...")
        
        # Stop fill processor
        self.fill_processor.stop()
        
        # Shutdown all groups
        for group in self._groups:
            group.shutdown()
        
        LOG.info("ParallelExecutionEngine shutdown complete")
