"""
MVCC Portfolio State Manager

Implements Multi-Version Concurrency Control for portfolio management.

Architecture:
- Single-threaded atomic updater (no locks needed for writes)
- Immutable snapshots for parallel reads (MVCC)
- Compare-And-Swap (CAS) for optimistic locking
- Position reservation system
- Broker reconciliation

Thread Safety Model:
    Read: Lock-free (immutable snapshots)
    Write: Single-threaded updater with CAS
    No reader-writer locks needed!
"""

import threading
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple, FrozenSet
from decimal import Decimal
from collections import deque
from dataclasses import replace

from arbitrex.risk_portfolio_manager.portfolio_state import (
    PortfolioSnapshot,
    Position,
    PositionReservation,
    PositionSide,
    ReservationStatus,
    AccountMetrics,
    PortfolioUpdate
)
from arbitrex.risk_portfolio_manager.broker_reconciliation import (
    BrokerReconciliationEngine,
    ReconciliationReport,
    DriftSeverity,
    DriftAction,
)

LOG = logging.getLogger(__name__)


class CASFailureException(Exception):
    """Raised when Compare-And-Swap fails (version mismatch)"""
    pass


class OverAllocationException(Exception):
    """Raised when position reservation would cause over-allocation"""
    pass


class PortfolioStateManager:
    """
    MVCC Portfolio State Manager
    
    Provides:
    - Immutable snapshots for parallel readers
    - Atomic updates with CAS (Compare-And-Swap)
    - Position reservation system
    - Broker reconciliation
    
    Thread Safety:
    - Readers: Lock-free (read snapshot pointer)
    - Writers: Single-threaded (enforced by updater thread)
    """
    
    def __init__(
        self,
        initial_equity: Decimal = Decimal('100000'),
        enable_reconciliation: bool = True,
        reconciliation_interval: float = 60.0,  # seconds (default 60s for production safety)
        reservation_ttl: float = 300.0,  # 5 minutes
        emit_events: bool = True,
        auto_correct_drift: bool = True,  # Enable auto-correction for minor drift
        halt_on_catastrophic_drift: bool = True,  # Halt trading on >5% drift
    ):
        """
        Initialize portfolio state manager.
        
        Args:
            initial_equity: Starting account equity
            enable_reconciliation: Enable broker reconciliation
            reconciliation_interval: Reconciliation frequency (seconds, default 60s)
            reservation_ttl: Default reservation time-to-live (seconds)
            emit_events: Whether to emit events to event bus
            auto_correct_drift: Allow automatic drift correction for 1-5% drift
            halt_on_catastrophic_drift: Emergency halt on >5% drift
        """
        self.initial_equity = initial_equity
        self.reservation_ttl = reservation_ttl
        
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
                    EventType.ORDER_FILLED,
                    self._on_order_filled
                )
                LOG.info("PortfolioStateManager subscribed to signal and execution events")
            except ImportError:
                self._emit_events = False
                LOG.warning("Event bus not available for PortfolioStateManager")
        
        # Current snapshot (atomic pointer)
        # Readers always get consistent snapshot without locks
        self._current_snapshot = PortfolioSnapshot(
            version=0,
            timestamp=datetime.utcnow(),
            positions=frozenset(),
            reservations=frozenset(),
            metrics=AccountMetrics(
                total_equity=initial_equity,
                cash_available=initial_equity,
                cash_reserved=Decimal('0'),
                margin_used=Decimal('0'),
                margin_available=initial_equity,
                total_exposure=Decimal('0'),
                net_exposure=Decimal('0'),
                gross_exposure=Decimal('0'),
                leverage=Decimal('0'),
                total_unrealized_pnl=Decimal('0'),
                total_realized_pnl=Decimal('0'),
                daily_pnl=Decimal('0'),
                total_positions=0,
                long_positions=0,
                short_positions=0
            )
        )
        
        # Update queue for atomic updater thread
        self._update_queue: deque = deque()
        self._update_lock = threading.Lock()  # Only for queue access
        
        # Atomic updater - NEW COMPREHENSIVE SYSTEM
        self._enable_reconciliation = enable_reconciliation
        self._reconciliation_interval = reconciliation_interval
        self._reconciliation_thread: Optional[threading.Thread] = None
        self._broker_connector: Optional[Callable] = None
        
        # Initialize broker reconciliation engine
        self._reconciliation_engine = BrokerReconciliationEngine(
            reconciliation_interval=reconciliation_interval,
            minimal_drift_threshold=0.005,     # 0.5% - log warning
            warning_drift_threshold=0.01,      # 1% - send alert
            critical_drift_threshold=0.05,     # 5% - auto-correct or halt
            auto_correct_enabled=auto_correct_drift,
            halt_on_catastrophic=halt_on_catastrophic_drift,
            alert_callback=self._send_drift_alert,
        )
        self._enable_reconciliation = enable_reconciliation
        self._reconciliation_interval = reconciliation_interval
        self._reconciliation_thread: Optional[threading.Thread] = None
        self._broker_connector: Optional[Callable] = None
        
        # Statistics
        self._total_updates = 0
        self._cas_failures = 0
        self._successful_updates = 0
        self._reservation_conflicts = 0
        self._last_reconciliation_time: Optional[datetime] = None
        
        # Start updater thread
        self.start()
    
    def _on_signal_generated(self, event):
        """
        Handle signal generated events.
        
        This would trigger risk evaluation for the signal.
        """
        signal_id = event.data.get('signal_id')
        strategy_id = event.data.get('strategy_id')
        symbol = event.symbol
        
        LOG.debug(f"Signal {signal_id} generated by {strategy_id} for {symbol}")
        # Risk evaluation would happen here
        # For now, just log
    
    def _on_order_filled(self, event):
        """
        Handle order filled events.
        
        Updates position after order execution.
        """
        symbol = event.symbol
        side = event.data.get('side')
        quantity = event.data.get('quantity')
        price = event.data.get('price')
        
        LOG.debug(f"Order filled: {symbol} {side} {quantity}@{price}")
        # Would call commit_reservation or update_position here
    
    def _publish_event(self, event_type, symbol: Optional[str] = None, data: Dict = None):
        """Publish event to event bus"""
        if self._event_bus is None:
            return
        
        try:
            event = self._Event(
                event_type=event_type,
                symbol=symbol,
                data=data or {}
            )
            self._event_bus.publish(event)
        except Exception as e:
            LOG.warning(f"Failed to publish {event_type.value} event: {e}")
    
    def start(self):
        """Start atomic updater and reconciliation threads"""
        if self._running:
            return
        
        self._running = True
        
        # Start atomic updater
        self._updater_thread = threading.Thread(
            target=self._updater_loop,
            name="PortfolioAtomicUpdater",
            daemon=True
        )
        self._updater_thread.start()
        LOG.info("Portfolio atomic updater started")
        
        # Start reconciliation thread
        if self._enable_reconciliation:
            self._reconciliation_thread = threading.Thread(
                target=self._reconciliation_loop,
                name="BrokerReconciliation",
                daemon=True
            )
            self._reconciliation_thread.start()
            LOG.info("Broker reconciliation started")
    
    def stop(self):
        """Stop all threads"""
        if not self._running:
            return
        
        self._running = False
        
        if self._updater_thread:
            self._updater_thread.join(timeout=5.0)
        
        if self._reconciliation_thread:
            self._reconciliation_thread.join(timeout=5.0)
        
        LOG.info("Portfolio state manager stopped")
    
    def get_snapshot(self) -> PortfolioSnapshot:
        """
        Get current portfolio snapshot (lock-free read).
        
        Returns immutable snapshot - safe for parallel access.
        """
        return self._current_snapshot
    
    def reserve_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        signal_id: str,
        ttl_seconds: Optional[float] = None
    ) -> str:
        """
        Reserve position capacity (prevents over-allocation).
        
        Args:
            symbol: Symbol to reserve
            side: Position side (LONG/SHORT)
            quantity: Quantity to reserve
            signal_id: Signal that triggered reservation
            ttl_seconds: Time-to-live (uses default if None)
            
        Returns:
            Reservation ID
            
        Raises:
            OverAllocationException: If reservation would exceed limits
        """
        reservation_id = str(uuid.uuid4())
        ttl = ttl_seconds or self.reservation_ttl
        
        reservation = PositionReservation(
            reservation_id=reservation_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            signal_id=signal_id,
            status=ReservationStatus.PENDING,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
            updated_at=datetime.utcnow()
        )
        
        # Queue update
        snapshot = self.get_snapshot()
        update = PortfolioUpdate(
            operation="add_reservation",
            symbol=symbol,
            reservation_id=reservation_id,
            reservation=reservation,
            expected_version=snapshot.version
        )
        
        self._enqueue_update(update)
        
        # Publish reservation created event
        if self._emit_events:
            self._publish_event(
                self._EventType.RESERVATION_CREATED,
                symbol=symbol,
                data={
                    'reservation_id': reservation_id,
                    'signal_id': signal_id,
                    'side': side.value,
                    'quantity': str(quantity)
                }
            )
        
        LOG.debug(f"Reserved {quantity} {symbol} {side.value} (reservation_id={reservation_id})")
        return reservation_id
    
    def commit_reservation(
        self,
        reservation_id: str,
        executed_quantity: Decimal,
        avg_price: Decimal
    ) -> Optional[str]:
        """
        Commit reservation to actual position.
        
        Args:
            reservation_id: Reservation to commit
            executed_quantity: Quantity actually executed
            avg_price: Average execution price
            
        Returns:
            Position ID or None if failed
        """
        snapshot = self.get_snapshot()
        reservation = snapshot.get_reservation(reservation_id)
        
        if not reservation:
            LOG.error(f"Reservation {reservation_id} not found")
            return None
        
        # Update reservation status
        updated_reservation = replace(
            reservation,
            status=ReservationStatus.COMMITTED,
            executed_quantity=executed_quantity,
            avg_execution_price=avg_price,
            updated_at=datetime.utcnow()
        )
        
        # Create or update position
        position_id = f"pos_{uuid.uuid4()}"
        existing_position = snapshot.get_position(reservation.symbol)
        
        if existing_position:
            # Update existing position
            new_quantity = existing_position.quantity + executed_quantity
            new_avg_price = (
                (existing_position.quantity * existing_position.avg_entry_price + 
                 executed_quantity * avg_price) / new_quantity
            )
            
            updated_position = replace(
                existing_position,
                quantity=new_quantity,
                avg_entry_price=new_avg_price,
                last_updated=datetime.utcnow()
            )
            position_id = existing_position.position_id
        else:
            # Create new position
            updated_position = Position(
                symbol=reservation.symbol,
                side=reservation.side,
                quantity=executed_quantity,
                avg_entry_price=avg_price,
                current_price=avg_price,  # Will update with market data
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                opened_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                position_id=position_id
            )
        
        # Queue updates (reservation + position)
        self._enqueue_update(PortfolioUpdate(
            operation="update_reservation",
            reservation_id=reservation_id,
            reservation=updated_reservation,
            expected_version=snapshot.version
        ))
        
        self._enqueue_update(PortfolioUpdate(
            operation="update_position" if existing_position else "add_position",
            symbol=reservation.symbol,
            position_id=position_id,
            position=updated_position,
            expected_version=snapshot.version
        ))
        
        # Publish events
        if self._emit_events:
            self._publish_event(
                self._EventType.RESERVATION_COMMITTED,
                symbol=reservation.symbol,
                data={
                    'reservation_id': reservation_id,
                    'position_id': position_id,
                    'executed_quantity': str(executed_quantity),
                    'avg_price': str(avg_price)
                }
            )
            self._publish_event(
                self._EventType.POSITION_UPDATED,
                symbol=reservation.symbol,
                data={
                    'position_id': position_id,
                    'quantity': str(updated_position.quantity),
                    'avg_price': str(updated_position.avg_entry_price),
                    'unrealized_pnl': str(updated_position.unrealized_pnl)
                }
            )
        
        # Publish events
        if self._emit_events:
            self._publish_event(
                self._EventType.RESERVATION_COMMITTED,
                symbol=reservation.symbol,
                data={
                    'reservation_id': reservation_id,
                    'position_id': position_id,
                    'executed_quantity': str(executed_quantity),
                    'avg_price': str(avg_price)
                }
            )
            self._publish_event(
                self._EventType.POSITION_UPDATED,
                symbol=reservation.symbol,
                data={
                    'position_id': position_id,
                    'quantity': str(updated_position.quantity),
                    'avg_price': str(updated_position.avg_entry_price),
                    'unrealized_pnl': str(updated_position.unrealized_pnl)
                }
            )
        
        LOG.info(f"Committed reservation {reservation_id} → position {position_id}")
        return position_id
    
    def release_reservation(self, reservation_id: str, reason: str = "cancelled"):
        """
        Release reservation (cancel).
        
        Args:
            reservation_id: Reservation to release
            reason: Release reason
        """
        snapshot = self.get_snapshot()
        reservation = snapshot.get_reservation(reservation_id)
        
        if not reservation:
            LOG.warning(f"Reservation {reservation_id} not found")
            return
        
        updated_reservation = replace(
            reservation,
            status=ReservationStatus.RELEASED,
            updated_at=datetime.utcnow()
        )
        
        self._enqueue_update(PortfolioUpdate(
            operation="update_reservation",
            reservation_id=reservation_id,
            reservation=updated_reservation,
            expected_version=snapshot.version
        ))
        
        # Publish reservation released event
        if self._emit_events:
            self._publish_event(
                self._EventType.RESERVATION_RELEASED,
                symbol=reservation.symbol,
                data={
                    'reservation_id': reservation_id,
                    'reason': reason
                }
            )
        
        LOG.debug(f"Released reservation {reservation_id}: {reason}")
    
    def update_position_price(self, symbol: str, current_price: Decimal):
        """
        Update position market price and recalculate P&L.
        
        Args:
            symbol: Symbol to update
            current_price: Current market price
        """
        snapshot = self.get_snapshot()
        position = snapshot.get_position(symbol)
        
        if not position:
            return
        
        # Calculate unrealized P&L
        price_diff = current_price - position.avg_entry_price
        if position.side == PositionSide.SHORT:
            price_diff = -price_diff
        unrealized_pnl = position.quantity * price_diff
        
        updated_position = replace(
            position,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            last_updated=datetime.utcnow()
        )
        
        self._enqueue_update(PortfolioUpdate(
            operation="update_position",
            symbol=symbol,
            position_id=position.position_id,
            position=updated_position,
            expected_version=snapshot.version
        ))
    
    def _enqueue_update(self, update: PortfolioUpdate):
        """Add update to queue (thread-safe)"""
        with self._update_lock:
            self._update_queue.append(update)
            self._total_updates += 1
    
    def _updater_loop(self):
        """
        Single-threaded atomic updater loop.
        
        All writes go through this thread → No write-write conflicts!
        """
        while self._running:
            try:
                # Get updates from queue
                updates_to_process = []
                with self._update_lock:
                    while self._update_queue:
                        updates_to_process.append(self._update_queue.popleft())
                
                # Process updates
                for update in updates_to_process:
                    try:
                        self._apply_update(update)
                        self._successful_updates += 1
                    except CASFailureException:
                        self._cas_failures += 1
                        LOG.debug(f"CAS failure for operation {update.operation} (retry #{self._cas_failures})")
                        # Re-queue with updated version
                        update.expected_version = self._current_snapshot.version
                        with self._update_lock:
                            self._update_queue.append(update)
                    except Exception as e:
                        LOG.error(f"Update failed: {e}")
                
                # Sleep if no updates
                if not updates_to_process:
                    threading.Event().wait(0.001)  # 1ms
                    
            except Exception as e:
                LOG.error(f"Updater loop error: {e}")
    
    def _apply_update(self, update: PortfolioUpdate):
        """
        Apply update with CAS (Compare-And-Swap).
        
        Raises:
            CASFailureException: If version mismatch
        """
        current_snapshot = self._current_snapshot
        
        # CAS check
        if update.expected_version != current_snapshot.version:
            raise CASFailureException(
                f"Expected version {update.expected_version}, got {current_snapshot.version}"
            )
        
        # Apply operation
        if update.operation == "add_reservation":
            new_snapshot = self._add_reservation(current_snapshot, update.reservation)
        elif update.operation == "update_reservation":
            new_snapshot = self._update_reservation(current_snapshot, update.reservation)
        elif update.operation == "add_position":
            new_snapshot = self._add_position(current_snapshot, update.position)
        elif update.operation == "update_position":
            new_snapshot = self._update_position(current_snapshot, update.position)
        elif update.operation == "remove_position":
            new_snapshot = self._remove_position(current_snapshot, update.position_id)
        else:
            raise ValueError(f"Unknown operation: {update.operation}")
        
        # Atomic snapshot swap
        self._current_snapshot = new_snapshot
        LOG.debug(f"Applied {update.operation}, version {current_snapshot.version} → {new_snapshot.version}")
    
    def _add_reservation(
        self,
        snapshot: PortfolioSnapshot,
        reservation: PositionReservation
    ) -> PortfolioSnapshot:
        """Add reservation and create new snapshot"""
        new_reservations = frozenset(list(snapshot.reservations) + [reservation])
        new_metrics = self._recalculate_metrics(snapshot.positions, new_reservations)
        
        return replace(
            snapshot,
            version=snapshot.version + 1,
            timestamp=datetime.utcnow(),
            reservations=new_reservations,
            metrics=new_metrics
        )
    
    def _update_reservation(
        self,
        snapshot: PortfolioSnapshot,
        reservation: PositionReservation
    ) -> PortfolioSnapshot:
        """Update reservation and create new snapshot"""
        new_reservations = frozenset(
            reservation if r.reservation_id == reservation.reservation_id else r
            for r in snapshot.reservations
        )
        new_metrics = self._recalculate_metrics(snapshot.positions, new_reservations)
        
        return replace(
            snapshot,
            version=snapshot.version + 1,
            timestamp=datetime.utcnow(),
            reservations=new_reservations,
            metrics=new_metrics
        )
    
    def _add_position(
        self,
        snapshot: PortfolioSnapshot,
        position: Position
    ) -> PortfolioSnapshot:
        """Add position and create new snapshot"""
        new_positions = frozenset(list(snapshot.positions) + [position])
        new_metrics = self._recalculate_metrics(new_positions, snapshot.reservations)
        
        return replace(
            snapshot,
            version=snapshot.version + 1,
            timestamp=datetime.utcnow(),
            positions=new_positions,
            metrics=new_metrics
        )
    
    def _update_position(
        self,
        snapshot: PortfolioSnapshot,
        position: Position
    ) -> PortfolioSnapshot:
        """Update position and create new snapshot"""
        new_positions = frozenset(
            position if p.position_id == position.position_id else p
            for p in snapshot.positions
        )
        new_metrics = self._recalculate_metrics(new_positions, snapshot.reservations)
        
        return replace(
            snapshot,
            version=snapshot.version + 1,
            timestamp=datetime.utcnow(),
            positions=new_positions,
            metrics=new_metrics
        )
    
    def _remove_position(
        self,
        snapshot: PortfolioSnapshot,
        position_id: str
    ) -> PortfolioSnapshot:
        """Remove position and create new snapshot"""
        new_positions = frozenset(
            p for p in snapshot.positions if p.position_id != position_id
        )
        new_metrics = self._recalculate_metrics(new_positions, snapshot.reservations)
        
        return replace(
            snapshot,
            version=snapshot.version + 1,
            timestamp=datetime.utcnow(),
            positions=new_positions,
            metrics=new_metrics
        )
    
    def _recalculate_metrics(
        self,
        positions: FrozenSet[Position],
        reservations: FrozenSet[PositionReservation]
    ) -> AccountMetrics:
        """Recalculate account metrics from positions and reservations"""
        # Calculate position metrics
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)
        total_exposure = sum(p.market_value for p in positions)
        
        long_positions = sum(1 for p in positions if p.side == PositionSide.LONG)
        short_positions = sum(1 for p in positions if p.side == PositionSide.SHORT)
        
        # Calculate reserved cash
        active_reservations = [r for r in reservations if r.is_active]
        cash_reserved = sum(
            r.remaining_quantity * Decimal('1000')  # Simplified - should use actual price
            for r in active_reservations
        )
        
        # Calculate equity
        total_equity = self.initial_equity + total_unrealized_pnl + total_realized_pnl
        cash_available = total_equity - cash_reserved - total_exposure
        
        return AccountMetrics(
            total_equity=total_equity,
            cash_available=max(Decimal('0'), cash_available),
            cash_reserved=cash_reserved,
            margin_used=total_exposure,
            margin_available=total_equity - total_exposure,
            total_exposure=total_exposure,
            net_exposure=total_exposure,  # Simplified
            gross_exposure=total_exposure,
            leverage=total_exposure / total_equity if total_equity > 0 else Decimal('0'),
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            daily_pnl=total_unrealized_pnl,  # Simplified
            total_positions=len(positions),
            long_positions=long_positions,
            short_positions=short_positions
        )
    
    def _reconciliation_loop(self):
        """
        Periodic broker reconciliation loop.
        
        Perform comprehensive broker reconciliation with drift detection.
        
        This is the CRITICAL safety mechanism that prevents catastrophic
        position drift between internal state and broker reality.
        
        Process:
        1. Query broker for current positions and account state
        2. Compare with internal PortfolioSnapshot
        3. Detect drift (position quantity, equity, P&L)
        4. Take action based on severity:
           - < 0.5%: Log warning
           - 0.5-1%: Alert ops team
           - 1-5%: Auto-correct or alert (configurable)
           - > 5%: HALT TRADING, require manual intervention
        """
        while self._running:
            try:
                if self._broker_connector:
                    self._perform_reconciliation()
                
                # Wait for next reconciliation interval
                threading.Event().wait(self._reconciliation_interval)
                
            except Exception as e:
                LOG.error(f"Reconciliation loop error: {e}", exc_info=True)
                threading.Event().wait(self._reconciliation_interval)
    
    def _perform_reconciliation(self):
        """Perform a single reconciliation cycle"""
        if not self._broker_connector:
            LOG.debug("Broker connector not set, skipping reconciliation")
            return
        
        try:
            # Get broker positions and account info
            broker_data = self._broker_connector()
            if not broker_data:
                LOG.warning("Broker connector returned no data")
                return
            
            # Get current portfolio snapshot
            snapshot = self.get_snapshot()
            
            # Extract broker data
            broker_positions = broker_data.get('positions', [])
            broker_equity = broker_data.get('equity', 0)
            
            # Convert broker positions to dict[symbol, position_data]
            broker_positions_dict = {}
            for pos in broker_positions:
                if isinstance(pos, dict):
                    broker_positions_dict[pos.get('symbol')] = {
                        'symbol': pos.get('symbol'),
                        'quantity': pos.get('volume', 0),  # MT5 uses 'volume' for quantity
                        'price_current': pos.get('price_current'),
                        'profit': pos.get('profit'),
                        'type': pos.get('type'),  # 0=BUY, 1=SELL
                    }
            
            # Convert internal positions to dict[symbol, position_data]
            internal_positions_dict = {}
            for pos in snapshot.positions:
                internal_positions_dict[pos.symbol] = {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'side': pos.side.value,
                }
            
            # Perform reconciliation using the engine
            report = self._reconciliation_engine.reconcile(
                internal_positions=internal_positions_dict,
                broker_positions=broker_positions_dict,
                internal_equity=snapshot.metrics.total_equity,
                broker_equity=broker_equity,
            )
            
            # Log reconciliation results
            self._log_reconciliation_report(report)
            
            # If catastrophic drift and trading was halted, trigger kill switch
            if report.trading_halted:
                self._trigger_kill_switch_on_drift(report)
            
            # Update reconciliation timestamp
            self._last_reconciliation_time = datetime.utcnow()
            
        except Exception as e:
            LOG.error(f"Reconciliation error: {e}", exc_info=True)
            
            # On reconciliation failure, consider halting trading as safety measure
            LOG.critical("Reconciliation failed - consider manual trading halt")
    
    def get_stats(self) -> dict:
        """Get manager statistics including reconciliation metrics"""
        snapshot = self.get_snapshot()
        
        # Get reconciliation engine stats
        reconciliation_stats = self._reconciliation_engine.get_stats()
        
        return {
            'current_version': snapshot.version,
            'total_updates': self._total_updates,
            'successful_updates': self._successful_updates,
            'cas_failures': self._cas_failures,
            'cas_success_rate': self._successful_updates / max(self._total_updates, 1),
            'cas_retry_rate': self._cas_failures / max(self._total_updates, 1),
            'reservation_conflicts': self._reservation_conflicts,
            'last_reconciliation': self._last_reconciliation_time.isoformat() if self._last_reconciliation_time else None,
            'positions': snapshot.position_count,
            'reservations': snapshot.reservation_count,
            'active_reservations': snapshot.active_reservation_count,
            
            # NEW: Broker reconciliation metrics
            'reconciliation': reconciliation_stats,
        }
    
    def get_reconciliation_report(self) -> Optional[Dict]:
        """Get the last reconciliation report"""
        last_report = self._reconciliation_engine.last_reconciliation
        return last_report.to_dict() if last_report else None
    
    def force_reconciliation(self) -> Optional[ReconciliationReport]:
        """
        Force immediate reconciliation (for testing or manual intervention).
        
        Returns:
            ReconciliationReport if successful, None otherwise
        """
        LOG.info("Force reconciliation requested")
        self._perform_reconciliation()
        return self._reconciliation_engine.last_reconciliation
    
    def _log_reconciliation_report(self, report: ReconciliationReport):
        """Log reconciliation report with appropriate severity"""
        severity = report.overall_severity
        
        if severity == DriftSeverity.NONE:
            LOG.debug(
                f"Reconciliation OK: {report.matched_positions} positions matched, "
                f"equity drift: {report.equity_drift_pct*100:.3f}%"
            )
        
        elif severity == DriftSeverity.MINIMAL:
            LOG.info(
                f"Minimal drift detected: {report.total_drift_pct*100:.2f}% "
                f"(equity: {report.equity_drift_pct*100:.2f}%)"
            )
        
        elif severity == DriftSeverity.WARNING:
            LOG.warning(
                f"⚠️ WARNING: Drift detected: {report.total_drift_pct*100:.2f}% "
                f"(equity: {report.equity_drift_pct*100:.2f}%) "
                f"- Missing: {report.missing_positions}, Phantom: {report.phantom_positions}"
            )
        
        elif severity == DriftSeverity.CRITICAL:
            LOG.error(
                f"🚨 CRITICAL DRIFT: {report.total_drift_pct*100:.2f}% "
                f"(equity: {report.equity_drift_pct*100:.2f}%) "
                f"- Action: {report.action_taken.value}"
            )
        
        else:  # CATASTROPHIC
            LOG.critical(
                f"🛑 CATASTROPHIC DRIFT: {report.total_drift_pct*100:.2f}% "
                f"(equity: {report.equity_drift_pct*100:.2f}%) "
                f"- Missing: {report.missing_positions}, Phantom: {report.phantom_positions} "
                f"- Trading halted: {report.trading_halted}"
            )
    
    def _send_drift_alert(self, alert_data: Dict):
        """
        Send drift alert to ops team.
        
        Integrates with existing alert systems (Slack, PagerDuty).
        """
        LOG.error(f"DRIFT ALERT: {alert_data.get('message', 'Unknown drift')}")
        
        # Emit event if event bus available
        if self._emit_events and self._event_bus:
            try:
                event = self._Event(
                    event_type=self._EventType.RISK_LIMIT_BREACHED,
                    symbol="PORTFOLIO",
                    data={
                        'alert_type': 'broker_drift',
                        'severity': alert_data.get('severity'),
                        'timestamp': alert_data.get('timestamp'),
                        'drift_pct': alert_data.get('total_drift_pct'),
                        'equity_drift_pct': alert_data.get('equity_drift_pct'),
                        'message': alert_data.get('message'),
                    },
                    timestamp=datetime.utcnow(),
                )
                self._event_bus.publish(event)
            except Exception as e:
                LOG.error(f"Failed to publish drift alert event: {e}")
        
        # TODO: Integrate with external alerting (Slack, PagerDuty)
        # This would call kill_switch.py's alert system
    
    def _trigger_kill_switch_on_drift(self, report: ReconciliationReport):
        """
        Trigger kill switch activation due to catastrophic drift.
        
        This is the nuclear option - halt all trading immediately.
        """
        LOG.critical(
            "Triggering kill switch due to catastrophic broker drift "
            f"({report.total_drift_pct*100:.2f}%)"
        )
        
        # Emit kill switch event
        if self._emit_events and self._event_bus:
            try:
                event = self._Event(
                    event_type=self._EventType.KILL_SWITCH_ACTIVATED,
                    symbol="GLOBAL",
                    data={
                        'reason': 'catastrophic_broker_drift',
                        'drift_pct': report.total_drift_pct * 100,
                        'equity_drift_pct': report.equity_drift_pct * 100,
                        'missing_positions': report.missing_positions,
                        'phantom_positions': report.phantom_positions,
                        'timestamp': report.timestamp.isoformat(),
                        'requires_manual_intervention': True,
                    },
                    timestamp=datetime.utcnow(),
                )
                self._event_bus.publish(event)
                LOG.critical("Kill switch activation event published")
            except Exception as e:
                LOG.error(f"Failed to publish kill switch event: {e}")
        
        # TODO: Direct integration with KillSwitchManager
        # from arbitrex.risk_portfolio_manager.kill_switch import KillSwitchManager
        # kill_switch_manager.activate_kill_switch('global', 'catastrophic_drift')
    
    def set_broker_connector(self, connector: Callable):
        """Set broker connector for reconciliation"""
        self._broker_connector = connector