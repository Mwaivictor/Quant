"""
MVCC Portfolio State Schemas

Implements Multi-Version Concurrency Control for portfolio/position management.

Key Concepts:
- Immutable snapshots (frozen dataclasses) for parallel reads
- Versioned state with monotonic version numbers
- Position reservations for pending trades
- No locking required for reads (snapshot isolation)

Design:
    PortfolioSnapshot (immutable) ← Multiple readers (parallel risk evaluation)
            ↑
    PortfolioStateManager (single writer thread with CAS)
            ↑
    Position updates (atomic, optimistic locking)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set, FrozenSet
from enum import Enum
from decimal import Decimal


class PositionSide(str, Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ReservationStatus(str, Enum):
    """Reservation lifecycle status"""
    PENDING = "pending"           # Reservation active
    COMMITTED = "committed"       # Converted to actual position
    RELEASED = "released"         # Cancelled/expired
    FAILED = "failed"             # Failed to execute


@dataclass(frozen=True)
class Position:
    """
    Immutable position record.
    
    Thread-safe due to immutability.
    """
    symbol: str
    side: PositionSide
    quantity: Decimal              # Number of units
    avg_entry_price: Decimal       # Average entry price
    current_price: Decimal         # Current market price
    unrealized_pnl: Decimal        # Unrealized profit/loss
    realized_pnl: Decimal          # Realized profit/loss (closed portion)
    
    # Metadata
    opened_at: datetime
    last_updated: datetime
    position_id: str               # Unique position identifier
    
    # Broker tracking
    broker_position_id: Optional[str] = None
    last_reconciled: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': float(self.quantity),
            'avg_entry_price': float(self.avg_entry_price),
            'current_price': float(self.current_price),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'opened_at': self.opened_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'position_id': self.position_id,
            'broker_position_id': self.broker_position_id,
            'last_reconciled': self.last_reconciled.isoformat() if self.last_reconciled else None,
        }
    
    @property
    def market_value(self) -> Decimal:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """Total cost basis"""
        return self.quantity * self.avg_entry_price


@dataclass(frozen=True)
class PositionReservation:
    """
    Immutable position reservation for pending trades.
    
    Prevents over-allocation by "reserving" capacity before execution.
    """
    reservation_id: str            # Unique reservation ID
    symbol: str
    side: PositionSide
    quantity: Decimal              # Reserved quantity
    signal_id: str                 # Signal that created reservation
    
    # Lifecycle
    status: ReservationStatus
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    
    # Execution tracking
    executed_quantity: Decimal = Decimal('0')
    avg_execution_price: Optional[Decimal] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'reservation_id': self.reservation_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': float(self.quantity),
            'signal_id': self.signal_id,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'executed_quantity': float(self.executed_quantity),
            'avg_execution_price': float(self.avg_execution_price) if self.avg_execution_price else None,
        }
    
    @property
    def is_active(self) -> bool:
        """Check if reservation is still active"""
        return self.status == ReservationStatus.PENDING and datetime.utcnow() < self.expires_at
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to execute"""
        return self.quantity - self.executed_quantity


@dataclass(frozen=True)
class AccountMetrics:
    """
    Immutable account-level metrics.
    
    Thread-safe snapshot of account state.
    """
    # Capital
    total_equity: Decimal          # Total account equity
    cash_available: Decimal        # Available cash (not reserved)
    cash_reserved: Decimal         # Cash reserved for pending orders
    margin_used: Decimal           # Margin currently used
    margin_available: Decimal      # Available margin
    
    # Risk metrics
    total_exposure: Decimal        # Total position exposure
    net_exposure: Decimal          # Net long/short exposure
    gross_exposure: Decimal        # Sum of absolute exposures
    leverage: Decimal              # Current leverage ratio
    
    # P&L
    total_unrealized_pnl: Decimal  # Total unrealized P&L
    total_realized_pnl: Decimal    # Total realized P&L
    daily_pnl: Decimal             # Today's P&L
    
    # Position counts
    total_positions: int           # Number of open positions
    long_positions: int            # Number of long positions
    short_positions: int           # Number of short positions
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_equity': float(self.total_equity),
            'cash_available': float(self.cash_available),
            'cash_reserved': float(self.cash_reserved),
            'margin_used': float(self.margin_used),
            'margin_available': float(self.margin_available),
            'total_exposure': float(self.total_exposure),
            'net_exposure': float(self.net_exposure),
            'gross_exposure': float(self.gross_exposure),
            'leverage': float(self.leverage),
            'total_unrealized_pnl': float(self.total_unrealized_pnl),
            'total_realized_pnl': float(self.total_realized_pnl),
            'daily_pnl': float(self.daily_pnl),
            'total_positions': self.total_positions,
            'long_positions': self.long_positions,
            'short_positions': self.short_positions,
        }


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Immutable portfolio snapshot for MVCC.
    
    Represents complete portfolio state at a specific version.
    Thread-safe - multiple readers can access simultaneously without locking.
    """
    # Version control
    version: int                   # Monotonic version number
    timestamp: datetime            # Snapshot timestamp
    
    # Positions (immutable dict)
    positions: FrozenSet[Position] = field(default_factory=frozenset)
    
    # Reservations (immutable dict)
    reservations: FrozenSet[PositionReservation] = field(default_factory=frozenset)
    
    # Account metrics
    metrics: AccountMetrics = field(default_factory=lambda: AccountMetrics(
        total_equity=Decimal('100000'),
        cash_available=Decimal('100000'),
        cash_reserved=Decimal('0'),
        margin_used=Decimal('0'),
        margin_available=Decimal('100000'),
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
    ))
    
    # Reconciliation tracking
    last_broker_sync: Optional[datetime] = None
    broker_sync_status: str = "unknown"  # "synced", "drift_detected", "failed"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'positions': [pos.to_dict() for pos in self.positions],
            'reservations': [res.to_dict() for res in self.reservations],
            'metrics': self.metrics.to_dict(),
            'last_broker_sync': self.last_broker_sync.isoformat() if self.last_broker_sync else None,
            'broker_sync_status': self.broker_sync_status,
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def get_reservation(self, reservation_id: str) -> Optional[PositionReservation]:
        """Get reservation by ID"""
        for res in self.reservations:
            if res.reservation_id == reservation_id:
                return res
        return None
    
    def get_active_reservations(self, symbol: Optional[str] = None) -> Set[PositionReservation]:
        """Get active reservations, optionally filtered by symbol"""
        active = {res for res in self.reservations if res.is_active}
        if symbol:
            active = {res for res in active if res.symbol == symbol}
        return active
    
    def get_reserved_quantity(self, symbol: str, side: PositionSide) -> Decimal:
        """Get total reserved quantity for symbol and side"""
        return sum(
            res.remaining_quantity
            for res in self.reservations
            if res.symbol == symbol and res.side == side and res.is_active
        )
    
    @property
    def position_count(self) -> int:
        """Get number of positions"""
        return len(self.positions)
    
    @property
    def reservation_count(self) -> int:
        """Get number of reservations"""
        return len(self.reservations)
    
    @property
    def active_reservation_count(self) -> int:
        """Get number of active reservations"""
        return len(self.get_active_reservations())


@dataclass
class PortfolioUpdate:
    """
    Portfolio update operation (mutable).
    
    Used internally by atomic updater to describe state changes.
    Not exposed to external readers (they get immutable snapshots).
    """
    # Operation type
    operation: str  # "add_position", "update_position", "remove_position", 
                    # "add_reservation", "update_reservation", "remove_reservation"
    
    # Target
    symbol: Optional[str] = None
    position_id: Optional[str] = None
    reservation_id: Optional[str] = None
    
    # Data
    position: Optional[Position] = None
    reservation: Optional[PositionReservation] = None
    
    # Optimistic locking
    expected_version: int = 0      # Expected current version (CAS)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'operation': self.operation,
            'symbol': self.symbol,
            'position_id': self.position_id,
            'reservation_id': self.reservation_id,
            'position': self.position.to_dict() if self.position else None,
            'reservation': self.reservation.to_dict() if self.reservation else None,
            'expected_version': self.expected_version,
        }
