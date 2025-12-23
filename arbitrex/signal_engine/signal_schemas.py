"""
Standardized Signal Schemas for Single and Multi-Leg Strategies

Provides unified signal representation for:
- Single-leg directional trades (EURUSD LONG)
- Multi-leg spread trades (EURUSD LONG + GBPUSD SHORT)
- Multi-leg basket trades (Portfolio of correlated pairs)
- Complex synthetic positions

Design Principles:
- Unified schema for all signal types
- Supports arbitrary number of legs
- Immutable signal objects
- Full audit trail
- Thread-safe by design (immutable)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid


class SignalType(str, Enum):
    """Signal type classification"""
    SINGLE_LEG = "single_leg"           # Single instrument directional
    SPREAD = "spread"                   # Two-leg spread (long/short)
    BASKET = "basket"                   # Multiple correlated instruments
    HEDGE = "hedge"                     # Hedging position
    ARBITRAGE = "arbitrage"             # Multi-leg arbitrage


class LegDirection(int, Enum):
    """Direction for individual leg"""
    LONG = 1
    SHORT = -1
    FLAT = 0


class SignalStatus(str, Enum):
    """Signal lifecycle status"""
    PENDING = "pending"                 # Generated, awaiting risk approval
    APPROVED = "approved"               # Risk approved, ready for execution
    REJECTED = "rejected"               # Risk rejected
    ACTIVE = "active"                   # Execution in progress
    FILLED = "filled"                   # Fully executed
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"             # Cancelled before execution
    EXPIRED = "expired"                 # Timeout expired
    FAILED = "failed"                   # Execution failed


@dataclass(frozen=True)
class SignalLeg:
    """
    Single leg of a potentially multi-leg signal.
    
    Immutable to ensure thread safety.
    """
    
    # Instrument identification
    symbol: str                         # e.g., "EURUSD"
    
    # Direction and sizing
    direction: LegDirection             # LONG (1) or SHORT (-1)
    weight: float = 1.0                 # Relative weight in multi-leg (0-1)
    
    # Confidence metrics
    confidence: float = 1.0             # 0-1, leg-specific confidence
    
    # Entry conditions (optional, for execution)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __post_init__(self):
        """Validate leg parameters"""
        assert 0.0 <= self.weight <= 1.0, f"Weight must be 0-1, got {self.weight}"
        assert 0.0 <= self.confidence <= 1.0, f"Confidence must be 0-1, got {self.confidence}"
        if self.entry_price is not None:
            assert self.entry_price > 0, f"Entry price must be positive, got {self.entry_price}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'weight': float(self.weight),
            'confidence': float(self.confidence),
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
        }


@dataclass(frozen=True)
class Signal:
    """
    Standardized signal representation supporting single and multi-leg strategies.
    
    Immutable for thread safety - once created, cannot be modified.
    Status changes create new Signal objects.
    """
    
    # Unique identification
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Signal metadata
    signal_type: SignalType = SignalType.SINGLE_LEG
    strategy_id: str = ""               # e.g., "momentum_v1", "spread_arb_v2"
    strategy_version: str = ""          # Config hash for reproducibility
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeframe: str = "1H"               # Source timeframe
    bar_index: int = 0                  # Bar index when generated
    
    # Signal legs (1+ legs)
    legs: Tuple[SignalLeg, ...] = field(default_factory=tuple)
    
    # Overall signal confidence
    confidence_score: float = 1.0       # 0-1, overall signal confidence
    
    # Signal status
    status: SignalStatus = SignalStatus.PENDING
    
    # Metadata and audit trail
    metadata: Dict[str, any] = field(default_factory=dict)
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate signal"""
        assert len(self.legs) > 0, "Signal must have at least one leg"
        assert 0.0 <= self.confidence_score <= 1.0, f"Confidence must be 0-1, got {self.confidence_score}"
        
        # Validate weights sum to ~1.0 for multi-leg
        if len(self.legs) > 1:
            total_weight = sum(abs(leg.weight) for leg in self.legs)
            assert 0.99 <= total_weight <= 1.01, f"Multi-leg weights must sum to ~1.0, got {total_weight}"
    
    @property
    def is_single_leg(self) -> bool:
        """Check if single-leg signal"""
        return len(self.legs) == 1
    
    @property
    def is_multi_leg(self) -> bool:
        """Check if multi-leg signal"""
        return len(self.legs) > 1
    
    @property
    def symbols(self) -> List[str]:
        """Get all symbols in signal"""
        return [leg.symbol for leg in self.legs]
    
    @property
    def primary_symbol(self) -> str:
        """Get primary symbol (first leg or highest weight)"""
        if not self.legs:
            return ""
        if len(self.legs) == 1:
            return self.legs[0].symbol
        # Return leg with highest weight
        return max(self.legs, key=lambda leg: abs(leg.weight)).symbol
    
    def with_status(self, new_status: SignalStatus) -> 'Signal':
        """Create new Signal with updated status (immutable pattern)"""
        from dataclasses import replace
        return replace(self, status=new_status, updated_at=datetime.utcnow())
    
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type.value,
            'strategy_id': self.strategy_id,
            'strategy_version': self.strategy_version,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'bar_index': self.bar_index,
            'legs': [leg.to_dict() for leg in self.legs],
            'confidence_score': float(self.confidence_score),
            'status': self.status.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_single_leg': self.is_single_leg,
            'is_multi_leg': self.is_multi_leg,
            'symbols': self.symbols,
            'primary_symbol': self.primary_symbol,
        }


# Helper functions for common signal patterns

def create_single_leg_signal(
    symbol: str,
    direction: LegDirection,
    confidence: float,
    strategy_id: str,
    timeframe: str = "1H",
    bar_index: int = 0,
    **kwargs
) -> Signal:
    """Create single-leg directional signal"""
    leg = SignalLeg(
        symbol=symbol,
        direction=direction,
        weight=1.0,
        confidence=confidence
    )
    return Signal(
        signal_type=SignalType.SINGLE_LEG,
        strategy_id=strategy_id,
        timeframe=timeframe,
        bar_index=bar_index,
        legs=(leg,),
        confidence_score=confidence,
        **kwargs
    )


def create_spread_signal(
    long_symbol: str,
    short_symbol: str,
    confidence: float,
    strategy_id: str,
    long_weight: float = 0.5,
    short_weight: float = 0.5,
    timeframe: str = "1H",
    bar_index: int = 0,
    **kwargs
) -> Signal:
    """Create two-leg spread signal (long one, short another)"""
    leg1 = SignalLeg(
        symbol=long_symbol,
        direction=LegDirection.LONG,
        weight=long_weight,
        confidence=confidence
    )
    leg2 = SignalLeg(
        symbol=short_symbol,
        direction=LegDirection.SHORT,
        weight=short_weight,
        confidence=confidence
    )
    return Signal(
        signal_type=SignalType.SPREAD,
        strategy_id=strategy_id,
        timeframe=timeframe,
        bar_index=bar_index,
        legs=(leg1, leg2),
        confidence_score=confidence,
        **kwargs
    )


def create_basket_signal(
    legs: List[Tuple[str, LegDirection, float]],  # (symbol, direction, weight)
    confidence: float,
    strategy_id: str,
    timeframe: str = "1H",
    bar_index: int = 0,
    **kwargs
) -> Signal:
    """Create multi-leg basket signal"""
    signal_legs = tuple(
        SignalLeg(symbol=sym, direction=direction, weight=weight, confidence=confidence)
        for sym, direction, weight in legs
    )
    return Signal(
        signal_type=SignalType.BASKET,
        strategy_id=strategy_id,
        timeframe=timeframe,
        bar_index=bar_index,
        legs=signal_legs,
        confidence_score=confidence,
        **kwargs
    )
