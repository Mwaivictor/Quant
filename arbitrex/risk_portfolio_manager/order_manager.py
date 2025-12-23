"""
Order Management Module

Tracks pending orders, partial fills, and slippage.
Provides interface between RPM and execution engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class OrderStatus(str, Enum):
    """Order lifecycle states"""
    PENDING = "PENDING"  # Approved by RPM, sent to execution
    PARTIAL = "PARTIAL"  # Partially filled
    FILLED = "FILLED"    # Completely filled
    REJECTED = "REJECTED"  # Rejected by broker/exchange
    CANCELLED = "CANCELLED"  # Cancelled before fill
    EXPIRED = "EXPIRED"  # Time-based expiry


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """
    Represents a trading order in its lifecycle.
    
    Tracks order from RPM approval through execution.
    """
    order_id: str
    symbol: str
    direction: int  # 1=LONG, -1=SHORT
    approved_units: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_units: float = 0.0
    remaining_units: float = 0.0
    average_fill_price: float = 0.0
    slippage_bps: float = 0.0  # Basis points of slippage
    
    # Metadata
    rpm_config_hash: str = ""
    confidence_score: float = 0.0
    regime: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Fill history
    fills: List['OrderFill'] = field(default_factory=list)
    
    def __post_init__(self):
        self.remaining_units = self.approved_units
    
    def add_fill(
        self,
        fill_units: float,
        fill_price: float,
        fill_timestamp: Optional[datetime] = None,
    ):
        """
        Record a fill (complete or partial).
        
        Args:
            fill_units: Units filled in this execution
            fill_price: Price of this fill
            fill_timestamp: When fill occurred
        """
        if fill_timestamp is None:
            fill_timestamp = datetime.now()
        
        # Create fill record
        fill = OrderFill(
            order_id=self.order_id,
            fill_units=fill_units,
            fill_price=fill_price,
            fill_timestamp=fill_timestamp,
        )
        self.fills.append(fill)
        
        # Update order state
        self.filled_units += fill_units
        self.remaining_units = self.approved_units - self.filled_units
        
        # Recalculate average fill price
        total_value = sum(f.fill_units * f.fill_price for f in self.fills)
        self.average_fill_price = total_value / self.filled_units if self.filled_units > 0 else 0.0
        
        # Update status
        if self.remaining_units <= 0.01:  # Tolerance for rounding
            self.status = OrderStatus.FILLED
            self.filled_at = fill_timestamp
        else:
            self.status = OrderStatus.PARTIAL
    
    def calculate_slippage(self, expected_price: float):
        """
        Calculate slippage vs expected price.
        
        Args:
            expected_price: Expected execution price
        """
        if self.average_fill_price > 0 and expected_price > 0:
            # Slippage in basis points
            # For LONG: negative if filled above expected (bad)
            # For SHORT: negative if filled below expected (bad)
            price_diff = (self.average_fill_price - expected_price) * self.direction
            self.slippage_bps = (price_diff / expected_price) * 10000
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'approved_units': float(self.approved_units),
            'order_type': self.order_type.value,
            'limit_price': float(self.limit_price) if self.limit_price else None,
            'stop_price': float(self.stop_price) if self.stop_price else None,
            'status': self.status.value,
            'filled_units': float(self.filled_units),
            'remaining_units': float(self.remaining_units),
            'average_fill_price': float(self.average_fill_price),
            'slippage_bps': float(self.slippage_bps),
            'rpm_config_hash': self.rpm_config_hash,
            'confidence_score': float(self.confidence_score),
            'regime': self.regime,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'fills': [f.to_dict() for f in self.fills],
        }


@dataclass
class OrderFill:
    """Individual fill within an order (for partial fills)"""
    order_id: str
    fill_units: float
    fill_price: float
    fill_timestamp: datetime
    fill_id: str = ""
    
    def __post_init__(self):
        if not self.fill_id:
            self.fill_id = f"{self.order_id}_{self.fill_timestamp.strftime('%Y%m%d%H%M%S%f')}"
    
    def to_dict(self) -> dict:
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'fill_units': float(self.fill_units),
            'fill_price': float(self.fill_price),
            'fill_timestamp': self.fill_timestamp.isoformat(),
        }


class OrderManager:
    """
    Manages order lifecycle and execution tracking.
    
    Bridges RPM and execution engine.
    """
    
    def __init__(self):
        """Initialize order manager"""
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.order_counter = 0
    
    def create_order(
        self,
        symbol: str,
        direction: int,
        approved_units: float,
        order_type: OrderType = OrderType.MARKET,
        rpm_config_hash: str = "",
        confidence_score: float = 0.0,
        regime: str = "",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Create new order from RPM approval.
        
        Args:
            symbol: Trading symbol
            direction: 1=LONG, -1=SHORT
            approved_units: Units approved by RPM
            order_type: Order type
            rpm_config_hash: Config version for audit trail
            confidence_score: Signal confidence
            regime: Market regime
            limit_price: Limit price (if LIMIT order)
            stop_price: Stop price (if STOP order)
        
        Returns:
            Order: Created order
        """
        self.order_counter += 1
        order_id = f"RPM{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            approved_units=approved_units,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            rpm_config_hash=rpm_config_hash,
            confidence_score=confidence_score,
            regime=regime,
        )
        
        self.active_orders[order_id] = order
        return order
    
    def update_order_fill(
        self,
        order_id: str,
        fill_units: float,
        fill_price: float,
        expected_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Update order with fill information.
        
        Args:
            order_id: Order identifier
            fill_units: Units filled
            fill_price: Execution price
            expected_price: Expected price (for slippage calculation)
        
        Returns:
            Order: Updated order, or None if not found
        """
        order = self.active_orders.get(order_id)
        if not order:
            return None
        
        order.add_fill(fill_units, fill_price)
        
        if expected_price:
            order.calculate_slippage(expected_price)
        
        # Move to completed if fully filled
        if order.status == OrderStatus.FILLED:
            self.completed_orders.append(order)
            del self.active_orders[order_id]
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order to cancel
        
        Returns:
            bool: True if cancelled successfully
        """
        order = self.active_orders.get(order_id)
        if not order:
            return False
        
        order.status = OrderStatus.CANCELLED
        self.completed_orders.append(order)
        del self.active_orders[order_id]
        return True
    
    def get_pending_exposure(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate pending exposure from unfilled orders.
        
        Args:
            symbol: Optional symbol filter
        
        Returns:
            Dict[str, float]: Pending units by symbol
        """
        pending = {}
        
        for order in self.active_orders.values():
            if symbol is None or order.symbol == symbol:
                if order.symbol not in pending:
                    pending[order.symbol] = 0.0
                
                # Add remaining units (considering direction)
                pending[order.symbol] += order.remaining_units * order.direction
        
        return pending
    
    def get_slippage_statistics(self, lookback_orders: int = 100) -> dict:
        """
        Calculate slippage statistics from recent orders.
        
        Args:
            lookback_orders: Number of recent orders to analyze
        
        Returns:
            dict: Slippage statistics
        """
        recent_orders = self.completed_orders[-lookback_orders:]
        filled_orders = [o for o in recent_orders if o.status == OrderStatus.FILLED]
        
        if not filled_orders:
            return {
                'count': 0,
                'avg_slippage_bps': 0.0,
                'median_slippage_bps': 0.0,
                'max_slippage_bps': 0.0,
                'positive_slippage_pct': 0.0,
            }
        
        slippages = [o.slippage_bps for o in filled_orders]
        
        import numpy as np
        return {
            'count': len(filled_orders),
            'avg_slippage_bps': float(np.mean(slippages)),
            'median_slippage_bps': float(np.median(slippages)),
            'max_slippage_bps': float(np.max(np.abs(slippages))),
            'positive_slippage_pct': float(np.mean([s > 0 for s in slippages]) * 100),
        }
    
    def get_order_statistics(self) -> dict:
        """
        Get order execution statistics.
        
        Returns:
            dict: Order statistics
        """
        total_orders = len(self.active_orders) + len(self.completed_orders)
        
        status_counts = {}
        for order in list(self.active_orders.values()) + self.completed_orders:
            status_counts[order.status.value] = status_counts.get(order.status.value, 0) + 1
        
        return {
            'total_orders': total_orders,
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'status_breakdown': status_counts,
            'fill_rate': status_counts.get('FILLED', 0) / max(1, total_orders),
        }
    
    def to_dict(self) -> dict:
        """Export state to dictionary"""
        return {
            'active_orders': {oid: order.to_dict() for oid, order in self.active_orders.items()},
            'completed_orders': [order.to_dict() for order in self.completed_orders],
            'order_counter': self.order_counter,
        }
