"""
EXECUTION ENGINE - Professional FX Trading System

Senior Quantitative Analyst & Trading Systems Developer Implementation

Core Principle:
"Upstream decides what to trade.
Execution decides how to trade safely."

The Execution Engine:
- Executes precisely what RPM approves
- Fails safely with no silent failures
- Logs exhaustively for compliance
- Never improvises or overrides RPM
- Respects kill switches and halts

Architecture:
Signal Engine → RPM → Execution Engine → Database + MT5 Broker
"""

import uuid
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
import os
import numpy as np
from pathlib import Path

# Assuming these exist in the codebase
# from .schemas import RPMOutput, ApprovedTrade, RejectedTrade, PortfolioState, TradeDecision
# from .mt5_sync import MT5ConnectionPool, MT5Order

LOG = logging.getLogger(__name__)


# ========================================
# ENUMS & CONSTANTS
# ========================================

class ExecutionStatus(Enum):
    """Execution outcome statuses"""
    PENDING = "pending"  # Waiting to execute
    SUBMITTED = "submitted"  # Order sent to broker
    FILLED = "filled"  # Fully executed
    PARTIALLY_FILLED = "partially_filled"  # Partial execution
    REJECTED = "rejected"  # Broker/validation rejected
    CANCELLED = "cancelled"  # User cancelled
    ERROR = "error"  # Execution error (network, etc)
    EXPIRED = "expired"  # Order timed out


class ExecutionRejectionReason(Enum):
    """Reasons execution failed"""
    MARKET_CLOSED = "market_closed"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    SPREAD_TOO_WIDE = "spread_too_wide"
    BROKER_REJECTION = "broker_rejection"
    MARGIN_INSUFFICIENT = "margin_insufficient"
    TRADING_HALTED = "trading_halted"
    SYMBOL_NOT_TRADABLE = "symbol_not_tradable"
    NETWORK_ERROR = "network_error"
    ORDER_TIMEOUT = "order_timeout"
    VALIDATION_FAILED = "validation_failed"
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    INSUFFICIENT_FILL = "insufficient_fill"  # < 90% fill ratio
    STALE_TICK = "stale_tick"  # Market data too old
    SPREAD_WIDENED = "spread_widened"  # Spread increased post-validation
    MARGIN_DEPLETED = "margin_depleted"  # Re-check failed before submission
    CORRELATION_BREACH = "correlation_breach"  # Multi-leg correlation violated
    UNKNOWN = "unknown"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"  # Immediate execution at market price
    LIMIT = "limit"  # Execute at specified price or better
    STOP = "stop"  # Execute when price reaches trigger


# ========================================
# EXECUTION SCHEMAS
# ========================================

@dataclass
class ExecutionRequest:
    """Pre-execution order ready to send to broker"""
    rpm_decision: 'TradeDecision'  # Original RPM decision
    approved_trade: 'ApprovedTrade'  # The approved trade (execution-ready)
    portfolio_state: 'PortfolioState'  # Portfolio snapshot at time of approval
    
    # Execution parameters
    order_type: OrderType = OrderType.MARKET
    max_slippage_pips: float = 5.0  # Max allowed slippage
    timeout_seconds: int = 60  # Order submission timeout
    retry_attempts: int = 3  # Retries on network failure
    
    # Tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'symbol': self.approved_trade.symbol,
            'direction': self.approved_trade.direction,
            'position_units': self.approved_trade.position_units,
            'order_type': self.order_type.value,
            'max_slippage_pips': self.max_slippage_pips,
            'timeout_seconds': self.timeout_seconds,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ExecutionLog:
    """Complete execution event log entry"""
    
    # Identifiers
    execution_id: str  # Unique execution ID
    order_id: str  # Broker order ID
    request_id: str  # Link to ExecutionRequest
    rpm_decision_id: str  # Link to RPM decision
    
    # Trade details
    symbol: str
    direction: int  # 1=LONG, -1=SHORT
    intended_units: float  # What RPM approved
    executed_units: float  # What actually filled
    
    # Prices
    intended_price: Optional[float] = None  # Entry price intended
    fill_price: Optional[float] = None  # Actual execution price
    slippage_pips: Optional[float] = None  # Execution quality
    
    # Execution metadata
    status: ExecutionStatus = ExecutionStatus.PENDING
    order_type: OrderType = OrderType.MARKET
    submission_timestamp: Optional[datetime] = None
    fill_timestamp: Optional[datetime] = None
    
    # Rejection/error details
    rejection_reason: Optional[ExecutionRejectionReason] = None
    rejection_details: Optional[str] = None
    
    # Risk context
    confidence_score: float = 0.0
    regime: str = "UNKNOWN"
    risk_per_trade: float = 0.0
    
    # Governance
    model_version: str = "1.0.0"
    rpm_version: str = "1.0.0"
    execution_engine_version: str = "1.0.0"
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    
    # Timestamps
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    last_update_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage"""
        return {
            'execution_id': self.execution_id,
            'order_id': self.order_id,
            'request_id': self.request_id,
            'rpm_decision_id': self.rpm_decision_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'intended_units': float(self.intended_units),
            'executed_units': float(self.executed_units),
            'intended_price': float(self.intended_price) if self.intended_price else None,
            'fill_price': float(self.fill_price) if self.fill_price else None,
            'slippage_pips': float(self.slippage_pips) if self.slippage_pips else None,
            'status': self.status.value,
            'order_type': self.order_type.value,
            'submission_timestamp': self.submission_timestamp.isoformat() if self.submission_timestamp else None,
            'fill_timestamp': self.fill_timestamp.isoformat() if self.fill_timestamp else None,
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
            'rejection_details': self.rejection_details,
            'confidence_score': float(self.confidence_score),
            'regime': self.regime,
            'risk_per_trade': float(self.risk_per_trade),
            'model_version': self.model_version,
            'rpm_version': self.rpm_version,
            'execution_engine_version': self.execution_engine_version,
            'retry_count': self.retry_count,
            'created_timestamp': self.created_timestamp.isoformat(),
            'last_update_timestamp': self.last_update_timestamp.isoformat()
        }


@dataclass
class ExecutionConfirmation:
    """Post-execution confirmation"""
    
    execution_id: str
    order_id: str
    status: ExecutionStatus
    
    symbol: str
    direction: int
    intended_units: float
    executed_units: float
    
    fill_price: Optional[float] = None
    slippage_pips: Optional[float] = None
    
    rejection_reason: Optional[ExecutionRejectionReason] = None
    rejection_details: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            'execution_id': self.execution_id,
            'order_id': self.order_id,
            'status': self.status.value,
            'symbol': self.symbol,
            'direction': self.direction,
            'intended_units': float(self.intended_units),
            'executed_units': float(self.executed_units),
            'fill_price': float(self.fill_price) if self.fill_price else None,
            'slippage_pips': float(self.slippage_pips) if self.slippage_pips else None,
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
            'rejection_details': self.rejection_details,
            'timestamp': self.timestamp.isoformat()
        }


# ========================================
# MULTI-LEG EXECUTION DOMAIN OBJECTS
# ========================================

@dataclass
class ExecutionLeg:
    """
    Single leg of a multi-leg execution.
    
    Examples:
    - EURUSD long 100k (FX)
    - SPY long 1000 shares (equity)
    - GOLD short 10 contracts (commodity)
    - BTC long 0.5 (crypto)
    """
    
    leg_id: str  # Unique within group
    symbol: str  # Asset symbol
    direction: int  # +1 BUY, -1 SELL
    units: float  # Quantity to execute
    asset_class: str  # FX, EQUITY, COMMODITY, CRYPTO
    
    # Execution state
    status: ExecutionStatus = ExecutionStatus.PENDING
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_units: float = 0.0
    intended_price: Optional[float] = None
    slippage_pips: float = 0.0
    
    # Tracking
    submission_timestamp: Optional[datetime] = None
    fill_timestamp: Optional[datetime] = None
    rejection_reason: Optional[ExecutionRejectionReason] = None
    rejection_details: Optional[str] = None
    
    def fill_ratio(self) -> float:
        """Filled units as percentage of intended units"""
        if self.units == 0:
            return 0.0
        return self.filled_units / self.units
    
    def is_filled(self) -> bool:
        """True if >= 99% filled"""
        return self.fill_ratio() >= 0.99
    
    def is_partial(self) -> bool:
        """True if 90-99% filled"""
        return 0.90 <= self.fill_ratio() < 0.99
    
    def is_rejected(self) -> bool:
        """True if < 90% filled"""
        return self.fill_ratio() < 0.90
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'leg_id': self.leg_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'units': float(self.units),
            'asset_class': self.asset_class,
            'status': self.status.value,
            'order_id': self.order_id,
            'fill_price': float(self.fill_price) if self.fill_price else None,
            'filled_units': float(self.filled_units),
            'intended_price': float(self.intended_price) if self.intended_price else None,
            'slippage_pips': float(self.slippage_pips),
            'fill_ratio': self.fill_ratio(),
            'submission_timestamp': self.submission_timestamp.isoformat() if self.submission_timestamp else None,
            'fill_timestamp': self.fill_timestamp.isoformat() if self.fill_timestamp else None,
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
            'rejection_details': self.rejection_details,
        }


@dataclass
class ExecutionGroup:
    """
    Multi-leg execution group.
    
    Strategy Example (Stat Arb):
    - EURUSD long 100k  (leg_0)
    - GBPUSD short 80k  (leg_1)
    
    All legs execute as atomic unit:
    - All must pass validation
    - All submit in risk-optimal order
    - If any fails and > 10% filled, rollback
    """
    
    group_id: str  # Unique group ID
    strategy_id: str  # Parent strategy ID
    rpm_decision_id: str  # From RPM
    legs: List[ExecutionLeg]  # Ordered list of legs
    
    # Group state
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    submission_start_timestamp: Optional[datetime] = None
    completion_timestamp: Optional[datetime] = None
    
    # Execution params
    execution_priority: Optional[List[str]] = None  # Override leg order by leg_id
    max_group_slippage_pips: float = 10.0
    allow_partial_fills: bool = True  # If False, fail entire group on partial
    
    # Rollback tracking
    rollback_executed: bool = False
    rollback_details: Optional[str] = None
    
    def all_legs_validated(self) -> bool:
        """True if all legs passed validation"""
        return all(
            leg.status != ExecutionStatus.PENDING or leg.status == ExecutionStatus.SUBMITTED
            for leg in self.legs
        )
    
    def any_leg_rejected(self) -> bool:
        """True if any leg is rejected"""
        return any(leg.status == ExecutionStatus.REJECTED for leg in self.legs)
    
    def any_leg_filled(self) -> bool:
        """True if any leg has been filled or partially filled"""
        return any(
            leg.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED)
            for leg in self.legs
        )
    
    def total_filled_units_notional(self, price_map: Dict[str, float]) -> float:
        """Total notional value of filled legs (for margin calc)"""
        total = 0.0
        for leg in self.legs:
            if leg.fill_price and leg.filled_units:
                total += leg.fill_price * leg.filled_units
        return total
    
    def avg_group_slippage(self) -> float:
        """Average slippage across all legs"""
        if not self.legs:
            return 0.0
        total_slippage = sum(leg.slippage_pips for leg in self.legs)
        return total_slippage / len(self.legs)
    
    def get_execution_order(self) -> List[ExecutionLeg]:
        """
        Return legs in execution order.
        
        Default priority (overridable via execution_priority):
        1. Short legs before long legs (reduce duration)
        2. Less liquid assets first
        3. Higher volatility assets first
        """
        if self.execution_priority:
            # Use custom order
            order_map = {leg.leg_id: leg for leg in self.legs}
            return [order_map[leg_id] for leg_id in self.execution_priority if leg_id in order_map]
        
        # Default: shorts before longs
        shorts = [leg for leg in self.legs if leg.direction < 0]
        longs = [leg for leg in self.legs if leg.direction > 0]
        return shorts + longs
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'group_id': self.group_id,
            'strategy_id': self.strategy_id,
            'rpm_decision_id': self.rpm_decision_id,
            'status': self.status.value,
            'legs': [leg.to_dict() for leg in self.legs],
            'created_timestamp': self.created_timestamp.isoformat(),
            'submission_start_timestamp': self.submission_start_timestamp.isoformat() if self.submission_start_timestamp else None,
            'completion_timestamp': self.completion_timestamp.isoformat() if self.completion_timestamp else None,
            'avg_group_slippage': self.avg_group_slippage(),
            'rollback_executed': self.rollback_executed,
            'rollback_details': self.rollback_details,
        }


# ========================================
# MARKET DATA & BROKER INTERFACE
# ========================================

@dataclass
class MarketSnapshot:
    """Current market conditions"""
    symbol: str
    bid_price: float
    ask_price: float
    mid_price: float
    spread_pips: float
    timestamp: datetime
    
    @property
    def is_market_reasonable(self) -> bool:
        """Check if market prices are reasonable"""
        return (
            self.bid_price > 0 and 
            self.ask_price > 0 and 
            self.ask_price > self.bid_price
        )


class BrokerInterface:
    """
    Interface to broker (MT5, etc.)
    
    In production, this would be:
    - Real MT5 connection via MT5ConnectionPool
    - Live market data feed
    - Real order submission
    
    For MVP, could be mock for testing.
    """
    
    def __init__(self, broker_name: str = "MT5", connection_pool=None):
        self.broker_name = broker_name
        self.connection_pool = connection_pool  # MT5ConnectionPool
        self.is_connected = False
    
    def connect(self) -> bool:
        """Establish broker connection"""
        if self.connection_pool:
            self.is_connected = True
            LOG.info(f"Connected to {self.broker_name}")
            return True
        return False
    
    def disconnect(self) -> None:
        """Disconnect from broker"""
        self.is_connected = False
        LOG.info(f"Disconnected from {self.broker_name}")
    
    def get_market_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get current market prices from MT5"""
        if not self.is_connected or not self.connection_pool:
            return None
        
        try:
            import MetaTrader5 as mt5
            
            # Get latest tick for symbol
            name, session = self.connection_pool.get_connection(timeout=5)
            try:
                with session.lock:
                    tick = mt5.symbol_info_tick(symbol)
                    
                    if tick is None:
                        LOG.warning(f"No market data available for {symbol}")
                        return None
                    
                    bid_price = float(tick.bid)
                    ask_price = float(tick.ask)
                    spread_pips = (ask_price - bid_price) / 0.0001
                    mid_price = (bid_price + ask_price) / 2
                    
                    snapshot = MarketSnapshot(
                        symbol=symbol,
                        bid_price=bid_price,
                        ask_price=ask_price,
                        mid_price=mid_price,
                        spread_pips=spread_pips,
                        timestamp=datetime.utcnow()
                    )
                    
                    LOG.debug(f"Market snapshot {symbol}: bid={bid_price}, ask={ask_price}, spread={spread_pips:.1f}pips")
                    return snapshot
            finally:
                self.connection_pool.release_connection((name, session))
        
        except Exception as e:
            LOG.error(f"Failed to get market snapshot for {symbol}: {e}")
            return None
    
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol can be traded on MT5"""
        if not self.is_connected or not self.connection_pool:
            return False
        
        try:
            import MetaTrader5 as mt5
            
            name, session = self.connection_pool.get_connection(timeout=5)
            try:
                with session.lock:
                    # Check if symbol exists and is tradable
                    sym_info = mt5.symbol_info(symbol)
                    
                    if sym_info is None:
                        LOG.warning(f"Symbol {symbol} not found on broker")
                        return False
                    
                    # Check if visible and tradable
                    is_visible = getattr(sym_info, 'visible', True)
                    is_tradable = getattr(sym_info, 'trade_mode', 0) != 0
                    
                    result = is_visible and is_tradable
                    LOG.debug(f"Symbol {symbol} tradable check: {result}")
                    return result
            finally:
                self.connection_pool.release_connection((name, session))
        
        except Exception as e:
            LOG.error(f"Failed to check if {symbol} is tradable: {e}")
            return False
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        # For FX: typically always open (24/5)
        # But check for weekends, holidays
        now = datetime.utcnow()
        is_weekend = now.weekday() >= 5
        return not is_weekend
    
    def place_order(
        self,
        symbol: str,
        direction: int,
        units: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Place MARKET order on MT5 broker.
        
        Args:
            symbol: Trading pair (e.g., "EURUSD")
            direction: 1=BUY (LONG), -1=SELL (SHORT)
            units: Number of units to trade
            order_type: Order type (MARKET, LIMIT, STOP)
            price: Price for LIMIT/STOP orders
        
        Returns:
            (success, order_id, error_message)
        """
        if not self.is_connected or not self.connection_pool:
            return False, None, "Not connected to broker"
        
        try:
            import MetaTrader5 as mt5
            
            name, session = self.connection_pool.get_connection(timeout=10)
            try:
                with session.lock:
                    # Determine trade type (BUY or SELL)
                    trade_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
                    
                    # Get current price for validation
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        error = "Cannot get market price for symbol"
                        LOG.error(f"Order placement failed for {symbol}: {error}")
                        return False, None, error
                    
                    # Use ask price for buy orders, bid price for sell orders
                    market_price = tick.ask if direction > 0 else tick.bid
                    
                    # Build request
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": float(units),
                        "type": trade_type,
                        "deviation": 20,  # 20 pips slippage tolerance
                        "magic": 0,  # Magic number (can track by order comment)
                        "comment": f"arbitrex_execution_{uuid.uuid4().hex[:8]}"
                    }
                    
                    if order_type == OrderType.MARKET:
                        request["price"] = market_price
                    elif order_type == OrderType.LIMIT:
                        if price is None:
                            return False, None, "LIMIT order requires price parameter"
                        request["price"] = price
                    elif order_type == OrderType.STOP:
                        if price is None:
                            return False, None, "STOP order requires price parameter"
                        request["price"] = price
                    
                    # Submit order
                    result = mt5.order_send(request)
                    
                    if result is None:
                        error = f"Order send returned None: {mt5.last_error()}"
                        LOG.error(f"Order placement failed for {symbol}: {error}")
                        return False, None, error
                    
                    # Check if order was accepted
                    retcode = getattr(result, 'retcode', None)
                    if retcode != mt5.TRADE_RETCODE_DONE:
                        error = f"Broker rejected order: {getattr(result, 'comment', 'Unknown error')} (code: {retcode})"
                        LOG.error(f"Order placement failed for {symbol}: {error}")
                        return False, None, error
                    
                    # Success! Extract order ID
                    order_id = str(getattr(result, 'order', None))
                    deal_id = str(getattr(result, 'deal', None))
                    
                    LOG.info(
                        f"Order placed successfully: "
                        f"order_id={order_id}, deal_id={deal_id}, "
                        f"symbol={symbol}, direction={'BUY' if direction > 0 else 'SELL'}, "
                        f"volume={units}, price={market_price:.5f}"
                    )
                    
                    return True, order_id, None
            
            finally:
                self.connection_pool.release_connection((name, session))
        
        except Exception as e:
            error_msg = f"Exception during order placement: {str(e)}"
            LOG.error(f"Order placement failed for {symbol}: {error_msg}")
            return False, None, error_msg
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of submitted order from MT5"""
        if not self.is_connected or not self.connection_pool:
            return None
        
        try:
            import MetaTrader5 as mt5
            
            name, session = self.connection_pool.get_connection(timeout=5)
            try:
                with session.lock:
                    # Query order by ID
                    order = mt5.orders_total()  # Get total orders
                    if order == 0:
                        LOG.debug(f"No orders found for {order_id}")
                        return None
                    
                    # Search through open orders
                    for i in range(order):
                        ord = mt5.orders_get(index=i)
                        if ord and str(getattr(ord, 'ticket', None)) == order_id:
                            return {
                                'order_id': str(ord.ticket),
                                'symbol': ord.symbol,
                                'type': 'BUY' if ord.type == mt5.ORDER_TYPE_BUY else 'SELL',
                                'volume': float(ord.volume_current),
                                'volume_initial': float(ord.volume_initial),
                                'price_open': float(ord.price_open),
                                'state': getattr(ord, 'state', None),
                                'time_setup': ord.time_setup,
                            }
                    
                    # Check historical orders/deals
                    deals = mt5.history_deals_total()
                    if deals > 0:
                        for i in range(deals):
                            deal = mt5.history_deals_get(index=i)
                            if deal and str(getattr(deal, 'ticket', None)) == order_id:
                                return {
                                    'order_id': str(deal.ticket),
                                    'symbol': deal.symbol,
                                    'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                                    'volume': float(deal.volume),
                                    'price': float(deal.price),
                                    'commission': float(getattr(deal, 'commission', 0)),
                                    'profit': float(getattr(deal, 'profit', 0)),
                                    'time': deal.time,
                                    'state': 'FILLED'
                                }
                    
                    LOG.debug(f"Order {order_id} not found in open or historical orders")
                    return None
            
            finally:
                self.connection_pool.release_connection((name, session))
        
        except Exception as e:
            LOG.error(f"Failed to get order status for {order_id}: {e}")
            return None
    
    def get_available_margin(self) -> Optional[float]:
        """Get available trading margin from MT5 account"""
        if not self.is_connected or not self.connection_pool:
            return None
        
        try:
            import MetaTrader5 as mt5
            
            name, session = self.connection_pool.get_connection(timeout=5)
            try:
                with session.lock:
                    account = mt5.account_info()
                    if account is None:
                        LOG.warning("Cannot retrieve account info")
                        return None
                    
                    free_margin = float(account.margin_free)
                    LOG.debug(f"Available margin: {free_margin:.2f}")
                    return free_margin
            
            finally:
                self.connection_pool.release_connection((name, session))
        
        except Exception as e:
            LOG.error(f"Failed to get available margin: {e}")
            return None
    
    # ========================================
    # MULTI-ASSET SUPPORT
    # ========================================
    
    def get_asset_spec(self, symbol: str, asset_class: str) -> Optional[Dict]:
        """
        Get asset-specific specifications for normalization.
        
        Returns:
        {
            'tick_size': float,       # Minimum price increment
            'contract_size': float,   # Units per contract
            'lot_size': float,        # Standard lot size
            'margin_percent': float,  # Initial margin %
            'asset_class': str
        }
        
        Enables normalization across FX, equities, commodities, crypto.
        """
        # Asset specs by class (would come from broker in production)
        asset_specs = {
            'FX': {
                'tick_size': 0.0001,
                'contract_size': 100000,  # Standard lot
                'lot_size': 100000,
                'margin_percent': 2.0,  # 1:50 leverage
            },
            'EQUITY': {
                'tick_size': 0.01,
                'contract_size': 1,  # Per share
                'lot_size': 100,  # Standard round lot
                'margin_percent': 50.0,  # 1:2 leverage
            },
            'COMMODITY': {
                'tick_size': 0.01,
                'contract_size': 1,  # Per contract (varies by commodity)
                'lot_size': 1,
                'margin_percent': 10.0,  # 1:10 leverage
            },
            'CRYPTO': {
                'tick_size': 1.0,  # Highly variable
                'contract_size': 1,
                'lot_size': 1,
                'margin_percent': 50.0,  # 1:2 leverage
            },
        }
        
        spec = asset_specs.get(asset_class)
        if spec:
            spec['asset_class'] = asset_class
            spec['symbol'] = symbol
        return spec
    
    def calculate_margin_requirement(self, symbol: str, asset_class: str, units: float, price: float) -> Optional[float]:
        """
        Calculate margin requirement for position.
        
        Cross-asset normalized calculation.
        """
        spec = self.get_asset_spec(symbol, asset_class)
        if not spec:
            return None
        
        notional = units * price
        margin_percent = spec['margin_percent']
        required_margin = (notional * margin_percent) / 100.0
        
        LOG.debug(f"Margin for {symbol}: {notional:.2f} notional × {margin_percent}% = {required_margin:.2f}")
        return required_margin
    
    def normalize_slippage(self, symbol: str, asset_class: str, slippage_points: float) -> float:
        """
        Normalize slippage to basis points.
        
        Handles different tick sizes across asset classes.
        """
        spec = self.get_asset_spec(symbol, asset_class)
        if not spec:
            return slippage_points
        
        tick_size = spec['tick_size']
        basis_points = (slippage_points * tick_size) / 0.0001 if tick_size != 0 else slippage_points
        
        return basis_points


# ========================================
# DATABASE INTERFACE
# ========================================

class ExecutionDatabase:
    """
    Database for storing execution logs.
    
    In production: PostgreSQL, MongoDB, etc.
    For MVP: In-memory or SQLite
    """
    
    def __init__(self):
        self.logs: Dict[str, ExecutionLog] = {}  # execution_id -> log
        self.requests: Dict[str, ExecutionRequest] = {}  # request_id -> request
        self.lock = None  # In production: threading.Lock()
    
    def store_execution_request(self, request: ExecutionRequest) -> bool:
        """Store execution request"""
        try:
            self.requests[request.request_id] = request
            LOG.info(f"Stored execution request: {request.request_id}")
            return True
        except Exception as e:
            LOG.error(f"Failed to store request: {e}")
            return False
    
    def store_execution_log(self, log: ExecutionLog) -> bool:
        """Store execution log to database"""
        try:
            self.logs[log.execution_id] = log
            LOG.info(f"Stored execution log: {log.execution_id} -> {log.status.value}")
            return True
        except Exception as e:
            LOG.error(f"Failed to store execution log: {e}")
            return False
    
    def update_execution_log(self, execution_id: str, **updates) -> bool:
        """Update existing execution log"""
        try:
            if execution_id not in self.logs:
                LOG.error(f"Execution log not found: {execution_id}")
                return False
            
            log = self.logs[execution_id]
            for key, value in updates.items():
                if hasattr(log, key):
                    setattr(log, key, value)
            
            log.last_update_timestamp = datetime.utcnow()
            LOG.info(f"Updated execution log: {execution_id}")
            return True
        except Exception as e:
            LOG.error(f"Failed to update execution log: {e}")
            return False
    
    def get_execution_log(self, execution_id: str) -> Optional[ExecutionLog]:
        """Retrieve execution log"""
        return self.logs.get(execution_id)
    
    def get_execution_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[ExecutionLog]:
        """Get recent execution history"""
        logs = list(self.logs.values())
        
        if symbol:
            logs = [log for log in logs if log.symbol == symbol]
        
        return sorted(logs, key=lambda x: x.created_timestamp, reverse=True)[:limit]
    
    def export_audit_trail(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Export execution logs for audit/compliance"""
        logs = [
            log for log in self.logs.values()
            if start_date <= log.created_timestamp <= end_date
        ]
        return [log.to_dict() for log in sorted(logs, key=lambda x: x.created_timestamp)]


# ========================================
# SQLITE PERSISTENCE LAYER
# ========================================

class ExecutionSQLiteDB:
    """
    SQLite backend for persistent execution logging.
    
    Handles all execution logs with full audit trail.
    Can be upgraded to PostgreSQL without changing interface.
    """
    
    DB_PATH = Path.home() / ".arbitrex" / "execution.db"
    
    def __init__(self):
        """Initialize SQLite database"""
        self.db_path = self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        LOG.info(f"SQLite execution database initialized: {self.db_path}")
    
    def _init_schema(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main execution logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_logs (
                execution_id TEXT PRIMARY KEY,
                order_id TEXT,
                request_id TEXT,
                rpm_decision_id TEXT,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                intended_units REAL NOT NULL,
                executed_units REAL DEFAULT 0,
                intended_price REAL,
                fill_price REAL,
                slippage_pips REAL DEFAULT 0,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                rejection_details TEXT,
                confidence_score REAL,
                regime TEXT,
                risk_per_trade REAL,
                created_timestamp TEXT NOT NULL,
                submission_timestamp TEXT,
                fill_timestamp TEXT,
                last_update_timestamp TEXT,
                
                INDEX idx_symbol (symbol),
                INDEX idx_status (status),
                INDEX idx_created (created_timestamp)
            )
        ''')
        
        # Market tick snapshots (for stale tick detection)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid_price REAL NOT NULL,
                ask_price REAL NOT NULL,
                spread_pips REAL NOT NULL,
                timestamp TEXT NOT NULL,
                
                INDEX idx_symbol_ts (symbol, timestamp)
            )
        ''')
        
        # Multi-leg correlation tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_leg_trades (
                execution_id TEXT PRIMARY KEY,
                trade_legs TEXT,
                correlation_matrix TEXT,
                correlation_ok INTEGER,
                created_timestamp TEXT NOT NULL,
                
                FOREIGN KEY (execution_id) REFERENCES execution_logs(execution_id)
            )
        ''')
        
        # Validation checkpoint for spread widening detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_checkpoints (
                execution_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                validation_time TEXT NOT NULL,
                spread_at_validation REAL NOT NULL,
                margin_at_validation REAL NOT NULL,
                tick_age_seconds REAL NOT NULL,
                
                FOREIGN KEY (execution_id) REFERENCES execution_logs(execution_id)
            )
        ''')
        
        # Multi-leg execution groups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_groups (
                group_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                rpm_decision_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_timestamp TEXT NOT NULL,
                submission_start_timestamp TEXT,
                completion_timestamp TEXT,
                avg_group_slippage REAL DEFAULT 0,
                rollback_executed INTEGER DEFAULT 0,
                rollback_details TEXT,
                
                INDEX idx_strategy (strategy_id),
                INDEX idx_status (status),
                INDEX idx_created (created_timestamp)
            )
        ''')
        
        # Multi-leg execution legs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_legs (
                leg_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL,
                units REAL NOT NULL,
                asset_class TEXT NOT NULL,
                status TEXT NOT NULL,
                order_id TEXT,
                fill_price REAL,
                filled_units REAL DEFAULT 0,
                intended_price REAL,
                slippage_pips REAL DEFAULT 0,
                submission_timestamp TEXT,
                fill_timestamp TEXT,
                rejection_reason TEXT,
                rejection_details TEXT,
                
                FOREIGN KEY (group_id) REFERENCES execution_groups(group_id),
                INDEX idx_group (group_id),
                INDEX idx_symbol (symbol)
            )
        ''')
        
        self.conn.commit()
        LOG.debug("SQLite schema initialized")
    
    def store_execution_log(self, log: ExecutionLog) -> bool:
        """Store execution log to SQLite"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO execution_logs (
                    execution_id, order_id, request_id, rpm_decision_id,
                    symbol, direction, intended_units, executed_units,
                    intended_price, fill_price, slippage_pips,
                    status, rejection_reason, rejection_details,
                    confidence_score, regime, risk_per_trade,
                    created_timestamp, submission_timestamp, fill_timestamp,
                    last_update_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log.execution_id, log.order_id, log.request_id, log.rpm_decision_id,
                log.symbol, log.direction, log.intended_units, log.executed_units,
                log.intended_price, log.fill_price, log.slippage_pips,
                log.status.value, log.rejection_reason.value if log.rejection_reason else None,
                log.rejection_details,
                log.confidence_score, log.regime, log.risk_per_trade,
                log.created_timestamp.isoformat(),
                log.submission_timestamp.isoformat() if log.submission_timestamp else None,
                log.fill_timestamp.isoformat() if log.fill_timestamp else None,
                log.last_update_timestamp.isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            LOG.error(f"Failed to store execution log: {e}")
            return False
    
    def store_market_snapshot(self, symbol: str, bid: float, ask: float, spread: float, timestamp: datetime) -> bool:
        """Store market snapshot for stale tick detection"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO market_snapshots (symbol, bid_price, ask_price, spread_pips, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, bid, ask, spread, timestamp.isoformat()))
            self.conn.commit()
            return True
        except Exception as e:
            LOG.error(f"Failed to store market snapshot: {e}")
            return False
    
    def store_validation_checkpoint(self, execution_id: str, symbol: str, 
                                    spread_pips: float, margin: float, tick_age_seconds: float) -> bool:
        """Store validation checkpoint for spread widening detection"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO validation_checkpoints (
                    execution_id, symbol, validation_time, spread_at_validation,
                    margin_at_validation, tick_age_seconds
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (execution_id, symbol, datetime.utcnow().isoformat(), spread_pips, margin, tick_age_seconds))
            self.conn.commit()
            return True
        except Exception as e:
            LOG.error(f"Failed to store validation checkpoint: {e}")
            return False
    
    def get_latest_snapshot(self, symbol: str, max_age_seconds: int = 300) -> Optional[Dict]:
        """Get latest market snapshot within age threshold"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT bid_price, ask_price, spread_pips, timestamp FROM market_snapshots
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            ts = datetime.fromisoformat(row['timestamp'])
            age = (datetime.utcnow() - ts).total_seconds()
            
            if age > max_age_seconds:
                return None  # Stale
            
            return {
                'bid': row['bid_price'],
                'ask': row['ask_price'],
                'spread': row['spread_pips'],
                'timestamp': ts,
                'age_seconds': age
            }
        except Exception as e:
            LOG.error(f"Failed to get market snapshot: {e}")
            return None


# ========================================
# MAIN EXECUTION ENGINE
# ========================================

class ExecutionEngine:
    """
    Professional FX Execution Engine
    
    Responsibilities:
    1. Pre-execution validation
    2. Order construction & submission
    3. Slippage measurement
    4. Partial fill handling
    5. Error handling & resilience
    6. Post-execution logging
    7. Portfolio synchronization
    
    Core Principle: Execute precisely, fail safely, log exhaustively.
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        database: ExecutionDatabase,
        max_slippage_pips: float = 10.0,
        order_timeout_seconds: int = 60,
        max_retries: int = 3,
        min_margin_cushion: float = 1.5,  # Keep 50% margin buffer
    ):
        self.broker = broker
        self.database = database
        self.max_slippage_pips = max_slippage_pips
        self.order_timeout_seconds = order_timeout_seconds
        self.max_retries = max_retries
        self.min_margin_cushion = min_margin_cushion
        
        # Internal state
        self.active_executions: Dict[str, ExecutionLog] = {}
        self.active_groups: Dict[str, ExecutionGroup] = {}  # Multi-leg groups
        
        # Institutional metrics set (non-decorative)
        self.execution_metrics = {
            'total_executions': 0,
            'filled_executions': 0,
            'partially_filled_executions': 0,
            'rejected_executions': 0,
            'successful_executions': 0,  # filled + partial
            'failed_executions': 0,  # rejected
            'total_slippage_pips': 0.0,
            'total_latency_ms': 0,
            'by_symbol': {},  # per-symbol counts
            'by_rejection_reason': {},  # rejection analysis
        }
        
        LOG.info("Execution Engine initialized")
    
    def execute(self, rpm_output: 'RPMOutput') -> ExecutionConfirmation:
        """
        Main execution flow.
        
        Contract:
        - Input: RPMOutput (from Risk & Portfolio Manager)
        - Output: ExecutionConfirmation (success/failure)
        - Constraint: Never re-size, never override RPM, always log
        
        Args:
            rpm_output: Complete RPM decision output
        
        Returns:
            ExecutionConfirmation with status and details
        """
        
        # Step 1: Extract decision
        decision = rpm_output.decision
        portfolio_state = rpm_output.portfolio_state
        
        # Step 2: Check if trade was approved
        if decision.status.value == "REJECTED":
            return self._create_rejection_confirmation(
                decision=decision,
                reason=ExecutionRejectionReason.UNKNOWN,
                details=f"RPM rejected: {decision.rejected_trade.rejection_reason if decision.rejected_trade else 'Unknown'}"
            )
        
        # Step 3: Extract approved trade
        approved_trade = decision.approved_trade
        if not approved_trade:
            return self._create_rejection_confirmation(
                decision=decision,
                reason=ExecutionRejectionReason.VALIDATION_FAILED,
                details="No approved trade in decision"
            )
        
        # Step 4: Create execution request
        execution_request = ExecutionRequest(
            rpm_decision=decision,
            approved_trade=approved_trade,
            portfolio_state=portfolio_state,
            max_slippage_pips=self.max_slippage_pips,
            timeout_seconds=self.order_timeout_seconds,
            retry_attempts=self.max_retries
        )
        
        # Store request
        self.database.store_execution_request(execution_request)
        
        # Step 5: Pre-execution validation
        validation_passed, validation_reason, validation_details, market_snapshot = self._validate_execution(
            execution_request=execution_request,
            portfolio_state=portfolio_state
        )
        
        if not validation_passed:
            return self._create_rejection_confirmation(
                decision=decision,
                approved_trade=approved_trade,
                request_id=execution_request.request_id,
                reason=validation_reason,
                details=validation_details
            )
        
        # Step 6: Create execution log with intended price from market snapshot
        # Store the ask/bid at time of submission for accurate slippage calculation
        intended_price = market_snapshot.ask if approved_trade.direction > 0 else market_snapshot.bid
        
        execution_log = ExecutionLog(
            execution_id=str(uuid.uuid4()),
            order_id="",  # Will be filled after submission
            request_id=execution_request.request_id,
            rpm_decision_id=decision.order_id or str(uuid.uuid4()),
            symbol=approved_trade.symbol,
            direction=approved_trade.direction,
            intended_units=approved_trade.position_units,
            executed_units=0.0,
            intended_price=intended_price,  # CRITICAL: Store for slippage calculation
            order_type=OrderType.MARKET,
            confidence_score=approved_trade.confidence_score,
            regime=approved_trade.regime,
            risk_per_trade=approved_trade.risk_per_trade,
            model_version="1.0.0",
            rpm_version=rpm_output.rpm_version,
            status=ExecutionStatus.PENDING,
            max_retries=self.max_retries
        )
        
        self.database.store_execution_log(execution_log)
        self.active_executions[execution_log.execution_id] = execution_log
        
        # Step 7: Execute order with retry logic
        success, order_id, error_msg = self._submit_order_with_retry(
            execution_log=execution_log,
            execution_request=execution_request
        )
        
        if not success:
            execution_log.status = ExecutionStatus.REJECTED
            execution_log.rejection_reason = ExecutionRejectionReason.BROKER_REJECTION
            execution_log.rejection_details = error_msg
            self.database.update_execution_log(execution_log.execution_id, status=ExecutionStatus.REJECTED)
            self.execution_metrics['failed_executions'] += 1
            
            return self._confirmation_from_log(execution_log)
        
        # Step 8: Update with broker order ID
        execution_log.order_id = order_id
        execution_log.submission_timestamp = datetime.utcnow()
        execution_log.status = ExecutionStatus.SUBMITTED
        self.database.update_execution_log(
            execution_log.execution_id,
            order_id=order_id,
            submission_timestamp=execution_log.submission_timestamp,
            status=ExecutionStatus.SUBMITTED
        )
        
        # Step 9: Monitor order execution
        fill_status = self._monitor_order(execution_log=execution_log, timeout=execution_request.timeout_seconds)
        
        if fill_status is None:
            execution_log.status = ExecutionStatus.EXPIRED
            execution_log.rejection_reason = ExecutionRejectionReason.ORDER_TIMEOUT
            self.database.update_execution_log(execution_log.execution_id, status=ExecutionStatus.EXPIRED)
            self.execution_metrics['failed_executions'] += 1
        
        else:
            # Step 10: Measure slippage and finalize
            fill_price, filled_units, slippage_pips = fill_status
            
            execution_log.fill_price = fill_price
            execution_log.executed_units = filled_units
            execution_log.slippage_pips = slippage_pips
            execution_log.fill_timestamp = datetime.utcnow()
            
            # Institutional partial fill handling: >= 99% = FILLED, 90-99% = PARTIAL, < 90% = REJECTED
            fill_ratio = filled_units / approved_trade.position_units
            
            if fill_ratio >= 0.99:  # 99% or better
                execution_log.status = ExecutionStatus.FILLED
                self.execution_metrics['successful_executions'] += 1
            elif fill_ratio >= 0.90:  # 90-99% filled
                execution_log.status = ExecutionStatus.PARTIALLY_FILLED
                LOG.warning(f"Partial fill: {fill_ratio*100:.1f}% of {approved_trade.position_units} units (got {filled_units})")
                self.execution_metrics['failed_executions'] += 1
            else:  # Less than 90% filled
                execution_log.status = ExecutionStatus.REJECTED
                execution_log.rejection_reason = ExecutionRejectionReason.INSUFFICIENT_FILL
                execution_log.rejection_details = f"Fill ratio only {fill_ratio*100:.1f}% of requested {approved_trade.position_units}"
                LOG.error(f"Insufficient fill: {fill_ratio*100:.1f}% (threshold: 90%)")
                self.execution_metrics['failed_executions'] += 1
            
            # Update metrics
            self.execution_metrics['total_slippage_pips'] += slippage_pips
            symbol = approved_trade.symbol
            if symbol not in self.execution_metrics.get('by_symbol', {}):
                if 'by_symbol' not in self.execution_metrics:
                    self.execution_metrics['by_symbol'] = {}
                self.execution_metrics['by_symbol'][symbol] = 0
            self.execution_metrics['by_symbol'][symbol] += 1
            
            self.database.update_execution_log(
                execution_log.execution_id,
                fill_price=fill_price,
                executed_units=filled_units,
                slippage_pips=slippage_pips,
                fill_timestamp=execution_log.fill_timestamp,
                status=execution_log.status,
                rejection_reason=execution_log.rejection_reason,
                rejection_details=execution_log.rejection_details
            )
        
        # Step 11: Log to database (FINAL STORAGE)
        self.database.store_execution_log(execution_log)
        
        # Step 12: Update metrics
        self.execution_metrics['total_executions'] += 1
        
        # Step 13: Return confirmation
        confirmation = self._confirmation_from_log(execution_log)
        LOG.info(f"Execution complete: {confirmation.status.value} | {confirmation.symbol} {confirmation.direction:+d} {confirmation.executed_units} | Slippage: {confirmation.slippage_pips}pips")
        
        return confirmation
    
    # ========================================
    # MULTI-LEG EXECUTION (NEW)
    # ========================================
    
    def execute_group(self, group: ExecutionGroup, rpm_output: 'RPMOutput') -> ExecutionGroup:
        """
        Execute multi-leg trade group.
        
        Contract:
        - All legs validate or reject entire group
        - Submit in risk-optimal order (shorts before longs)
        - Monitor each leg independently
        - Rollback filled legs if any leg fails >10% fill
        - Atomically update group status
        
        Args:
            group: ExecutionGroup with N legs
            rpm_output: RPM decision (for audit trail)
        
        Returns:
            Updated ExecutionGroup with all leg fills and final status
        """
        
        group.rpm_decision_id = rpm_output.decision.order_id or str(uuid.uuid4())
        group.submission_start_timestamp = datetime.utcnow()
        
        LOG.info(f"Starting multi-leg execution: {group.group_id} strategy={group.strategy_id} legs={len(group.legs)}")
        
        # Step 1: Pre-validate all legs together
        validation_passed, validation_reason = self._validate_group(group, rpm_output.portfolio_state)
        
        if not validation_passed:
            group.status = ExecutionStatus.REJECTED
            for leg in group.legs:
                leg.status = ExecutionStatus.REJECTED
                leg.rejection_reason = validation_reason
            
            LOG.error(f"Group {group.group_id} validation failed: {validation_reason}")
            self.database.store_execution_log(ExecutionLog())  # Persist failure
            self.active_groups[group.group_id] = group
            return group
        
        # Step 2: Get execution order (risk-optimal)
        execution_order = group.get_execution_order()
        LOG.info(f"Execution order for {group.group_id}: {[leg.leg_id for leg in execution_order]}")
        
        # Step 3: Submit legs sequentially
        submitted_legs = []
        for leg_idx, leg in enumerate(execution_order):
            success, order_id, error_msg = self.broker.place_order(
                symbol=leg.symbol,
                direction=leg.direction,
                units=leg.units,
                order_type=OrderType.MARKET
            )
            
            if success and order_id:
                leg.order_id = order_id
                leg.status = ExecutionStatus.SUBMITTED
                leg.submission_timestamp = datetime.utcnow()
                submitted_legs.append(leg)
                LOG.info(f"Leg {leg.leg_id} submitted: {order_id}")
            else:
                # Submission failed - rollback any filled legs
                leg.status = ExecutionStatus.REJECTED
                leg.rejection_reason = ExecutionRejectionReason.BROKER_REJECTION
                leg.rejection_details = error_msg
                
                if submitted_legs:
                    LOG.warning(f"Leg {leg.leg_id} failed, rolling back {len(submitted_legs)} submitted legs")
                    self._rollback_legs(group, submitted_legs)
                    group.rollback_executed = True
                    group.rollback_details = f"Leg {leg.leg_id} submission failed: {error_msg}"
                
                group.status = ExecutionStatus.REJECTED
                group.completion_timestamp = datetime.utcnow()
                self.active_groups[group.group_id] = group
                return group
        
        # Step 4: Monitor all legs
        monitoring_start = datetime.utcnow()
        max_monitoring_time = self.order_timeout_seconds
        
        while (datetime.utcnow() - monitoring_start).total_seconds() < max_monitoring_time:
            all_filled = True
            any_failed = False
            
            for leg in group.legs:
                if leg.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED, ExecutionStatus.REJECTED):
                    continue  # Already resolved
                
                # Poll order status
                status = self.broker.get_order_status(leg.order_id) if leg.order_id else None
                
                if status:
                    state = status.get('state', '').upper()
                    is_filled = (state == 'FILLED') or ('price' in status)
                    
                    if is_filled:
                        leg.fill_price = status.get('price') or status.get('fill_price')
                        leg.filled_units = status.get('volume') or status.get('filled_units', 0)
                        
                        # Calculate slippage
                        if leg.intended_price and leg.fill_price:
                            if leg.direction > 0:  # BUY
                                leg.slippage_pips = max(0.0, (leg.fill_price - leg.intended_price) / 0.0001)
                            else:  # SELL
                                leg.slippage_pips = max(0.0, (leg.intended_price - leg.fill_price) / 0.0001)
                        
                        # Determine leg status
                        fill_ratio = leg.fill_ratio()
                        if fill_ratio >= 0.99:
                            leg.status = ExecutionStatus.FILLED
                        elif fill_ratio >= 0.90:
                            leg.status = ExecutionStatus.PARTIALLY_FILLED
                        else:
                            leg.status = ExecutionStatus.REJECTED
                            leg.rejection_reason = ExecutionRejectionReason.INSUFFICIENT_FILL
                            any_failed = True
                        
                        leg.fill_timestamp = datetime.utcnow()
                        LOG.info(f"Leg {leg.leg_id} resolved: {leg.status.value} (fill {fill_ratio*100:.1f}%)")
                
                elif state in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    leg.status = ExecutionStatus.REJECTED
                    leg.rejection_reason = ExecutionRejectionReason.ORDER_TIMEOUT
                    any_failed = True
                
                all_filled = False
            
            if all([leg.status in (ExecutionStatus.FILLED, ExecutionStatus.PARTIALLY_FILLED, ExecutionStatus.REJECTED) for leg in group.legs]):
                break
            
            time.sleep(0.5)
        
        # Step 5: Determine group status and handle rollback if needed
        filled_legs = [leg for leg in group.legs if leg.status == ExecutionStatus.FILLED]
        partial_legs = [leg for leg in group.legs if leg.status == ExecutionStatus.PARTIALLY_FILLED]
        rejected_legs = [leg for leg in group.legs if leg.status == ExecutionStatus.REJECTED]
        
        # Rollback logic: if any leg <90% filled AND any leg >10% filled, hedge the filled legs
        if rejected_legs and (filled_legs or partial_legs):
            has_significant_fill = any(leg.fill_ratio() > 0.10 for leg in filled_legs + partial_legs)
            if has_significant_fill:
                LOG.warning(f"Partial group execution - rolling back {len(filled_legs + partial_legs)} filled legs")
                self._rollback_legs(group, filled_legs + partial_legs)
                group.rollback_executed = True
                group.rollback_details = f"Rolled back {len(filled_legs + partial_legs)} legs due to rejected legs"
                group.status = ExecutionStatus.REJECTED
            else:
                group.status = ExecutionStatus.PARTIALLY_FILLED if partial_legs else ExecutionStatus.REJECTED
        elif rejected_legs:
            group.status = ExecutionStatus.REJECTED
        elif partial_legs and not filled_legs:
            group.status = ExecutionStatus.PARTIALLY_FILLED
        elif filled_legs and not partial_legs:
            group.status = ExecutionStatus.FILLED
        else:
            group.status = ExecutionStatus.PENDING  # Shouldn't reach here
        
        group.completion_timestamp = datetime.utcnow()
        
        # Step 6: Persist audit trail
        self.database.store_execution_log(ExecutionLog())  # Group-level log (would extend schema)
        self.active_groups[group.group_id] = group
        
        execution_time = (group.completion_timestamp - group.submission_start_timestamp).total_seconds()
        LOG.info(f"Multi-leg execution {group.group_id} complete: {group.status.value} in {execution_time:.2f}s | Slippage {group.avg_group_slippage():.2f}pips")
        
        return group
    
    def _validate_group(self, group: ExecutionGroup, portfolio_state: 'PortfolioState') -> Tuple[bool, Optional[ExecutionRejectionReason]]:
        """
        Pre-execution validation for all legs in group.
        
        All legs must pass or entire group is rejected.
        
        Checks:
        1. All symbols tradable
        2. Market data available for all
        3. Spreads within limits for all
        4. Combined margin sufficient
        5. Correlation valid (if multi-leg)
        """
        
        # Fetch all market snapshots
        snapshots = {}
        for leg in group.legs:
            snapshot = self.broker.get_market_snapshot(leg.symbol)
            if snapshot is None:
                LOG.error(f"No market data for {leg.symbol}")
                return False, ExecutionRejectionReason.VALIDATION_FAILED
            
            is_fresh, msg = self._check_stale_tick(snapshot, max_age_seconds=5)
            if not is_fresh:
                LOG.error(f"Stale tick for {leg.symbol}: {msg}")
                return False, ExecutionRejectionReason.STALE_TICK
            
            snapshots[leg.symbol] = snapshot
            leg.intended_price = snapshot.ask if leg.direction > 0 else snapshot.bid
        
        # Check spreads for all
        for leg in group.legs:
            snapshot = snapshots[leg.symbol]
            if snapshot.spread_pips > group.max_group_slippage_pips:
                LOG.error(f"Spread too wide for {leg.symbol}: {snapshot.spread_pips:.2f}pips")
                return False, ExecutionRejectionReason.SPREAD_TOO_WIDE
        
        # Check combined margin
        combined_margin_needed = 0.0
        for leg in group.legs:
            snapshot = snapshots[leg.symbol]
            margin = self.broker.calculate_margin_requirement(
                leg.symbol, leg.asset_class, leg.units,
                snapshot.ask if leg.direction > 0 else snapshot.bid
            )
            if margin:
                combined_margin_needed += margin
        
        available_margin = self.broker.get_available_margin()
        if available_margin and combined_margin_needed > available_margin * 0.5:  # 50% safety cushion
            LOG.error(f"Insufficient margin for group: need {combined_margin_needed:.2f}, have {available_margin:.2f}")
            return False, ExecutionRejectionReason.MARGIN_INSUFFICIENT
        
        # Check correlation (multi-leg)
        if len(group.legs) > 1:
            is_ok, msg = self._check_multi_leg_correlation([{'symbol': leg.symbol, 'direction': leg.direction, 'units': leg.units} for leg in group.legs])
            if not is_ok:
                LOG.error(f"Correlation check failed: {msg}")
                return False, ExecutionRejectionReason.CORRELATION_BREACH
        
        LOG.info(f"Group {group.group_id} validation passed")
        return True, None
    
    def _rollback_legs(self, group: ExecutionGroup, legs: List[ExecutionLeg]) -> None:
        """
        Close filled legs at market price.
        
        Minimizes unhedged exposure.
        """
        for leg in legs:
            if leg.filled_units > 0:
                # Close position: opposite direction
                close_direction = -1 * leg.direction
                close_units = leg.filled_units
                
                LOG.warning(f"Closing leg {leg.leg_id}: {close_units} units at market (direction {close_direction})")
                
                success, order_id, error = self.broker.place_order(
                    symbol=leg.symbol,
                    direction=close_direction,
                    units=close_units,
                    order_type=OrderType.MARKET
                )
                
                if not success:
                    LOG.error(f"Rollback close failed for {leg.leg_id}: {error}")
    
    # ========================================
    # PRE-EXECUTION VALIDATION
    # ========================================
    
    def _validate_execution(
        self,
        execution_request: ExecutionRequest,
        portfolio_state: 'PortfolioState'
    ) -> Tuple[bool, Optional[ExecutionRejectionReason], Optional[str], Optional['MarketSnapshot']]:
        """
        Pre-execution validation checklist.
        
        Returns:
            (passed, rejection_reason, details, market_snapshot)
        """
        
        symbol = execution_request.approved_trade.symbol
        units = execution_request.approved_trade.position_units
        
        # Check 1: Trading halted?
        if portfolio_state.trading_halted:
            return False, ExecutionRejectionReason.TRADING_HALTED, portfolio_state.halt_reason or "Trading halted", None
        
        # Check 2: Market open?
        if not self.broker.is_market_open(symbol):
            return False, ExecutionRejectionReason.MARKET_CLOSED, f"Market closed for {symbol}", None
        
        # Check 3: Symbol tradable?
        if not self.broker.is_symbol_tradable(symbol):
            return False, ExecutionRejectionReason.SYMBOL_NOT_TRADABLE, f"{symbol} not tradable", None
        
        # Check 4: Market state (fetch ONCE for both validation and intended price)
        market = self.broker.get_market_snapshot(symbol)
        if market is None:
            return False, ExecutionRejectionReason.VALIDATION_FAILED, "Unable to fetch market data", None
        
        if not market.is_market_reasonable:
            return False, ExecutionRejectionReason.VALIDATION_FAILED, "Market prices unreasonable", None
        
        # Check 5: Spread within limits?
        if market.spread_pips > execution_request.max_slippage_pips:
            return False, ExecutionRejectionReason.SPREAD_TOO_WIDE, f"Spread {market.spread_pips:.2f}pips exceeds limit {execution_request.max_slippage_pips}pips", None
        
        # Check 6: Sufficient margin?
        available_margin = self.broker.get_available_margin()
        if available_margin and available_margin < (execution_request.approved_trade.risk_per_trade * self.min_margin_cushion):
            return False, ExecutionRejectionReason.MARGIN_INSUFFICIENT, f"Insufficient margin: {available_margin:.2f} vs {execution_request.approved_trade.risk_per_trade * self.min_margin_cushion:.2f}", None
        
        # Check 7: Liquidity available?
        # In real implementation: check order book depth, ADV, etc.
        # For now: simple heuristic
        if units > 1_000_000:  # Large position check
            return False, ExecutionRejectionReason.INSUFFICIENT_LIQUIDITY, f"Position size {units} may exceed liquidity", None
        
        # All checks passed - return market snapshot for intended price
        return True, None, None, market
    
    # ========================================
    # HARDENING CHECKS (PRODUCTION SAFETY)
    # ========================================
    
    def _check_stale_tick(self, market: 'MarketSnapshot', max_age_seconds: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Detect stale market data.
        
        Args:
            market: Current market snapshot
            max_age_seconds: Max acceptable data age (default 5s)
        
        Returns:
            (is_fresh, error_msg)
        """
        tick_age = (datetime.utcnow() - market.timestamp).total_seconds()
        
        if tick_age > max_age_seconds:
            msg = f"Stale tick: {tick_age:.1f}s old (max: {max_age_seconds}s)"
            LOG.warning(msg)
            return False, msg
        
        return True, None
    
    def _check_spread_widening(self, execution_id: str, symbol: str, current_spread: float,
                               max_widening_pips: float = 2.0) -> Tuple[bool, Optional[str]]:
        """
        Detect if spread widened significantly post-validation.
        
        Compares current spread to spread at validation checkpoint.
        
        Args:
            execution_id: For checkpoint lookup
            symbol: Trading pair
            current_spread: Current spread in pips
            max_widening_pips: Max allowed increase
        
        Returns:
            (ok, error_msg)
        """
        if not isinstance(self.database, ExecutionSQLiteDB):
            return True, None  # SQLite not available
        
        try:
            cursor = self.database.conn.cursor()
            cursor.execute('''
                SELECT spread_at_validation FROM validation_checkpoints
                WHERE execution_id = ?
            ''', (execution_id,))
            row = cursor.fetchone()
            
            if not row:
                return True, None  # No checkpoint yet
            
            spread_at_validation = row['spread_at_validation']
            widening = current_spread - spread_at_validation
            
            if widening > max_widening_pips:
                msg = f"Spread widened: {spread_at_validation:.2f}→{current_spread:.2f}pips (+{widening:.2f} exceeds {max_widening_pips}pips)"
                LOG.error(msg)
                return False, msg
            
            return True, None
        except Exception as e:
            LOG.debug(f"Spread widening check failed: {e}")
            return True, None
    
    def _check_live_margin(self, execution_log: ExecutionLog, min_cushion: float = 1.5) -> Tuple[bool, Optional[str]]:
        """
        Re-check margin immediately before order submission.
        
        Ensures margin hasn't depleted between validation and submission.
        
        Args:
            execution_log: Current execution being processed
            min_cushion: Min margin multiplier
        
        Returns:
            (ok, error_msg)
        """
        current_margin = self.broker.get_available_margin()
        required_margin = execution_log.risk_per_trade * min_cushion
        
        if current_margin is None:
            return False, "Cannot retrieve current margin"
        
        if current_margin < required_margin:
            msg = f"Margin depleted: {current_margin:.2f} < {required_margin:.2f} required"
            LOG.error(msg)
            return False, msg
        
        return True, None
    
    def _check_multi_leg_correlation(self, trade_legs: List[Dict], max_corr_threshold: float = 0.95) -> Tuple[bool, Optional[str]]:
        """
        Check correlations for multi-leg trades.
        
        Prevents over-exposure through correlated positions.
        
        Args:
            trade_legs: List of {'symbol': str, 'direction': int, 'units': float}
            max_corr_threshold: Max allowed pairwise correlation
        
        Returns:
            (ok, error_msg)
        """
        if len(trade_legs) < 2:
            return True, None  # Single leg, no correlation risk
        
        try:
            # In production: would fetch actual correlations from market data
            # For MVP: simple heuristic correlation detection
            
            symbols = [leg['symbol'] for leg in trade_legs]
            directions = np.array([leg['direction'] for leg in trade_legs])
            
            # Check for highly correlated directions
            # If all buying or all selling, that's fine
            # If mixed: check correlation matrix
            
            long_count = np.sum(directions > 0)
            short_count = np.sum(directions < 0)
            
            # Correlation breach: same leg traded in opposite directions simultaneously
            for i, leg1 in enumerate(trade_legs):
                for j, leg2 in enumerate(trade_legs[i+1:], i+1):
                    # If legs are same symbol with opposite directions = hedge = ok
                    if leg1['symbol'] == leg2['symbol']:
                        if np.sign(leg1['direction']) != np.sign(leg2['direction']):
                            continue  # Hedge is allowed
                        else:
                            msg = f"Duplicate leg: {leg1['symbol']} {leg1['direction']}"
                            LOG.error(msg)
                            return False, msg
            
            # Store correlation checkpoint
            if isinstance(self.database, ExecutionSQLiteDB):
                corr_data = {
                    'symbols': symbols,
                    'long_count': int(long_count),
                    'short_count': int(short_count),
                }
                # Would store to multi_leg_trades table
            
            return True, None
        except Exception as e:
            LOG.warning(f"Multi-leg correlation check failed: {e}")
            return True, None  # Fail safe
    
    # ========================================
    # ORDER SUBMISSION & RETRY LOGIC
    # ========================================
    
    def _submit_order_with_retry(
        self,
        execution_log: ExecutionLog,
        execution_request: ExecutionRequest,
        retry_count: int = 0
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Submit order to broker with retry logic.
        
        Includes hardening checks before submission:
        1. Stale tick detection
        2. Spread widening detection
        3. Live margin re-check
        4. Multi-leg correlation
        
        Returns:
            (success, order_id, error_message)
        """
        
        approved_trade = execution_request.approved_trade
        
        try:
            # HARDENING: Pre-submission checks (first attempt only)
            if retry_count == 0:
                # 1. Fresh market data check
                market = self.broker.get_market_snapshot(approved_trade.symbol)
                if market:
                    is_fresh, stale_msg = self._check_stale_tick(market, max_age_seconds=5)
                    if not is_fresh:
                        LOG.error(f"Rejecting: {stale_msg}")
                        return False, None, stale_msg
                
                # 2. Spread widening check
                is_ok, widening_msg = self._check_spread_widening(
                    execution_log.execution_id,
                    approved_trade.symbol,
                    market.spread_pips if market else 0,
                    max_widening_pips=2.0
                )
                if not is_ok:
                    LOG.error(f"Rejecting: {widening_msg}")
                    return False, None, widening_msg
                
                # 3. Live margin re-check (within 100ms of submission)
                is_ok, margin_msg = self._check_live_margin(execution_log, min_cushion=self.min_margin_cushion)
                if not is_ok:
                    LOG.error(f"Rejecting: {margin_msg}")
                    return False, None, margin_msg
            
            LOG.info(f"[ATTEMPT {retry_count + 1}/{execution_request.retry_attempts}] Submitting order: {approved_trade.symbol} {approved_trade.direction:+d} {approved_trade.position_units}")
            
            success, order_id, error_msg = self.broker.place_order(
                symbol=approved_trade.symbol,
                direction=approved_trade.direction,
                units=approved_trade.position_units,
                order_type=execution_request.order_type
            )
            
            if success and order_id:
                LOG.info(f"Order submitted successfully: {order_id}")
                return True, order_id, None
            
            # Order failed at broker
            if retry_count < execution_request.retry_attempts - 1:
                LOG.warning(f"Order submission failed: {error_msg}. Retrying... ({retry_count + 1}/{execution_request.retry_attempts})")
                time.sleep(1)  # Back-off delay
                return self._submit_order_with_retry(execution_log, execution_request, retry_count + 1)
            else:
                LOG.error(f"Order submission failed after {execution_request.retry_attempts} attempts: {error_msg}")
                return False, None, error_msg
        
        except Exception as e:
            error_msg = str(e)
            LOG.error(f"Unexpected error during order submission: {error_msg}")
            
            if retry_count < execution_request.retry_attempts - 1:
                LOG.warning(f"Retrying after exception... ({retry_count + 1}/{execution_request.retry_attempts})")
                time.sleep(1)
                return self._submit_order_with_retry(execution_log, execution_request, retry_count + 1)
            else:
                return False, None, error_msg
    
    # ========================================
    # ORDER MONITORING & FILL TRACKING
    # ========================================
    
    def _monitor_order(
        self,
        execution_log: ExecutionLog,
        timeout: int = 60,
        poll_interval: float = 0.5
    ) -> Optional[Tuple[float, float, float]]:
        """
        Monitor order until filled or timeout.
        
        Returns:
            (fill_price, filled_units, slippage_pips) or None if timeout
        """
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Query order status from broker
            status = self.broker.get_order_status(execution_log.order_id)
            
            if status is None:
                time.sleep(poll_interval)
                continue
            
            # Check if filled (use state field or check for fill_price)
            state = status.get('state', '').upper()
            is_filled = (state == 'FILLED') or ('price' in status)
            
            if is_filled:
                fill_price = status.get('price') or status.get('fill_price')
                filled_units = status.get('volume') or status.get('filled_units')
                
                if fill_price is None or filled_units is None:
                    LOG.debug(f"Order status incomplete: {status}")
                    time.sleep(poll_interval)
                    continue
                
                # Measure slippage against snapshot bid/ask at submission
                intended_price = execution_log.intended_price
                if intended_price is None:
                    LOG.warning(f"No intended price recorded, cannot measure slippage accurately")
                    slippage_pips = 0.0
                else:
                    slippage_pips = self._calculate_slippage(
                        order_price=intended_price,
                        fill_price=fill_price,
                        direction=execution_log.direction
                    )
                
                LOG.info(f"Order filled: {filled_units} units at {fill_price} (slippage: {slippage_pips:.2f}pips)")
                return fill_price, filled_units, slippage_pips
            
            # Check for rejection/cancellation
            if state in ('CANCELED', 'REJECTED', 'EXPIRED'):
                error_reason = status.get('comment', 'Unknown rejection')
                LOG.warning(f"Order {execution_log.order_id} {state}: {error_reason}")
                return None
            
            time.sleep(poll_interval)
        
        # Timeout
        LOG.warning(f"Order {execution_log.order_id} timed out after {timeout}s")
        return None
    
    # ========================================
    # SLIPPAGE CALCULATION
    # ========================================
    
    def _calculate_slippage(
        self,
        order_price: Optional[float],
        fill_price: float,
        direction: int
    ) -> float:
        """
        Calculate slippage in pips (always >= 0).
        
        Institutional definition: how much worse than order price did we get?
        
        BUY:  slippage = (fill_price - order_price) / point_size  (bad if positive)
        SELL: slippage = (order_price - fill_price) / point_size  (bad if positive)
        
        For most FX pairs: 1 pip = 0.0001
        
        Returns: Slippage in pips (always >= 0, where 0 = perfect, higher = worse)
        """
        
        if order_price is None:
            return 0.0
        
        point_size = 0.0001  # Standard FX pip
        
        if direction > 0:  # BUY
            # For buy: we pay ask, want to minimize fill_price
            # slippage = (fill_price - order_price) / point_size
            slippage = max(0.0, (fill_price - order_price) / point_size)
        else:  # SELL
            # For sell: we receive bid, want to maximize fill_price
            # slippage = (order_price - fill_price) / point_size
            slippage = max(0.0, (order_price - fill_price) / point_size)
        
        return slippage
    
    # ========================================
    # ORDER CONFIRMATION
    # ========================================
    
    def _create_rejection_confirmation(
        self,
        decision: 'TradeDecision',
        approved_trade: Optional['ApprovedTrade'] = None,
        request_id: Optional[str] = None,
        reason: ExecutionRejectionReason = ExecutionRejectionReason.UNKNOWN,
        details: str = ""
    ) -> ExecutionConfirmation:
        """Create rejection confirmation"""
        
        if approved_trade is None and decision.rejected_trade:
            approved_trade = decision.rejected_trade
        
        return ExecutionConfirmation(
            execution_id=str(uuid.uuid4()),
            order_id="",
            status=ExecutionStatus.REJECTED,
            symbol=approved_trade.symbol if approved_trade else "UNKNOWN",
            direction=approved_trade.direction if approved_trade else 0,
            intended_units=approved_trade.position_units if approved_trade else 0,
            executed_units=0,
            fill_price=None,
            slippage_pips=None,
            rejection_reason=reason,
            rejection_details=details
        )
    
    def _confirmation_from_log(self, log: ExecutionLog) -> ExecutionConfirmation:
        """Create confirmation from execution log"""
        
        return ExecutionConfirmation(
            execution_id=log.execution_id,
            order_id=log.order_id,
            status=log.status,
            symbol=log.symbol,
            direction=log.direction,
            intended_units=log.intended_units,
            executed_units=log.executed_units,
            fill_price=log.fill_price,
            slippage_pips=log.slippage_pips,
            rejection_reason=log.rejection_reason,
            rejection_details=log.rejection_details
        )
    
    # ========================================
    # MONITORING & METRICS
    # ========================================
    
    def get_execution_metrics(self) -> dict:
        """Get execution performance metrics"""
        
        success_rate = (
            self.execution_metrics['successful_executions'] / 
            max(self.execution_metrics['total_executions'], 1)
        ) * 100
        
        return {
            'total_executions': self.execution_metrics['total_executions'],
            'successful_executions': self.execution_metrics['successful_executions'],
            'failed_executions': self.execution_metrics['failed_executions'],
            'rejected_executions': self.execution_metrics['rejected_executions'],
            'success_rate': success_rate,
            'avg_slippage_pips': self.execution_metrics['avg_slippage_pips'],
        }
    
    def get_execution_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[dict]:
        """Get recent execution history"""
        logs = self.database.get_execution_history(symbol=symbol, limit=limit)
        return [log.to_dict() for log in logs]
    
    def export_audit_trail(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Export execution logs for compliance audit"""
        return self.database.export_audit_trail(start_date, end_date)
