"""
Risk & Portfolio Manager Schemas

Defines data structures for trade decisions, portfolio state, and risk metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class TradeApprovalStatus(str, Enum):
    """Trade approval status"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ADJUSTED = "ADJUSTED"


class RejectionReason(str, Enum):
    """Reasons for trade rejection"""
    MAX_DRAWDOWN_EXCEEDED = "MAX_DRAWDOWN_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    SYMBOL_EXPOSURE_LIMIT = "SYMBOL_EXPOSURE_LIMIT"
    CURRENCY_EXPOSURE_LIMIT = "CURRENCY_EXPOSURE_LIMIT"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"
    STRESSED_REGIME = "STRESSED_REGIME"
    LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    TRADING_HALTED = "TRADING_HALTED"
    ZERO_POSITION_SIZE = "ZERO_POSITION_SIZE"


@dataclass
class ApprovedTrade:
    """
    Approved trade with final position sizing.
    
    Execution-ready instruction for downstream Execution Engine.
    Includes comprehensive sizing breakdown for institutional-grade auditability.
    """
    
    # Core trade parameters
    symbol: str
    direction: int  # 1=LONG, -1=SHORT
    position_units: float  # Final sized position
    
    # Risk metadata
    confidence_score: float  # From signal engine
    regime: str  # Current market regime
    
    # Sizing breakdown
    base_units: float  # Before adjustments
    confidence_adjustment: float  # Multiplier applied
    regime_adjustment: float  # Multiplier applied
    
    # Risk metrics
    atr: float  # ATR used for sizing
    vol_percentile: float  # Volatility regime
    risk_per_trade: float  # Capital risked
    
    # Institutional enhancements (optional)
    kelly_fraction: Optional[float] = None  # Kelly Criterion %
    kelly_capped: Optional[bool] = None  # Was Kelly cap applied?
    expectancy: Optional[float] = None  # Expectancy (p·W - (1-p)·L)
    expectancy_multiplier: Optional[float] = None  # Expectancy adjustment
    liquidity_capped: Optional[bool] = None  # Was liquidity limit applied?
    market_impact_pct: Optional[float] = None  # Estimated market impact %
    portfolio_vol_multiplier: Optional[float] = None  # Portfolio vol adjustment
    
    # Optional adjustments (with defaults must come last)
    correlation_adjustment: float = 1.0  # Multiplier applied (default: no adjustment)
    
    # Flags
    risk_override: bool = False  # Manual override applied
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    rpm_version: str = "1.1.0"  # Updated to reflect institutional enhancements
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = {
            'symbol': self.symbol,
            'direction': self.direction,
            'position_units': float(self.position_units),
            'confidence_score': float(self.confidence_score),
            'regime': self.regime,
            'base_units': float(self.base_units),
            'confidence_adjustment': float(self.confidence_adjustment),
            'regime_adjustment': float(self.regime_adjustment),
            'correlation_adjustment': float(self.correlation_adjustment),
            'atr': float(self.atr),
            'vol_percentile': float(self.vol_percentile),
            'risk_per_trade': float(self.risk_per_trade),
            'risk_override': bool(self.risk_override),
            'timestamp': self.timestamp.isoformat(),
            'rpm_version': self.rpm_version,
        }
        
        # Add institutional fields if present
        if self.kelly_fraction is not None:
            result['kelly_fraction'] = float(self.kelly_fraction)
        if self.kelly_capped is not None:
            result['kelly_capped'] = bool(self.kelly_capped)
        if self.expectancy is not None:
            result['expectancy'] = float(self.expectancy)
        if self.expectancy_multiplier is not None:
            result['expectancy_multiplier'] = float(self.expectancy_multiplier)
        if self.liquidity_capped is not None:
            result['liquidity_capped'] = bool(self.liquidity_capped)
        if self.market_impact_pct is not None:
            result['market_impact_pct'] = float(self.market_impact_pct)
        if self.portfolio_vol_multiplier is not None:
            result['portfolio_vol_multiplier'] = float(self.portfolio_vol_multiplier)
        
        return result


@dataclass
class RejectedTrade:
    """
    Rejected trade with detailed reason.
    
    Trade was suppressed by RPM - no execution.
    """
    
    # Original trade intent
    symbol: str
    direction: int
    confidence_score: float
    
    # Rejection details
    rejection_reason: RejectionReason
    rejection_details: str
    
    # Context at rejection
    current_drawdown: Optional[float] = None
    symbol_exposure: Optional[float] = None
    currency_exposure: Optional[float] = None
    vol_percentile: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence_score': float(self.confidence_score),
            'rejection_reason': self.rejection_reason.value,
            'rejection_details': self.rejection_details,
            'current_drawdown': float(self.current_drawdown) if self.current_drawdown is not None else None,
            'symbol_exposure': float(self.symbol_exposure) if self.symbol_exposure is not None else None,
            'currency_exposure': float(self.currency_exposure) if self.currency_exposure is not None else None,
            'vol_percentile': float(self.vol_percentile) if self.vol_percentile is not None else None,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TradeDecision:
    """
    Complete RPM decision with approval/rejection.
    
    Includes full audit trail of decision logic.
    """
    
    # Decision outcome
    status: TradeApprovalStatus
    approved_trade: Optional[ApprovedTrade] = None
    rejected_trade: Optional[RejectedTrade] = None
    
    # Order tracking (NEW)
    order_id: Optional[str] = None
    
    # Decision checkpoints
    kill_switch_triggered: bool = False
    kill_switch_reason: Optional[str] = None
    
    portfolio_constraints_passed: bool = True
    portfolio_constraint_violations: List[str] = field(default_factory=list)
    
    position_sizing_applied: bool = False
    sizing_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'approved_trade': self.approved_trade.to_dict() if self.approved_trade else None,
            'rejected_trade': self.rejected_trade.to_dict() if self.rejected_trade else None,
            'order_id': self.order_id,
            'kill_switch_triggered': bool(self.kill_switch_triggered),
            'kill_switch_reason': self.kill_switch_reason,
            'portfolio_constraints_passed': bool(self.portfolio_constraints_passed),
            'portfolio_constraint_violations': list(self.portfolio_constraint_violations),
            'position_sizing_applied': bool(self.position_sizing_applied),
            'sizing_adjustments': {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in self.sizing_adjustments.items()},
            'processing_time_ms': float(self.processing_time_ms),
        }


@dataclass
class Position:
    """Current open position"""
    symbol: str
    direction: int
    units: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'units': float(self.units),
            'entry_price': float(self.entry_price),
            'entry_time': self.entry_time.isoformat(),
            'unrealized_pnl': float(self.unrealized_pnl),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Restore position from dictionary"""
        from dateutil.parser import parse
        return cls(
            symbol=data['symbol'],
            direction=data['direction'],
            units=data['units'],
            entry_price=data['entry_price'],
            entry_time=parse(data['entry_time']),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
        )


@dataclass
class PortfolioState:
    """
    Current portfolio state.
    
    Tracks positions, exposure, PnL, and risk metrics.
    """
    
    # Positions
    open_positions: Dict[str, Position] = field(default_factory=dict)
    
    # Exposure tracking
    symbol_exposure: Dict[str, float] = field(default_factory=dict)  # Units per symbol
    currency_exposure: Dict[str, float] = field(default_factory=dict)  # Net exposure per currency
    
    # PnL tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    
    # Capital tracking
    total_capital: float = 100000.0  # Default starting capital
    equity: float = 100000.0  # Capital + unrealized PnL
    peak_equity: float = 100000.0  # High water mark
    current_drawdown: float = 0.0  # (peak - current) / peak
    
    # Trading state
    trading_halted: bool = False
    halt_reason: Optional[str] = None
    halt_timestamp: Optional[datetime] = None
    
    # Timestamp
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'open_positions': {k: v.to_dict() for k, v in self.open_positions.items()},
            'symbol_exposure': {k: float(v) for k, v in self.symbol_exposure.items()},
            'currency_exposure': {k: float(v) for k, v in self.currency_exposure.items()},
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'daily_pnl': float(self.daily_pnl),
            'weekly_pnl': float(self.weekly_pnl),
            'total_capital': float(self.total_capital),
            'equity': float(self.equity),
            'peak_equity': float(self.peak_equity),
            'current_drawdown': float(self.current_drawdown),
            'trading_halted': bool(self.trading_halted),
            'halt_reason': self.halt_reason,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'last_update': self.last_update.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PortfolioState':
        """Restore portfolio state from dictionary"""
        from dateutil.parser import parse
        
        # Restore positions
        open_positions = {
            k: Position.from_dict(v) 
            for k, v in data.get('open_positions', {}).items()
        }
        
        return cls(
            open_positions=open_positions,
            symbol_exposure=data.get('symbol_exposure', {}),
            currency_exposure=data.get('currency_exposure', {}),
            realized_pnl=data.get('realized_pnl', 0.0),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            daily_pnl=data.get('daily_pnl', 0.0),
            weekly_pnl=data.get('weekly_pnl', 0.0),
            total_capital=data.get('total_capital', 100000.0),
            equity=data.get('equity', 100000.0),
            peak_equity=data.get('peak_equity', 100000.0),
            current_drawdown=data.get('current_drawdown', 0.0),
            trading_halted=data.get('trading_halted', False),
            halt_reason=data.get('halt_reason'),
            halt_timestamp=parse(data['halt_timestamp']) if data.get('halt_timestamp') else None,
            last_update=parse(data['last_update']) if data.get('last_update') else datetime.utcnow(),
        )


@dataclass
class RiskMetrics:
    """Risk metrics for monitoring"""
    
    # Decisions
    total_decisions: int = 0
    trades_approved: int = 0
    trades_rejected: int = 0
    trades_adjusted: int = 0
    approval_rate: float = 0.0
    
    # Rejections by reason
    rejections_by_drawdown: int = 0
    rejections_by_loss_limit: int = 0
    rejections_by_exposure: int = 0
    rejections_by_volatility: int = 0
    rejections_by_regime: int = 0
    rejections_by_confidence: int = 0
    
    # Kill switches
    kill_switch_activations: int = 0
    trading_halt_count: int = 0
    
    # Sizing
    avg_position_size: float = 0.0
    min_position_size: float = 0.0
    max_position_size: float = 0.0
    
    # Performance
    avg_processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_decisions': self.total_decisions,
            'trades_approved': self.trades_approved,
            'trades_rejected': self.trades_rejected,
            'trades_adjusted': self.trades_adjusted,
            'approval_rate': float(self.approval_rate),
            'rejections_by_drawdown': self.rejections_by_drawdown,
            'rejections_by_loss_limit': self.rejections_by_loss_limit,
            'rejections_by_exposure': self.rejections_by_exposure,
            'rejections_by_volatility': self.rejections_by_volatility,
            'rejections_by_regime': self.rejections_by_regime,
            'rejections_by_confidence': self.rejections_by_confidence,
            'kill_switch_activations': self.kill_switch_activations,
            'trading_halt_count': self.trading_halt_count,
            'avg_position_size': float(self.avg_position_size),
            'min_position_size': float(self.min_position_size),
            'max_position_size': float(self.max_position_size),
            'avg_processing_time_ms': float(self.avg_processing_time_ms),
        }


@dataclass
class RPMOutput:
    """
    Complete RPM output.
    
    Combines decision, portfolio state, and risk metrics.
    """
    
    # Core decision
    decision: TradeDecision
    
    # Portfolio snapshot
    portfolio_state: PortfolioState
    
    # Risk metrics
    risk_metrics: RiskMetrics
    
    # Configuration
    config_hash: str
    rpm_version: str = "1.0.0"
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'decision': self.decision.to_dict(),
            'portfolio_state': self.portfolio_state.to_dict(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'config_hash': self.config_hash,
            'rpm_version': self.rpm_version,
            'timestamp': self.timestamp.isoformat(),
        }
