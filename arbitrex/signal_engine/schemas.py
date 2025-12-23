"""
Signal Engine Output Schemas

Defines data structures for trade intents, signal decisions, and state management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class TradeDirection(int, Enum):
    """Trade direction (deterministic from momentum)"""
    LONG = 1
    SHORT = -1
    FLAT = 0


class SignalState(str, Enum):
    """Signal state machine states"""
    NO_TRADE = "NO_TRADE"           # No valid signal
    VALID_SIGNAL = "VALID_SIGNAL"   # Signal generated, not yet active
    ACTIVE_TRADE = "ACTIVE_TRADE"   # Signal confirmed and active
    EXITED = "EXITED"               # Signal closed/exited


@dataclass
class TradeIntent:
    """
    Pure trade intent object.
    
    NO execution parameters - only direction and confidence.
    Consumed by Risk Manager for position sizing.
    """
    
    # Identification
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Core trade intent
    direction: TradeDirection  # LONG (1) or SHORT (-1)
    confidence_score: float    # 0-1, used for position sizing downstream
    
    # Signal source
    signal_source: str         # e.g., "momentum_v1"
    signal_version: str        # Config hash for reproducibility
    
    # Metadata
    bar_index: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction.value,
            'confidence_score': float(self.confidence_score),
            'signal_source': self.signal_source,
            'signal_version': self.signal_version,
            'bar_index': self.bar_index,
        }


@dataclass
class SignalDecision:
    """
    Complete signal decision with all gate results.
    
    Provides full audit trail of decision logic.
    """
    
    # Final decision
    trade_allowed: bool = False
    trade_intent: Optional[TradeIntent] = None
    
    # Gate results
    regime_gate_passed: bool = False
    quant_gate_passed: bool = False
    ml_gate_passed: bool = False
    
    # Gate details
    regime_gate_reason: str = ""
    quant_gate_reason: str = ""
    ml_gate_reason: str = ""
    
    # Suppression reasons (if trade not allowed)
    suppression_reasons: List[str] = field(default_factory=list)
    
    # Deterministic inputs
    momentum_direction: int = 0  # Raw momentum sign
    
    # Intermediate scores
    trend_consistency: float = 0.0
    regime_weight: float = 0.0
    ml_confidence: float = 0.0
    raw_confidence_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'trade_allowed': bool(self.trade_allowed),
            'trade_intent': self.trade_intent.to_dict() if self.trade_intent else None,
            'regime_gate_passed': bool(self.regime_gate_passed),
            'quant_gate_passed': bool(self.quant_gate_passed),
            'ml_gate_passed': bool(self.ml_gate_passed),
            'regime_gate_reason': self.regime_gate_reason,
            'quant_gate_reason': self.quant_gate_reason,
            'ml_gate_reason': self.ml_gate_reason,
            'suppression_reasons': list(self.suppression_reasons),
            'momentum_direction': int(self.momentum_direction),
            'trend_consistency': float(self.trend_consistency),
            'regime_weight': float(self.regime_weight),
            'ml_confidence': float(self.ml_confidence),
            'raw_confidence_score': float(self.raw_confidence_score),
        }


@dataclass
class SignalStateRecord:
    """
    Current signal state for a symbol/timeframe.
    
    Ensures single active signal per symbol.
    """
    
    # Current state
    state: SignalState
    
    # Active intent (if any)
    active_intent: Optional[TradeIntent] = None
    
    # State transition tracking
    state_entry_time: Optional[datetime] = None
    state_entry_bar: Optional[int] = None
    previous_state: Optional[SignalState] = None
    state_change_time: Optional[datetime] = None
    
    # Trade tracking
    last_trade_direction: Optional[TradeDirection] = None
    last_trade_time: Optional[datetime] = None
    
    # Prevent oscillation
    bars_since_state_change: int = 0
    min_bars_between_signals: int = 5
    
    def can_generate_new_signal(self) -> bool:
        """Check if enough time has passed for new signal"""
        # Allow first signal (state == NO_TRADE and no previous state)
        if self.state == SignalState.NO_TRADE and self.previous_state is None:
            return True
        return self.bars_since_state_change >= self.min_bars_between_signals
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'state': self.state.value,
            'active_intent': self.active_intent.to_dict() if self.active_intent else None,
            'state_entry_time': self.state_entry_time.isoformat() if self.state_entry_time else None,
            'state_entry_bar': self.state_entry_bar,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'state_change_time': self.state_change_time.isoformat() if self.state_change_time else None,
            'last_trade_direction': self.last_trade_direction.value if self.last_trade_direction else None,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'bars_since_state_change': self.bars_since_state_change,
            'min_bars_between_signals': self.min_bars_between_signals,
        }


@dataclass
class SignalEngineOutput:
    """
    Complete Signal Engine output for a single bar.
    
    Combines decision, state, and metadata for auditability.
    """
    
    # Identification
    timestamp: datetime
    symbol: str
    timeframe: str
    bar_index: int
    
    # Core outputs
    decision: SignalDecision
    state: SignalStateRecord
    
    # Configuration versioning
    config_hash: str
    engine_version: str
    
    # Processing metadata
    processing_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'bar_index': self.bar_index,
            'decision': self.decision.to_dict(),
            'state': self.state.to_dict(),
            'config_hash': self.config_hash,
            'engine_version': self.engine_version,
            'processing_time_ms': float(self.processing_time_ms),
        }


@dataclass
class SignalEngineHealth:
    """Health metrics for Signal Engine monitoring"""
    
    # Signal generation stats
    total_bars_processed: int = 0
    signals_generated: int = 0
    signals_suppressed: int = 0
    signal_generation_rate: float = 0.0
    
    # Gate statistics
    regime_gate_pass_rate: float = 0.0
    quant_gate_pass_rate: float = 0.0
    ml_gate_pass_rate: float = 0.0
    
    # Suppression reasons distribution
    suppression_by_regime: int = 0
    suppression_by_quant: int = 0
    suppression_by_ml: int = 0
    
    # Direction distribution
    long_signals: int = 0
    short_signals: int = 0
    
    # Confidence distribution
    avg_confidence_score: float = 0.0
    min_confidence_score: float = 1.0
    max_confidence_score: float = 0.0
    
    # State tracking
    active_signals: int = 0
    
    # Performance
    avg_processing_time_ms: float = 0.0
    last_signal_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_bars_processed': self.total_bars_processed,
            'signals_generated': self.signals_generated,
            'signals_suppressed': self.signals_suppressed,
            'signal_generation_rate': float(self.signal_generation_rate),
            'regime_gate_pass_rate': float(self.regime_gate_pass_rate),
            'quant_gate_pass_rate': float(self.quant_gate_pass_rate),
            'ml_gate_pass_rate': float(self.ml_gate_pass_rate),
            'suppression_by_regime': self.suppression_by_regime,
            'suppression_by_quant': self.suppression_by_quant,
            'suppression_by_ml': self.suppression_by_ml,
            'long_signals': self.long_signals,
            'short_signals': self.short_signals,
            'avg_confidence_score': float(self.avg_confidence_score),
            'min_confidence_score': float(self.min_confidence_score),
            'max_confidence_score': float(self.max_confidence_score),
            'active_signals': self.active_signals,
            'avg_processing_time_ms': float(self.avg_processing_time_ms),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
        }
