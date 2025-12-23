"""
Integration Bridge Between Old and New Signal Systems

Enables seamless coexistence of:
- Old: SignalGenerationEngine → TradeIntent (single-leg gate-based)
- New: StrategyRunner → Signal (multi-leg parallel strategies)

Both systems publish to unified SignalBuffer for downstream processing.
"""

import logging
from typing import Optional
from datetime import datetime

from arbitrex.signal_engine.schemas import TradeIntent, TradeDirection
from arbitrex.signal_engine.signal_schemas import (
    Signal,
    SignalLeg,
    LegDirection,
    SignalType,
    SignalStatus,
    create_single_leg_signal
)
from arbitrex.signal_engine.signal_buffer import SignalBuffer, get_signal_buffer

LOG = logging.getLogger(__name__)


def convert_trade_intent_to_signal(
    trade_intent: TradeIntent,
    timeframe: str = "1H"
) -> Signal:
    """
    Convert old TradeIntent to new Signal schema.
    
    Enables SignalGenerationEngine output to flow through new SignalBuffer.
    
    Args:
        trade_intent: TradeIntent from SignalGenerationEngine
        timeframe: Timeframe (not stored in TradeIntent)
        
    Returns:
        Signal object compatible with new system
    """
    # Convert direction enum
    if trade_intent.direction == TradeDirection.LONG:
        leg_direction = LegDirection.LONG
    elif trade_intent.direction == TradeDirection.SHORT:
        leg_direction = LegDirection.SHORT
    else:
        leg_direction = LegDirection.FLAT
    
    # Create signal using helper function
    signal = create_single_leg_signal(
        symbol=trade_intent.symbol,
        direction=leg_direction,
        confidence=trade_intent.confidence_score,
        strategy_id=trade_intent.signal_source,
        timeframe=timeframe,
        bar_index=trade_intent.bar_index,
        strategy_version=trade_intent.signal_version,
        metadata={
            'source': 'SignalGenerationEngine',
            'original_type': 'TradeIntent',
            'converted': True
        }
    )
    
    return signal


def convert_signal_to_trade_intent(
    signal: Signal,
    raise_on_multileg: bool = True
) -> Optional[TradeIntent]:
    """
    Convert new Signal to old TradeIntent (for backward compatibility).
    
    Note: Only single-leg signals can be converted.
    
    Args:
        signal: Signal from new system
        raise_on_multileg: Raise error if multi-leg (vs return None)
        
    Returns:
        TradeIntent or None if multi-leg
    """
    # Check if single-leg
    if not signal.is_single_leg:
        if raise_on_multileg:
            raise ValueError(f"Cannot convert multi-leg signal {signal.signal_id} to TradeIntent")
        LOG.warning(f"Skipping multi-leg signal {signal.signal_id} (not compatible with TradeIntent)")
        return None
    
    leg = signal.legs[0]
    
    # Convert direction
    if leg.direction == LegDirection.LONG:
        trade_direction = TradeDirection.LONG
    elif leg.direction == LegDirection.SHORT:
        trade_direction = TradeDirection.SHORT
    else:
        trade_direction = TradeDirection.FLAT
    
    # Create TradeIntent
    trade_intent = TradeIntent(
        timestamp=signal.timestamp,
        symbol=leg.symbol,
        timeframe=signal.timeframe,
        direction=trade_direction,
        confidence_score=signal.confidence_score,
        signal_source=signal.strategy_id,
        signal_version=signal.strategy_version,
        bar_index=signal.bar_index
    )
    
    return trade_intent


class UnifiedSignalPublisher:
    """
    Unified publisher that accepts both TradeIntent and Signal objects.
    
    Automatically converts TradeIntent → Signal and publishes to SignalBuffer.
    """
    
    def __init__(self, signal_buffer: Optional[SignalBuffer] = None):
        """
        Initialize unified publisher.
        
        Args:
            signal_buffer: SignalBuffer instance (uses global if None)
        """
        self.signal_buffer = signal_buffer or get_signal_buffer()
        self._trade_intents_published = 0
        self._signals_published = 0
    
    def publish_trade_intent(
        self,
        trade_intent: TradeIntent,
        timeframe: str = "1H"
    ) -> bool:
        """
        Publish TradeIntent (auto-converts to Signal).
        
        Args:
            trade_intent: TradeIntent from SignalGenerationEngine
            timeframe: Timeframe (not in TradeIntent)
            
        Returns:
            True if published successfully
        """
        try:
            signal = convert_trade_intent_to_signal(trade_intent, timeframe)
            success = self.signal_buffer.publish(signal)
            
            if success:
                self._trade_intents_published += 1
                LOG.debug(f"Published TradeIntent as Signal: {signal.signal_id}")
            
            return success
            
        except Exception as e:
            LOG.error(f"Failed to publish TradeIntent: {e}")
            return False
    
    def publish_signal(self, signal: Signal) -> bool:
        """
        Publish Signal directly.
        
        Args:
            signal: Signal from new strategy system
            
        Returns:
            True if published successfully
        """
        success = self.signal_buffer.publish(signal)
        
        if success:
            self._signals_published += 1
            LOG.debug(f"Published Signal: {signal.signal_id}")
        
        return success
    
    def publish(self, signal_or_intent) -> bool:
        """
        Publish either Signal or TradeIntent (auto-detect type).
        
        Args:
            signal_or_intent: Signal or TradeIntent object
            
        Returns:
            True if published successfully
        """
        if isinstance(signal_or_intent, TradeIntent):
            return self.publish_trade_intent(signal_or_intent)
        elif isinstance(signal_or_intent, Signal):
            return self.publish_signal(signal_or_intent)
        else:
            LOG.error(f"Unknown type: {type(signal_or_intent)}")
            return False
    
    def get_stats(self) -> dict:
        """Get publishing statistics"""
        return {
            'trade_intents_published': self._trade_intents_published,
            'signals_published': self._signals_published,
            'total_published': self._trade_intents_published + self._signals_published,
            'buffer_metrics': self.signal_buffer.get_metrics().to_dict()
        }


class HybridSignalRouter:
    """
    Routes signals from both old and new systems to appropriate consumers.
    
    Architecture:
        SignalGenerationEngine (old) ─┐
                                       ├→ HybridSignalRouter → SignalBuffer → Risk Manager
        StrategyRunner (new) ─────────┘
    """
    
    def __init__(self, signal_buffer: Optional[SignalBuffer] = None):
        """
        Initialize hybrid router.
        
        Args:
            signal_buffer: SignalBuffer instance
        """
        self.signal_buffer = signal_buffer or get_signal_buffer()
        self.publisher = UnifiedSignalPublisher(signal_buffer)
        
        # Statistics
        self._signals_from_old_system = 0
        self._signals_from_new_system = 0
        self._conversion_failures = 0
    
    def route_from_signal_engine(
        self,
        trade_intent: TradeIntent,
        timeframe: str = "1H"
    ) -> bool:
        """
        Route signal from SignalGenerationEngine (old system).
        
        Args:
            trade_intent: TradeIntent from SignalGenerationEngine
            timeframe: Timeframe
            
        Returns:
            True if routed successfully
        """
        success = self.publisher.publish_trade_intent(trade_intent, timeframe)
        
        if success:
            self._signals_from_old_system += 1
        else:
            self._conversion_failures += 1
        
        return success
    
    def route_from_strategy_runner(self, signal: Signal) -> bool:
        """
        Route signal from StrategyRunner (new system).
        
        Args:
            signal: Signal from strategy actor
            
        Returns:
            True if routed successfully
        """
        success = self.publisher.publish_signal(signal)
        
        if success:
            self._signals_from_new_system += 1
        
        return success
    
    def route(self, signal_or_intent, **kwargs) -> bool:
        """
        Auto-detect and route signal from either system.
        
        Args:
            signal_or_intent: TradeIntent or Signal
            **kwargs: Additional args (e.g., timeframe for TradeIntent)
            
        Returns:
            True if routed successfully
        """
        if isinstance(signal_or_intent, TradeIntent):
            timeframe = kwargs.get('timeframe', '1H')
            return self.route_from_signal_engine(signal_or_intent, timeframe)
        elif isinstance(signal_or_intent, Signal):
            return self.route_from_strategy_runner(signal_or_intent)
        else:
            LOG.error(f"Unknown signal type: {type(signal_or_intent)}")
            return False
    
    def get_routing_stats(self) -> dict:
        """Get routing statistics"""
        return {
            'signals_from_old_system': self._signals_from_old_system,
            'signals_from_new_system': self._signals_from_new_system,
            'conversion_failures': self._conversion_failures,
            'total_routed': self._signals_from_old_system + self._signals_from_new_system,
            'publisher_stats': self.publisher.get_stats()
        }


# Global router instance
_global_router: Optional[HybridSignalRouter] = None


def get_hybrid_router() -> HybridSignalRouter:
    """Get global hybrid router instance"""
    global _global_router
    if _global_router is None:
        _global_router = HybridSignalRouter()
    return _global_router
