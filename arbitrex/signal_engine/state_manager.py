"""
Signal State Manager

Manages signal state machine and ensures:
- Single active signal per symbol
- No duplicate entries
- No conflicting directions
- Proper state transitions
"""

from typing import Dict, Optional, Tuple
from datetime import datetime

from arbitrex.signal_engine.config import StateManagementConfig
from arbitrex.signal_engine.schemas import (
    SignalState,
    SignalStateRecord,
    TradeIntent,
    TradeDirection
)
from arbitrex.ml_layer.schemas import MLOutput, RegimeLabel
from arbitrex.quant_stats.schemas import QuantStatsOutput


class SignalStateManager:
    """
    Manages signal state machine per symbol/timeframe.
    
    State transitions:
        NO_TRADE → VALID_SIGNAL → ACTIVE_TRADE → NO_TRADE
                        ↓             ↓
                    NO_TRADE      NO_TRADE
    """
    
    def __init__(self, config: StateManagementConfig):
        self.config = config
        
        # State tracking per symbol/timeframe
        self._states: Dict[str, SignalStateRecord] = {}
    
    def _get_key(self, symbol: str, timeframe: str) -> str:
        """Generate state key"""
        return f"{symbol}_{timeframe}"
    
    def get_state(self, symbol: str, timeframe: str) -> SignalStateRecord:
        """
        Get current state for symbol/timeframe.
        
        Creates new state if doesn't exist.
        """
        key = self._get_key(symbol, timeframe)
        
        if key not in self._states:
            self._states[key] = SignalStateRecord(
                state=SignalState.NO_TRADE,
                min_bars_between_signals=self.config.min_bars_between_signals
            )
        
        return self._states[key]
    
    def can_generate_signal(
        self,
        symbol: str,
        timeframe: str,
        new_direction: TradeDirection
    ) -> Tuple[bool, str]:
        """
        Check if new signal can be generated.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            new_direction: Proposed trade direction
            
        Returns:
            (allowed: bool, reason: str)
        """
        state = self.get_state(symbol, timeframe)
        
        # Check if in cooldown period
        if not state.can_generate_new_signal():
            return (
                False,
                f"Cooldown active: {state.bars_since_state_change}/{state.min_bars_between_signals} bars"
            )
        
        # Check if active trade exists
        if state.state == SignalState.ACTIVE_TRADE:
            if state.active_intent is None:
                # Inconsistent state - allow reset
                return True, "Resetting inconsistent state"
            
            # Check if reversal allowed
            if state.active_intent.direction != new_direction:
                if not self.config.allow_reversal:
                    return False, f"Active {state.active_intent.direction.name} trade, reversal not allowed"
                else:
                    return True, f"Reversal from {state.active_intent.direction.name} to {new_direction.name} allowed"
            else:
                # Same direction - don't duplicate
                return False, f"Active {state.active_intent.direction.name} trade already exists"
        
        # No active trade - allowed
        return True, "No active trade, signal allowed"
    
    def transition_to_valid_signal(
        self,
        symbol: str,
        timeframe: str,
        trade_intent: TradeIntent,
        timestamp: datetime,
        bar_index: int
    ) -> SignalStateRecord:
        """
        Transition to VALID_SIGNAL state.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            trade_intent: Generated trade intent
            timestamp: Current timestamp
            bar_index: Current bar index
            
        Returns:
            Updated state record
        """
        state = self.get_state(symbol, timeframe)
        
        # Update state
        previous_state = state.state
        state.previous_state = previous_state
        state.state = SignalState.VALID_SIGNAL
        state.active_intent = trade_intent
        state.state_entry_time = timestamp
        state.state_entry_bar = bar_index
        state.state_change_time = timestamp
        state.bars_since_state_change = 0
        
        return state
    
    def transition_to_active_trade(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime
    ) -> SignalStateRecord:
        """
        Transition from VALID_SIGNAL to ACTIVE_TRADE.
        
        Typically triggered by external confirmation (e.g., order filled).
        """
        state = self.get_state(symbol, timeframe)
        
        if state.state != SignalState.VALID_SIGNAL:
            raise ValueError(f"Cannot transition to ACTIVE_TRADE from state {state.state}")
        
        # Update state
        state.previous_state = state.state
        state.state = SignalState.ACTIVE_TRADE
        state.state_change_time = timestamp
        state.last_trade_direction = state.active_intent.direction if state.active_intent else None
        state.last_trade_time = timestamp
        
        return state
    
    def transition_to_no_trade(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        reason: str = "Exit condition met"
    ) -> SignalStateRecord:
        """
        Transition to NO_TRADE state (exit).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            timestamp: Current timestamp
            reason: Reason for exit
            
        Returns:
            Updated state record
        """
        state = self.get_state(symbol, timeframe)
        
        # Update state
        state.previous_state = state.state
        state.state = SignalState.NO_TRADE
        state.active_intent = None
        state.state_change_time = timestamp
        state.bars_since_state_change = 0
        
        return state
    
    def check_exit_conditions(
        self,
        symbol: str,
        timeframe: str,
        qse_output: QuantStatsOutput,
        ml_output: MLOutput
    ) -> Tuple[bool, str]:
        """
        Check if current active trade should be exited.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            qse_output: Current quant stats
            ml_output: Current ML output
            
        Returns:
            (should_exit: bool, reason: str)
        """
        state = self.get_state(symbol, timeframe)
        
        # Only check if trade is active
        if state.state != SignalState.ACTIVE_TRADE:
            return False, "No active trade"
        
        if state.active_intent is None:
            return True, "Inconsistent state: no active intent"
        
        # Check regime change
        if self.config.exit_on_regime_change:
            current_regime = ml_output.prediction.regime.regime_label
            if current_regime != RegimeLabel.TRENDING:
                return True, f"Regime changed to {current_regime.value}"
        
        # Check quant stats failure
        if self.config.exit_on_quant_failure:
            if not qse_output.validation.signal_validity_flag:
                reasons = ", ".join(qse_output.validation.failure_reasons)
                return True, f"Quant stats failed: {reasons}"
        
        # Check ML exit signal
        signal_pred = ml_output.prediction.signal
        if signal_pred.should_exit:
            return True, f"ML exit signal (prob={signal_pred.momentum_success_prob:.3f})"
        
        # Check opposite direction signal
        if self.config.exit_on_opposite_signal:
            # This would be set by the main engine when detecting momentum reversal
            pass
        
        return False, "No exit conditions met"
    
    def increment_bars(self, symbol: str, timeframe: str):
        """Increment bar counter for state tracking"""
        state = self.get_state(symbol, timeframe)
        state.bars_since_state_change += 1
    
    def reset(self, symbol: str, timeframe: str):
        """Reset state for symbol/timeframe"""
        key = self._get_key(symbol, timeframe)
        if key in self._states:
            del self._states[key]
    
    def reset_all(self):
        """Reset all states"""
        self._states.clear()
    
    def get_all_active_signals(self) -> Dict[str, SignalStateRecord]:
        """Get all active signal states"""
        return {
            key: state 
            for key, state in self._states.items() 
            if state.state == SignalState.ACTIVE_TRADE
        }
    
    def get_state_summary(self) -> dict:
        """Get summary of all states"""
        summary = {
            'total_symbols': len(self._states),
            'active_trades': 0,
            'valid_signals': 0,
            'no_trade': 0,
            'states': {}
        }
        
        for key, state in self._states.items():
            if state.state == SignalState.ACTIVE_TRADE:
                summary['active_trades'] += 1
            elif state.state == SignalState.VALID_SIGNAL:
                summary['valid_signals'] += 1
            elif state.state == SignalState.NO_TRADE:
                summary['no_trade'] += 1
            
            summary['states'][key] = state.to_dict()
        
        return summary
