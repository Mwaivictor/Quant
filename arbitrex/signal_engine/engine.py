"""
Signal Generation Engine

Core engine that orchestrates:
1. Gate filtering (Regime → Quant Stats → ML)
2. Direction assignment (deterministic from momentum)
3. Confidence score computation
4. State management
5. Trade intent emission

Design Principles:
    - Conservative: When in doubt, do nothing
    - Deterministic: Same inputs → same outputs
    - Auditable: Full decision trail
    - Causal: No future leakage
"""

import time
from typing import Optional, Tuple
from datetime import datetime
import numpy as np

from arbitrex.signal_engine.config import SignalEngineConfig
from arbitrex.signal_engine.schemas import (
    TradeIntent,
    TradeDirection,
    SignalDecision,
    SignalEngineOutput,
    SignalEngineHealth
)
from arbitrex.signal_engine.filters import (
    RegimeGate,
    QuantStatsGate,
    MLConfidenceGate
)
from arbitrex.signal_engine.state_manager import SignalStateManager
from arbitrex.feature_engine.schemas import FeatureVector
from arbitrex.quant_stats.schemas import QuantStatsOutput
from arbitrex.ml_layer.schemas import MLOutput


class SignalGenerationEngine:
    """
    Signal Generation Engine
    
    Converts validated features, statistics, and ML confidence
    into actionable trade intents.
    """
    
    def __init__(self, config: Optional[SignalEngineConfig] = None):
        """
        Initialize Signal Generation Engine.
        
        Args:
            config: Engine configuration (uses defaults if None)
        """
        self.config = config or SignalEngineConfig()
        self.config_hash = self.config.compute_hash()
        
        # Initialize gates
        self.regime_gate = RegimeGate(self.config.regime_gate)
        self.quant_gate = QuantStatsGate(self.config.quant_gate)
        self.ml_gate = MLConfidenceGate(self.config.ml_gate)
        
        # Initialize state manager
        self.state_manager = SignalStateManager(self.config.state_management)
        
        # Health tracking
        self.health = SignalEngineHealth()
        
        # Internal tracking
        self._regime_gate_passes = 0
        self._quant_gate_passes = 0
        self._ml_gate_passes = 0
        self._processing_times = []
    
    def process_bar(
        self,
        feature_vector: FeatureVector,
        qse_output: QuantStatsOutput,
        ml_output: MLOutput,
        bar_index: int
    ) -> SignalEngineOutput:
        """
        Process single bar and generate signal decision.
        
        This is the main entry point called per bar.
        
        Args:
            feature_vector: Feature vector from Feature Engine
            qse_output: Quantitative statistics from QSE
            ml_output: ML predictions from ML Layer
            bar_index: Current bar index
            
        Returns:
            SignalEngineOutput with decision and state
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self._validate_inputs(feature_vector, qse_output, ml_output)
        
        # Get current state
        state = self.state_manager.get_state(
            feature_vector.symbol,
            feature_vector.timeframe
        )
        
        # Check exit conditions for active trade
        should_exit, exit_reason = self.state_manager.check_exit_conditions(
            feature_vector.symbol,
            feature_vector.timeframe,
            qse_output,
            ml_output
        )
        
        if should_exit:
            state = self.state_manager.transition_to_no_trade(
                feature_vector.symbol,
                feature_vector.timeframe,
                feature_vector.timestamp_utc,
                reason=exit_reason
            )
        
        # Generate signal decision
        decision = self._generate_decision(
            feature_vector,
            qse_output,
            ml_output,
            bar_index
        )
        
        # Update state if new signal generated
        if decision.trade_allowed and decision.trade_intent:
            state = self.state_manager.transition_to_valid_signal(
                feature_vector.symbol,
                feature_vector.timeframe,
                decision.trade_intent,
                feature_vector.timestamp_utc,
                bar_index
            )
        
        # Increment bar counter
        self.state_manager.increment_bars(
            feature_vector.symbol,
            feature_vector.timeframe
        )
        
        # Update health metrics
        self._update_health(decision, start_time)
        
        # Compute processing time
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Create output
        output = SignalEngineOutput(
            timestamp=feature_vector.timestamp_utc,
            symbol=feature_vector.symbol,
            timeframe=feature_vector.timeframe,
            bar_index=bar_index,
            decision=decision,
            state=state,
            config_hash=self.config_hash,
            engine_version="1.0.0",
            processing_time_ms=processing_time
        )
        
        return output
    
    def _generate_decision(
        self,
        feature_vector: FeatureVector,
        qse_output: QuantStatsOutput,
        ml_output: MLOutput,
        bar_index: int
    ) -> SignalDecision:
        """
        Generate signal decision by applying all gates.
        
        Gate order (strict):
            1. Regime Gate
            2. Quant Stats Gate
            3. ML Confidence Gate
            4. Direction Assignment
            5. Confidence Score Computation
        """
        decision = SignalDecision()
        suppression_reasons = []
        
        # Gate 1: Regime Check
        regime_passed, regime_reason, regime_weight = self.regime_gate.check(ml_output)
        decision.regime_gate_passed = regime_passed
        decision.regime_gate_reason = regime_reason
        decision.regime_weight = regime_weight
        
        if not regime_passed:
            suppression_reasons.append(f"Regime: {regime_reason}")
            decision.trade_allowed = False
            decision.suppression_reasons = suppression_reasons
            self.health.suppression_by_regime += 1
            return decision
        
        self._regime_gate_passes += 1
        
        # Gate 2: Quant Stats Check
        quant_passed, quant_reason = self.quant_gate.check(qse_output)
        decision.quant_gate_passed = quant_passed
        decision.quant_gate_reason = quant_reason
        decision.trend_consistency = qse_output.validation.trend_consistency
        
        if not quant_passed:
            suppression_reasons.append(f"QuantStats: {quant_reason}")
            decision.trade_allowed = False
            decision.suppression_reasons = suppression_reasons
            self.health.suppression_by_quant += 1
            return decision
        
        self._quant_gate_passes += 1
        
        # Gate 3: ML Confidence Check
        ml_passed, ml_reason = self.ml_gate.check(ml_output)
        decision.ml_gate_passed = ml_passed
        decision.ml_gate_reason = ml_reason
        decision.ml_confidence = ml_output.prediction.signal.momentum_success_prob
        
        if not ml_passed:
            suppression_reasons.append(f"ML: {ml_reason}")
            decision.trade_allowed = False
            decision.suppression_reasons = suppression_reasons
            self.health.suppression_by_ml += 1
            return decision
        
        self._ml_gate_passes += 1
        
        # All gates passed - determine direction
        direction = self._assign_direction(feature_vector, qse_output)
        decision.momentum_direction = direction.value
        
        # Check state constraints
        can_signal, state_reason = self.state_manager.can_generate_signal(
            feature_vector.symbol,
            feature_vector.timeframe,
            direction
        )
        
        if not can_signal:
            suppression_reasons.append(f"State: {state_reason}")
            decision.trade_allowed = False
            decision.suppression_reasons = suppression_reasons
            return decision
        
        # Compute confidence score
        confidence_score = self._compute_confidence_score(
            ml_confidence=decision.ml_confidence,
            trend_consistency=decision.trend_consistency,
            regime_weight=decision.regime_weight
        )
        decision.raw_confidence_score = confidence_score
        
        # Create trade intent
        trade_intent = TradeIntent(
            timestamp=feature_vector.timestamp_utc,
            symbol=feature_vector.symbol,
            timeframe=feature_vector.timeframe,
            direction=direction,
            confidence_score=confidence_score,
            signal_source=self.config.signal_source_name,
            signal_version=self.config_hash,
            bar_index=bar_index
        )
        
        # Signal allowed
        decision.trade_allowed = True
        decision.trade_intent = trade_intent
        
        return decision
    
    def _assign_direction(
        self,
        feature_vector: FeatureVector,
        qse_output: QuantStatsOutput
    ) -> TradeDirection:
        """
        Assign trade direction from deterministic momentum.
        
        Direction comes from feature vector momentum signal.
        NO PREDICTION - purely deterministic.
        
        Args:
            feature_vector: Feature vector (contains momentum features)
            qse_output: Quant stats (contains trend regime)
            
        Returns:
            TradeDirection (LONG or SHORT)
        """
        # Extract momentum direction from feature vector
        # Assuming feature vector contains a momentum or return signal
        # For demonstration, use trend_regime from QSE
        
        trend_regime = qse_output.regime.trend_regime
        
        # Map trend regime to direction
        if trend_regime in ["STRONG_UP", "UP"]:
            return TradeDirection.LONG
        elif trend_regime in ["STRONG_DOWN", "DOWN"]:
            return TradeDirection.SHORT
        else:
            # Default to checking returns sign
            # This should ideally come from a specific momentum feature
            if qse_output.metrics.rolling_mean > 0:
                return TradeDirection.LONG
            else:
                return TradeDirection.SHORT
    
    def _compute_confidence_score(
        self,
        ml_confidence: float,
        trend_consistency: float,
        regime_weight: float
    ) -> float:
        """
        Compute final confidence score for position sizing.
        
        Formula:
            confidence = (
                ml_confidence * ml_weight +
                trend_consistency * trend_weight +
                regime_weight * regime_contribution
            )
        
        Args:
            ml_confidence: P(momentum_success) from ML
            trend_consistency: Trend consistency score from QSE
            regime_weight: Regime quality weight
            
        Returns:
            Confidence score in [0, 1]
        """
        score = (
            ml_confidence * self.config.confidence_score.ml_confidence_weight +
            trend_consistency * self.config.confidence_score.trend_consistency_weight +
            regime_weight * self.config.confidence_score.regime_weight_contribution
        )
        
        # Clamp to valid range
        score = max(
            self.config.confidence_score.min_output_confidence,
            min(score, self.config.confidence_score.max_output_confidence)
        )
        
        return score
    
    def _validate_inputs(
        self,
        feature_vector: FeatureVector,
        qse_output: QuantStatsOutput,
        ml_output: MLOutput
    ):
        """Validate input consistency"""
        # Check timestamps match
        if not (feature_vector.timestamp_utc == qse_output.timestamp == ml_output.timestamp):
            raise ValueError(
                f"Timestamp mismatch: FE={feature_vector.timestamp_utc}, "
                f"QSE={qse_output.timestamp}, ML={ml_output.timestamp}"
            )
        
        # Check symbols match
        if not (feature_vector.symbol == qse_output.symbol == ml_output.symbol):
            raise ValueError(
                f"Symbol mismatch: FE={feature_vector.symbol}, "
                f"QSE={qse_output.symbol}, ML={ml_output.symbol}"
            )
        
        # Check timeframes match
        if not (feature_vector.timeframe == qse_output.timeframe == ml_output.timeframe):
            raise ValueError(
                f"Timeframe mismatch: FE={feature_vector.timeframe}, "
                f"QSE={qse_output.timeframe}, ML={ml_output.timeframe}"
            )
    
    def _update_health(self, decision: SignalDecision, start_time: float):
        """Update health metrics"""
        self.health.total_bars_processed += 1
        
        if decision.trade_allowed:
            self.health.signals_generated += 1
            self.health.last_signal_time = datetime.utcnow()
            
            if decision.trade_intent:
                if decision.trade_intent.direction == TradeDirection.LONG:
                    self.health.long_signals += 1
                else:
                    self.health.short_signals += 1
                
                # Update confidence stats
                conf = decision.trade_intent.confidence_score
                self.health.avg_confidence_score = (
                    (self.health.avg_confidence_score * (self.health.signals_generated - 1) + conf) /
                    self.health.signals_generated
                )
                self.health.min_confidence_score = min(self.health.min_confidence_score, conf)
                self.health.max_confidence_score = max(self.health.max_confidence_score, conf)
        else:
            self.health.signals_suppressed += 1
        
        # Update generation rate
        if self.health.total_bars_processed > 0:
            self.health.signal_generation_rate = (
                self.health.signals_generated / self.health.total_bars_processed
            )
        
        # Update gate pass rates
        if self.health.total_bars_processed > 0:
            self.health.regime_gate_pass_rate = self._regime_gate_passes / self.health.total_bars_processed
            self.health.quant_gate_pass_rate = self._quant_gate_passes / self.health.total_bars_processed
            self.health.ml_gate_pass_rate = self._ml_gate_passes / self.health.total_bars_processed
        
        # Update processing time
        processing_time = (time.perf_counter() - start_time) * 1000
        self._processing_times.append(processing_time)
        if len(self._processing_times) > 1000:
            self._processing_times.pop(0)
        self.health.avg_processing_time_ms = np.mean(self._processing_times)
        
        # Update active signals count
        self.health.active_signals = len(self.state_manager.get_all_active_signals())
    
    def get_health(self) -> SignalEngineHealth:
        """Get current health metrics"""
        return self.health
    
    def get_state_summary(self) -> dict:
        """Get state manager summary"""
        return self.state_manager.get_state_summary()
    
    def reset(self):
        """Reset engine state"""
        self.state_manager.reset_all()
        self.health = SignalEngineHealth()
        self._regime_gate_passes = 0
        self._quant_gate_passes = 0
        self._ml_gate_passes = 0
        self._processing_times = []
