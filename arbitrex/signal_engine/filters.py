"""
Signal Engine Filtering Gates

Implements the three primary gates:
1. Regime Gate - Only allow TRENDING regimes
2. Quant Stats Gate - Validate statistical robustness
3. ML Confidence Gate - Filter by momentum success probability
"""

from typing import Tuple
from arbitrex.signal_engine.config import (
    RegimeGateConfig,
    QuantStatsGateConfig,
    MLGateConfig
)
from arbitrex.ml_layer.schemas import MLOutput, RegimeLabel
from arbitrex.quant_stats.schemas import QuantStatsOutput


class RegimeGate:
    """
    Gate 1: Regime Filter
    
    Suppresses all signals outside allowed regimes (typically TRENDING only).
    """
    
    def __init__(self, config: RegimeGateConfig):
        self.config = config
    
    def check(self, ml_output: MLOutput) -> Tuple[bool, str, float]:
        """
        Check if regime allows trade.
        
        Args:
            ml_output: ML Layer output with regime prediction
            
        Returns:
            (passed: bool, reason: str, regime_weight: float)
        """
        regime_prediction = ml_output.prediction.regime
        regime_label = regime_prediction.regime_label.value
        regime_confidence = regime_prediction.regime_confidence
        
        # Check if regime is allowed
        if regime_label not in self.config.allowed_regimes:
            return False, f"Regime {regime_label} not in allowed list", 0.0
        
        # Check regime stability if required
        if self.config.require_stable_regime:
            if not regime_prediction.regime_stable:
                return False, "Regime not stable (recent change detected)", 0.0
        
        # Check regime confidence
        if regime_confidence < self.config.min_regime_confidence:
            return (
                False, 
                f"Regime confidence {regime_confidence:.3f} < threshold {self.config.min_regime_confidence}",
                0.0
            )
        
        # Get regime weight for confidence scoring
        regime_weight = self.config.regime_weights.get(regime_label, 0.0)
        
        return True, f"Regime {regime_label} allowed (confidence: {regime_confidence:.3f})", regime_weight


class QuantStatsGate:
    """
    Gate 2: Quantitative Statistics Filter
    
    Validates statistical robustness of the signal.
    """
    
    def __init__(self, config: QuantStatsGateConfig):
        self.config = config
    
    def check(self, qse_output: QuantStatsOutput) -> Tuple[bool, str]:
        """
        Check if quantitative statistics validate the signal.
        
        Args:
            qse_output: Quant Stats Engine output
            
        Returns:
            (passed: bool, reason: str)
        """
        validation = qse_output.validation
        metrics = qse_output.metrics
        regime = qse_output.regime
        
        # Primary gate: signal validity flag
        if self.config.require_signal_validity_flag:
            if not validation.signal_validity_flag:
                reasons = ", ".join(validation.failure_reasons) if validation.failure_reasons else "unknown"
                return False, f"Signal validity flag = False ({reasons})"
        
        # Trend consistency check
        if validation.trend_consistency < self.config.min_trend_consistency:
            return (
                False,
                f"Trend consistency {validation.trend_consistency:.3f} < threshold {self.config.min_trend_consistency}"
            )
        
        # Volatility regime check
        if metrics.volatility_regime not in self.config.allowed_volatility_regimes:
            return False, f"Volatility regime {metrics.volatility_regime} not allowed"
        
        # Volatility percentile check
        if not (self.config.min_volatility_percentile <= 
                metrics.volatility_percentile <= 
                self.config.max_volatility_percentile):
            return (
                False,
                f"Volatility percentile {metrics.volatility_percentile:.1f} outside range "
                f"[{self.config.min_volatility_percentile}, {self.config.max_volatility_percentile}]"
            )
        
        # Cross-correlation check (prevent crowded trades)
        if metrics.max_cross_correlation > self.config.max_cross_correlation:
            return (
                False,
                f"Max cross-correlation {metrics.max_cross_correlation:.3f} > threshold {self.config.max_cross_correlation}"
            )
        
        # Distribution stability check
        if self.config.require_distribution_stable:
            if not validation.distribution_check_passed:
                return False, "Distribution stability check failed"
        
        # Autocorrelation check
        if self.config.require_autocorr_check:
            if not validation.autocorr_check_passed:
                return False, "Autocorrelation check failed"
        
        # Stationarity check
        if self.config.require_stationarity_check:
            if not validation.stationarity_check_passed:
                return False, "Stationarity check failed"
        
        # All checks passed
        return True, "All quantitative statistics checks passed"


class MLConfidenceGate:
    """
    Gate 3: ML Confidence Filter
    
    Filters signals by momentum continuation probability.
    ML never generates signals - only filters deterministic ones.
    """
    
    def __init__(self, config: MLGateConfig):
        self.config = config
    
    def check(self, ml_output: MLOutput) -> Tuple[bool, str]:
        """
        Check if ML confidence allows trade entry.
        
        Args:
            ml_output: ML Layer output with signal prediction
            
        Returns:
            (passed: bool, reason: str)
        """
        signal_prediction = ml_output.prediction.signal
        prob_success = signal_prediction.momentum_success_prob
        confidence_level = signal_prediction.confidence_level
        
        # Primary gate: momentum success probability
        if prob_success < self.config.entry_threshold:
            return (
                False,
                f"Momentum success probability {prob_success:.3f} < entry threshold {self.config.entry_threshold}"
            )
        
        # Confidence level check (optional stricter filter)
        confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_level = confidence_order.get(self.config.min_confidence_level, 0)
        current_level = confidence_order.get(confidence_level, 0)
        
        if current_level < min_level:
            return (
                False,
                f"Confidence level {confidence_level} < minimum {self.config.min_confidence_level}"
            )
        
        # Check ML allow_trade flag (combines regime + signal)
        if not ml_output.prediction.allow_trade:
            reasons = ", ".join(ml_output.prediction.decision_reasons)
            return False, f"ML layer disallowed trade: {reasons}"
        
        # All checks passed
        return True, f"ML confidence passed (prob={prob_success:.3f}, level={confidence_level})"
