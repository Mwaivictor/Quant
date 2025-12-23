"""
Signal Engine Configuration

Defines all filtering thresholds and decision parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import hashlib
import json


@dataclass
class RegimeGateConfig:
    """Regime filter configuration"""
    
    # Allowed regimes
    allowed_regimes: list = field(default_factory=lambda: ["TRENDING"])
    
    # Regime stability requirement
    require_stable_regime: bool = True
    min_regime_confidence: float = 0.6
    
    # Regime weights for confidence scoring
    regime_weights: Dict[str, float] = field(default_factory=lambda: {
        "TRENDING": 1.0,
        "RANGING": 0.0,
        "STRESSED": 0.0,
        "UNKNOWN": 0.0
    })


@dataclass
class QuantStatsGateConfig:
    """Quantitative statistics filter configuration"""
    
    # Signal validity (primary gate)
    require_signal_validity_flag: bool = True
    
    # Trend consistency threshold
    min_trend_consistency: float = 0.5  # 0-1 score
    
    # Volatility regime constraints
    allowed_volatility_regimes: list = field(default_factory=lambda: ["NORMAL", "LOW"])
    min_volatility_percentile: float = 20.0
    max_volatility_percentile: float = 80.0
    
    # Correlation constraints
    max_cross_correlation: float = 0.85  # Prevent crowded trades
    
    # Distribution stability
    require_distribution_stable: bool = True
    
    # Autocorrelation requirements
    require_autocorr_check: bool = True
    
    # Stationarity requirements
    require_stationarity_check: bool = True


@dataclass
class MLGateConfig:
    """ML confidence filter configuration"""
    
    # Entry threshold (primary gate)
    entry_threshold: float = 0.55  # P(momentum_success) must exceed this
    
    # Exit threshold (for position management)
    exit_threshold: float = 0.45
    
    # Confidence level requirements
    min_confidence_level: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH"
    
    # ML weight in confidence scoring
    ml_weight: float = 1.0


@dataclass
class ConfidenceScoreConfig:
    """Confidence score computation configuration"""
    
    # Component weights (must sum to 1.0)
    ml_confidence_weight: float = 0.5
    trend_consistency_weight: float = 0.3
    regime_weight_contribution: float = 0.2
    
    # Normalization
    min_output_confidence: float = 0.0
    max_output_confidence: float = 1.0
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.ml_confidence_weight + 
                self.trend_consistency_weight + 
                self.regime_weight_contribution)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")


@dataclass
class StateManagementConfig:
    """Signal state management configuration"""
    
    # Minimum bars between signals (prevent oscillation)
    min_bars_between_signals: int = 5
    
    # Allow signal reversal (LONG → SHORT or vice versa)
    allow_reversal: bool = True
    
    # Exit on opposite signal
    exit_on_opposite_signal: bool = True
    
    # Exit on regime change
    exit_on_regime_change: bool = True
    
    # Exit on failed quant stats
    exit_on_quant_failure: bool = True


@dataclass
class SignalEngineConfig:
    """
    Complete Signal Engine configuration.
    
    All thresholds and decision parameters.
    """
    
    # Sub-configurations
    regime_gate: RegimeGateConfig = field(default_factory=RegimeGateConfig)
    quant_gate: QuantStatsGateConfig = field(default_factory=QuantStatsGateConfig)
    ml_gate: MLGateConfig = field(default_factory=MLGateConfig)
    confidence_score: ConfidenceScoreConfig = field(default_factory=ConfidenceScoreConfig)
    state_management: StateManagementConfig = field(default_factory=StateManagementConfig)
    
    # Signal source identification
    signal_source_name: str = "momentum_v1"
    
    # Version tracking
    config_version: str = "1.0.0"
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash of configuration.
        
        Used for versioning and reproducibility.
        """
        config_dict = {
            'regime_gate': {
                'allowed_regimes': self.regime_gate.allowed_regimes,
                'require_stable_regime': self.regime_gate.require_stable_regime,
                'min_regime_confidence': self.regime_gate.min_regime_confidence,
                'regime_weights': self.regime_gate.regime_weights,
            },
            'quant_gate': {
                'require_signal_validity_flag': self.quant_gate.require_signal_validity_flag,
                'min_trend_consistency': self.quant_gate.min_trend_consistency,
                'allowed_volatility_regimes': self.quant_gate.allowed_volatility_regimes,
                'min_volatility_percentile': self.quant_gate.min_volatility_percentile,
                'max_volatility_percentile': self.quant_gate.max_volatility_percentile,
                'max_cross_correlation': self.quant_gate.max_cross_correlation,
                'require_distribution_stable': self.quant_gate.require_distribution_stable,
                'require_autocorr_check': self.quant_gate.require_autocorr_check,
                'require_stationarity_check': self.quant_gate.require_stationarity_check,
            },
            'ml_gate': {
                'entry_threshold': self.ml_gate.entry_threshold,
                'exit_threshold': self.ml_gate.exit_threshold,
                'min_confidence_level': self.ml_gate.min_confidence_level,
                'ml_weight': self.ml_gate.ml_weight,
            },
            'confidence_score': {
                'ml_confidence_weight': self.confidence_score.ml_confidence_weight,
                'trend_consistency_weight': self.confidence_score.trend_consistency_weight,
                'regime_weight_contribution': self.confidence_score.regime_weight_contribution,
            },
            'state_management': {
                'min_bars_between_signals': self.state_management.min_bars_between_signals,
                'allow_reversal': self.state_management.allow_reversal,
                'exit_on_opposite_signal': self.state_management.exit_on_opposite_signal,
                'exit_on_regime_change': self.state_management.exit_on_regime_change,
                'exit_on_quant_failure': self.state_management.exit_on_quant_failure,
            },
            'signal_source_name': self.signal_source_name,
            'config_version': self.config_version,
        }
        
        # Compute hash
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'regime_gate': {
                'allowed_regimes': self.regime_gate.allowed_regimes,
                'require_stable_regime': self.regime_gate.require_stable_regime,
                'min_regime_confidence': self.regime_gate.min_regime_confidence,
                'regime_weights': self.regime_gate.regime_weights,
            },
            'quant_gate': {
                'require_signal_validity_flag': self.quant_gate.require_signal_validity_flag,
                'min_trend_consistency': self.quant_gate.min_trend_consistency,
                'allowed_volatility_regimes': self.quant_gate.allowed_volatility_regimes,
                'min_volatility_percentile': self.quant_gate.min_volatility_percentile,
                'max_volatility_percentile': self.quant_gate.max_volatility_percentile,
                'max_cross_correlation': self.quant_gate.max_cross_correlation,
                'require_distribution_stable': self.quant_gate.require_distribution_stable,
                'require_autocorr_check': self.quant_gate.require_autocorr_check,
                'require_stationarity_check': self.quant_gate.require_stationarity_check,
            },
            'ml_gate': {
                'entry_threshold': self.ml_gate.entry_threshold,
                'exit_threshold': self.ml_gate.exit_threshold,
                'min_confidence_level': self.ml_gate.min_confidence_level,
                'ml_weight': self.ml_gate.ml_weight,
            },
            'confidence_score': {
                'ml_confidence_weight': self.confidence_score.ml_confidence_weight,
                'trend_consistency_weight': self.confidence_score.trend_consistency_weight,
                'regime_weight_contribution': self.confidence_score.regime_weight_contribution,
                'min_output_confidence': self.confidence_score.min_output_confidence,
                'max_output_confidence': self.confidence_score.max_output_confidence,
            },
            'state_management': {
                'min_bars_between_signals': self.state_management.min_bars_between_signals,
                'allow_reversal': self.state_management.allow_reversal,
                'exit_on_opposite_signal': self.state_management.exit_on_opposite_signal,
                'exit_on_regime_change': self.state_management.exit_on_regime_change,
                'exit_on_quant_failure': self.state_management.exit_on_quant_failure,
            },
            'signal_source_name': self.signal_source_name,
            'config_version': self.config_version,
            'config_hash': self.compute_hash(),
        }
