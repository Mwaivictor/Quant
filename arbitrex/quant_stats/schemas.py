"""
Output Schemas for Quantitative Statistics Engine

Defines structured outputs for statistical metrics, validation results,
and regime states.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
class StatisticalMetrics:
    """
    Statistical metrics computed per bar.
    
    All metrics are causal (use data ≤ t only).
    """
    
    # Autocorrelation metrics
    autocorr_lag1: float = np.nan
    autocorr_lag5: float = np.nan
    autocorr_lag10: float = np.nan
    autocorr_lag20: float = np.nan
    trend_persistence_score: float = 0.0  # Average significant autocorr
    
    # Stationarity
    adf_stationary: bool = False
    adf_pvalue: float = 1.0
    adf_test_statistic: float = 0.0
    
    # Distribution stability
    z_score: float = 0.0
    is_outlier: bool = False
    distribution_stable: bool = True
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    
    # Cross-pair correlation (when applicable)
    avg_cross_correlation: float = 0.0
    max_cross_correlation: float = 0.0
    correlation_dispersion: float = 0.0
    
    # Volatility regime
    volatility_percentile: float = 50.0
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    current_volatility: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'autocorr_lag1': float(self.autocorr_lag1),
            'autocorr_lag5': float(self.autocorr_lag5),
            'autocorr_lag10': float(self.autocorr_lag10),
            'autocorr_lag20': float(self.autocorr_lag20),
            'trend_persistence_score': float(self.trend_persistence_score),
            'adf_stationary': self.adf_stationary,
            'adf_pvalue': float(self.adf_pvalue),
            'adf_test_statistic': float(self.adf_test_statistic),
            'z_score': float(self.z_score),
            'is_outlier': self.is_outlier,
            'distribution_stable': self.distribution_stable,
            'rolling_mean': float(self.rolling_mean),
            'rolling_std': float(self.rolling_std),
            'avg_cross_correlation': float(self.avg_cross_correlation),
            'max_cross_correlation': float(self.max_cross_correlation),
            'correlation_dispersion': float(self.correlation_dispersion),
            'volatility_percentile': float(self.volatility_percentile),
            'volatility_regime': self.volatility_regime,
            'current_volatility': float(self.current_volatility),
        }


@dataclass
class SignalValidation:
    """
    Signal validation result per bar.
    
    Determines if signal passes all statistical gates.
    """
    
    # Overall validation
    signal_validity_flag: bool = False
    
    # Individual checks
    autocorr_check_passed: bool = False
    stationarity_check_passed: bool = False
    distribution_check_passed: bool = False
    correlation_check_passed: bool = False
    volatility_check_passed: bool = False
    
    # Composite scores
    trend_consistency: float = 0.0  # 0-1 score
    regime_quality_score: float = 0.0  # 0-1 score
    
    # Failure reasons (if any)
    failure_reasons: List[str] = field(default_factory=list)
    
    # Metadata
    bars_used: int = 0
    computation_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'signal_validity_flag': self.signal_validity_flag,
            'autocorr_check_passed': self.autocorr_check_passed,
            'stationarity_check_passed': self.stationarity_check_passed,
            'distribution_check_passed': self.distribution_check_passed,
            'correlation_check_passed': self.correlation_check_passed,
            'volatility_check_passed': self.volatility_check_passed,
            'trend_consistency': float(self.trend_consistency),
            'regime_quality_score': float(self.regime_quality_score),
            'failure_reasons': self.failure_reasons,
            'bars_used': self.bars_used,
            'computation_timestamp': self.computation_timestamp.isoformat() if self.computation_timestamp else None,
        }


@dataclass
class RegimeState:
    """
    Current market regime characterization.
    
    Used for regime-aware signal generation.
    """
    
    # Trend regime
    trend_regime: str = "NEUTRAL"  # STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN
    trend_strength: float = 0.0
    
    # Volatility regime
    volatility_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    volatility_level: float = 0.0
    
    # Market structure
    efficiency_ratio: float = 0.5  # 0=random, 1=trending
    market_phase: str = "CONSOLIDATION"  # TRENDING, CONSOLIDATION, REVERSAL
    
    # Correlation regime (multi-symbol)
    correlation_regime: str = "NORMAL"  # LOW, NORMAL, HIGH
    avg_correlation: float = 0.0
    
    # Regime stability
    regime_stable: bool = False
    regime_change_detected: bool = False
    
    # Timestamps
    regime_start: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'trend_regime': self.trend_regime,
            'trend_strength': float(self.trend_strength),
            'volatility_regime': self.volatility_regime,
            'volatility_level': float(self.volatility_level),
            'efficiency_ratio': float(self.efficiency_ratio),
            'market_phase': self.market_phase,
            'correlation_regime': self.correlation_regime,
            'avg_correlation': float(self.avg_correlation),
            'regime_stable': self.regime_stable,
            'regime_change_detected': self.regime_change_detected,
            'regime_start': self.regime_start.isoformat() if self.regime_start else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class QuantStatsOutput:
    """
    Complete QSE output for a single symbol/timeframe/timestamp.
    
    Combines metrics, validation, and regime state.
    """
    
    # Identification
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Core outputs
    metrics: StatisticalMetrics
    validation: SignalValidation
    regime: RegimeState
    
    # Configuration
    config_hash: str
    config_version: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'metrics': self.metrics.to_dict(),
            'validation': self.validation.to_dict(),
            'regime': self.regime.to_dict(),
            'config_hash': self.config_hash,
            'config_version': self.config_version,
        }
