"""
Quantitative Statistics Engine Configuration

Defines thresholds and parameters for statistical validation.
"""

from dataclasses import dataclass
from typing import List, Tuple
import hashlib
import json


@dataclass
class AutocorrelationConfig:
    """Autocorrelation & trend persistence settings"""
    lags: List[int] = None  # [1, 5, 10, 20]
    min_autocorr_threshold: float = 0.15  # Minimum autocorr for trend signal
    rolling_window: int = 60  # Bars for rolling autocorr
    enabled: bool = True
    
    def __post_init__(self):
        if self.lags is None:
            self.lags = [1, 5, 10, 20]


@dataclass
class StationarityConfig:
    """ADF stationarity test settings"""
    significance_level: float = 0.05  # p-value threshold
    rolling_window: int = 60  # Minimum bars for ADF
    max_lag: int = 10  # Max lag for ADF test
    enabled: bool = True


@dataclass
class DistributionConfig:
    """Distribution stability & outlier detection"""
    rolling_window: int = 60  # Window for mean/std
    z_score_threshold: float = 3.0  # Outlier threshold
    min_samples: int = 30  # Minimum samples for stability
    enabled: bool = True


@dataclass
class CorrelationConfig:
    """Cross-pair correlation settings"""
    rolling_window: int = 60  # Correlation window
    max_correlation_threshold: float = 0.85  # Suppress if corr too high
    min_pairs: int = 2  # Minimum pairs for correlation check
    enabled: bool = True


@dataclass
class VolatilityConfig:
    """Volatility regime filtering"""
    rolling_window: int = 60  # Window for vol percentile
    min_percentile: float = 10.0  # Suppress ultra-low vol
    max_percentile: float = 90.0  # Suppress extreme vol
    atr_window: int = 14  # ATR calculation window
    enabled: bool = True


@dataclass
class ValidationConfig:
    """Signal validation thresholds"""
    min_trend_consistency: float = 0.20  # Minimum trend score
    require_trend_persistence: bool = True  # Require trend check
    require_stationarity: bool = True  # Must pass ADF
    require_distribution_stability: bool = True  # No outliers
    require_correlation_check: bool = False  # Cross-pair check (optional)
    require_volatility_filter: bool = True  # Vol regime check


@dataclass
class QuantStatsConfig:
    """
    Master configuration for Quantitative Statistics Engine.
    
    All thresholds and windows are based on statistical significance
    and empirical testing for financial time series.
    """
    
    # Component configs
    autocorrelation: AutocorrelationConfig = None
    stationarity: StationarityConfig = None
    distribution: DistributionConfig = None
    correlation: CorrelationConfig = None
    volatility: VolatilityConfig = None
    validation: ValidationConfig = None
    
    # General settings
    config_version: str = "1.0.0"
    min_bars_required: int = 120  # Minimum history for all tests
    verbose_logging: bool = True
    fail_on_error: bool = False  # Continue on individual test errors
    
    def __post_init__(self):
        """Initialize sub-configs with defaults"""
        if self.autocorrelation is None:
            self.autocorrelation = AutocorrelationConfig()
        if self.stationarity is None:
            self.stationarity = StationarityConfig()
        if self.distribution is None:
            self.distribution = DistributionConfig()
        if self.correlation is None:
            self.correlation = CorrelationConfig()
        if self.volatility is None:
            self.volatility = VolatilityConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'config_version': self.config_version,
            'min_bars_required': self.min_bars_required,
            'verbose_logging': self.verbose_logging,
            'fail_on_error': self.fail_on_error,
            'autocorrelation': {
                'lags': self.autocorrelation.lags,
                'min_autocorr_threshold': self.autocorrelation.min_autocorr_threshold,
                'rolling_window': self.autocorrelation.rolling_window,
                'enabled': self.autocorrelation.enabled,
            },
            'stationarity': {
                'significance_level': self.stationarity.significance_level,
                'rolling_window': self.stationarity.rolling_window,
                'max_lag': self.stationarity.max_lag,
                'enabled': self.stationarity.enabled,
            },
            'distribution': {
                'rolling_window': self.distribution.rolling_window,
                'z_score_threshold': self.distribution.z_score_threshold,
                'min_samples': self.distribution.min_samples,
                'enabled': self.distribution.enabled,
            },
            'correlation': {
                'rolling_window': self.correlation.rolling_window,
                'max_correlation_threshold': self.correlation.max_correlation_threshold,
                'min_pairs': self.correlation.min_pairs,
                'enabled': self.correlation.enabled,
            },
            'volatility': {
                'rolling_window': self.volatility.rolling_window,
                'min_percentile': self.volatility.min_percentile,
                'max_percentile': self.volatility.max_percentile,
                'atr_window': self.volatility.atr_window,
                'enabled': self.volatility.enabled,
            },
            'validation': {
                'min_trend_consistency': self.validation.min_trend_consistency,
                'require_trend_persistence': self.validation.require_trend_persistence,
                'require_stationarity': self.validation.require_stationarity,
                'require_distribution_stability': self.validation.require_distribution_stability,
                'require_correlation_check': self.validation.require_correlation_check,
                'require_volatility_filter': self.validation.require_volatility_filter,
            }
        }
    
    def get_config_hash(self) -> str:
        """
        Generate deterministic hash of configuration.
        
        Returns:
            Hash string for versioning
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'QuantStatsConfig':
        """Create config from dictionary"""
        autocorr_dict = config_dict.get('autocorrelation', {})
        stat_dict = config_dict.get('stationarity', {})
        dist_dict = config_dict.get('distribution', {})
        corr_dict = config_dict.get('correlation', {})
        vol_dict = config_dict.get('volatility', {})
        val_dict = config_dict.get('validation', {})
        
        return cls(
            config_version=config_dict.get('config_version', '1.0.0'),
            min_bars_required=config_dict.get('min_bars_required', 120),
            verbose_logging=config_dict.get('verbose_logging', True),
            fail_on_error=config_dict.get('fail_on_error', False),
            autocorrelation=AutocorrelationConfig(**autocorr_dict) if autocorr_dict else None,
            stationarity=StationarityConfig(**stat_dict) if stat_dict else None,
            distribution=DistributionConfig(**dist_dict) if dist_dict else None,
            correlation=CorrelationConfig(**corr_dict) if corr_dict else None,
            volatility=VolatilityConfig(**vol_dict) if vol_dict else None,
            validation=ValidationConfig(**val_dict) if val_dict else None,
        )
