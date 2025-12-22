"""
Feature Engine Configuration

Defines all feature computation parameters, windows, and versioning.
All parameters must be explicitly versioned for reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import hashlib


@dataclass
class ReturnsMomentumConfig:
    """Category A: Returns & Momentum"""
    
    # Rolling return windows
    return_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # Momentum calculation
    momentum_window: int = 12
    momentum_vol_window: int = 12
    
    # Enable/disable
    enabled: bool = True


@dataclass
class VolatilityConfig:
    """Category B: Volatility Structure"""
    
    # Rolling volatility windows
    vol_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    
    # Normalized ATR
    atr_window: int = 14
    
    # Enable/disable
    enabled: bool = True


@dataclass
class TrendConfig:
    """Category C: Trend Structure (Descriptive)"""
    
    # Moving average windows
    ma_windows: List[int] = field(default_factory=lambda: [12, 24, 50])
    
    # MA slope calculation
    slope_window: int = 3
    
    # Price-to-MA distance
    distance_atr_window: int = 14
    
    # Enable/disable
    enabled: bool = True


@dataclass
class EfficiencyConfig:
    """Category D: Range & Market Efficiency"""
    
    # Efficiency Ratio (Kaufman)
    er_direction_window: int = 10
    er_volatility_window: int = 10
    
    # Range compression
    range_window: int = 12
    range_atr_window: int = 14
    
    # Enable/disable
    enabled: bool = True


@dataclass
class RegimeConfig:
    """Category E: Regime Features (Daily Only)"""
    
    # Trend regime detection
    trend_ma_fast: int = 20
    trend_ma_slow: int = 50
    trend_buffer: float = 0.02  # 2% buffer for regime flip
    
    # Stress indicator
    stress_short_window: int = 6
    stress_long_window: int = 24
    
    # Enable/disable
    enabled: bool = True
    
    # Daily timeframe only
    daily_only: bool = True


@dataclass
class ExecutionConfig:
    """Category F: Execution/Cost Filters (Optional)"""
    
    # Spread ratio
    spread_atr_window: int = 14
    spread_avg_window: int = 6
    
    # Enable/disable (never passed to ML)
    enabled: bool = False
    
    # Execution filter only
    ml_excluded: bool = True


@dataclass
class NormalizationConfig:
    """Normalization parameters"""
    
    # Rolling z-score windows
    norm_window: int = 60
    
    # Minimum bars required before normalization
    min_bars_required: int = 60
    
    # Clip outliers
    z_score_clip: float = 5.0
    
    # Use robust statistics (median/MAD)
    use_robust: bool = False


@dataclass
class FeatureEngineConfig:
    """
    Master configuration for Feature Engine.
    
    All parameters versioned for reproducibility.
    """
    
    # Configuration version
    config_version: str = "1.0.0"
    
    # Feature categories
    returns_momentum: ReturnsMomentumConfig = field(default_factory=ReturnsMomentumConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Normalization
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    # Quality gates
    min_valid_bars_required: int = 100  # Minimum bars before feature computation
    min_valid_bar_pct: float = 0.95  # Require 95% valid bars in window
    
    # Execution
    fail_on_critical_error: bool = True
    verbose_logging: bool = True
    
    def get_config_hash(self) -> str:
        """
        Generate deterministic hash of configuration.
        
        Used for feature versioning and cache invalidation.
        """
        config_dict = self._to_dict_no_hash()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _to_dict_no_hash(self) -> dict:
        """Internal method: serialize config without hash (prevents recursion)"""
        return {
            "config_version": self.config_version,
            "returns_momentum": {
                "return_windows": self.returns_momentum.return_windows,
                "momentum_window": self.returns_momentum.momentum_window,
                "momentum_vol_window": self.returns_momentum.momentum_vol_window,
                "enabled": self.returns_momentum.enabled,
            },
            "volatility": {
                "vol_windows": self.volatility.vol_windows,
                "atr_window": self.volatility.atr_window,
                "enabled": self.volatility.enabled,
            },
            "trend": {
                "ma_windows": self.trend.ma_windows,
                "slope_window": self.trend.slope_window,
                "distance_atr_window": self.trend.distance_atr_window,
                "enabled": self.trend.enabled,
            },
            "efficiency": {
                "er_direction_window": self.efficiency.er_direction_window,
                "er_volatility_window": self.efficiency.er_volatility_window,
                "range_window": self.efficiency.range_window,
                "range_atr_window": self.efficiency.range_atr_window,
                "enabled": self.efficiency.enabled,
            },
            "regime": {
                "trend_ma_fast": self.regime.trend_ma_fast,
                "trend_ma_slow": self.regime.trend_ma_slow,
                "trend_buffer": self.regime.trend_buffer,
                "stress_short_window": self.regime.stress_short_window,
                "stress_long_window": self.regime.stress_long_window,
                "enabled": self.regime.enabled,
                "daily_only": self.regime.daily_only,
            },
            "execution": {
                "spread_atr_window": self.execution.spread_atr_window,
                "spread_avg_window": self.execution.spread_avg_window,
                "enabled": self.execution.enabled,
                "ml_excluded": self.execution.ml_excluded,
            },
            "normalization": {
                "norm_window": self.normalization.norm_window,
                "min_bars_required": self.normalization.min_bars_required,
                "z_score_clip": self.normalization.z_score_clip,
                "use_robust": self.normalization.use_robust,
            },
            "quality_gates": {
                "min_valid_bars_required": self.min_valid_bars_required,
                "min_valid_bar_pct": self.min_valid_bar_pct,
            },
            "execution": {
                "fail_on_critical_error": self.fail_on_critical_error,
                "verbose_logging": self.verbose_logging,
            }
        }
    
    def to_dict(self) -> dict:
        """Serialize configuration to dictionary with hash"""
        d = self._to_dict_no_hash()
        d["config_hash"] = self.get_config_hash()
        return d
    
    def to_json(self) -> str:
        """Serialize configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


# Default configuration instance
DEFAULT_CONFIG = FeatureEngineConfig()
