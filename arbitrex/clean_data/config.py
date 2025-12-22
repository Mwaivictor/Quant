"""
Clean Data Layer Configuration

Defines thresholds, schedules, and validation parameters.
All parameters must be explicitly versioned for reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import time
import json


@dataclass
class OutlierThresholds:
    """Outlier detection parameters"""
    
    # Price jump detection (in standard deviations)
    price_jump_std_multiplier: float = 5.0
    
    # Rolling window for volatility estimation (bars)
    volatility_window: int = 20
    
    # Minimum bars required for outlier detection
    min_bars_required: int = 20
    
    # Maximum allowed return magnitude (absolute log return)
    max_abs_log_return: float = 0.15  # ~16% single-bar move
    
    # Zero price tolerance
    zero_price_tolerance: float = 1e-10


@dataclass
class MissingBarThresholds:
    """Missing bar detection parameters"""
    
    # Time tolerance for bar matching (seconds)
    timestamp_tolerance_seconds: int = 60
    
    # Maximum consecutive missing bars before symbol exclusion
    max_consecutive_missing: int = 3
    
    # Maximum missing bar percentage before symbol exclusion
    max_missing_percentage: float = 0.05  # 5%


@dataclass
class TimeAlignment:
    """Canonical time grid schedules"""
    
    # 1H schedule: every hour on the hour
    schedule_1H: List[time] = field(default_factory=lambda: [
        time(hour=h, minute=0) for h in range(24)
    ])
    
    # 4H schedule: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    schedule_4H: List[time] = field(default_factory=lambda: [
        time(hour=h, minute=0) for h in [0, 4, 8, 12, 16, 20]
    ])
    
    # 1D schedule: 00:00 UTC daily close
    schedule_1D: List[time] = field(default_factory=lambda: [
        time(hour=0, minute=0)
    ])
    
    # Timezone: always UTC
    timezone: str = "UTC"


@dataclass
class ValidationRules:
    """Bar validation logic parameters"""
    
    # OHLC consistency checks enabled
    enforce_ohlc_consistency: bool = True
    
    # Require valid returns for valid_bar=True
    require_valid_returns: bool = True
    
    # Require non-missing for valid_bar=True
    require_non_missing: bool = True
    
    # Require non-outlier for valid_bar=True
    require_non_outlier: bool = True
    
    # Minimum volume threshold (0 = no minimum)
    min_volume: float = 0.0


@dataclass
class SpreadEstimation:
    """Spread estimation parameters (optional)"""
    
    # Enable spread estimation
    enabled: bool = False
    
    # High-low spread as percentage of close
    use_hl_spread: bool = True
    
    # Exponential smoothing factor for spread
    smoothing_alpha: float = 0.1


@dataclass
class CleanDataConfig:
    """
    Complete configuration for Clean Data Layer.
    
    This configuration must be versioned and stored with each
    clean dataset for full reproducibility.
    """
    
    # Version identifier for this configuration
    config_version: str = "1.0.0"
    
    # Component configurations
    outlier_thresholds: OutlierThresholds = field(default_factory=OutlierThresholds)
    missing_bar_thresholds: MissingBarThresholds = field(default_factory=MissingBarThresholds)
    time_alignment: TimeAlignment = field(default_factory=TimeAlignment)
    validation_rules: ValidationRules = field(default_factory=ValidationRules)
    spread_estimation: SpreadEstimation = field(default_factory=SpreadEstimation)
    
    # Execution parameters
    fail_on_critical_error: bool = True
    verbose_logging: bool = True
    
    def to_dict(self) -> Dict:
        """Serialize configuration to dictionary"""
        return {
            "config_version": self.config_version,
            "outlier_thresholds": {
                "price_jump_std_multiplier": self.outlier_thresholds.price_jump_std_multiplier,
                "volatility_window": self.outlier_thresholds.volatility_window,
                "min_bars_required": self.outlier_thresholds.min_bars_required,
                "max_abs_log_return": self.outlier_thresholds.max_abs_log_return,
                "zero_price_tolerance": self.outlier_thresholds.zero_price_tolerance,
            },
            "missing_bar_thresholds": {
                "timestamp_tolerance_seconds": self.missing_bar_thresholds.timestamp_tolerance_seconds,
                "max_consecutive_missing": self.missing_bar_thresholds.max_consecutive_missing,
                "max_missing_percentage": self.missing_bar_thresholds.max_missing_percentage,
            },
            "time_alignment": {
                "timezone": self.time_alignment.timezone,
                "schedule_1H_count": len(self.time_alignment.schedule_1H),
                "schedule_4H_count": len(self.time_alignment.schedule_4H),
                "schedule_1D_count": len(self.time_alignment.schedule_1D),
            },
            "validation_rules": {
                "enforce_ohlc_consistency": self.validation_rules.enforce_ohlc_consistency,
                "require_valid_returns": self.validation_rules.require_valid_returns,
                "require_non_missing": self.validation_rules.require_non_missing,
                "require_non_outlier": self.validation_rules.require_non_outlier,
                "min_volume": self.validation_rules.min_volume,
            },
            "spread_estimation": {
                "enabled": self.spread_estimation.enabled,
                "use_hl_spread": self.spread_estimation.use_hl_spread,
                "smoothing_alpha": self.spread_estimation.smoothing_alpha,
            },
            "execution": {
                "fail_on_critical_error": self.fail_on_critical_error,
                "verbose_logging": self.verbose_logging,
            }
        }
    
    def to_json(self) -> str:
        """Serialize configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CleanDataConfig":
        """Deserialize configuration from dictionary"""
        return cls(
            config_version=data.get("config_version", "1.0.0"),
            outlier_thresholds=OutlierThresholds(**data.get("outlier_thresholds", {})),
            missing_bar_thresholds=MissingBarThresholds(**data.get("missing_bar_thresholds", {})),
            # time_alignment and other complex objects require custom handling
            fail_on_critical_error=data.get("execution", {}).get("fail_on_critical_error", True),
            verbose_logging=data.get("execution", {}).get("verbose_logging", True),
        )


# Default configuration instance
DEFAULT_CONFIG = CleanDataConfig()
