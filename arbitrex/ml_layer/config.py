"""
ML Layer Configuration

Defines thresholds, model parameters, and operational settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import hashlib
import json


@dataclass
class RegimeConfig:
    """Configuration for regime classification"""
    
    # Regime detection thresholds
    trending_min_efficiency: float = 0.65  # ER threshold for trending
    ranging_max_volatility_pct: float = 20  # Below 20th percentile = ranging
    stressed_min_volatility_pct: float = 90  # Above 90th percentile = stressed
    
    # Minimum confidence for regime classification
    min_confidence: float = 0.60
    
    # Smoothing (temporal consistency)
    regime_smoothing_window: int = 3  # bars
    
    # Feature requirements
    min_bars_required: int = 100


@dataclass
class SignalFilterConfig:
    """Configuration for signal filter (momentum continuation)"""
    
    # Entry/exit thresholds (prevent flip-flopping)
    entry_threshold: float = 0.55  # P(success) > 0.55 to enter
    exit_threshold: float = 0.45   # P(success) < 0.45 to exit
    
    # Confidence bands
    high_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.40
    
    # Allowed regimes for trading
    allowed_regimes: List[str] = field(default_factory=lambda: ['TRENDING'])
    
    # Feature requirements
    min_bars_required: int = 100


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Model type
    model_type: str = "lightgbm"  # "lightgbm", "xgboost", "logistic"
    
    # Training parameters
    max_depth: int = 6
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_samples: int = 20
    
    # Validation
    n_splits: int = 5  # Time-series cross-validation
    test_size: float = 0.2
    
    # Feature engineering
    use_lag_features: bool = True
    max_lags: int = 3
    use_time_encoding: bool = True
    
    # Model selection criteria
    min_auc: float = 0.55  # Minimum AUC for deployment
    min_accuracy: float = 0.52
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'model_type': self.model_type,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'min_child_samples': self.min_child_samples,
            'n_splits': self.n_splits,
            'test_size': self.test_size,
            'use_lag_features': self.use_lag_features,
            'max_lags': self.max_lags,
            'use_time_encoding': self.use_time_encoding,
            'min_auc': self.min_auc,
            'min_accuracy': self.min_accuracy
        }


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Walk-forward window
    training_window: int = 5000  # bars
    retraining_frequency: int = 500  # retrain every N bars
    min_training_samples: int = 1000
    
    # Label construction
    momentum_horizon: int = 10  # bars ahead for success label
    
    # Data filtering
    exclude_stressed: bool = True  # Exclude stressed periods from training
    exclude_gaps: bool = True  # Exclude data gaps
    max_gap_bars: int = 5
    
    # Feature versioning
    feature_version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'training_window': self.training_window,
            'retraining_frequency': self.retraining_frequency,
            'min_training_samples': self.min_training_samples,
            'momentum_horizon': self.momentum_horizon,
            'exclude_stressed': self.exclude_stressed,
            'exclude_gaps': self.exclude_gaps,
            'max_gap_bars': self.max_gap_bars,
            'feature_version': self.feature_version
        }


@dataclass
class GovernanceConfig:
    """Configuration for model governance and auditability"""
    
    # Model versioning
    model_version_format: str = "v{major}.{minor}.{patch}"
    auto_version: bool = True
    
    # Logging
    log_predictions: bool = True
    enable_prediction_logging: bool = True  # Enable disk logging
    log_feature_importance: bool = True
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_window: int = 100  # bars
    drift_threshold: float = 0.10  # 10% change in distribution
    
    # Performance monitoring
    monitor_performance: bool = True
    performance_window: int = 500  # bars
    performance_alert_threshold: float = 0.05  # 5% drop in AUC
    
    # Model storage
    save_models: bool = True
    model_storage_path: str = "arbitrex/ml_layer/models"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'model_version_format': self.model_version_format,
            'auto_version': self.auto_version,
            'log_predictions': self.log_predictions,
            'enable_prediction_logging': self.enable_prediction_logging,
            'log_feature_importance': self.log_feature_importance,
            'enable_drift_detection': self.enable_drift_detection,
            'drift_window': self.drift_window,
            'drift_threshold': self.drift_threshold,
            'monitor_performance': self.monitor_performance,
            'performance_window': self.performance_window,
            'performance_alert_threshold': self.performance_alert_threshold,
            'save_models': self.save_models,
            'model_storage_path': self.model_storage_path
        }


@dataclass
class MLConfig:
    """
    Master configuration for ML Layer.
    
    All thresholds, parameters, and operational settings.
    """
    
    config_version: str = "1.0.0"
    
    # Sub-configurations
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signal_filter: SignalFilterConfig = field(default_factory=SignalFilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    
    # Timeframes
    regime_timeframe: str = "4H"  # Daily or 4H for regime
    signal_timeframe: str = "4H"  # 4H for signal filtering
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'config_version': self.config_version,
            'regime_timeframe': self.regime_timeframe,
            'signal_timeframe': self.signal_timeframe,
            'regime': {
                'trending_min_efficiency': self.regime.trending_min_efficiency,
                'ranging_max_volatility_pct': self.regime.ranging_max_volatility_pct,
                'stressed_min_volatility_pct': self.regime.stressed_min_volatility_pct,
                'min_confidence': self.regime.min_confidence,
                'regime_smoothing_window': self.regime.regime_smoothing_window,
                'min_bars_required': self.regime.min_bars_required
            },
            'signal_filter': {
                'entry_threshold': self.signal_filter.entry_threshold,
                'exit_threshold': self.signal_filter.exit_threshold,
                'high_confidence_threshold': self.signal_filter.high_confidence_threshold,
                'low_confidence_threshold': self.signal_filter.low_confidence_threshold,
                'allowed_regimes': self.signal_filter.allowed_regimes,
                'min_bars_required': self.signal_filter.min_bars_required
            },
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'governance': self.governance.to_dict()
        }
    
    def get_config_hash(self) -> str:
        """
        Generate deterministic hash of configuration.
        Used for versioning and reproducibility.
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MLConfig':
        """Create config from dictionary"""
        return cls(
            config_version=config_dict.get('config_version', '1.0.0'),
            regime_timeframe=config_dict.get('regime_timeframe', '4H'),
            signal_timeframe=config_dict.get('signal_timeframe', '4H'),
            regime=RegimeConfig(**config_dict.get('regime', {})),
            signal_filter=SignalFilterConfig(**config_dict.get('signal_filter', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            governance=GovernanceConfig(**config_dict.get('governance', {}))
        )
