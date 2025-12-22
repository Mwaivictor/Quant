"""
ML Layer Output Schemas

Defines data structures for ML predictions, regime labels, and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


class RegimeLabel(str, Enum):
    """Market regime classification"""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    STRESSED = "STRESSED"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimePrediction:
    """Regime classification output"""
    
    # Primary output
    regime_label: RegimeLabel
    regime_confidence: float  # Probability of predicted regime
    
    # Probabilities for all regimes
    prob_trending: float
    prob_ranging: float
    prob_stressed: float
    
    # Regime metrics (used for classification)
    efficiency_ratio: float
    volatility_percentile: float
    correlation_regime: str
    
    # Temporal consistency
    regime_stable: bool  # Same regime for N consecutive bars
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'regime_label': self.regime_label.value,
            'regime_confidence': float(self.regime_confidence),
            'prob_trending': float(self.prob_trending),
            'prob_ranging': float(self.prob_ranging),
            'prob_stressed': float(self.prob_stressed),
            'efficiency_ratio': float(self.efficiency_ratio),
            'volatility_percentile': float(self.volatility_percentile),
            'correlation_regime': self.correlation_regime,
            'regime_stable': bool(self.regime_stable)
        }


@dataclass
class SignalPrediction:
    """Signal filter output (momentum continuation probability)"""
    
    # Primary output
    momentum_success_prob: float  # P(momentum_success | X_t) ∈ [0,1]
    
    # Decision flags
    should_enter: bool  # prob > entry_threshold
    should_exit: bool   # prob < exit_threshold
    
    # Confidence bands
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    
    # Feature contributions (top 5)
    top_features: Dict[str, float]  # Feature importance for this prediction
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'momentum_success_prob': float(self.momentum_success_prob),
            'should_enter': bool(self.should_enter),
            'should_exit': bool(self.should_exit),
            'confidence_level': self.confidence_level,
            'top_features': {k: float(v) for k, v in self.top_features.items()}
        }


@dataclass
class ModelMetadata:
    """Model versioning and governance metadata"""
    
    # Model identification
    model_version: str
    model_type: str  # "regime_classifier" or "signal_filter"
    
    # Training info
    trained_on: str  # ISO datetime
    training_samples: int
    feature_version: str
    config_hash: str
    
    # Performance metrics
    train_auc: Optional[float] = None
    test_auc: Optional[float] = None
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'model_version': self.model_version,
            'model_type': self.model_type,
            'trained_on': self.trained_on,
            'training_samples': self.training_samples,
            'feature_version': self.feature_version,
            'config_hash': self.config_hash,
            'train_auc': float(self.train_auc) if self.train_auc else None,
            'test_auc': float(self.test_auc) if self.test_auc else None,
            'train_accuracy': float(self.train_accuracy) if self.train_accuracy else None,
            'test_accuracy': float(self.test_accuracy) if self.test_accuracy else None,
            'feature_importance': {k: float(v) for k, v in self.feature_importance.items()} if self.feature_importance else None
        }


@dataclass
class MLPrediction:
    """Combined ML prediction (regime + signal filter)"""
    
    # Regime prediction
    regime: RegimePrediction
    
    # Signal filter prediction
    signal: SignalPrediction
    
    # Final decision
    allow_trade: bool  # Regime allowed AND signal confidence sufficient
    
    # Reasons for decision
    decision_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'regime': self.regime.to_dict(),
            'signal': self.signal.to_dict(),
            'allow_trade': bool(self.allow_trade),
            'decision_reasons': list(self.decision_reasons)
        }


@dataclass
class MLOutput:
    """
    Complete ML Layer output.
    
    This is what flows to Signal Generator and Risk Manager.
    """
    
    # Identification
    timestamp: datetime
    symbol: str
    timeframe: str
    bar_index: int
    
    # ML predictions
    prediction: MLPrediction
    
    # Model metadata
    regime_model: ModelMetadata
    signal_model: ModelMetadata
    
    # Config versioning
    config_hash: str
    ml_version: str
    
    # Processing metadata
    processing_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'bar_index': self.bar_index,
            'prediction': self.prediction.to_dict(),
            'regime_model': self.regime_model.to_dict(),
            'signal_model': self.signal_model.to_dict(),
            'config_hash': self.config_hash,
            'ml_version': self.ml_version,
            'processing_time_ms': float(self.processing_time_ms)
        }


@dataclass
class MLHealthMetrics:
    """Health metrics for ML Layer monitoring"""
    
    # Performance
    recent_regime_accuracy: float
    recent_signal_accuracy: float
    avg_processing_time_ms: float
    
    # Predictions
    total_predictions: int
    trades_allowed: int
    trades_suppressed: int
    allow_rate: float
    
    # Regime distribution
    trending_count: int
    ranging_count: int
    stressed_count: int
    
    # Model status
    regime_model_version: str
    signal_model_version: str
    last_prediction_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'recent_regime_accuracy': float(self.recent_regime_accuracy),
            'recent_signal_accuracy': float(self.recent_signal_accuracy),
            'avg_processing_time_ms': float(self.avg_processing_time_ms),
            'total_predictions': self.total_predictions,
            'trades_allowed': self.trades_allowed,
            'trades_suppressed': self.trades_suppressed,
            'allow_rate': float(self.allow_rate),
            'trending_count': self.trending_count,
            'ranging_count': self.ranging_count,
            'stressed_count': self.stressed_count,
            'regime_model_version': self.regime_model_version,
            'signal_model_version': self.signal_model_version,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }
