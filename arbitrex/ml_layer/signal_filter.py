"""
Signal Filter Model

Predicts probability of momentum signal success: P(momentum_success | X_t)
Uses gradient-boosted trees (LightGBM) or logistic regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

from arbitrex.ml_layer.config import MLConfig, SignalFilterConfig
from arbitrex.ml_layer.schemas import SignalPrediction, ModelMetadata

LOG = logging.getLogger(__name__)


class SignalFilter:
    """
    Signal filter for momentum continuation probability.
    
    Answers: "Is this validated signal likely to succeed?"
    
    Output: P(momentum_success | X_t) where success means:
        sign(return_{t+1 → t+H}) == momentum_direction_t
    
    Decision Logic:
        - Enter if P(success) > entry_threshold (e.g., 0.55)
        - Exit if P(success) < exit_threshold (e.g., 0.45)
        - Hysteresis prevents flip-flopping
    """
    
    def __init__(self, config: SignalFilterConfig):
        """
        Initialize signal filter.
        
        Args:
            config: Signal filter configuration
        """
        self.config = config
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        
        LOG.info("SignalFilter initialized (rule-based fallback)")
    
    def extract_signal_features(
        self,
        feature_df: pd.DataFrame,
        qse_output: Optional[Dict] = None,
        regime_label: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract features for signal filtering.
        
        Args:
            feature_df: Features from Feature Engine
            qse_output: QSE validation output
            regime_label: Current regime
        
        Returns:
            Dictionary of signal features
        """
        if feature_df.empty:
            return {}
        
        latest = feature_df.iloc[-1]
        features = {}
        
        # ===== MOMENTUM FEATURES =====
        if 'momentum_score' in feature_df.columns:
            features['momentum_score'] = latest['momentum_score']
        
        if 'rolling_return_20' in feature_df.columns:
            features['rolling_return_20'] = latest['rolling_return_20']
        
        if 'trend_consistency' in feature_df.columns:
            features['trend_consistency'] = latest['trend_consistency']
        
        # MA distances (trend strength)
        for ma in ['ma_distance_20', 'ma_distance_50']:
            if ma in feature_df.columns:
                features[ma] = latest[ma]
        
        # ===== VOLATILITY FEATURES =====
        if 'atr' in feature_df.columns:
            features['atr'] = latest['atr']
        
        if 'rolling_vol' in feature_df.columns:
            features['rolling_vol'] = latest['rolling_vol']
        
        if 'vol_percentile' in feature_df.columns:
            features['vol_percentile'] = latest['vol_percentile']
        
        if 'vol_slope' in feature_df.columns:
            features['vol_slope'] = latest['vol_slope']
        
        # ===== MARKET STRUCTURE FEATURES =====
        if 'cross_pair_correlation' in feature_df.columns:
            features['cross_pair_correlation'] = latest['cross_pair_correlation']
        
        if 'dispersion' in feature_df.columns:
            features['dispersion'] = latest['dispersion']
        
        # ===== QSE FEATURES =====
        if qse_output:
            metrics = qse_output.get('metrics', {})
            validation = qse_output.get('validation', {})
            regime_state = qse_output.get('regime', {})
            
            # Statistical metrics
            features['trend_persistence_score'] = metrics.get('trend_persistence_score', 0.0)
            features['adf_stationary'] = 1.0 if metrics.get('adf_stationary', False) else 0.0
            features['z_score_abs'] = abs(metrics.get('z_score', 0.0))
            
            # Validation flags
            features['autocorr_check'] = 1.0 if validation.get('autocorr_check_passed', False) else 0.0
            features['stationarity_check'] = 1.0 if validation.get('stationarity_check_passed', False) else 0.0
            features['distribution_check'] = 1.0 if validation.get('distribution_check_passed', False) else 0.0
            
            # Regime state
            features['trend_strength'] = regime_state.get('trend_strength', 0.0)
            features['efficiency_ratio'] = regime_state.get('efficiency_ratio', 0.5)
        
        # ===== REGIME ENCODING =====
        if regime_label:
            features['regime_trending'] = 1.0 if regime_label == 'TRENDING' else 0.0
            features['regime_ranging'] = 1.0 if regime_label == 'RANGING' else 0.0
            features['regime_stressed'] = 1.0 if regime_label == 'STRESSED' else 0.0
        
        return features
    
    def predict_rule_based(self, features: Dict[str, float]) -> float:
        """
        Rule-based fallback for signal probability.
        
        Uses weighted combination of key indicators.
        
        Args:
            features: Signal features
        
        Returns:
            Probability of momentum success [0, 1]
        """
        # Base probability
        prob = 0.50
        
        # Momentum contribution (±0.15)
        momentum_score = features.get('momentum_score', 0.0)
        prob += np.clip(momentum_score * 0.15, -0.15, 0.15)
        
        # Trend consistency contribution (±0.10)
        trend_consistency = features.get('trend_consistency', 0.5)
        prob += (trend_consistency - 0.5) * 0.20
        
        # QSE trend persistence contribution (±0.10)
        trend_persistence = features.get('trend_persistence_score', 0.2)
        if trend_persistence > 0.2:
            prob += min((trend_persistence - 0.2) * 0.25, 0.10)
        else:
            prob -= min((0.2 - trend_persistence) * 0.25, 0.10)
        
        # Volatility penalty (if too high or too low)
        vol_percentile = features.get('vol_percentile', 50)
        if vol_percentile > 85 or vol_percentile < 15:
            prob -= 0.10
        
        # Regime bonus/penalty
        if features.get('regime_trending', 0.0) == 1.0:
            prob += 0.05
        elif features.get('regime_stressed', 0.0) == 1.0:
            prob -= 0.10
        
        # Statistical checks bonus
        if features.get('autocorr_check', 0.0) == 1.0:
            prob += 0.03
        if features.get('stationarity_check', 0.0) == 1.0:
            prob += 0.02
        if features.get('distribution_check', 0.0) == 1.0:
            prob += 0.03
        
        # Clip to valid range
        prob = np.clip(prob, 0.0, 1.0)
        
        return prob
    
    def get_top_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get top contributing features for explainability.
        
        Args:
            features: Input features
        
        Returns:
            Top 5 features with their contributions
        """
        # If trained model exists with feature importance, use that
        if self.feature_importance:
            # Get top features by importance
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            return dict(sorted_features)
        
        # Otherwise, use rule-based importance
        contributions = {}
        
        if 'momentum_score' in features:
            contributions['momentum_score'] = abs(features['momentum_score'])
        
        if 'trend_consistency' in features:
            contributions['trend_consistency'] = abs(features['trend_consistency'] - 0.5)
        
        if 'trend_persistence_score' in features:
            contributions['trend_persistence_score'] = abs(features['trend_persistence_score'] - 0.2)
        
        if 'vol_percentile' in features:
            contributions['vol_percentile'] = abs(features['vol_percentile'] - 50) / 50
        
        if 'efficiency_ratio' in features:
            contributions['efficiency_ratio'] = features['efficiency_ratio']
        
        # Sort and get top 5
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return dict(sorted_contrib)
    
    def predict(
        self,
        feature_df: pd.DataFrame,
        qse_output: Optional[Dict] = None,
        regime_label: Optional[str] = None
    ) -> SignalPrediction:
        """
        Predict momentum success probability.
        
        Args:
            feature_df: Features from Feature Engine
            qse_output: QSE validation output
            regime_label: Current regime
        
        Returns:
            SignalPrediction with probability and decision flags
        """
        # Extract features
        features = self.extract_signal_features(feature_df, qse_output, regime_label)
        
        if not features:
            # Not enough data - return neutral
            return SignalPrediction(
                momentum_success_prob=0.50,
                should_enter=False,
                should_exit=False,
                confidence_level="LOW",
                top_features={}
            )
        
        # Predict probability
        if self.model:
            # Use trained model (future implementation)
            prob = self.model.predict_proba([list(features.values())])[0][1]
        else:
            # Use rule-based fallback
            prob = self.predict_rule_based(features)
        
        # Decision flags
        should_enter = prob > self.config.entry_threshold
        should_exit = prob < self.config.exit_threshold
        
        # Confidence level
        if prob >= self.config.high_confidence_threshold:
            confidence_level = "HIGH"
        elif prob <= self.config.low_confidence_threshold:
            confidence_level = "LOW"
        else:
            confidence_level = "MEDIUM"
        
        # Get top contributing features
        top_features = self.get_top_features(features)
        
        return SignalPrediction(
            momentum_success_prob=prob,
            should_enter=should_enter,
            should_exit=should_exit,
            confidence_level=confidence_level,
            top_features=top_features
        )
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        if self.metadata:
            return self.metadata
        
        # Create default metadata for rule-based model
        return ModelMetadata(
            model_version="v1.0.0-rule-based",
            model_type="signal_filter",
            trained_on=datetime.now().isoformat(),
            training_samples=0,
            feature_version="1.0.0",
            config_hash="rule_based",
            train_auc=None,
            test_auc=None,
            feature_importance=self.feature_importance
        )
