"""
Regime Classification Model

Detects market regime: Trending, Ranging, or Stressed.
Uses gradient-boosted trees for classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from arbitrex.ml_layer.config import MLConfig, RegimeConfig
from arbitrex.ml_layer.schemas import RegimeLabel, RegimePrediction, ModelMetadata

LOG = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Market regime classification using rule-based + ML hybrid.
    
    Regime Types:
        TRENDING: High efficiency ratio, directional persistence
        RANGING: Low volatility, mean-reverting behavior
        STRESSED: High volatility, low correlation structure
    
    Primary method: Rule-based classification with ML fallback
    """
    
    def __init__(self, config: RegimeConfig):
        """
        Initialize regime classifier.
        
        Args:
            config: Regime configuration
        """
        self.config = config
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.recent_regimes = []  # For smoothing
        
        LOG.info("RegimeClassifier initialized (rule-based)")
    
    def extract_regime_features(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for regime classification.
        
        Args:
            feature_df: DataFrame with computed features
        
        Returns:
            Dictionary of regime features
        """
        if feature_df.empty:
            return {}
        
        # Get latest bar
        latest = feature_df.iloc[-1]
        
        features = {}
        
        # Momentum features
        if 'momentum_score' in feature_df.columns:
            features['momentum_score'] = latest['momentum_score']
        
        if 'trend_consistency' in feature_df.columns:
            features['trend_consistency'] = latest['trend_consistency']
        
        # Volatility features
        if 'atr' in feature_df.columns:
            features['atr'] = latest['atr']
            
        if 'rolling_vol' in feature_df.columns:
            features['rolling_vol'] = latest['rolling_vol']
            
        if 'vol_percentile' in feature_df.columns:
            features['vol_percentile'] = latest['vol_percentile']
        
        # Market structure features
        if 'cross_pair_correlation' in feature_df.columns:
            features['cross_pair_correlation'] = latest['cross_pair_correlation']
        
        # Efficiency ratio (key for trending vs ranging)
        if 'ma_distance_20' in feature_df.columns and 'atr' in feature_df.columns:
            # ER = net_move / sum_of_moves
            net_move = abs(feature_df['ma_distance_20'].iloc[-1] - feature_df['ma_distance_20'].iloc[-20])
            sum_moves = feature_df['atr'].iloc[-20:].sum()
            features['efficiency_ratio'] = net_move / max(sum_moves, 1e-8)
        else:
            features['efficiency_ratio'] = 0.5  # Neutral default
        
        return features
    
    def classify_regime_rule_based(
        self,
        features: Dict[str, float]
    ) -> Tuple[RegimeLabel, Dict[str, float]]:
        """
        Classify regime using rule-based logic.
        
        Args:
            features: Regime features
        
        Returns:
            (regime_label, probabilities)
        """
        efficiency_ratio = features.get('efficiency_ratio', 0.5)
        vol_percentile = features.get('vol_percentile', 50)
        
        # Initialize probabilities
        prob_trending = 0.33
        prob_ranging = 0.33
        prob_stressed = 0.33
        
        # Rule 1: High volatility → STRESSED
        if vol_percentile >= self.config.stressed_min_volatility_pct:
            prob_stressed = 0.70
            prob_trending = 0.20
            prob_ranging = 0.10
            regime = RegimeLabel.STRESSED
        
        # Rule 2: High efficiency ratio → TRENDING
        elif efficiency_ratio >= self.config.trending_min_efficiency:
            prob_trending = 0.70
            prob_ranging = 0.20
            prob_stressed = 0.10
            regime = RegimeLabel.TRENDING
        
        # Rule 3: Low volatility → RANGING
        elif vol_percentile <= self.config.ranging_max_volatility_pct:
            prob_ranging = 0.70
            prob_trending = 0.20
            prob_stressed = 0.10
            regime = RegimeLabel.RANGING
        
        # Default: Use efficiency ratio as tiebreaker
        else:
            if efficiency_ratio > 0.5:
                prob_trending = 0.50
                prob_ranging = 0.35
                prob_stressed = 0.15
                regime = RegimeLabel.TRENDING
            else:
                prob_ranging = 0.50
                prob_trending = 0.35
                prob_stressed = 0.15
                regime = RegimeLabel.RANGING
        
        probabilities = {
            'prob_trending': prob_trending,
            'prob_ranging': prob_ranging,
            'prob_stressed': prob_stressed
        }
        
        return regime, probabilities
    
    def smooth_regime(self, regime: RegimeLabel) -> Tuple[RegimeLabel, bool]:
        """
        Apply temporal smoothing to regime prediction.
        
        Args:
            regime: Current regime prediction
        
        Returns:
            (smoothed_regime, is_stable)
        """
        # Add to recent history
        self.recent_regimes.append(regime)
        
        # Keep only recent N
        if len(self.recent_regimes) > self.config.regime_smoothing_window:
            self.recent_regimes.pop(0)
        
        # Not enough history yet
        if len(self.recent_regimes) < self.config.regime_smoothing_window:
            return regime, False
        
        # Check if all recent regimes are same
        is_stable = all(r == self.recent_regimes[0] for r in self.recent_regimes)
        
        # If stable, keep current
        if is_stable:
            return regime, True
        
        # Otherwise, use most common regime
        from collections import Counter
        regime_counts = Counter(self.recent_regimes)
        smoothed_regime = regime_counts.most_common(1)[0][0]
        
        return smoothed_regime, False
    
    def predict(
        self,
        feature_df: pd.DataFrame,
        qse_output: Optional[Dict] = None
    ) -> RegimePrediction:
        """
        Predict market regime.
        
        Args:
            feature_df: Features from Feature Engine
            qse_output: Optional QSE output for additional context
        
        Returns:
            RegimePrediction with regime label and probabilities
        """
        # Extract features
        features = self.extract_regime_features(feature_df)
        
        if not features:
            # Not enough data
            return RegimePrediction(
                regime_label=RegimeLabel.UNKNOWN,
                regime_confidence=0.0,
                prob_trending=0.33,
                prob_ranging=0.33,
                prob_stressed=0.33,
                efficiency_ratio=0.0,
                volatility_percentile=50.0,
                correlation_regime="UNKNOWN",
                regime_stable=False
            )
        
        # Classify using rules
        regime, probabilities = self.classify_regime_rule_based(features)
        
        # Apply temporal smoothing
        smoothed_regime, is_stable = self.smooth_regime(regime)
        
        # Get confidence (probability of predicted regime)
        regime_confidence = probabilities[f'prob_{smoothed_regime.value.lower()}']
        
        # Get correlation regime from QSE if available
        correlation_regime = "UNKNOWN"
        if qse_output and 'regime' in qse_output:
            correlation_regime = qse_output['regime'].get('correlation_regime', 'UNKNOWN')
        
        return RegimePrediction(
            regime_label=smoothed_regime,
            regime_confidence=regime_confidence,
            prob_trending=probabilities['prob_trending'],
            prob_ranging=probabilities['prob_ranging'],
            prob_stressed=probabilities['prob_stressed'],
            efficiency_ratio=features['efficiency_ratio'],
            volatility_percentile=features.get('vol_percentile', 50.0),
            correlation_regime=correlation_regime,
            regime_stable=is_stable
        )
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        if self.metadata:
            return self.metadata
        
        # Create default metadata for rule-based model
        return ModelMetadata(
            model_version="v1.0.0-rule-based",
            model_type="regime_classifier",
            trained_on=datetime.now().isoformat(),
            training_samples=0,
            feature_version="1.0.0",
            config_hash="rule_based",
            train_auc=None,
            test_auc=None,
            feature_importance=None
        )
    
    def reset_smoothing(self):
        """Reset regime smoothing (useful for new session)"""
        self.recent_regimes = []
        LOG.info("Regime smoothing reset")
