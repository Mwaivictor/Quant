"""
ML Inference Engine

Orchestrates regime classification and signal filtering.
Provides unified interface for ML predictions.
"""

import time
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import logging

from arbitrex.ml_layer.config import MLConfig
from arbitrex.ml_layer.schemas import (
    MLOutput,
    MLPrediction,
    RegimeLabel,
    RegimePrediction,
    SignalPrediction,
    ModelMetadata
)
from arbitrex.ml_layer.regime_classifier import RegimeClassifier
from arbitrex.ml_layer.signal_filter import SignalFilter

LOG = logging.getLogger(__name__)


class MLInferenceEngine:
    """
    ML Layer inference engine.
    
    Orchestrates:
        1. Regime classification
        2. Signal filtering (momentum continuation probability)
        3. Final trade decision (regime + signal)
    
    Flow:
        Feature Engine → QSE → ML Inference Engine → Signal Generator
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        """
        Initialize ML inference engine.
        
        Args:
            config: ML configuration (uses defaults if None)
        """
        self.config = config or MLConfig()
        
        # Initialize models
        self.regime_classifier = RegimeClassifier(self.config.regime)
        self.signal_filter = SignalFilter(self.config.signal_filter)
        
        # Config hash for versioning
        self.config_hash = self.config.get_config_hash()
        
        LOG.info(f"ML Inference Engine initialized (config hash: {self.config_hash})")
    
    def check_data_requirements(self, feature_df: pd.DataFrame) -> bool:
        """
        Check if we have enough data for prediction.
        
        Args:
            feature_df: Feature DataFrame
        
        Returns:
            True if sufficient data
        """
        min_bars = max(
            self.config.regime.min_bars_required,
            self.config.signal_filter.min_bars_required
        )
        
        if len(feature_df) < min_bars:
            LOG.warning(f"Insufficient data: {len(feature_df)} < {min_bars} bars")
            return False
        
        return True
    
    def predict(
        self,
        symbol: str,
        timeframe: str,
        feature_df: pd.DataFrame,
        qse_output: Optional[Dict] = None,
        bar_index: Optional[int] = None
    ) -> MLOutput:
        """
        Generate ML predictions.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "4H")
            feature_df: Features from Feature Engine
            qse_output: QSE validation output
            bar_index: Current bar index (for causality)
        
        Returns:
            MLOutput with regime, signal probability, and trade decision
        """
        start_time = time.time()
        
        # Default bar index
        if bar_index is None:
            bar_index = len(feature_df) - 1
        
        # Check data requirements
        if not self.check_data_requirements(feature_df):
            # Not enough data - return default
            return self._create_insufficient_data_output(
                symbol, timeframe, bar_index, start_time
            )
        
        # ===== STEP 1: REGIME CLASSIFICATION =====
        regime_pred = self.regime_classifier.predict(feature_df, qse_output)
        
        LOG.debug(f"{symbol} regime: {regime_pred.regime_label.value} "
                  f"(conf: {regime_pred.regime_confidence:.3f})")
        
        # ===== STEP 2: SIGNAL FILTERING =====
        signal_pred = self.signal_filter.predict(
            feature_df,
            qse_output,
            regime_pred.regime_label.value
        )
        
        LOG.debug(f"{symbol} signal prob: {signal_pred.momentum_success_prob:.3f} "
                  f"(conf: {signal_pred.confidence_level})")
        
        # ===== STEP 3: FINAL DECISION =====
        allow_trade, decision_reasons = self._make_trade_decision(
            regime_pred, signal_pred, qse_output
        )
        
        # Create ML prediction
        ml_prediction = MLPrediction(
            regime=regime_pred,
            signal=signal_pred,
            allow_trade=allow_trade,
            decision_reasons=decision_reasons
        )
        
        # Calculate processing time
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Create output
        output = MLOutput(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            bar_index=bar_index,
            prediction=ml_prediction,
            regime_model=self.regime_classifier.get_metadata(),
            signal_model=self.signal_filter.get_metadata(),
            config_hash=self.config_hash,
            ml_version="1.0.0",
            processing_time_ms=elapsed_ms
        )
        
        LOG.info(f"{symbol} ML decision: {'ALLOW' if allow_trade else 'SUPPRESS'} "
                 f"({elapsed_ms:.2f}ms)")
        
        return output
    
    def _make_trade_decision(
        self,
        regime: RegimePrediction,
        signal: SignalPrediction,
        qse_output: Optional[Dict]
    ) -> tuple[bool, List[str]]:
        """
        Make final trade decision based on regime and signal.
        
        Args:
            regime: Regime prediction
            signal: Signal prediction
            qse_output: QSE output
        
        Returns:
            (allow_trade, decision_reasons)
        """
        allow_trade = True
        reasons = []
        
        # Check 1: QSE must pass
        if qse_output:
            qse_valid = qse_output.get('validation', {}).get('signal_validity_flag', False)
            if not qse_valid:
                allow_trade = False
                reasons.append("QSE validation failed")
                return allow_trade, reasons
        
        # Check 2: Regime must be allowed
        if regime.regime_label.value not in self.config.signal_filter.allowed_regimes:
            allow_trade = False
            reasons.append(f"Regime not allowed: {regime.regime_label.value}")
        
        # Check 3: Regime confidence must be sufficient
        if regime.regime_confidence < self.config.regime.min_confidence:
            allow_trade = False
            reasons.append(f"Low regime confidence: {regime.regime_confidence:.3f}")
        
        # Check 4: Signal probability must exceed entry threshold
        if not signal.should_enter:
            allow_trade = False
            reasons.append(f"Signal prob below entry threshold: {signal.momentum_success_prob:.3f} < {self.config.signal_filter.entry_threshold}")
        
        # Check 5: Regime should be stable (warning, not blocking)
        if not regime.regime_stable:
            reasons.append("Warning: Regime not stable (recent change)")
        
        # Success reasons
        if allow_trade:
            reasons.append(f"Regime: {regime.regime_label.value} (conf: {regime.regime_confidence:.3f})")
            reasons.append(f"Signal prob: {signal.momentum_success_prob:.3f}")
            reasons.append(f"Confidence: {signal.confidence_level}")
        
        return allow_trade, reasons
    
    def _create_insufficient_data_output(
        self,
        symbol: str,
        timeframe: str,
        bar_index: int,
        start_time: float
    ) -> MLOutput:
        """Create output for insufficient data case"""
        
        regime_pred = RegimePrediction(
            regime_label=RegimeLabel.UNKNOWN,
            regime_confidence=0.0,
            prob_trending=0.33,
            prob_ranging=0.33,
            prob_stressed=0.33,
            efficiency_ratio=0.5,
            volatility_percentile=50.0,
            correlation_regime="UNKNOWN",
            regime_stable=False
        )
        
        signal_pred = SignalPrediction(
            momentum_success_prob=0.50,
            should_enter=False,
            should_exit=False,
            confidence_level="LOW",
            top_features={}
        )
        
        ml_prediction = MLPrediction(
            regime=regime_pred,
            signal=signal_pred,
            allow_trade=False,
            decision_reasons=["Insufficient data for prediction"]
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return MLOutput(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            bar_index=bar_index,
            prediction=ml_prediction,
            regime_model=self.regime_classifier.get_metadata(),
            signal_model=self.signal_filter.get_metadata(),
            config_hash=self.config_hash,
            ml_version="1.0.0",
            processing_time_ms=elapsed_ms
        )
    
    def batch_predict(
        self,
        symbols: List[str],
        timeframe: str,
        feature_dfs: Dict[str, pd.DataFrame],
        qse_outputs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, MLOutput]:
        """
        Batch prediction for multiple symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe
            feature_dfs: Dictionary mapping symbol to feature DataFrame
            qse_outputs: Dictionary mapping symbol to QSE output
        
        Returns:
            Dictionary mapping symbol to MLOutput
        """
        results = {}
        
        for symbol in symbols:
            if symbol not in feature_dfs:
                LOG.warning(f"No features for {symbol}, skipping")
                continue
            
            qse_out = qse_outputs.get(symbol) if qse_outputs else None
            
            results[symbol] = self.predict(
                symbol=symbol,
                timeframe=timeframe,
                feature_df=feature_dfs[symbol],
                qse_output=qse_out
            )
        
        return results
    
    def reset(self):
        """Reset engine state (useful for new session)"""
        self.regime_classifier.reset_smoothing()
        LOG.info("ML Inference Engine reset")
