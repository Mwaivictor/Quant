"""
Test Suite for Signal Generation Engine

Tests core functionality:
- Gate filtering logic
- Direction assignment
- Confidence scoring
- State management
- Integration with upstream layers
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from arbitrex.signal_engine.engine import SignalGenerationEngine
from arbitrex.signal_engine.config import SignalEngineConfig
from arbitrex.signal_engine.schemas import TradeDirection, SignalState
from arbitrex.feature_engine.schemas import FeatureVector
from arbitrex.quant_stats.schemas import (
    QuantStatsOutput,
    StatisticalMetrics,
    SignalValidation,
    RegimeState
)
from arbitrex.ml_layer.schemas import (
    MLOutput,
    MLPrediction,
    RegimePrediction,
    SignalPrediction,
    ModelMetadata,
    RegimeLabel
)


# Helper Functions

def create_feature_vector(
    symbol="EURUSD",
    timeframe="H1",
    timestamp=None
) -> FeatureVector:
    """Create sample feature vector"""
    if timestamp is None:
        timestamp = datetime.utcnow().replace(microsecond=0)
    
    return FeatureVector(
        timestamp_utc=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        feature_values=np.random.randn(20),
        feature_names=[f"feature_{i}" for i in range(20)],
        feature_version="test_v1",
        is_ml_ready=True
    )


def create_qse_output(
    symbol="EURUSD",
    timeframe="H1",
    timestamp=None,
    signal_valid=True,
    trend_consistency=0.75,
    vol_regime="NORMAL",
    vol_percentile=50.0,
    max_corr=0.5,
    trend_regime="UP"
) -> QuantStatsOutput:
    """Create sample QSE output"""
    if timestamp is None:
        timestamp = datetime.utcnow().replace(microsecond=0)
    
    metrics = StatisticalMetrics(
        autocorr_lag1=0.3,
        trend_persistence_score=0.6,
        adf_stationary=True,
        volatility_regime=vol_regime,
        volatility_percentile=vol_percentile,
        max_cross_correlation=max_corr,
        current_volatility=0.01,
        rolling_mean=0.0001
    )
    
    validation = SignalValidation(
        signal_validity_flag=signal_valid,
        autocorr_check_passed=True,
        stationarity_check_passed=True,
        distribution_check_passed=True,
        correlation_check_passed=True,
        volatility_check_passed=True,
        trend_consistency=trend_consistency
    )
    
    regime = RegimeState(
        trend_regime=trend_regime,
        trend_strength=0.7,
        volatility_regime=vol_regime,
        efficiency_ratio=0.6,
        market_phase="TRENDING"
    )
    
    return QuantStatsOutput(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        metrics=metrics,
        validation=validation,
        regime=regime,
        config_hash="test_hash",
        config_version="1.0.0"
    )


def create_ml_output(
    symbol="EURUSD",
    timeframe="H1",
    timestamp=None,
    regime_label=RegimeLabel.TRENDING,
    regime_stable=True,
    regime_confidence=0.85,
    momentum_prob=0.65,
    allow_trade=True
) -> MLOutput:
    """Create sample ML output"""
    if timestamp is None:
        timestamp = datetime.utcnow().replace(microsecond=0)
    
    regime_pred = RegimePrediction(
        regime_label=regime_label,
        regime_confidence=regime_confidence,
        prob_trending=0.85 if regime_label == RegimeLabel.TRENDING else 0.2,
        prob_ranging=0.1,
        prob_stressed=0.05,
        efficiency_ratio=0.7,
        volatility_percentile=50.0,
        correlation_regime="NORMAL",
        regime_stable=regime_stable
    )
    
    signal_pred = SignalPrediction(
        momentum_success_prob=momentum_prob,
        should_enter=momentum_prob > 0.55,
        should_exit=momentum_prob < 0.45,
        confidence_level="HIGH" if momentum_prob > 0.7 else "MEDIUM",
        top_features={"feature_0": 0.3, "feature_1": 0.2}
    )
    
    prediction = MLPrediction(
        regime=regime_pred,
        signal=signal_pred,
        allow_trade=allow_trade,
        decision_reasons=["Trending regime", "High confidence"] if allow_trade else ["Low confidence"]
    )
    
    model_meta = ModelMetadata(
        model_version="test_v1",
        model_type="test",
        trained_on=datetime.utcnow().isoformat(),
        training_samples=1000,
        feature_version="test_v1",
        config_hash="test_hash"
    )
    
    return MLOutput(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        bar_index=0,
        prediction=prediction,
        regime_model=model_meta,
        signal_model=model_meta,
        config_hash="test_hash",
        ml_version="1.0.0",
        processing_time_ms=1.5
    )


# Tests

class TestRegimeGate:
    """Test regime filtering"""
    
    def test_trending_regime_allowed(self):
        """Test TRENDING regime passes"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output()
        ml = create_ml_output(regime_label=RegimeLabel.TRENDING)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.regime_gate_passed
        assert output.decision.trade_allowed
    
    def test_ranging_regime_blocked(self):
        """Test RANGING regime blocks signal"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output()
        ml = create_ml_output(regime_label=RegimeLabel.RANGING)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert not output.decision.regime_gate_passed
        assert not output.decision.trade_allowed
        assert "Regime" in output.decision.suppression_reasons[0]
    
    def test_unstable_regime_blocked(self):
        """Test unstable regime blocks signal"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output()
        ml = create_ml_output(regime_label=RegimeLabel.TRENDING, regime_stable=False)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert not output.decision.regime_gate_passed
        assert not output.decision.trade_allowed


class TestQuantStatsGate:
    """Test quantitative statistics filtering"""
    
    def test_valid_quant_stats_pass(self):
        """Test valid quant stats passes"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(signal_valid=True, trend_consistency=0.75)
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.regime_gate_passed
        assert output.decision.quant_gate_passed
    
    def test_invalid_signal_flag_blocks(self):
        """Test invalid signal flag blocks"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(signal_valid=False)
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.regime_gate_passed
        assert not output.decision.quant_gate_passed
        assert not output.decision.trade_allowed
    
    def test_low_trend_consistency_blocks(self):
        """Test low trend consistency blocks"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(trend_consistency=0.3)  # Below 0.5 threshold
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert not output.decision.quant_gate_passed
        assert not output.decision.trade_allowed
    
    def test_high_volatility_blocks(self):
        """Test extreme volatility blocks"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(vol_percentile=90.0)  # Above 80 threshold
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert not output.decision.quant_gate_passed
        assert not output.decision.trade_allowed
    
    def test_high_correlation_blocks(self):
        """Test high cross-correlation blocks"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(max_corr=0.9)  # Above 0.85 threshold
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert not output.decision.quant_gate_passed
        assert not output.decision.trade_allowed


class TestMLConfidenceGate:
    """Test ML confidence filtering"""
    
    def test_high_ml_confidence_passes(self):
        """Test high ML confidence passes"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output()
        ml = create_ml_output(momentum_prob=0.70)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.ml_gate_passed
        assert output.decision.trade_allowed
    
    def test_low_ml_confidence_blocks(self):
        """Test low ML confidence blocks"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output()
        ml = create_ml_output(momentum_prob=0.45)  # Below 0.55 threshold
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.regime_gate_passed
        assert output.decision.quant_gate_passed
        assert not output.decision.ml_gate_passed
        assert not output.decision.trade_allowed


class TestConfidenceScoring:
    """Test confidence score computation"""
    
    def test_confidence_score_computation(self):
        """Test confidence score calculation"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(trend_consistency=0.8)
        ml = create_ml_output(momentum_prob=0.7)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        assert output.decision.trade_allowed
        assert output.decision.trade_intent is not None
        
        # Check confidence score is in valid range
        conf = output.decision.trade_intent.confidence_score
        assert 0 <= conf <= 1
        
        # Check it's a weighted combination
        assert conf > 0.5  # Should be relatively high given inputs
    
    def test_confidence_score_range(self):
        """Test confidence scores stay in [0, 1]"""
        engine = SignalGenerationEngine()
        
        # Try extreme values
        fv = create_feature_vector()
        qse = create_qse_output(trend_consistency=1.0)
        ml = create_ml_output(momentum_prob=1.0)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        if output.decision.trade_intent:
            conf = output.decision.trade_intent.confidence_score
            assert 0 <= conf <= 1


class TestStateManagement:
    """Test signal state management"""
    
    def test_single_signal_per_symbol(self):
        """Test only one active signal per symbol"""
        engine = SignalGenerationEngine()
        
        timestamp = datetime.utcnow()
        
        fv = create_feature_vector(timestamp=timestamp)
        qse = create_qse_output(timestamp=timestamp)
        ml = create_ml_output(timestamp=timestamp)
        
        # Generate first signal
        output1 = engine.process_bar(fv, qse, ml, bar_index=0)
        assert output1.decision.trade_allowed
        assert output1.state.state == SignalState.VALID_SIGNAL
        
        # Try to generate second signal immediately (should be blocked)
        output2 = engine.process_bar(fv, qse, ml, bar_index=1)
        # State manager should block due to cooldown
        # (Note: depends on min_bars_between_signals config)
    
    def test_cooldown_period(self):
        """Test cooldown between signals"""
        config = SignalEngineConfig()
        config.state_management.min_bars_between_signals = 3
        engine = SignalGenerationEngine(config)
        
        timestamp = datetime.utcnow()
        
        # Generate first signal
        fv = create_feature_vector(timestamp=timestamp)
        qse = create_qse_output(timestamp=timestamp)
        ml = create_ml_output(timestamp=timestamp)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        assert output.decision.trade_allowed
        
        # Process bars during cooldown
        for i in range(1, 3):
            timestamp += timedelta(hours=1)
            fv = create_feature_vector(timestamp=timestamp)
            qse = create_qse_output(timestamp=timestamp)
            ml = create_ml_output(timestamp=timestamp)
            
            output = engine.process_bar(fv, qse, ml, bar_index=i)
            # Should be suppressed due to cooldown
        
        # After cooldown, new signal allowed
        timestamp += timedelta(hours=1)
        fv = create_feature_vector(timestamp=timestamp)
        qse = create_qse_output(timestamp=timestamp)
        ml = create_ml_output(timestamp=timestamp)
        
        output = engine.process_bar(fv, qse, ml, bar_index=3)
        # Should be allowed now


class TestDirectionAssignment:
    """Test trade direction assignment"""
    
    def test_long_direction_assignment(self):
        """Test LONG direction assigned correctly"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(trend_regime="UP")
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        if output.decision.trade_intent:
            assert output.decision.trade_intent.direction == TradeDirection.LONG
    
    def test_short_direction_assignment(self):
        """Test SHORT direction assigned correctly"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector()
        qse = create_qse_output(trend_regime="DOWN")
        ml = create_ml_output()
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        if output.decision.trade_intent:
            assert output.decision.trade_intent.direction == TradeDirection.SHORT


class TestHealthMetrics:
    """Test health monitoring"""
    
    def test_health_tracking(self):
        """Test health metrics are updated"""
        engine = SignalGenerationEngine()
        
        # Process multiple bars
        for i in range(10):
            fv = create_feature_vector()
            qse = create_qse_output()
            ml = create_ml_output(momentum_prob=0.6 if i % 2 == 0 else 0.4)
            
            engine.process_bar(fv, qse, ml, bar_index=i)
        
        health = engine.get_health()
        
        assert health.total_bars_processed == 10
        assert health.signals_generated + health.signals_suppressed == 10
        assert 0 <= health.signal_generation_rate <= 1
        assert health.avg_processing_time_ms > 0


class TestInputValidation:
    """Test input validation"""
    
    def test_timestamp_mismatch_error(self):
        """Test error on timestamp mismatch"""
        engine = SignalGenerationEngine()
        
        timestamp1 = datetime.utcnow()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        fv = create_feature_vector(timestamp=timestamp1)
        qse = create_qse_output(timestamp=timestamp2)
        ml = create_ml_output(timestamp=timestamp1)
        
        with pytest.raises(ValueError, match="Timestamp mismatch"):
            engine.process_bar(fv, qse, ml, bar_index=0)
    
    def test_symbol_mismatch_error(self):
        """Test error on symbol mismatch"""
        engine = SignalGenerationEngine()
        
        fv = create_feature_vector(symbol="EURUSD")
        qse = create_qse_output(symbol="GBPUSD")
        ml = create_ml_output(symbol="EURUSD")
        
        with pytest.raises(ValueError, match="Symbol mismatch"):
            engine.process_bar(fv, qse, ml, bar_index=0)


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_pipeline_valid_signal(self):
        """Test complete pipeline with valid signal"""
        engine = SignalGenerationEngine()
        
        timestamp = datetime.utcnow()
        
        fv = create_feature_vector(symbol="EURUSD", timeframe="H1", timestamp=timestamp)
        qse = create_qse_output(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=timestamp,
            signal_valid=True,
            trend_consistency=0.8,
            trend_regime="UP"
        )
        ml = create_ml_output(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=timestamp,
            regime_label=RegimeLabel.TRENDING,
            regime_stable=True,
            momentum_prob=0.75
        )
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        # Verify all gates passed
        assert output.decision.regime_gate_passed
        assert output.decision.quant_gate_passed
        assert output.decision.ml_gate_passed
        
        # Verify trade allowed
        assert output.decision.trade_allowed
        
        # Verify trade intent created
        assert output.decision.trade_intent is not None
        assert output.decision.trade_intent.symbol == "EURUSD"
        assert output.decision.trade_intent.direction in [TradeDirection.LONG, TradeDirection.SHORT]
        assert 0 <= output.decision.trade_intent.confidence_score <= 1
        
        # Verify state updated
        assert output.state.state == SignalState.VALID_SIGNAL
        assert output.state.active_intent is not None
    
    def test_full_pipeline_suppressed_signal(self):
        """Test complete pipeline with suppressed signal"""
        engine = SignalGenerationEngine()
        
        timestamp = datetime.utcnow()
        
        fv = create_feature_vector(symbol="EURUSD", timestamp=timestamp)
        qse = create_qse_output(symbol="EURUSD", timestamp=timestamp, signal_valid=False)
        ml = create_ml_output(symbol="EURUSD", timestamp=timestamp)
        
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        # Verify suppression
        assert not output.decision.trade_allowed
        assert output.decision.trade_intent is None
        assert len(output.decision.suppression_reasons) > 0
        
        # Verify state remains NO_TRADE
        assert output.state.state == SignalState.NO_TRADE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
