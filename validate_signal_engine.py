"""
Signal Engine Integration Validator

Validates that the Signal Engine integrates correctly with existing system layers.
Tests import chain, data flow, and API availability.
"""

import sys
from datetime import datetime
import numpy as np


def test_imports():
    """Test all Signal Engine imports"""
    print("=" * 80)
    print("SIGNAL ENGINE INTEGRATION VALIDATION")
    print("=" * 80)
    print()
    
    print("Testing imports...")
    
    try:
        # Core engine
        from arbitrex.signal_engine import SignalGenerationEngine
        print("✓ SignalGenerationEngine imported")
        
        # Configuration
        from arbitrex.signal_engine import SignalEngineConfig
        print("✓ SignalEngineConfig imported")
        
        # Schemas
        from arbitrex.signal_engine import (
            TradeIntent,
            TradeDirection,
            SignalState,
            SignalDecision,
            SignalEngineOutput
        )
        print("✓ Schemas imported")
        
        # Filters
        from arbitrex.signal_engine import (
            RegimeGate,
            QuantStatsGate,
            MLConfidenceGate
        )
        print("✓ Filters imported")
        
        # State manager
        from arbitrex.signal_engine import SignalStateManager
        print("✓ SignalStateManager imported")
        
        print()
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_upstream_integration():
    """Test integration with upstream layers"""
    print("Testing upstream layer integration...")
    
    try:
        # Feature Engine
        from arbitrex.feature_engine.schemas import FeatureVector
        print("✓ Feature Engine schemas accessible")
        
        # Quant Stats Engine
        from arbitrex.quant_stats.schemas import (
            QuantStatsOutput,
            StatisticalMetrics,
            SignalValidation,
            RegimeState
        )
        print("✓ Quant Stats Engine schemas accessible")
        
        # ML Layer
        from arbitrex.ml_layer.schemas import (
            MLOutput,
            MLPrediction,
            RegimePrediction,
            SignalPrediction,
            ModelMetadata,
            RegimeLabel
        )
        print("✓ ML Layer schemas accessible")
        
        print()
        return True
        
    except ImportError as e:
        print(f"✗ Upstream integration error: {e}")
        return False


def test_engine_initialization():
    """Test engine initialization"""
    print("Testing engine initialization...")
    
    try:
        from arbitrex.signal_engine import SignalGenerationEngine, SignalEngineConfig
        
        # Default config
        engine1 = SignalGenerationEngine()
        print(f"✓ Engine initialized with default config")
        print(f"  Config hash: {engine1.config_hash}")
        
        # Custom config
        config = SignalEngineConfig()
        config.quant_gate.min_trend_consistency = 0.6
        engine2 = SignalGenerationEngine(config)
        print(f"✓ Engine initialized with custom config")
        print(f"  Config hash: {engine2.config_hash}")
        
        # Verify different hashes
        if engine1.config_hash != engine2.config_hash:
            print("✓ Config hashing working (different configs → different hashes)")
        else:
            print("⚠ Config hashing may need review")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False


def test_basic_processing():
    """Test basic bar processing"""
    print("Testing basic bar processing...")
    
    try:
        from arbitrex.signal_engine import SignalGenerationEngine
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
        
        # Initialize engine
        engine = SignalGenerationEngine()
        
        # Create sample inputs
        timestamp = datetime.utcnow()
        
        fv = FeatureVector(
            timestamp_utc=timestamp,
            symbol="EURUSD",
            timeframe="H1",
            feature_values=np.random.randn(20),
            feature_names=[f"feature_{i}" for i in range(20)],
            feature_version="test_v1",
            is_ml_ready=True
        )
        
        metrics = StatisticalMetrics(
            autocorr_lag1=0.3,
            trend_persistence_score=0.6,
            adf_stationary=True,
            volatility_regime="NORMAL",
            volatility_percentile=50.0,
            max_cross_correlation=0.5,
            current_volatility=0.01
        )
        
        validation = SignalValidation(
            signal_validity_flag=True,
            autocorr_check_passed=True,
            stationarity_check_passed=True,
            distribution_check_passed=True,
            correlation_check_passed=True,
            volatility_check_passed=True,
            trend_consistency=0.8
        )
        
        regime = RegimeState(
            trend_regime="UP",
            trend_strength=0.7,
            volatility_regime="NORMAL",
            efficiency_ratio=0.6,
            market_phase="TRENDING"
        )
        
        qse = QuantStatsOutput(
            timestamp=timestamp,
            symbol="EURUSD",
            timeframe="H1",
            metrics=metrics,
            validation=validation,
            regime=regime,
            config_hash="test_hash",
            config_version="1.0.0"
        )
        
        regime_pred = RegimePrediction(
            regime_label=RegimeLabel.TRENDING,
            regime_confidence=0.85,
            prob_trending=0.85,
            prob_ranging=0.1,
            prob_stressed=0.05,
            efficiency_ratio=0.7,
            volatility_percentile=50.0,
            correlation_regime="NORMAL",
            regime_stable=True
        )
        
        signal_pred = SignalPrediction(
            momentum_success_prob=0.7,
            should_enter=True,
            should_exit=False,
            confidence_level="HIGH",
            top_features={"feature_0": 0.3}
        )
        
        prediction = MLPrediction(
            regime=regime_pred,
            signal=signal_pred,
            allow_trade=True,
            decision_reasons=["Valid conditions"]
        )
        
        model_meta = ModelMetadata(
            model_version="test_v1",
            model_type="test",
            trained_on=datetime.utcnow().isoformat(),
            training_samples=1000,
            feature_version="test_v1",
            config_hash="test_hash"
        )
        
        ml = MLOutput(
            timestamp=timestamp,
            symbol="EURUSD",
            timeframe="H1",
            bar_index=0,
            prediction=prediction,
            regime_model=model_meta,
            signal_model=model_meta,
            config_hash="test_hash",
            ml_version="1.0.0",
            processing_time_ms=1.5
        )
        
        # Process bar
        output = engine.process_bar(fv, qse, ml, bar_index=0)
        
        print(f"✓ Bar processed successfully")
        print(f"  Timestamp: {output.timestamp}")
        print(f"  Symbol: {output.symbol}")
        print(f"  Trade allowed: {output.decision.trade_allowed}")
        
        if output.decision.trade_allowed:
            intent = output.decision.trade_intent
            print(f"  Direction: {intent.direction.name}")
            print(f"  Confidence: {intent.confidence_score:.3f}")
        else:
            print(f"  Suppression reasons: {output.decision.suppression_reasons}")
        
        print(f"  State: {output.state.state.value}")
        print(f"  Processing time: {output.processing_time_ms:.2f}ms")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_metrics():
    """Test health metrics"""
    print("Testing health metrics...")
    
    try:
        from arbitrex.signal_engine import SignalGenerationEngine
        
        engine = SignalGenerationEngine()
        health = engine.get_health()
        
        print(f"✓ Health metrics retrieved")
        print(f"  Total bars processed: {health.total_bars_processed}")
        print(f"  Signals generated: {health.signals_generated}")
        print(f"  Active signals: {health.active_signals}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Health metrics error: {e}")
        return False


def test_api_availability():
    """Test if API module is importable"""
    print("Testing API availability...")
    
    try:
        from arbitrex.signal_engine.api import app
        print(f"✓ API module importable")
        print(f"  Title: {app.title}")
        print(f"  Version: {app.version}")
        print(f"  Start with: python start_signal_api.py")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ API availability error: {e}")
        return False


def run_validation():
    """Run all validation tests"""
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Upstream Integration", test_upstream_integration()))
    results.append(("Engine Initialization", test_engine_initialization()))
    results.append(("Basic Processing", test_basic_processing()))
    results.append(("Health Metrics", test_health_metrics()))
    results.append(("API Availability", test_api_availability()))
    
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if failed == 0:
        print()
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print()
        print("Signal Engine is ready for integration:")
        print("  1. Start API: python start_signal_api.py")
        print("  2. Run tests: pytest test_signal_engine.py -v")
        print("  3. Run demo: python demo_signal_engine.py")
        print("  4. Check docs: SIGNAL_ENGINE.md")
        return True
    else:
        print()
        print("⚠ Some validation tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
