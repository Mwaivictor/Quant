"""
Signal Engine Demo

Demonstrates the Signal Generation Engine with synthetic data.
"""

import numpy as np
from datetime import datetime, timedelta

from arbitrex.signal_engine.engine import SignalGenerationEngine
from arbitrex.signal_engine.config import SignalEngineConfig
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


def create_synthetic_bar(
    bar_index: int,
    base_time: datetime,
    scenario: str = "valid_trending"
):
    """
    Create synthetic bar data for different scenarios.
    
    Scenarios:
        - valid_trending: All gates pass, trending regime
        - ranging_market: Regime gate blocks
        - weak_trend: Quant stats gate blocks
        - low_ml_confidence: ML gate blocks
    """
    timestamp = base_time + timedelta(hours=bar_index)
    
    # Feature Vector
    feature_vector = FeatureVector(
        timestamp_utc=timestamp,
        symbol="EURUSD",
        timeframe="H1",
        feature_values=np.random.randn(20),
        feature_names=[f"feature_{i}" for i in range(20)],
        feature_version="demo_v1",
        is_ml_ready=True
    )
    
    # Configure based on scenario
    if scenario == "valid_trending":
        regime_label = RegimeLabel.TRENDING
        regime_stable = True
        signal_valid = True
        trend_consistency = 0.8
        momentum_prob = 0.7
        trend_regime = "UP"
    elif scenario == "ranging_market":
        regime_label = RegimeLabel.RANGING
        regime_stable = True
        signal_valid = True
        trend_consistency = 0.8
        momentum_prob = 0.7
        trend_regime = "NEUTRAL"
    elif scenario == "weak_trend":
        regime_label = RegimeLabel.TRENDING
        regime_stable = True
        signal_valid = False
        trend_consistency = 0.3
        momentum_prob = 0.7
        trend_regime = "NEUTRAL"
    elif scenario == "low_ml_confidence":
        regime_label = RegimeLabel.TRENDING
        regime_stable = True
        signal_valid = True
        trend_consistency = 0.8
        momentum_prob = 0.4
        trend_regime = "UP"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # QSE Output
    metrics = StatisticalMetrics(
        autocorr_lag1=0.3,
        trend_persistence_score=0.6,
        adf_stationary=True,
        volatility_regime="NORMAL",
        volatility_percentile=50.0,
        max_cross_correlation=0.5,
        current_volatility=0.01,
        rolling_mean=0.0001 if trend_regime == "UP" else -0.0001
    )
    
    validation = SignalValidation(
        signal_validity_flag=signal_valid,
        autocorr_check_passed=signal_valid,
        stationarity_check_passed=signal_valid,
        distribution_check_passed=signal_valid,
        correlation_check_passed=signal_valid,
        volatility_check_passed=signal_valid,
        trend_consistency=trend_consistency,
        failure_reasons=[] if signal_valid else ["Low trend consistency"]
    )
    
    regime = RegimeState(
        trend_regime=trend_regime,
        trend_strength=0.7 if signal_valid else 0.3,
        volatility_regime="NORMAL",
        efficiency_ratio=0.6,
        market_phase="TRENDING" if regime_label == RegimeLabel.TRENDING else "CONSOLIDATION"
    )
    
    qse_output = QuantStatsOutput(
        timestamp=timestamp,
        symbol="EURUSD",
        timeframe="H1",
        metrics=metrics,
        validation=validation,
        regime=regime,
        config_hash="demo_hash",
        config_version="1.0.0"
    )
    
    # ML Output
    regime_pred = RegimePrediction(
        regime_label=regime_label,
        regime_confidence=0.85,
        prob_trending=0.85 if regime_label == RegimeLabel.TRENDING else 0.2,
        prob_ranging=0.8 if regime_label == RegimeLabel.RANGING else 0.1,
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
        confidence_level="HIGH" if momentum_prob > 0.7 else "MEDIUM" if momentum_prob > 0.5 else "LOW",
        top_features={"momentum_10": 0.3, "volatility_ratio": 0.2}
    )
    
    prediction = MLPrediction(
        regime=regime_pred,
        signal=signal_pred,
        allow_trade=regime_label == RegimeLabel.TRENDING and momentum_prob > 0.55,
        decision_reasons=["Valid regime", "Sufficient confidence"] if regime_label == RegimeLabel.TRENDING else ["Non-trending regime"]
    )
    
    model_meta = ModelMetadata(
        model_version="demo_v1",
        model_type="demo",
        trained_on=datetime.utcnow().isoformat(),
        training_samples=10000,
        feature_version="demo_v1",
        config_hash="demo_hash"
    )
    
    ml_output = MLOutput(
        timestamp=timestamp,
        symbol="EURUSD",
        timeframe="H1",
        bar_index=bar_index,
        prediction=prediction,
        regime_model=model_meta,
        signal_model=model_meta,
        config_hash="demo_hash",
        ml_version="1.0.0",
        processing_time_ms=1.5
    )
    
    return feature_vector, qse_output, ml_output


def run_demo():
    """Run comprehensive demo of Signal Engine"""
    
    print("=" * 80)
    print("SIGNAL GENERATION ENGINE - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize engine
    print("Initializing Signal Generation Engine...")
    config = SignalEngineConfig()
    engine = SignalGenerationEngine(config)
    print(f"✓ Engine initialized with config hash: {engine.config_hash}")
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Allowed Regimes: {config.regime_gate.allowed_regimes}")
    print(f"  Min Trend Consistency: {config.quant_gate.min_trend_consistency}")
    print(f"  ML Entry Threshold: {config.ml_gate.entry_threshold}")
    print(f"  Min Bars Between Signals: {config.state_management.min_bars_between_signals}")
    print()
    
    base_time = datetime.utcnow()
    scenarios = [
        ("valid_trending", "✓ All gates pass - Signal Generated"),
        ("ranging_market", "✗ Regime gate blocks - Ranging market"),
        ("weak_trend", "✗ Quant stats gate blocks - Weak trend"),
        ("low_ml_confidence", "✗ ML gate blocks - Low confidence"),
        ("valid_trending", "✓ Valid signal after cooldown"),
    ]
    
    print("Processing Bars:")
    print("-" * 80)
    
    for bar_idx, (scenario, description) in enumerate(scenarios):
        fv, qse, ml = create_synthetic_bar(bar_idx, base_time, scenario)
        
        output = engine.process_bar(fv, qse, ml, bar_idx)
        
        print(f"\nBar {bar_idx}: {description}")
        print(f"  Timestamp: {output.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Regime Gate: {'PASS' if output.decision.regime_gate_passed else 'FAIL'}")
        print(f"    └─ {output.decision.regime_gate_reason}")
        print(f"  Quant Stats Gate: {'PASS' if output.decision.quant_gate_passed else 'FAIL'}")
        print(f"    └─ {output.decision.quant_gate_reason}")
        print(f"  ML Confidence Gate: {'PASS' if output.decision.ml_gate_passed else 'FAIL'}")
        print(f"    └─ {output.decision.ml_gate_reason}")
        
        if output.decision.trade_allowed:
            intent = output.decision.trade_intent
            print(f"  ✓ TRADE INTENT GENERATED")
            print(f"    Direction: {intent.direction.name}")
            print(f"    Confidence: {intent.confidence_score:.3f}")
            print(f"    Signal Source: {intent.signal_source}")
        else:
            print(f"  ✗ SIGNAL SUPPRESSED")
            print(f"    Reasons: {', '.join(output.decision.suppression_reasons)}")
        
        print(f"  State: {output.state.state.value}")
        print(f"  Processing Time: {output.processing_time_ms:.2f}ms")
    
    print()
    print("=" * 80)
    
    # Display health metrics
    health = engine.get_health()
    print("\nEngine Health Metrics:")
    print("-" * 80)
    print(f"Total Bars Processed: {health.total_bars_processed}")
    print(f"Signals Generated: {health.signals_generated}")
    print(f"Signals Suppressed: {health.signals_suppressed}")
    print(f"Signal Generation Rate: {health.signal_generation_rate:.2%}")
    print()
    print(f"Gate Pass Rates:")
    print(f"  Regime Gate: {health.regime_gate_pass_rate:.2%}")
    print(f"  Quant Stats Gate: {health.quant_gate_pass_rate:.2%}")
    print(f"  ML Confidence Gate: {health.ml_gate_pass_rate:.2%}")
    print()
    print(f"Suppression Breakdown:")
    print(f"  By Regime: {health.suppression_by_regime}")
    print(f"  By Quant Stats: {health.suppression_by_quant}")
    print(f"  By ML: {health.suppression_by_ml}")
    print()
    print(f"Direction Distribution:")
    print(f"  Long Signals: {health.long_signals}")
    print(f"  Short Signals: {health.short_signals}")
    print()
    print(f"Confidence Statistics:")
    print(f"  Average: {health.avg_confidence_score:.3f}")
    print(f"  Min: {health.min_confidence_score:.3f}")
    print(f"  Max: {health.max_confidence_score:.3f}")
    print()
    print(f"Performance:")
    print(f"  Avg Processing Time: {health.avg_processing_time_ms:.2f}ms")
    print(f"  Active Signals: {health.active_signals}")
    
    print()
    print("=" * 80)
    
    # Display state summary
    state_summary = engine.get_state_summary()
    print("\nState Summary:")
    print("-" * 80)
    print(f"Total Symbols Tracked: {state_summary['total_symbols']}")
    print(f"Active Trades: {state_summary['active_trades']}")
    print(f"Valid Signals: {state_summary['valid_signals']}")
    print(f"No Trade: {state_summary['no_trade']}")
    
    if state_summary['states']:
        print("\nDetailed States:")
        for key, state in state_summary['states'].items():
            print(f"  {key}:")
            print(f"    State: {state['state']}")
            print(f"    Bars Since Change: {state['bars_since_state_change']}")
            if state['active_intent']:
                print(f"    Active Direction: {state['active_intent']['direction']}")
                print(f"    Confidence: {state['active_intent']['confidence_score']:.3f}")
    
    print()
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
