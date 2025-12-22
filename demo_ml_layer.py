"""
Quick integration demo showing Feature Engine → QSE → ML Layer
"""

from arbitrex.ml_layer import MLConfig, MLInferenceEngine
import pandas as pd
import numpy as np


def demo_ml_layer():
    """Quick demo of ML Layer"""
    print("\n" + "="*60)
    print("  ArbitreX ML Layer - Quick Demo")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing ML Layer...")
    config = MLConfig()
    ml_engine = MLInferenceEngine(config)
    print("   ✓ ML Engine ready")
    
    # Create synthetic features
    print("\n2. Generating test features...")
    np.random.seed(42)
    n = 150
    
    feature_df = pd.DataFrame({
        'momentum_score': np.random.uniform(-0.5, 0.5, n),
        'rolling_return_20': np.random.normal(0.002, 0.01, n),
        'trend_consistency': np.random.uniform(0.5, 0.8, n),
        'ma_distance_20': np.random.uniform(-0.01, 0.01, n),
        'atr': np.random.uniform(0.001, 0.003, n),
        'rolling_vol': np.random.uniform(0.01, 0.02, n),
        'vol_percentile': np.random.uniform(30, 70, n),
        'cross_pair_correlation': np.random.uniform(0.4, 0.6, n),
    })
    print(f"   ✓ Generated {len(feature_df)} bars")
    
    # Mock QSE output
    print("\n3. QSE validation...")
    qse_output = {
        'validation': {'signal_validity_flag': True},
        'metrics': {
            'trend_persistence_score': 0.28,
            'adf_stationary': True,
            'z_score': -1.5
        },
        'regime': {
            'trend_regime': 'TRENDING',
            'efficiency_ratio': 0.72
        }
    }
    print("   ✓ QSE passed validation")
    
    # ML Prediction
    print("\n4. ML Layer prediction...")
    output = ml_engine.predict(
        symbol="EURUSD",
        timeframe="4H",
        feature_df=feature_df,
        qse_output=qse_output
    )
    
    print(f"\n{'─'*60}")
    print("  PREDICTION RESULT")
    print(f"{'─'*60}")
    print(f"  Regime: {output.prediction.regime.regime_label.value}")
    print(f"  Regime Confidence: {output.prediction.regime.regime_confidence:.1%}")
    print(f"  Signal Probability: {output.prediction.signal.momentum_success_prob:.1%}")
    print(f"  Confidence Level: {output.prediction.signal.confidence_level}")
    print(f"  Trade Decision: {'✓ ALLOW' if output.prediction.allow_trade else '✗ SUPPRESS'}")
    print(f"{'─'*60}")
    
    if output.prediction.allow_trade:
        print("\n  → Proceed to Signal Generator")
    else:
        print("\n  → Signal suppressed")
        if output.prediction.decision_reasons:
            print(f"  Reason: {output.prediction.decision_reasons[0]}")
    
    print(f"\n  Processing time: {output.processing_time_ms:.2f}ms")
    print(f"  Config hash: {output.config_hash}")
    
    print("\n" + "="*60)
    print("  Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo_ml_layer()
