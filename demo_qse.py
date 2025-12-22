"""
Quantitative Statistics Engine Demo

Demonstrates the 5-gate validation system of the QSE.
Shows how QSE sits between Feature Engine and ML layer.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arbitrex.quant_stats import (
    QuantStatsConfig,
    QuantitativeStatisticsEngine,
    VolatilityRegime
)
from arbitrex.feature_engine import FeaturePipeline, FeatureEngineConfig


def generate_test_data(n_bars: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 10, n_bars)  # Upward trend
    noise = np.random.normal(0, 2, n_bars)  # Random noise
    close = base_price + trend + noise
    
    # Generate OHLC from close
    high = close + np.random.uniform(0.5, 2.0, n_bars)
    low = close - np.random.uniform(0.5, 2.0, n_bars)
    open_price = close + np.random.uniform(-1, 1, n_bars)
    volume = np.random.uniform(1000, 10000, n_bars)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.index = pd.date_range('2024-01-01', periods=n_bars, freq='1H')
    
    return df


def demo_qse_standalone():
    """Demo QSE with standalone return series."""
    print("=" * 80)
    print("DEMO 1: QSE Standalone Analysis")
    print("=" * 80)
    
    # Generate test data
    df = generate_test_data(n_bars=200)
    
    # Compute returns
    returns = df['close'].pct_change().fillna(0)
    
    print(f"\nGenerated {len(df)} bars of synthetic data")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Average return: {returns.mean()*100:.4f}%")
    print(f"Return volatility: {returns.std()*100:.4f}%")
    
    # Initialize QSE
    config = QuantStatsConfig()
    qse = QuantitativeStatisticsEngine(config)
    
    print(f"\nQSE initialized with config: {config.get_config_hash()}")
    
    # Process last 50 bars
    print("\n" + "-" * 80)
    print("Processing last 50 bars...")
    print("-" * 80)
    
    results = []
    for bar_idx in range(150, 200):
        output = qse.process_bar(
            symbol="TEST",
            returns=returns,
            bar_index=bar_idx
        )
        results.append(output)
    
    # Print summary
    summary = qse.get_validation_summary(results)
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total bars processed: {summary['total_bars']}")
    print(f"Valid signals: {summary['valid_signals']}")
    print(f"Invalid signals: {summary['invalid_signals']}")
    print(f"Validity rate: {summary['validity_rate']*100:.2f}%")
    print(f"Avg trend persistence: {summary['avg_trend_persistence']:.4f}")
    print(f"Avg |z-score|: {summary['avg_abs_z_score']:.4f}")
    
    if summary['failure_breakdown']:
        print("\nFailure Breakdown:")
        for reason, count in summary['failure_breakdown'].items():
            print(f"  - {reason}: {count}")
    
    # Show detailed metrics for last bar
    last_result = results[-1]
    print("\n" + "=" * 80)
    print(f"DETAILED METRICS - Bar {last_result.bar_index}")
    print("=" * 80)
    
    print("\n1. AUTOCORRELATION:")
    print(f"   Lag 1:  {last_result.metrics.autocorr_lag1:.4f}")
    print(f"   Lag 5:  {last_result.metrics.autocorr_lag5:.4f}")
    print(f"   Lag 10: {last_result.metrics.autocorr_lag10:.4f}")
    print(f"   Lag 20: {last_result.metrics.autocorr_lag20:.4f}")
    print(f"   Trend Persistence Score: {last_result.metrics.trend_persistence_score:.4f}")
    
    print("\n2. STATIONARITY:")
    print(f"   ADF Stationary: {last_result.metrics.adf_stationary}")
    print(f"   ADF p-value: {last_result.metrics.adf_pvalue:.6f}")
    print(f"   ADF Test Statistic: {last_result.metrics.adf_test_statistic:.4f}")
    
    print("\n3. DISTRIBUTION:")
    print(f"   Z-score: {last_result.metrics.z_score:.4f}")
    print(f"   Is Outlier: {last_result.metrics.is_outlier}")
    print(f"   Rolling Mean: {last_result.metrics.rolling_mean:.6f}")
    print(f"   Rolling Std: {last_result.metrics.rolling_std:.6f}")
    print(f"   Distribution Stable: {last_result.metrics.distribution_stable}")
    
    print("\n4. VOLATILITY:")
    print(f"   Current Volatility: {last_result.metrics.current_volatility:.6f}")
    print(f"   Volatility Percentile: {last_result.metrics.volatility_percentile:.2f}%")
    print(f"   Volatility Regime: {last_result.metrics.volatility_regime}")
    
    print("\n5. REGIME STATE:")
    print(f"   Trend Regime: {last_result.regime_state.trend_regime}")
    print(f"   Volatility Regime: {last_result.regime_state.volatility_regime}")
    print(f"   Market Phase: {last_result.regime_state.market_phase}")
    print(f"   Efficiency Ratio: {last_result.regime_state.efficiency_ratio:.4f}")
    
    print("\n6. VALIDATION:")
    print(f"   Signal Valid: {last_result.validation.signal_validity_flag}")
    print(f"   Trend Persistence Check: {last_result.validation.autocorr_check_passed}")
    print(f"   Stationarity Check: {last_result.validation.stationarity_check_passed}")
    print(f"   Distribution Check: {last_result.validation.distribution_check_passed}")
    print(f"   Volatility Regime Check: {last_result.validation.volatility_check_passed}")
    
    if last_result.validation.failure_reasons:
        print(f"\n   Failure Reasons:")
        for reason in last_result.validation.failure_reasons:
            print(f"   - {reason}")


def demo_qse_with_feature_engine():
    """Demo QSE integrated with Feature Engine."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: QSE + Feature Engine Integration")
    print("=" * 80)
    
    # Generate test data
    df = generate_test_data(n_bars=200)
    
    print(f"\nGenerated {len(df)} bars of synthetic data")
    
    # Initialize Feature Engine
    fe_config = FeatureEngineConfig(
        symbols=['TEST'],
        lookback_periods={'short': 10, 'medium': 20, 'long': 50}
    )
    feature_pipeline = FeaturePipeline(fe_config)
    
    print("Feature Pipeline initialized")
    
    # Compute features for last bar
    bar_idx = len(df) - 1
    features = feature_pipeline.compute_features('TEST', df, bar_idx)
    
    print(f"\nComputed {len(features)} features:")
    for key, value in features.items():
        print(f"  {key}: {value:.6f}")
    
    # Initialize QSE
    qse_config = QuantStatsConfig()
    qse = QuantitativeStatisticsEngine(qse_config)
    
    print("\nQSE initialized")
    
    # Extract returns from features
    returns = df['close'].pct_change().fillna(0)
    
    # Process bar through QSE
    qse_output = qse.process_bar(
        symbol='TEST',
        returns=returns,
        bar_index=bar_idx
    )
    
    # Integration Decision Logic
    print("\n" + "=" * 80)
    print("INTEGRATION DECISION LOGIC")
    print("=" * 80)
    
    print(f"\nFeature Vector (X_t): {len(features)} features computed")
    print(f"QSE Validation: signal_validity_flag = {qse_output.validation.signal_validity_flag}")
    
    if qse_output.validation.signal_validity_flag:
        print("\n[PASS] Features are statistically valid")
        print("  -> Forward to ML Layer for prediction")
        print("  -> Use ML output in Signal Generator")
    else:
        print("\n[FAIL] Features do not meet statistical criteria")
        print("  -> Do NOT forward to ML Layer")
        print("  -> Signal Generator uses default/fallback logic")
        print(f"\n  Failure reasons ({len(qse_output.validation.failure_reasons)}):")
        for reason in qse_output.validation.failure_reasons:
            print(f"    - {reason}")
    
    print(f"\nRegime State:")
    print(f"  Trend: {qse_output.regime_state.trend_regime}")
    print(f"  Volatility: {qse_output.regime_state.volatility_regime}")
    print(f"  Market Phase: {qse_output.regime_state.market_phase}")


def demo_multi_symbol_correlation():
    """Demo cross-pair correlation analysis."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Multi-Symbol Correlation Analysis")
    print("=" * 80)
    
    # Generate correlated return series
    np.random.seed(42)
    n_bars = 200
    
    # Base returns
    base_returns = np.random.normal(0, 0.01, n_bars)
    
    # Create correlated symbols
    symbols = {
        'EURUSD': base_returns + np.random.normal(0, 0.005, n_bars),
        'GBPUSD': 0.7 * base_returns + np.random.normal(0, 0.005, n_bars),  # High correlation
        'USDJPY': -0.5 * base_returns + np.random.normal(0, 0.008, n_bars),  # Negative correlation
        'XAUUSD': 0.2 * base_returns + np.random.normal(0, 0.015, n_bars)   # Low correlation
    }
    
    returns_dict = {sym: pd.Series(rets) for sym, rets in symbols.items()}
    
    print(f"\nGenerated {len(symbols)} correlated return series")
    
    # Initialize QSE
    config = QuantStatsConfig()
    qse = QuantitativeStatisticsEngine(config)
    
    # Process bar with correlation analysis
    bar_idx = 150
    
    output = qse.process_bar(
        symbol='EURUSD',
        returns=returns_dict['EURUSD'],
        bar_index=bar_idx,
        returns_dict=returns_dict
    )
    
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    print(f"\nAverage Correlation: {output.metrics.avg_cross_correlation:.4f}")
    print(f"Max Correlation: {output.metrics.max_cross_correlation:.4f}")
    print(f"Correlation Dispersion: {output.metrics.correlation_dispersion:.4f}")
    
    print(f"\nCorrelation Check: {output.validation.correlation_check}")
    
    if output.metrics.max_cross_correlation > config.correlation.max_correlation_threshold:
        print(f"WARNING: Correlation exceeds threshold "
              f"({output.metrics.max_cross_correlation:.4f} > "
              f"{config.correlation.max_correlation_threshold})")
        print("  -> Risk of concentration")
        print("  -> Suppress signals on highly correlated pairs")


def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print(" " * 15 + "QUANTITATIVE STATISTICS ENGINE DEMO")
    print("=" * 80)
    
    try:
        demo_qse_standalone()
        demo_qse_with_feature_engine()
        demo_multi_symbol_correlation()
        
        print("\n\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\nQSE Integration Summary:")
        print("  1. Feature Engine computes X_t (feature vector)")
        print("  2. QSE validates X_t with 5 statistical gates")
        print("  3. If valid -> forward to ML layer")
        print("  4. If invalid -> suppress signal, use fallback")
        print("  5. Regime state informs risk management")
        
        print("\n5-Gate Validation:")
        print("  [x] Autocorrelation & Trend Persistence")
        print("  [x] Stationarity (ADF Test)")
        print("  [x] Distribution Stability & Outlier Detection")
        print("  [x] Cross-Pair Correlation")
        print("  [x] Volatility Regime Filtering")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
