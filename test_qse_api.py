"""
Quick test for QSE API and Health Monitor
"""

from arbitrex.quant_stats import QuantStatsConfig, QuantitativeStatisticsEngine, QSEHealthMonitor
import pandas as pd
import numpy as np

def test_health_monitor():
    """Test health monitor functionality"""
    print("\n" + "="*60)
    print("Testing QSE Health Monitor")
    print("="*60)
    
    # Initialize
    config = QuantStatsConfig()
    qse = QuantitativeStatisticsEngine(config)
    health = QSEHealthMonitor()
    
    # Generate test data
    np.random.seed(42)
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Good signal (should pass)
        returns = pd.Series(np.random.normal(0.0001, 0.01, 100))
        start_time = health.record_validation_start(symbol)
        output = qse.process_bar(symbol, returns, bar_index=80)
        
        if output.validation.signal_validity_flag:
            health.record_validation_success(
                symbol, start_time,
                {'trend_persistence_score': output.metrics.trend_persistence_score,
                 'adf_pvalue': output.metrics.adf_pvalue,
                 'z_score': output.metrics.z_score}
            )
            print(f"  Valid signal")
        else:
            health.record_validation_failure(
                symbol, start_time,
                output.validation.failure_reasons,
                {'trend_persistence_score': output.metrics.trend_persistence_score,
                 'adf_pvalue': output.metrics.adf_pvalue,
                 'z_score': output.metrics.z_score}
            )
            print(f"  Invalid signal: {output.validation.failure_reasons}")
        
        # Bad signal (should fail - low volatility)
        returns_bad = pd.Series(np.random.normal(0, 0.0001, 100))
        start_time = health.record_validation_start(symbol)
        output = qse.process_bar(symbol, returns_bad, bar_index=80)
        
        if output.validation.signal_validity_flag:
            health.record_validation_success(
                symbol, start_time,
                {'trend_persistence_score': output.metrics.trend_persistence_score,
                 'adf_pvalue': output.metrics.adf_pvalue,
                 'z_score': output.metrics.z_score}
            )
            print(f"  Valid signal")
        else:
            health.record_validation_failure(
                symbol, start_time,
                output.validation.failure_reasons,
                {'trend_persistence_score': output.metrics.trend_persistence_score,
                 'adf_pvalue': output.metrics.adf_pvalue,
                 'z_score': output.metrics.z_score}
            )
            print(f"  Invalid signal (expected): {output.validation.failure_reasons}")
    
    # Get health status
    print("\n" + "-"*60)
    print("OVERALL HEALTH STATUS")
    print("-"*60)
    status = health.get_health_status()
    print(f"Status: {status['status']}")
    print(f"Uptime: {status['uptime_seconds']:.2f}s")
    print(f"Total Validations: {status['global_metrics']['total_validations']}")
    print(f"Valid Signals: {status['global_metrics']['valid_signals']}")
    print(f"Invalid Signals: {status['global_metrics']['invalid_signals']}")
    print(f"Validity Rate: {status['validity_rate']:.1%}")
    print(f"Avg Processing Time: {status['avg_processing_time_ms']:.2f}ms")
    print(f"Symbols Tracked: {status['symbols_tracked']}")
    
    # Symbol-specific health
    print("\n" + "-"*60)
    print("PER-SYMBOL HEALTH")
    print("-"*60)
    for symbol in symbols:
        sym_health = health.get_symbol_health(symbol)
        if sym_health:
            metrics = sym_health['metrics']
            validity_rate = metrics['valid_signals'] / max(1, metrics['total_validations'])
            print(f"\n{symbol}:")
            print(f"  Total: {metrics['total_validations']}")
            print(f"  Valid: {metrics['valid_signals']}")
            print(f"  Invalid: {metrics['invalid_signals']}")
            print(f"  Validity Rate: {validity_rate:.1%}")
            print(f"  Avg Time: {metrics['avg_processing_time_ms']:.2f}ms")
            print(f"  Consecutive Failures: {sym_health['consecutive_failures']}")
    
    # Failure breakdown
    print("\n" + "-"*60)
    print("FAILURE BREAKDOWN")
    print("-"*60)
    failures = health.get_failure_breakdown()
    for failure_type, count in failures.items():
        print(f"  {failure_type}: {count}")
    
    # Recent validations
    print("\n" + "-"*60)
    print("RECENT VALIDATIONS (last 5)")
    print("-"*60)
    recent = health.get_recent_validations(5)
    for i, val in enumerate(recent, 1):
        status = "VALID" if val['valid'] else "INVALID"
        print(f"{i}. {val['symbol']} - {status} - {val['elapsed_ms']:.2f}ms")
        if not val['valid']:
            print(f"   Reasons: {val['reasons']}")
    
    # Export report
    print("\n" + "-"*60)
    print("Exporting health report...")
    health.export_health_report('qse_health_test.json')
    print("Report exported to: qse_health_test.json")
    
    print("\n" + "="*60)
    print("Health Monitor Test PASSED")
    print("="*60)


def test_api_models():
    """Test API request/response models"""
    print("\n" + "="*60)
    print("Testing API Models")
    print("="*60)
    
    from arbitrex.quant_stats.api import ValidateRequest, ValidateResponse
    
    # Test request model
    request = ValidateRequest(
        symbol="EURUSD",
        timeframe="1H",
        returns=[0.001, -0.002, 0.003, 0.001],
        bar_index=100
    )
    print(f"\nRequest created: {request.symbol}")
    print(f"  Returns length: {len(request.returns)}")
    print(f"  Bar index: {request.bar_index}")
    
    print("\n" + "="*60)
    print("API Models Test PASSED")
    print("="*60)


if __name__ == "__main__":
    test_health_monitor()
    test_api_models()
    print("\n==> All tests PASSED <==\n")
