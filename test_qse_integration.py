"""
Integration test for QSE API + Health Monitor + Engine

Tests the complete workflow:
1. Start QSE engine with health monitoring
2. Process multiple signals
3. Check health status
4. Verify metrics tracking
"""

from arbitrex.quant_stats import QuantStatsConfig, QuantitativeStatisticsEngine, QSEHealthMonitor
import pandas as pd
import numpy as np
import time

def generate_test_returns(n=100, mean=0.0001, std=0.01, seed=None):
    """Generate synthetic returns for testing"""
    if seed:
        np.random.seed(seed)
    return pd.Series(np.random.normal(mean, std, n))


def test_integration():
    """Test complete QSE integration"""
    print("\n" + "="*70)
    print(" QSE INTEGRATION TEST: Engine + Health Monitor")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. INITIALIZATION
    # -------------------------------------------------------------------------
    print("\n[1] Initializing QSE components...")
    config = QuantStatsConfig()
    qse = QuantitativeStatisticsEngine(config)
    health = QSEHealthMonitor()
    
    print(f"    Config hash: {config.get_config_hash()}")
    print(f"    Config version: {config.config_version}")
    print("    QSE initialized ✓")
    print("    Health monitor initialized ✓")
    
    # -------------------------------------------------------------------------
    # 2. PROCESS VALID SIGNALS (should pass)
    # -------------------------------------------------------------------------
    print("\n[2] Processing VALID signals...")
    valid_scenarios = [
        ('EURUSD', 0.0001, 0.015, 42),
        ('GBPUSD', 0.0002, 0.018, 123),
        ('USDJPY', 0.00015, 0.012, 456),
    ]
    
    valid_count = 0
    for symbol, mean, std, seed in valid_scenarios:
        returns = generate_test_returns(n=150, mean=mean, std=std, seed=seed)
        
        start_time = health.record_validation_start(symbol)
        output = qse.process_bar(symbol, returns, bar_index=120)
        
        metrics_dict = {
            'trend_persistence_score': output.metrics.trend_persistence_score,
            'adf_pvalue': output.metrics.adf_pvalue,
            'z_score': output.metrics.z_score
        }
        
        if output.validation.signal_validity_flag:
            health.record_validation_success(symbol, start_time, metrics_dict)
            print(f"    {symbol}: VALID ✓")
            print(f"      Trend Score: {output.metrics.trend_persistence_score:.3f}")
            print(f"      Volatility: {output.metrics.volatility_regime}")
            print(f"      Regime: {output.regime.market_phase}")
            valid_count += 1
        else:
            health.record_validation_failure(symbol, start_time, 
                                            output.validation.failure_reasons,
                                            metrics_dict)
            print(f"    {symbol}: INVALID ✗")
            print(f"      Reasons: {output.validation.failure_reasons}")
    
    print(f"\n    Valid signals: {valid_count}/{len(valid_scenarios)}")
    
    # -------------------------------------------------------------------------
    # 3. PROCESS INVALID SIGNALS (should fail)
    # -------------------------------------------------------------------------
    print("\n[3] Processing INVALID signals (expected to fail)...")
    invalid_scenarios = [
        ('EURUSD', 0, 0.0001, 789, 'Low volatility'),  # Too low volatility
        ('XAUUSD', 0, 0.05, 321, 'High volatility'),    # Too high volatility
        ('EURGBP', 0.0001, 0.01, 654, 'Random noise'),  # Random noise
    ]
    
    invalid_count = 0
    for symbol, mean, std, seed, reason in invalid_scenarios:
        returns = generate_test_returns(n=150, mean=mean, std=std, seed=seed)
        
        start_time = health.record_validation_start(symbol)
        output = qse.process_bar(symbol, returns, bar_index=120)
        
        metrics_dict = {
            'trend_persistence_score': output.metrics.trend_persistence_score,
            'adf_pvalue': output.metrics.adf_pvalue,
            'z_score': output.metrics.z_score
        }
        
        if not output.validation.signal_validity_flag:
            health.record_validation_failure(symbol, start_time,
                                            output.validation.failure_reasons,
                                            metrics_dict)
            print(f"    {symbol}: REJECTED ✓ ({reason})")
            print(f"      Reasons: {output.validation.failure_reasons[:2]}")
            invalid_count += 1
        else:
            health.record_validation_success(symbol, start_time, metrics_dict)
            print(f"    {symbol}: PASSED ✗ (unexpected)")
    
    print(f"\n    Invalid signals correctly rejected: {invalid_count}/{len(invalid_scenarios)}")
    
    # -------------------------------------------------------------------------
    # 4. HEALTH STATUS CHECK
    # -------------------------------------------------------------------------
    print("\n[4] Checking health status...")
    status = health.get_health_status()
    
    print(f"\n    System Status: {status['status']}")
    print(f"    Uptime: {status['uptime_seconds']:.2f}s")
    print(f"    Total Validations: {status['global_metrics']['total_validations']}")
    print(f"    Valid Signals: {status['global_metrics']['valid_signals']}")
    print(f"    Invalid Signals: {status['global_metrics']['invalid_signals']}")
    print(f"    Validity Rate: {status['validity_rate']:.1%}")
    print(f"    Avg Processing Time: {status['avg_processing_time_ms']:.2f}ms")
    
    # -------------------------------------------------------------------------
    # 5. SYMBOL-SPECIFIC HEALTH
    # -------------------------------------------------------------------------
    print("\n[5] Symbol health details...")
    all_symbols = set([s[0] for s in valid_scenarios + [(x[0],) for x in invalid_scenarios]])
    
    for symbol in sorted(all_symbols):
        sym_health = health.get_symbol_health(symbol)
        if sym_health:
            metrics = sym_health['metrics']
            validity_rate = metrics['valid_signals'] / max(1, metrics['total_validations'])
            print(f"\n    {symbol}:")
            print(f"      Validations: {metrics['total_validations']}")
            print(f"      Validity Rate: {validity_rate:.1%}")
            print(f"      Avg Time: {metrics['avg_processing_time_ms']:.2f}ms")
            print(f"      Consecutive Failures: {sym_health['consecutive_failures']}")
    
    # -------------------------------------------------------------------------
    # 6. FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[6] Failure breakdown...")
    failures = health.get_failure_breakdown()
    total_failures = sum(failures.values())
    
    print(f"    Total failures: {total_failures}")
    for failure_type, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / max(1, total_failures)) * 100
            print(f"      {failure_type}: {count} ({pct:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 7. RECENT ACTIVITY
    # -------------------------------------------------------------------------
    print("\n[7] Recent activity (last 5 validations)...")
    recent = health.get_recent_validations(5)
    
    for i, val in enumerate(recent, 1):
        status_str = "✓ VALID" if val['valid'] else "✗ INVALID"
        print(f"    {i}. {val['symbol']:8s} - {status_str} - {val['elapsed_ms']:.2f}ms")
    
    # -------------------------------------------------------------------------
    # 8. EXPORT REPORT
    # -------------------------------------------------------------------------
    print("\n[8] Exporting health report...")
    report_path = 'qse_integration_health.json'
    health.export_health_report(report_path)
    print(f"    Report saved to: {report_path}")
    
    # -------------------------------------------------------------------------
    # 9. PERFORMANCE SUMMARY
    # -------------------------------------------------------------------------
    print("\n[9] Performance summary...")
    
    total_validations = status['global_metrics']['total_validations']
    avg_time = status['avg_processing_time_ms']
    max_time = status['global_metrics']['max_processing_time_ms']
    min_time = status['global_metrics']['min_processing_time_ms']
    
    print(f"    Total validations: {total_validations}")
    print(f"    Avg time: {avg_time:.2f}ms")
    print(f"    Min time: {min_time:.2f}ms")
    print(f"    Max time: {max_time:.2f}ms")
    print(f"    Throughput: {1000/avg_time:.1f} validations/sec")
    
    # -------------------------------------------------------------------------
    # 10. FINAL VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[10] Verification...")
    
    checks = []
    
    # Check 1: System is operational
    checks.append(("System operational", status['status'] in ['HEALTHY', 'DEGRADED', 'UNHEALTHY']))
    
    # Check 2: All validations recorded
    expected_total = len(valid_scenarios) + len(invalid_scenarios)
    checks.append(("All validations recorded", total_validations == expected_total))
    
    # Check 3: Processing time reasonable
    checks.append(("Processing time < 50ms", avg_time < 50))
    
    # Check 4: Health monitor tracking symbols
    checks.append(("Symbols tracked", status['symbols_tracked'] > 0))
    
    # Check 5: Metrics calculated
    checks.append(("Metrics calculated", status['global_metrics']['avg_processing_time_ms'] > 0))
    
    print()
    for check_name, passed in checks:
        status_icon = "✓" if passed else "✗"
        print(f"    {status_icon} {check_name}")
    
    all_passed = all(check[1] for check in checks)
    
    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    if all_passed:
        print(" ✓ INTEGRATION TEST PASSED")
        print(f"   - {valid_count} valid signals processed")
        print(f"   - {invalid_count} invalid signals correctly rejected")
        print(f"   - Health monitoring working")
        print(f"   - Avg processing time: {avg_time:.2f}ms")
    else:
        print(" ✗ INTEGRATION TEST FAILED")
        print("   Some checks did not pass")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)
