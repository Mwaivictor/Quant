"""
Test Feature Engine Health Monitor and API
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from arbitrex.feature_engine.health_monitor import FeatureEngineHealthMonitor


def test_health_monitor():
    """Test health monitor functionality"""
    
    print("=" * 80)
    print("TESTING FEATURE ENGINE HEALTH MONITOR")
    print("=" * 80)
    print()
    
    # Initialize monitor
    monitor = FeatureEngineHealthMonitor()
    print("✓ Health monitor initialized")
    print()
    
    # Simulate successful computations
    print("Simulating successful feature computations...")
    for i in range(5):
        symbol = ['EURUSD', 'GBPUSD', 'XAUUSD'][i % 3]
        timeframe = ['1H', '4H'][i % 2]
        
        start = monitor.record_computation_start(symbol, timeframe)
        
        # Simulate computation
        import time
        time.sleep(0.01)  # Small delay
        
        monitor.record_computation_success(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start,
            features_computed=16,
            bars_processed=200,
            feature_coverage_pct=98.5,
            normalization_applied=True
        )
    
    print(f"✓ Recorded {5} successful computations")
    print()
    
    # Simulate failures
    print("Simulating failures...")
    monitor.record_computation_failure('BTCUSD', '1H', 'Insufficient data')
    monitor.record_validation_result(False, ['Missing required columns'])
    print("✓ Recorded failures")
    print()
    
    # Simulate storage operations
    print("Simulating storage operations...")
    monitor.record_storage_write(True, "abc123")
    monitor.record_storage_read(True)
    print("✓ Recorded storage operations")
    print()
    
    # Get health status
    print("-" * 80)
    print("HEALTH STATUS:")
    print("-" * 80)
    status = monitor.get_health_status()
    print(f"Overall Status: {status['status']}")
    print(f"Success Rate: {status['success_rate_pct']:.2f}%")
    print(f"Total Computations: {status['total_computations']}")
    print()
    
    print("Metrics:")
    print(f"  Features Computed: {status['metrics']['computation']['features_computed_total']}")
    print(f"  Features Failed: {status['metrics']['computation']['features_failed']}")
    print(f"  Symbols Processed: {status['metrics']['computation']['symbols_processed']}")
    print(f"  Avg Computation Time: {status['metrics']['performance']['avg_computation_time_ms']:.2f}ms")
    print(f"  Feature Coverage: {status['metrics']['data_quality']['avg_feature_coverage_pct']:.2f}%")
    print(f"  Validation Pass Rate: {status['metrics']['data_quality']['validation_pass_pct']:.2f}%")
    print()
    
    # Print summary
    print(monitor.get_summary())
    
    # Export metrics
    export_path = monitor.export_metrics(Path("./test_health_export.json"))
    print(f"✓ Metrics exported to: {export_path}")
    print()
    
    print("=" * 80)
    print("HEALTH MONITOR TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_health_monitor()
