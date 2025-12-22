"""Quick test of QSE implementation"""

import numpy as np
import pandas as pd
from arbitrex.quant_stats import QuantStatsConfig, QuantitativeStatisticsEngine

# Generate simple test data
np.random.seed(42)
n = 150
returns = pd.Series(np.random.normal(0.001, 0.01, n))

# Initialize QSE
print("Initializing QSE...")
config = QuantStatsConfig()
qse = QuantitativeStatisticsEngine(config)
print(f"QSE initialized with config: {config.get_config_hash()}")

# Process a single bar
print("\nProcessing single bar (index=100)...")
output = qse.process_bar(
    symbol='TEST',
    returns=returns,
    bar_index=100
)

print(f"\nSymbol: {output.symbol}")
print(f"Timeframe: {output.timeframe}")
print(f"Signal Valid: {output.validation.signal_validity_flag}")
print(f"Trend Persistence Score: {output.metrics.trend_persistence_score:.4f}")
print(f"ADF Stationary: {output.metrics.adf_stationary}")
print(f"Z-Score: {output.metrics.z_score:.4f}")
print(f"Volatility Regime: {output.metrics.volatility_regime}")

if output.validation.failure_reasons:
    print(f"\nFailure Reasons:")
    for reason in output.validation.failure_reasons:
        print(f"  - {reason}")

print("\n SUCCESS!")
