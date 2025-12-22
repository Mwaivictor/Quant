"""
Quantitative Statistics Engine (QSE)

Statistical gatekeeper between Feature Engine and ML layers.
Validates signals, prevents noise-based learning, ensures regime stability.

Core Responsibilities:
    - Autocorrelation & trend persistence analysis
    - Stationarity validation (ADF tests)
    - Distribution stability & outlier detection
    - Cross-pair correlation monitoring
    - Volatility regime filtering
    - Signal validation logic

Flow:
    Feature Engine → QSE → ML Layer → Signal Generator
"""

from arbitrex.quant_stats.config import QuantStatsConfig
from arbitrex.quant_stats.engine import QuantitativeStatisticsEngine
from arbitrex.quant_stats.health_monitor import QSEHealthMonitor
from arbitrex.quant_stats.schemas import (
    StatisticalMetrics,
    SignalValidation,
    RegimeState,
    QuantStatsOutput
)
from arbitrex.quant_stats.autocorrelation import AutocorrelationAnalyzer
from arbitrex.quant_stats.stationarity import StationarityTester
from arbitrex.quant_stats.distribution import DistributionAnalyzer
from arbitrex.quant_stats.correlation import CorrelationAnalyzer
from arbitrex.quant_stats.volatility import VolatilityFilter, VolatilityRegime

__version__ = "1.0.0"

__all__ = [
    'QuantStatsConfig',
    'QuantitativeStatisticsEngine',
    'QSEHealthMonitor',
    'StatisticalMetrics',
    'SignalValidation',
    'RegimeState',
    'QuantStatsOutput',
    'AutocorrelationAnalyzer',
    'StationarityTester',
    'DistributionAnalyzer',
    'CorrelationAnalyzer',
    'VolatilityFilter',
    'VolatilityRegime',
]
