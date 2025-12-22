"""
ARBITREX Feature Engine

Transforms clean OHLCV bars into stationary, normalized feature vectors.

Philosophy:
    - Causality: No lookahead, all windows end at time t
    - Stationarity: No raw prices, only returns/ratios/normalized distances
    - Determinism: Same input → same output, fully reproducible
    - Timeframe Isolation: No mixing timeframes at computation time
    - Data Trust: Only consume valid_bar == True from Clean Data Layer
    - No Retail: No RSI, MACD, Stochastic, CCI

Features describe market condition, NOT next price move.
"""

from arbitrex.feature_engine.config import FeatureEngineConfig
from arbitrex.feature_engine.pipeline import FeaturePipeline
from arbitrex.feature_engine.schemas import FeatureVector, FeatureMetadata

__all__ = [
    'FeatureEngineConfig',
    'FeaturePipeline',
    'FeatureVector',
    'FeatureMetadata',
]

__version__ = '1.0.0'
