"""
ML Layer - Adaptive Filter for Signal Validation

The ML Layer is a controlled, regime-aware filter that provides:
- Regime classification (Trending/Ranging/Stressed)
- Signal confidence scoring (momentum continuation probability)
- Explainable feature importance
- Auditable model versioning

CRITICAL PRINCIPLE:
    ML does NOT predict prices, generate signals, or override risk rules.
    It answers: "Is this validated signal likely to succeed under current conditions?"

Flow:
    Feature Engine → QSE → ML Layer → Signal Generator → Risk Manager
"""

from arbitrex.ml_layer.config import MLConfig
from arbitrex.ml_layer.schemas import (
    MLOutput,
    RegimeLabel,
    MLPrediction,
    ModelMetadata
)
from arbitrex.ml_layer.inference import MLInferenceEngine
from arbitrex.ml_layer.monitoring import MLMonitor
from arbitrex.ml_layer.model_registry import ModelRegistry

__version__ = "1.0.0"

__all__ = [
    'MLConfig',
    'MLInferenceEngine',
    'MLMonitor',
    'ModelRegistry',
    'MLOutput',
    'RegimeLabel',
    'MLPrediction',
    'ModelMetadata',
    'MLInferenceEngine',
]
