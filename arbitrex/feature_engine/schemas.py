"""
Feature Vector Schemas

Defines output structure for feature vectors and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FeatureVector:
    """
    Single feature vector snapshot.
    
    Immutable once created.
    """
    
    # Identification
    timestamp_utc: datetime
    symbol: str
    timeframe: str
    
    # Feature values (ordered)
    feature_values: np.ndarray
    feature_names: List[str]
    
    # Versioning
    feature_version: str  # Config hash
    schema_version: str = "1.0.0"
    
    # Normalization metadata
    normalization_metadata: Optional[Dict] = None
    
    # Flags
    is_ml_ready: bool = True  # False if execution features included
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'timestamp_utc': self.timestamp_utc.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'feature_values': self.feature_values.tolist(),
            'feature_names': self.feature_names,
            'feature_version': self.feature_version,
            'schema_version': self.schema_version,
            'is_ml_ready': self.is_ml_ready,
            'normalization_metadata': self.normalization_metadata,
        }


@dataclass
class FeatureMetadata:
    """
    Metadata for feature computation batch.
    
    Required for auditability and reproducibility.
    """
    
    # Processing info
    processing_timestamp: datetime
    config_version: str
    config_hash: str
    
    # Source data
    source_symbol: str
    source_timeframe: str
    source_start: datetime
    source_end: datetime
    
    # Processing statistics
    total_bars_input: int
    valid_bars_processed: int
    features_computed: int
    feature_names: List[str]
    
    # Normalization info
    normalization_applied: bool
    normalization_window: int
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'config_version': self.config_version,
            'config_hash': self.config_hash,
            'source': {
                'symbol': self.source_symbol,
                'timeframe': self.source_timeframe,
                'start': self.source_start.isoformat(),
                'end': self.source_end.isoformat(),
            },
            'statistics': {
                'total_bars_input': self.total_bars_input,
                'valid_bars_processed': self.valid_bars_processed,
                'features_computed': self.features_computed,
                'feature_names': self.feature_names,
            },
            'normalization': {
                'applied': self.normalization_applied,
                'window': self.normalization_window,
            },
            'warnings': self.warnings,
            'errors': self.errors,
        }


@dataclass
class FeatureSchema:
    """
    Feature vector schema definition.
    
    Defines expected features for each timeframe.
    """
    
    # Schema version
    version: str = "1.0.0"
    
    # Feature categories (in order)
    returns_momentum_features: List[str] = field(default_factory=lambda: [
        'rolling_return_3',
        'rolling_return_6',
        'rolling_return_12',
        'momentum_score',
    ])
    
    volatility_features: List[str] = field(default_factory=lambda: [
        'vol_6',
        'vol_12',
        'vol_24',
        'atr_normalized',
    ])
    
    trend_features: List[str] = field(default_factory=lambda: [
        'ma_12_slope',
        'ma_24_slope',
        'ma_50_slope',
        'distance_to_ma_12',
        'distance_to_ma_24',
        'distance_to_ma_50',
    ])
    
    efficiency_features: List[str] = field(default_factory=lambda: [
        'efficiency_ratio',
        'range_compression',
    ])
    
    regime_features: List[str] = field(default_factory=lambda: [
        'trend_regime',
        'stress_indicator',
    ])
    
    execution_features: List[str] = field(default_factory=lambda: [
        'spread_ratio',
    ])
    
    def get_ml_features(self, timeframe: str) -> List[str]:
        """
        Get features suitable for ML models.
        
        Excludes:
            - Execution features (always)
            - Regime features for non-daily timeframes
        """
        features = []
        
        # Always include
        features.extend(self.returns_momentum_features)
        features.extend(self.volatility_features)
        features.extend(self.trend_features)
        features.extend(self.efficiency_features)
        
        # Regime features only for daily
        if timeframe == '1D':
            features.extend(self.regime_features)
        
        # Execution features NEVER included in ML
        
        return features
    
    def get_all_features(self, timeframe: str) -> List[str]:
        """Get all features including execution filters"""
        features = self.get_ml_features(timeframe)
        features.extend(self.execution_features)
        return features
