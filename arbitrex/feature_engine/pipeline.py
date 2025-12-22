"""
Feature Pipeline

Orchestrates feature computation from clean OHLCV bars.

Pipeline Stages:
    1. Input validation (valid_bar gate)
    2. Primitive transforms
    3. Rolling statistics
    4. Structural/regime metrics
    5. Normalization
    6. Feature vector freeze
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import logging

from arbitrex.feature_engine.config import FeatureEngineConfig
from arbitrex.feature_engine.validation import FeatureInputValidator
from arbitrex.feature_engine.returns_momentum import ReturnsMomentumFeatures
from arbitrex.feature_engine.volatility import VolatilityFeatures
from arbitrex.feature_engine.trend import TrendFeatures
from arbitrex.feature_engine.efficiency import EfficiencyFeatures
from arbitrex.feature_engine.regime import RegimeFeatures
from arbitrex.feature_engine.execution import ExecutionFeatures
from arbitrex.feature_engine.normalization import FeatureNormalizer
from arbitrex.feature_engine.schemas import (
    FeatureVector,
    FeatureMetadata,
    FeatureSchema
)

LOG = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature computation pipeline.
    
    Transforms clean OHLCV bars into normalized feature vectors.
    
    Philosophy:
        - Causal: No lookahead
        - Stationary: No raw prices
        - Deterministic: Same input → same output
        - Immutable: Features never recomputed
    """
    
    def __init__(self, config: Optional[FeatureEngineConfig] = None):
        self.config = config or FeatureEngineConfig()
        
        # Initialize components
        self.validator = FeatureInputValidator(self.config)
        self.returns_momentum = ReturnsMomentumFeatures(self.config.returns_momentum)
        self.volatility = VolatilityFeatures(self.config.volatility)
        self.trend = TrendFeatures(self.config.trend)
        self.efficiency = EfficiencyFeatures(self.config.efficiency)
        self.regime = RegimeFeatures(self.config.regime)
        self.execution = ExecutionFeatures(self.config.execution)
        self.normalizer = FeatureNormalizer(self.config.normalization)
        self.schema = FeatureSchema()
        
        # Processing metadata
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
        LOG.info(f"Feature Pipeline initialized with config version {self.config.config_version}")
    
    def compute_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        normalize: bool = True
    ) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """
        Compute features from clean OHLCV data.
        
        Args:
            df: Clean OHLCV DataFrame from Clean Data Layer
            symbol: Symbol being processed
            timeframe: Timeframe (1H, 4H, 1D)
            normalize: Apply rolling z-score normalization
        
        Returns:
            (feature_df, metadata)
        
        Pipeline:
            1. Input validation
            2. Feature computation (all categories)
            3. Normalization (if enabled)
            4. Feature vector freeze
        """
        start_time = datetime.utcnow()
        
        LOG.info(f"Computing features: {symbol} {timeframe} ({len(df)} bars)")
        
        # ===================================================================
        # STAGE 1: INPUT VALIDATION
        # ===================================================================
        is_valid, df_valid, errors = self.validator.validate_input(df, symbol, timeframe)
        
        if not is_valid:
            error_msg = f"Input validation failed: {errors}"
            LOG.error(error_msg)
            self.errors.extend(errors)
            
            if self.config.fail_on_critical_error:
                raise ValueError(error_msg)
            
            # Return empty result
            metadata = self._create_error_metadata(
                symbol, timeframe, start_time, errors
            )
            return pd.DataFrame(), metadata
        
        LOG.info(f"✓ Stage 1: Input validation passed ({len(df_valid)} valid bars)")
        
        # ===================================================================
        # STAGE 2-4: FEATURE COMPUTATION
        # ===================================================================
        
        # Category A: Returns & Momentum
        df_features = self.returns_momentum.compute(df_valid)
        LOG.info(f"✓ Stage 2: Returns & Momentum computed")
        
        # Category B: Volatility
        df_features = self.volatility.compute(df_features)
        LOG.info(f"✓ Stage 3: Volatility computed")
        
        # Category C: Trend
        df_features = self.trend.compute(df_features)
        LOG.info(f"✓ Stage 4: Trend computed")
        
        # Category D: Efficiency
        df_features = self.efficiency.compute(df_features)
        LOG.info(f"✓ Stage 5: Efficiency computed")
        
        # Category E: Regime (daily only)
        df_features = self.regime.compute(df_features, timeframe)
        LOG.info(f"✓ Stage 6: Regime computed")
        
        # Category F: Execution (optional)
        df_features = self.execution.compute(df_features)
        LOG.info(f"✓ Stage 7: Execution computed")
        
        # ===================================================================
        # STAGE 5: NORMALIZATION
        # ===================================================================
        
        if normalize:
            # Get feature columns for normalization
            feature_cols = self.schema.get_ml_features(timeframe)
            
            # Filter to existing columns only
            feature_cols_existing = [
                col for col in feature_cols if col in df_features.columns
            ]
            
            df_features, norm_metadata = self.normalizer.normalize(
                df_features,
                feature_cols_existing
            )
            LOG.info(f"✓ Stage 8: Normalization applied to {len(feature_cols_existing)} features")
        else:
            norm_metadata = {}
            LOG.info(f"✓ Stage 8: Normalization skipped")
        
        # ===================================================================
        # STAGE 6: METADATA GENERATION
        # ===================================================================
        
        metadata = FeatureMetadata(
            processing_timestamp=datetime.utcnow(),
            config_version=self.config.config_version,
            config_hash=self.config.get_config_hash(),
            source_symbol=symbol,
            source_timeframe=timeframe,
            source_start=df_features['timestamp_utc'].iloc[0],
            source_end=df_features['timestamp_utc'].iloc[-1],
            total_bars_input=len(df),
            valid_bars_processed=len(df_features),
            features_computed=len(self.schema.get_all_features(timeframe)),
            feature_names=self.schema.get_all_features(timeframe),
            normalization_applied=normalize,
            normalization_window=self.config.normalization.norm_window if normalize else 0,
            warnings=self.warnings,
            errors=self.errors,
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        LOG.info(f"✓ Feature computation complete: {symbol} {timeframe} "
                f"({processing_time:.2f}s, {len(df_features)} bars)")
        
        return df_features, metadata
    
    def freeze_feature_vector(
        self,
        df: pd.DataFrame,
        timestamp: datetime,
        symbol: str,
        timeframe: str,
        ml_only: bool = True
    ) -> FeatureVector:
        """
        Freeze feature vector for single timestamp.
        
        Args:
            df: DataFrame with computed features
            timestamp: Timestamp to freeze
            symbol: Symbol
            timeframe: Timeframe
            ml_only: If True, exclude execution features
        
        Returns:
            FeatureVector (immutable)
        
        Use Case:
            - Live trading: freeze at bar close
            - Backtesting: freeze each bar
            - ML training: freeze training samples
        """
        # Get row for timestamp
        row = df[df['timestamp_utc'] == timestamp]
        
        if len(row) == 0:
            raise ValueError(f"Timestamp {timestamp} not found in DataFrame")
        
        row = row.iloc[0]
        
        # Get feature columns
        if ml_only:
            feature_cols = self.schema.get_ml_features(timeframe)
        else:
            feature_cols = self.schema.get_all_features(timeframe)
        
        # Get normalized feature names
        norm_cols = [f'{col}_norm' for col in feature_cols if f'{col}_norm' in df.columns]
        
        # Extract feature values
        feature_values = row[norm_cols].values
        
        # Create feature vector
        vector = FeatureVector(
            timestamp_utc=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            feature_values=feature_values,
            feature_names=norm_cols,
            feature_version=self.config.get_config_hash(),
            schema_version=self.schema.version,
            normalization_metadata=self.normalizer.normalization_metadata,
            is_ml_ready=ml_only,
        )
        
        return vector
    
    def _create_error_metadata(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        errors: List[str]
    ) -> FeatureMetadata:
        """Create metadata for failed computation"""
        return FeatureMetadata(
            processing_timestamp=start_time,
            config_version=self.config.config_version,
            config_hash=self.config.get_config_hash(),
            source_symbol=symbol,
            source_timeframe=timeframe,
            source_start=start_time,
            source_end=start_time,
            total_bars_input=0,
            valid_bars_processed=0,
            features_computed=0,
            feature_names=[],
            normalization_applied=False,
            normalization_window=0,
            warnings=[],
            errors=errors,
        )
