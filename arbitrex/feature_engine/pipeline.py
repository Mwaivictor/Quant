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
    
Parallelism:
    - Per-symbol Tier 1 features (fully independent)
    - Coordinated Tier 2 features (correlation matrix, versioned)
    - Event-driven feature emission
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        - Parallel: Per-symbol independence
    """
    
    def __init__(self, config: Optional[FeatureEngineConfig] = None, emit_events: bool = False, max_workers: int = 10):
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
        
        # Event-driven processing
        self._emit_events = emit_events
        self._event_bus = None
        if emit_events:
            try:
                from arbitrex.event_bus import get_event_bus, Event, EventType
                self._event_bus = get_event_bus()
                self._Event = Event
                self._EventType = EventType
                
                # Subscribe to NormalizedBarReady events
                self._event_bus.subscribe(
                    EventType.NORMALIZED_BAR_READY,
                    self._on_normalized_bar_ready
                )
            except ImportError:
                self._emit_events = False
                LOG.warning("Event bus not available")
        
        # Parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FeatureTier1Worker")
        self._lock = threading.Lock()
        
        # Correlation matrix computation (Tier 2)
        self._correlation_version = 0
        self._correlation_matrix = None
        self._feature_cache: Dict[str, pd.DataFrame] = {}  # Cache for correlation computation
        
        LOG.info(f"Feature Pipeline initialized with config version {self.config.config_version}")
    
    def _on_normalized_bar_ready(self, event):
        """Event handler for normalized bars"""
        # This would trigger feature computation in event-driven mode
        # For now, just log
        LOG.debug(f"Received NormalizedBarReady for {event.symbol}")
    
    def compute_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        normalize: bool = True
    ) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """
        Compute Tier 1 features from clean OHLCV data.
        
        This is thread-safe and can be called in parallel for different symbols.
        
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
        
        # Cache features for correlation computation
        with self._lock:
            self._feature_cache[symbol] = df_features.copy()
        
        # Emit FeatureTier1Ready event
        if self._emit_events and self._event_bus:
            try:
                event = self._Event(
                    event_type=self._EventType.FEATURE_TIER1_READY,
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    data={
                        'timeframe': timeframe,
                        'bar_count': len(df_features),
                        'feature_count': len(self.schema.get_all_features(timeframe)),
                        'processing_time_seconds': processing_time,
                        'normalized': normalize
                    }
                )
                self._event_bus.publish(event)
            except Exception as e:
                LOG.warning(f"Failed to emit FeatureTier1Ready event: {e}")
        
        return df_features, metadata
        
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
    
    def compute_correlation_matrix(self, symbols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], int]:
        """
        Compute Tier 2 correlation matrix across symbols.
        
        Args:
            symbols: List of symbols to include (None = all cached symbols)
        
        Returns:
            (correlation_matrix, symbols, version)
        
        This is a coordinated operation that requires snapshots from all symbols.
        """
        with self._lock:
            if symbols is None:
                symbols = list(self._feature_cache.keys())
            
            if len(symbols) < 2:
                LOG.warning("Need at least 2 symbols for correlation matrix")
                return np.eye(len(symbols)), symbols, self._correlation_version
            
            # Extract returns from each symbol
            returns_data = {}
            for symbol in symbols:
                if symbol not in self._feature_cache:
                    continue
                df = self._feature_cache[symbol]
                # Look for log_return_1 (from input) or rolling_return_1 (from features)
                if 'log_return_1' in df.columns:
                    returns_data[symbol] = df['log_return_1'].values
                elif 'rolling_return_1' in df.columns:
                    returns_data[symbol] = df['rolling_return_1'].values
            
            if len(returns_data) < 2:
                LOG.warning("Insufficient return data for correlation")
                return np.eye(len(symbols)), symbols, self._correlation_version
            
            # Align lengths (use minimum)
            min_length = min(len(v) for v in returns_data.values())
            returns_matrix = np.array([returns_data[s][-min_length:] for s in symbols])
            
            # Compute correlation
            correlation_matrix = np.corrcoef(returns_matrix)
            
            # Increment version
            self._correlation_version += 1
            self._correlation_matrix = correlation_matrix
            
            LOG.info(f"Computed correlation matrix v{self._correlation_version} for {len(symbols)} symbols")
            
            # Emit FeatureTier2Ready event
            if self._emit_events and self._event_bus:
                try:
                    event = self._Event(
                        event_type=self._EventType.FEATURE_TIER2_READY,
                        timestamp=datetime.utcnow(),
                        symbol=None,  # Global event
                        data={
                            'symbols': symbols,
                            'matrix_shape': correlation_matrix.shape,
                            'version': self._correlation_version
                        },
                        version=self._correlation_version
                    )
                    self._event_bus.publish(event)
                except Exception as e:
                    LOG.warning(f"Failed to emit FeatureTier2Ready event: {e}")
            
            return correlation_matrix, symbols, self._correlation_version
    
    def compute_features_parallel(
        self,
        data: Dict[str, pd.DataFrame],
        timeframe: str,
        normalize: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, FeatureMetadata]]:
        """
        Compute features for multiple symbols in parallel.
        
        Args:
            data: Dict mapping symbol to clean OHLCV dataframe
            timeframe: Timeframe for all symbols
            normalize: Apply normalization
        
        Returns:
            Dict mapping symbol to (features_df, metadata)
        """
        LOG.info(f"Computing features for {len(data)} symbols in parallel")
        start_time = datetime.utcnow()
        
        results = {}
        
        # Submit all symbols to thread pool
        future_to_symbol = {
            self._executor.submit(self.compute_features, df, symbol, timeframe, normalize): symbol
            for symbol, df in data.items()
        }
        
        # Collect results
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                features_df, metadata = future.result()
                results[symbol] = (features_df, metadata)
            except Exception as e:
                LOG.error(f"Feature computation failed for {symbol}: {e}")
                results[symbol] = (pd.DataFrame(), self._create_error_metadata(
                    symbol, timeframe, start_time, [str(e)]
                ))
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        LOG.info(f"Computed features for {len(results)} symbols in {processing_time:.2f}s")
        
        # Compute correlation matrix (Tier 2)
        if len(results) > 1:
            try:
                self.compute_correlation_matrix(list(results.keys()))
            except Exception as e:
                LOG.error(f"Correlation matrix computation failed: {e}")
        
        return results
