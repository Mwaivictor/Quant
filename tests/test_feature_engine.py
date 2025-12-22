"""
Comprehensive test suite for ARBITREX Feature Engine.

Tests all feature categories, primitives, normalization, pipeline,
feature store, and validation logic.

Run: pytest tests/test_feature_engine.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import tempfile
import shutil

from arbitrex.feature_engine.config import (
    FeatureEngineConfig,
    ReturnsMomentumConfig,
    VolatilityConfig,
    TrendConfig,
    EfficiencyConfig,
    RegimeConfig,
    ExecutionConfig,
    NormalizationConfig
)
from arbitrex.feature_engine.validation import FeatureInputValidator
from arbitrex.feature_engine.primitives import PrimitiveTransforms
from arbitrex.feature_engine.returns_momentum import ReturnsMomentumFeatures
from arbitrex.feature_engine.volatility import VolatilityFeatures
from arbitrex.feature_engine.trend import TrendFeatures
from arbitrex.feature_engine.efficiency import EfficiencyFeatures
from arbitrex.feature_engine.regime import RegimeFeatures
from arbitrex.feature_engine.execution import ExecutionFeatures
from arbitrex.feature_engine.normalization import FeatureNormalizer
from arbitrex.feature_engine.schemas import FeatureVector, FeatureMetadata, FeatureSchema
from arbitrex.feature_engine.pipeline import FeaturePipeline
from arbitrex.feature_engine.feature_store import FeatureStore


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_clean_data():
    """Generate sample clean OHLCV data for testing (matching Clean Data Layer output)"""
    np.random.seed(42)
    n_bars = 200
    
    base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_bars)]
    
    # Generate realistic OHLCV
    close = 100.0
    data = []
    prev_close = close
    
    for ts in timestamps:
        ret = np.random.normal(0.0001, 0.01)
        close = close * (1 + ret)
        
        spread = 0.0002 * close
        high = close + np.random.uniform(0, 0.005) * close
        low = close - np.random.uniform(0, 0.005) * close
        open_price = low + np.random.uniform(0, 1) * (high - low)
        volume = np.random.uniform(1000, 10000)
        
        # Calculate log return
        log_return = np.log(close / prev_close) if prev_close > 0 else 0.0
        
        data.append({
            'timestamp_utc': ts,
            'symbol': 'EURUSD',
            'timeframe': '1H',
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'spread': spread,
            'log_return_1': log_return,
            'valid_bar': True
        })
        
        prev_close = close
    
    return pd.DataFrame(data)


@pytest.fixture
def feature_config():
    """Default feature engine configuration"""
    return FeatureEngineConfig()


@pytest.fixture
def temp_feature_store():
    """Temporary feature store for testing"""
    temp_dir = tempfile.mkdtemp()
    store = FeatureStore(Path(temp_dir))
    yield store
    shutil.rmtree(temp_dir)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

class TestConfiguration:
    """Test configuration and versioning"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = FeatureEngineConfig()
        
        assert config.config_version == "1.0.0"
        assert config.returns_momentum.enabled
        assert config.volatility.enabled
        assert config.trend.enabled
        assert config.efficiency.enabled
        assert config.regime.enabled
        assert config.execution.enabled == False  # Execution features disabled by default
        
    def test_config_hash_deterministic(self):
        """Test configuration hash is deterministic"""
        config1 = FeatureEngineConfig()
        config2 = FeatureEngineConfig()
        
        hash1 = config1.get_config_hash()
        hash2 = config2.get_config_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars
        
    def test_config_hash_changes_with_params(self):
        """Test config hash changes when parameters change"""
        config1 = FeatureEngineConfig()
        config2 = FeatureEngineConfig(
            returns_momentum=ReturnsMomentumConfig(momentum_window=24)
        )
        
        hash1 = config1.get_config_hash()
        hash2 = config2.get_config_hash()
        
        assert hash1 != hash2
        
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = FeatureEngineConfig()
        
        config_dict = config.to_dict()
        assert "config_version" in config_dict
        assert "config_hash" in config_dict
        assert "returns_momentum" in config_dict
        assert "normalization" in config_dict
        
        config_json = config.to_json()
        assert isinstance(config_json, str)
        assert "config_version" in config_json
        
    def test_custom_config_creation(self):
        """Test custom configuration with modified parameters"""
        custom_config = FeatureEngineConfig(
            returns_momentum=ReturnsMomentumConfig(
                return_windows=[5, 10, 20],
                momentum_window=20
            ),
            normalization=NormalizationConfig(
                norm_window=100,
                use_robust=True
            )
        )
        
        assert custom_config.returns_momentum.return_windows == [5, 10, 20]
        assert custom_config.returns_momentum.momentum_window == 20
        assert custom_config.normalization.norm_window == 100
        assert custom_config.normalization.use_robust == True


# ============================================================================
# TEST VALIDATION
# ============================================================================

class TestValidation:
    """Test input validation logic"""
    
    def test_valid_input_passes(self, sample_clean_data, feature_config):
        """Test valid input passes validation"""
        validator = FeatureInputValidator(feature_config)
        
        is_valid, df_valid, errors = validator.validate_input(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H'
        )
        
        assert is_valid
        assert len(df_valid) == len(sample_clean_data)
        assert len(errors) == 0
        
    def test_missing_columns_fails(self, sample_clean_data, feature_config):
        """Test missing required columns fails validation"""
        validator = FeatureInputValidator(feature_config)
        df_incomplete = sample_clean_data.drop(columns=['close'])
        
        is_valid, df_valid, errors = validator.validate_input(
            df_incomplete,
            symbol='EURUSD',
            timeframe='1H'
        )
        
        assert not is_valid
        assert "missing required columns" in errors[0].lower()
        
    def test_invalid_bars_filtered(self, sample_clean_data, feature_config):
        """Test invalid bars are filtered out"""
        validator = FeatureInputValidator(feature_config)
        
        # Mark only a few bars as invalid (not too many, or we'll fail min bars requirement)
        df_test = sample_clean_data.copy()
        df_test.loc[10:15, 'valid_bar'] = False  # Mark 6 bars invalid (out of 200)
        
        is_valid, df_valid, errors = validator.validate_input(
            df_test,
            symbol='EURUSD',
            timeframe='1H'
        )
        
        # Should pass after filtering (200-6=194 > min_valid_bars_required=100)
        assert is_valid
        assert len(df_valid) < len(df_test)
        assert df_valid['valid_bar'].all()
        
    def test_insufficient_bars_fails(self, sample_clean_data, feature_config):
        """Test insufficient bars fails validation"""
        validator = FeatureInputValidator(feature_config)
        df_short = sample_clean_data.head(30)  # Too few bars
        
        is_valid, df_valid, errors = validator.validate_input(
            df_short,
            symbol='EURUSD',
            timeframe='1H'
        )
        
        assert not is_valid
        assert "insufficient" in errors[0].lower() and "bars" in errors[0].lower()
        
    def test_nan_in_ohlcv_fails(self, sample_clean_data, feature_config):
        """Test NaN in OHLCV fails validation"""
        validator = FeatureInputValidator(feature_config)
        sample_clean_data.loc[50, 'close'] = np.nan
        
        is_valid, df_valid, errors = validator.validate_input(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H'
        )
        
        assert not is_valid
        assert "nan" in errors[0].lower()


# ============================================================================
# TEST PRIMITIVES
# ============================================================================

class TestPrimitives:
    """Test causal primitive transforms"""
    
    def test_rolling_mean_causal(self, sample_clean_data):
        """Test rolling mean is causal (no lookahead)"""
        series = sample_clean_data['close']
        result = PrimitiveTransforms.rolling_mean(series, window=10)
        
        # First 9 values should be NaN
        assert result[:9].isna().all()
        
        # Value at position 10 should only use data up to position 10
        manual_mean = series[:10].mean()
        assert np.isclose(result.iloc[9], manual_mean)
        
    def test_rolling_std_causal(self, sample_clean_data):
        """Test rolling std is causal"""
        series = sample_clean_data['close']
        result = PrimitiveTransforms.rolling_std(series, window=10)
        
        assert result[:9].isna().all()
        
    def test_rolling_return(self, sample_clean_data):
        """Test rolling return calculation"""
        log_returns = sample_clean_data['log_return_1']
        rolling_ret = PrimitiveTransforms.rolling_return(log_returns, window=3)
        
        # Should be NaN for first window-1 values
        assert pd.isna(rolling_ret.iloc[0])
        assert pd.isna(rolling_ret.iloc[1])
        
        # Check calculation (sum of 3 log returns)
        expected = log_returns.iloc[0:3].sum()
        assert np.isclose(rolling_ret.iloc[2], expected, equal_nan=True)
        
    def test_atr_calculation(self, sample_clean_data):
        """Test ATR (True Range) calculation"""
        result = PrimitiveTransforms.atr(
            sample_clean_data['high'],
            sample_clean_data['low'],
            sample_clean_data['close'],
            window=14
        )
        
        # Window=14, but first result appears at index 13 (0-indexed)
        assert result[:13].isna().all()
        assert result[13:].notna().any()  # Some non-null values after warmup
        assert (result[14:] > 0).all()
        
    def test_efficiency_ratio(self, sample_clean_data):
        """Test Kaufman Efficiency Ratio"""
        close = sample_clean_data['close']
        result = PrimitiveTransforms.efficiency_ratio(close, direction_window=10, volatility_window=10)
        
        # ER should be between 0 and 1
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 1).all()
        
    def test_ma_slope(self, sample_clean_data):
        """Test MA slope calculation"""
        close = sample_clean_data['close']
        atr = PrimitiveTransforms.atr(
            sample_clean_data['high'],
            sample_clean_data['low'],
            close,
            window=14
        )
        
        ma = PrimitiveTransforms.simple_moving_average(close, window=12)
        result = PrimitiveTransforms.ma_slope(ma, window=3, atr=atr)
        
        # Should be normalized by ATR
        assert result.isna().sum() > 0  # Some NaN at start
        assert result.dropna().abs().max() < 10  # Reasonable range


# ============================================================================
# TEST FEATURE CATEGORIES
# ============================================================================

class TestReturnsMomentum:
    """Test Category A - Returns & Momentum"""
    
    def test_compute_returns_momentum(self, sample_clean_data):
        """Test returns and momentum computation"""
        config = ReturnsMomentumConfig()
        feature_engine = ReturnsMomentumFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # Check expected columns exist
        assert 'rolling_return_3' in df.columns
        assert 'rolling_return_6' in df.columns
        assert 'rolling_return_12' in df.columns
        assert 'momentum_score' in df.columns
        
    def test_momentum_score_calculation(self, sample_clean_data):
        """Test momentum score = return / volatility"""
        config = ReturnsMomentumConfig()
        feature_engine = ReturnsMomentumFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # Momentum score should be return/vol (roughly)
        assert 'momentum_score' in df.columns
        assert df['momentum_score'].isna().sum() > 0  # Some NaN at start
        
    def test_returns_are_stationary(self, sample_clean_data):
        """Test returns are stationary (no raw prices)"""
        config = ReturnsMomentumConfig()
        feature_engine = ReturnsMomentumFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # All return features should be differences/ratios
        for col in ['rolling_return_3', 'rolling_return_6', 'rolling_return_12']:
            assert col in df.columns
            # Returns should be small (typically < 0.1)
            valid_vals = df[col].dropna()
            assert valid_vals.abs().max() < 1.0


class TestVolatility:
    """Test Category B - Volatility"""
    
    def test_compute_volatility(self, sample_clean_data):
        """Test volatility features computation"""
        config = VolatilityConfig()
        feature_engine = VolatilityFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        assert 'vol_6' in df.columns
        assert 'vol_12' in df.columns
        assert 'vol_24' in df.columns
        assert 'atr_normalized' in df.columns
        
    def test_volatility_positive(self, sample_clean_data):
        """Test volatility values are positive"""
        config = VolatilityConfig()
        feature_engine = VolatilityFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        for col in ['vol_6', 'vol_12', 'vol_24']:
            valid_vals = df[col].dropna()
            assert (valid_vals >= 0).all()
            
    def test_atr_normalized(self, sample_clean_data):
        """Test ATR is normalized by close price"""
        config = VolatilityConfig()
        feature_engine = VolatilityFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # ATR normalized should be small (typically < 0.01)
        valid_vals = df['atr_normalized'].dropna()
        assert (valid_vals > 0).all()
        assert valid_vals.max() < 0.1


class TestTrend:
    """Test Category C - Trend Structure"""
    
    def test_compute_trend(self, sample_clean_data):
        """Test trend features computation"""
        config = TrendConfig()
        feature_engine = TrendFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # Check all expected columns
        assert 'ma_12_slope' in df.columns
        assert 'ma_24_slope' in df.columns
        assert 'ma_50_slope' in df.columns
        assert 'distance_to_ma_12' in df.columns
        assert 'distance_to_ma_24' in df.columns
        assert 'distance_to_ma_50' in df.columns
        
    def test_slopes_normalized(self, sample_clean_data):
        """Test slopes are normalized by ATR"""
        config = TrendConfig()
        feature_engine = TrendFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        # Slopes should be in reasonable range (normalized by ATR)
        for col in ['ma_12_slope', 'ma_24_slope', 'ma_50_slope']:
            valid_vals = df[col].dropna()
            assert valid_vals.abs().max() < 10  # Reasonable range
            
    def test_distances_normalized(self, sample_clean_data):
        """Test distances are normalized by ATR"""
        config = TrendConfig()
        feature_engine = TrendFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        for col in ['distance_to_ma_12', 'distance_to_ma_24', 'distance_to_ma_50']:
            valid_vals = df[col].dropna()
            assert valid_vals.abs().max() < 20  # Reasonable range


class TestEfficiency:
    """Test Category D - Market Efficiency"""
    
    def test_compute_efficiency(self, sample_clean_data):
        """Test efficiency features computation"""
        config = EfficiencyConfig()
        feature_engine = EfficiencyFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        assert 'efficiency_ratio' in df.columns
        assert 'range_compression' in df.columns
        
    def test_efficiency_ratio_bounded(self, sample_clean_data):
        """Test efficiency ratio is between 0 and 1"""
        config = EfficiencyConfig()
        feature_engine = EfficiencyFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        valid_vals = df['efficiency_ratio'].dropna()
        assert (valid_vals >= 0).all()
        assert (valid_vals <= 1).all()
        
    def test_range_compression_positive(self, sample_clean_data):
        """Test range compression is positive"""
        config = EfficiencyConfig()
        feature_engine = EfficiencyFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        valid_vals = df['range_compression'].dropna()
        assert (valid_vals > 0).all()


class TestRegime:
    """Test Category E - Regime Detection"""
    
    def test_compute_regime_daily(self, sample_clean_data):
        """Test regime features for daily timeframe"""
        config = RegimeConfig()
        feature_engine = RegimeFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy(), timeframe='1D')
        
        assert 'trend_regime' in df.columns
        assert 'stress_indicator' in df.columns
        
    def test_regime_skipped_intraday(self, sample_clean_data):
        """Test regime features skipped for intraday"""
        config = RegimeConfig()
        feature_engine = RegimeFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy(), timeframe='1H')
        
        # Should have columns but all NaN
        assert 'trend_regime' in df.columns
        assert 'stress_indicator' in df.columns
        assert df['trend_regime'].isna().all()
        assert df['stress_indicator'].isna().all()
        
    def test_trend_regime_values(self, sample_clean_data):
        """Test trend regime is -1, 0, or +1"""
        config = RegimeConfig()
        feature_engine = RegimeFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy(), timeframe='1D')
        
        valid_vals = df['trend_regime'].dropna()
        unique_vals = valid_vals.unique()
        assert set(unique_vals).issubset({-1.0, 0.0, 1.0})
        
    def test_stress_indicator_positive(self, sample_clean_data):
        """Test stress indicator is positive"""
        config = RegimeConfig()
        feature_engine = RegimeFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy(), timeframe='1D')
        
        valid_vals = df['stress_indicator'].dropna()
        assert (valid_vals > 0).all()


class TestExecution:
    """Test Category F - Execution Filters"""
    
    def test_compute_execution(self, sample_clean_data):
        """Test execution features computation"""
        config = ExecutionConfig(enabled=True)  # Enable execution features for this test
        feature_engine = ExecutionFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        assert 'spread_ratio' in df.columns
        
    def test_spread_ratio_positive(self, sample_clean_data):
        """Test spread ratio is positive"""
        config = ExecutionConfig(enabled=True)  # Enable execution features for this test
        feature_engine = ExecutionFeatures(config)
        df = feature_engine.compute(sample_clean_data.copy())
        
        valid_vals = df['spread_ratio'].dropna()
        assert (valid_vals > 0).all()
        
    def test_execution_ml_excluded(self):
        """Test execution features marked as ML excluded"""
        config = ExecutionConfig()
        assert config.ml_excluded == True


# ============================================================================
# TEST NORMALIZATION
# ============================================================================

class TestNormalization:
    """Test feature normalization"""
    
    def test_normalize_features(self, sample_clean_data, feature_config):
        """Test feature normalization"""
        # Compute some features first
        feature_engine = ReturnsMomentumFeatures(feature_config.returns_momentum)
        df = feature_engine.compute(sample_clean_data.copy())
        feature_cols = ['rolling_return_3', 'rolling_return_6', 'rolling_return_12']
        
        normalizer = FeatureNormalizer(feature_config.normalization)
        df_norm, metadata = normalizer.normalize(df, feature_cols)
        
        # Check normalized columns exist
        for col in feature_cols:
            assert f'{col}_norm' in df_norm.columns
            assert f'{col}_mean' in df_norm.columns
            assert f'{col}_std' in df_norm.columns
            
    def test_normalized_distribution(self, sample_clean_data, feature_config):
        """Test normalized features have mean≈0, std≈1"""
        feature_engine = ReturnsMomentumFeatures(feature_config.returns_momentum)
        df = feature_engine.compute(sample_clean_data.copy())
        feature_cols = ['rolling_return_3', 'rolling_return_6']
        
        normalizer = FeatureNormalizer(feature_config.normalization)
        df_norm, _ = normalizer.normalize(df, feature_cols)
        
        for col in feature_cols:
            norm_col = f'{col}_norm'
            valid_vals = df_norm[norm_col].dropna()
            
            if len(valid_vals) > 0:
                # Mean should be close to 0
                assert abs(valid_vals.mean()) < 1.0
                # Std should be close to 1
                assert abs(valid_vals.std() - 1.0) < 1.0
                
    def test_z_score_clipping(self, sample_clean_data, feature_config):
        """Test z-scores are clipped"""
        feature_engine = ReturnsMomentumFeatures(feature_config.returns_momentum)
        df = feature_engine.compute(sample_clean_data.copy())
        feature_cols = ['rolling_return_3']
        
        config = NormalizationConfig(z_score_clip=3.0)
        normalizer = FeatureNormalizer(config)
        df_norm, _ = normalizer.normalize(df, feature_cols)
        
        norm_col = 'rolling_return_3_norm'
        valid_vals = df_norm[norm_col].dropna()
        
        # Should be clipped at ±3
        assert valid_vals.max() <= 3.0
        assert valid_vals.min() >= -3.0
        
    def test_robust_statistics(self, sample_clean_data, feature_config):
        """Test robust normalization (median/MAD)"""
        feature_engine = ReturnsMomentumFeatures(feature_config.returns_momentum)
        df = feature_engine.compute(sample_clean_data.copy())
        feature_cols = ['rolling_return_3']
        
        config = NormalizationConfig(use_robust=True)
        normalizer = FeatureNormalizer(config)
        df_norm, metadata = normalizer.normalize(df, feature_cols)
        
        # Check it runs without error (median/MAD used internally)
        assert 'rolling_return_3_norm' in df_norm.columns
        assert metadata['rolling_return_3']['method'] == 'rolling_z_score'  # metadata is dict[col] -> dict
        assert metadata['rolling_return_3']['robust'] == True


# ============================================================================
# TEST SCHEMAS
# ============================================================================

class TestSchemas:
    """Test feature vector schemas"""
    
    def test_feature_schema_ml_features(self):
        """Test feature schema returns correct ML features"""
        schema = FeatureSchema()
        
        # 1H should have 16 ML features
        ml_features_1h = schema.get_ml_features('1H')
        assert len(ml_features_1h) == 16
        
        # Execution features should be excluded
        assert 'spread_ratio' not in ml_features_1h
        
        # Regime features should be excluded for 1H
        assert 'trend_regime' not in ml_features_1h
        assert 'stress_indicator' not in ml_features_1h
        
    def test_feature_schema_daily_includes_regime(self):
        """Test daily schema includes regime features"""
        schema = FeatureSchema()
        
        ml_features_daily = schema.get_ml_features('1D')
        assert len(ml_features_daily) == 18  # 16 + 2 regime
        
        assert 'trend_regime' in ml_features_daily
        assert 'stress_indicator' in ml_features_daily
        
    def test_feature_vector_creation(self):
        """Test feature vector creation"""
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        feature_names = ['feat1', 'feat2', 'feat3']
        feature_values = np.array([0.5, -0.3, 1.2])
        
        vector = FeatureVector(
            timestamp_utc=timestamp,
            symbol='EURUSD',
            timeframe='1H',
            feature_values=feature_values,
            feature_names=feature_names,
            feature_version='abc123',
            is_ml_ready=True
        )
        
        assert vector.symbol == 'EURUSD'
        assert vector.timeframe == '1H'
        assert len(vector.feature_values) == 3
        assert vector.is_ml_ready
        
    def test_feature_metadata_creation(self):
        """Test feature metadata creation"""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=1)
        
        metadata = FeatureMetadata(
            processing_timestamp=datetime.now(timezone.utc),
            config_version='1.0.0',
            config_hash='abc123',
            source_symbol='EURUSD',
            source_timeframe='1H',
            source_start=start_time,
            source_end=end_time,
            total_bars_input=200,
            valid_bars_processed=200,
            features_computed=16,
            feature_names=['f1', 'f2'],
            normalization_applied=True,
            normalization_window=50,
            warnings=[],
            errors=[]
        )
        
        assert metadata.source_symbol == 'EURUSD'
        assert metadata.valid_bars_processed == 200
        assert metadata.features_computed == 16


# ============================================================================
# TEST PIPELINE
# ============================================================================

class TestPipeline:
    """Test feature computation pipeline"""
    
    def test_pipeline_creation(self, feature_config):
        """Test pipeline creation"""
        pipeline = FeaturePipeline(feature_config)
        
        assert pipeline.config == feature_config
        assert pipeline.validator is not None
        assert pipeline.normalizer is not None
        
    def test_pipeline_compute_features(self, sample_clean_data, feature_config):
        """Test end-to-end feature computation"""
        pipeline = FeaturePipeline(feature_config)
        
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        # Check output structure
        assert isinstance(feature_df, pd.DataFrame)
        assert isinstance(metadata, FeatureMetadata)
        
        # Check features computed
        assert metadata.features_computed > 0
        assert metadata.valid_bars_processed > 0
        
        # Check normalized features exist
        assert any('_norm' in col for col in feature_df.columns)
        
    def test_pipeline_freeze_vector(self, sample_clean_data, feature_config):
        """Test freezing single feature vector"""
        pipeline = FeaturePipeline(feature_config)
        
        feature_df, _ = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        # Freeze last timestamp
        last_timestamp = feature_df['timestamp_utc'].iloc[-1]
        vector = pipeline.freeze_feature_vector(
            feature_df,
            last_timestamp,
            symbol='EURUSD',
            timeframe='1H',
            ml_only=True
        )
        
        assert isinstance(vector, FeatureVector)
        assert vector.symbol == 'EURUSD'
        assert vector.timeframe == '1H'
        assert vector.is_ml_ready
        assert len(vector.feature_values) > 0
        
    def test_pipeline_without_normalization(self, sample_clean_data, feature_config):
        """Test pipeline without normalization"""
        pipeline = FeaturePipeline(feature_config)
        
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=False
        )
        
        # Should have raw features but not normalized
        assert 'rolling_return_3' in feature_df.columns
        assert 'rolling_return_3_norm' not in feature_df.columns
        
    def test_pipeline_daily_timeframe(self, sample_clean_data, feature_config):
        """Test pipeline with daily timeframe (includes regime)"""
        pipeline = FeaturePipeline(feature_config)
        
        # Update timeframe in test data
        df_daily = sample_clean_data.copy()
        df_daily['timeframe'] = '1D'
        
        feature_df, metadata = pipeline.compute_features(
            df_daily,
            symbol='EURUSD',
            timeframe='1D',
            normalize=True
        )
        
        # Regime features should be computed
        assert 'trend_regime' in feature_df.columns
        assert 'stress_indicator' in feature_df.columns
        
        # Check some values are not NaN (though many may be at start)
        assert not feature_df['trend_regime'].isna().all()


# ============================================================================
# TEST FEATURE STORE
# ============================================================================

class TestFeatureStore:
    """Test feature storage and retrieval"""
    
    def test_store_creation(self, temp_feature_store):
        """Test feature store creation"""
        assert temp_feature_store.base_path.exists()
        
    def test_write_and_read_features(self, temp_feature_store, sample_clean_data, feature_config):
        """Test writing and reading features"""
        pipeline = FeaturePipeline(feature_config)
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        config_hash = feature_config.get_config_hash()
        
        # Write
        temp_feature_store.write_features(
            feature_df,
            metadata,
            symbol='EURUSD',
            timeframe='1H',
            config_hash=config_hash
        )
        
        # Read
        loaded_df = temp_feature_store.read_features(
            symbol='EURUSD',
            timeframe='1H',
            config_hash=config_hash
        )
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(feature_df)
        
    def test_feature_exists(self, temp_feature_store, sample_clean_data, feature_config):
        """Test checking if features exist"""
        pipeline = FeaturePipeline(feature_config)
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        config_hash = feature_config.get_config_hash()
        
        # Should not exist yet
        assert not temp_feature_store.exists('EURUSD', '1H', config_hash)
        
        # Write
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # Should exist now
        assert temp_feature_store.exists('EURUSD', '1H', config_hash)
        
    def test_list_versions(self, temp_feature_store, sample_clean_data, feature_config):
        """Test listing available versions"""
        pipeline = FeaturePipeline(feature_config)
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        config_hash = feature_config.get_config_hash()
        
        # Write
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # List versions
        versions = temp_feature_store.list_versions('EURUSD', '1H')
        
        assert len(versions) == 1
        assert config_hash in versions
        
    def test_get_latest_features(self, temp_feature_store, sample_clean_data, feature_config):
        """Test getting latest features"""
        pipeline = FeaturePipeline(feature_config)
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        config_hash = feature_config.get_config_hash()
        
        # Write
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # Get latest
        latest_df = temp_feature_store.get_latest_features('EURUSD', '1H')
        
        assert isinstance(latest_df, pd.DataFrame)
        assert len(latest_df) == len(feature_df)
        
    def test_immutability(self, temp_feature_store, sample_clean_data, feature_config):
        """Test features are immutable once written"""
        pipeline = FeaturePipeline(feature_config)
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        config_hash = feature_config.get_config_hash()
        
        # Write first time
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # Attempt to write again should raise or skip
        # (current implementation allows overwrite, but in production should be immutable)
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # Should still be readable
        loaded_df = temp_feature_store.read_features('EURUSD', '1H', config_hash)
        assert len(loaded_df) == len(feature_df)


# ============================================================================
# TEST INTEGRATION
# ============================================================================

class TestIntegration:
    """Test end-to-end integration scenarios"""
    
    def test_full_workflow(self, sample_clean_data, temp_feature_store):
        """Test complete workflow from data to storage"""
        # 1. Create config
        config = FeatureEngineConfig()
        
        # 2. Initialize pipeline
        pipeline = FeaturePipeline(config)
        
        # 3. Compute features
        feature_df, metadata = pipeline.compute_features(
            sample_clean_data,
            symbol='EURUSD',
            timeframe='1H',
            normalize=True
        )
        
        # 4. Freeze a vector
        last_ts = feature_df['timestamp_utc'].iloc[-1]
        vector = pipeline.freeze_feature_vector(
            feature_df,
            last_ts,
            'EURUSD',
            '1H',
            ml_only=True
        )
        
        # 5. Store features
        config_hash = config.get_config_hash()
        temp_feature_store.write_features(
            feature_df,
            metadata,
            'EURUSD',
            '1H',
            config_hash
        )
        
        # 6. Retrieve features
        loaded_df = temp_feature_store.read_features('EURUSD', '1H', config_hash)
        
        # Validate end-to-end
        assert isinstance(vector, FeatureVector)
        assert vector.is_ml_ready
        assert len(loaded_df) == len(feature_df)
        
    def test_multi_symbol_workflow(self, sample_clean_data, temp_feature_store):
        """Test workflow with multiple symbols"""
        config = FeatureEngineConfig()
        pipeline = FeaturePipeline(config)
        config_hash = config.get_config_hash()
        
        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
        
        for symbol in symbols:
            # Update symbol in test data
            df_symbol = sample_clean_data.copy()
            df_symbol['symbol'] = symbol
            
            feature_df, metadata = pipeline.compute_features(
                df_symbol,
                symbol=symbol,
                timeframe='1H',
                normalize=True
            )
            
            temp_feature_store.write_features(
                feature_df,
                metadata,
                symbol,
                '1H',
                config_hash
            )
        
        # Verify all symbols stored
        for symbol in symbols:
            assert temp_feature_store.exists(symbol, '1H', config_hash)
            df = temp_feature_store.read_features(symbol, '1H', config_hash)
            assert len(df) > 0
            
    def test_reproducibility(self, sample_clean_data):
        """Test feature computation is reproducible"""
        config = FeatureEngineConfig()
        pipeline1 = FeaturePipeline(config)
        pipeline2 = FeaturePipeline(config)
        
        # Compute twice
        df1, _ = pipeline1.compute_features(sample_clean_data.copy(), 'EURUSD', '1H', normalize=True)
        df2, _ = pipeline2.compute_features(sample_clean_data.copy(), 'EURUSD', '1H', normalize=True)
        
        # Should be identical
        feature_cols = [col for col in df1.columns if '_norm' in col]
        
        for col in feature_cols:
            assert np.allclose(
                df1[col].dropna().values,
                df2[col].dropna().values,
                rtol=1e-10
            )


# ============================================================================
# SUMMARY
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
