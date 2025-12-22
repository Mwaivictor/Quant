"""
Tests for Quantitative Statistics Engine

Tests all 5 statistical modules and the main QSE orchestrator.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from arbitrex.quant_stats import (
    QuantStatsConfig,
    QuantitativeStatisticsEngine,
    AutocorrelationAnalyzer,
    StationarityTester,
    DistributionAnalyzer,
    CorrelationAnalyzer,
    VolatilityFilter,
    VolatilityRegime
)


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    n = 200
    # Trending returns with some autocorrelation
    returns = np.random.normal(0.001, 0.01, n)
    # Add some trend
    returns[:100] += np.linspace(0, 0.005, 100)
    return pd.Series(returns)


@pytest.fixture
def multiple_returns():
    """Generate multiple correlated return series."""
    np.random.seed(42)
    n = 200
    base = np.random.normal(0, 0.01, n)
    
    return {
        'SYM1': pd.Series(base + np.random.normal(0, 0.005, n)),
        'SYM2': pd.Series(0.8 * base + np.random.normal(0, 0.005, n)),
        'SYM3': pd.Series(0.3 * base + np.random.normal(0, 0.008, n))
    }


class TestAutocorrelationAnalyzer:
    """Test autocorrelation analyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = AutocorrelationAnalyzer(
            lags=[1, 5, 10],
            min_threshold=0.15,
            rolling_window=60
        )
        assert analyzer.lags == [1, 5, 10]
        assert analyzer.min_threshold == 0.15
        assert analyzer.rolling_window == 60
    
    def test_compute_autocorrelation(self, sample_returns):
        """Test autocorrelation computation."""
        analyzer = AutocorrelationAnalyzer()
        autocorr = analyzer.compute_autocorrelation(sample_returns, lag=1)
        
        assert isinstance(autocorr, pd.Series)
        assert len(autocorr) == len(sample_returns)
    
    def test_analyze_bar(self, sample_returns):
        """Test bar-by-bar analysis."""
        analyzer = AutocorrelationAnalyzer()
        result = analyzer.analyze_bar(sample_returns, bar_index=100)
        
        assert 'lag_1' in result
        assert 'lag_5' in result
        assert 'lag_10' in result
        assert 'lag_20' in result
        assert 'trend_persistence_score' in result
        
        # Check types
        assert isinstance(result['trend_persistence_score'], float)
        assert 0 <= result['trend_persistence_score'] <= 1


class TestStationarityTester:
    """Test stationarity tester."""
    
    def test_initialization(self):
        """Test tester initialization."""
        tester = StationarityTester(
            significance_level=0.05,
            rolling_window=60,
            max_lag=10
        )
        assert tester.significance_level == 0.05
        assert tester.rolling_window == 60
    
    def test_run_adf_test(self, sample_returns):
        """Test ADF test."""
        tester = StationarityTester()
        is_stationary, p_value, test_stat = tester.run_adf_test(sample_returns)
        
        assert isinstance(is_stationary, bool)
        assert isinstance(p_value, float)
        assert isinstance(test_stat, float)
        assert 0 <= p_value <= 1
    
    def test_test_bar(self, sample_returns):
        """Test bar-by-bar testing."""
        tester = StationarityTester()
        is_stat, pval, stat = tester.test_bar(sample_returns, bar_index=100)
        
        assert isinstance(is_stat, bool)
        assert isinstance(pval, float)
        assert isinstance(stat, float)


class TestDistributionAnalyzer:
    """Test distribution analyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = DistributionAnalyzer(
            rolling_window=60,
            z_score_threshold=3.0,
            min_samples=30
        )
        assert analyzer.rolling_window == 60
        assert analyzer.z_score_threshold == 3.0
    
    def test_compute_rolling_stats(self, sample_returns):
        """Test rolling statistics."""
        analyzer = DistributionAnalyzer()
        mean, std = analyzer.compute_rolling_stats(sample_returns)
        
        assert isinstance(mean, pd.Series)
        assert isinstance(std, pd.Series)
        assert len(mean) == len(sample_returns)
        assert len(std) == len(sample_returns)
    
    def test_compute_z_scores(self, sample_returns):
        """Test z-score computation."""
        analyzer = DistributionAnalyzer()
        mean, std = analyzer.compute_rolling_stats(sample_returns)
        z_scores = analyzer.compute_z_scores(sample_returns, mean, std)
        
        assert isinstance(z_scores, pd.Series)
        assert len(z_scores) == len(sample_returns)
    
    def test_analyze_bar(self, sample_returns):
        """Test bar-by-bar analysis."""
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_bar(sample_returns, bar_index=100)
        
        assert 'z_score' in result
        assert 'is_outlier' in result
        assert 'rolling_mean' in result
        assert 'rolling_std' in result
        assert 'distribution_stable' in result
        
        assert isinstance(result['is_outlier'], bool)
        assert isinstance(result['distribution_stable'], bool)


class TestCorrelationAnalyzer:
    """Test correlation analyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CorrelationAnalyzer(
            rolling_window=60,
            max_correlation_threshold=0.85,
            min_pairs=2
        )
        assert analyzer.rolling_window == 60
        assert analyzer.max_correlation_threshold == 0.85
    
    def test_compute_correlation_matrix(self, multiple_returns):
        """Test correlation matrix computation."""
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.compute_correlation_matrix(multiple_returns)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == len(multiple_returns)
    
    def test_get_correlation_metrics(self, multiple_returns):
        """Test correlation metrics extraction."""
        analyzer = CorrelationAnalyzer()
        corr_matrix = analyzer.compute_correlation_matrix(multiple_returns)
        metrics = analyzer.get_correlation_metrics(corr_matrix)
        
        assert 'avg_correlation' in metrics
        assert 'max_correlation' in metrics
        assert 'correlation_dispersion' in metrics
        
        assert 0 <= metrics['avg_correlation'] <= 1
        assert 0 <= metrics['max_correlation'] <= 1
    
    def test_analyze_bar(self, multiple_returns):
        """Test bar-by-bar analysis."""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze_bar(multiple_returns, bar_index=100)
        
        assert 'avg_correlation' in result
        assert 'max_correlation' in result
        assert 'correlation_acceptable' in result
        assert isinstance(result['correlation_acceptable'], bool)


class TestVolatilityFilter:
    """Test volatility filter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = VolatilityFilter(
            rolling_window=60,
            min_percentile=10.0,
            max_percentile=90.0
        )
        assert filter.rolling_window == 60
        assert filter.min_percentile == 10.0
    
    def test_compute_volatility(self, sample_returns):
        """Test volatility computation."""
        filter = VolatilityFilter()
        vol = filter.compute_volatility(sample_returns)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_returns)
    
    def test_classify_regime(self):
        """Test regime classification."""
        filter = VolatilityFilter()
        
        regime_low = filter.classify_regime(0.005, 0.008, 0.015)
        assert regime_low == VolatilityRegime.LOW
        
        regime_normal = filter.classify_regime(0.010, 0.008, 0.015)
        assert regime_normal == VolatilityRegime.NORMAL
        
        regime_high = filter.classify_regime(0.020, 0.008, 0.015)
        assert regime_high == VolatilityRegime.HIGH
    
    def test_analyze_bar(self, sample_returns):
        """Test bar-by-bar analysis."""
        filter = VolatilityFilter()
        result = filter.analyze_bar(sample_returns, bar_index=100)
        
        assert 'current_volatility' in result
        assert 'volatility_percentile' in result
        assert 'regime' in result
        
        assert isinstance(result['regime'], VolatilityRegime)


class TestQuantStatsConfig:
    """Test QSE configuration."""
    
    def test_default_initialization(self):
        """Test default config."""
        config = QuantStatsConfig()
        
        assert config.autocorr.lags == [1, 5, 10, 20]
        assert config.stationarity.significance_level == 0.05
        assert config.distribution.z_score_threshold == 3.0
        assert config.correlation.max_correlation_threshold == 0.85
        assert config.volatility.min_percentile == 10.0
    
    def test_config_hash(self):
        """Test config hashing."""
        config1 = QuantStatsConfig()
        hash1 = config1.get_config_hash()
        
        config2 = QuantStatsConfig()
        hash2 = config2.get_config_hash()
        
        # Same config should produce same hash
        assert hash1 == hash2
        
        # Modified config should produce different hash
        config3 = QuantStatsConfig()
        config3.autocorr.min_threshold = 0.25
        hash3 = config3.get_config_hash()
        
        assert hash1 != hash3
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        config1 = QuantStatsConfig()
        config_dict = config1.to_dict()
        
        assert isinstance(config_dict, dict)
        
        config2 = QuantStatsConfig.from_dict(config_dict)
        
        # Should have same hash
        assert config1.get_config_hash() == config2.get_config_hash()


class TestQuantitativeStatisticsEngine:
    """Test main QSE engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        config = QuantStatsConfig()
        qse = QuantitativeStatisticsEngine(config)
        
        assert qse.config == config
        assert qse.autocorr_analyzer is not None
        assert qse.stationarity_tester is not None
        assert qse.distribution_analyzer is not None
        assert qse.correlation_analyzer is not None
        assert qse.volatility_filter is not None
    
    def test_compute_statistical_metrics(self, sample_returns):
        """Test metrics computation."""
        qse = QuantitativeStatisticsEngine()
        metrics = qse.compute_statistical_metrics(
            returns=sample_returns,
            bar_index=100
        )
        
        # Check all fields present
        assert hasattr(metrics, 'autocorr_lag1')
        assert hasattr(metrics, 'adf_stationary')
        assert hasattr(metrics, 'z_score')
        assert hasattr(metrics, 'avg_correlation')
        assert hasattr(metrics, 'volatility_percentile')
    
    def test_validate_signal(self, sample_returns):
        """Test signal validation."""
        qse = QuantitativeStatisticsEngine()
        metrics = qse.compute_statistical_metrics(
            returns=sample_returns,
            bar_index=100
        )
        validation = qse.validate_signal(metrics)
        
        assert hasattr(validation, 'signal_validity_flag')
        assert hasattr(validation, 'trend_persistence_check')
        assert hasattr(validation, 'stationarity_check')
        assert hasattr(validation, 'distribution_check')
        assert hasattr(validation, 'failure_reasons')
        
        assert isinstance(validation.signal_validity_flag, bool)
        assert isinstance(validation.failure_reasons, list)
    
    def test_compute_regime_state(self, sample_returns):
        """Test regime state computation."""
        qse = QuantitativeStatisticsEngine()
        metrics = qse.compute_statistical_metrics(
            returns=sample_returns,
            bar_index=100
        )
        regime = qse.compute_regime_state(metrics, sample_returns, 100)
        
        assert hasattr(regime, 'trend_regime')
        assert hasattr(regime, 'volatility_regime')
        assert hasattr(regime, 'market_phase')
        assert hasattr(regime, 'efficiency_ratio')
        
        assert isinstance(regime.trend_regime, str)
        assert isinstance(regime.market_phase, str)
    
    def test_process_bar(self, sample_returns):
        """Test full bar processing."""
        qse = QuantitativeStatisticsEngine()
        output = qse.process_bar(
            symbol='TEST',
            returns=sample_returns,
            bar_index=100
        )
        
        assert output.symbol == 'TEST'
        assert output.bar_index == 100
        assert output.metrics is not None
        assert output.validation is not None
        assert output.regime_state is not None
        assert isinstance(output.timestamp, datetime)
    
    def test_process_series(self, sample_returns):
        """Test full series processing."""
        qse = QuantitativeStatisticsEngine()
        results = qse.process_series(
            symbol='TEST',
            returns=sample_returns.iloc[100:120]  # Process 20 bars
        )
        
        assert len(results) == 20
        assert all(r.symbol == 'TEST' for r in results)
    
    def test_get_validation_summary(self, sample_returns):
        """Test validation summary."""
        qse = QuantitativeStatisticsEngine()
        results = qse.process_series(
            symbol='TEST',
            returns=sample_returns.iloc[100:150]
        )
        
        summary = qse.get_validation_summary(results)
        
        assert 'total_bars' in summary
        assert 'valid_signals' in summary
        assert 'invalid_signals' in summary
        assert 'validity_rate' in summary
        
        assert summary['total_bars'] == len(results)
        assert summary['valid_signals'] + summary['invalid_signals'] == summary['total_bars']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
