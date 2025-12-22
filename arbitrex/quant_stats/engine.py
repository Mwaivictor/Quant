"""
Quantitative Statistics Engine (QSE)

Main orchestrator for statistical validation of feature vectors.
Applies 5-gate validation:
1. Autocorrelation & Trend Persistence
2. Stationarity (ADF Test)
3. Distribution Stability & Outlier Detection
4. Cross-Pair Correlation
5. Volatility-Based Filtering

Output: signal_validity_flag + detailed metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from .config import QuantStatsConfig
from .schemas import (
    StatisticalMetrics,
    SignalValidation,
    RegimeState,
    QuantStatsOutput
)
from .autocorrelation import AutocorrelationAnalyzer
from .stationarity import StationarityTester
from .distribution import DistributionAnalyzer
from .correlation import CorrelationAnalyzer
from .volatility import VolatilityFilter, VolatilityRegime

LOG = logging.getLogger(__name__)


class QuantitativeStatisticsEngine:
    """
    Quantitative Statistics Engine (QSE).
    
    Statistical gatekeeper between Feature Engine and ML layers.
    Validates feature vectors using 5 statistical checks.
    """
    
    def __init__(self, config: Optional[QuantStatsConfig] = None):
        """
        Initialize QSE with configuration.
        
        Args:
            config: QSE configuration (uses defaults if None)
        """
        self.config = config or QuantStatsConfig()
        
        # Initialize statistical modules
        self.autocorr_analyzer = AutocorrelationAnalyzer(
            lags=self.config.autocorrelation.lags,
            min_threshold=self.config.autocorrelation.min_autocorr_threshold,
            rolling_window=self.config.autocorrelation.rolling_window
        )
        
        self.stationarity_tester = StationarityTester(
            significance_level=self.config.stationarity.significance_level,
            rolling_window=self.config.stationarity.rolling_window,
            max_lag=self.config.stationarity.max_lag
        )
        
        self.distribution_analyzer = DistributionAnalyzer(
            rolling_window=self.config.distribution.rolling_window,
            z_score_threshold=self.config.distribution.z_score_threshold,
            min_samples=self.config.distribution.min_samples
        )
        
        self.correlation_analyzer = CorrelationAnalyzer(
            rolling_window=self.config.correlation.rolling_window,
            max_correlation_threshold=self.config.correlation.max_correlation_threshold,
            min_pairs=self.config.correlation.min_pairs
        )
        
        self.volatility_filter = VolatilityFilter(
            rolling_window=self.config.volatility.rolling_window,
            min_percentile=self.config.volatility.min_percentile,
            max_percentile=self.config.volatility.max_percentile
        )
        
        LOG.info(f"QSE initialized with config version: {self.config.get_config_hash()}")
    
    def compute_statistical_metrics(
        self,
        returns: pd.Series,
        bar_index: int,
        returns_dict: Optional[Dict[str, pd.Series]] = None
    ) -> StatisticalMetrics:
        """
        Compute statistical metrics for a single bar.
        
        Args:
            returns: Return series for primary symbol
            bar_index: Index of current bar
            returns_dict: Optional dict of multiple symbols for correlation
        
        Returns:
            StatisticalMetrics object
        """
        # 1. Autocorrelation Analysis
        autocorr_result = self.autocorr_analyzer.analyze_bar(returns, bar_index)
        
        # 2. Stationarity Test
        is_stationary, adf_pvalue, adf_stat = self.stationarity_tester.test_bar(
            returns, bar_index
        )
        
        # 3. Distribution Analysis
        dist_result = self.distribution_analyzer.analyze_bar(returns, bar_index)
        
        # 4. Correlation Analysis (if multiple symbols provided)
        if returns_dict and len(returns_dict) >= self.config.correlation.min_pairs:
            corr_result = self.correlation_analyzer.analyze_bar(returns_dict, bar_index)
        else:
            corr_result = {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_dispersion': 0.0
            }
        
        # 5. Volatility Analysis
        vol_result = self.volatility_filter.analyze_bar(returns, bar_index)
        
        # Construct StatisticalMetrics
        metrics = StatisticalMetrics(
            # Autocorrelation
            autocorr_lag1=autocorr_result.get('autocorr_lag1', 0.0),
            autocorr_lag5=autocorr_result.get('autocorr_lag5', 0.0),
            autocorr_lag10=autocorr_result.get('autocorr_lag10', 0.0),
            autocorr_lag20=autocorr_result.get('autocorr_lag20', 0.0),
            trend_persistence_score=autocorr_result['trend_persistence_score'],
            
            # Stationarity
            adf_stationary=is_stationary,
            adf_pvalue=adf_pvalue,
            adf_test_statistic=adf_stat,
            
            # Distribution
            z_score=dist_result['z_score'],
            is_outlier=dist_result['is_outlier'],
            rolling_mean=dist_result['rolling_mean'],
            rolling_std=dist_result['rolling_std'],
            distribution_stable=dist_result['distribution_stable'],
            
            # Correlation
            avg_cross_correlation=corr_result['avg_correlation'],
            max_cross_correlation=corr_result['max_correlation'],
            correlation_dispersion=corr_result['correlation_dispersion'],
            
            # Volatility
            volatility_percentile=vol_result['volatility_percentile'],
            volatility_regime=vol_result['regime'].value,
            current_volatility=vol_result['current_volatility']
        )
        
        return metrics
    
    def validate_signal(
        self,
        metrics: StatisticalMetrics
    ) -> SignalValidation:
        """
        Validate signal based on statistical metrics.
        
        Applies all 5 validation checks and determines overall validity.
        
        Args:
            metrics: Statistical metrics
        
        Returns:
            SignalValidation object
        """
        failure_reasons = []
        
        # Check 1: Trend Persistence
        trend_check = (
            self.config.validation.require_trend_persistence and
            metrics.trend_persistence_score >= self.config.validation.min_trend_consistency
        )
        if self.config.validation.require_trend_persistence and not trend_check:
            failure_reasons.append(
                f"Insufficient trend persistence: {metrics.trend_persistence_score:.3f} < "
                f"{self.config.validation.min_trend_consistency}"
            )
        
        # Check 2: Stationarity
        stationarity_check = (
            not self.config.validation.require_stationarity or
            metrics.adf_stationary
        )
        if self.config.validation.require_stationarity and not stationarity_check:
            failure_reasons.append(
                f"Non-stationary series: ADF p-value={metrics.adf_pvalue:.4f}"
            )
        
        # Check 3: Distribution Stability
        distribution_check = (
            not self.config.validation.require_distribution_stability or
            (metrics.distribution_stable and not metrics.is_outlier)
        )
        if self.config.validation.require_distribution_stability and not distribution_check:
            if metrics.is_outlier:
                failure_reasons.append(f"Outlier detected: z-score={metrics.z_score:.3f}")
            if not metrics.distribution_stable:
                failure_reasons.append("Distribution unstable")
        
        # Check 4: Correlation
        correlation_check = (
            not self.config.validation.require_correlation_check or
            metrics.max_cross_correlation < self.config.correlation.max_correlation_threshold
        )
        if self.config.validation.require_correlation_check and not correlation_check:
            failure_reasons.append(
                f"Excessive correlation: {metrics.max_cross_correlation:.3f} >= "
                f"{self.config.correlation.max_correlation_threshold}"
            )
        
        # Check 5: Volatility Regime
        vol_regime = VolatilityRegime(metrics.volatility_regime)
        volatility_check = (
            not self.config.validation.require_volatility_filter or
            vol_regime == VolatilityRegime.NORMAL
        )
        if self.config.validation.require_volatility_filter and not volatility_check:
            failure_reasons.append(f"Invalid volatility regime: {vol_regime.value}")
        
        # Overall validity: ALL checks must pass
        signal_valid = all([
            trend_check,
            stationarity_check,
            distribution_check,
            correlation_check,
            volatility_check
        ])
        
        # Compute composite scores
        # Trend consistency score (0-1)
        trend_consistency = min(1.0, metrics.trend_persistence_score)
        
        # Regime quality score (0-1)
        # 1.0 if NORMAL regime, 0.5 if LOW/HIGH, 0.0 if UNKNOWN
        regime_quality_map = {
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.LOW: 0.5,
            VolatilityRegime.HIGH: 0.5,
            VolatilityRegime.UNKNOWN: 0.0
        }
        regime_quality = regime_quality_map.get(vol_regime, 0.0)
        
        validation = SignalValidation(
            signal_validity_flag=signal_valid,
            autocorr_check_passed=trend_check,
            stationarity_check_passed=stationarity_check,
            distribution_check_passed=distribution_check,
            correlation_check_passed=correlation_check,
            volatility_check_passed=volatility_check,
            trend_consistency=trend_consistency,
            regime_quality_score=regime_quality,
            failure_reasons=failure_reasons
        )
        
        return validation
    
    def compute_regime_state(
        self,
        metrics: StatisticalMetrics,
        returns: pd.Series,
        bar_index: int
    ) -> RegimeState:
        """
        Compute market regime state.
        
        Args:
            metrics: Statistical metrics
            returns: Return series
            bar_index: Current bar index
        
        Returns:
            RegimeState object
        """
        # Trend regime based on autocorrelation
        if metrics.trend_persistence_score > 0.5:
            trend_regime = "STRONG_TREND"
        elif metrics.trend_persistence_score > 0.2:
            trend_regime = "WEAK_TREND"
        else:
            trend_regime = "MEAN_REVERTING"
        
        # Volatility regime
        volatility_regime = metrics.volatility_regime
        
        # Correlation regime (if available)
        if metrics.max_cross_correlation > 0.85:
            correlation_regime = "HIGH_CORRELATION"
        elif metrics.max_cross_correlation < 0.3:
            correlation_regime = "LOW_CORRELATION"
        else:
            correlation_regime = "NORMAL_CORRELATION"
        
        # Market phase (simplified)
        if trend_regime == "STRONG_TREND" and volatility_regime == "HIGH":
            market_phase = "BREAKOUT"
        elif trend_regime == "MEAN_REVERTING" and volatility_regime == "LOW":
            market_phase = "RANGING"
        elif volatility_regime == "HIGH":
            market_phase = "VOLATILE"
        else:
            market_phase = "NORMAL"
        
        # Efficiency ratio (simplified: |autocorr_lag1|)
        efficiency_ratio = abs(metrics.autocorr_lag1)
        
        # Regime stability
        trend_regime_stable = metrics.distribution_stable
        regime_stable = (volatility_regime == "NORMAL") and trend_regime_stable
        
        regime_state = RegimeState(
            trend_regime=trend_regime,
            trend_strength=metrics.trend_persistence_score,
            volatility_regime=volatility_regime,
            volatility_level=metrics.current_volatility,
            correlation_regime=correlation_regime,
            avg_correlation=metrics.avg_cross_correlation,
            market_phase=market_phase,
            efficiency_ratio=efficiency_ratio,
            regime_stable=regime_stable,
            regime_change_detected=False
        )
        
        return regime_state
    
    def process_bar(
        self,
        symbol: str,
        returns: pd.Series,
        bar_index: int,
        returns_dict: Optional[Dict[str, pd.Series]] = None
    ) -> QuantStatsOutput:
        """
        Process a single bar through full QSE pipeline.
        
        Args:
            symbol: Symbol identifier
            returns: Return series for symbol
            bar_index: Index of current bar
            returns_dict: Optional dict for cross-pair analysis
        
        Returns:
            QuantStatsOutput with complete analysis
        """
        LOG.debug(f"Processing bar {bar_index} for {symbol}")
        
        # Compute statistical metrics
        metrics = self.compute_statistical_metrics(
            returns=returns,
            bar_index=bar_index,
            returns_dict=returns_dict
        )
        
        # Validate signal
        validation = self.validate_signal(metrics)
        
        # Compute regime state
        regime_state = self.compute_regime_state(metrics, returns, bar_index)
        
        # Create output
        output = QuantStatsOutput(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe="unknown",  # Can be added as parameter
            metrics=metrics,
            validation=validation,
            regime=regime_state,
            config_hash=self.config.get_config_hash(),
            config_version=self.config.config_version
        )
        
        if validation.signal_validity_flag:
            LOG.info(f"Bar {bar_index} ({symbol}): VALID signal")
        else:
            LOG.info(f"Bar {bar_index} ({symbol}): INVALID signal - "
                    f"{len(validation.failure_reasons)} failures")
        
        return output
    
    def process_series(
        self,
        symbol: str,
        returns: pd.Series,
        returns_dict: Optional[Dict[str, pd.Series]] = None
    ) -> List[QuantStatsOutput]:
        """
        Process full time series.
        
        Args:
            symbol: Symbol identifier
            returns: Return series
            returns_dict: Optional dict for cross-pair analysis
        
        Returns:
            List of QuantStatsOutput objects
        """
        results = []
        
        for i in range(len(returns)):
            try:
                output = self.process_bar(
                    symbol=symbol,
                    returns=returns,
                    bar_index=i,
                    returns_dict=returns_dict
                )
                results.append(output)
            except Exception as e:
                LOG.error(f"Error processing bar {i}: {e}")
        
        return results
    
    def get_validation_summary(
        self,
        results: List[QuantStatsOutput]
    ) -> Dict[str, any]:
        """
        Get summary statistics from validation results.
        
        Args:
            results: List of QSE outputs
        
        Returns:
            Summary dict
        """
        if not results:
            return {}
        
        total_bars = len(results)
        valid_bars = sum(1 for r in results if r.validation.signal_validity_flag)
        
        # Failure breakdown
        failure_counts = {}
        for result in results:
            for reason in result.validation.failure_reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        # Average metrics
        avg_trend_score = np.mean([r.metrics.trend_persistence_score for r in results])
        avg_z_score = np.mean([abs(r.metrics.z_score) for r in results])
        
        summary = {
            'total_bars': total_bars,
            'valid_signals': valid_bars,
            'invalid_signals': total_bars - valid_bars,
            'validity_rate': valid_bars / total_bars if total_bars > 0 else 0.0,
            'avg_trend_persistence': avg_trend_score,
            'avg_abs_z_score': avg_z_score,
            'failure_breakdown': failure_counts
        }
        
        return summary
